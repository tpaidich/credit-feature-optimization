from fastapi import FastAPI, UploadFile, File
import pandas as pd
import numpy as np
import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder
import shap

app = FastAPI()

# --- Helper Functions ---

def preprocess(df):
    drop_cols = ['model_decision', 'final_decision', 'user_id', 'date']
    df = df.drop(columns=drop_cols, errors='ignore')

    for col in ['region', 'officer_id']:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Feature engineering
    if 'credit_score' in df.columns:
        df['credit_score_squared'] = df['credit_score'] ** 2
    if 'loan_amount' in df.columns and 'income' in df.columns:
        df['loan_amount_to_income_ratio'] = df['loan_amount'] / df['income'].replace(0, 1)
    if 'interest_rate' in df.columns and 'income' in df.columns:
        df['interest_income_ratio'] = df['interest_rate'] / df['income'].replace(0, 1)
    if 'business_revenue' in df.columns and 'num_employees' in df.columns:
        df['revenue_per_employee'] = df['business_revenue'] / df['num_employees'].replace(0, 1)
    if 'collateral_value' in df.columns and 'loan_amount' in df.columns:
        df['collateral_coverage_ratio'] = df['collateral_value'] / df['loan_amount'].replace(0, 1)

    return df

def calculate_gini(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})

def calculate_iv(df, target):
    iv_values = {}
    for col in df.columns:
        if col == target:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            try:
                bins = pd.qcut(df[col], q=10, labels=False, duplicates='drop')
                data = df.groupby(bins)[target].agg(['count', 'sum']).reset_index()
            except:
                data = df.groupby(col)[target].agg(['count', 'sum']).reset_index()
        else:
            data = df.groupby(col)[target].agg(['count', 'sum']).reset_index()

        data.columns = [col, 'total', 'bad']
        data['good'] = data['total'] - data['bad']
        total_good = data['good'].sum()
        total_bad = data['bad'].sum()
        data['perc_good'] = data['good'] / total_good if total_good > 0 else 0
        data['perc_bad'] = data['bad'] / total_bad if total_bad > 0 else 0
        data['woe'] = np.log(data['perc_good'] / data['perc_bad'].replace(0, 0.001))
        data['woe'] = data['woe'].replace([np.inf, -np.inf], 0)
        data['iv'] = (data['perc_good'] - data['perc_bad']) * data['woe']
        iv_values[col] = data['iv'].sum()

    return pd.DataFrame(list(iv_values.items()), columns=['feature', 'importance'])

def calculate_permutation(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
    return pd.DataFrame({'feature': X.columns, 'importance': result.importances_mean})

def calculate_shap(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap_abs = np.abs(shap_values[1] if isinstance(shap_values, list) else shap_values).mean(axis=0)
    return pd.DataFrame({'feature': X.columns, 'importance': shap_abs})

# --- API Endpoint ---

@app.post("/feature-importances")
async def feature_importances(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        if 'default' not in df.columns:
            return [{"error": "Missing 'default' column."}]

        y = df['default']
        df = preprocess(df)
        X = df.drop(columns=['default'], errors='ignore').select_dtypes(include=['number'])

        results = {
            "gini": calculate_gini(X, y),
            "information_value": calculate_iv(df, 'default'),
            "permutation": calculate_permutation(X, y),
            "shap": calculate_shap(X, y),
        }

        # Flatten results into one list with method name per row
        output = []
        for method, df_result in results.items():
            for _, row in df_result.iterrows():
                output.append({
                    "method": method,
                    "feature": row["feature"],
                    "importance": row["importance"]
                })

        return output

    except Exception as e:
        return [{"error": str(e)}]
