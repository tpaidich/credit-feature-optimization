from fastapi import FastAPI, UploadFile, File
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import io

app = FastAPI()

@app.post("/feature-importances")
async def feature_importances(file: UploadFile = File(...)):
    try:
        # Read uploaded file
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        # Drop label columns if present
        drop_cols = ['default', 'model_decision', 'final_decision']
        y = None

        if 'default' in df.columns:
            y = df['default']
            drop_cols.remove('default')  # don't drop if it's the target
        else:
            return {"error": "Missing required 'default' column in the data."}

        # Optional label encoding for categorical features
        label_cols = ['region', 'officer_id']
        for col in label_cols:
            if col in df.columns:
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))

        # Feature engineering
        df['credit_score_squared'] = df['credit_score'] ** 2
        df['loan_amount_to_income_ratio'] = df['loan_amount'] / df['income'].replace(0, 1)
        df['interest_income_ratio'] = df['interest_rate'] / df['income'].replace(0, 1)
        df['revenue_per_employee'] = df['business_revenue'] / df['num_employees'].replace(0, 1)
        df['collateral_coverage_ratio'] = df['collateral_value'] / df['loan_amount'].replace(0, 1)

        # Handle datetime columns
        for col in df.columns:
            if df[col].dtype == 'object' and df[col].str.contains(r'\d{4}-\d{2}-\d{2}', na=False).any():
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    df[f"{col}_year"] = df[col].dt.year
                    df[f"{col}_month"] = df[col].dt.month
                    df.drop(columns=[col], inplace=True)
                except Exception:
                    continue  # if can't convert, ignore

        # Drop irrelevant or non-numeric columns
        df = df.drop(columns=drop_cols, errors='ignore')
        X = df.select_dtypes(include=['number'])

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        # Get importances
        importances = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values(by='importance', ascending=False)

        return {"importances": importances.to_dict(orient='records')}

    except Exception as e:
        return {"error": str(e)}
