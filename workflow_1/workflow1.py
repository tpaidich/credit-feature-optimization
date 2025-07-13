import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
import shap

# Read input from n8n
items = _input.all()
data = [item.json for item in items]
file_path = data[0].get('filepath')

try:
    df = pd.read_csv(file_path)
except Exception as e:
    return [{"error": f"Could not load file: {str(e)}"}]

# --- Helper Functions ---

def preprocess(df):
    drop_cols = ['user_id', 'date', 'model_decision', 'final_decision']
    df = df.drop(columns=drop_cols, errors='ignore')
    if 'is_commercial' in df.columns:
        df['is_commercial'] = df['is_commercial'].astype(object)
    for col in ['region', 'officer_id']:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df

def calculate_gini_importance(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_}).sort_values(by='importance', ascending=False)

def calculate_information_value(df, target):
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
    return pd.DataFrame(list(iv_values.items()), columns=['feature', 'importance']).sort_values(by='importance', ascending=False)

def calculate_permutation_importance(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
    return pd.DataFrame({'feature': X.columns, 'importance': result.importances_mean}).sort_values(by='importance', ascending=False)

def calculate_shap_importance(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap_abs = np.abs(shap_values[1] if isinstance(shap_values, list) else shap_values).mean(axis=0)
    return pd.DataFrame({'feature': X.columns, 'importance': shap_abs}).sort_values(by='importance', ascending=False)

# --- Run all methods ---
df = preprocess(df)

if 'default' not in df.columns:
    return [{"error": "Missing 'default' column in dataset."}]

y = df['default']
X = df.drop(columns=['default'], errors='ignore')

# Create outputs for each method
gini = calculate_gini_importance(X, y).to_dict(orient='records')
iv = calculate_information_value(df, 'default').to_dict(orient='records')
perm = calculate_permutation_importance(X, y).to_dict(orient='records')
shap_vals = calculate_shap_importance(X, y).to_dict(orient='records')

# Return as separate n8n tables
output = []

for method_name, result in [
    ("gini", gini),
    ("information_value", iv),
    ("permutation", perm),
    ("shap", shap_vals)
]:
    for row in result:
        output.append({"json": {"method": method_name, **row}})

return output
