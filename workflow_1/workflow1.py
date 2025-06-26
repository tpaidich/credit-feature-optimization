from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import json
import pandas as pd

items = _input.all()
data = [item.json for item in items]
df = pd.DataFrame(data)

if 'is_commercial' in df.columns:
    df = df.drop(columns=['is_commercial'])

# encode categorical variables
for col in ['region', 'officer_id']:
    df[col] = LabelEncoder().fit_transform(df[col])

# separate features and target variable
X = df.drop(columns=['default', 'model_decision', 'final_decision'])
y = df['default']

# train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# get feature importances
importance_df = pd.DataFrame({
'feature': X.columns,
'importance': model.feature_importances_
}).sort_values(by='importance', ascending=False)

# (optional) drift detection - "how are feature importances changing?"
try:
    with open('feature_drift_log.json', 'r') as f:
        previous = pd.DataFrame(json.load(f))
    joined = importance_df.merge(previous, on='feature', suffixes=('', '_previous'))
    joined['drift'] = (joined['importance'] - joined['importance_previous']).abs()
    drift_df = joined.sort_values(by='drift', ascending=False)
    drift_df = drift_df[['feature', 'importance', 'importance_previous', 'drift']]
    drift_df.to_csv('feature_drift_report.csv', index=False)
except FileNotFoundError:
    # first run
    drift_df = importance_df.head(10)

# save current importances
importance_df.to_json('feature_drift_log.json', orient='records')

# convert to Python list for n8n JSON output
feature_importances = importance_df.to_dict(orient='records')
print(feature_importances)
