import pandas as pd
import json

def analyze_overrides(file_path):
    df = pd.read_csv(file_path)

    df['override_flag'] = (df['model_decision'] != df['final_decision']).astype(int)
    df['date'] = pd.date_range(start='2021-01-01', periods=len(df), freq='D')
    df['month'] = df['date'].dt.to_period('M').astype(str)

    monthly_override = df.groupby('month')['override_flag'].mean().reset_index()
    officer_stats = df.groupby('officer_id')['override_flag'].agg(['mean', 'count']).reset_index()
    officer_stats.columns = ['officer_id', 'override_rate', 'total_cases']
    outliers = officer_stats[officer_stats['override_rate'] > 0.5]

    summary = {
        'total_overrides': int(df['override_flag'].sum()),
        'total_decisions': len(df),
        'override_rate': round(df['override_flag'].mean(), 3),
        'num_outlier_officers': len(outliers),
        'max_officer_override_rate': round(officer_stats['override_rate'].max(), 3)
    }

    output = {
        "summary": summary,
        #"monthly_trend": monthly_override.to_dict(orient="records"),
        "officer_outliers": outliers.to_dict(orient="records")
    }

    return output
