# threshold_summary.py

import pandas as pd
import json

def analyze_thresholds(file_path):
    df = pd.read_csv(file_path)
    df['is_commercial'] = df['is_commercial'].astype(int)
    df['portfolio'] = df['is_commercial'].map({0: 'retail', 1: 'commercial'})

    thresh_df = df.groupby('portfolio').agg({
        'credit_score': 'mean',
        'loan_amount': 'mean',
        'debt_to_income': 'mean'
    }).reset_index()

    thresh_df['credit_score_thresh'] = (thresh_df['credit_score'] * 0.95).round(1)
    thresh_df['loan_amount_thresh'] = (thresh_df['loan_amount'] * 1.10).round(0)
    thresh_df['debt_to_income_thresh'] = (thresh_df['debt_to_income'] * 1.10).round(2)

    alerts = []
    for _, row in df.iterrows():
        portfolio = 'retail' if row['is_commercial'] == 0 else 'commercial'
        limits = thresh_df[thresh_df['portfolio'] == portfolio].iloc[0]

        if row['credit_score'] < limits['credit_score_thresh']:
            alerts.append({'portfolio': portfolio, 'metric': 'credit_score', 'value': row['credit_score'], 'threshold': limits['credit_score_thresh']})
        if row['loan_amount'] > limits['loan_amount_thresh']:
            alerts.append({'portfolio': portfolio, 'metric': 'loan_amount', 'value': row['loan_amount'], 'threshold': limits['loan_amount_thresh']})
        if row['debt_to_income'] > limits['debt_to_income_thresh']:
            alerts.append({'portfolio': portfolio, 'metric': 'debt_to_income', 'value': row['debt_to_income'], 'threshold': limits['debt_to_income_thresh']})

    return {
        "thresholds": thresh_df.to_dict(orient='records'),
        "alerts": alerts
    }
