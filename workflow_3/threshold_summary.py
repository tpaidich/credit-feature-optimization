import pandas as pd
import numpy as np

def calculate_rolling_thresholds(df, metrics, window_size, std_multiplier):
    df = df.copy()
    for metric in metrics:
        if metric in df.columns:
            df[f'{metric}_rolling_mean'] = df.groupby('user_id')[metric].rolling(window=window_size, min_periods=1).mean().reset_index(level=0, drop=True)
            df[f'{metric}_rolling_std'] = df.groupby('user_id')[metric].rolling(window=window_size, min_periods=1).std().reset_index(level=0, drop=True)
            df[f'{metric}_upper_bound'] = df[f'{metric}_rolling_mean'] + std_multiplier * df[f'{metric}_rolling_std']
            df[f'{metric}_lower_bound'] = df[f'{metric}_rolling_mean'] - std_multiplier * df[f'{metric}_rolling_std']
            if metric in ['loan_amount', 'account_balance', 'business_revenue', 'num_employees', 'collateral_value', 'income']:
                df[f'{metric}_lower_bound'] = df[f'{metric}_lower_bound'].clip(lower=0)
    return df

def calculate_percentile_thresholds(df, metrics, window_size, percentile):
    df = df.copy()
    for metric in metrics:
        if metric in df.columns:
            percentile_val = percentile / 100.0
            df[f'{metric}_percentile_threshold'] = df.groupby('user_id')[metric].rolling(window=window_size, min_periods=1).quantile(percentile_val).reset_index(level=0, drop=True)
    return df

def analyze_thresholds(file_path):
    try:
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
    except Exception as e:
        raise RuntimeError(f"Could not load or parse file: {e}")

    metrics = ['credit_score', 'loan_amount', 'debt_to_income']
    df['portfolio'] = df['is_commercial'].astype(int).map({0: 'retail', 1: 'commercial'})

    thresh_df_original = df.groupby('portfolio')[metrics].mean().reset_index()
    thresh_df_original.rename(columns={
        'credit_score': 'credit_score_mean',
        'loan_amount': 'loan_amount_mean',
        'debt_to_income': 'debt_to_income_mean'
    }, inplace=True)

    selected = {
        'Mean': True,
        'Rolling': True,
        'Percentile': True,
        'Percentile_Direction': {
            'credit_score': 'lower',
            'loan_amount': 'upper',
            'debt_to_income': 'upper'
        }
    }

    df_thresh = df.copy()

    if selected['Rolling']:
        df_roll = calculate_rolling_thresholds(df.copy(), metrics, window_size=12, std_multiplier=2.0)
        df_thresh = df_thresh.merge(df_roll[['date', 'user_id'] + [c for c in df_roll.columns if '_rolling_' in c or '_bound' in c]], on=['date', 'user_id'], how='left')

    if selected['Percentile']:
        df_perc_cs = calculate_percentile_thresholds(df.copy(), ['credit_score'], 12, 5)
        df_perc_ld = calculate_percentile_thresholds(df.copy(), ['loan_amount', 'debt_to_income'], 12, 95)
        df_perc = df_perc_cs.merge(
            df_perc_ld[['date', 'user_id', 'loan_amount_percentile_threshold', 'debt_to_income_percentile_threshold']],
            on=['date', 'user_id'],
            how='left'
        )
        df_thresh = df_thresh.merge(
            df_perc[['date', 'user_id'] + [c for c in df_perc.columns if '_percentile_' in c]],
            on=['date', 'user_id'],
            how='left'
        )

    alerts = []

    for col in df_thresh.columns:
        if any(metric in col for metric in metrics):
            df_thresh[col] = pd.to_numeric(df_thresh[col], errors='coerce')

    for _, row in df_thresh.iterrows():
        uid = row['user_id']
        date = str(row['date'].date())
        portfolio = row['portfolio']

        for metric in metrics:
            val = row[metric]
            mean_val = thresh_df_original.loc[thresh_df_original['portfolio'] == portfolio, f'{metric}_mean'].values[0]

            if selected['Mean']:
                if metric == 'credit_score' and pd.notna(val) and val < mean_val * 0.8:
                    alerts.append({"user_id": uid, "date": date, "metric": metric, "value": val, "threshold": mean_val * 0.8, "method": "Mean", "portfolio": portfolio})
                elif metric != 'credit_score' and pd.notna(val) and val > mean_val * 1.2:
                    alerts.append({"user_id": uid, "date": date, "metric": metric, "value": val, "threshold": mean_val * 1.2, "method": "Mean", "portfolio": portfolio})

            if selected['Rolling']:
                upper = row.get(f'{metric}_upper_bound')
                lower = row.get(f'{metric}_lower_bound')
                if pd.notna(val) and pd.notna(upper) and pd.notna(lower):
                    if metric == 'credit_score' and val < lower:
                        alerts.append({"user_id": uid, "date": date, "metric": metric, "value": val, "threshold": lower, "method": "Rolling", "portfolio": portfolio})
                    elif metric != 'credit_score' and val > upper:
                        alerts.append({"user_id": uid, "date": date, "metric": metric, "value": val, "threshold": upper, "method": "Rolling", "portfolio": portfolio})

            if selected['Percentile']:
                direction = selected['Percentile_Direction'][metric]
                perc_thresh = row.get(f'{metric}_percentile_threshold')
                if pd.notna(val) and pd.notna(perc_thresh):
                    if direction == 'lower' and val < perc_thresh:
                        alerts.append({"user_id": uid, "date": date, "metric": metric, "value": val, "threshold": perc_thresh, "method": "Percentile", "portfolio": portfolio})
                    elif direction == 'upper' and val > perc_thresh:
                        alerts.append({"user_id": uid, "date": date, "metric": metric, "value": val, "threshold": perc_thresh, "method": "Percentile", "portfolio": portfolio})

    alerts_df = pd.DataFrame(alerts)
    stats = {
        "alerts_by_metric": alerts_df['metric'].value_counts().to_dict() if not alerts_df.empty else {},
        "alerts_by_year": alerts_df['date'].apply(lambda x: pd.to_datetime(x).year).value_counts().sort_index().to_dict() if not alerts_df.empty else {},
        "alerts_by_portfolio": alerts_df['portfolio'].value_counts().to_dict() if not alerts_df.empty else {}
    }

    return {
        "alert_summary_statistics": stats,
        "thresholds_mean_by_portfolio": thresh_df_original.to_dict(orient='records')
    }
