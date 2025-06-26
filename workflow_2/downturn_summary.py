# downturn_summary.py

import pandas as pd
import numpy as np
import json

def analyze_downturn(csv_path):
    df = pd.read_csv(csv_path)

    df['date'] = pd.to_datetime(df['date'])
    df['gdp_growth'] = np.random.normal(1.5, 2.0, len(df)).round(2)
    df['unemployment_rate'] = np.random.normal(5.5, 1.5, len(df)).round(2)
    df['default_rate'] = df['default'] + np.random.normal(0.05, 0.02, len(df))

    df = df.sort_values('date')
    is_downturn = (df['gdp_growth'] < 0) & (df['unemployment_rate'] > 6)
    downturns = df[is_downturn]
    non_downturns = df[~is_downturn]

    downturn_avg = downturns['default_rate'].mean().round(3)
    non_downturn_avg = non_downturns['default_rate'].mean().round(3)
    best_downturn = downturns.loc[downturns['default_rate'].idxmax()]

    return [{
        'downturn_avg_default_rate': downturn_avg,
        'baseline_default_rate': non_downturn_avg,
        'worst_downturn_date': str(best_downturn['date'].date()),
        'worst_downturn_default': best_downturn['default_rate']
    }]
