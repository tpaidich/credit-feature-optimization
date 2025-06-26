# workflow_4.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

def analyze_stability(input_path: str):
    df = pd.read_csv(input_path)
    df['date'] = pd.to_datetime(df['date'])
    df['default'] = pd.to_numeric(df['default'], errors='coerce')

    # sort and compute rolling stats
    df = df.sort_values('date')
    df['rolling_default_rate'] = df['default'].rolling(window=12, min_periods=1).mean()
    df['rolling_std'] = df['default'].rolling(window=12, min_periods=1).std()
    df['stability_flag'] = (df['rolling_std'] > 0.2).astype(int)

    # summary stats
    summary = {
        'periods_flagged': int(df['stability_flag'].sum()),
        'most_unstable_month': str(df.loc[df['rolling_std'].idxmax(), 'date'].date()),
        'least_unstable_month': str(df.loc[df['rolling_std'].idxmin(), 'date'].date()),
        'max_std': round(df['rolling_std'].max(), 3),
        'min_std': round(df['rolling_std'].min(), 3)
    }

    # plot
    plt.figure(figsize=(10, 5))
    plt.plot(df['date'], df['rolling_default_rate'], label='12M Rolling Default Rate', color='blue')
    plt.fill_between(df['date'],
                     df['rolling_default_rate'] - df['rolling_std'],
                     df['rolling_default_rate'] + df['rolling_std'],
                     alpha=0.2, label='±1 STD', color='gray')
    plt.title('Backtesting Model Stability')
    plt.xlabel('Date')
    plt.ylabel('Default Rate')
    plt.legend()
    plt.tight_layout()
    plot_path = "model_stability_plot.png"
    plt.savefig(plot_path)
    plt.close()

    # output json
    output = {
        "summary": summary,
        "plot_path": plot_path
    }

    with open('backtesting.json', 'w') as f:
        json.dump(output, f)

    return output
