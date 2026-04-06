# preprocessing.py
import pandas as pd
import numpy as np

def preprocess_data(df, window_size=3):
    """
    df: raw dataframe with columns: well_id, timestamp, water_level, ph, ec, tds, turbidity, temperature
    Returns cleaned and smoothed dataframe.
    """
    # Sort
    df = df.sort_values(['well_id', 'timestamp']).copy()

    # Introduce random missing values (5%) to simulate real gaps
    np.random.seed(0)
    for col in ['water_level', 'ph', 'ec', 'tds', 'turbidity', 'temperature']:
        mask = np.random.random(len(df)) < 0.05
        df.loc[mask, col] = np.nan

    # Handle missing: forward fill within each well
    df = df.groupby('well_id').apply(lambda group: group.fillna(method='ffill').fillna(method='bfill')).reset_index(drop=True)

    # Outlier removal using IQR per well
    def remove_outliers(group):
        for col in ['water_level', 'ph', 'ec', 'tds', 'turbidity', 'temperature']:
            Q1 = group[col].quantile(0.25)
            Q3 = group[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            group.loc[(group[col] < lower) | (group[col] > upper), col] = np.nan
        return group

    df = df.groupby('well_id').apply(remove_outliers).reset_index(drop=True)

    # Fill outliers again (with forward fill after removal)
    df = df.groupby('well_id').apply(lambda g: g.fillna(method='ffill').fillna(method='bfill')).reset_index(drop=True)

    # Noise filtering: moving average (window=3) per well
    def smooth(group):
        for col in ['water_level', 'ph', 'ec', 'tds', 'turbidity', 'temperature']:
            group[col] = group[col].rolling(window=window_size, min_periods=1, center=True).mean()
        return group

    df = df.groupby('well_id').apply(smooth).reset_index(drop=True)

    # Temporal windowing: we can create lag features later in model building, but here we just keep the series.
    # Optionally, we could resample to a fixed frequency (already monthly).

    return df

if __name__ == "__main__":
    raw = pd.read_csv('data/synthetic_raw/groundwater_monthly.csv', parse_dates=['timestamp'])
    preprocessed = preprocess_data(raw)
    preprocessed.to_csv('data/preprocessed/groundwater_preprocessed.csv', index=False)
    print("Preprocessing complete.")