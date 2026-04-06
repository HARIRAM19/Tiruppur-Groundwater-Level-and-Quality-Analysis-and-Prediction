# generate_data.py
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances
from scipy.linalg import cholesky
from datetime import datetime

# Load wells
gwl = pd.read_csv('data/GWL_Well_Data.csv')
gwq = pd.read_csv('data/GWQ_Well_Data.csv')

# Combine and deduplicate by coordinates
wells = pd.concat([
    gwl[['Block', 'Village', 'Latitude', 'Longitude']].rename(columns={'Block': 'block'}),
    gwq[['Block', 'Village', 'Latitude', 'Longitude']]
]).drop_duplicates(subset=['Latitude', 'Longitude']).reset_index(drop=True)
wells['well_id'] = wells.index

# Parameters
np.random.seed(42)
n_wells = len(wells)
dates = pd.date_range('1996-01-01', '2025-12-31', freq='M')
n_time = len(dates)

# Distance matrix (km)
coords_rad = np.radians(wells[['Latitude', 'Longitude']].values)
dist_km = haversine_distances(coords_rad) * 6371

# Spatial covariance (exponential)
phi = 30  # range in km
K = np.exp(-dist_km / phi)
K += np.eye(n_wells) * 1e-6  # nugget
L = cholesky(K, lower=True)

# Time components
t = np.arange(n_time)
trend_level = -0.15 * t / 12                     # m per year decline
trend_tds = 20 * t / 12                           # TDS increase (mg/L per year)
seasonal_level = 3.0 * np.sin(2 * np.pi * t / 12 + 1.5)
seasonal_temp = 4.0 * np.sin(2 * np.pi * t / 12 + 2.0)

# Baseline per well (uniform within given ranges)
baseline_level = np.random.uniform(10, 30, n_wells)
baseline_ph = np.random.uniform(7.2, 8.5, n_wells)
baseline_tds = np.random.uniform(750, 3600, n_wells)
baseline_temp = 28.0  # base temperature

# Generate spatial random effects (one per time step)
spatial_effects = np.zeros((n_time, n_wells))
for i in range(n_time):
    z = np.random.normal(size=n_wells)
    spatial_effects[i] = L @ z

# Create dataframe
rows = []
for wi in range(n_wells):
    for ti, date in enumerate(dates):
        # Water level
        level = (baseline_level[wi] + trend_level[ti] + seasonal_level[ti] +
                 spatial_effects[ti, wi] * 0.5 + np.random.normal(0, 0.3))
        level = np.clip(level, 5, 35)  # keep within plausible bounds

        # pH (little trend, mainly random + spatial)
        ph = (baseline_ph[wi] +
              0.1 * np.sin(2 * np.pi * ti / 12) +
              spatial_effects[ti, wi] * 0.1 +
              np.random.normal(0, 0.1))
        ph = np.clip(ph, 6.5, 9.0)

        # TDS
        tds = (baseline_tds[wi] + trend_tds[ti] +
               150 * np.sin(2 * np.pi * ti / 12 + 1.5) +
               spatial_effects[ti, wi] * 50 +
               np.random.normal(0, 50))
        tds = np.clip(tds, 500, 4000)

        # EC = 1.46 * TDS
        ec = 1.46 * tds + np.random.normal(0, 20)

        # Turbidity (Gamma distribution, correlated with TDS?)
        # Use shape=2, scale=5 => mean=10, but vary with location/time
        turb_base = np.random.gamma(2, 5)
        turb = turb_base + 0.01 * tds + spatial_effects[ti, wi] * 2 + np.random.normal(0, 2)
        turb = np.clip(turb, 0, 100)

        # Temperature
        temp = baseline_temp + seasonal_temp[ti] + spatial_effects[ti, wi] * 0.5 + np.random.normal(0, 0.5)

        rows.append([
            wells.loc[wi, 'well_id'],
            wells.loc[wi, 'block'],
            wells.loc[wi, 'Village'],
            wells.loc[wi, 'Latitude'],
            wells.loc[wi, 'Longitude'],
            date,
            level, ph, ec, tds, turb, temp
        ])

df = pd.DataFrame(rows, columns=[
    'well_id', 'block', 'village', 'lat', 'lon',
    'timestamp', 'water_level', 'ph', 'ec', 'tds', 'turbidity', 'temperature'
])

# Save raw data
df.to_csv('data/synthetic_raw/groundwater_monthly.csv', index=False)
print("Raw synthetic data generated.")