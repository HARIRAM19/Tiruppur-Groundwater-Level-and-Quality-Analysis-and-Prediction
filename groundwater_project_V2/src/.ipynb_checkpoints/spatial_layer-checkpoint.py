# spatial_layer.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def add_spatial_features(df, wells_metadata):
    """
    df: preprocessed data (with well_id, lat, lon, block, etc.)
    wells_metadata: dataframe with well_id, block, lat, lon (from original wells)
    Returns df with added spatial features.
    """
    # Merge to ensure block is consistent (if not already)
    # Assuming df already has block, but we can reattach from metadata if needed.

    # 1. Block encoding (one-hot)
    block_encoder = OneHotEncoder(sparse=False)
    block_encoded = block_encoder.fit_transform(df[['block']])
    block_cols = [f"block_{cat}" for cat in block_encoder.categories_[0]]
    block_df = pd.DataFrame(block_encoded, columns=block_cols, index=df.index)
    df = pd.concat([df, block_df], axis=1)

    # 2. Simple geohash (first 4 characters) – approximate zone tagging
    import geohash2  # need to install python-geohash
    df['geohash'] = df.apply(lambda row: geohash2.encode(row['lat'], row['lon'], precision=4), axis=1)

    # 3. Spatial feature: distance to a reference point (e.g., Tiruppur city center)
    ref_lat, ref_lon = 11.1085, 77.3411  # approximate center of Tiruppur
    from math import radians, sin, cos, sqrt, atan2
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c
    df['dist_to_tiruppur_km'] = df.apply(lambda row: haversine(row['lat'], row['lon'], ref_lat, ref_lon), axis=1)

    # 4. You could also add elevation if you have DEM data – omitted here.

    return df

if __name__ == "__main__":
    pre = pd.read_csv('data/preprocessed/groundwater_preprocessed.csv', parse_dates=['timestamp'])
    # Load original wells to get metadata if needed (but pre already has block, lat, lon)
    # Ensure block is string
    pre['block'] = pre['block'].astype(str)
    spatial_df = add_spatial_features(pre, None)  # metadata not needed separately
    spatial_df.to_csv('data/spatial/groundwater_spatial.csv', index=False)
    print("Spatial features added.")