import pandas as pd
import numpy as np

def compute_bust_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the target variable `is_bust` based on operational thresholds 
    outlined in the SIH ML Engineering Guide.
    
    Expected columns in df:
    - region_id (int)
    - variable (str: 'rainfall', 'tmax', 'tmin', 'wind')
    - nwp_forecast (float)
    - ground_truth_actual (float)
    """
    df = df.copy()
    
    # Calculate absolute forecast error
    df['absolute_error'] = (df['nwp_forecast'] - df['ground_truth_actual']).abs()
    
    # Initialize the bust target
    df['is_bust'] = 0
    
    # Apply operational threshold floors
    # Rainfall: Error > 35.0 mm/24h
    rainfall_bust_idx = (df['variable'] == 'rainfall') & (df['absolute_error'] > 35.0)
    df.loc[rainfall_bust_idx, 'is_bust'] = 1
    
    # Maximum Temperature: Error > 3.5 C
    tmax_bust_idx = (df['variable'] == 'tmax') & (df['absolute_error'] > 3.5)
    df.loc[tmax_bust_idx, 'is_bust'] = 1
    
    # Minimum Temperature: Error > 3.0 C
    tmin_bust_idx = (df['variable'] == 'tmin') & (df['absolute_error'] > 3.0)
    df.loc[tmin_bust_idx, 'is_bust'] = 1
    
    # Wind Speed: Error > 6.0 m/s
    wind_bust_idx = (df['variable'] == 'wind') & (df['absolute_error'] > 6.0)
    df.loc[wind_bust_idx, 'is_bust'] = 1
    
    # NOTE: To be fully rigorous, we must also check if the absolute_error > 90th percentile 
    # of historical errors for that specific region/variable/season.
    # This acts as a basic baseline constraint.
    
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers the core feature matrix for LightGBM model.
    """
    # Example template for adding features:
    # 1. Multi-Model Spread (e.g. |ECMWF_IFS - GFS| spread in rain, temp)
    # df['ecmwf_gfs_spread'] = (df['ecmwf_val'] - df['gfs_val']).abs()
    
    # 2. Dynamic Instability
    # df['cape'] = df['cape_from_nwp']
    # df['vertical_wind_shear'] = df['shear_0_6km']
    
    # 3. Severe Synoptics
    # df['sst_anomaly'] = df['sst'] - df['sst_climatology']
    
    # 4. Lead Time Dynamics
    # df['lead_day'] = df['lead_time_hours'] // 24
    
    return df

if __name__ == "__main__":
    print("Dataset preparation script initialized. Ready to process NWP data.")
