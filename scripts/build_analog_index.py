import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pickle
import os

def build_analog_index():
    print("Building historical analog index for fast similarity search...")
    
    # 1. Simulate pulling a historical dataset (e.g., from WeatherBench 2)
    # In a real scenario, this would be the same DataFrame you extracted in Colab
    np.random.seed(42)
    num_historical_events = 5000
    
    historical_df = pd.DataFrame({
        'date': pd.date_range(start='2016-01-01', periods=num_historical_events, freq='D'),
        'lat': np.random.uniform(8, 38, size=num_historical_events),
        'lon': np.random.uniform(68, 98, size=num_historical_events),
        'month': np.random.randint(1, 13, size=num_historical_events),
        'forecast_temp': np.random.uniform(280, 315, size=num_historical_events),
        'forecast_wind': np.random.uniform(0, 15, size=num_historical_events),
        'forecast_precip': np.random.uniform(0, 100, size=num_historical_events),
        'is_bust': np.random.choice([0, 1], p=[0.85, 0.15], size=num_historical_events),
        'error_magnitude': np.random.uniform(0, 10, size=num_historical_events)
    })
    
    # 2. Define the exact features used to find "similar" weather patterns
    feature_cols = ['lat', 'lon', 'month', 'forecast_temp', 'forecast_wind', 'forecast_precip']
    X_historical = historical_df[feature_cols].values
    
    # 3. Fit a Nearest Neighbors model (KDTree algorithm is great for this)
    # This acts as our mini Vector Database!
    print("Fitting Nearest Neighbors KDTree...")
    nn_model = NearestNeighbors(n_neighbors=5, algorithm='kd_tree')
    nn_model.fit(X_historical)
    
    # 4. Save the fitted index and the reference data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(current_dir, "..", "bustlib", "data")
    os.makedirs(out_dir, exist_ok=True)
    
    # Save the NearestNeighbors model
    with open(os.path.join(out_dir, "analog_index.pkl"), "wb") as f:
        pickle.dump(nn_model, f)
        
    # Save the historical reference data (so we know WHICH dates were matched)
    historical_df.to_pickle(os.path.join(out_dir, "historical_reference.pkl"))
    
    print(f"Index successfully saved to {out_dir}/")

if __name__ == "__main__":
    build_analog_index()
