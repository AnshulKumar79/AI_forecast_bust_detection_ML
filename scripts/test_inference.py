import pandas as pd
import bustlib
import json

def run_tests():
    print("--- 1. Testing model_info() ---")
    info = bustlib.model_info()
    print(json.dumps(info, indent=2))
    
    print("\n--- 2. Testing predict() ---")
    # Simulate an incoming weather forecast for New Delhi (Lat: 28.6, Lon: 77.2)
    test_df = pd.DataFrame({
        "region_id": [101, 102],
        "variable": ["tmax", "rainfall"],
        "lead_day": [3, 5],
        "lat": [28.6, 19.0], # Delhi, Mumbai
        "lon": [77.2, 72.8],
        "month": [5, 7],     # May (Heat), July (Monsoon)
        "forecast_temp": [314.15, 303.15], # ~41 C, 30 C
        "forecast_wind": [2.5, 8.0],
        "forecast_precip": [0.0, 45.5]
    })
    
    preds = bustlib.predict(test_df)
    print(preds.to_string())
    
    print("\n--- 3. Testing explain() ---")
    explanations = bustlib.explain(test_df, preds)
    print(json.dumps(explanations, indent=2, ensure_ascii=False))
    
    print("\n--- 4. Testing find_analogs() ---")
    # Find analogs for the first row (Delhi Heatwave)
    analogs = bustlib.find_analogs(test_df.iloc[[0]], k=2)
    print(json.dumps(analogs, indent=2))
    
    print("\nAll internal package tests passed successfully! ✅")

if __name__ == "__main__":
    run_tests()
