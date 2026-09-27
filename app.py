from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import bustlib
import uvicorn
from typing import List

app = FastAPI(
    title="AI Forecast Bust Detection API",
    description="Machine Learning service for predicting numerical weather prediction (NWP) failures in India.",
    version="1.0"
)

# Define the expected JSON payload from the backend team
class ForecastInput(BaseModel):
    region_id: int
    variable: str
    lead_day: int
    lat: float
    lon: float
    month: int
    forecast_temp: float
    forecast_wind: float
    forecast_precip: float

@app.get("/api/info")
def get_model_info():
    """Returns ML metadata, calibration scores, and versioning info."""
    return bustlib.model_info()

@app.post("/api/predict")
def predict_bust(inputs: List[ForecastInput]):
    """
    Receives an array of weather forecasts, processes them through LightGBM + Isotonic Regression,
    generates SHAP explanations, and finds historical analogs.
    """
    try:
        # 1. Convert JSON list to Pandas DataFrame for the ML package
        df = pd.DataFrame([inp.model_dump() for inp in inputs])
        
        # 2. Get Calibrated Predictions & Confidence Bands
        preds_df = bustlib.predict(df)
        
        # 3. Generate SHAP AI Explanations
        explanations = bustlib.explain(df, preds_df)
        
        # 4. Construct JSON Response
        results = []
        for i, row in preds_df.iterrows():
            result = row.to_dict()
            result["explanation"] = explanations[i]
            
            # For demonstration, we fetch historical analogs for every requested row
            # In a massive production batch, you might only fetch analogs if band == "High" risk
            result["analogs"] = bustlib.find_analogs(df.iloc[[i]], k=3)
            
            results.append(result)
            
        return {"status": "success", "predictions": results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Starting ML API Server on http://localhost:8000")
    print("Test the API docs at http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
