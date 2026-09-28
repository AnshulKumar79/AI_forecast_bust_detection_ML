FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by LightGBM (OpenMP)
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies including the web server components
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn shap lightgbm scikit-learn

# Copy the rest of the application
COPY . .

# Expose the API port
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
