import mlflow.pyfunc
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import pandas as pd
import time
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response


app = FastAPI(
    title="Diabetes Prediction API",
    description="API de prédiction du diabète avec MLflow et monitoring Prometheus",
    version="1.0.0"
)

REQUEST_COUNT = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status']
)

PREDICTION_COUNT = Counter(
    'predictions_total',
    'Total number of predictions made',
    ['outcome']
)

ERROR_COUNT = Counter(
    'api_errors_total',
    'Total number of API errors',
    ['endpoint', 'error_type']
)

REQUEST_LATENCY = Histogram(
    'api_request_duration_seconds',
    'API request latency in seconds',
    ['method', 'endpoint']
)

INFERENCE_TIME = Histogram(
    'model_inference_duration_seconds',
    'Model inference time in seconds',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

# Gauges pour l'état
MODEL_LOADED = Gauge(
    'model_loaded',
    'Whether the ML model is loaded (1) or not (0)'
)

ACTIVE_REQUESTS = Gauge(
    'api_active_requests',
    'Number of requests currently being processed'
)


@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    """Middleware pour monitorer toutes les requêtes"""
    ACTIVE_REQUESTS.inc()
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Calculate metrics
    duration = time.time() - start_time
    
    # Record metrics
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    ACTIVE_REQUESTS.dec()
    
    return response


class DiabetesInput(BaseModel):
    """Modèle Pydantic pour la validation des données d'entrée"""
    Pregnancies: float = Field(..., description="Nombre de grossesses", ge=0)
    Glucose: float = Field(..., description="Niveau de glucose", ge=0)
    BloodPressure: float = Field(..., description="Pression artérielle (mm Hg)", ge=0)
    SkinThickness: float = Field(..., description="Épaisseur de la peau (mm)", ge=0)
    Insulin: float = Field(..., description="Niveau d'insuline (mu U/ml)", ge=0)
    BMI: float = Field(..., description="Indice de masse corporelle", ge=0)
    DiabetesPedigreeFunction: float = Field(..., description="Fonction de pedigree du diabète", ge=0)
    Age: float = Field(..., description="Âge", ge=0)
    
    class Config:
        schema_extra = {
            "example": {
                "Pregnancies": 6,
                "Glucose": 148,
                "BloodPressure": 72,
                "SkinThickness": 35,
                "Insulin": 0,
                "BMI": 33.6,
                "DiabetesPedigreeFunction": 0.627,
                "Age": 50
            }
        }


model = None


@app.on_event("startup")
def loadmodel():
    """Charge le modèle depuis MLflow au démarrage de l'API"""
    global model
    model_name = "DiabetesClusterClassifier"
    stage = "Production"
    
    MODEL_LOADED.set(0)
    
    try:
        model = mlflow.pyfunc.load_model(f"models:/{model_name}/{stage}")
        print(f"✅ Model {model_name} (Stage: {stage}) loaded successfully!")
        MODEL_LOADED.set(1)
    except Exception as e:
        print(f" Error loading model: {e}")
        print(f"  Attempting to load latest version...")
        try:
            model = mlflow.pyfunc.load_model(f"models:/{model_name}/latest")
            print(f"Model {model_name} (latest) loaded successfully!")
            MODEL_LOADED.set(1)
        except Exception as e2:
            print(f"Error loading latest model: {e2}")
            MODEL_LOADED.set(0)


@app.get("/metrics")
def metrics():
    """Endpoint pour exposer les métriques Prometheus"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(data: DiabetesInput):
    """Endpoint de prédiction avec monitoring"""
    if model is None:
        ERROR_COUNT.labels(endpoint="/predict", error_type="model_not_loaded").inc()
        raise HTTPException(
            status_code=503,
            detail="Modèle non chargé. Veuillez attendre le démarrage complet de l'API."
        )
    
    try:
        inference_start = time.time()
        
        df = pd.DataFrame([data.dict()])
        
        column_order = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        df = df[column_order]
        
        prediction = model.predict(df)
        
        inference_duration = time.time() - inference_start
        INFERENCE_TIME.observe(inference_duration)
        
        outcome = int(prediction[0])
        PREDICTION_COUNT.labels(outcome=outcome).inc()
        
        try:
            probabilities = model.predict_proba(df)
            return {
                "prediction": outcome,
                "cluster": outcome,
                "probabilities": probabilities[0].tolist(),
                "inference_time_seconds": round(inference_duration, 4),
                "input_data": data.dict()
            }
        except AttributeError:
            return {
                "prediction": outcome,
                "cluster": outcome,
                "inference_time_seconds": round(inference_duration, 4),
                "input_data": data.dict()
            }
            
    except Exception as e:
        ERROR_COUNT.labels(endpoint="/predict", error_type=type(e).__name__).inc()
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la prédiction: {str(e)}"
        )