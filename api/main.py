import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import time


app = FastAPI(
    title="Diabetes Prediction API",
    description="API de prédiction du diabète avec MLflow",
    version="1.0.0"
)

# Prometheus metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('api_request_duration_seconds', 'API request duration')
PREDICTION_COUNT = Counter('predictions_total', 'Total predictions made', ['result'])


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
scaler = None

@app.on_event("startup")
def loadmodel():
    global model, scaler
    model_name = "DiabetesClusterClassifier"
    stage = "Production"
    try:
        model = mlflow.pyfunc.load_model(f"models:/{model_name}/{stage}")
        run_id = model.metadata.run_id
        scaler = mlflow.sklearn.load_model(f"runs:/{run_id}/scaler")
        print(f"Model and scaler loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")


@app.post("/predict")
def predict(data: DiabetesInput):
    start_time = time.time()
    REQUEST_COUNT.labels(method='POST', endpoint='/predict').inc()
    
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503, 
            detail="Modèle ou scaler non chargé. Veuillez attendre le démarrage complet de l'API."
        )
    
    try:
        df = pd.DataFrame([data.dict()])
        
        column_order = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        df = df[column_order]
        
        df_scaled = scaler.transform(df)
        prediction = model.predict(df_scaled)
        
        # Record prediction metrics
        PREDICTION_COUNT.labels(result=str(prediction[0])).inc()
        
        probabilities = None
        try:
            probabilities = model.predict_proba(df_scaled)
        except AttributeError:
            pass
        
        result = {
            "prediction": int(prediction[0]),
            "cluster": int(prediction[0]),
            "input_data": data.dict()
        }
        
        if probabilities is not None:
            result["probabilities"] = probabilities[0].tolist()
        
        # Record request duration
        REQUEST_DURATION.observe(time.time() - start_time)
        return result
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la prédiction: {str(e)}"
        )