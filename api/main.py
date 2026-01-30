import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd


app = FastAPI(
    title="Diabetes Prediction API",
    description="API de prédiction du diabète avec MLflow",
    version="1.0.0"
)


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
    try:
        model = mlflow.pyfunc.load_model(f"models:/{model_name}/{stage}")
        print(f"✅ Model {model_name} (Stage: {stage}) loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(f"⚠️  Attempting to load latest version...")
        try:
            # Essayer de charger la dernière version si Production n'existe pas
            model = mlflow.pyfunc.load_model(f"models:/{model_name}/latest")
            print(f"✅ Model {model_name} (latest) loaded successfully!")
        except Exception as e2:
            print(f"❌ Error loading latest model: {e2}")


@app.get("/")
def root():
    return {
        "message": "API de prédiction du diabète",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "docs": "/docs"
        }
    }


@app.post("/predict")
def predict(data: DiabetesInput):

    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Modèle non chargé. Veuillez attendre le démarrage complet de l'API."
        )
    
    try:
        # Convertir les données d'entrée en DataFrame
        df = pd.DataFrame([data.dict()])
        
        # Assurer l'ordre des colonnes
        column_order = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        df = df[column_order]
        
        prediction = model.predict(df)
        
        try:
            probabilities = model.predict_proba(df)
            return {
                "prediction": int(prediction[0]),
                "cluster": int(prediction[0]),
                "probabilities": probabilities[0].tolist(),
                "input_data": data.dict()
            }
        except AttributeError:
            return {
                "prediction": int(prediction[0]),
                "cluster": int(prediction[0]),
                "input_data": data.dict()
            }
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la prédiction: {str(e)}"
        )