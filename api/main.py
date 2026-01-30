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
scaler = None

@app.on_event("startup")
def loadmodel():
    global model, scaler
    model_name = "DiabetesClusterClassifier"
    stage = "Production"
    try:
        model = mlflow.pyfunc.load_model(f"models:/{model_name}/{stage}")
        # Load the scaler from the same run
        run_id = model.metadata.run_id
        scaler = mlflow.sklearn.load_model(f"runs:/{run_id}/scaler")
        print(f"Model and scaler loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")



@app.post("/predict")
def predict(data: DiabetesInput):
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
        
        try:
            probabilities = model.predict_proba(df_scaled)
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