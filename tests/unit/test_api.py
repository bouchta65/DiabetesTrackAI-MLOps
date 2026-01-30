import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np
from api.main import app

client = TestClient(app)

def test_predict_success():
    """Test successful prediction"""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
    
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    
    with patch('api.main.model', mock_model), patch('api.main.scaler', mock_scaler):
        response = client.post("/predict", json={
            "Pregnancies": 6, "Glucose": 148, "BloodPressure": 72,
            "SkinThickness": 35, "Insulin": 0, "BMI": 33.6,
            "DiabetesPedigreeFunction": 0.627, "Age": 50
        })
        
        assert response.status_code == 200
        assert response.json()["prediction"] == 1

def test_predict_model_not_loaded():
    """Test when model not loaded"""
    with patch('api.main.model', None):
        response = client.post("/predict", json={
            "Pregnancies": 1, "Glucose": 120, "BloodPressure": 70,
            "SkinThickness": 20, "Insulin": 80, "BMI": 25.5,
            "DiabetesPedigreeFunction": 0.5, "Age": 25
        })
        
        assert response.status_code == 503

def test_predict_invalid_input():
    """Test with invalid input"""
    response = client.post("/predict", json={
        "Pregnancies": -1, "Glucose": 120, "BloodPressure": 70,
        "SkinThickness": 20, "Insulin": 80, "BMI": 25.5,
        "DiabetesPedigreeFunction": 0.5, "Age": 25
    })
    
    assert response.status_code == 422