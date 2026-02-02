# DiabetesTrackAI-MLOps

## Contexte du projet

L’objectif de ce projet est de mettre en place un pipeline **MLOps de bout en bout** pour un service de prédiction basé sur un modèle de Machine Learning. Ce service est exposé via une **API REST** et doit répondre à des critères de :

- Fiabilité  
- Rapidité (latence p95 < 1 s)  
- Maintenabilité  

Pour gérer l’ensemble du cycle de vie du modèle — entraînement, déploiement et monitoring — nous utilisons :

- **GitHub Actions** : CI/CD, tests, validation de code et déploiement automatisé  
- **MLflow** : suivi des expérimentations, versionnement des modèles, registre de modèles  
- **Prometheus & Grafana** : monitoring en temps réel de la santé et des performances de l’API  

---

## Fonctionnalités

### 1. MLflow

- Intégration de **MLflow Tracking** dans le script d’entraînement  
- Logging des **paramètres**, **métriques** et **artefacts du modèle** à chaque run  
- Comparaison des runs via l’UI ou l’API MLflow  
- Enregistrement du meilleur modèle dans le **Model Registry**  
- Promotion des modèles de **Staging à Production**  
- Endpoint de prédiction via **FastAPI** :
  - Chargement du modèle depuis MLflow au démarrage de l’API  
  - Validation des données d’entrée avec **Pydantic**  
  - Tests via Swagger UI  

### 2. Pipeline CI/CD avec GitHub Actions

- Workflow déclenché à chaque push  
- Validation automatique de :
  - La qualité des données  
  - Les performances du modèle  
  - La qualité du code  
- Déploiement continu après validation  
- Création d’une **image Docker** du service (API + modèle)  
- Versionnement de l’image Docker associée au modèle déployé  

### 3. Monitoring avec Prometheus & Grafana

- Exposition des métriques applicatives via `/metrics`  
- Configuration de Prometheus pour collecter ces métriques  
- Connexion de Grafana à Prometheus  
- Dashboard Grafana pour suivre :
  - Nombre de requêtes  
  - Latence  
  - Taux d’erreurs  
  - Temps d’inférence  
  - CPU, RAM, réseau des conteneurs Docker  
- Configuration d’alertes sur métriques critiques (optionnel)  

---

## Technologies utilisées

- **Python**  
- **FastAPI**  
- **Docker**  
- **MLflow**  
- **Prometheus**  
- **Grafana**  
- **GitHub Actions**  



