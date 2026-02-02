# Monitoring avec Prometheus & Grafana

## 🚀 Démarrage

```bash
cd docker
docker-compose up -d
```

## 📊 Accès aux services

- **API**: http://localhost:8002
- **API Docs**: http://localhost:8002/docs
- **API Health**: http://localhost:8002/health
- **Métriques**: http://localhost:8002/metrics
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3001 (admin/admin)
- **cAdvisor**: http://localhost:8080
- **MLflow**: http://localhost:5000

## 📈 Métriques disponibles

### Métriques API
- `api_requests_total` - Nombre total de requêtes (par méthode, endpoint, status)
- `api_request_duration_seconds` - Latence des requêtes (histogramme)
- `api_active_requests` - Nombre de requêtes en cours de traitement
- `api_errors_total` - Nombre total d'erreurs (par endpoint et type)

### Métriques Modèle
- `model_loaded` - État du chargement du modèle (1=chargé, 0=non chargé)
- `model_inference_duration_seconds` - Temps d'inférence du modèle (histogramme)
- `predictions_total` - Nombre de prédictions par outcome

### Métriques Docker (via cAdvisor)
- CPU usage par conteneur
- Memory usage par conteneur
- Network I/O
- Disk I/O
- Container states

## 🔔 Alertes configurées

Les alertes suivantes sont définies dans `monitoring/alerts.yml` :

1. **APIDown** (Critical)
   - Condition : API indisponible
   - Durée : > 1 minute
   - Action : Vérifier les logs du conteneur

2. **ModelNotLoaded** (Critical)
   - Condition : Modèle ML non chargé
   - Durée : > 2 minutes
   - Action : Vérifier MLflow et les artifacts

3. **HighErrorRate** (Warning)
   - Condition : Taux d'erreur > 0.1 req/sec
   - Durée : > 5 minutes
   - Action : Examiner les logs d'erreurs

4. **HighLatency** (Warning)
   - Condition : p95 latence > 1 seconde
   - Durée : > 5 minutes
   - Action : Vérifier les performances de l'API

5. **SlowInference** (Warning)
   - Condition : p95 inférence > 0.5 seconde
   - Durée : > 5 minutes
   - Action : Optimiser le modèle ou les ressources

6. **HighConcurrentRequests** (Warning)
   - Condition : > 10 requêtes simultanées
   - Durée : > 2 minutes
   - Action : Considérer le scaling horizontal

## 🧪 Test des métriques

### Faire une prédiction

```bash
curl -X POST "http://localhost:8002/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Pregnancies": 6,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50
  }'
```

### Voir les métriques

```bash
# Métriques Prometheus
curl http://localhost:8002/metrics

# Health check
curl http://localhost:8002/health
```

### Générer du trafic pour les tests

```bash
# Script pour générer 100 requêtes
for i in {1..100}; do
  curl -X POST "http://localhost:8002/predict" \
    -H "Content-Type: application/json" \
    -d '{
      "Pregnancies": 6,
      "Glucose": 148,
      "BloodPressure": 72,
      "SkinThickness": 35,
      "Insulin": 0,
      "BMI": 33.6,
      "DiabetesPedigreeFunction": 0.627,
      "Age": 50
    }' &
done
wait
```

## 📊 Dashboard Grafana

Le dashboard **ML API Monitoring** est automatiquement provisionné au démarrage.

### Panneaux disponibles :

1. **API Status** - État UP/DOWN de l'API
2. **Model Status** - État LOADED/NOT LOADED du modèle
3. **Request Rate** - Taux de requêtes par minute (par endpoint)
4. **Request Latency** - Percentiles de latence (p50, p95, p99)
5. **Model Inference Time** - Temps d'inférence (p50, p95, p99)
6. **Predictions per Outcome** - Distribution des prédictions
7. **Error Rate** - Taux d'erreurs par endpoint et type
8. **Active Requests** - Gauge des requêtes en cours

### Accéder au dashboard :

1. Ouvrir http://localhost:3001
2. Login : `admin` / Password : `admin`
3. Le dashboard "ML API Monitoring Dashboard" est disponible automatiquement

## 🔍 Queries Prometheus utiles

### Taux de requêtes

```promql
# Taux de requêtes par minute
rate(api_requests_total[1m])

# Taux par endpoint
rate(api_requests_total{endpoint="/predict"}[1m])

# Nombre total de requêtes
sum(api_requests_total)
```

### Latence

```promql
# Latence p95 sur 5 minutes
histogram_quantile(0.95, rate(api_request_duration_seconds_bucket[5m]))

# Latence p99
histogram_quantile(0.99, rate(api_request_duration_seconds_bucket[5m]))

# Latence moyenne
rate(api_request_duration_seconds_sum[5m]) / rate(api_request_duration_seconds_count[5m])
```

### Inférence du modèle

```promql
# Temps d'inférence p95
histogram_quantile(0.95, rate(model_inference_duration_seconds_bucket[5m]))

# Temps d'inférence moyen
rate(model_inference_duration_seconds_sum[5m]) / rate(model_inference_duration_seconds_count[5m])
```

### Erreurs

```promql
# Taux d'erreur
rate(api_errors_total[5m])

# Erreurs par type
sum by (error_type) (rate(api_errors_total[5m]))
```

### Prédictions

```promql
# Nombre de prédictions par outcome
sum by (outcome) (rate(predictions_total[1m]))

# Total des prédictions
sum(predictions_total)
```

### Métriques système (cAdvisor)

```promql
# CPU usage du conteneur ml-api
rate(container_cpu_usage_seconds_total{name="ml-api"}[1m])

# Memory usage
container_memory_usage_bytes{name="ml-api"}

# Network I/O
rate(container_network_receive_bytes_total{name="ml-api"}[1m])
rate(container_network_transmit_bytes_total{name="ml-api"}[1m])
```

## 🛠️ Troubleshooting

### Prometheus ne collecte pas les métriques

```bash
# Vérifier que l'API expose les métriques
curl http://localhost:8002/metrics

# Vérifier les targets dans Prometheus
# Ouvrir http://localhost:9090/targets
# ml-api devrait être UP

# Vérifier les logs Prometheus
docker logs prometheus
```

### Grafana ne se connecte pas à Prometheus

```bash
# Vérifier que les conteneurs sont sur le même réseau
docker network inspect mlops-network

# Tester la connexion depuis Grafana
docker exec grafana curl http://prometheus:9090/-/healthy

# Vérifier les logs Grafana
docker logs grafana
```

### Dashboard vide ou sans données

- Attendre quelques minutes pour collecter les données initiales
- Faire des requêtes à l'API pour générer des métriques
- Ajuster la plage de temps dans Grafana (dernières 30 min)
- Vérifier que Prometheus collecte bien les métriques

### cAdvisor ne démarre pas

Sur Windows, cAdvisor peut avoir des limitations. Solutions :

```bash
# Option 1 : Retirer cAdvisor du docker-compose
# Commenter ou supprimer le service cadvisor

# Option 2 : Utiliser une alternative
# Utiliser Docker stats API ou Windows Performance Counters
```

### Alertes ne se déclenchent pas

```bash
# Vérifier que les rules sont chargées
# Ouvrir http://localhost:9090/rules

# Forcer le rechargement de la config
curl -X POST http://localhost:9090/-/reload

# Vérifier les logs
docker logs prometheus
```

## 📚 Ressources supplémentaires

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [PromQL Guide](https://prometheus.io/docs/prometheus/latest/querying/basics/)
- [FastAPI Monitoring](https://fastapi.tiangolo.com/advanced/advanced-middleware/)

## 🔄 Mise à jour du monitoring

### Ajouter une nouvelle métrique

1. Modifier `api/main.py` pour ajouter la métrique
2. Redémarrer le conteneur : `docker-compose restart api`
3. Créer un nouveau panneau dans Grafana

### Modifier les alertes

1. Éditer `monitoring/alerts.yml`
2. Recharger Prometheus : `curl -X POST http://localhost:9090/-/reload`
3. Vérifier dans http://localhost:9090/rules

### Mettre à jour le dashboard

1. Modifier directement dans Grafana UI
2. Exporter le JSON depuis Grafana
3. Remplacer le contenu dans `docker/monitoring/grafana/provisioning/dashboards/json/ml-api-dashboard.json`

## 🎯 Best Practices

1. **Monitoring continu** : Consulter le dashboard régulièrement
2. **Seuils d'alertes** : Ajuster selon votre usage réel
3. **Rétention des données** : Configurer selon vos besoins de storage
4. **Sécurité** : Changer les mots de passe par défaut en production
5. **Backup** : Sauvegarder régulièrement les configurations Grafana
