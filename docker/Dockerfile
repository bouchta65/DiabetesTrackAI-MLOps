FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY tests/ ./tests/

RUN mkdir -p /app/models /app/mlruns

ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=file:///app/mlruns
ENV MODEL_NAME=diabetes_model
ENV MODEL_VERSION=latest

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]