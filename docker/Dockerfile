FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p api
COPY api/main.py ./api/
COPY notebooks/ ./notebooks/
COPY data/ ./data/

EXPOSE 8000 8888

CMD ["bash", "-c", "uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload & jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password='' & wait"]