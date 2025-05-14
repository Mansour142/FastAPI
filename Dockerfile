FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8
WORKDIR /app
COPY ./app /app
COPY requirements.txt /app/
COPY data.csv /app/
COPY savings_predictor_forest.joblib /app/
COPY categorical_features.joblib /app/
COPY encoder.joblib /app/
COPY feature_order.joblib /app/
COPY numerical_features.joblib /app/
COPY scaler.joblib /app/
RUN pip install -r requirements.txt
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
