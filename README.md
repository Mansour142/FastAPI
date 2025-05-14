# FastAPI ML App

This is a FastAPI application that predicts potential savings across various expense categories using a RandomForestRegressor model.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run locally: `uvicorn app.main:app --reload`
3. Access the API at `http://localhost:8000` or `/docs` for Swagger UI.

## Endpoints
- `GET /`: Check if the API is running.
- `POST /receive-data`: Submit financial data.
- `GET /send-data`: Get savings predictions.

## Deployment
Deployed on Render using the provided `render.yaml` and `Dockerfile`.
