from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .model import predict_plan
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PlanRequest(BaseModel):
    age: int
    dependents: int
    occupation: str
    city_tier: str
    goalAmount: float
    income: float
    rent: float
    loanPayment: float
    insurance: float
    groceries: float
    transport: float
    eatingOut: float
    education: float
    entertainment: float
    utilities: float
    healthcare: float
    otherMoney: float

class PlanResponse(BaseModel):
    groceriesSavings: float
    transportSavings: float
    eatingOutSavings: float
    entertainmentSavings: float
    utilitiesSavings: float
    healthcareSavings: float
    educationSavings: float
    otherMoneySavings: float
    endDate: int

latest_plan = PlanRequest(
    age=0, dependents=0, occupation="", city_tier="", goalAmount=0.0, income=0.0,
    rent=0.0, loanPayment=0.0, insurance=0.0, groceries=0.0, transport=0.0,
    eatingOut=0.0, education=0.0, entertainment=0.0, utilities=0.0,
    healthcare=0.0, otherMoney=0.0
)

@app.post("/receive-data")
async def receive_data(request: Request):
    try:
        data = await request.json()
        global latest_plan
        latest_plan = PlanRequest(**data)
        return {"status": "success", "message": "Data received"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/send-data", response_model=PlanResponse)
async def send_data():
    try:
        return predict_plan(latest_plan)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "FastAPI ML App is running!"}
