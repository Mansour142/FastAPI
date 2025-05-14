import joblib
import pandas as pd
import numpy as np
from pydantic import BaseModel
from .main import PlanRequest, PlanResponse
from math import ceil

def load_model():
    model = joblib.load("../savings_predictor_forest.joblib")
    encoder = joblib.load("../encoder.joblib")
    scaler = joblib.load("../scaler.joblib")
    feature_order = joblib.load("../feature_order.joblib")
    numerical_features = joblib.load("../numerical_features.joblib")
    categorical_features = joblib.load("../categorical_features.joblib")
    return model, encoder, scaler, feature_order, numerical_features, categorical_features

def predict_plan(plan: PlanRequest) -> PlanResponse:
    try:
        model, encoder, scaler, feature_order, numerical_features, categorical_features = load_model()
        
        input_data = {
            'Income': plan.income,
            'Age': plan.age,
            'Dependents': plan.dependents,
            'Occupation': plan.occupation,
            'City_Tier': plan.city_tier,
            'Rent': plan.rent,
            'Loan_Repayment': plan.loanPayment,
            'Insurance': plan.insurance,
            'Groceries': plan.groceries,
            'Transport': plan.transport,
            'Eating_Out': plan.eatingOut,
            'Entertainment': plan.entertainment,
            'Utilities': plan.utilities,
            'Healthcare': plan.healthcare,
            'Education': plan.education,
            'Miscellaneous': plan.otherMoney,
            'Desired_Savings': plan.goalAmount
        }

        total_expenses = sum([
            input_data['Rent'], input_data['Loan_Repayment'], input_data['Insurance'],
            input_data['Groceries'], input_data['Transport'], input_data['Eating_Out'],
            input_data['Entertainment'], input_data['Utilities'],
            input_data['Healthcare'], input_data['Education'], input_data['Miscellaneous']
        ])

        input_data['Disposable_Income'] = input_data['Income'] - total_expenses
        input_data['Desired_Savings_Percentage'] = (
            (input_data['Desired_Savings'] / input_data['Income']) * 100
            if input_data['Income'] != 0 else 0.0
        )

        df = pd.DataFrame([input_data])
        cat_encoded = encoder.transform(df[categorical_features]).toarray()
        num_scaled = scaler.transform(df[numerical_features])
        processed_input = np.concatenate([num_scaled, cat_encoded], axis=1)
        processed_df = pd.DataFrame(processed_input, columns=feature_order)

        prediction = model.predict(processed_df.to_numpy())[0]

        if len(prediction) != 8:
            raise ValueError(f"Expected 8 output values, got {len(prediction)}")

        if plan.utilities == 0:
            prediction[4] = 0.0

        total_savings = sum(prediction)
        months_needed = plan.goalAmount / total_savings if total_savings > 0 else 12

        return PlanResponse(
            groceriesSavings=float(prediction[0]),
            transportSavings=float(prediction[1]),
            eatingOutSavings=float(prediction[2]),
            entertainmentSavings=float(prediction[3]),
            utilitiesSavings=float(prediction[4]),
            healthcareSavings=float(prediction[5]),
            educationSavings=float(prediction[6]),
            otherMoneySavings=float(prediction[7]),
            endDate=ceil(months_needed)
        )
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")
