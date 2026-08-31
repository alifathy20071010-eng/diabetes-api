import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

model = joblib.load("logistic.pkl")
scaler = joblib.load("scaler.pkl")

COLUMNS_ORDER = [
    'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level',
    'blood_glucose_level', 'gender_Male', 'gender_Other',
    'smoking_history_current', 'smoking_history_ever',
    'smoking_history_former', 'smoking_history_never',
    'smoking_history_not current'
]


class DiabetesInput(BaseModel):
    gender: str
    age: int
    hypertension: int
    heart_disease: int
    smoking_history: str
    bmi: float
    HbA1c_level: float
    blood_glucose_level: int


@app.post("/predict")
def predict_diabetes(data: DiabetesInput):

    row = {
        "age": data.age,
        "hypertension": data.hypertension,
        "heart_disease": data.heart_disease,
        "bmi": data.bmi,
        "HbA1c_level": data.HbA1c_level,
        "blood_glucose_level": data.blood_glucose_level,
        "gender_Male": 1 if data.gender.lower() == "male" else 0,
        "gender_Other": 1 if data.gender.lower() == "other" else 0,
        "smoking_history_current": 1 if data.smoking_history == "current" else 0,
        "smoking_history_ever": 1 if data.smoking_history == "ever" else 0,
        "smoking_history_former": 1 if data.smoking_history == "former" else 0,
        "smoking_history_never": 1 if data.smoking_history == "never" else 0,
        "smoking_history_not current": 1 if data.smoking_history == "not current" else 0,
    }

    input_df = pd.DataFrame([row])
    input_df = input_df[COLUMNS_ORDER]
    print(input_df)
    print(input_df.dtypes)


    input_scaled = scaler.transform(input_df)
    print(input_scaled)

    prediction = model.predict(input_scaled)[0]
    print("prediction = " , prediction)

    return {
        "prediction": int(prediction),
        "result": "Diabetic" if prediction == 1 else "Not Diabetic"
    }