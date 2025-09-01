import joblib
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from app.schema.customer import CustomerData
import pandas as pd


router = APIRouter(prefix="/customer", tags=["Customer Insurance Prediction"])


@router.post("/predict")
def predict(customer: CustomerData):
    model_pkl_file = "app\model\model_rf.pkl"
    model_rf = joblib.load(model_pkl_file)

    input_data = pd.DataFrame([[customer.age, 
                                customer.bmi, 
                                customer.children, 
                                customer.smoker]],
                                columns=['age', 'bmi', 'children', 'smoker'])

    prediction = model_rf.predict(input_data)

    prediction = int(prediction[0])
    if prediction < 0:
        raise HTTPException(status_code=400, detail="Invalid prediction")

    return JSONResponse(content={"Insurance Charges suggested": prediction},
        status_code= status.HTTP_200_OK
    )