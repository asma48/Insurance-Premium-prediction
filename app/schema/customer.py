from pydantic import BaseModel



class CustomerData(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str