from fastapi import FastAPI, HTTPException
from app.route import customer



app = FastAPI(title="Insurance Premimum prediction", description="API for predicting insurance charges")
app.include_router(customer.router)


