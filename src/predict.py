from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# Load the trained model
with open("artifacts/model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()


class SalesInput(BaseModel):
    ProductCategory: str
    Region: str
    CustomerSegment: str
    IsPromotionApplied: str  
    ProductionCost: float
    MarketingSpend: float
    SeasonalDemandIndex: float
    CompetitorPrice: float
    CustomerRating: float
    EconomicIndex: float
    StoreCount: int

@app.get("/")
def home():
    return {"message": "Sales Revenue Prediction API is running"}

@app.post("/predict")
def predict(data: SalesInput):
    
    input_data = pd.DataFrame([data.dict()])

    
    prediction = model.predict(input_data)[0]

    return {"Predicted_SalesRevenue": round(float(prediction), 2)}
