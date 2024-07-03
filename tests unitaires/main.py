import pickle
from lightgbm import LGBMClassifier
import pandas as pd
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import joblib
import nest_asyncio
import asyncio

nest_asyncio.apply()

# Charger le modèle LightGBM
model = joblib.load("C:\\Users\\amal9\\OneDrive\\Documents\\9-Openclassroom\\8-PROJET 7\\GIT\\lightgbm_model_df1.pkl")
df = pd.read_csv("C:\\Users\\amal9\\OneDrive\\Documents\\9-Openclassroom\\8-PROJET 7\\GIT\\df1_final.csv")

def predict(customer_id):
    customer_df = df[df.SK_ID_CURR == customer_id]
    if customer_df.shape[0] == 0:
        return -1
    X = customer_df.iloc[:, :-2]
    score = model.predict_proba(X)[0, 1]
    return round(score*100, 2)

app = FastAPI()

class post_data(BaseModel):
    customer_id: int

@app.get('/')
async def root():
    return {"Message": "Welcome to Score Prediction API."}

@app.post("/predict")
def score(input_data: post_data):
    score = predict(input_data.customer_id)
    return {"score": score}

if __name__ == "__main__":
    config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="info")
    server = uvicorn.Server(config)
    asyncio.run(server.serve())

