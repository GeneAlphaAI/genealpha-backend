from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
import numpy as np
import datetime
import xgboost as xgb

from config import settings
from db import get_db, Base, engine
from models import Prediction
from logger import logger
from model_loader import load_xgb, load_lgb, load_rf

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI()

# Load models once
xgb_model = load_xgb()
lgb_model = load_lgb()
rf_model  = load_rf()

# Placeholder mapping: token name → token hash
TOKEN_HASHES = {
    "BTC": "0xabc123placeholder",
    "ETH": "0xdef456placeholder",
    "XRP": "0xghi789placeholder",
}

# Pydantic schema for incoming request
from pydantic import BaseModel

class PredictRequest(BaseModel):
    token: str


def fetch_features(token_hash: str) -> np.ndarray:
    """
    TODO: replace with real feature fetching logic using token_hash.
    For now, returns a dummy 1×9 feature vector matching the model's inputs.
    """
    return np.array([[
        np.random.randint(0, 10),             # Buy Count
        np.random.randint(0, 10),             # Sell Count
        np.random.randint(1, 50),             # Active Address Count
        np.random.uniform(0.00003, 0.00008),  # Avg Token Price (USD)
        np.random.uniform(1e4, 1e8),          # Token Volume
        np.random.uniform(10, 5e3),           # Token Volume (USD)
        np.random.uniform(1000, 3000),        # ETH Price (USD)
        np.random.uniform(20000, 120000),     # BTC Price (USD)
        np.random.uniform(-0.1, 0.1),         # momentum
    ]])

@app.post("/predict")
async def predict(req: PredictRequest,
    db: Session = Depends(get_db)):

    token = req.token.upper()
    token_hash = TOKEN_HASHES.get(token)
    if token_hash is None:
        raise HTTPException(status_code=400, detail=f"Unknown token: {token}")

    # Fetch (dummy) features
    features = fetch_features(token_hash)
    now = datetime.datetime.utcnow()

    try:
        # XGBoost expects a DMatrix
        # xgb_log = float(xgb_model.predict(xgb.DMatrix(features)[0])
        # LightGBM & RF use the raw array
        lgb_log = float(lgb_model.predict(features)[0])
        rf_log  = float(rf_model.predict(features)[0])
    except Exception as e:
        logger.error(f"Prediction failed for {token}: {e}")
        raise HTTPException(status_code=500, detail="Prediction error")

    # Convert log-return to actual next-hour price
    # xgb_ratio = np.exp(xgb_log)
    lgb_ratio = np.exp(lgb_log)
    rf_ratio  = np.exp(rf_log)
    current_price = features[0][3]  # Avg Token Price from dummy features
    # xgb_price = current_price * xgb_ratio
    lgb_price = current_price * lgb_ratio
    rf_price  = current_price * rf_ratio

    # Persist
    record = Prediction(
        token=token,
        timestamp=now,
        features=features.tolist(),
        xgboost=float(0),
        lightgbm=float(lgb_price),
        random_forest=float(rf_price),
    )
    db.add(record)
    db.commit()
    logger.info(f"Saved prediction for {token} at {now.isoformat()}")

    return {
        "token": token,
        "timestamp": now.isoformat(),
        "predicted_price_next_hour": {
            # "xgboost": xgb_price,
            "lightgbm": lgb_price,
            "random_forest": rf_price,
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
