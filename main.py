from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
import numpy as np
import datetime
import requests

from config import settings
from db import get_db, Base, engine
from models import Prediction
from logger import logger
from model_loader import load_xgb, load_lgb, load_rf

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI()

# ---- CORS setup ----------------------------------------------------
origins = [
    "http://localhost:5180",        # local front-end during dev
    "https://hive.genealpha.ai",    # production front-end (HTTPS)
    "http://hive.genealpha.ai",     # non-TLS fallback if ever needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,    
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Load models once
# xgb_model = load_xgb()
lgb_model = load_lgb()
rf_model  = load_rf()

# Placeholder mapping: token name → token hash
TOKEN_HASHES = {
    "COCORO": "0xa93d86af16fe83f064e3c0e2f3d129f7b7b002b0",
}

# Pydantic schema for incoming request

class PredictRequest(BaseModel):
    token: str


def fetch_features(address: str, base_url: str) -> dict:
    """
    Fetches the transaction summary for a given Ethereum address and
    returns only the features needed for prediction, including momentum.
    """
    endpoint = "/api/token/transactions/summary"
    params = {"address": address}
    url = f"{base_url}{endpoint}"

    try:
        logger.info(f"Fetching transaction summary for address: {address}")
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json().get("data", [])
        if not data:
            logger.error(f"No data returned for address: {address}")
            return {}
        item = data[0]
        # Extract and convert fields
        buy_count = int(item.get("buyCount", 0))
        sell_count = int(item.get("sellCount", 0))
        active_address_count = int(item.get("activeAddressCount", 0))
        avg_token_price = float(item.get("avgTokenPrice", 0.0))
        token_volume = float(item.get("tokenVolume", 0.0))
        token_volume_usd = float(item.get("tokenVolumeUSD", 0.0))
        eth_price = float(item.get("ethPrice", 0.0))
        btc_price = float(item.get("btcPrice", 0.0))

        last_price = float(item.get("lastTokenPrice", 0.0))
        latest_price = float(item.get("latestTokenPrice", 0.0))
        # Calculate momentum: (latest - last) / last
        momentum = (latest_price - last_price) / last_price if last_price else 0.0

        return {
            "buy_count": buy_count,
            "sell_count": sell_count,
            "active_address_count": active_address_count,
            "latest_price": latest_price,
            "token_volume": token_volume,
            "token_volume_usd": token_volume_usd,
            "eth_price": eth_price,
            "btc_price": btc_price,
            "momentum": momentum,
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching transaction summary: {e}")
        return {}

@app.post("/predict")
async def predict(req: PredictRequest,
    db: Session = Depends(get_db)):

    token = req.token.upper()
    token_hash = TOKEN_HASHES.get(token)
    if token_hash is None:
        raise HTTPException(status_code=400, detail=f"Unknown token: {token}")

    # Fetch features dict
    features_dict = fetch_features(token_hash, base_url=settings.FEATURES_URL)
    if not features_dict:
        raise HTTPException(status_code=500, detail="Failed to fetch features")

    # Convert to array for prediction order
    features = np.array([[
        features_dict["buy_count"],
        features_dict["sell_count"],
        features_dict["active_address_count"],
        features_dict["latest_price"],
        features_dict["token_volume"],
        features_dict["token_volume_usd"],
        features_dict["eth_price"],
        features_dict["btc_price"],
        features_dict["momentum"],
    ]])
    now = datetime.datetime.utcnow()

    try:
        # xgb_log = float(xgb_model.predict(xgb.DMatrix(features))[0])
        lgb_log = float(lgb_model.predict(features)[0])
        rf_log  = float(rf_model.predict(features)[0])
    except Exception as e:
        logger.error(f"Prediction failed for {token}: {e}")
        raise HTTPException(status_code=500, detail="Prediction error")

    # calculate actual prices
    # xgb_price = features_dict["latest_price"] * np.exp(xgb_log)
    lgb_price = features_dict["latest_price"] * np.exp(lgb_log)
    rf_price  = features_dict["latest_price"] * np.exp(rf_log)

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
