from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
import numpy as np
import datetime

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
rf_model = load_rf()


def fetch_features(token: str) -> np.ndarray:
    # TODO: replace with real API call
    return np.random.rand(1, settings.FEATURE_DIM)

@app.get("/predict")
async def predict(db: Session = Depends(get_db)):
    tokens = ["BTC", "ETH", "XRP"]
    now = datetime.datetime.utcnow()
    results = {}

    for token in tokens:
        features = fetch_features(token)
        try:
            xgb_pred = float(xgb_model.predict(xgb.DMatrix(features))[0])
            lgb_pred = float(lgb_model.predict(features)[0])
            rf_pred = float(rf_model.predict(features)[0])
        except Exception as e:
            logger.error(f"Prediction failed for {token}: {e}")
            raise HTTPException(status_code=500, detail="Prediction error")

        # Persist
        record = Prediction(
            token=token,
            timestamp=now,
            features=features.tolist(),
            xgboost=xgb_pred,
            lightgbm=lgb_pred,
            random_forest=rf_pred,
        )
        db.add(record)
        db.commit()
        logger.info(f"Saved prediction for {token} at {now.isoformat()}")

        results[token] = {
            "xgboost": xgb_pred,
            "lightgbm": lgb_pred,
            "random_forest": rf_pred,
        }

    return {"predictions": results}
