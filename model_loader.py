import xgboost as xgb
import lightgbm as lgb
import joblib
from sklearn.ensemble import RandomForestRegressor
from config import settings
from logger import logger


def load_xgb():
    booster = xgb.Booster()
    try:
        booster.load_model(settings.XGB_MODEL_PATH)
        logger.info(f"Loaded XGBoost model from {settings.XGB_MODEL_PATH}")
    except Exception:
        logger.warning("XGBoost model not found, initializing empty Booster.")
    return booster


def load_lgb():
    try:
        booster = lgb.Booster(model_file=settings.LGB_MODEL_PATH)
        logger.info(f"Loaded LightGBM model from {settings.LGB_MODEL_PATH}")
    except Exception:
        booster = lgb.Booster()
        logger.warning("LightGBM model not found, initializing empty Booster.")
    return booster


def load_rf():
    try:
        model = joblib.load(settings.RF_MODEL_PATH)
        logger.info(f"Loaded Random Forest model from {settings.RF_MODEL_PATH}")
    except Exception:
        model = RandomForestRegressor()
        logger.warning("RF model not found, initializing untrained estimator.")
    return model
