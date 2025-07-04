from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = Field(..., env="DATABASE_URL")

    # Model paths
    XGB_MODEL_PATH: str = Field("models/xgb_model.json", env="XGB_MODEL_PATH")
    LGB_MODEL_PATH: str = Field("models/lgb_model.txt", env="LGB_MODEL_PATH")
    RF_MODEL_PATH: str = Field("models/rf_model.joblib", env="RF_MODEL_PATH")

    # Feature settings
    FEATURE_DIM: int = Field(10, env="FEATURE_DIM")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
