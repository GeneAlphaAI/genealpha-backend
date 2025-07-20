from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # 1) Tell Pydantic where to load your .env
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore', 
    )

    # 2) Map each Python attribute to the actual ENV var name via alias
    USER_NAME: str = Field(..., alias='POSTGRES_USER')
    PASSWORD:  str = Field(..., alias='POSTGRES_PASSWORD')
    DB_NAME:   str = Field(..., alias='POSTGRES_DB')
    HOST:      str = Field(..., alias='HOST')
    FEATURES_URL: str = Field(..., alias='FEATURES_URL')

    # (your other settings...)
    XGB_MODEL_PATH: str = Field('models/xgb_model.json', alias='XGB_MODEL_PATH')
    LGB_MODEL_PATH: str = Field('models/lgbm_model.txt', alias='LGB_MODEL_PATH')
    RF_MODEL_PATH:  str = Field('models/rf_model.joblib', alias='RF_MODEL_PATH')
    FEATURE_DIM:    int = Field(10, alias='FEATURE_DIM')

settings = Settings()
