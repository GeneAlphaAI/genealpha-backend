from sqlalchemy import Column, Integer, DateTime, JSON, Float, String
from db import Base
import datetime

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    token = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    features = Column(JSON)
    xgboost = Column(Float)
    lightgbm = Column(Float)
    random_forest = Column(Float)
