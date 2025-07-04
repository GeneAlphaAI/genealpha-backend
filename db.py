from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from config import settings

engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# models.py (SQLAlchemy ORM)
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
