# genealpha-backend
Backend Template for GeneAlpha Prediction Engine

## Project Structure
```
├── config.py            # Application settings (Pydantic)
├── db.py                # SQLAlchemy engine & session
├── models.py            # ORM models / DB schema
├── logger.py            # Structured logging setup
├── model_loader.py      # ML model load functions
├── main.py              # FastAPI app & endpoints
├── models/              # Directory for saved model files
│   ├── xgb_model.json
│   ├── lgb_model.txt
│   └── rf_model.joblib
├── .env                 # Environment variable definitions
└── README.md            # This documentation
```
