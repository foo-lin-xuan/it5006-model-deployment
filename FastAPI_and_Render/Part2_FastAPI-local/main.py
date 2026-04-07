"""
FastAPI Fraud Detection Service with Inference Logging

Features:
- Loads complete pipelines (preprocessing + model)
- Returns individual + ensemble predictions
- Logs all predictions for monitoring, drift detection, and error analysis

Updated to use sklearn Pipelines - no manual encoding needed!
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import joblib
import json
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import os
from pathlib import Path

# ============================================================
# LOGGING SETUP
# ============================================================
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "inference.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Separate logger for predictions (structured for analysis)
prediction_logger = logging.getLogger("predictions")
prediction_handler = logging.FileHandler(LOG_DIR / "predictions.jsonl")
prediction_handler.setFormatter(logging.Formatter('%(message)s'))
prediction_logger.addHandler(prediction_handler)
prediction_logger.setLevel(logging.INFO)

# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(
    title="Fraud Detection API",
    description="Ensemble fraud detection using XGBoost + Random Forest Pipelines",
    version="2.0.0"
)

# ============================================================
# LOAD PIPELINES & METADATA
# ============================================================
MODEL_DIR = Path("models")

try:
    # Load COMPLETE pipelines (preprocessor + model)
    # These handle raw data directly - no manual encoding needed!
    xgb_pipeline = joblib.load(MODEL_DIR / "xgboost_pipeline.pkl")
    rf_pipeline = joblib.load(MODEL_DIR / "random_forest_pipeline.pkl")
    
    with open(MODEL_DIR / "model_metadata.json") as f:
        metadata = json.load(f)
    
    with open(MODEL_DIR / "feature_stats.json") as f:
        feature_stats = json.load(f)
    
    # Extract feature info from metadata (handle both old and new key formats)
    # For pipeline-based models, features should be raw column names (not encoded)
    raw_features = metadata.get("feature_columns", metadata.get("features", []))
    
    # If metadata has 'category_encoded', convert to 'category' for pipeline input
    FEATURE_COLS = ['category' if f == 'category_encoded' else f for f in raw_features]
    
    CATEGORICAL_COLS = metadata.get("categorical_columns", ["category"])
    NUMERIC_COLS = metadata.get("numeric_columns", [f for f in FEATURE_COLS if f != "category"])
    CATEGORY_CLASSES = metadata.get("category_classes", [])
    
    # If FEATURE_COLS is empty, define default based on pipeline structure
    if not FEATURE_COLS:
        FEATURE_COLS = ['category', 'amount', 'age_at_transaction', 'days_until_card_expires',
                        'loc_delta', 'trans_volume_mavg', 'trans_volume_mstd', 'trans_freq', 'loc_delta_mavg']
        NUMERIC_COLS = FEATURE_COLS[1:]  # All except category
    
    logger.info("Pipelines loaded successfully!")
    logger.info(f"   Feature columns: {FEATURE_COLS}")
    logger.info(f"   Category classes: {CATEGORY_CLASSES}")
    
except Exception as e:
    logger.error(f"Failed to load pipelines: {e}")
    raise

# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================
class TransactionInput(BaseModel):
    """Input features for fraud prediction (RAW features - no encoding needed)."""
    category: str                          # Categorical - pipeline handles encoding
    amount: float                          # Transaction amount
    age_at_transaction: float              # Customer age
    days_until_card_expires: float         # Days until card expiry
    loc_delta: float = 0.0                 # Distance from last transaction
    trans_volume_mavg: float               # 4h avg transaction amount
    trans_volume_mstd: float = 0.0         # 4h std dev of amount
    trans_freq: float = 1.0                # Transactions in 4h window
    loc_delta_mavg: float = 0.0            # 4h avg location change
    
    class Config:
        json_schema_extra = {
            "example": {
                "category": "grocery_pos",
                "amount": 150.50,
                "age_at_transaction": 35.5,
                "days_until_card_expires": 365.0,
                "loc_delta": 0.05,
                "trans_volume_mavg": 120.0,
                "trans_volume_mstd": 45.0,
                "trans_freq": 3.0,
                "loc_delta_mavg": 0.03
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response with individual and ensemble results."""
    transaction_id: str
    timestamp: str
    
    # Individual model predictions
    xgboost_prediction: int
    xgboost_probability: float
    random_forest_prediction: int
    random_forest_probability: float
    
    # Ensemble prediction
    ensemble_prediction: int
    ensemble_probability: float
    ensemble_verdict: str
    
    # Drift warning (if features are outside training distribution)
    drift_warnings: List[str]


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def check_drift(features: dict) -> list:
    """Check if numeric input features are outside training distribution."""
    warnings = []
    
    for feature, value in features.items():
        # Only check numeric features (skip categorical)
        if feature in feature_stats:
            stats = feature_stats[feature]
            mean = stats.get('mean', 0)
            std = stats.get('std', 1)
            min_val = stats.get('min', float('-inf'))
            max_val = stats.get('max', float('inf'))
            
            # Check if value is > 3 std from mean
            if std > 0 and abs(value - mean) > 3 * std:
                warnings.append(f"{feature}: {value:.2f} is >3 sigma from training mean ({mean:.2f})")
            
            # Check if value is outside training range
            if value < min_val or value > max_val:
                warnings.append(f"{feature}: {value:.2f} outside training range [{min_val:.2f}, {max_val:.2f}]")
    
    return warnings


def log_prediction(request_id: str, input_data: dict, prediction: dict, drift_warnings: list):
    """Log prediction for monitoring and analysis."""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "request_id": request_id,
        "input": input_data,
        "predictions": {
            "xgboost": {"pred": prediction["xgb_pred"], "prob": prediction["xgb_prob"]},
            "random_forest": {"pred": prediction["rf_pred"], "prob": prediction["rf_prob"]},
            "ensemble": {"pred": prediction["ensemble_pred"], "prob": prediction["ensemble_prob"]}
        },
        "drift_warnings": drift_warnings,
        "has_drift": len(drift_warnings) > 0
    }
    
    # Write to JSONL file (one JSON object per line for easy parsing)
    prediction_logger.info(json.dumps(log_entry))


# Global request counter
request_counter = 0

# ============================================================
# API ENDPOINTS
# ============================================================
@app.get("/")
def root():
    """API health check."""
    return {
        "status": "healthy",
        "message": "Fraud Detection API is running",
        "version": "2.0.0 (Pipeline-based)",
        "models": ["XGBoost Pipeline", "Random Forest Pipeline"],
        "ensemble_method": "Soft Voting (Average)"
    }


@app.get("/health")
def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "pipelines_loaded": True,
        "xgboost_pipeline": "ready",
        "random_forest_pipeline": "ready",
        "total_predictions": request_counter
    }


@app.get("/model-info")
def model_info():
    """Get model metadata and metrics."""
    return {
        "feature_columns": FEATURE_COLS,
        "categorical_columns": CATEGORICAL_COLS,
        "numeric_columns": NUMERIC_COLS,
        "category_classes": CATEGORY_CLASSES,
        "metrics": metadata.get("metrics", {}),
        "training_samples": metadata.get("training_samples", 0),
        "test_samples": metadata.get("test_samples", 0),
        "pipeline_steps": metadata.get("pipeline_steps", [])
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionInput):
    """
    Predict fraud probability for a transaction.
    
    Input: RAW transaction features (no preprocessing needed)
    
    Returns predictions from:
    - XGBoost Pipeline (gradient boosting)
    - Random Forest Pipeline (bagging)
    - Ensemble (soft voting average)
    """
    global request_counter
    request_counter += 1
    request_id = f"txn_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{request_counter:06d}"
    
    try:
        # Validate category
        if transaction.category not in CATEGORY_CLASSES:
            logger.warning(f"[{request_id}] Unknown category: {transaction.category}")
            # Pipeline's OrdinalEncoder will handle unknown as -1
        
        # Create DataFrame with RAW features (pipeline handles preprocessing!)
        input_data = {
            'category': [transaction.category],
            'amount': [transaction.amount],
            'age_at_transaction': [transaction.age_at_transaction],
            'days_until_card_expires': [transaction.days_until_card_expires],
            'loc_delta': [transaction.loc_delta],
            'trans_volume_mavg': [transaction.trans_volume_mavg],
            'trans_volume_mstd': [transaction.trans_volume_mstd],
            'trans_freq': [transaction.trans_freq],
            'loc_delta_mavg': [transaction.loc_delta_mavg]
        }
        
        # Create DataFrame with correct column order
        X = pd.DataFrame(input_data)[FEATURE_COLS]
        
        # Check for drift (numeric features only)
        numeric_features = {
            'amount': transaction.amount,
            'age_at_transaction': transaction.age_at_transaction,
            'days_until_card_expires': transaction.days_until_card_expires,
            'loc_delta': transaction.loc_delta,
            'trans_volume_mavg': transaction.trans_volume_mavg,
            'trans_volume_mstd': transaction.trans_volume_mstd,
            'trans_freq': transaction.trans_freq,
            'loc_delta_mavg': transaction.loc_delta_mavg
        }
        drift_warnings = check_drift(numeric_features)
        
        if drift_warnings:
            logger.warning(f"[{request_id}] Drift detected: {drift_warnings}")
        
        # Get predictions from PIPELINES (they handle preprocessing internally!)
        xgb_prob = float(xgb_pipeline.predict_proba(X)[0, 1])
        xgb_pred = int(xgb_prob >= 0.5)
        
        rf_prob = float(rf_pipeline.predict_proba(X)[0, 1])
        rf_pred = int(rf_prob >= 0.5)
        
        # Ensemble (soft voting)
        ensemble_prob = (xgb_prob + rf_prob) / 2
        ensemble_pred = int(ensemble_prob >= 0.5)
        
        # Determine verdict
        if ensemble_prob >= 0.7:
            verdict = "HIGH RISK - Likely Fraud"
        elif ensemble_prob >= 0.5:
            verdict = "SUSPICIOUS - Review Recommended"
        elif ensemble_prob >= 0.3:
            verdict = "LOW RISK - Monitor"
        else:
            verdict = "LEGITIMATE - Approved"
        
        # Log prediction
        log_prediction(
            request_id=request_id,
            input_data=transaction.model_dump(),
            prediction={
                "xgb_pred": xgb_pred, "xgb_prob": xgb_prob,
                "rf_pred": rf_pred, "rf_prob": rf_prob,
                "ensemble_pred": ensemble_pred, "ensemble_prob": ensemble_prob
            },
            drift_warnings=drift_warnings
        )
        
        logger.info(f"[{request_id}] Prediction: {verdict} (prob={ensemble_prob:.3f})")
        
        return PredictionResponse(
            transaction_id=request_id,
            timestamp=datetime.utcnow().isoformat(),
            xgboost_prediction=xgb_pred,
            xgboost_probability=round(xgb_prob, 4),
            random_forest_prediction=rf_pred,
            random_forest_probability=round(rf_prob, 4),
            ensemble_prediction=ensemble_pred,
            ensemble_probability=round(ensemble_prob, 4),
            ensemble_verdict=verdict,
            drift_warnings=drift_warnings
        )
        
    except Exception as e:
        logger.error(f"[{request_id}] Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/logs/summary")
def get_log_summary():
    """Get summary of logged predictions for monitoring."""
    log_file = LOG_DIR / "predictions.jsonl"
    
    if not log_file.exists():
        return {"message": "No predictions logged yet", "total": 0}
    
    total = 0
    fraud_count = 0
    drift_count = 0
    
    with open(log_file) as f:
        for line in f:
            try:
                entry = json.loads(line)
                total += 1
                if entry["predictions"]["ensemble"]["pred"] == 1:
                    fraud_count += 1
                if entry.get("has_drift", False):
                    drift_count += 1
            except:
                continue
    
    return {
        "total_predictions": total,
        "fraud_predictions": fraud_count,
        "fraud_rate": round(fraud_count / total * 100, 2) if total > 0 else 0,
        "predictions_with_drift": drift_count,
        "drift_rate": round(drift_count / total * 100, 2) if total > 0 else 0
    }


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
