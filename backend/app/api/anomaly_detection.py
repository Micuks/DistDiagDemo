from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import joblib
import numpy as np
from datetime import datetime
import logging
from ..services.graph_service import AnomalyGraphService
import os

router = APIRouter()
logger = logging.getLogger(__name__)

# Load models
try:
    model_path = os.getenv("MODEL_PATH", "../models")
    xgboost_model = joblib.load(f"{model_path}/xgboost_model.joblib")
    xgboost_scaler = joblib.load(f"{model_path}/xgboost_scaler.joblib")
    logger.info("Successfully loaded XGBoost model and scaler")
except Exception as e:
    logger.error(f"Failed to load models: {str(e)}")
    raise

# Initialize graph service
neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
neo4j_user = os.getenv("NEO4J_USER", "neo4j")
neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
graph_service = AnomalyGraphService(neo4j_uri, neo4j_user, neo4j_password)

class MetricsData(BaseModel):
    node_id: str
    timestamp: datetime
    metrics: Dict[str, float]

class AnomalyResponse(BaseModel):
    node_id: str
    timestamp: datetime
    anomaly_detected: bool
    anomaly_probability: float
    anomaly_type: Optional[str]
    metrics: Dict[str, float]

@router.post("/detect", response_model=AnomalyResponse)
async def detect_anomaly(data: MetricsData):
    """Detect anomalies in the provided metrics data."""
    try:
        # Prepare features for prediction
        features = [
            data.metrics.get("cpu_usage", 0),
            data.metrics.get("memory_usage", 0),
            data.metrics.get("io_usage", 0),
            data.metrics.get("network_in", 0),
            data.metrics.get("network_out", 0),
            data.metrics.get("disk_read", 0),
            data.metrics.get("disk_write", 0),
            data.metrics.get("active_sessions", 0)
        ]
        
        # Scale features
        features_scaled = xgboost_scaler.transform(np.array(features).reshape(1, -1))
        
        # Get prediction and probability
        prediction = xgboost_model.predict(features_scaled)[0]
        probabilities = xgboost_model.predict_proba(features_scaled)[0]
        max_prob = max(probabilities)
        
        # Update anomaly graph
        graph_service.create_node(
            data.node_id,
            data.metrics,
            max_prob if prediction != 0 else 0  # Only set probability if anomaly detected
        )
        
        # Map anomaly types
        anomaly_types = {
            0: "normal",
            1: "cpu_bottleneck",
            2: "memory_leak",
            3: "io_bottleneck"
        }
        
        return AnomalyResponse(
            node_id=data.node_id,
            timestamp=data.timestamp,
            anomaly_detected=prediction != 0,
            anomaly_probability=float(max_prob),
            anomaly_type=anomaly_types.get(prediction),
            metrics=data.metrics
        )
        
    except Exception as e:
        logger.error(f"Error detecting anomaly: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

class RootCauseResponse(BaseModel):
    timestamp: datetime
    root_causes: List[Dict[str, float]]
    subgraph: Dict

@router.get("/root-causes/{node_id}", response_model=RootCauseResponse)
async def get_root_causes(node_id: str, top_k: int = 5):
    """Get root causes for anomalies related to a specific node."""
    try:
        # Get root causes
        root_causes = graph_service.get_root_causes(top_k)
        
        # Get relevant subgraph
        node_ids = [rc["node"] for rc in root_causes]
        if node_id not in node_ids:
            node_ids.append(node_id)
        subgraph = graph_service.get_subgraph(node_ids)
        
        return RootCauseResponse(
            timestamp=datetime.now(),
            root_causes=root_causes,
            subgraph=subgraph
        )
        
    except Exception as e:
        logger.error(f"Error getting root causes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.on_event("shutdown")
def shutdown_event():
    """Clean up resources on shutdown."""
    graph_service.close() 