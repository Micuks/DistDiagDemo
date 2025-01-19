from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Dict, Optional
from datetime import datetime
import asyncio
from pydantic import BaseModel
from app.services.k8s_service import K8sService
from app.services.metrics_service import MetricsService
from app.services.diagnosis_service import DiagnosisService
from app.services.training_service import training_service
from app.schemas.anomaly import AnomalyRequest, MetricsResponse, AnomalyRankResponse, ActiveAnomalyResponse
import logging

logger = logging.getLogger(__name__)

router = APIRouter()
k8s_service = K8sService()
metrics_service = MetricsService()
diagnosis_service = DiagnosisService()

async def retry_with_backoff(func, max_retries=3, initial_delay=1):
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            last_exception = e
            if "AlreadyExists" in str(e) and "is being deleted" in str(e):
                await asyncio.sleep(delay)
                delay *= 2
                continue
            raise e
    
    raise last_exception

class AnomalyRequest(BaseModel):
    type: str
    node: Optional[str] = None
    collect_training_data: Optional[bool] = False

@router.post("/inject")
async def inject_anomaly(request: AnomalyRequest):
    """Inject an anomaly into the OceanBase cluster"""
    try:
        # Start collecting training data if requested
        if request.collect_training_data:
            training_service.start_collection(request.type, request.node)
            logger.info(f"Started collecting training data for {request.type} anomaly on {request.node}")

        # Inject the anomaly
        await k8s_service.apply_chaos_experiment(request.type)
        return {"status": "success", "message": f"Injected {request.type} anomaly"}
    except Exception as e:
        logger.error(f"Failed to inject anomaly: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clear")
async def clear_anomaly(request: AnomalyRequest):
    """Clear an anomaly from the OceanBase cluster"""
    try:
        # Stop collecting training data if it was being collected
        if request.collect_training_data:
            training_service.stop_collection()
            logger.info("Stopped collecting training data")

        # Clear the anomaly
        await k8s_service.delete_chaos_experiment(request.type)
        return {"status": "success", "message": f"Cleared {request.type} anomaly"}
    except Exception as e:
        logger.error(f"Failed to clear anomaly: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/active", response_model=List[ActiveAnomalyResponse])
async def get_active_anomalies():
    """Get list of active anomalies in the OceanBase cluster"""
    try:
        active_anomalies = await k8s_service.get_active_anomalies()
        return active_anomalies
    except Exception as e:
        logger.error(f"Failed to get active anomalies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics", response_model=List[MetricsResponse])
async def get_metrics():
    """Get system metrics from the OceanBase cluster"""
    try:
        # Fetch metrics from Prometheus
        metrics = await metrics_service.get_system_metrics()
        return [
            MetricsResponse(
                timestamp=entry["timestamp"],
                cpu=entry["cpu"],
                memory=entry["memory"],
                network=entry["network"]
            ) for entry in metrics
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ranks", response_model=List[AnomalyRankResponse])
async def get_anomaly_ranks():
    """Get ranked anomalies from the OceanBase cluster"""
    try:
        # Fetch latest metrics
        metrics = metrics_service.get_metrics()
        
        # Get anomaly ranks from diagnosis service
        ranked_anomalies = diagnosis_service.analyze_metrics(metrics.get('metrics', {}))
        
        # Convert to response format
        return [
            AnomalyRankResponse(
                timestamp=anomaly["timestamp"],
                node=anomaly["node"],
                type=anomaly["type"],
                score=anomaly["score"]
            ) for anomaly in ranked_anomalies
        ]
    except Exception as e:
        logger.error(f"Failed to get anomaly ranks: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/train")
async def train_model():
    """Train the anomaly detection model with collected data"""
    try:
        # Get training data
        X, y = training_service.get_training_data()
        if len(X) == 0 or len(y) == 0:
            raise ValueError("No training data available")
            
        # Train the model
        diagnosis_service.train(X, y)
        return {"status": "success", "message": "Model trained successfully"}
    except Exception as e:
        logger.error(f"Failed to train model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/normal/start")
async def start_normal_collection():
    """Start collecting normal state metrics data for training."""
    try:
        training_service.start_normal_collection()
        return {"status": "success", "message": "Started collecting normal state data"}
    except Exception as e:
        logger.error(f"Failed to start normal state collection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/normal/stop")
async def stop_normal_collection():
    """Stop collecting normal state metrics data and save it."""
    try:
        training_service.stop_normal_collection()
        return {"status": "success", "message": "Stopped collecting normal state data"}
    except Exception as e:
        logger.error(f"Failed to stop normal state collection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/training/stats")
async def get_training_stats():
    """Get statistics about the collected training data."""
    try:
        stats = training_service.get_dataset_stats()
        is_balanced = training_service.is_dataset_balanced()
        return {
            "status": "success",
            "stats": stats,
            "is_balanced": is_balanced,
            "total_samples": stats["normal"] + stats["anomaly"],
            "normal_ratio": stats["normal"] / (stats["normal"] + stats["anomaly"]) if stats["normal"] + stats["anomaly"] > 0 else 0,
            "anomaly_ratio": stats["anomaly"] / (stats["normal"] + stats["anomaly"]) if stats["normal"] + stats["anomaly"] > 0 else 0
        }
    except Exception as e:
        logger.error(f"Failed to get training stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 