from fastapi import APIRouter, HTTPException
from typing import List, Dict
from datetime import datetime
import asyncio
from app.services.k8s_service import K8sService
from app.services.metrics_service import MetricsService
from app.services.diagnosis_service import DiagnosisService
from app.schemas.anomaly import AnomalyRequest, MetricsResponse, AnomalyRankResponse, ActiveAnomalyResponse

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

@router.post("/start")
async def start_anomaly(request: AnomalyRequest):
    """Start an anomaly experiment in the OceanBase cluster"""
    try:
        # Apply the chaos mesh experiment with retry logic
        async def start_experiment():
            return await k8s_service.apply_chaos_experiment(request.type)
            
        await retry_with_backoff(start_experiment)
        return {"status": "success", "message": f"Started {request.type} anomaly"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stop")
async def stop_anomaly(request: AnomalyRequest = None):
    """Stop specific or all running anomaly experiments"""
    try:
        if request and request.type:
            # Delete specific chaos mesh experiment
            await k8s_service.delete_chaos_experiment(request.type)
            return {"status": "success", "message": f"Stopped {request.type} anomaly"}
        else:
            # Delete all chaos mesh experiments
            await k8s_service.delete_all_chaos_experiments()
            return {"status": "success", "message": "Stopped all anomalies"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/active", response_model=List[ActiveAnomalyResponse])
async def get_active_anomalies():
    """Get list of currently active anomalies"""
    try:
        active_anomalies = await k8s_service.get_active_anomalies()
        return active_anomalies
    except Exception as e:
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
        metrics = await metrics_service.get_system_metrics()
        
        # Get anomaly ranks from diagnosis service
        ranked_anomalies = await diagnosis_service.analyze_metrics(metrics)
        
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
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/train")
async def train_model():
    """Train the anomaly detection model with historical data"""
    try:
        # Fetch historical metrics with labels
        training_data = await metrics_service.get_training_metrics()
        
        # Train the model
        diagnosis_service.train(training_data)
        
        return {"status": "success", "message": "Model trained successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 