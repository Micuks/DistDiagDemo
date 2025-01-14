from fastapi import APIRouter, HTTPException
from typing import List, Dict
from datetime import datetime
from app.services.k8s_service import K8sService
from app.services.metrics_service import MetricsService
from app.services.diagnosis_service import DiagnosisService
from app.schemas.anomaly import AnomalyRequest, MetricsResponse, AnomalyRankResponse

router = APIRouter()
k8s_service = K8sService()
metrics_service = MetricsService()
diagnosis_service = DiagnosisService()

@router.post("/start")
async def start_anomaly(request: AnomalyRequest):
    """Start an anomaly experiment in the OceanBase cluster"""
    try:
        # Apply the chaos mesh experiment
        await k8s_service.apply_chaos_experiment(request.type)
        return {"status": "success", "message": f"Started {request.type} anomaly"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stop")
async def stop_anomaly():
    """Stop all running anomaly experiments"""
    try:
        # Delete all chaos mesh experiments
        await k8s_service.delete_all_chaos_experiments()
        return {"status": "success", "message": "Stopped all anomalies"}
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