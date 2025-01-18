from fastapi import APIRouter, HTTPException
from ..services.metrics_service import metrics_service

router = APIRouter()

@router.get("/")
async def get_metrics():
    """Get current system metrics"""
    try:
        return metrics_service.get_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/start")
async def start_metrics_collection():
    """Start collecting metrics"""
    try:
        metrics_service.start_collection()
        return {"status": "success", "message": "Started metrics collection"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stop")
async def stop_metrics_collection():
    """Stop collecting metrics"""
    try:
        metrics_service.stop_collection()
        return {"status": "success", "message": "Stopped metrics collection"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))