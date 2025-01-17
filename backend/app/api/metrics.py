from fastapi import APIRouter, HTTPException
from ..services.metrics_service import metrics_service
from ..schemas.metrics import MetricsData

router = APIRouter()

@router.get("/metrics", response_model=MetricsData)
async def get_metrics():
    """Get the latest system metrics for all nodes"""
    try:
        return metrics_service.get_latest_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/metrics/start")
async def start_metrics_collection():
    """Start periodic metrics collection"""
    try:
        await metrics_service.start_collection()
        return {"status": "success", "message": "Started metrics collection"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/metrics/stop")
async def stop_metrics_collection():
    """Stop periodic metrics collection"""
    try:
        await metrics_service.stop_collection()
        return {"status": "success", "message": "Stopped metrics collection"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))