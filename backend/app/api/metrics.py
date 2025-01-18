from fastapi import APIRouter, HTTPException
from ..services.metrics_service import metrics_service

router = APIRouter()

@router.get("/")
async def get_metrics():
    """Get current system metrics."""
    return metrics_service.get_metrics()

@router.get("/detailed")
async def get_detailed_metrics(node_ip: str, category: str):
    """Get detailed metrics with historical data for a specific node and category."""
    result = metrics_service.get_detailed_metrics(node_ip, category)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@router.post("/start")
async def start_metrics_collection():
    """Start collecting metrics."""
    metrics_service.start_collection()
    return {"status": "started"}

@router.post("/stop")
async def stop_metrics_collection():
    """Stop collecting metrics."""
    metrics_service.stop_collection()
    return {"status": "stopped"}