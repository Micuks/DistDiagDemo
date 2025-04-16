from fastapi import APIRouter, HTTPException, Depends, Query
from ..services.metrics_service import metrics_service
import json
from typing import Dict, Any, Optional
from fastapi.responses import JSONResponse
import warnings
import logging
from app.services.metric_inferor_service import analyze_node_metrics
import os

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/")
async def get_metrics():
    """Get current system metrics."""
    return metrics_service.get_last_metric_point()

@router.get("/detailed", deprecated=True)
async def get_detailed_metrics(node_ip: str, category: str):
    """Get detailed metrics with historical data for a specific node and category.
    
    Deprecated: Use /detailed/selected endpoint instead for better performance.
    This endpoint fetches all metrics for a category, which can be inefficient.
    """
    # Log usage of deprecated endpoint
    warnings.warn(f"Deprecated endpoint /metrics/detailed called for {node_ip}/{category}", DeprecationWarning)
    
    result = metrics_service.get_detailed_metrics(node_ip, category)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@router.get("/detailed/selected")
async def get_selected_detailed_metrics(node_ip: str, metrics: str):
    """Get detailed metrics with historical data for only the selected metrics for each category.
    
    Args:
        node_ip: The IP address of the node to get metrics for
        metrics: JSON string of category -> metric mappings (e.g. {"cpu": "cpu usage", "memory": "total memstore used"})
    """
    try:
        # Parse the metrics parameter
        selected_metrics = json.loads(metrics)
        if not isinstance(selected_metrics, dict):
            raise HTTPException(status_code=400, detail="metrics parameter must be a JSON object")
            
        # Get metrics for each category
        result = metrics_service.get_selected_detailed_metrics(node_ip, selected_metrics)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in metrics parameter")

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

@router.get("/fluctuations")
async def get_metrics_fluctuations():
    """Get metrics with fluctuation data for all nodes."""
    try:
        metrics = metrics_service.get_last_metric_point()
        
        # Count total metrics and metrics with fluctuations
        total_metrics = 0
        fluctuating_metrics = 0
        
        # Check if any metrics have fluctuations
        for node_ip, node_data in metrics.get("metrics", {}).items():
            for category, category_metrics in node_data.items():
                for metric_name, metric_data in category_metrics.items():
                    total_metrics += 1
                    if metric_data.get("has_fluctuation", False):
                        fluctuating_metrics += 1
        
        return {
            "metrics": metrics.get("metrics", {}),
            "timestamp": metrics.get("timestamp"),
            "summary": {
                "total_metrics": total_metrics,
                "fluctuating_metrics": fluctuating_metrics,
                "has_fluctuations": fluctuating_metrics > 0
            }
        }
    except Exception as e:
        logger.error(f"Error getting metrics fluctuations: {str(e)}")
        return JSONResponse(
            status_code=500, 
            content={"error": str(e)},
            headers={
                "Access-Control-Allow-Credentials": "true"
            }
        )