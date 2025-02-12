from fastapi import APIRouter, HTTPException, BackgroundTasks, Response
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
import threading
from fastapi_cache import FastAPICache
from fastapi_cache.decorator import cache
from fastapi_cache.backends.redis import RedisBackend
from redis import asyncio as aioredis
from sse_starlette.sse import EventSourceResponse
import json
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi_cache.coder import JsonCoder
from fastapi.responses import JSONResponse

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
        # Clear cache for active anomalies
        FastAPICache.clear_all()
        
        # Inject the anomaly
        await retry_with_backoff(
            lambda: k8s_service.apply_chaos_experiment(request.type)
        )
        
        # Start collecting training data if requested
        if request.collect_training_data:
            training_service.start_collection(request.type, request.node)
            logger.info(f"Started collecting training data for {request.type} anomaly on {request.node}")
            
        return {
            "status": "success", 
            "message": f"Injected {request.type} anomaly into node {request.node}"
        }
    except Exception as e:
        logger.error(f"Failed to inject anomaly: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clear")
async def clear_anomaly(request: AnomalyRequest):
    """Clear an anomaly from the OceanBase cluster"""
    try:
        # Clear cache for active anomalies
        FastAPICache.clear_all()
        
        # Clear the anomaly first
        await retry_with_backoff(
            lambda: k8s_service.delete_chaos_experiment(request.type)
        )
        
        # Stop collecting training data if it was being collected
        # This will automatically start post-anomaly data collection
        if request.collect_training_data:
            training_service.stop_collection()
            logger.info(f"Stopped collecting training data for {request.type} anomaly on {request.node}")
            
        return {
            "status": "success", 
            "message": f"Cleared {request.type} anomaly from node {request.node}"
        }
    except Exception as e:
        logger.error(f"Failed to clear anomaly: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/active", response_model=List[ActiveAnomalyResponse])
async def get_active_anomalies():
    """Get list of active anomalies in the OceanBase cluster"""
    try:
        active_anomalies = await k8s_service.get_active_anomalies()
        response = jsonable_encoder(active_anomalies)
        return JSONResponse(
            content=response,
            headers={
                "Access-Control-Allow-Origin": "http://10.101.168.97:3000",
                "Access-Control-Allow-Credentials": "true",
            }
        )
    except Exception as e:
        logger.error(f"Failed to get active anomalies: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)},
            headers={
                "Access-Control-Allow-Origin": "http://10.101.168.97:3000",
                "Access-Control-Allow-Credentials": "true",
            }
        )

@router.get("/ranks", response_model=List[AnomalyRankResponse])
@cache(expire=5)
async def get_anomaly_ranks(response: Response):
    """Get ranked list of potential anomalies"""
    try:
        metrics = metrics_service.get_metrics()
        if not metrics or not metrics.get("metrics"):
            return JSONResponse(
                content=[],
                headers={
                    "Access-Control-Allow-Origin": "http://10.101.168.97:3000",
                    "Access-Control-Allow-Credentials": "true"
                }
            )
            
        ranks = diagnosis_service.analyze_metrics(metrics["metrics"])
        return JSONResponse(
            content=ranks,
            headers={
                "Access-Control-Allow-Origin": "http://10.101.168.97:3000",
                "Access-Control-Allow-Credentials": "true"
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get anomaly ranks: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
            headers={
                "Access-Control-Allow-Origin": "http://10.101.168.97:3000",
                "Access-Control-Allow-Credentials": "true"
            }
        )

@router.post("/train")
async def train_model():
    """Train the anomaly detection model."""
    try:
        # Get training data
        X, y = training_service.get_training_data()
        if len(X) == 0 or len(y) == 0:
            raise HTTPException(status_code=400, detail="No training data available")
            
        # Train the model
        diagnosis_service.train(X, y)
        return {"status": "success", "message": "Model trained successfully"}
    except Exception as e:
        logger.error(f"Failed to train model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/normal/start")
async def start_normal_collection():
    """Start collecting normal state data"""
    try:
        training_service.start_normal_collection()
        return {"status": "success", "message": "Started normal state data collection"}
    except Exception as e:
        logger.error(f"Failed to start normal state collection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/normal/stop")
async def stop_normal_collection():
    """Stop collecting normal state data"""
    try:
        training_service.stop_normal_collection()
        return {"status": "success", "message": "Stopped normal state data collection"}
    except Exception as e:
        logger.error(f"Failed to stop normal state collection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/training/stats")
async def get_training_stats():
    """Get statistics about the collected training data."""
    try:
        try:
            stats = await asyncio.wait_for(training_service.get_dataset_stats(), timeout=20)
        except asyncio.TimeoutError:
            logger.warning("Dataset stats computation timed out")
            stats = training_service._stats_cache
        
        if not stats:
            return JSONResponse(
                status_code=404,
                content={"status": "error", "message": "No training data available"},
                headers={
                    "Access-Control-Allow-Origin": "http://10.101.168.97:3000",
                    "Access-Control-Allow-Credentials": "true",
                }
            )

        # Use the stats directly from the service without modification
        response = {
            "status": "success",
            "stats": {
                "normal": stats["normal"],  # Use direct access instead of .get()
                "anomaly": stats["anomaly"],
                "anomaly_types": stats["anomaly_types"],
                "total_samples": stats["total_samples"],
                "normal_ratio": stats["normal_ratio"],
                "anomaly_ratio": stats["anomaly_ratio"],
                "is_balanced": stats["is_balanced"]
            }
        }
        
        return JSONResponse(
            content=jsonable_encoder(response),
            headers={
                "Access-Control-Allow-Origin": "http://10.101.168.97:3000",
                "Access-Control-Allow-Credentials": "true",
            }
        )
    except Exception as e:
        logger.error(f"Failed to get training stats: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)},
            headers={
                "Access-Control-Allow-Origin": "http://10.101.168.97:3000",
                "Access-Control-Allow-Credentials": "true",
            }
        )

@router.post("/training/auto_balance")
async def auto_balance_dataset():
    """Start auto-balancing the training dataset."""
    try:
        # Start auto-balance in a background thread
        threading.Thread(target=training_service.auto_balance_dataset).start()
        return {"status": "success", "message": "Auto-balance process started"}
    except Exception as e:
        logger.error(f"Failed to start auto-balance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stream")
async def anomaly_stream():
    """Stream anomaly updates using Server-Sent Events"""
    async def event_generator():
        last_data = None
        while True:
            try:
                # Get current anomalies directly from the service's state
                current_data = list(k8s_service.active_anomalies.values())
                
                # Only send if data changed
                if current_data != last_data:
                    last_data = current_data
                    yield {
                        "event": "anomaly_update",
                        "data": json.dumps(current_data),
                        "retry": 3000  # Retry connection after 3s if dropped
                    }
                
                # Check every second
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error in anomaly stream: {str(e)}")
                # Add small delay on error
                await asyncio.sleep(1)
                continue

    return EventSourceResponse(event_generator()) 