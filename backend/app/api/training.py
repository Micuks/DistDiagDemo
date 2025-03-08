from fastapi import APIRouter, HTTPException
from typing import List, Dict, Optional
from pydantic import BaseModel
import logging
from app.services.training_service import training_service
from app.services.metrics_service import metrics_service
logger = logging.getLogger(__name__)

router = APIRouter()

class AnomalyRequest(BaseModel):
    type: str
    node: Optional[str] = None
    pre_collect: Optional[bool] = True
    post_collect: Optional[bool] = True
    experiment_name: Optional[str] = None

class TrainingStopRequest(BaseModel):
    save_post_data: Optional[bool] = True

@router.post("/collect")
async def start_training_collection(request: AnomalyRequest):
    """Start collecting training data for a specific anomaly type"""
    try:
        metrics_service.toggle_send_to_training(True);
        await training_service.start_collection(
            anomaly_type=request.type,
            node=request.node,
        )
        return {
            "status": "success",
            "message": f"Started collecting training data for {request.type}",
            "type": request.type,
            "node": request.node
        }
    except Exception as e:
        logger.error(f"Error starting training collection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting training collection: {str(e)}")

@router.post("/stop")
async def stop_training_collection(request: TrainingStopRequest):
    """Stop collecting training data for an anomaly type"""
    try:
        metrics_service.toggle_send_to_training(False);
        if hasattr(training_service, "stop_anomaly_collection"):
            training_service.stop_anomaly_collection(save_post_data=request.save_post_data)
        else:
            # Fallback to generic stop
            training_service.stop_collection()
            
        return {
            "status": "success",
            "message": "Stopped training data collection"
        }
    except Exception as e:
        logger.error(f"Error stopping training collection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error stopping training collection: {str(e)}")

@router.post("/normal/start")
async def start_normal_collection():
    """Start collecting normal state data"""
    try:
        await training_service.start_normal_collection()
        return {"status": "success", "message": "Started normal state data collection"}
    except Exception as e:
        logger.error(f"Failed to start normal state collection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/normal/stop")
async def stop_normal_collection():
    """Stop collecting normal state data"""
    try:
        await training_service.stop_normal_collection()
        return {"status": "success", "message": "Stopped normal state data collection"}
    except Exception as e:
        logger.error(f"Failed to stop normal state collection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_training_stats():
    """Get statistics about the collected training data."""
    try:
        import asyncio
        from fastapi.responses import JSONResponse
        from fastapi.encoders import jsonable_encoder
        
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

        # Use the stats directly from the service
        response = {
            "status": "success",
            "stats": {
                "normal": stats["normal"],
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

@router.post("/auto_balance")
async def auto_balance_dataset():
    """Auto-balance the training dataset"""
    try:
        await training_service.auto_balance_dataset()
        return {"status": "success", "message": "Auto-balance started"}
    except Exception as e:
        logger.error(f"Error in auto-balancing dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error auto-balancing dataset: {str(e)}")

@router.get("/collection-status")
async def get_collection_status():
    """Get current data collection status"""
    try:
        from fastapi.responses import JSONResponse
        
        async with training_service._lock:  # Use async lock to ensure atomic access
            is_normal = training_service.is_collecting_normal and not training_service.current_anomaly
            is_anomaly = training_service.current_anomaly is not None
            
            status = {
                "is_collecting_normal": is_normal,
                "is_collecting_anomaly": is_anomaly,
                "current_type": "normal" if is_normal else training_service.current_anomaly["type"] if is_anomaly else None,
                "collection_options": {
                    "pre_collect": training_service.current_anomaly.get("pre_collect", True) if is_anomaly else False,
                    "post_collect": training_service.current_anomaly.get("post_collect", True) if is_anomaly else False
                } if is_anomaly else None,
                "collection_phase": "pre_anomaly" if is_anomaly and training_service.is_collecting_normal else
                                  "anomaly" if is_anomaly else
                                  "normal" if is_normal else None
            }
        
        return JSONResponse(
            content=status,
            headers={
                "Access-Control-Allow-Origin": "http://10.101.168.97:3000",
                "Access-Control-Allow-Credentials": "true"
            }
        )
    except Exception as e:
        logger.error(f"Failed to get collection status: {str(e)}")
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
    """Train the anomaly detection model using existing collected data."""
    try:
        logger.info("Starting model training process with existing collected data")
        
        # Temporarily disable workload check to ensure training can proceed
        metrics_service.set_check_workload_active(False)
        
        # Get training data from disk (not from current metrics collection)
        X, y = training_service.get_training_data()
        if len(X) == 0 or len(y) == 0:
            logger.error("No training data available in the dataset")
            raise HTTPException(status_code=400, detail="No training data available in the dataset")
            
        # Get access to diagnosis service
        from app.services.diagnosis_service import DiagnosisService
        diagnosis_service = DiagnosisService()
        
        # Train the model with the collected data
        result = diagnosis_service.train(X, y)
        logger.info(f"Model training completed successfully: {result['model_name']}")
        
        # Restore workload check setting
        metrics_service.set_check_workload_active(True)
        
        return {
            "status": "success", 
            "message": "Model trained successfully", 
            "model": result['model_name']
        }
    except Exception as e:
        logger.error(f"Failed to train model: {str(e)}")
        # Ensure workload check is restored even on error
        metrics_service.set_check_workload_active(True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reset")
async def reset_training_state():
    """Reset the training service state to clear any active collection."""
    try:
        result = await training_service.reset_state()
        return {
            "status": "success",
            "message": "Training state reset successfully"
        }
    except Exception as e:
        logger.error(f"Error resetting training state: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error resetting training state: {str(e)}")

@router.post("/workload/active")
async def set_workload_active(active: bool = True):
    """Set the workload active state to control metrics flow to training service."""
    try:
        result = metrics_service.set_workload_active(active)
        return {
            "status": "success",
            "message": f"Workload active state set to {active}",
            "previous_state": result["previous_state"],
            "current_state": result["current_state"]
        }
    except Exception as e:
        logger.error(f"Error setting workload active state: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error setting workload active state: {str(e)}")