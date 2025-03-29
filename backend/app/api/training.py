from fastapi import APIRouter, HTTPException
from typing import List, Dict, Optional
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
from app.services.training_service import training_service
from app.services.metrics_service import metrics_service
import threading
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from fastapi.applications import FastAPI
from starlette.requests import Request
from starlette.responses import Response
import traceback
from datetime import datetime

from backend.app.schemas.anomaly import AnomalyRequest

# Import and initialize diagnosis service properly
try:
    from app.services.diagnosis_service import diagnosis_service
    logger = logging.getLogger(__name__)
    logger.info("Successfully imported DiagnosisService")
except Exception as e:
    logger = logging.getLogger(__name__)
    logger.error(f"Error importing DiagnosisService: {str(e)}")
    diagnosis_service = None

router = APIRouter()

def _init_training_status(stats: Dict = {}):
    return {
    "stage": "idle",  # idle, preprocessing, training, evaluating, saving, completed, failed
    "progress": 0,    # 0-100 percentage
    "message": "",    # Detailed status message
    "error": None,    # Error message if any
    "stats": stats       # Training stats like accuracy, etc.
} 

# Global training state object
TRAINING_STATE = {
    "in_progress": False,
    "status": _init_training_status(),
    "finished_at": None
}
_training_executor = ThreadPoolExecutor(max_workers=1)  # Dedicated executor for training

# Add global variables for tracking collection state
_normal_collection_in_progress = False
_anomaly_collection_in_progress = False
_collection_lock = asyncio.Lock()

# Add this after _training_executor initialization
_status_sync_task = None

class TrainingStopRequest(BaseModel):
    save_post_data: Optional[bool] = True

class CompoundAnomalyRequest(BaseModel):
    anomaly_types: List[str]
    node: Optional[str] = None

@router.post("/collect")
async def start_training_collection(request: AnomalyRequest):
    """Start collecting training data for a specific anomaly type"""
    global _anomaly_collection_in_progress
    
    # Check if collection is already in progress
    async with _collection_lock:
        if _anomaly_collection_in_progress:
            logger.warning("Anomaly collection already in progress, ignoring duplicate request")
            return {
                "status": "pending", 
                "message": "Anomaly collection is already in progress",
                "type": request.type,
                "node": request.node
            }
        # Mark collection as in progress
        _anomaly_collection_in_progress = True
    
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
        # Reset collection flag on error
        async with _collection_lock:
            _anomaly_collection_in_progress = False
        logger.error(f"Error starting training collection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting training collection: {str(e)}")

@router.post("/stop")
async def stop_training_collection(request: TrainingStopRequest):
    """Stop collecting training data for an anomaly type"""
    global _anomaly_collection_in_progress
    
    # Check if there's no collection to stop
    async with _collection_lock:
        if not _anomaly_collection_in_progress:
            logger.warning("No anomaly collection in progress, ignoring stop request")
            return {
                "status": "warning", 
                "message": "No anomaly collection was in progress"
            }
    
    try:
        metrics_service.toggle_send_to_training(False);
        if hasattr(training_service, "stop_anomaly_collection"):
            training_service.stop_anomaly_collection(save_post_data=request.save_post_data)
        else:
            # Fallback to generic stop
            training_service.stop_collection()
            
        # Mark collection as stopped
        async with _collection_lock:
            _anomaly_collection_in_progress = False
            
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
    global _normal_collection_in_progress
    
    # Check if collection is already in progress
    async with _collection_lock:
        if _normal_collection_in_progress:
            logger.warning("Normal collection already in progress, ignoring duplicate request")
            return {
                "status": "pending", 
                "message": "Normal state collection is already in progress"
            }
        # Mark normal collection as in progress
        _normal_collection_in_progress = True
    
    try:
        await training_service.start_normal_collection()
        return {"status": "success", "message": "Started normal state data collection"}
    except Exception as e:
        # Reset collection flag on error
        async with _collection_lock:
            _normal_collection_in_progress = False
        logger.error(f"Failed to start normal state collection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/normal/stop")
async def stop_normal_collection():
    """Stop collecting normal state data"""
    global _normal_collection_in_progress
    
    # Check if there's no collection to stop
    async with _collection_lock:
        if not _normal_collection_in_progress:
            logger.warning("No normal collection in progress, ignoring stop request")
            return {
                "status": "warning", 
                "message": "No normal state collection was in progress"
            }
    
    try:
        await training_service.stop_normal_collection()
        
        # Mark collection as stopped
        async with _collection_lock:
            _normal_collection_in_progress = False
            
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
        result = await training_service.auto_balance_dataset()
        if isinstance(result, dict):
            return {
                "status": "success" if result.get("success", False) else "warning",
                "message": result.get("message", "Auto-balance process completed"),
                "action_taken": result.get("message", "No specific action needed")
            }
        else:
            # For backward compatibility
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
            
            # Add current timestamp and collection start time for frontend state verification
            current_time = datetime.now().isoformat()
            collection_start_time = training_service.collection_start_time
            
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
                                  "normal" if is_normal else None,
                "last_update_time": current_time,
                "collection_start_time": collection_start_time,
                "collection_duration": training_service.collection_duration if hasattr(training_service, "collection_duration") else 180,
                "debug_info": {
                    "metrics_buffer_size": len(training_service.metrics_buffer) if hasattr(training_service, "metrics_buffer") else 0,
                    "normal_buffer_size": len(training_service.normal_state_buffer) if hasattr(training_service, "normal_state_buffer") else 0,
                    "has_current_case": training_service.current_case is not None,
                    "has_current_anomaly": training_service.current_anomaly is not None
                }
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

@router.get("/training-status")
async def get_training_status():
    """Get the current status of model training."""
    global TRAINING_STATE
    # If training finished more than 30 seconds ago, reset status to idle
    if (not TRAINING_STATE["in_progress"]) and TRAINING_STATE["finished_at"]:
        elapsed = (datetime.now() - TRAINING_STATE["finished_at"]).total_seconds()
        if elapsed > 5:
            TRAINING_STATE["status"] = _init_training_status()
            TRAINING_STATE["finished_at"] = None
    return TRAINING_STATE["status"]

@router.post("/train")
async def train_model():
    """Train the anomaly detection model using existing collected data."""
    global TRAINING_STATE

    if TRAINING_STATE["in_progress"]:
        logger.warning("Training already in progress, ignoring duplicate request")
        return {
            "status": "pending",
            "message": "Training is already in progress, please wait"
        }

    TRAINING_STATE["in_progress"] = True
    TRAINING_STATE["status"] = {
        "stage": "preprocessing",
        "progress": 0,
        "message": "Preparing training data...",
        "error": None,
        "stats": {}
    }

    asyncio.create_task(_run_training_in_background())
    return {
        "status": "accepted",
        "message": "Training started in the background"
    }

async def update_progress():
    """Periodically update training progress"""
    progress = 50
    while True:
        await asyncio.sleep(0.1)
        if progress < 75:
            progress += 1
        TRAINING_STATE["status"].update({
            "stage": "training",
            "progress": progress,
            "message": f"Training model... ({progress}%)"
        })
        logger.debug(f"Updated training progress to {progress}%")

async def _run_training_in_background():
    """Run the model training process in a background task."""
    global TRAINING_STATE
    try:
        logger.info("Starting model training process with existing collected data")

        metrics_service.set_check_workload_active(False)
        original_send_setting = metrics_service.send_to_training
        metrics_service.toggle_send_to_training(False)
        
        TRAINING_STATE["status"].update({
            "stage": "preprocessing",
            "progress": 10,
            "message": "Loading training data from disk..."
        })

        X, y = training_service.get_training_data()
        if len(X) == 0 or len(y) == 0:
            logger.error("No training data available in the dataset")
            TRAINING_STATE["status"].update({
                "stage": "failed",
                "message": "No training data available in the dataset",
                "error": "No training data available in the dataset",
                "progress": 0
            })
            return

        logger.info(f"Training data shapes: X={X.shape}, y={y.shape}")
        if diagnosis_service is None:
            logger.error("DiagnosisService not properly initialized")
            TRAINING_STATE["status"].update({
                "stage": "failed",
                "message": "Model service not properly initialized",
                "error": "DiagnosisService import failed",
                "progress": 0
            })
            return

        TRAINING_STATE["status"].update({
            "stage": "preprocessing",
            "progress": 30,
            "message": f"Processing training data: {X.shape[0]} samples..."
        })

        progress_task = asyncio.create_task(update_progress())

        TRAINING_STATE["status"].update({
            "stage": "training",
            "progress": 50,
            "message": "Training model..."
        })

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                _training_executor,
                diagnosis_service.train, 
                X, 
                y
            )
            if result is None or not isinstance(result, dict):
                raise ValueError(f"Invalid result from model training: {result}")
            logger.info(f"Model training completed successfully: {result.get('model_name', 'unnamed')}")
            TRAINING_STATE["status"].update({
                "stage": "completed",
                "progress": 100,
                "message": "Model training completed successfully",
                "stats": result.get("metrics", {})
            })
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise ValueError(f"Model training failed: {str(e)}")
        finally:
            if progress_task and not progress_task.done():
                progress_task.cancel()
                try:
                    await progress_task
                except asyncio.CancelledError:
                    pass

        TRAINING_STATE["finished_at"] = datetime.now()
        TRAINING_STATE["status"].update({
            "stage": "evaluating",
            "progress": 80,
            "message": "Evaluating model performance..."
        })
        # If there are evaluation metrics
        model_metrics = {}
        if 'metrics' in result:
            model_metrics = result['metrics']
            logger.info(f"Model evaluation metrics: {model_metrics}")
        TRAINING_STATE["status"].update({
            "stage": "completed",
            "progress": 100,
            "message": "Model training completed successfully",
            "stats": model_metrics
        })
        logger.info("Training process completed successfully")
    except Exception as e:
        logger.error(f"Failed to train model: {str(e)}")
        logger.error(traceback.format_exc())
        TRAINING_STATE["status"].update({
            "stage": "failed",
            "progress": 0,
            "message": f"Training failed: {str(e)}",
            "error": str(e)
        })
        TRAINING_STATE["finished_at"] = datetime.now()
    finally:
        metrics_service.set_check_workload_active(True)
        metrics_service.toggle_send_to_training(original_send_setting)
        TRAINING_STATE["in_progress"] = False
        logger.info("Training process marked as complete")

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

@router.post("/force-reset")
async def force_reset_collection_state():
    """Force reset the collection state flags in case of desynchronization between API and backend."""
    global _anomaly_collection_in_progress, _normal_collection_in_progress
    
    try:
        # Reset API flags
        async with _collection_lock:
            was_anomaly_collecting = _anomaly_collection_in_progress
            was_normal_collecting = _normal_collection_in_progress
            
            _anomaly_collection_in_progress = False
            _normal_collection_in_progress = False
        
        # Reset actual training service state
        try:
            await training_service.reset_state()
            logger.info("Successfully reset training service state")
        except Exception as e:
            logger.error(f"Error resetting training service state: {str(e)}")
        
        return {
            "status": "success",
            "message": "Collection state forcefully reset",
            "previous_state": {
                "anomaly_collection": was_anomaly_collecting,
                "normal_collection": was_normal_collecting
            },
            "current_state": {
                "anomaly_collection": False,
                "normal_collection": False
            }
        }
    except Exception as e:
        logger.error(f"Error performing force reset: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error performing force reset: {str(e)}")

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

@router.get("/process-status")
async def get_process_status():
    """Get the status of all training processes (training, anomaly collection, normal collection)"""
    async with _collection_lock, _training_executor:
        return {
            "training_in_progress": TRAINING_STATE["in_progress"],
            "anomaly_collection_in_progress": _anomaly_collection_in_progress,
            "normal_collection_in_progress": _normal_collection_in_progress
        }

@router.post("/compound")
async def start_compound_collection(request: CompoundAnomalyRequest):
    """
    Start collecting training data for compound anomaly scenarios (multiple simultaneous anomalies)
    
    Compound anomalies are situations where multiple issues occur simultaneously,
    creating complex interactions that are more challenging to diagnose.
    
    Args:
        request: CompoundAnomalyRequest with:
            - anomaly_types: List of anomaly types to simulate simultaneously (min 2)
            - node: Optional target node (auto-selected if not specified)
    """
    global _anomaly_collection_in_progress
    
    # Validate we have at least 2 anomaly types
    if len(request.anomaly_types) < 2:
        raise HTTPException(
            status_code=400, 
            detail="Compound collection requires at least 2 anomaly types"
        )
    
    # Check if collection is already in progress
    async with _collection_lock:
        if _anomaly_collection_in_progress:
            logger.warning("Anomaly collection already in progress, ignoring duplicate request")
            return {
                "status": "pending", 
                "message": "Anomaly collection is already in progress",
                "types": request.anomaly_types,
                "node": request.node
            }
        # Mark collection as in progress
        _anomaly_collection_in_progress = True
    
    try:
        # First ensure metrics collection is active
        metrics_service.toggle_send_to_training(True)
        
        # Start compound collection
        await training_service.start_compound_collection(
            anomaly_types=request.anomaly_types,
            node=request.node,
        )
        
        logger.info(f"Started compound anomaly collection with types: {request.anomaly_types}")
        
        # Inform user about longer collection period for compound anomalies
        collection_duration = getattr(training_service, 'collection_duration', 180)
        extended_duration = int(collection_duration * 1.5)  # 50% longer for compound anomalies
        
        return {
            "status": "success",
            "message": f"Started compound anomaly collection with {len(request.anomaly_types)} anomaly types",
            "types": request.anomaly_types,
            "node": request.node or "auto-selected",
            "expected_duration": extended_duration,
            "note": "Collection period is extended by 50% for compound anomalies to ensure sufficient data"
        }
    except Exception as e:
        # Reset collection flag on error
        async with _collection_lock:
            _anomaly_collection_in_progress = False
        logger.error(f"Error starting compound anomaly collection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting compound anomaly collection: {str(e)}")

@router.post("/train_compound_model")
async def train_compound_model():
    """
    Train a specialized model for detecting compound anomalies
    
    This endpoint will train a model specifically optimized for:
    1. Identifying multiple simultaneous anomalies on the same node 
    2. Understanding anomaly propagation across nodes
    3. Determining the root cause in complex scenarios
    """
    try:
        # Check if sufficient compound anomaly data exists
        stats = await training_service.get_dataset_stats()
        compound_cases = stats.get("compound", 0)
        
        if compound_cases < 5:  # Require at least 5 compound anomaly samples
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": f"Insufficient compound anomaly data. Found {compound_cases} cases but require at least 5. Use the /training/compound endpoint to collect more data."
                }
            )
        
        # Get training data specifically optimized for compound scenarios
        X, y = await training_service.get_training_data()
        
        if len(X) == 0 or len(y) == 0:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "Could not extract valid training data"
                }
            )
            
        # Train the model with the multi-node, multi-label dataset
        result = diagnosis_service.train(X, y)
        
        # Add metadata about compound anomaly training
        result["compound_optimized"] = True
        result["compound_cases_used"] = compound_cases
        
        return JSONResponse(
            content={
                "status": "success",
                "message": f"Successfully trained compound anomaly model with {len(X)} samples, including {compound_cases} compound cases",
                "model": result
            }
        )
    except Exception as e:
        logger.error(f"Error training compound model: {str(e)}")
        traceback_str = traceback.format_exc()
        logger.debug(f"Traceback: {traceback_str}")
        
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Failed to train compound anomaly model: {str(e)}"
            }
        )

# Add this function before the routes
async def _sync_collection_status():
    """Periodically check the actual collection status and sync with API flags"""
    global _anomaly_collection_in_progress, _normal_collection_in_progress
    try:
        while True:
            # Check every 5 seconds
            await asyncio.sleep(5)
            
            try:
                # Check training service state and sync with our flags
                async with training_service._lock:
                    actual_normal_active = training_service.is_collecting_normal and not training_service.current_anomaly
                    actual_anomaly_active = training_service.current_anomaly is not None
                    
                    # Sync API flags with actual state
                    async with _collection_lock:
                        if _anomaly_collection_in_progress != actual_anomaly_active:
                            logger.warning(f"Fixing desynchronized anomaly collection state: API={_anomaly_collection_in_progress}, actual={actual_anomaly_active}")
                            _anomaly_collection_in_progress = actual_anomaly_active
                            
                        if _normal_collection_in_progress != actual_normal_active:
                            logger.warning(f"Fixing desynchronized normal collection state: API={_normal_collection_in_progress}, actual={actual_normal_active}")
                            _normal_collection_in_progress = actual_normal_active
                
                # Additionally check collection status from debug info
                collection_status = await training_service.get_collection_status()
                if collection_status and isinstance(collection_status, dict):
                    debug_info = collection_status.get("debug_info", {})
                    has_current_anomaly = debug_info.get("has_current_anomaly", False)
                    
                    async with _collection_lock:
                        if _anomaly_collection_in_progress and not has_current_anomaly:
                            logger.warning(f"Collection flag inconsistency detected: flag={_anomaly_collection_in_progress}, actual={has_current_anomaly}")
                            _anomaly_collection_in_progress = False
            except Exception as e:
                logger.error(f"Error in collection status sync: {str(e)}")
                # Reset flags if error occurs during status check
                async with _collection_lock:
                    _anomaly_collection_in_progress = False
                    _normal_collection_in_progress = False
                    
    except asyncio.CancelledError:
        logger.debug("Collection status sync task cancelled")
        raise
    except Exception as e:
        logger.error(f"Error in collection status sync task: {str(e)}")

# Add this function at the end of the file, right before setup_shutdown_handler
def start_status_sync_task():
    """Start the background task to sync collection status between API and service"""
    global _status_sync_task
    
    if _status_sync_task is None or _status_sync_task.done():
        _status_sync_task = asyncio.create_task(_sync_collection_status())
        logger.info("Started collection status sync task")

# Modify setup_shutdown_handler to include stopping the sync task
def setup_shutdown_handler(app: FastAPI):
    """Set up a shutdown handler to gracefully close resources."""
    @app.on_event("startup")
    async def startup_event():
        logger.info("Starting training API and background tasks...")
        start_status_sync_task()
    
    @app.on_event("shutdown")
    async def shutdown_event():
        global _status_sync_task
        
        logger.info("Shutting down training executor and tasks...")
        try:
            # Cancel status sync task if running
            if _status_sync_task and not _status_sync_task.done():
                _status_sync_task.cancel()
                try:
                    await _status_sync_task
                except asyncio.CancelledError:
                    pass
                
            _training_executor.shutdown(wait=False)
            logger.info("Training resources shutdown complete")
        except Exception as e:
            logger.error(f"Error shutting down training resources: {str(e)}")

@router.post("/test-training")
async def test_training_process():
    """
    Test the training process by manually loading training data and training the model.
    This is a diagnostic endpoint to help identify and fix training issues.
    """
    try:
        # Import diagnosis service
        from app.services.diagnosis_service import diagnosis_service
        
        logger.info("Starting direct training test")
        
        # Get training data
        X, y = training_service.get_training_data()
        
        # Check if we have any data
        if len(X) == 0 or len(y) == 0:
            logger.error("No training data available")
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "No training data available",
                    "data_shapes": {
                        "X": str((0, 0)),
                        "y": str((0, 0))
                    }
                }
            )
        
        # Log data shapes
        logger.info(f"Training data shapes: X={X.shape}, y={y.shape}")
        
        # Use the direct_train method for more detailed diagnosis
        result = diagnosis_service.direct_train(X, y)
        
        if result.get("success", False):
            return JSONResponse(
                content={
                    "status": "success",
                    "message": "Direct training test completed successfully",
                    "data_shapes": {
                        "X": str(X.shape),
                        "y": str(y.shape),
                    },
                    "model_info": result.get("result", {})
                }
            )
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "Direct training test failed",
                    "error": result.get("error", "Unknown error"),
                    "data_shapes": {
                        "X": str(X.shape),
                        "y": str(y.shape),
                    }
                }
            )
    except Exception as e:
        logger.error(f"Error in test training endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error testing training process: {str(e)}"
            }
        )