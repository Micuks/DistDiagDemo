from fastapi import APIRouter, HTTPException, Query
from app.services.diagnosis_service import DiagnosisService
from fastapi.responses import JSONResponse
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from app.services.metrics_service import MetricsService

logger = logging.getLogger(__name__)

router = APIRouter()
diagnosis_service = DiagnosisService()
metrics_service = MetricsService()

class CompareModelsRequest(BaseModel):
    model_names: List[str] 

@router.get("/list")
async def list_models():
    """Get list of available models"""
    try:
        models = diagnosis_service.get_available_models()
        return JSONResponse(
            content=models,
            headers={
                "Access-Control-Allow-Origin": "http://10.101.168.97:3000",
                "Access-Control-Allow-Credentials": "true"
            }
        )
    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
            headers={
                "Access-Control-Allow-Origin": "http://10.101.168.97:3000",
                "Access-Control-Allow-Credentials": "true"
            }
        )

@router.get("/ranks")
async def get_model_ranks(model_names: Optional[List[str]] = Query(None, alias="model_names[]")):
    """Get RCA results for specified models using time series metrics"""
    try:
        logger.info(f"Received request for models: {model_names}")
        # Get detailed metrics with time series data
        metrics = metrics_service.get_detailed_metrics()
        if not metrics:
            logger.warning("No metrics data available")
            return JSONResponse(
                content={},
                headers={
                    "Access-Control-Allow-Origin": "http://10.101.168.97:3000",
                    "Access-Control-Allow-Credentials": "true"
                }
            )
        
        # Store current model to restore it later
        current_model = diagnosis_service.active_model
        logger.info(f"Starting model analysis. Current active model: {current_model}")
        
        # Results dictionary to hold analysis for each model
        results = {}
        
        # If no models specified, use current model
        if not model_names:
            logger.info("No models specified, using current model")
            ranks = diagnosis_service.analyze_metrics(metrics)
            results[current_model or "default"] = {
                "ranks": ranks,
            }
            logger.info(f"Default model analysis complete. Found {len(ranks)} anomalies")
        else:
            logger.info(f"Processing requested models: {model_names}")
            # Check if all models exist first
            available_models = set(diagnosis_service.get_available_models())
            logger.debug(f"Available models: {available_models}")
            
            for model_name in model_names:
                if model_name not in available_models:
                    logger.warning(f"Model not found: {model_name}")
                    # Include an error message for models that don't exist
                    results[model_name] = {
                        "error": f"Model '{model_name}' not found",
                        "ranks": [],
                    }
            
            # Analyze metrics for each requested model
            for model_name in model_names:
                # Skip models we already know don't exist
                if model_name in results:
                    logger.debug(f"Skipping already processed model: {model_name}")
                    continue
                    
                try:
                    logger.info(f"Processing model: {model_name}")
                    # Store original model and attempt switch
                    original_active_model = diagnosis_service.active_model
                    success = diagnosis_service.switch_model(model_name)
                    
                    # Verify model actually switched
                    if not success or diagnosis_service.active_model != model_name:
                        logger.error(f"Failed to verify model switch to {model_name}")
                        results[model_name] = {
                            "error": f"Model switch failed for '{model_name}'",
                            "ranks": [],
                        }
                        continue
                    
                    # Analyze metrics with this model
                    logger.debug(f"Analyzing metrics with model {model_name}")
                    ranks = diagnosis_service.analyze_metrics(metrics)
                    logger.info(f"Analysis complete for {model_name}. Found {len(ranks)} anomalies")
                    
                    # Store results for this model
                    results[model_name] = {
                        "ranks": ranks,
                    }
                    logger.debug(f"Results for {model_name}: {len(ranks)} ranks")
                    
                except Exception as model_error:
                    logger.error(f"Error analyzing with model {model_name}: {str(model_error)}")
                    results[model_name] = {
                        "error": str(model_error),
                        "ranks": [],
                    }
                    # Attempt to restore original model if possible
                    if original_active_model:
                        try:
                            diagnosis_service.switch_model(original_active_model)
                            logger.info(f"Restored to original model after error: {original_active_model}")
                        except Exception as restore_error:
                            logger.error(f"Failed to restore original model: {str(restore_error)}")
        
        # Restore original model
        if current_model:
            logger.info(f"Restoring to initial model: {current_model}")
            try:
                success = diagnosis_service.switch_model(current_model)
                if not success:
                    logger.error(f"Failed to restore original model {current_model}")
                else:
                    logger.info(f"Successfully restored to original model")
            except Exception as restore_error:
                logger.error(f"Error restoring original model: {str(restore_error)}")
        
        # Log final results summary
        logger.info(f"Analysis complete. Results summary:")
        for model_name, result in results.items():
            if "error" in result:
                logger.info(f"  {model_name}: ERROR - {result['error']}")
            else:
                logger.info(f"  {model_name}: {len(result['ranks'])} anomalies")
            
        return JSONResponse(
            content=results,
            headers={
                "Access-Control-Allow-Origin": "http://10.101.168.97:3000",
                "Access-Control-Allow-Credentials": "true"
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get model ranks: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
            headers={
                "Access-Control-Allow-Origin": "http://10.101.168.97:3000",
                "Access-Control-Allow-Credentials": "true"
            }
        )