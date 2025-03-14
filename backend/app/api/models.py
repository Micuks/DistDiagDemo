from fastapi import APIRouter, HTTPException, Query
from app.services.diagnosis_service import DiagnosisService
from fastapi.responses import JSONResponse
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from app.services.metrics_service import MetricsService
import os
import json

logger = logging.getLogger(__name__)

router = APIRouter()
diagnosis_service = DiagnosisService()
metrics_service = MetricsService()

class CompareModelsRequest(BaseModel):
    model_names: List[str] 

class ModelValidationRequest(BaseModel):
    model_name: str

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

@router.post("/validate")
async def validate_model(request: ModelValidationRequest):
    """Validate a model's integrity before loading"""
    try:
        model_name = request.model_name
        logger.info(f"Validating model: {model_name}")
        
        # 1. Check if model exists
        available_models = diagnosis_service.get_available_models()
        if model_name not in available_models:
            logger.warning(f"Model '{model_name}' not found")
            return JSONResponse(
                content={"valid": False, "error": f"Model '{model_name}' not found"},
                headers={
                    "Access-Control-Allow-Origin": "http://10.101.168.97:3000",
                    "Access-Control-Allow-Credentials": "true"
                }
            )
        
        # 2. Validate model directory structure
        model_path = os.path.join(diagnosis_service.models_path, model_name)
        if not os.path.exists(model_path):
            logger.warning(f"Model path not found: {model_path}")
            return JSONResponse(
                content={"valid": False, "error": "Model directory not found"},
                headers={
                    "Access-Control-Allow-Origin": "http://10.101.168.97:3000",
                    "Access-Control-Allow-Credentials": "true"
                }
            )
        
        # 3. Check if feature_info.json exists and is valid
        feature_info_path = os.path.join(model_path, 'feature_info.json')
        if not os.path.exists(feature_info_path):
            logger.warning(f"feature_info.json not found in {model_name}")
            return JSONResponse(
                content={"valid": False, "error": "feature_info.json not found"},
                headers={
                    "Access-Control-Allow-Origin": "http://10.101.168.97:3000",
                    "Access-Control-Allow-Credentials": "true"
                }
            )
        
        # 4. Try to parse feature_info.json
        try:
            with open(feature_info_path, 'r') as f:
                feature_info = json.load(f)
                logger.info(f"Successfully loaded feature_info.json: {feature_info}")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing feature_info.json: {str(e)}")
            return JSONResponse(
                content={"valid": False, "error": f"Invalid JSON in feature_info.json: {str(e)}"},
                headers={
                    "Access-Control-Allow-Origin": "http://10.101.168.97:3000",
                    "Access-Control-Allow-Credentials": "true"
                }
            )
        
        # 5. Check for required classifier files
        classifier_files = [f for f in os.listdir(model_path) if f.startswith('classifier_') and f.endswith('.joblib')] + [f for f in os.listdir(model_path) if f.startswith('clf_') and f.endswith('.pkl')]
        if not classifier_files:
            logger.warning(f"No classifier files found in {model_name}")
            return JSONResponse(
                content={"valid": False, "error": "No classifier files found"},
                headers={
                    "Access-Control-Allow-Origin": "http://10.101.168.97:3000",
                    "Access-Control-Allow-Credentials": "true"
                }
            )
        
        # All checks passed
        return JSONResponse(
            content={
                "valid": True, 
                "message": f"Model {model_name} is valid",
                "details": {
                    "classifiers": len(classifier_files),
                    "feature_info": feature_info
                }
            },
            headers={
                "Access-Control-Allow-Origin": "http://10.101.168.97:3000",
                "Access-Control-Allow-Credentials": "true"
            }
        )
        
    except Exception as e:
        logger.error(f"Error validating model: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"valid": False, "error": str(e)},
            headers={
                "Access-Control-Allow-Origin": "http://10.101.168.97:3000",
                "Access-Control-Allow-Credentials": "true"
            }
        )

@router.get("/ranks")
async def get_model_ranks(
    model_names: Optional[List[str]] = Query(None, alias="model_names[]"),
    threshold: float = Query(0.001, description="Score threshold for including anomalies")
):
    """Get RCA results for specified models using time series metrics"""
    try:
        logger.info(f"Received request for models: {model_names} with threshold: {threshold}")
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
            diagnosis_result = diagnosis_service.diagnose(metrics)
            
            # Filter anomalies by threshold from all_ranked_anomalies instead of using pre-filtered anomalies
            all_anomalies = diagnosis_result.get('all_ranked_anomalies', [])
            # Make sure to use the actual threshold from the request here
            filtered_anomalies = [a for a in all_anomalies if a.get('score', 0) > threshold]
            
            # Log the correct threshold being used
            logger.info(f"Filtering anomalies using threshold: {threshold}")
            logger.debug(f"Found {len(filtered_anomalies)} anomalies above threshold {threshold} from {len(all_anomalies)} total possibilities")
            
            # Include metric ranks for each anomaly's RCA
            for anomaly in filtered_anomalies:
                # Add metric ranks related to this anomaly type
                anomaly['related_metrics'] = get_related_metrics_for_anomaly(
                    metrics, 
                    anomaly.get('node', ''), 
                    anomaly.get('type', '')
                )
            
            results[current_model or "default"] = {
                "ranks": filtered_anomalies,
                "propagation_graph": diagnosis_result.get('propagation_graph', {}),
                "node_names": diagnosis_result.get('node_names', [])
            }
            logger.info(f"Default model analysis complete. Found {len(filtered_anomalies)} anomalies above threshold {threshold}")
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
                    diagnosis_result = diagnosis_service.diagnose(metrics)
                    
                    # Filter anomalies by threshold from all ranked anomalies
                    all_anomalies = diagnosis_result.get('all_ranked_anomalies', [])
                    # Make sure to use the correct threshold from the request
                    filtered_anomalies = [a for a in all_anomalies if a.get('score', 0) > threshold]
                    
                    # Include metric ranks for each anomaly's RCA
                    for anomaly in filtered_anomalies:
                        # Add metric ranks related to this anomaly type
                        anomaly['related_metrics'] = get_related_metrics_for_anomaly(
                            metrics, 
                            anomaly.get('node', ''), 
                            anomaly.get('type', '')
                        )
                    
                    logger.info(f"Analysis complete for {model_name}. Found {len(filtered_anomalies)} anomalies above threshold {threshold}")
                    
                    # Store results for this model
                    results[model_name] = {
                        "ranks": filtered_anomalies,
                        "propagation_graph": diagnosis_result.get('propagation_graph', {}),
                        "node_names": diagnosis_result.get('node_names', [])
                    }
                    logger.debug(f"Results for {model_name}: {len(filtered_anomalies)} ranks")
                    
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

def get_related_metrics_for_anomaly(metrics, node_name, anomaly_type):
    """Get related metrics for an anomaly's root cause analysis"""
    try:
        if not node_name or not anomaly_type or node_name not in metrics:
            return []
            
        node_metrics = metrics.get(node_name, {})
        related_metrics = []
        
        # Map anomaly types to their most relevant metric categories
        metric_categories = {
            "cpu_stress": ["cpu"],
            "io_bottleneck": ["io"],
            "network_bottleneck": ["network"],
            "cache_bottleneck": ["io", "memory"],
            "too_many_indexes": ["memory", "io"]
        }
        
        # Get the relevant categories for this anomaly type
        categories = metric_categories.get(anomaly_type, [])
        
        # Collect metrics from relevant categories
        for category in categories:
            category_metrics = node_metrics.get(category, {})
            for metric_name, metric_data in category_metrics.items():
                # Extract latest metric value if available
                metric_value = None
                if isinstance(metric_data, dict) and 'latest' in metric_data:
                    metric_value = metric_data['latest']
                elif isinstance(metric_data, (int, float)):
                    metric_value = metric_data
                
                if metric_value is not None:
                    related_metrics.append({
                        "name": metric_name,
                        "category": category,
                        "value": metric_value
                    })
        
        # Sort by value (descending) to highlight most significant metrics
        related_metrics.sort(key=lambda x: x.get('value', 0), reverse=True)
        
        # Return top 5 most significant metrics
        return related_metrics[:5]
        
    except Exception as e:
        logger.error(f"Error getting related metrics for anomaly: {str(e)}")
        return []