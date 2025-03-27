from fastapi import APIRouter, HTTPException, Query
from app.services.diagnosis_service import DiagnosisService
from fastapi.responses import JSONResponse
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from app.services.metrics_service import MetricsService
import os
import json
import numpy as np
from collections import defaultdict
from app.services.metric_inferor_service import analyze_node_metrics, get_metric_summary

logger = logging.getLogger(__name__)

router = APIRouter()
diagnosis_service = DiagnosisService()
metrics_service = MetricsService()

class CompareModelsRequest(BaseModel):
    model_names: List[str] 

class ModelValidationRequest(BaseModel):
    model_name: str

@router.get("/{model_name}/performance")
async def get_model_performance(model_name: str):
    """Get performance metrics for a specific model"""
    try:
        # Ensure model exists
        available_models = diagnosis_service.get_available_models()
        if model_name not in available_models:
            logger.warning(f"Model '{model_name}' not found")
            return JSONResponse(
                status_code=404,
                content={"error": f"Model '{model_name}' not found"},
                headers={
                    "Access-Control-Allow-Origin": "http://10.101.168.97:3000",
                    "Access-Control-Allow-Credentials": "true"
                }
            )
        
        # Check for metrics file with the correct naming pattern
        model_path = os.path.join(diagnosis_service.models_path, model_name)
        metrics_path = os.path.join(model_path, 'metrics', f"{model_name}_metrics.json")
        
        if not os.path.exists(metrics_path):
            logger.warning(f"Metrics file not found at primary path: {metrics_path}")
            # Try directory listing to find any metrics file
            metrics_dir = os.path.join(model_path, 'metrics')
            if os.path.exists(metrics_dir):
                metrics_files = [f for f in os.listdir(metrics_dir) if f.endswith('_metrics.json')]
                if metrics_files:
                    metrics_path = os.path.join(metrics_dir, metrics_files[0])
                    logger.info(f"Found alternative metrics file: {metrics_path}")
                else:
                    logger.warning(f"No metrics files found in directory: {metrics_dir}")
                    return JSONResponse(
                        status_code=404,
                        content={"error": f"No performance metrics found for model '{model_name}'"},
                        headers={
                            "Access-Control-Allow-Origin": "http://10.101.168.97:3000",
                            "Access-Control-Allow-Credentials": "true"
                        }
                    )
            else:
                logger.warning(f"Metrics directory not found: {metrics_dir}")
                return JSONResponse(
                    status_code=404,
                    content={"error": f"No performance metrics found for model '{model_name}'"},
                    headers={
                        "Access-Control-Allow-Origin": "http://10.101.168.97:3000",
                        "Access-Control-Allow-Credentials": "true"
                    }
                )
        
        # Load metrics from file
        try:
            logger.info(f"Loading metrics from: {metrics_path}")
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                logger.info(f"Successfully loaded metrics for model {model_name}")
                
                return JSONResponse(
                    content=metrics,
                    headers={
                        "Access-Control-Allow-Origin": "http://10.101.168.97:3000",
                        "Access-Control-Allow-Credentials": "true"
                    }
                )
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing metrics for model '{model_name}': {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": f"Invalid metrics data for model '{model_name}': {str(e)}"},
                headers={
                    "Access-Control-Allow-Origin": "http://10.101.168.97:3000",
                    "Access-Control-Allow-Credentials": "true"
                }
            )
            
    except Exception as e:
        logger.error(f"Failed to get model performance: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
            headers={
                "Access-Control-Allow-Origin": "http://10.101.168.97:3000",
                "Access-Control-Allow-Credentials": "true"
            }
        )

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
        if not node_name or not anomaly_type:
            return []
            
        # Translate node name if needed
        translated_node = translate_node_name(node_name, metrics)
        if translated_node != node_name:
            logger.debug(f"Translated node name from {node_name} to {translated_node}")
            node_name = translated_node
        
        # Check if node exists in metrics
        if node_name not in metrics:
            logger.warning(f"Node {node_name} not found in metrics data")
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

@router.get("/metrics_ranks")
async def get_metric_ranks(
    node: str,
    model_name: Optional[str] = None,
):
    """Get detailed metric rankings for a specific node using MetricRooter analysis, independent of anomaly type"""
    try:
        logger.info(f"Received request for metric rankings: node={node}, model={model_name}")
        
        # Get detailed metrics with time series data
        metrics = metrics_service.get_detailed_metrics()
        if not metrics:
            logger.warning("No metrics data available")
            return JSONResponse(
                content={"error": "No metrics data available"},
                headers={
                    "Access-Control-Allow-Origin": "http://10.101.168.97:3000",
                    "Access-Control-Allow-Credentials": "true"
                }
            )
        
        # Translate node name if it's an IP address
        translated_node = translate_node_name(node, metrics)
        if translated_node != node:
            logger.info(f"Translated node name from {node} to {translated_node}")
            node = translated_node
        
        # Check if node exists in metrics
        if node not in metrics:
            logger.warning(f"Node {node} not found in metrics data")
            return JSONResponse(
                content={"error": f"Node {node} not found in metrics data"},
                headers={
                    "Access-Control-Allow-Origin": "http://10.101.168.97:3000",
                    "Access-Control-Allow-Credentials": "true"
                }
            )
        
        node_metrics = metrics[node]

        # Perform the metric ranking analysis
        analysis_result = analyze_node_metrics(node_metrics, node)
        
        # Return the results
        return JSONResponse(
            content=analysis_result,
            headers={
                "Access-Control-Allow-Origin": "http://10.101.168.97:3000",
                "Access-Control-Allow-Credentials": "true"
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get metric rankings: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
            headers={
                "Access-Control-Allow-Origin": "http://10.101.168.97:3000",
                "Access-Control-Allow-Credentials": "true"
            }
        )

@router.get("/metrics_summary")
async def get_metrics_summary(
    node: str,
    languages: Optional[List[str]] = Query(["Chinese", "English"], alias="languages[]"),
):
    """Get an LLM-generated summary and actions for metrics analysis separately"""
    try:
        logger.info(f"Received request for metrics summary: node={node}, languages={languages}")
        
        # Get detailed metrics with time series data
        metrics = metrics_service.get_detailed_metrics()
        if not metrics:
            logger.warning("No metrics data available")
            return JSONResponse(
                content={"error": "No metrics data available"},
                headers={
                    "Access-Control-Allow-Origin": "http://10.101.168.97:3000",
                    "Access-Control-Allow-Credentials": "true"
                }
            )
        
        # Translate node name if it's an IP address
        translated_node = translate_node_name(node, metrics)
        if translated_node != node:
            logger.info(f"Translated node name from {node} to {translated_node}")
            node = translated_node
        
        # Check if node exists in metrics
        if node not in metrics:
            logger.warning(f"Node {node} not found in metrics data")
            return JSONResponse(
                content={"error": f"Node {node} not found in metrics data"},
                headers={
                    "Access-Control-Allow-Origin": "http://10.101.168.97:3000",
                    "Access-Control-Allow-Credentials": "true"
                }
            )
        
        node_metrics = metrics[node]

        # First get metrics analysis to get ranked metrics
        analysis_result = analyze_node_metrics(node_metrics, node)
        ranked_metrics = analysis_result.get("metrics", [])
        
        # Generate summary and actions with LLM
        result = get_metric_summary(ranked_metrics, node, languages)
        
        # Return summary and actions
        return JSONResponse(
            content={
                "summary": result.get("summary", ""),
                "actions": result.get("actions", [])
            },
            headers={
                "Access-Control-Allow-Origin": "http://10.101.168.97:3000",
                "Access-Control-Allow-Credentials": "true"
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get metrics summary: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
            headers={
                "Access-Control-Allow-Origin": "http://10.101.168.97:3000",
                "Access-Control-Allow-Credentials": "true"
            }
        )

def translate_node_name(node_name, metrics=None):
    """
    Utility function to translate between IP addresses and Kubernetes pod names.
    This helps maintain consistency across different parts of the application.
    
    Args:
        node_name: The node name or IP address to translate
        metrics: Optional metrics dictionary to check for node names
        
    Returns:
        Translated node name if a match is found, otherwise the original node name
    """
    try:
        # If metrics is provided and node already exists, no translation needed
        if metrics and node_name in metrics:
            return node_name
            
        # Get available nodes from k8s_service
        k8s_nodes = []
        try:
            # Try to get K8s nodes from service
            from app.services.k8s_service import k8s_service
            try:
                # Use event loop if available
                import asyncio
                loop = asyncio.get_event_loop()
                k8s_nodes = loop.run_until_complete(k8s_service.get_available_nodes())
            except Exception as e:
                logger.debug(f"Error getting nodes from k8s_service: {str(e)}")
                
                # Fallback to WorkloadService
                from app.services.workload_service import workload_service
                k8s_nodes = workload_service.get_available_nodes()
        except Exception as e:
            logger.warning(f"Failed to get nodes from services: {str(e)}")
        
        # First check if node_name is directly in K8s nodes
        if node_name in k8s_nodes:
            return node_name
            
        # Then check against metric nodes if provided
        if metrics:
            for metric_node in metrics.keys():
                # If one contains the other (partial match)
                if node_name in metric_node or metric_node in node_name:
                    logger.info(f"Translated node {node_name} to {metric_node} based on metrics")
                    return metric_node
        
        # Finally check against k8s nodes for partial matches
        for k8s_node in k8s_nodes:
            if node_name in k8s_node or k8s_node in node_name:
                logger.info(f"Translated node {node_name} to {k8s_node} based on K8s nodes")
                return k8s_node
        
        # No translation found, return original
        return node_name
        
    except Exception as e:
        logger.error(f"Error translating node name: {str(e)}")
        return node_name