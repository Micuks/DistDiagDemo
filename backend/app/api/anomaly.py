from fastapi import APIRouter, HTTPException, BackgroundTasks, Response, Depends, Query
from typing import List, Dict, Optional, Union
from datetime import datetime
import asyncio
from pydantic import BaseModel
from app.services.k8s_service import K8sService
from app.services.metrics_service import MetricsService
from app.services.diagnosis_service import DiagnosisService
from app.services.training_service import training_service
from app.schemas.anomaly import AnomalyRequest, MetricsResponse, AnomalyRankResponse, ActiveAnomalyResponse, AnomalyResponse, AnomalyStatus, AnomalyConfig, AnomalyInfo
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
from fastapi.responses import JSONResponse, RedirectResponse

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

async def clear_caches():
    """Clear all caches including Redis and K8s service caches"""
    try:
        # Get the Redis backend from FastAPICache
        backend = FastAPICache.get_backend()
        if backend and hasattr(backend, '_redis'):
            # Clear all keys with our prefix
            redis = backend._redis
            if redis:
                keys = await redis.keys(f"{FastAPICache.get_prefix()}*")
                if keys:
                    await redis.delete(*keys)
                logger.info("Cleared Redis cache")
    except Exception as e:
        logger.warning(f"Failed to clear Redis cache: {str(e)}")
    
    try:
        # Clear K8s service cache
        k8s_service.invalidate_cache()
        
        # Force the cache to refresh on next request by resetting timestamps
        k8s_service._last_request_time = 0
        k8s_service._last_cache_update = 0
        
        # Reset any other internal caches
        if hasattr(k8s_service, '_cache'):
            k8s_service._cache = {}
        
        logger.info("Cleared K8s service cache")
    except Exception as e:
        logger.warning(f"Failed to clear K8s service cache: {str(e)}")

@router.get("/nodes", response_model=List[str])
async def get_available_nodes():
    """Get list of available nodes for running workloads"""
    try:
        nodes = await k8s_service.get_available_nodes()
        return nodes
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/inject")
async def inject_anomaly(request: AnomalyRequest):
    """Inject an anomaly into the OceanBase cluster"""
    try:
        # Use target_node if available, otherwise fall back to node for backward compatibility
        target_node = request.target_node if request.target_node is not None else request.node
        
        # Log the request details
        logger.info(f"Injecting anomaly: {request.type} on node {target_node}")
        
        # Clear all caches to ensure fresh data
        await clear_caches()
        
        # Start collecting training data first if requested
        # This ensures data collection is ready before the anomaly is injected
        if request.collect_training_data:
            try:
                await training_service.start_collection(
                    request.type, 
                    target_node
                )
                logger.info(f"Started collecting training data for {request.type} anomaly on {target_node}")
            except Exception as e:
                logger.error(f"Failed to start data collection: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to start data collection: {str(e)}"
                )
        
        # Inject the anomaly
        try:
            # Use retry_with_backoff to handle potential race conditions
            await retry_with_backoff(
                lambda: k8s_service.apply_chaos_experiment(
                    request.type,
                    target_node=target_node
                )
            )
            logger.info(f"Successfully injected {request.type} anomaly on node {target_node}")
            
            # Double check that the anomaly is tracked in active_anomalies
            logger.info(f"Active anomalies after injection: {list(k8s_service.active_anomalies.keys())}")
            
            # Force invalidate all caches again after injection
            await clear_caches()
            
        except Exception as e:
            # If anomaly injection fails and we started collection, stop it
            if request.collect_training_data:
                try:
                    training_service.stop_collection()
                except Exception as stop_error:
                    logger.error(f"Failed to stop collection after failed injection: {str(stop_error)}")
            
            logger.error(f"Failed to inject anomaly: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to inject anomaly: {str(e)}"
            )
            
        return {
            "status": "success", 
            "message": f"Injected {request.type} anomaly into node {target_node}"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in inject_anomaly: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clear")
async def clear_anomaly(request: AnomalyRequest):
    """Clear an anomaly from the OceanBase cluster"""
    try:
        # Use target_node if available, otherwise fall back to node for backward compatibility
        target_node = request.target_node if request.target_node is not None else request.node
        
        # Log the request details
        logger.info(f"Clearing anomaly - type: {request.type}, experiment: {request.experiment_name}")
        
        # Clear all caches first
        await clear_caches()
        
        # First priority: Use experiment_name if provided (precise deletion by ID)
        if request.experiment_name:
            # Find the anomaly type if not provided
            if not request.type:
                active_anomalies = await k8s_service.get_active_anomalies()
                target_anomaly = next((a for a in active_anomalies if a.get('name') == request.experiment_name), None)
                
                if target_anomaly:
                    request.type = target_anomaly.get('type')
                    logger.info(f"Found anomaly type {request.type} for experiment {request.experiment_name}")
                else:
                    logger.warning(f"Experiment {request.experiment_name} not found in active anomalies")
            
            if not request.type:
                return {
                    "status": "warning",
                    "message": f"Anomaly with ID {request.experiment_name} not found or missing type information"
                }
            
            # Delete the specific experiment by ID
            logger.info(f"Deleting specific experiment: {request.experiment_name}")
            deleted_experiments = await retry_with_backoff(
                lambda: k8s_service.delete_chaos_experiment(request.type, request.experiment_name)
            )
        else:
            # Second priority: Delete by type (will delete all experiments of this type)
            if not request.type:
                return {
                    "status": "error",
                    "message": "Either type or experiment_name must be provided"
                }
            
            logger.info(f"Deleting all experiments of type: {request.type}")
            deleted_experiments = await retry_with_backoff(
                lambda: k8s_service.delete_chaos_experiment(request.type, None)
            )
        
        # Log the deleted experiments
        logger.info(f"Deleted experiments: {deleted_experiments}")
        
        # Stop collecting training data if it was being collected
        if request.collect_training_data:
            training_service.stop_anomaly_collection(save_post_data=request.save_post_data)
            logger.info(f"Stopped collecting training data for {request.type} anomaly on {target_node}")
        
        # Clear caches again to ensure latest state
        await clear_caches()
        
        # Get current active anomalies to verify cleanup
        try:
            current_anomalies = await k8s_service.get_active_anomalies()
            logger.info(f"Active anomalies after clear: {[a.get('name', '') for a in current_anomalies]}")
            
            # If we were deleting by type, check if any anomalies of that type remain
            if not request.experiment_name and request.type:
                remaining = [a for a in current_anomalies if a.get('type') == request.type]
                if remaining:
                    logger.warning(f"Still have {len(remaining)} anomalies of type {request.type} after deletion")
                    
                    # Try one more time to delete them specifically
                    for anomaly in remaining:
                        try:
                            name = anomaly.get('name')
                            if name:
                                logger.info(f"Attempting to delete remaining anomaly: {name}")
                                await k8s_service.delete_chaos_experiment(request.type, name)
                        except Exception as e:
                            logger.error(f"Failed to delete remaining anomaly: {str(e)}")
        except Exception as e:
            logger.error(f"Error checking remaining anomalies: {str(e)}")
            
        return {
            "status": "success", 
            "message": f"Cleared anomaly successfully",
            "deleted": deleted_experiments
        }
    except Exception as e:
        logger.error(f"Failed to clear anomaly: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/active", response_model=ActiveAnomalyResponse)
@cache(expire=3)
async def get_active_anomalies():
    """Get list of active anomalies in the OceanBase cluster"""
    try:
        logger.info("Requesting active anomalies...")
        try:
            k8s_service.invalidate_cache()
            k8s_service._last_cache_update = 0
            active_anomalies = await k8s_service.get_active_anomalies()
            logger.info(f"Retrieved {len(active_anomalies)} active anomalies from K8s service")
        except AttributeError as e:
            logger.warning(f"AttributeError in get_active_anomalies: {str(e)}, using fallback...")
            active_anomalies = list(k8s_service.active_anomalies.values())
            logger.info(f"Retrieved {len(active_anomalies)} active anomalies from fallback")
        except Exception as e:
            logger.error(f"Exception in get_active_anomalies: {str(e)}, using fallback...")
            active_anomalies = list(k8s_service.active_anomalies.values())
            logger.info(f"Retrieved {len(active_anomalies)} active anomalies from fallback")
        logger.info(f"Active anomalies to return: {json.dumps(active_anomalies)}")

        response = jsonable_encoder({"anomalies": active_anomalies})
        return JSONResponse(
            content=response,
            headers={
                "Access-Control-Allow-Origin": "http://10.101.168.97:3000",
                "Access-Control-Allow-Credentials": "true",
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    except Exception as e:
        logger.error(f"Failed to get active anomalies: {str(e)}")
        return JSONResponse(
            content={"anomalies": []},
            headers={
                "Access-Control-Allow-Origin": "http://10.101.168.97:3000",
                "Access-Control-Allow-Credentials": "true",
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )

@router.get("/ranks", response_model=List[AnomalyRankResponse])
@cache(expire=5)
async def get_anomaly_ranks(response: Response):
    """Get ranked list of potential anomalies using time series metrics"""
    try:
        # Get detailed metrics with time series data
        metrics = metrics_service.get_detailed_metrics()
        if not metrics:
            return JSONResponse(
                content=[],
                headers={
                    "Access-Control-Allow-Origin": "http://10.101.168.97:3000",
                    "Access-Control-Allow-Credentials": "true"
                }
            )
            
        # Pass the detailed metrics directly to diagnosis service
        # The diagnosis service's _process_metrics method will handle the time window processing
        ranks = diagnosis_service.analyze_metrics(metrics)
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

@router.get("/compound")
@cache(expire=5)
async def get_compound_anomalies():
    """Get comprehensive anomaly diagnosis including compound anomalies"""
    try:
        # Get detailed metrics with time series data
        metrics = metrics_service.get_detailed_metrics()
        if not metrics:
            return JSONResponse(
                content={
                    "anomalies": [],
                    "compound_anomalies": {},
                    "propagation_graph": {}
                },
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Credentials": "true"
                }
            )
            
        # Get full diagnosis results including compound anomalies
        diagnosis_result = diagnosis_service.diagnose(metrics)
        
        # Check if the result contains compound anomalies
        has_compound_anomalies = False
        compound_anomalies = diagnosis_result.get("compound_anomalies", {})
        if compound_anomalies and len(compound_anomalies) > 0:
            has_compound_anomalies = True
            
            # For each node with compound anomalies, sort by overall impact
            for node, anomalies in compound_anomalies.items():
                # Calculate combined impact score
                combined_score = sum(a['score'] for a in anomalies)
                
                # Add combined details to each compound anomaly
                for anomaly in anomalies:
                    # Calculate contribution percentage
                    anomaly['contribution'] = (anomaly['score'] / combined_score) * 100.0 if combined_score > 0 else 0.0
                
                # Add total score to facilitate sorting
                compound_anomalies[node] = {
                    'anomalies': anomalies,
                    'total_score': combined_score,
                    'anomaly_count': len(anomalies)
                }
            
            # Sort compound anomalies by severity
            sorted_compound = {
                k: v for k, v in sorted(
                    compound_anomalies.items(),
                    key=lambda item: item[1]['total_score'],
                    reverse=True
                )
            }
            diagnosis_result["compound_anomalies"] = sorted_compound
        
        # Expand propagation graph with compound anomaly information
        if has_compound_anomalies and "propagation_graph" in diagnosis_result:
            propagation_graph = diagnosis_result["propagation_graph"]
            
            # Mark nodes with compound anomalies in the graph
            for node in propagation_graph:
                if node in compound_anomalies:
                    propagation_graph[node]["has_compound_anomaly"] = True
                    propagation_graph[node]["compound_score"] = compound_anomalies[node]["total_score"]
                    propagation_graph[node]["anomaly_count"] = compound_anomalies[node]["anomaly_count"]
                else:
                    propagation_graph[node]["has_compound_anomaly"] = False
            
            # Update the graph with compound anomaly connections
            for src_node in propagation_graph:
                for dst_node in propagation_graph.get(src_node, {}):
                    if isinstance(propagation_graph[src_node][dst_node], (int, float)):
                        # This is a correlation value, convert to object
                        corr_value = propagation_graph[src_node][dst_node]
                        propagation_graph[src_node][dst_node] = {
                            "correlation": corr_value,
                            "compound_path": (
                                src_node in compound_anomalies and 
                                dst_node in compound_anomalies
                            )
                        }
        
        # Create a more simplified response structure
        response_data = {
            "anomalies": diagnosis_result.get("anomalies", []),
            "compound_anomalies": diagnosis_result.get("compound_anomalies", {}),
            "propagation_graph": diagnosis_result.get("propagation_graph", {}),
            "node_names": diagnosis_result.get("node_names", []),
            "has_compound_anomalies": has_compound_anomalies,
            "timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(
            content=response_data,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Credentials": "true"
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get compound anomalies: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Credentials": "true"
            }
        )

@router.post("/train")
async def train_model():
    """Train the anomaly detection model."""
    try:
        # Redirect to the new training endpoint
        return RedirectResponse(url="/api/training/train", status_code=307)
    except Exception as e:
        logger.error(f"Failed to redirect to training endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/normal/start")
async def start_normal_collection():
    """Redirect to the new training endpoint for normal collection"""
    try:
        return RedirectResponse(url="/api/training/normal/start", status_code=307)
    except Exception as e:
        logger.error(f"Failed to redirect to normal collection endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/normal/stop")
async def stop_normal_collection():
    """Redirect to the new training endpoint for stopping normal collection"""
    try:
        return RedirectResponse(url="/api/training/normal/stop", status_code=307)
    except Exception as e:
        logger.error(f"Failed to redirect to stop normal collection endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/collection-status")
async def get_collection_status():
    """Redirect to the new training endpoint for collection status"""
    try:
        return RedirectResponse(url="/api/training/collection-status", status_code=307)
    except Exception as e:
        logger.error(f"Failed to redirect to collection status endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/training/stats")
async def get_training_stats():
    """Redirect to the new training endpoint for stats"""
    try:
        return RedirectResponse(url="/api/training/stats", status_code=307)
    except Exception as e:
        logger.error(f"Failed to redirect to training stats endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/training/auto_balance")
async def auto_balance_dataset():
    """Redirect to the new training endpoint for auto-balancing"""
    try:
        return RedirectResponse(url="/api/training/auto_balance", status_code=307)
    except Exception as e:
        logger.error(f"Failed to redirect to auto balance endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/training/collect")
async def start_training_collection(request: AnomalyRequest):
    """Redirect to the new training endpoint for collection"""
    try:
        return RedirectResponse(url="/api/training/collect", status_code=307)
    except Exception as e:
        logger.error(f"Failed to redirect to training collection endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/training/stop")
async def stop_training_collection():
    """Redirect to the new training endpoint for stopping collection"""
    try:
        return RedirectResponse(url="/api/training/stop", status_code=307)
    except Exception as e:
        logger.error(f"Failed to redirect to stop training endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stream")
async def anomaly_stream():
    """Stream anomaly updates using Server-Sent Events"""
    async def event_generator():
        last_data = None
        error_count = 0
        max_errors = 10
        update_interval = 2  # Seconds between updates
        
        logger.info("Starting anomaly stream event generator")
        
        while True:
            try:
                # Don't clear cache on every iteration to reduce load
                # Only clear it occasionally or when we've hit errors
                if error_count > 0:
                    k8s_service.invalidate_cache()
                    k8s_service._last_cache_update = 0
                    logger.debug("Cleared cache due to previous errors")
                
                # Get current anomalies by calling the API method
                try:
                    logger.debug("Fetching active anomalies for stream")
                    current_data = await k8s_service.get_active_anomalies()
                    logger.debug(f"Stream fetched {len(current_data)} active anomalies")
                    error_count = 0  # Reset error count on success
                except AttributeError as e:
                    logger.warning(f"AttributeError in anomaly stream: {str(e)}, trying to recover...")
                    # If the method is missing, use fallback and wait
                    current_data = list(k8s_service.active_anomalies.values())
                    logger.debug(f"Stream fallback: {len(current_data)} active anomalies from direct cache")
                    error_count += 1
                except Exception as e:
                    logger.error(f"Error getting active anomalies for stream: {str(e)}")
                    # Use fallback to direct dictionary access if the method fails
                    current_data = list(k8s_service.active_anomalies.values())
                    logger.debug(f"Stream error fallback: {len(current_data)} active anomalies from direct cache")
                    error_count += 1
                    
                    # If too many errors, reset cache completely but less frequently
                    if error_count > max_errors:
                        logger.warning("Too many errors in stream, attempting to clear caches")
                        try:
                            await clear_caches()
                            error_count = max_errors // 2  # Reduce error count but not to zero
                        except Exception as clear_error:
                            logger.error(f"Failed to clear caches in stream: {str(clear_error)}")
                
                # Only send if data changed to reduce unnecessary updates
                if current_data != last_data:
                    logger.info(f"Stream data changed, sending update with {len(current_data)} anomalies")
                    last_data = current_data
                    yield {
                        "event": "anomaly_update",
                        "data": json.dumps(current_data),
                        "retry": 5000  # Increased retry time to 5s if connection drops
                    }
                
                # Use a longer wait interval to reduce processing load
                await asyncio.sleep(update_interval)
            except Exception as e:
                logger.error(f"Critical error in anomaly stream generator: {str(e)}")
                error_count += 1
                # Add small delay on error
                await asyncio.sleep(2)
                continue

    return EventSourceResponse(
        event_generator(),
        headers={
            "Access-Control-Allow-Origin": "http://10.101.168.97:3000",
            "Access-Control-Allow-Credentials": "true",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )

@router.post("/stop-all")
async def stop_all_anomalies():
    """Stop all active anomalies in the cluster"""
    try:
        logger.info("Stopping all active anomalies")
        
        # Clear all caches first
        await clear_caches()
        
        # Get active anomalies before deletion for verification
        active_anomalies = await k8s_service.get_active_anomalies()
        active_count = len(active_anomalies)
        logger.info(f"Found {active_count} active anomalies before deletion")
        
        # Delete all active experiments
        deleted_experiments = await k8s_service.delete_all_chaos_experiments()
        logger.info(f"Deleted {len(deleted_experiments) if deleted_experiments else 0} experiments")
        
        # Clear caches again to ensure latest state
        await clear_caches()
        
        # Verify deletion
        remaining_anomalies = await k8s_service.get_active_anomalies()
        if remaining_anomalies:
            logger.warning(f"Still have {len(remaining_anomalies)} anomalies after deletion attempt")
            
            # Try one more time for each remaining anomaly
            for anomaly in remaining_anomalies:
                try:
                    anomaly_type = anomaly.get('type')
                    name = anomaly.get('name')
                    if anomaly_type and name:
                        logger.info(f"Attempting to delete remaining anomaly: {name}")
                        await k8s_service.delete_chaos_experiment(anomaly_type, name)
                except Exception as e:
                    logger.error(f"Failed to delete remaining anomaly: {str(e)}")
        
        return {
            "status": "success", 
            "message": f"Stopped all anomalies ({active_count} total)",
            "deleted_count": active_count,
            "deleted": deleted_experiments
        }
    except Exception as e:
        logger.error(f"Failed to stop all anomalies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 