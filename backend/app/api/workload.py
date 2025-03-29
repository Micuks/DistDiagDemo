import logging
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, List, Optional
from ..services.workload_service import workload_service, WorkloadType
from ..services.k8s_service import k8s_service
from ..schemas.workload import WorkloadResponse, WorkloadStatus, WorkloadConfig, WorkloadRequest, WorkloadInfo, Task, CreateTaskRequest

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/prepare")
async def prepare_database(request: WorkloadRequest):
    """Prepare the database for the specified workload type"""
    logger.info(f"Received request to prepare database for workload type: {request.type}")
    try:
        success = False
        if request.type == WorkloadType.SYSBENCH:
            success = workload_service.prepare_database(request.type.value)
        elif request.type == WorkloadType.TPCC:
            success = workload_service.prepare_tpcc()
        elif request.type == WorkloadType.TPCH:
            success = workload_service.prepare_tpch()
        else:
            logger.error(f"Unsupported workload type: {request.type}")
            raise HTTPException(status_code=400, detail=f"Unsupported workload type: {request.type}")

        if success:
            logger.info("Database preparation completed successfully")
            return {"status": "success", "message": "Database prepared successfully"}
        else:
            logger.error("Database preparation failed")
            raise HTTPException(status_code=500, detail="Failed to prepare database")
    except Exception as e:
        logger.exception("Error during database preparation")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/start", response_model=WorkloadResponse)
async def start_workload(config: WorkloadRequest):
    """Start a new workload with custom configuration using obproxy connection"""
    try:
        logger.info(f"Received start workload request: type={config.type}, threads={config.threads}, task_name={config.task_name}")
        logger.info(f"Received options: {config.options.dict() if config.options else {}}")
        
        if config.type == 'tpcc':
            type = WorkloadType.TPCC
        elif config.type == 'sysbench':
            type = WorkloadType.SYSBENCH
        else:
            type = WorkloadType.UNKNOWN
            
        success = workload_service.start_workload(
            workload_type=type,
            num_threads=config.threads,
            options=config.options.dict() if config.options else {},
            task_name=config.task_name
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to start workload")
            
        return WorkloadResponse(
            workload_id=f"{config.type.value}_{config.threads}",
            status=WorkloadStatus.RUNNING,
            message="Workload started successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/create-task", response_model=Task)
async def create_task(request: CreateTaskRequest):
    """Create a task with anomalies for an existing workload"""
    try:
        logger.info(f"Creating task for existing workload: workload_id={request.workload_id}, task_name={request.task_name}")
        
        # Determine workload type from string to enum
        workload_type = WorkloadType.UNKNOWN
        if request.type == 'tpcc':
            workload_type = WorkloadType.TPCC
        elif request.type == 'sysbench':
            workload_type = WorkloadType.SYSBENCH
        elif request.type == 'tpch':
            workload_type = WorkloadType.TPCH
            
        # Get workload details to create appropriate workload_config
        active_workloads = workload_service.get_active_workloads()
        target_workload = next((w for w in active_workloads if w['id'] == request.workload_id), None)
        
        if not target_workload:
            raise HTTPException(status_code=404, detail=f"Workload with ID {request.workload_id} not found")
            
        # Create a workload config based on the existing workload
        workload_config = {
            "workload_type": workload_type,
            "num_threads": target_workload.get('threads', 1)
        }
        
        # Create the task
        task = workload_service.create_task(
            name=request.task_name,
            workload_type=workload_type,
            workload_config=workload_config,
            anomalies=request.anomalies
        )
        
        if not task:
            raise HTTPException(status_code=500, detail="Failed to create task")
            
        return task
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error creating task: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stop-all")
async def stop_all_workloads():
    """Stop all active workloads and anomalies"""
    logger.info("Received request to stop all workloads")
    try:
        # First try to stop all workloads
        success = workload_service.stop_all_workloads()
        
        # Also stop all active anomalies
        try:
            logger.info("Stopping all active anomalies")
            await k8s_service.delete_all_chaos_experiments()
            logger.info("All anomalies stopped")
        except Exception as e:
            logger.warning(f"Error stopping all anomalies: {str(e)}")
            # Continue anyway to check workload status
        
        # Verify if workloads are actually stopped by checking active workloads
        active_workloads = workload_service.get_active_workloads()
        
        if not active_workloads:
            # No active workloads, so operation was successful regardless of the service's return value
            logger.info("All workloads verified as stopped")
            return {"status": "success", "message": "All workloads and anomalies stopped successfully"}
        else:
            # Some workloads are still active - this is an actual failure
            workload_ids = [w['id'] for w in active_workloads]
            logger.error(f"Failed to stop all workloads, remaining: {workload_ids}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to stop all workloads. Remaining: {', '.join(workload_ids)}"
            )
    except Exception as e:
        logger.exception("Error stopping all workloads")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/active", response_model=List[WorkloadInfo])
async def get_active_workloads():
    """Get all active workloads"""
    try:
        workloads = workload_service.get_active_workloads()
        return workloads
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{workload_id}/stop", response_model=WorkloadResponse)
async def stop_workload(workload_id: str):
    """Stop a running workload"""
    try:
        success = workload_service.stop_workload(workload_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Workload not found or already stopped")
            
        return WorkloadResponse(
            workload_id=workload_id,
            status=WorkloadStatus.STOPPED,
            message="Workload stopped successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{workload_id}/status", response_model=WorkloadResponse)
async def get_workload_status(workload_id: str):
    """Get the status of a workload"""
    try:
        status = workload_service.get_workload_status(workload_id)
        
        if status["status"] == "not_found":
            raise HTTPException(status_code=404, detail="Workload not found")
            
        return WorkloadResponse(
            workload_id=workload_id,
            status=WorkloadStatus(status["status"]),
            message="Workload status retrieved successfully",
            output=status["output"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tasks", response_model=List[Task])
async def get_all_tasks():
    """Get all tasks"""
    try:
        tasks = workload_service.get_all_tasks()
        return tasks
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tasks/active", response_model=List[Task])
async def get_active_tasks():
    """Get all active tasks"""
    try:
        tasks = workload_service.get_active_tasks()
        
        # Log task count for debugging
        logger.info(f"Retrieved {len(tasks)} active tasks")
        if tasks:
            for task in tasks:
                logger.debug(f"Active task: {task.id}, status: {task.status}, name: {task.name}")
                
        return tasks
    except Exception as e:
        logger.error(f"Error getting active tasks: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tasks/cleanup", response_model=dict)
async def cleanup_tasks():
    """Remove stopped tasks older than a certain threshold"""
    try:
        removed_count = workload_service.cleanup_old_tasks()
        logger.info(f"Cleaned up {removed_count} old tasks")
        return {
            "status": "success", 
            "message": f"Cleaned up {removed_count} old tasks",
            "removed_count": removed_count
        }
    except Exception as e:
        logger.error(f"Error cleaning up tasks: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tasks/{task_id}", response_model=Task)
async def get_task(task_id: str):
    """Get a specific task by ID"""
    try:
        task = workload_service.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        return task
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tasks/{task_id}/stop", response_model=Task)
async def stop_task(task_id: str):
    """Stop a running task and all its associated workloads and anomalies"""
    try:
        logger.info(f"Received request to stop task: {task_id}")
        
        # First get the task
        task = workload_service.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
            
        # If task is already stopped, just return it
        if task.status != WorkloadStatus.RUNNING:
            return task
            
        # Try to stop associated workload if it exists
        if task.workload_id:
            try:
                workload_service.stop_workload(task.workload_id)
            except Exception as e:
                logger.warning(f"Failed to stop workload {task.workload_id}: {str(e)}")
                # Continue anyway to update task status

        # Stop all associated anomalies
        if task.anomalies:
            logger.info(f"Stopping {len(task.anomalies)} anomalies associated with task {task_id}")
            for anomaly in task.anomalies:
                try:
                    anomaly_type = anomaly.get('type')
                    experiment_name = anomaly.get('name')
                    if anomaly_type:
                        logger.info(f"Stopping anomaly: {anomaly_type} (name: {experiment_name})")
                        await k8s_service.delete_chaos_experiment(anomaly_type, experiment_name)
                except Exception as e:
                    logger.warning(f"Failed to stop anomaly {anomaly.get('type')}: {str(e)}")
                    # Continue anyway to update task status
        
        # Update task status
        updated_task = workload_service.update_task_status(task_id, WorkloadStatus.STOPPED)
        
        if not updated_task:
            raise HTTPException(status_code=500, detail="Failed to update task status")
            
        return updated_task
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error stopping task: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 