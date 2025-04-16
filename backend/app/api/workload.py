import logging
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, List, Optional
from ..services.workload_service import workload_service
from ..schemas.workload import WorkloadRequest, WorkloadStatus, WorkloadResponse, WorkloadInfo
from datetime import datetime
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()

class WorkloadRunInfo(BaseModel):
    run_id: str
    pid: int
    type: str
    start_time: Optional[datetime] = None

@router.post("/prepare")
async def prepare_database(request: WorkloadRequest):
    """
    Endpoint to trigger database preparation for a given workload type.
    This is often needed before running benchmarks like TPC-C or TPC-H.
    """
    try:
        logger.info(f"Received request to prepare database for workload: {request.workload_type}")
        # Assuming workload_service has a method to handle preparation
        # The actual implementation will depend on the workload (e.g., running a specific script)
        result = await workload_service.prepare_workload_environment(
            workload_type=request.workload_type,
            config=request.config
        )
        if not result or not result.get("success"):
             raise HTTPException(status_code=500, detail=f"Failed to prepare database for {request.workload_type}: {result.get('message', 'Unknown error')}")

        logger.info(f"Database preparation for {request.workload_type} completed successfully.")
        return {"status": "success", "message": f"Database prepared successfully for {request.workload_type}."}
    except Exception as e:
        logger.exception(f"Error preparing database for workload {request.workload_type}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during database preparation: {str(e)}")

@router.get("/active_runs", response_model=List[WorkloadRunInfo])
async def get_active_workload_runs():
    """Get a list of currently active workload process runs with basic info (PID, type)."""
    try:
        active_runs = await workload_service.get_active_runs() # Assumes service returns list of dicts/objects matching WorkloadRunInfo
        return active_runs
    except Exception as e:
        logger.error(f"Error retrieving active workload runs: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve active workload runs")

@router.get("/active", response_model=List[WorkloadInfo])
async def get_active_workloads():
    """Get details of currently active workloads (potentially linked to tasks)."""
    # This seems like it should use TaskService if workload == task?
    # If it's meant to be independent processes, WorkloadService is correct.
    try:
        active_workloads = await workload_service.get_active_workloads_info() # Assuming this method exists
        return active_workloads
    except Exception as e:
        logger.error(f"Error retrieving active workloads: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve active workloads")

@router.post("/{workload_id}/stop", response_model=WorkloadResponse)
async def stop_workload(workload_id: str):
    """
    Stop a specific running workload process by its ID.
    Note: This stops the workload process directly. If it's part of a Task,
    use the /api/tasks/{task_id}/stop endpoint instead for proper cleanup.
    """
    try:
        logger.info(f"Received request to stop workload: {workload_id}")
        response = await workload_service.stop_workload(run_id=workload_id)
        if not response or not response.get("success"):
            detail = response.get("message", f"Workload {workload_id} not found or already stopped.")
            status_code = 404 if "not found" in detail.lower() else 500
            raise HTTPException(status_code=status_code, detail=detail)

        logger.info(f"Workload {workload_id} stopped successfully.")
        # Return a standard WorkloadResponse format
        return WorkloadResponse(
            workload_id=workload_id,
            status=WorkloadStatus.STOPPED, # Assuming stop was successful
            message=f"Workload {workload_id} stopped successfully."
        )
    except HTTPException:
        raise # Re-raise HTTP exceptions
    except Exception as e:
        logger.exception(f"Error stopping workload {workload_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error stopping workload: {str(e)}")

@router.get("/{workload_id}/status", response_model=WorkloadResponse)
async def get_workload_status(workload_id: str):
    """Get the status of a specific workload process."""
    # Again, consider if this should come from TaskService via task ID?
    try:
        status_info = await workload_service.get_workload_status(run_id=workload_id) # Assuming this method exists
        if not status_info:
            raise HTTPException(status_code=404, detail=f"Workload with ID '{workload_id}' not found.")

        # Adapt the result from the service to the WorkloadResponse model
        return WorkloadResponse(
             workload_id=workload_id,
             status=status_info.get("status", WorkloadStatus.UNKNOWN), # Map service status to schema enum
             message=status_info.get("message", ""),
             details=status_info.get("details")
         )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving status for workload {workload_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve workload status")

# Example placeholder for a dependency - adapt if needed
# async def get_workload_service_dependency():
#    return workload_service # Return the singleton instance

# @router.get("/some_other_endpoint")
# async def some_other_endpoint(ws: WorkloadService = Depends(get_workload_service_dependency)):
#    pass 