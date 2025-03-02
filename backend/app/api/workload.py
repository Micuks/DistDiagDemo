import logging
from fastapi import APIRouter, HTTPException
from ..services.workload_service import WorkloadService
from ..schemas.workload import WorkloadRequest, WorkloadInfo, WorkloadType

logger = logging.getLogger(__name__)
router = APIRouter()
workload_service = WorkloadService()

@router.post("/prepare")
async def prepare_database(request: WorkloadRequest):
    """Prepare the database for the specified workload type"""
    logger.info(f"Received request to prepare database for workload type: {request.type}")
    try:
        success = False
        if request.type == WorkloadType.SYSBENCH:
            success = workload_service.prepare_database()
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

@router.post("/start")
async def start_workload(request: WorkloadRequest):
    """Start a new workload"""
    logger.info(f"Received request to start workload: type={request.type}, threads={request.threads}")
    try:
        success = workload_service.start_workload(request.type, request.threads)
        if success:
            logger.info("Workload started successfully")
            return {"status": "success", "message": "Workload started successfully"}
        else:
            logger.error("Failed to start workload")
            raise HTTPException(status_code=500, detail="Failed to start workload")
    except Exception as e:
        logger.exception("Error starting workload")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stop/{workload_id}")
async def stop_workload(workload_id: str):
    """Stop a specific workload"""
    logger.info(f"Received request to stop workload: {workload_id}")
    try:
        success = workload_service.stop_workload(workload_id)
        if success:
            logger.info(f"Workload {workload_id} stopped successfully")
            return {"status": "success", "message": f"Workload {workload_id} stopped successfully"}
        else:
            logger.error(f"Failed to stop workload {workload_id}")
            raise HTTPException(status_code=404, detail=f"Workload {workload_id} not found or already stopped")
    except Exception as e:
        logger.exception(f"Error stopping workload {workload_id}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stop-all")
async def stop_all_workloads():
    """Stop all active workloads"""
    logger.info("Received request to stop all workloads")
    try:
        # First try to stop all workloads
        success = workload_service.stop_all_workloads()
        
        # Verify if workloads are actually stopped by checking active workloads
        active_workloads = workload_service.get_active_workloads()
        
        if not active_workloads:
            # No active workloads, so operation was successful regardless of the service's return value
            logger.info("All workloads verified as stopped")
            return {"status": "success", "message": "All workloads stopped successfully"}
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

@router.get("/active")
async def get_active_workloads() -> list[WorkloadInfo]:
    """Get list of active workloads"""
    logger.info("Received request to get active workloads")
    try:
        workloads = workload_service.get_active_workloads()
        logger.info(f"Found {len(workloads)} active workloads")
        return workloads
    except Exception as e:
        logger.exception("Error getting active workloads")
        raise HTTPException(status_code=500, detail=str(e)) 