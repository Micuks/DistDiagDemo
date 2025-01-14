from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from app.services.ob_metrics_service import OBMetricsService
from app.services.metrics_service import MetricsService
from app.schemas.metrics import DatabaseMetricsResponse, TenantMetricsResponse, SystemMetricsResponse
import logging

router = APIRouter()
logger = logging.getLogger(__name__)
metrics_service = OBMetricsService()
system_metrics_service = MetricsService()

@router.get("/database", response_model=DatabaseMetricsResponse)
async def get_database_metrics():
    """Get OceanBase database metrics"""
    try:
        metrics = await metrics_service.get_database_metrics()
        return DatabaseMetricsResponse(metrics=metrics)
    except Exception as e:
        logger.error(f"Error fetching database metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tenant", response_model=TenantMetricsResponse)
async def get_tenant_metrics(tenant_name: Optional[str] = Query(None, description="Filter metrics by tenant name")):
    """Get OceanBase tenant metrics"""
    try:
        metrics = await metrics_service.get_tenant_metrics(tenant_name)
        return TenantMetricsResponse(metrics=metrics)
    except Exception as e:
        logger.error(f"Error fetching tenant metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system", response_model=SystemMetricsResponse)
async def get_system_metrics():
    """Get system metrics"""
    try:
        metrics = await system_metrics_service.get_system_metrics()
        return SystemMetricsResponse(metrics=metrics)
    except Exception as e:
        logger.error(f"Error fetching system metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))