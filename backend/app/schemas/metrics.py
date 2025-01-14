from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional, Dict, Any

class MetricBase(BaseModel):
    timestamp: datetime

class DatabaseMetrics(MetricBase):
    """Response model for database metrics"""
    qps: float
    tps: float
    active_sessions: float
    sql_response_time: float
    disk_io_bytes: float
    disk_iops: float
    memory_usage: float
    cache_hit_ratio: float
    slow_queries: float
    deadlocks: float
    replication_lag: float
    connection_count: float

class TenantMetrics(MetricBase):
    """Response model for tenant metrics"""
    tenant: str
    cpu_percent: float
    memory_used: float
    disk_used: float
    iops: float
    session_count: float
    active_session_count: float

class SystemMetric(MetricBase):
    cpu: float
    memory: float
    network: float
    disk_io_bytes: float
    disk_iops: float
    active_sessions: int
    sql_response_time: float
    cache_hit_ratio: float

class DatabaseMetricsResponse(BaseModel):
    """Response model for database metrics list"""
    metrics: List[DatabaseMetrics]

class TenantMetricsResponse(BaseModel):
    """Response model for tenant metrics list"""
    metrics: List[TenantMetrics]

class SystemMetricsResponse(BaseModel):
    metrics: List[SystemMetric]