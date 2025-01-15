from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List

class AnomalyRequest(BaseModel):
    """Request to start an anomaly experiment"""
    type: str
    duration: Optional[int] = None  # Duration in seconds

class MetricsResponse(BaseModel):
    """Response containing system metrics"""
    timestamp: datetime
    cpu: float
    memory: float
    network: float
    disk_io_bytes: Optional[float] = None
    disk_iops: Optional[float] = None
    active_sessions: Optional[float] = None
    sql_response_time: Optional[float] = None
    cache_hit_ratio: Optional[float] = None

class AnomalyRankResponse(BaseModel):
    """Response containing anomaly ranks"""
    timestamp: datetime
    node: str  # Node where anomaly was detected
    type: str  # Type of anomaly (cpu, io, buffer, net, load)
    score: float  # Anomaly score between 0 and 1

class ActiveAnomalyResponse(BaseModel):
    """Response containing information about an active anomaly"""
    start_time: str
    status: str
    type: str
    target: str 