from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional

class AnomalyRequest(BaseModel):
    """Request model for starting an anomaly"""
    type: str

class MetricsResponse(BaseModel):
    """Response model for system metrics"""
    timestamp: datetime
    cpu: float
    memory: float
    network: float

class AnomalyRankResponse(BaseModel):
    """Response model for anomaly ranks"""
    timestamp: datetime
    rank: float 