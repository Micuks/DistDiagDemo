from pydantic import BaseModel
from typing import Dict, List, Optional, Union
from enum import Enum

class AnomalyType(str, Enum):
    CPU_STRESS = "cpu_stress"
    IO_BOTTLENECK = "io_bottleneck"
    NETWORK_BOTTLENECK = "network_bottleneck"
    MEMORY_LEAK = "memory_leak"
    TOO_MANY_INDEXES = "too_many_indexes"

class AnomalyStatus(str, Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    NOT_FOUND = "not_found"

class AnomalyConfig(BaseModel):
    anomaly_type: AnomalyType
    target_node: Optional[Union[List[str], str]] = None
    severity: int = 5
    duration: Optional[int] = None

class AnomalyRequest(BaseModel):
    type: str
    target_node: Optional[Union[List[str], str]] = None
    node: Optional[Union[List[str], str]] = None  # For backwards compatibility
    severity: int = 5
    duration: Optional[int] = None
    collect_training_data: Optional[bool] = False
    save_post_data: Optional[bool] = True
    experiment_name: Optional[str] = None

class AnomalyInfo(BaseModel):
    id: str
    type: str
    target_node: str
    severity: int
    start_time: float
    elapsed_time: float

class AnomalyResponse(BaseModel):
    anomaly_id: str
    status: AnomalyStatus
    message: str
    output: Optional[List[str]] = None
    details: Optional[Dict] = None

class MetricsResponse(BaseModel):
    timestamp: float
    metrics: Dict[str, float]

class AnomalyRankResponse(BaseModel):
    ranks: List[Dict[str, float]]
    timestamp: float

class ActiveAnomalyResponse(BaseModel):
    anomalies: List[Dict] 