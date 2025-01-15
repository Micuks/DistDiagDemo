from enum import Enum
from pydantic import BaseModel
from typing import Dict, Optional

class WorkloadType(str, Enum):
    SYSBENCH = "sysbench"
    TPCC = "tpcc"
    TPCH = "tpch"

class WorkloadMetrics(BaseModel):
    cpu_usage: float
    memory_usage: float
    disk_usage: float

class WorkloadInfo(BaseModel):
    id: str
    type: str
    threads: int
    pid: int
    metrics: WorkloadMetrics

class WorkloadRequest(BaseModel):
    type: WorkloadType
    threads: Optional[int] = 1 