from enum import Enum
from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime

class WorkloadType(str, Enum):
    SYSBENCH = "sysbench"
    TPCC = "tpcc"
    TPCH = "tpch"
    UNKNOWN = "unknown"

class WorkloadStatus(str, Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    NOT_FOUND = "not_found"

class WorkloadMetrics(BaseModel):
    cpu_usage: float
    memory_usage: float
    disk_usage: float

class WorkloadOptions(BaseModel):
    # Node selection
    target_node: Optional[str] = None
    
    # Common options
    prepare_db: Optional[bool] = False
    reportInterval: Optional[int] = 10
    
    # Sysbench options
    tables: Optional[int] = 10
    tableSize: Optional[int] = 100000
    randType: Optional[str] = "uniform"
    
    # TPCC options
    warehouses: Optional[int] = 10
    warmupTime: Optional[int] = 10
    runningTime: Optional[int] = 60

class WorkloadConfig(BaseModel):
    workload_type: WorkloadType
    num_threads: int = 4
    options: Optional[WorkloadOptions] = None

class WorkloadRequest(BaseModel):
    type: WorkloadType
    threads: Optional[int] = 1
    options: Optional[WorkloadOptions] = None
    task_name: Optional[str] = None

class WorkloadInfo(BaseModel):
    id: str
    type: str
    threads: int
    pid: int
    start_time: Optional[datetime] = None
    status: WorkloadStatus

class WorkloadResponse(BaseModel):
    workload_id: str
    status: WorkloadStatus
    message: str
    output: Optional[List[str]] = None

class Task(BaseModel):
    id: str
    name: str
    workload_id: str
    workload_type: WorkloadType
    workload_config: WorkloadConfig
    anomalies: List[Dict] = []
    start_time: datetime
    status: WorkloadStatus
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None

class CreateTaskRequest(BaseModel):
    type: str
    task_name: str
    workload_id: str
    anomalies: List[Dict] = [] 