from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime

class WorkloadType(str, Enum):
    SYSBENCH = "sysbench"
    TPCC = "tpcc"
    TPCH = "tpch"

class TaskStatus(str, Enum):
    PENDING = "pending"        # Task created, orchestration not started/in progress
    RUNNING = "running"        # Workload running, anomalies (if any) applied
    STOPPING = "stopping"      # Stop requested, cleanup in progress
    STOPPED = "stopped"        # Task completed successfully and cleaned up
    ERROR = "error"          # Task failed during orchestration or execution

class AnomalyConfig(BaseModel):
    """Configuration for a single anomaly within a task."""
    type: str = Field(..., description="The type of anomaly to apply (e.g., 'cpu_stress', 'network_bottleneck')")
    target: Optional[str] = Field(None, description="Target node/pod for the anomaly (required for node-specific anomalies)")
    severity: Optional[str] = Field("medium", description="Severity level (e.g., 'low', 'medium', 'high') affecting anomaly parameters")

class TaskBase(BaseModel):
    """Base model with common fields for task creation and representation."""
    name: str = Field(..., description="User-defined name for the task")
    workload_type: WorkloadType = Field(..., description="Type of workload to run")
    workload_config: Dict[str, Any] = Field(..., description="Configuration parameters for the workload (e.g., threads, tables, duration)")
    anomalies: Optional[List[AnomalyConfig]] = Field(None, description="List of anomalies to apply during the task execution")

class TaskCreate(TaskBase):
    """Schema used for creating a new task via the API."""
    pass # Inherits all fields from TaskBase

class Task(TaskBase):
    """Schema representing a task's full state, including runtime info."""
    id: str = Field(..., description="Unique identifier for the task")
    status: TaskStatus = Field(..., description="Current status of the task")
    start_time: datetime = Field(..., description="Timestamp when the task was created (UTC)")
    end_time: Optional[datetime] = Field(None, description="Timestamp when the task finished (stopped or errored, UTC)")
    workload_run_id: Optional[str] = Field(None, description="Unique identifier for the specific workload process run associated with this task")
    anomaly_ids: Optional[List[str]] = Field(None, description="List of unique IDs for the anomalies applied by this task")
    error_message: Optional[str] = Field(None, description="Details if the task ended in an error state")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
                    "name": "High CPU Sysbench Run",
                    "status": "running",
                    "start_time": "2023-10-27T10:00:00Z",
                    "end_time": None,
                    "workload_type": "sysbench",
                    "workload_config": {"threads": 16, "tables": 10, "time": 0, "reportInterval": 10},
                    "workload_run_id": "sysbench-a1b2c3d4",
                    "anomalies": [{"type": "cpu_stress", "target": "obcluster-0-0", "severity": "high"}],
                    "anomaly_ids": ["f0e9d8c7-b6a5-4321-fedc-ba9876543210"],
                    "error_message": None,
                },
                {
                    "id": "b2c3d4e5-f6a7-8901-2345-67890abcdef0",
                    "name": "TPCC Run - Stopped",
                    "status": "stopped",
                    "start_time": "2023-10-27T11:00:00Z",
                    "end_time": "2023-10-27T11:15:30Z",
                    "workload_type": "tpcc",
                    "workload_config": {"warehouses": 10, "threads": 8},
                    "workload_run_id": "tpcc-b2c3d4e5",
                    "anomalies": None,
                    "anomaly_ids": None,
                    "error_message": None,
                }
            ]
        }
    }

# Potential future additions:
# class TaskUpdate(BaseModel): # For modifying tasks? Maybe not applicable.
#    pass 