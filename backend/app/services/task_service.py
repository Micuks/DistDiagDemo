import os
import uuid
import logging
import json
import asyncio # Import asyncio for create_task
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Any

import redis
from dotenv import load_dotenv

# Load environment variables from .env file
# Adjust the path based on the final location of this file
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env'))

# Placeholder imports - these will be the services TaskService orchestrates
# Assume these have singleton instances or methods we can call
from .workload_service import workload_service, WorkloadType
from .k8s_service import k8s_service

# Import schemas from the dedicated module
from ..schemas.tasks import Task, TaskCreate, TaskStatus, AnomalyConfig, WorkloadType

# Import ConnectionManager for type hinting (if possible, adjust based on actual location)
# Assuming tasks.py is in ../api/
from ..api.tasks import ConnectionManager # Relative import

logger = logging.getLogger(__name__)

class TaskService:
    def __init__(self):
        self.redis_host = os.getenv('REDIS_HOST', 'localhost')
        self.redis_port = int(os.getenv('REDIS_PORT', 6379))
        self.redis_db = int(os.getenv('REDIS_DB_TASKS', 1)) # Use a separate DB for tasks?
        self.redis_client = None
        self._connect_redis()
        self.connection_manager: Optional[ConnectionManager] = None # Use string literal for type hint

        # Redis key prefixes/patterns
        self.task_key_prefix = "task:"
        self.active_tasks_set_key = "active_tasks_set"
        self.completed_tasks_set_key = "completed_tasks_set"

        logger.info(f"TaskService initialized. Attempting to connect to Redis at {self.redis_host}:{self.redis_port} DB {self.redis_db}")
        if not self.redis_client:
            logger.error("TaskService failed to connect to Redis during initialization.")
        else:
            logger.info("TaskService connected to Redis successfully.")

    def _connect_redis(self):
        try:
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                decode_responses=True
            )
            self.redis_client.ping() # Test connection
        except redis.exceptions.ConnectionError as e:
            logger.error(f"Failed to connect to Redis for TaskService: {e}")
            self.redis_client = None

    # Method to inject the connection manager
    def set_connection_manager(self, manager: ConnectionManager):
        """Sets the WebSocket connection manager for broadcasting updates."""
        self.connection_manager = manager
        logger.info("WebSocket ConnectionManager injected into TaskService.")

    # --- Helper for Updating Redis and Broadcasting --- 
    async def _update_task_and_broadcast(self, task: Task):
        """Updates task data in Redis and broadcasts the update via WebSocket."""
        if not self.redis_client:
            logger.error(f"Cannot update task {task.id}: Redis connection lost.")
            return # Or raise an exception?
        
        task_key = f"{self.task_key_prefix}{task.id}"
        try:
            # 1. Update Redis
            task_json = task.model_dump_json()
            self.redis_client.set(task_key, task_json)
            logger.debug(f"Updated task {task.id} in Redis. Status: {task.status}")

            # 2. Broadcast if manager is available
            if self.connection_manager:
                await self.connection_manager.broadcast(task_json)
                logger.debug(f"Broadcasted update for task {task.id}")
            else:
                logger.warning(f"ConnectionManager not set in TaskService, cannot broadcast update for task {task.id}.")

        except redis.exceptions.ConnectionError as e:
            logger.error(f"Redis error updating/broadcasting task {task.id}: {e}")
            # Handle Redis error - broadcast might have failed or Redis set failed
        except Exception as e:
            logger.exception(f"Unexpected error during task update/broadcast for {task.id}: {e}")


    async def create_task(self, task_create: TaskCreate) -> Task:
        """Creates, stores, and starts orchestrating a new task."""
        if not self.redis_client:
            error_msg = "Redis connection not available. Cannot create task."
            logger.error(error_msg)
            return Task(
                id="error", name=task_create.name, status=TaskStatus.ERROR,
                start_time=datetime.now(timezone.utc), workload_type=task_create.workload_type,
                workload_config=task_create.workload_config, error_message=error_msg
            )

        task_id = str(uuid.uuid4())
        workload_run_id = f"{task_create.workload_type.value}-{task_id[:8]}"
        start_time = datetime.now(timezone.utc)

        task_data = Task(
            id=task_id,
            name=task_create.name,
            status=TaskStatus.PENDING, # Initial status
            start_time=start_time,
            end_time=None,
            workload_type=task_create.workload_type,
            workload_config=task_create.workload_config,
            workload_run_id=workload_run_id,
            anomalies=task_create.anomalies or [],
            anomaly_ids=[],
            error_message=None
        )

        try:
            # 1. Store initial task definition in Redis (PENDING)
            task_key = f"{self.task_key_prefix}{task_id}"
            self.redis_client.set(task_key, task_data.model_dump_json())
            self.redis_client.sadd(self.active_tasks_set_key, task_id)
            logger.info(f"Stored new task {task_id} in Redis (status: PENDING).")

            # Broadcast the initial PENDING state
            await self._update_task_and_broadcast(task_data)

            # 2. Initiate background orchestration
            asyncio.create_task(self._orchestrate_task_start(task_data))
            logger.info(f"Scheduled background processing for task {task_id}")

            # Return the initial task data (which is PENDING)
            return task_data

        except redis.exceptions.ConnectionError as e:
            error_msg = f"Redis error creating task {task_id}: {e}"
            logger.error(error_msg)
            task_data.status = TaskStatus.ERROR
            task_data.error_message = error_msg
            try: self.redis_client.srem(self.active_tasks_set_key, task_id)
            except: pass
            # No broadcast needed here as the task object itself signals the error
            return task_data
        except Exception as e:
            error_msg = f"Unexpected error creating task {task_id}: {e}"
            logger.exception(error_msg)
            task_data.status = TaskStatus.ERROR
            task_data.error_message = error_msg
            try: 
                self.redis_client.srem(self.active_tasks_set_key, task_id)
                self.redis_client.delete(f"{self.task_key_prefix}{task_id}")
            except: pass
            return task_data

    # Updated background task logic
    async def _orchestrate_task_start(self, task: Task):
        task_key = f"{self.task_key_prefix}{task.id}"
        logger.info(f"Starting orchestration for task {task.id} (run_id: {task.workload_run_id})")
        try:
            # 1. Start workload
            logger.info(f"Attempting to start workload {task.workload_run_id} for task {task.id}")
            success = await workload_service.start_workload(
                workload_type=task.workload_type,
                run_id=task.workload_run_id,
                config=task.workload_config
            )
            if not success:
                raise Exception(f"WorkloadService failed to start workload run_id {task.workload_run_id}")
            logger.info(f"Workload {task.workload_run_id} started successfully.")

            # 2. Apply anomalies
            applied_anomaly_details = []
            if task.anomalies:
                logger.info(f"Applying {len(task.anomalies)} anomalies for task {task.id}")
                for anomaly_config in task.anomalies:
                    try:
                        created_anomalies = await k8s_service.apply_anomaly(
                            anomaly_type=anomaly_config.type,
                            target_node=anomaly_config.target,
                            severity=anomaly_config.severity
                        )
                        applied_ids = [c.get('id') for c in created_anomalies if c.get('id')]
                        logger.info(f"Applied anomaly '{anomaly_config.type}' for task {task.id}. Resulting IDs: {applied_ids}")
                        applied_anomaly_details.extend(created_anomalies)
                    except Exception as anomaly_e:
                        logger.error(f"Failed to apply anomaly {anomaly_config.type} for task {task.id}: {anomaly_e}", exc_info=True)
                        # Decide: Raise immediately or collect errors?
                        # Let's raise immediately to mark task as error
                        raise Exception(f"Failed to apply anomaly {anomaly_config.type}: {anomaly_e}")

            # 3. Update task state to RUNNING and broadcast
            task.anomaly_ids = [d.get('id') for d in applied_anomaly_details if d.get('id')]
            task.status = TaskStatus.RUNNING
            task.error_message = None # Clear any potential previous errors
            await self._update_task_and_broadcast(task)
            logger.info(f"Task {task.id} orchestration complete. Status: RUNNING")

        except Exception as e:
            error_msg = f"Orchestration failed for task {task.id}: {e}"
            logger.exception(error_msg)
            # Update task status to ERROR and broadcast
            try:
                # Use the existing task object, update fields, then broadcast
                task.status = TaskStatus.ERROR
                task.error_message = error_msg
                task.end_time = datetime.now(timezone.utc)
                await self._update_task_and_broadcast(task) # Broadcast the ERROR state
                # Move from active to completed set in Redis AFTER broadcasting
                if self.redis_client:
                    self.redis_client.srem(self.active_tasks_set_key, task.id)
                    self.redis_client.sadd(self.completed_tasks_set_key, task.id)
                    logger.info(f"Moved task {task.id} to completed set (status: ERROR).")
            except Exception as update_e:
                logger.error(f"Failed to update/broadcast ERROR state for task {task.id}: {update_e}")


    async def get_task(self, task_id: str) -> Optional[Task]:
        """Retrieves task details from Redis."""
        if not self.redis_client:
            logger.error("Redis connection not available. Cannot get task.")
            return None
        try:
            task_key = f"{self.task_key_prefix}{task_id}"
            task_json = self.redis_client.get(task_key)
            if task_json:
                task = Task.model_validate_json(task_json)
                return task
            else:
                return None
        except redis.exceptions.ConnectionError as e:
            logger.error(f"Redis error getting task {task_id}: {e}")
            raise ConnectionError(f"Redis error getting task: {e}")
        except Exception as e:
            logger.exception(f"Error decoding task data for {task_id} from Redis: {e}")
            return None

    async def stop_task(self, task_id: str) -> Optional[Task]:
        """Initiates the stopping procedure for a task."""
        if not self.redis_client:
            logger.error("Redis connection not available. Cannot stop task.")
            raise ConnectionError("Redis connection not available.")

        task = await self.get_task(task_id)
        if not task or task.status not in [TaskStatus.RUNNING, TaskStatus.PENDING]:
            logger.warning(f"Task {task_id} not found or not in a stoppable state ({task.status if task else 'Not Found'}).")
            return None

        task_key = f"{self.task_key_prefix}{task.id}"
        try:
            # 1. Update status to STOPPING and broadcast
            task.status = TaskStatus.STOPPING
            await self._update_task_and_broadcast(task)
            logger.info(f"Set task {task.id} status to STOPPING and broadcasted.")

            # 2. Initiate background cleanup
            asyncio.create_task(self._orchestrate_task_stop(task))
            logger.info(f"Scheduled background cleanup for task {task_id}")

            return task

        except redis.exceptions.ConnectionError as e:
            error_msg = f"Redis error initiating stop for task {task_id}: {e}"
            logger.error(error_msg)
            raise ConnectionError(f"Redis error stopping task: {e}")
        except Exception as e:
            error_msg = f"Unexpected error initiating stop for task {task_id}: {e}"
            logger.exception(error_msg)
            # Try to update task to ERROR state and broadcast
            try:
                task.status = TaskStatus.ERROR
                task.error_message = error_msg
                task.end_time = datetime.now(timezone.utc)
                await self._update_task_and_broadcast(task)
                if self.redis_client:
                     self.redis_client.srem(self.active_tasks_set_key, task.id)
                     self.redis_client.sadd(self.completed_tasks_set_key, task.id)
            except Exception as final_update_e:
                 logger.error(f"Failed to update/broadcast final ERROR state for task {task.id} after stop failure: {final_update_e}")
            raise Exception(error_msg)

    # Updated background cleanup logic
    async def _orchestrate_task_stop(self, task: Task):
        task_key = f"{self.task_key_prefix}{task.id}"
        logger.info(f"Starting cleanup orchestration for task {task.id} (run_id: {task.workload_run_id})")
        errors = []

        # 1. Delete anomalies
        if task.anomaly_ids:
            logger.info(f"Deleting {len(task.anomaly_ids)} anomalies for task {task.id}")
            for anomaly_id in task.anomaly_ids:
                try:
                    deleted_ids = await k8s_service.delete_anomaly(anomaly_id=anomaly_id)
                    if anomaly_id in deleted_ids:
                        logger.info(f"Deleted anomaly {anomaly_id} for task {task.id}")
                    else:
                        logger.warning(f"Anomaly {anomaly_id} for task {task.id} was not confirmed deleted.")
                except Exception as e:
                    msg = f"Failed to delete anomaly {anomaly_id} for task {task.id}: {e}"
                    logger.error(msg, exc_info=True)
                    errors.append(msg)

        # 2. Stop workload process
        if task.workload_run_id:
            logger.info(f"Stopping workload process {task.workload_run_id} for task {task.id}")
            try:
                success = await workload_service.stop_workload(run_id=task.workload_run_id)
                if success:
                    logger.info(f"Stopped workload process {task.workload_run_id} successfully.")
                else:
                    msg = f"Workload service reported failure stopping run_id {task.workload_run_id}."
                    logger.warning(msg)
                    # errors.append(msg) # Optional: count this as error?
            except Exception as e:
                msg = f"Failed to stop workload {task.workload_run_id} for task {task.id}: {e}"
                logger.error(msg, exc_info=True)
                errors.append(msg)

        # 3. Update final task status and broadcast
        try:
            # Fetch the latest task data to avoid overwriting intermediate states if possible
            # (though unlikely in stop orchestration)
            current_task = await self.get_task(task.id)
            if not current_task:
                 logger.error(f"Task {task.id} disappeared before final stop update.")
                 return
            
            task = current_task # Use the latest fetched task object
            task.end_time = datetime.now(timezone.utc)
            if errors:
                task.status = TaskStatus.ERROR
                existing_error = task.error_message + "; " if task.error_message else ""
                task.error_message = existing_error + "Cleanup errors: " + "; ".join(errors)
            else:
                task.status = TaskStatus.STOPPED
                task.error_message = None
            
            # Update Redis and broadcast final state
            await self._update_task_and_broadcast(task)

            # Move from active to completed set AFTER successful broadcast
            if self.redis_client:
                self.redis_client.srem(self.active_tasks_set_key, task.id)
                self.redis_client.sadd(self.completed_tasks_set_key, task.id)
                logger.info(f"Moved task {task.id} to completed set (final status: {task.status})")

        except Exception as e:
             logger.error(f"Failed to finalize task {task.id} state and broadcast: {e}", exc_info=True)


    async def get_all_tasks(self) -> List[Task]:
        """Retrieves all tasks, active and completed."""
        active_tasks = await self._get_tasks_from_set(self.active_tasks_set_key)
        completed_tasks = await self._get_tasks_from_set(self.completed_tasks_set_key)
        return active_tasks + completed_tasks

    async def get_active_tasks(self) -> List[Task]:
        """Retrieves all tasks currently marked as active (running, pending, stopping)."""
        return await self._get_tasks_from_set(self.active_tasks_set_key)

    async def get_completed_tasks(self) -> List[Task]:
        """Retrieves all tasks marked as completed (stopped, error)."""
        return await self._get_tasks_from_set(self.completed_tasks_set_key)

    async def _get_tasks_from_set(self, set_key: str) -> List[Task]:
        if not self.redis_client:
            logger.error(f"Redis connection not available. Cannot get tasks from {set_key}.")
            return []
        tasks = []
        try:
            task_ids = self.redis_client.smembers(set_key)
            if not task_ids:
                return []

            task_keys = [f"{self.task_key_prefix}{tid}" for tid in task_ids]
            task_jsons = self.redis_client.mget(task_keys)

            valid_tasks = []
            missing_keys = []
            for i, task_json in enumerate(task_jsons):
                task_id = task_ids[i] # Get corresponding ID
                if task_json:
                    try:
                        task = Task.model_validate_json(task_json)
                        valid_tasks.append(task)
                    except Exception as e:
                        logger.warning(f"Failed to decode task data {task_id} from set {set_key}: {e}")
                        missing_keys.append(task_id)
                else:
                     logger.warning(f"Task ID {task_id} found in set {set_key}, but key {self.task_key_prefix}{task_id} is missing.")
                     missing_keys.append(task_id)

            # Optional: Clean up missing keys from the set if found
            if missing_keys:
                logger.info(f"Removing {len(missing_keys)} dangling task IDs from Redis set {set_key}")
                self.redis_client.srem(set_key, *missing_keys)

            # Sort by start time descending (most recent first)
            valid_tasks.sort(key=lambda t: t.start_time if t.start_time else datetime.min, reverse=True)
            return valid_tasks
        except redis.exceptions.ConnectionError as e:
            logger.error(f"Redis error getting tasks from set {set_key}: {e}")
            raise ConnectionError(f"Redis error getting tasks: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error getting tasks from set {set_key}: {e}")
            return []

    async def cleanup_old_tasks(self, retention_period: timedelta = timedelta(days=60)) -> None:
        """Cleans up tasks older than the retention period from both active and completed Redis sets."""
        if not self.redis_client:
            logger.error("Redis connection not available. Cannot perform cleanup of old tasks.")
            return

        cutoff_time = datetime.now() - retention_period
        for set_name, set_key in (("active", self.active_tasks_set_key), ("completed", self.completed_tasks_set_key)):
            try:
                task_ids = self.redis_client.smembers(set_key)
                if not task_ids:
                    continue

                for task_id in task_ids:
                    task_key = f"{self.task_key_prefix}{task_id}"
                    task_json = self.redis_client.get(task_key)
                    if task_json:
                        try:
                            task = Task.model_validate_json(task_json)
                            if task.start_time < cutoff_time:
                                self.redis_client.srem(set_key, task_id)
                                self.redis_client.delete(task_key)
                                logger.info(f"Cleaned up old {set_name} task {task.id}")
                        except Exception as e:
                            logger.warning(f"Failed to parse task {task_id} during cleanup: {e}")
                    else:
                        # Remove stale task IDs whose data is missing
                        self.redis_client.srem(set_key, task_id)
                        logger.info(f"Removed dangling task {task_id} from {set_name} set")
            except redis.exceptions.ConnectionError as e:
                logger.error(f"Redis connection error during cleanup of {set_name} tasks: {e}")
            except Exception as e:
                logger.exception(f"Unexpected error during cleanup of {set_name} tasks: {e}")

# Singleton instance (consider a factory or dependency injection for better testability)
task_service = TaskService() 