from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Body, status, WebSocket, WebSocketDisconnect
from typing import List, Optional, Set
import logging
import json # For WebSocket messages

# Import schemas from the dedicated module
from ..schemas.tasks import Task, TaskCreate, TaskStatus, AnomalyConfig, WorkloadType

# Placeholder imports - replace with actual service instance or dependency injection
# from ..services.task_service import task_service # Direct import
from ..services.task_service import TaskService # Import class for dependency

# --- Connection Manager for WebSockets ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        # Use logger defined later in the module or pass one in
        self.logger = logging.getLogger(__name__) 

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        self.logger.info(f"WebSocket connected: {websocket.client}. Total clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            self.logger.info(f"WebSocket disconnected: {websocket.client}. Total clients: {len(self.active_connections)}")
        else:
             self.logger.warning(f"Attempted to disconnect already removed client: {websocket.client}")


    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
             await websocket.send_text(message)
        except Exception as e:
             self.logger.error(f"Failed to send personal message to {websocket.client}: {e}")
             # Optionally disconnect on send error
             # self.disconnect(websocket)

    async def broadcast(self, message: str):
        # Create a list of tasks to avoid issues if the set changes during iteration
        # Use list comprehension for slightly cleaner iteration copy
        current_connections = list(self.active_connections)
        disconnected_clients = set()
        for connection in current_connections:
            try:
                # Check readyState before sending? (Requires JS knowledge, not directly applicable here)
                # FastAPI handles send errors gracefully typically
                await connection.send_text(message)
            except Exception as e: # Handle potential errors like closed connections during broadcast
                 self.logger.warning(f"Broadcast failed for {connection.client}: {e}. Marking for disconnect.")
                 # Don't disconnect immediately while iterating, mark them
                 disconnected_clients.add(connection)

        # Remove disconnected clients after iteration
        for client in disconnected_clients:
            self.disconnect(client) # Use the disconnect method


# Create a singleton manager instance
# This should ideally be managed within the app lifecycle or via dependency injection
manager = ConnectionManager()

# --- Task Service Dependency ---
async def get_task_service():
    from ..services.task_service import task_service # Import the singleton instance
    if not task_service.redis_client:
         raise HTTPException(status_code=503, detail="TaskService Redis connection not available")
    # Inject the connection manager instance into the task service
    if not task_service.connection_manager:
        task_service.set_connection_manager(manager)
    return task_service


router = APIRouter()
logger = logging.getLogger(__name__)

# --- WebSocket Endpoint ---
@router.websocket("/ws/tasks")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep the connection open, wait for client messages (if any)
            # Or just keep alive. For task updates, we mostly push from server.
            # If client sends data, handle here. Example:
            # data = await websocket.receive_text()
            # await manager.send_personal_message(f"You wrote: {data}", websocket)
            # For now, just keep listening passively to detect disconnects
            # Setting a receive timeout can help clean up dead connections if needed
            # but FastAPI handles disconnect exceptions well.
            data = await websocket.receive_text() # This will raise WebSocketDisconnect if client closes
            # Optional: If client sends data, log or process it
            # logger.debug(f"Received WebSocket data from {websocket.client}: {data}")
    except WebSocketDisconnect:
        # manager.disconnect is called within the finally block
        logger.info(f"WebSocket connection closed by client {websocket.client}")
    except Exception as e:
        logger.error(f"WebSocket Error for client {websocket.client}: {e}", exc_info=True)
        # Ensure disconnect happens even on unexpected errors (handled in finally)
    finally:
        # Ensure disconnection happens regardless of how the loop exits
        manager.disconnect(websocket)


@router.post(
    "/",
    response_model=Task,
    status_code=status.HTTP_201_CREATED,
    summary="Create and start a new Task",
    description="Defines a task (workload + optional anomalies) and initiates its execution. Updates are sent via WebSocket.",
)
async def create_task(
    task_create: TaskCreate = Body(...),
    task_service: TaskService = Depends(get_task_service)
):
    """
    Creates a new task definition. The service will then attempt to:
    1. Store the task definition persistently.
    2. Start the associated workload process.
    3. Apply any specified anomalies.

    The initial response returns the task details with a 'pending' or 'error' status.
    The actual execution happens asynchronously, and updates are sent via WebSocket.
    """
    try:
        logger.info(f"Received request to create task: {task_create.model_dump()}")
        # Task service now handles storing, starting orchestration, and broadcasting initial state
        created_task = await task_service.create_task(task_create)

        if created_task.status == TaskStatus.ERROR:
             raise HTTPException(status_code=500, detail=f"Failed to create task: {created_task.error_message}")

        # --- Broadcast Removed --- 
        # The TaskService._update_task_and_broadcast now handles this.
        # --- End Broadcast Removed ---

        # Return the initial PENDING state (or ERROR if immediate failure)
        return created_task
    except ConnectionError as e:
        logger.error(f"Task creation failed due to Redis connection error: {e}")
        raise HTTPException(status_code=503, detail=f"Task service unavailable: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error creating task: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error creating task: {str(e)}")


@router.get(
    "/",
    response_model=List[Task],
    summary="List all tasks (Initial Load)",
    description="Retrieves a list of all tasks. Use for initial page load. Real-time updates via WebSocket.",
)
async def list_tasks(
    task_service: TaskService = Depends(get_task_service)
):
    """
    Get the current list of all tasks. Primarily for initial data load.
    The list is sorted by start_time descending (most recent first).
    """
    try:
        all_tasks = []
        try:
            # Try using a dedicated get_all_tasks method if it exists
            all_tasks = await task_service.get_all_tasks()
            logger.debug("Fetched all tasks using get_all_tasks()")
        except AttributeError:
            # Fallback: Get active and completed separately
            logger.debug("get_all_tasks not found, fetching active and completed tasks separately.")
            active = await task_service.get_active_tasks()
            completed = await task_service.get_completed_tasks()
            all_tasks = active + completed

        # Sort tasks by start_time descending, handling None values
        all_tasks.sort(key=lambda t: t.start_time if t.start_time else datetime.min, reverse=True)
        logger.info(f"Returning {len(all_tasks)} tasks for initial load.")
        return all_tasks

    except ConnectionError as e:
        logger.error(f"Task listing failed due to Redis connection error: {e}")
        raise HTTPException(status_code=503, detail=f"Task service unavailable: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error listing tasks: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error listing tasks: {str(e)}")


@router.get(
    "/{task_id}",
    response_model=Task,
    summary="Get Task Details",
    description="Retrieves the full details and status of a specific task by its ID. Updates also sent via WebSocket.",
)
async def get_task(
    task_id: str,
    task_service: TaskService = Depends(get_task_service)
):
    """
    Fetch the current state of a task, including its configuration,
    status, associated workload/anomaly IDs, and timings.
    """
    try:
        task = await task_service.get_task(task_id)
        if task is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Task with ID '{task_id}' not found")
        return task
    except ConnectionError as e:
        logger.error(f"Get task {task_id} failed due to Redis connection error: {e}")
        raise HTTPException(status_code=503, detail=f"Task service unavailable: {e}")
    except HTTPException:
        raise # Re-raise 404 Not Found
    except Exception as e:
        logger.exception(f"Unexpected error getting task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error getting task: {str(e)}")


@router.post(
    "/{task_id}/stop",
    response_model=Task,
    summary="Stop a Task",
    description="Initiates the stopping procedure for a running task. Status updates sent via WebSocket.",
)
async def stop_task(
    task_id: str,
    task_service: TaskService = Depends(get_task_service)
):
    """
    Requests the TaskService to initiate the stop procedure.
    The operation is asynchronous. The response indicates the stop process has been initiated.
    Status updates (stopping -> stopped/error) are broadcast via WebSocket by the service.
    """
    try:
        logger.info(f"Received request to stop task {task_id}")
        # Task service handles setting status to STOPPING, broadcasting, and scheduling cleanup
        task = await task_service.stop_task(task_id)
        if task is None:
            # This might happen if the task was already stopped or didn't exist
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Task with ID '{task_id}' not found or cannot be stopped.")

        # --- Broadcast Removed --- 
        # TaskService._update_task_and_broadcast handles the STOPPING state broadcast.
        # Subsequent STOPPED/ERROR states are broadcast by _orchestrate_task_stop.
        # --- End Broadcast Removed ---

        # Return the task in its current state (likely STOPPING)
        return task
    except ConnectionError as e:
        logger.error(f"Stop task {task_id} failed due to Redis connection error: {e}")
        raise HTTPException(status_code=503, detail=f"Task service unavailable: {e}")
    except HTTPException:
        raise # Re-raise 404
    except Exception as e:
        logger.exception(f"Unexpected error stopping task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error stopping task: {str(e)}")

# TODO: Add DELETE endpoint for completed tasks?
# TODO: Add endpoint to get task output/logs? (Might require changes in WorkloadService) 