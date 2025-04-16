import axios from 'axios';

const API_BASE_URL =
    import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * Service for interacting with the Task API (/api/tasks).
 */
export const taskService = {
    /**
     * Creates a new task by sending the definition to the backend.
     * @param {object} taskCreateData - Data matching the TaskCreate schema.
     * @param {string} taskCreateData.name - User-defined name for the task.
     * @param {string} taskCreateData.workload_type - e.g., 'sysbench', 'tpcc'.
     * @param {object} taskCreateData.workload_config - Configuration for the workload.
     * @param {Array<object>} [taskCreateData.anomalies] - Optional list of anomalies.
     * @returns {Promise<object>} The created task object (initially likely in PENDING status).
     */
    createTask: async (taskCreateData) => {
        try {
            const response = await axios.post(`${API_BASE_URL}/api/tasks/`, taskCreateData);
            return response.data; // Should match the Task schema
        } catch (error) {
            console.error('Error creating task:', error);
            const errorMsg = error.response?.data?.detail || error.message || 'Failed to create task';
            throw new Error(errorMsg);
        }
    },

    /**
     * Retrieves a list of tasks. If status is omitted, fetches ALL tasks.
     * @param {string} [status] - Optional status to filter by (e.g., 'running', 'stopped'). If null/undefined, fetches all.
     * @returns {Promise<Array<object>>} A list of tasks matching the Task schema.
     */
    getTasks: async (status = null) => {
        try {
            // If status is provided, use it. Otherwise, send no status param to get all.
            const params = status ? { status } : {};
            console.log(`Fetching tasks with params:`, params); // Log what's being fetched
            const response = await axios.get(`${API_BASE_URL}/api/tasks/`, { params });
            return response.data; // Should be a list of Task objects
        } catch (error) {
            const statusMsg = status ? `(status: ${status})` : '(all)';
            console.error(`Error fetching tasks ${statusMsg}:`, error);
            const errorMsg = error.response?.data?.detail || error.message || 'Failed to fetch tasks';
            throw new Error(errorMsg);
        }
    },

     /**
     * Retrieves the details of a specific task by its ID.
     * @param {string} taskId - The ID of the task to fetch.
     * @returns {Promise<object>} The task object matching the Task schema.
     */
    getTask: async (taskId) => {
        try {
            const response = await axios.get(`${API_BASE_URL}/api/tasks/${taskId}`);
            return response.data; // Should match the Task schema
        } catch (error) {
            console.error(`Error fetching task ${taskId}:`, error);
            const errorMsg = error.response?.data?.detail || error.message || 'Failed to fetch task details';
            if (error.response?.status === 404) {
                throw new Error(`Task with ID '${taskId}' not found.`);
            }
            throw new Error(errorMsg);
        }
    },

    /**
     * Initiates the stop procedure for a specific task.
     * @param {string} taskId - The ID of the task to stop.
     * @returns {Promise<object>} The task object, likely in STOPPING status.
     */
    stopTask: async (taskId) => {
        try {
            const response = await axios.post(`${API_BASE_URL}/api/tasks/${taskId}/stop`);
            return response.data; // Should match the Task schema (in stopping state)
        } catch (error) {
            console.error(`Error stopping task ${taskId}:`, error);
            const errorMsg = error.response?.data?.detail || error.message || 'Failed to stop task';
             if (error.response?.status === 404) {
                throw new Error(`Task with ID '${taskId}' not found or cannot be stopped.`);
            }
            throw new Error(errorMsg);
        }
    },

    // Add other task-related API calls here if needed in the future
    // e.g., deleteTask, cleanupTasks, etc.
}; 