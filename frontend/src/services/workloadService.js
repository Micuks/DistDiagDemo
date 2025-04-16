import axios from 'axios';

const API_BASE_URL =
    import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const workloadService = {
    prepareDatabase: async (workloadType) => {
        try {
            // Ensure workloadType is a string value from the enum if necessary
            const response = await axios.post(`${API_BASE_URL}/api/workload/prepare`, { type: workloadType });
            return response.data;
        } catch (error) {
            console.error('Error preparing database:', error);
            const errorMsg = error.response?.data?.detail || error.message;
            throw new Error(`Failed to prepare database: ${errorMsg}`);
        }
    },

    // Removed startWorkload
    // Removed createTask
    // Removed stopWorkload
    // Removed stopAllWorkloads

    // Renamed getActiveWorkloads to getActiveWorkloadRuns
    getActiveWorkloadRuns: async () => {
        try {
            // Use the refactored endpoint /api/workload/active
            const response = await axios.get(`${API_BASE_URL}/api/workload/active`);
            // The response should be List[WorkloadRunInfo]
            return response.data;
        } catch (error) {
            console.error('Error getting active workload runs:', error);
            const errorMsg = error.response?.data?.detail || 'Network error or server unavailable';
            throw new Error(`Failed to get active workload runs: ${errorMsg}`);
        }
    },

    // Removed getAvailableNodes (was returning empty array anyway)

    // Removed formatWorkloadStatus
    // Removed getStatusColor
    // Removed getWorkloadStatus

    // --- Removed all Task related functions --- //
    // async getTasks() { ... }
    // async getActiveTasks() { ... }
    // async getTask(taskId) { ... }
    // async stopTask(taskId) { ... }
    // async cleanupOldTasks() { ... }
    // async getCompletedTasks() { ... }
    // async getTaskHistory() { ... }
};