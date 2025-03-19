import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const workloadService = {
  prepareDatabase: async (workloadType) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/workload/prepare`, { type: workloadType });
      return response.data;
    } catch (error) {
      console.error('Error preparing database:', error);
      const errorMsg = error.response?.data?.detail || error.message;
      throw new Error(`Failed to prepare database: ${errorMsg}`);
    }
  },

  startWorkload: async (workloadType, threads, options = {}) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/workload/start`, {
        type: workloadType,
        threads: threads,
        options: options
      });
      return response.data;
    } catch (error) {
      console.error('Error starting workload:', error);
      throw new Error('Failed to start workload: ' + (error.response?.data?.detail || error.message));
    }
  },

  stopWorkload: async (workloadId) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/workload/${workloadId}/stop`);
      return response.data;
    } catch (error) {
      console.error('Error stopping workload:', error);
      throw new Error('Failed to stop workload: ' + (error.response?.data?.detail || error.message));
    }
  },

  stopAllWorkloads: async () => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/workload/stop-all`);
      return response.data;
    } catch (error) {
      console.error('Error stopping all workloads:', error);
      const errorMsg = error.response?.data?.detail || error.message;
      throw new Error(`Failed to stop workloads: ${errorMsg}`);
    }
  },

  getActiveWorkloads: async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/workload/active`);
      return response.data;
    } catch (error) {
      console.error('Error getting active workloads:', error);
      throw new Error('Failed to get active workloads: ' + (error.response?.data?.detail || error.message));
    }
  },

  getAvailableNodes: async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/workload/nodes`);
      return response.data;
    } catch (error) {
      console.error('Error fetching available nodes:', error);
      return [];
    }
  },

  // Helper function to format workload status for display
  formatWorkloadStatus: (workload) => {
    if (!workload) return '';
    
    const status = workload.status.toLowerCase();
    const metrics = workload.metrics;
    
    let statusText = `${status.charAt(0).toUpperCase()}${status.slice(1)}`;
    
    if (metrics && status === 'running') {
      switch (workload.type) {
        case 'sysbench':
          statusText += ` (TPS: ${metrics.tps}, QPS: ${metrics.qps}, Latency: ${metrics.latency_ms}ms)`;
          break;
        case 'tpcc':
          statusText += ` (TPM: ${metrics.tpm}, Latency: ${metrics.latency_ms}ms)`;
          break;
        case 'tpch':
          // For TPCH, show the latest query execution time
          const queryTimes = Object.entries(metrics)
            .filter(([key]) => key.startsWith('q'))
            .sort(([a], [b]) => parseInt(b.slice(1)) - parseInt(a.slice(1)));
          if (queryTimes.length > 0) {
            const [query, time] = queryTimes[0];
            statusText += ` (${query.toUpperCase()}: ${time}s)`;
          }
          break;
      }
    }
    
    if (workload.error_message) {
      statusText += ` - Error: ${workload.error_message}`;
    }
    
    return statusText;
  },

  // Helper function to get status color
  getStatusColor: (status) => {
    switch (status.toLowerCase()) {
      case 'running':
        return 'success';
      case 'starting':
        return 'info';
      case 'stopping':
        return 'warning';
      case 'failed':
        return 'error';
      default:
        return 'default';
    }
  },

  getWorkloadStatus: async (workloadId) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/workload/${workloadId}/status`);
      return response.data;
    } catch (error) {
      console.error('Error getting workload status:', error);
      throw new Error('Failed to get workload status: ' + (error.response?.data?.detail || error.message));
    }
  }
}; 