import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://10.101.168.97:8001';

export const workloadService = {
  prepareDatabase: async () => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/workload/prepare`);
      return response.data;
    } catch (error) {
      console.error('Error preparing database:', error);
      const errorMsg = error.response?.data?.detail || error.message;
      throw new Error(`Failed to prepare database: ${errorMsg}`);
    }
  },

  startWorkload: async (workloadType, threads = 4) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/workload/start`, {
        type: workloadType,
        threads: threads
      });
      return response.data;
    } catch (error) {
      console.error('Error starting workload:', error);
      const errorMsg = error.response?.data?.detail || error.message;
      throw new Error(`Failed to start workload: ${errorMsg}`);
    }
  },

  stopWorkload: async (workloadId) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/workload/stop/${workloadId}`);
      return response.data;
    } catch (error) {
      console.error('Error stopping workload:', error);
      const errorMsg = error.response?.data?.detail || error.message;
      throw new Error(`Failed to stop workload: ${errorMsg}`);
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
      const workloads = response.data || [];
      
      // Extract system metrics from the first workload if available
      const systemMetrics = workloads[0]?.metrics || {
        cpu_usage: 0,
        memory_usage: 0,
        disk_usage: 0
      };

      return {
        workloads: workloads.map(workload => ({
          id: workload.id,
          type: workload.type,
          threads: workload.threads,
          pid: workload.pid,
          status: 'running',
          metrics: workload.metrics
        })),
        systemMetrics,
        totalCount: workloads.length
      };
    } catch (error) {
      console.error('Error fetching active workloads:', error);
      const errorMsg = error.response?.data?.detail || error.message;
      throw new Error(`Failed to fetch workloads: ${errorMsg}`);
    }
  },

  // Helper function to format workload status for display
  formatWorkloadStatus: (workload) => {
    if (!workload) return '';
    
    const status = workload.status.toLowerCase();
    const metrics = workload.metrics;
    
    let statusText = `${status.charAt(0).toUpperCase()}${status.slice(1)}`;
    
    if (metrics && status === 'running') {
      statusText += ` (TPS: ${metrics.tps}, QPS: ${metrics.qps}, Latency: ${metrics.latency_ms}ms)`;
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
  }
}; 