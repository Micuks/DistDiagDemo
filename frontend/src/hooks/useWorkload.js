import { useState, useEffect, useCallback } from 'react';
import { workloadService } from '../services/workloadService';
import { message } from 'antd';

export const useWorkload = () => {
  const [availableNodes, setAvailableNodes] = useState([]);
  const [loading, setLoading] = useState(false);
  const [activeWorkloads, setActiveWorkloads] = useState([]);
  const [error, setError] = useState(null);

  // Fetch available nodes
  const fetchAvailableNodes = useCallback(async () => {
    try {
      const nodes = await workloadService.getAvailableNodes();
      setAvailableNodes(nodes);
    } catch (err) {
      console.error('Error fetching available nodes:', err);
      setError('Failed to fetch available nodes');
    }
  }, []);

  // Fetch active workloads
  const fetchActiveWorkloads = useCallback(async () => {
    try {
      setLoading(true);
      const workloads = await workloadService.getActiveWorkloads();
      setActiveWorkloads(workloads);
      setError(null);
    } catch (err) {
      console.error('Error fetching active workloads:', err);
      setError('Failed to fetch active workloads');
    } finally {
      setLoading(false);
    }
  }, []);

  // Start a workload
  const startWorkload = useCallback(async (workloadType, threads, options) => {
    try {
      setLoading(true);
      const result = await workloadService.startWorkload(workloadType, threads, options);
      message.success(`Started ${workloadType} workload successfully`);
      await fetchActiveWorkloads();
      return result;
    } catch (err) {
      console.error('Error starting workload:', err);
      message.error(`Failed to start workload: ${err.message}`);
      setError(`Failed to start workload: ${err.message}`);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [fetchActiveWorkloads]);

  // Stop a workload
  const stopWorkload = useCallback(async (workloadId) => {
    try {
      setLoading(true);
      const result = await workloadService.stopWorkload(workloadId);
      message.success(`Stopped workload successfully`);
      await fetchActiveWorkloads();
      return result;
    } catch (err) {
      console.error('Error stopping workload:', err);
      message.error(`Failed to stop workload: ${err.message}`);
      setError(`Failed to stop workload: ${err.message}`);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [fetchActiveWorkloads]);

  // Get workload status
  const getWorkloadStatus = useCallback(async (workloadId) => {
    try {
      const status = await workloadService.getWorkloadStatus(workloadId);
      return status;
    } catch (err) {
      console.error('Error getting workload status:', err);
      setError(`Failed to get workload status: ${err.message}`);
      throw err;
    }
  }, []);

  // Initialize by fetching available nodes
  useEffect(() => {
    fetchAvailableNodes();
  }, [fetchAvailableNodes]);

  return {
    availableNodes,
    activeWorkloads,
    loading,
    error,
    fetchAvailableNodes,
    fetchActiveWorkloads,
    startWorkload,
    stopWorkload,
    getWorkloadStatus
  };
}; 