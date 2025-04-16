import { useState, useEffect, useCallback } from 'react';
import { workloadService } from '../services/workloadService';
import { message } from 'antd';

export const useWorkload = () => {
  // Remove state related to available nodes and general active workloads
  // const [availableNodes, setAvailableNodes] = useState([]);
  const [loading, setLoading] = useState(false);
  const [activeWorkloadRuns, setActiveWorkloadRuns] = useState([]); // Renamed state
  const [error, setError] = useState(null);

  // Remove fetchAvailableNodes
  // const fetchAvailableNodes = useCallback(async () => { ... });

  // Rename fetchActiveWorkloads and update its logic
  const fetchActiveWorkloadRuns = useCallback(async () => {
    try {
      setLoading(true);
      // Call the renamed service function
      const runs = await workloadService.getActiveWorkloadRuns();
      setActiveWorkloadRuns(runs); // Update the renamed state
      setError(null);
    } catch (err) {
      console.error('Error fetching active workload runs:', err);
      // Use Ant Design message for user feedback
      message.error(`Failed to fetch active workload runs: ${err.message}`);
      setError('Failed to fetch active workload runs');
    } finally {
      setLoading(false);
    }
  }, []);

  // Remove startWorkload function
  // const startWorkload = useCallback(async (...) => { ... }, [fetchActiveWorkloads]);

  // Remove stopWorkload function
  // const stopWorkload = useCallback(async (...) => { ... }, [fetchActiveWorkloads]);

  // Remove getWorkloadStatus function
  // const getWorkloadStatus = useCallback(async (...) => { ... }, []);

  // Remove useEffect that fetched available nodes
  // useEffect(() => { fetchAvailableNodes(); }, [fetchAvailableNodes]);

  // Keep prepareDatabase functionality if needed separately
  const prepareDatabase = useCallback(async (workloadType) => {
     setLoading(true);
     setError(null);
     try {
        await workloadService.prepareDatabase(workloadType);
        message.success(`Database preparation for ${workloadType} initiated successfully.`);
        // Optionally return true or the response data
        return true;
     } catch (err) {
        console.error(`Error preparing database for ${workloadType}:`, err);
        message.error(`Failed to prepare database for ${workloadType}: ${err.message}`);
        setError(`Failed to prepare database: ${err.message}`);
        return false;
     } finally {
        setLoading(false);
     }
  }, []);

  return {
    // Remove availableNodes
    activeWorkloadRuns, // Return renamed state
    loading,
    error,
    // Remove fetchAvailableNodes
    fetchActiveWorkloadRuns, // Return renamed fetch function
    // Remove startWorkload
    // Remove stopWorkload
    // Remove getWorkloadStatus
    prepareDatabase // Keep prepareDatabase if needed
  };
}; 