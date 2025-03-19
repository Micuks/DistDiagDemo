import { useState, useEffect, useCallback } from 'react';
import { anomalyService } from '../services/anomalyService';
import { message } from 'antd';

export const useAnomaly = () => {
  const [availableNodes, setAvailableNodes] = useState([]);
  const [anomalyTypes, setAnomalyTypes] = useState([]);
  const [loading, setLoading] = useState(false);
  const [activeAnomalies, setActiveAnomalies] = useState([]);
  const [error, setError] = useState(null);

  // Fetch available nodes
  const fetchAvailableNodes = useCallback(async () => {
    try {
      const nodes = await anomalyService.getAvailableNodes();
      setAvailableNodes(nodes);
    } catch (err) {
      console.error('Error fetching available nodes:', err);
      setError('Failed to fetch available nodes');
    }
  }, []);

  // Load anomaly types
  const loadAnomalyTypes = useCallback(() => {
    const types = anomalyService.getAnomalyTypes();
    setAnomalyTypes(types);
  }, []);

  // Fetch active anomalies
  const fetchActiveAnomalies = useCallback(async () => {
    try {
      setLoading(true);
      const anomalies = await anomalyService.getActiveAnomalies();
      setActiveAnomalies(anomalies);
      setError(null);
    } catch (err) {
      console.error('Error fetching active anomalies:', err);
      setError('Failed to fetch active anomalies');
    } finally {
      setLoading(false);
    }
  }, []);

  // Inject an anomaly
  const injectAnomaly = useCallback(async (anomalyType, targetNode, severity, duration) => {
    try {
      setLoading(true);
      const result = await anomalyService.injectAnomaly(anomalyType, targetNode, severity, duration);
      message.success(`Injected ${anomalyType} anomaly successfully`);
      await fetchActiveAnomalies();
      return result;
    } catch (err) {
      console.error('Error injecting anomaly:', err);
      message.error(`Failed to inject anomaly: ${err.message}`);
      setError(`Failed to inject anomaly: ${err.message}`);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [fetchActiveAnomalies]);

  // Stop an anomaly
  const stopAnomaly = useCallback(async (anomalyId) => {
    try {
      setLoading(true);
      const result = await anomalyService.stopAnomaly(anomalyId);
      message.success(`Stopped anomaly successfully`);
      await fetchActiveAnomalies();
      return result;
    } catch (err) {
      console.error('Error stopping anomaly:', err);
      message.error(`Failed to stop anomaly: ${err.message}`);
      setError(`Failed to stop anomaly: ${err.message}`);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [fetchActiveAnomalies]);

  // Stop all anomalies
  const stopAllAnomalies = useCallback(async () => {
    try {
      setLoading(true);
      const result = await anomalyService.stopAllAnomalies();
      message.success('Stopped all anomalies successfully');
      await fetchActiveAnomalies();
      return result;
    } catch (err) {
      console.error('Error stopping all anomalies:', err);
      message.error(`Failed to stop all anomalies: ${err.message}`);
      setError(`Failed to stop all anomalies: ${err.message}`);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [fetchActiveAnomalies]);

  // Get anomaly status
  const getAnomalyStatus = useCallback(async (anomalyId) => {
    try {
      const status = await anomalyService.getAnomalyStatus(anomalyId);
      return status;
    } catch (err) {
      console.error('Error getting anomaly status:', err);
      setError(`Failed to get anomaly status: ${err.message}`);
      throw err;
    }
  }, []);

  // Initialize by fetching available nodes and loading anomaly types
  useEffect(() => {
    fetchAvailableNodes();
    loadAnomalyTypes();
  }, [fetchAvailableNodes, loadAnomalyTypes]);

  return {
    availableNodes,
    anomalyTypes,
    activeAnomalies,
    loading,
    error,
    fetchAvailableNodes,
    fetchActiveAnomalies,
    injectAnomaly,
    stopAnomaly,
    stopAllAnomalies,
    getAnomalyStatus
  };
}; 