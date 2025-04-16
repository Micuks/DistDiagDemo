import { useState, useEffect, useCallback } from 'react';
import anomalyService from '../services/anomalyService';
import { useSnackbar } from 'notistack';

export const useAnomaly = (refreshInterval = 10000) => {
    const [availableNodes, setAvailableNodes] = useState([]);
    const [activeAnomalies, setActiveAnomalies] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [isSocketConnected, setIsSocketConnected] = useState(false);
    const [error, setError] = useState(null);
    const { enqueueSnackbar } = useSnackbar();
    
    // Fetch functions
    const fetchAvailableNodes = useCallback(async () => {
        try {
            setIsLoading(true);
            const nodes = await anomalyService.getAvailableNodes();
            setAvailableNodes(nodes);
            setError(null);
        } catch (err) {
            console.error('Error fetching available nodes:', err);
            setError(err?.message || 'Failed to fetch available nodes');
        } finally {
            setIsLoading(false);
        }
    }, []);

    const fetchActiveAnomalies = useCallback(async () => {
        try {
            setIsLoading(true);
            const anomalies = await anomalyService.getActiveAnomalies();
            setActiveAnomalies(anomalies);
            setError(null);
        } catch (err) {
            console.error('Error fetching active anomalies:', err);
            setError(err?.message || 'Failed to fetch active anomalies');
        } finally {
            setIsLoading(false);
        }
    }, []);

    // WebSocket handlers
    const handleWebSocketMessage = useCallback((data) => {
        if (data.type === 'anomalies_update') {
            setActiveAnomalies(data.anomalies || []);
        } else if (data.type === 'nodes_update') {
            setAvailableNodes(data.nodes || []);
        } else if (data.type === 'anomaly_started') {
            enqueueSnackbar(`Anomaly started: ${data.anomaly?.type} on ${data.anomaly?.node || 'system'}`, { 
                variant: 'info' 
            });
            fetchActiveAnomalies();
        } else if (data.type === 'anomaly_stopped') {
            enqueueSnackbar(`Anomaly stopped: ${data.anomaly?.type} on ${data.anomaly?.node || 'system'}`, { 
                variant: 'success' 
            });
            fetchActiveAnomalies();
        } else if (data.type === 'error') {
            enqueueSnackbar(data.message || 'Unknown error occurred', { 
                variant: 'error' 
            });
        }
    }, [enqueueSnackbar, fetchActiveAnomalies]);

    const setupWebSocket = useCallback(() => {
        anomalyService.connectToWebSocket({
            onMessage: handleWebSocketMessage,
            onConnect: () => {
                setIsSocketConnected(true);
                enqueueSnackbar('Connected to anomaly real-time updates', { 
                    variant: 'success' 
                });
            },
            onDisconnect: () => {
                setIsSocketConnected(false);
                enqueueSnackbar('Disconnected from anomaly real-time updates', { 
                    variant: 'warning' 
                });
            },
            onError: (error) => {
                console.error('WebSocket error:', error);
                setIsSocketConnected(false);
                enqueueSnackbar('Error connecting to anomaly updates', { 
                    variant: 'error' 
                });
            }
        });
    }, [handleWebSocketMessage, enqueueSnackbar]);

    // Anomaly injection and management
    const injectAnomaly = useCallback(async (type, node_or_params) => {
        try {
            setIsLoading(true);
            await anomalyService.startAnomaly(type, node_or_params);
            enqueueSnackbar(`Successfully injected anomaly: ${type}`, { variant: 'success' });
            fetchActiveAnomalies();
        } catch (err) {
            console.error('Error injecting anomaly:', err);
            enqueueSnackbar(err?.message || `Failed to inject anomaly: ${type}`, { variant: 'error' });
        } finally {
            setIsLoading(false);
        }
    }, [enqueueSnackbar, fetchActiveAnomalies]);

    const stopAnomaly = useCallback(async (typeOrId, experimentName) => {
        try {
            setIsLoading(true);
            await anomalyService.stopAnomaly(typeOrId, experimentName);
            enqueueSnackbar(`Successfully stopped anomaly`, { variant: 'success' });
            fetchActiveAnomalies();
        } catch (err) {
            console.error('Error stopping anomaly:', err);
            enqueueSnackbar(err?.message || 'Failed to stop anomaly', { variant: 'error' });
        } finally {
            setIsLoading(false);
        }
    }, [enqueueSnackbar, fetchActiveAnomalies]);

    const stopAllAnomalies = useCallback(async () => {
        try {
            setIsLoading(true);
            await anomalyService.stopAllAnomalies();
            enqueueSnackbar('Successfully stopped all anomalies', { variant: 'success' });
            fetchActiveAnomalies();
        } catch (err) {
            console.error('Error stopping all anomalies:', err);
            enqueueSnackbar(err?.message || 'Failed to stop all anomalies', { variant: 'error' });
        } finally {
            setIsLoading(false);
        }
    }, [enqueueSnackbar, fetchActiveAnomalies]);

    const refreshAnomalies = useCallback(() => {
        if (isSocketConnected) {
            anomalyService.refreshAnomalies();
        } else {
            fetchActiveAnomalies();
        }
    }, [fetchActiveAnomalies, isSocketConnected]);

    // Setup effects
    useEffect(() => {
        fetchAvailableNodes();
        fetchActiveAnomalies();
        setupWebSocket();

        // Cleanup on unmount
        return () => {
            anomalyService.disconnectFromWebSocket();
        };
    }, [fetchAvailableNodes, fetchActiveAnomalies, setupWebSocket]);

    // Set up polling if WebSocket is not connected
    useEffect(() => {
        if (!isSocketConnected && refreshInterval > 0) {
            const intervalId = setInterval(() => {
                fetchActiveAnomalies();
            }, refreshInterval);

            return () => clearInterval(intervalId);
        }
    }, [fetchActiveAnomalies, refreshInterval, isSocketConnected]);

    return {
        availableNodes,
        activeAnomalies,
        isLoading,
        isSocketConnected,
        error,
        injectAnomaly,
        stopAnomaly,
        stopAllAnomalies,
        refreshAnomalies,
        reconnectWebSocket: setupWebSocket
    };
};

export default useAnomaly; 