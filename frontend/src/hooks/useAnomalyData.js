import { useState, useEffect, useCallback, useRef } from 'react';
import { anomalyService } from '../services/anomalyService';

const cache = new Map();
const STALE_TIME = 10000; // Increase stale time to 10 seconds
const MAX_RETRIES = 3; // Reduce max retries to avoid hammering the server

export const useAnomalyData = () => {
  const [data, setData] = useState(cache.get('anomalies')?.data || []);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(!data.length);
  const eventSourceRef = useRef(null);
  const pollingIntervalRef = useRef(null);
  const lightPollingIntervalRef = useRef(null);

  const fetchData = useCallback(async (force = false) => {
    const now = Date.now();
    const cached = cache.get('anomalies');
    
    // Return cached data if fresh enough
    if (!force && cached && now - cached.timestamp < STALE_TIME) {
      return cached.data;
    }

    // Don't set loading if we already have cached data
    if (!cached?.data?.length) {
      setIsLoading(true);
    }
    
    try {
      const freshData = await anomalyService.getActiveAnomalies();
      
      // Ensure we always store an array
      const normalizedData = Array.isArray(freshData) ? freshData : [];
      
      cache.set('anomalies', {
        data: normalizedData,
        timestamp: now,
        errorCount: 0
      });
      
      setData(normalizedData);
      setError(null);
      return normalizedData;
    } catch (err) {
      console.error('Error fetching anomaly data:', err);
      const errorCount = (cache.get('anomalies')?.errorCount || 0) + 1;
      if (errorCount <= MAX_RETRIES) {
        cache.set('anomalies', { ...cache.get('anomalies'), errorCount });
        // Exponential backoff
        const delay = Math.min(1000 * Math.pow(1.5, errorCount - 1), 10000);
        await new Promise(resolve => setTimeout(resolve, delay));
        return fetchData(force);
      }
      setError(err);
      // Continue using previous data instead of setting to empty
      if (cached?.data) {
        setData(cached.data);
      } else {
        setData([]); // Set empty array if no cached data
      }
      return [];
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    // Clean up previous connections if they exist
    const cleanupConnections = () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
      
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
      
      if (lightPollingIntervalRef.current) {
        clearInterval(lightPollingIntervalRef.current);
        lightPollingIntervalRef.current = null;
      }
    };
    
    cleanupConnections();
    
    // Initial fetch
    fetchData(true);

    // Set up SSE connection
    const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://10.101.168.97:8001';
    const setupEventSource = () => {
      try {
        const eventSource = new EventSource(`${API_BASE_URL}/api/anomaly/stream`);
        eventSourceRef.current = eventSource;
        
        eventSource.onmessage = (event) => {
          try {
            const freshData = JSON.parse(event.data);
            
            // Ensure the data is an array
            const normalizedData = Array.isArray(freshData) ? freshData : [];
            
            cache.set('anomalies', {
              data: normalizedData,
              timestamp: Date.now(),
              errorCount: 0
            });
            setData(normalizedData);
          } catch (err) {
            console.error('Error processing SSE message:', err);
          }
        };
        
        eventSource.addEventListener('anomaly_update', (event) => {
          try {
            const freshData = JSON.parse(event.data);
            
            // Ensure the data is an array
            const normalizedData = Array.isArray(freshData) ? freshData : [];
            
            cache.set('anomalies', {
              data: normalizedData,
              timestamp: Date.now(),
              errorCount: 0
            });
            setData(normalizedData);
          } catch (err) {
            console.error('Error processing anomaly_update event:', err);
          }
        });

        eventSource.onerror = (error) => {
          console.error('SSE connection error:', error);
          // Close the current connection
          if (eventSourceRef.current) {
            eventSourceRef.current.close();
            eventSourceRef.current = null;
          }
          
          // Start polling as fallback
          if (!pollingIntervalRef.current) {
            pollingIntervalRef.current = setInterval(() => {
              if (!document.hidden && navigator.onLine) {
                fetchData(true);
              }
            }, 5000); // 5s fallback polling
          }
          
          // Try to reconnect SSE after a delay
          setTimeout(() => {
            if (!document.hidden && navigator.onLine) {
              setupEventSource();
            }
          }, 5000);
        };
        
        return eventSource;
      } catch (error) {
        console.error('Failed to setup EventSource:', error);
        return null;
      }
    };
    
    setupEventSource();

    // Also poll occasionally even with SSE to ensure data is fresh
    lightPollingIntervalRef.current = setInterval(() => {
      if (!document.hidden && navigator.onLine) {
        fetchData(false); // Use cached data if still valid
      }
    }, 30000); // 30s light polling as additional safety

    return () => {
      cleanupConnections();
    };
  }, [fetchData]);

  return {
    data,
    error,
    isLoading,
    refetch: () => fetchData(true)
  };
}; 