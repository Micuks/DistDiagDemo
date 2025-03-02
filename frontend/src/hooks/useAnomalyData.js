import { useState, useEffect, useCallback } from 'react';
import { anomalyService } from '../services/anomalyService';

const cache = new Map();
const STALE_TIME = 2000; // Reduce stale time to 2 seconds for faster updates
const MAX_RETRIES = 5; // Increase max retries

export const useAnomalyData = () => {
  const [data, setData] = useState(cache.get('anomalies')?.data || []);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(!data.length);

  const fetchData = useCallback(async (force = false) => {
    const now = Date.now();
    const cached = cache.get('anomalies');
    
    // Return cached data if fresh enough
    if (!force && cached && now - cached.timestamp < STALE_TIME) {
      console.log('Using cached anomaly data', cached.data);
      return cached.data;
    }

    console.log('Fetching fresh anomaly data...');
    try {
      setIsLoading(true);
      const freshData = await anomalyService.getActiveAnomalies();
      
      // Log the received data for debugging
      console.log('Received anomaly data:', freshData);
      
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
        console.log(`Retrying fetch in ${delay}ms (attempt ${errorCount}/${MAX_RETRIES})`);
        await new Promise(resolve => setTimeout(resolve, delay));
        return fetchData(force);
      }
      setError(err);
      console.error('Max retries exceeded:', err);
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
    const controller = new AbortController();
    let pollingInterval;
    
    // Initial fetch
    console.log('Performing initial anomaly data fetch');
    fetchData(true);

    // Set up SSE connection
    const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://10.101.168.97:8001';
    const setupEventSource = () => {
      try {
        console.log('Setting up SSE connection to:', `${API_BASE_URL}/api/anomaly/stream`);
        const eventSource = new EventSource(`${API_BASE_URL}/api/anomaly/stream`);
        console.log('SSE connection established');
        
        eventSource.onmessage = (event) => {
          try {
            console.log('Received generic SSE message:', event.data);
            const freshData = JSON.parse(event.data);
            console.log('Parsed SSE data:', freshData);
            
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
            console.log('Received anomaly_update event:', event.data);
            const freshData = JSON.parse(event.data);
            console.log('Parsed anomaly update data:', freshData);
            
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
          eventSource.close();
          
          // Start polling as fallback
          if (!pollingInterval) {
            console.log('Starting polling fallback');
            pollingInterval = setInterval(() => {
              if (!document.hidden && navigator.onLine) {
                console.log('Polling for anomaly data');
                fetchData(true);
              }
            }, 3000); // 3s fallback polling
          }
          
          // Try to reconnect SSE after a delay
          setTimeout(() => {
            if (!document.hidden && navigator.onLine) {
              console.log('Attempting to reconnect SSE');
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
    
    const eventSource = setupEventSource();

    // Also poll occasionally even with SSE to ensure data is fresh
    const lightPolling = setInterval(() => {
      if (!document.hidden && navigator.onLine) {
        console.log('Light polling for anomaly data');
        fetchData(true);
      }
    }, 10000); // 10s light polling as additional safety

    return () => {
      controller.abort();
      if (eventSource) eventSource.close();
      if (pollingInterval) clearInterval(pollingInterval);
      clearInterval(lightPolling);
    };
  }, [fetchData]);

  return {
    data,
    error,
    isLoading,
    refetch: () => fetchData(true)
  };
}; 