import { useState, useEffect, useCallback, useRef } from 'react';
import { anomalyService } from '../services/anomalyService';

const cache = new Map();
const STALE_TIME = 30000; // Increase stale time to 30 seconds
const MAX_RETRIES = 2; // Reduce max retries to avoid hammering the server
const REQUEST_DEBOUNCE = 500; // Minimum time between requests in ms

export const useAnomalyData = () => {
  const [data, setData] = useState(cache.get('anomalies')?.data || []);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(!data.length);
  const eventSourceRef = useRef(null);
  const pollingIntervalRef = useRef(null);
  const lastRequestTimeRef = useRef(0);
  const isSSEWorkingRef = useRef(false);
  const pendingRequestRef = useRef(null);

  const fetchData = useCallback(async (force = false) => {
    const now = Date.now();
    
    // Debounce requests - don't allow more than one request per REQUEST_DEBOUNCE ms
    if (now - lastRequestTimeRef.current < REQUEST_DEBOUNCE) {
      // If there's already a pending request, just return its promise
      if (pendingRequestRef.current) {
        return pendingRequestRef.current;
      }
      
      // Schedule a single fetch after the debounce period
      if (!force) {
        return new Promise(resolve => {
          setTimeout(() => {
            pendingRequestRef.current = fetchData(force);
            pendingRequestRef.current.then(resolve);
          }, REQUEST_DEBOUNCE - (now - lastRequestTimeRef.current));
        });
      }
    }
    
    // Set last request time
    lastRequestTimeRef.current = now;
    
    // Create a new pending request
    const fetchPromise = (async () => {
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
          cache.set('anomalies', { ...(cache.get('anomalies') || {}), errorCount });
          // Exponential backoff
          const delay = Math.min(1000 * Math.pow(2, errorCount - 1), 10000);
          await new Promise(resolve => setTimeout(resolve, delay));
          return fetchData(force);
        }
        setError(err);
        // Continue using previous data instead of setting to empty
        if (cached?.data) {
          setData(cached.data);
          return cached.data;
        } else {
          setData([]); // Set empty array if no cached data
          return [];
        }
      } finally {
        setIsLoading(false);
      }
    })();
    
    pendingRequestRef.current = fetchPromise;
    
    // Clear pending request ref when done
    fetchPromise.finally(() => {
      if (pendingRequestRef.current === fetchPromise) {
        pendingRequestRef.current = null;
      }
    });
    
    return fetchPromise;
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
    };
    
    cleanupConnections();
    
    // Initial fetch
    fetchData(true);

    // Set up SSE connection
    const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
    const setupEventSource = () => {
      try {
        const eventSource = new EventSource(`${API_BASE_URL}/api/anomaly/stream`);
        eventSourceRef.current = eventSource;
        
        // Track successful message receipt to know if SSE is working
        let receivedMessage = false;
        
        eventSource.onmessage = (event) => {
          try {
            receivedMessage = true;
            isSSEWorkingRef.current = true;
            
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
            receivedMessage = true;
            isSSEWorkingRef.current = true;
            
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
          isSSEWorkingRef.current = false;
          
          // Close the current connection
          if (eventSourceRef.current) {
            eventSourceRef.current.close();
            eventSourceRef.current = null;
          }
          
          // Start polling as fallback only if SSE isn't working
          if (!pollingIntervalRef.current) {
            pollingIntervalRef.current = setInterval(() => {
              if (!document.hidden && navigator.onLine) {
                fetchData(true);
              }
            }, 10000); // 10s fallback polling (increased from 5s)
          }
          
          // Try to reconnect SSE after a delay
          setTimeout(() => {
            if (!document.hidden && navigator.onLine) {
              setupEventSource();
            }
          }, 8000); // Increased reconnect delay
        };
        
        // Check if we received any messages after some time
        setTimeout(() => {
          if (!receivedMessage && eventSourceRef.current) {
            console.warn("SSE connection established but no messages received");
            isSSEWorkingRef.current = false;
            
            // Start polling as fallback
            if (!pollingIntervalRef.current) {
              pollingIntervalRef.current = setInterval(() => {
                if (!document.hidden && navigator.onLine) {
                  fetchData(true);
                }
              }, 10000); // 10s fallback polling
            }
          }
        }, 10000);
        
        return eventSource;
      } catch (error) {
        console.error('Failed to setup EventSource:', error);
        isSSEWorkingRef.current = false;
        return null;
      }
    };
    
    setupEventSource();

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