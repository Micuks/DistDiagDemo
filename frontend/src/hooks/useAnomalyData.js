import { useState, useEffect, useCallback } from 'react';
import { anomalyService } from '../services/anomalyService';

const cache = new Map();
const STALE_TIME = 5000; // 5 seconds
const MAX_RETRIES = 3;

export const useAnomalyData = () => {
  const [data, setData] = useState(cache.get('anomalies')?.data || null);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(!data);

  const fetchData = useCallback(async (force = false) => {
    const now = Date.now();
    const cached = cache.get('anomalies');
    
    // Return cached data if fresh enough
    if (!force && cached && now - cached.timestamp < STALE_TIME) {
      return cached.data;
    }

    try {
      setIsLoading(true);
      const freshData = await anomalyService.getActiveAnomalies();
      
      cache.set('anomalies', {
        data: freshData,
        timestamp: now,
        errorCount: 0
      });
      
      setData(freshData);
      setError(null);
      return freshData;
    } catch (err) {
      const errorCount = (cache.get('anomalies')?.errorCount || 0) + 1;
      if (errorCount <= MAX_RETRIES) {
        cache.set('anomalies', { ...cache.get('anomalies'), errorCount });
        // Exponential backoff
        const delay = Math.min(1000 * Math.pow(2, errorCount - 1), 30000);
        await new Promise(resolve => setTimeout(resolve, delay));
        return fetchData(force);
      }
      setError(err);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    
    // Initial fetch
    fetchData();

    // Set up SSE connection
    const eventSource = new EventSource('/api/anomaly/stream');
    
    eventSource.onmessage = (event) => {
      const freshData = JSON.parse(event.data);
      cache.set('anomalies', {
        data: freshData,
        timestamp: Date.now(),
        errorCount: 0
      });
      setData(freshData);
    };

    eventSource.onerror = () => {
      // If SSE fails, fall back to polling
      const interval = setInterval(() => {
        if (!document.hidden && navigator.onLine) {
          fetchData(true);
        }
      }, 30000); // 30s fallback polling
      
      return () => {
        clearInterval(interval);
      };
    };

    return () => {
      controller.abort();
      eventSource.close();
    };
  }, [fetchData]);

  return {
    data,
    error,
    isLoading,
    refetch: () => fetchData(true)
  };
}; 