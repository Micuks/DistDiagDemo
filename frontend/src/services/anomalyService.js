import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const anomalyService = {
  startAnomaly: async (anomalyType) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/anomaly/start`, {
        type: anomalyType
      });
      return response.data;
    } catch (error) {
      console.error('Error starting anomaly:', error);
      throw error;
    }
  },

  stopAnomaly: async () => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/anomaly/stop`);
      return response.data;
    } catch (error) {
      console.error('Error stopping anomaly:', error);
      throw error;
    }
  },

  getMetrics: async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/metrics`);
      return response.data;
    } catch (error) {
      console.error('Error fetching metrics:', error);
      throw error;
    }
  },

  getAnomalyRanks: async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/anomaly/ranks`);
      return response.data;
    } catch (error) {
      console.error('Error fetching anomaly ranks:', error);
      throw error;
    }
  }
}; 