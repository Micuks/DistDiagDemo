import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://10.101.168.97:8001';

export const anomalyService = {
  startAnomaly: async (anomalyType, collectTrainingData = false) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/anomaly/inject`, {
        type: anomalyType,
        collect_training_data: collectTrainingData
      });
      return response.data;
    } catch (error) {
      console.error('Error injecting anomaly:', error);
      throw error;
    }
  },

  stopAnomaly: async (anomalyType, collectTrainingData = false) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/anomaly/clear`, {
        type: anomalyType,
        collect_training_data: collectTrainingData
      });
      return response.data;
    } catch (error) {
      console.error('Error clearing anomaly:', error);
      throw error;
    }
  },

  stopAllAnomalies: async () => {
    try {
      // Get active anomalies first
      const activeAnomalies = await anomalyService.getActiveAnomalies();
      // Clear each anomaly
      await Promise.all(activeAnomalies.map(anomaly => 
        anomalyService.stopAnomaly(anomaly.type)
      ));
      return { status: 'success', message: 'All anomalies cleared' };
    } catch (error) {
      console.error('Error clearing all anomalies:', error);
      throw error;
    }
  },

  getActiveAnomalies: async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/anomaly/active`);
      return response.data;
    } catch (error) {
      console.error('Error fetching active anomalies:', error);
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
  },

  startNormalCollection: async () => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/anomaly/normal/start`);
      return response.data;
    } catch (error) {
      console.error('Error starting normal state collection:', error);
      throw error;
    }
  },

  stopNormalCollection: async () => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/anomaly/normal/stop`);
      return response.data;
    } catch (error) {
      console.error('Error stopping normal state collection:', error);
      throw error;
    }
  },

  getTrainingStats: async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/anomaly/training/stats`);
      return response.data;
    } catch (error) {
      console.error('Error fetching training stats:', error);
      throw error;
    }
  },

  trainModel: async () => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/anomaly/train`);
      return response.data;
    } catch (error) {
      console.error('Error training model:', error);
      throw error;
    }
  }
}; 