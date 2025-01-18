import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://10.101.168.97:8001';

export const fetchMetrics = async () => {
    try {
        console.log('Fetching metrics from:', `${API_BASE_URL}/api/metrics`);
        const response = await axios.get(`${API_BASE_URL}/api/metrics`);
        console.log('Raw response:', response);
        return response.data;
    } catch (error) {
        console.error('Error fetching metrics:', error.response || error);
        throw error;
    }
};

export const startMetricsCollection = async () => {
    try {
        const response = await axios.post(`${API_BASE_URL}/api/metrics/start`);
        return response.data;
    } catch (error) {
        console.error('Error starting metrics collection:', error);
        throw error;
    }
};

export const stopMetricsCollection = async () => {
    try {
        const response = await axios.post(`${API_BASE_URL}/api/metrics/stop`);
        return response.data;
    } catch (error) {
        console.error('Error stopping metrics collection:', error);
        throw error;
    }
}; 