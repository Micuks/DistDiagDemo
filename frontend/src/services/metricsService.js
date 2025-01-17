import api from './api';

export const fetchMetrics = async () => {
    try {
        const response = await api.get('/metrics');
        return response.data;
    } catch (error) {
        console.error('Error fetching metrics:', error);
        throw error;
    }
};

export const startMetricsCollection = async () => {
    try {
        const response = await api.post('/metrics/start');
        return response.data;
    } catch (error) {
        console.error('Error starting metrics collection:', error);
        throw error;
    }
};

export const stopMetricsCollection = async () => {
    try {
        const response = await api.post('/metrics/stop');
        return response.data;
    } catch (error) {
        console.error('Error stopping metrics collection:', error);
        throw error;
    }
}; 