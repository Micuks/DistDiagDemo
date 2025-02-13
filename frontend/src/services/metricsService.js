import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://10.101.168.97:8001';

export const fetchMetrics = async () => {
    try {
        const response = await axios.get(`${API_BASE_URL}/api/metrics`);
        return response.data;
    } catch (error) {
        console.error('Error fetching metrics:', error.response || error);
        throw error;
    }
};

export const fetchDetailedMetrics = async (nodeIp, category) => {
    try {
        const response = await axios.get(`${API_BASE_URL}/api/metrics/detailed`, {
            params: { node_ip: nodeIp, category }
        });
        return response.data;
    } catch (error) {
        console.error('Error fetching detailed metrics:', error.response || error);
        throw error;
    }
}; 