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

/**
 * @deprecated Use fetchAllDetailedMetrics instead which is more efficient
 * Fetches detailed metrics for a specific node and category
 */
export const fetchDetailedMetrics = async (nodeIp, category) => {
    console.warn('fetchDetailedMetrics is deprecated. Use fetchAllDetailedMetrics instead.');
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

// Optimized function to batch fetch detailed metrics for a node, only for the selected metrics
export const fetchAllDetailedMetrics = async (nodeIp, selectedMetrics) => {
    try {
        const response = await axios.get(`${API_BASE_URL}/api/metrics/detailed/selected`, {
            params: { 
                node_ip: nodeIp,
                // Convert the selectedMetrics object to a format the server can understand
                metrics: JSON.stringify(selectedMetrics)
            }
        });
        return response.data;
    } catch (error) {
        console.error('Error fetching all detailed metrics:', error.response || error);
        throw error;
    }
}; 