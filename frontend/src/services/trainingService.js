import axios from 'axios';

const API_BASE_URL = import.meta.env.REACT_APP_API_BASE_URL || 'http://10.101.168.97:8001';
const MAX_RETRIES = 3;

class TrainingService {
    constructor() {
        this.client = axios.create({
            baseURL: API_BASE_URL,
            timeout: 30000, // 30s timeout
        });
        
        // Add response interceptor for error handling
        this.client.interceptors.response.use(
            response => response.data,
            error => this._handleError(error)
        );
    }

    async _handleError(error) {
        if (error.response) {
            // Server responded with error
            console.error('Server error:', error.response.data);
            throw new Error(error.response.data.detail || 'Server error');
        } else if (error.request) {
            // Request made but no response
            console.error('No response from server:', error.request);
            throw new Error('No response from server');
        } else {
            // Request setup error
            console.error('Request setup error:', error.message);
            throw new Error('Failed to make request');
        }
    }

    async _retryableRequest(request, retries = 0) {
        try {
            return await request();
        } catch (error) {
            if (retries < MAX_RETRIES) {
                // Exponential backoff
                const delay = Math.min(1000 * Math.pow(2, retries), 10000);
                await new Promise(resolve => setTimeout(resolve, delay));
                return this._retryableRequest(request, retries + 1);
            }
            throw error;
        }
    }

    async getTrainingStats() {
        return this._retryableRequest(() => 
            this.client.get('/api/training/stats')
        );
    }

    async startAnomalyCollection(type, node, options = {}) {
        return this._retryableRequest(() => 
            this.client.post('/api/training/collect', {
                type: type,
                node: node || null,
            })
        );
    }

    async stopAnomalyCollection(savePostData = true) {
        return this._retryableRequest(() => 
            this.client.post('/api/training/stop', {
                save_post_data: savePostData
            })
        );
    }

    async startNormalCollection() {
        return this._retryableRequest(() => 
            this.client.post('/api/training/normal/start')
        );
    }

    async stopNormalCollection() {
        return this._retryableRequest(() => 
            this.client.post('/api/training/normal/stop')
        );
    }

    async getCollectionStatus() {
        return this._retryableRequest(() => 
            this.client.get('/api/training/collection-status')
        );
    }

    async getAvailableModels() {
        return this._retryableRequest(() =>
            this.client.get('/api/models/list')
        );
    }

    async getModelPerformance(modelName) {
        return this._retryableRequest(() =>
            this.client.get(`/api/models/${encodeURIComponent(modelName)}/performance`)
        );
    }

    async autoBalanceDataset() {
        return this._retryableRequest(() => 
            this.client.post('/api/training/auto_balance')
        );
    }

    async trainModel() {
        return this._retryableRequest(() => 
            this.client.post('/api/training/train')
        );
    }
}

export const trainingService = new TrainingService(); 