import axios from 'axios';

const API_BASE_URL = import.meta.env.REACT_APP_API_BASE_URL || 'http://10.101.168.97:8001';
const MAX_RETRIES = 3;

class AnomalyService {
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
            throw new Error(error.response.data.detail || 'Server error');
        } else if (error.request) {
            // Request made but no response
            throw new Error('No response from server');
        } else {
            // Request setup error
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

    async getActiveAnomalies() {
        return this._retryableRequest(() => 
            this.client.get('/api/anomaly/active')
        );
    }

    async startAnomaly(type, params = {}) {
        return this._retryableRequest(() => 
            this.client.post('/api/anomaly/inject', {
                type: type,
                node: params.node || null,
                collect_training_data: params.collect_training_data || false
            })
        );
    }

    async stopAnomaly(type) {
        return this._retryableRequest(() => 
            this.client.post('/api/anomaly/clear', {
                type: type,
                collect_training_data: false
            })
        );
    }

    async stopAllAnomalies() {
        // Get active anomalies first
        const activeAnomalies = await this.getActiveAnomalies();
        // Stop each anomaly
        for (const anomaly of activeAnomalies) {
            await this.stopAnomaly(anomaly.type);
        }
        return { status: "success", message: "All anomalies stopped" };
    }

    async getTrainingStats() {
        return this._retryableRequest(() => 
            this.client.get('/api/anomaly/training/stats')
        );
    }

    async startTrainingDataCollection() {
        return this._retryableRequest(() => 
            this.client.post('/api/anomaly/training/start')
        );
    }

    async stopTrainingDataCollection() {
        return this._retryableRequest(() => 
            this.client.post('/api/anomaly/training/stop')
        );
    }

    async getAnomalyRanks() {
        return this._retryableRequest(() => 
            this.client.get('/api/anomaly/ranks')
        );
    }

    async getCollectionStatus() {
        return this._retryableRequest(() => 
            this.client.get('/api/anomaly/collection-status')
        );
    }

    async startNormalCollection() {
        return this._retryableRequest(() => 
            this.client.post('/api/anomaly/normal/start')
        );
    }

    async stopNormalCollection() {
        return this._retryableRequest(() => 
            this.client.post('/api/anomaly/normal/stop')
        );
    }

    async getDetailedMetrics(metricType, duration) {
        return this._retryableRequest(() => 
            this.client.post('/api/metrics/detailed', {
                metric_type: metricType,
                duration: duration
            })
        ).then(response => response.data);
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

    async compareModels(modelNames) {
        return this._retryableRequest(() => 
            this.client.post('/api/models/compare', {
                models: modelNames
            })
        );
    }

    async startAnomalyCollection(type, node, options = {}) {
        return this._retryableRequest(() => 
            this.client.post('/api/training/collect', {
                type: type,
                node: node || null,
                pre_collect: options.preCollect,
                post_collect: options.postCollect
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
}

export const anomalyService = new AnomalyService(); 