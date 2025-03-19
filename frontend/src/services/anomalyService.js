import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
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

    async getActiveAnomalies() {
        console.log('Fetching active anomalies...');
        try {
            // Add timestamp to prevent caching
            const timestamp = new Date().getTime();
            const response = await this._retryableRequest(() => 
                this.client.get(`/api/anomaly/active?_=${timestamp}`, {
                    headers: {
                        'Cache-Control': 'no-cache, no-store, must-revalidate',
                        'Pragma': 'no-cache',
                        'Expires': '0'
                    }
                })
            );
            
            console.log('Active anomalies response:', response);
            
            // Ensure we return an array even if the response is null or not an array
            if (!response) {
                console.warn('Received null response from active anomalies endpoint');
                return [];
            }
            
            if (!Array.isArray(response)) {
                console.warn('Response is not an array:', response);
                return Array.isArray(response.data) ? response.data : [];
            }
            
            return response;
        } catch (error) {
            console.error('Error fetching active anomalies:', error);
            // Return empty array on error rather than throwing
            return [];
        }
    }

    async startAnomaly(type, node_or_params = {}) {
        // Handle both direct node parameter or a params object
        let node, collect_training_data;
        
        if (typeof node_or_params === 'string' || Array.isArray(node_or_params)) {
            // If second parameter is a string or array, it's the node
            node = node_or_params;
            collect_training_data = false;
        } else {
            // Otherwise it's a params object
            node = node_or_params.node || null;
            collect_training_data = node_or_params.collect_training_data || false;
        }
        
        return this._retryableRequest(() =>
            this.client.post('/api/anomaly/inject', {
                type: type,
                node: node,
                target_node: node, // Also send as target_node for schema compatibility
                collect_training_data: collect_training_data
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
        console.log('Stopping all anomalies...');
        try {
            // Get active anomalies first
            const activeAnomalies = await this.getActiveAnomalies();
            console.log('Active anomalies to stop:', activeAnomalies);
            
            if (!activeAnomalies || activeAnomalies.length === 0) {
                console.log('No active anomalies to stop');
                return { status: "success", message: "No active anomalies to stop" };
            }

            // Stop each anomaly with individual error handling
            const results = [];
            for (const anomaly of activeAnomalies) {
                try {
                    console.log(`Stopping anomaly: ${anomaly.type} (${anomaly.name})`);
                    const result = await this.stopAnomaly(anomaly.type);
                    results.push({ 
                        type: anomaly.type, 
                        name: anomaly.name,
                        success: true, 
                        message: result.message || "Stopped successfully" 
                    });
                } catch (error) {
                    console.error(`Failed to stop anomaly ${anomaly.type}:`, error);
                    results.push({ 
                        type: anomaly.type, 
                        name: anomaly.name,
                        success: false, 
                        error: error.message 
                    });
                }
                
                // Short delay between stops to avoid race conditions
                await new Promise(resolve => setTimeout(resolve, 500));
            }
            
            // Wait a moment for backend to process all deletions
            await new Promise(resolve => setTimeout(resolve, 1000));
            
            // Final verification - check if any anomalies are still active
            const remainingAnomalies = await this.getActiveAnomalies();
            if (remainingAnomalies && remainingAnomalies.length > 0) {
                console.warn('Some anomalies still active after stopAll:', remainingAnomalies);
                
                // Try once more to stop remaining anomalies
                for (const anomaly of remainingAnomalies) {
                    try {
                        console.log(`Second attempt to stop anomaly: ${anomaly.type} (${anomaly.name})`);
                        await this.stopAnomaly(anomaly.type);
                    } catch (error) {
                        console.error(`Failed second attempt to stop anomaly ${anomaly.type}:`, error);
                    }
                }
            }
            
            // One final check with force cache refresh
            const finalCheck = await this._retryableRequest(() => 
                this.client.get('/api/anomaly/active', {
                    headers: {
                        'Cache-Control': 'no-cache, no-store, must-revalidate',
                        'Pragma': 'no-cache',
                        'Expires': '0'
                    },
                    params: { _: Date.now() } // Add timestamp to bypass cache
                })
            );
            
            console.log('Final anomaly check after stopAll:', finalCheck);
            
            return {
                status: "success", 
                message: "All anomalies stopped",
                results: results
            };
        } catch (error) {
            console.error('Error in stopAllAnomalies:', error);
            throw error;
        }
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

    async validateModel(modelName) {
        return this._retryableRequest(() =>
            this.client.post('/api/models/validate', {
                model_name: modelName
            })
        ).catch(error => {
            console.error(`Model validation failed for ${modelName}:`, error);
            return { valid: false, error: error.message };
        });
    }

    async getModelPerformance(modelName) {
        return this._retryableRequest(() =>
            this.client.get(`/api/models/${encodeURIComponent(modelName)}/performance`)
        );
    }

    async getModelDiagnosis(modelName, threshold = 0.001) {
        return this._retryableRequest(() =>
            this.client.get('/api/models/ranks', {
                params: {
                    model_names: [modelName],  // Send as array even for single model
                    threshold: threshold       // Add threshold parameter
                }
            })
        );
    }

    async compareModels(modelNames, threshold = 0.001) {
        return this._retryableRequest(() => 
            this.client.get('/api/models/ranks', {
                params: {
                    model_names: modelNames,
                    threshold: threshold       // Add threshold parameter
                },
            })
        );
    }

    async startAnomalyCollection(type, node) {
        return this._retryableRequest(() =>
            this.client.post('/api/training/collect', {
                type: type,
                node: node || null
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

    async getAnomalyStatus(anomalyId) {
        return this._retryableRequest(() => 
            this.client.get(`/anomalies/${anomalyId}/status`)
        );
    }

    async getMetricsFluctuations() {
        try {
            // Add timestamp to prevent caching
            const timestamp = new Date().getTime();
            return await this._retryableRequest(() => 
                this.client.get(`/api/metrics/fluctuations?_=${timestamp}`, {
                    headers: {
                        'Cache-Control': 'no-cache, no-store, must-revalidate',
                        'Pragma': 'no-cache',
                        'Expires': '0'
                    }
                })
            );
        } catch (error) {
            console.error('Error fetching metrics fluctuations:', error);
            // Return empty object on error rather than throwing
            return { 
                metrics: {},
                summary: {
                    total_metrics: 0,
                    fluctuating_metrics: 0,
                    has_fluctuations: false
                }
            };
        }
    }

    async getAvailableNodes() {
        try {
            const response = await this.client.get('/api/anomaly/nodes');
            return response;
        } catch (error) {
            console.error('Error fetching available nodes:', error);
            return [];
        }
    }

    async injectAnomaly(anomalyType, targetNode, severity = 5, duration = null) {
        try {
            const response = await this.client.post('/api/anomaly/inject', {
                type: anomalyType,
                target_node: targetNode,
                severity: severity,
                duration: duration
            });
            return response.data;
        } catch (error) {
            console.error('Error injecting anomaly:', error);
            throw new Error('Failed to inject anomaly: ' + (error.response?.data?.detail || error.message));
        }
    }

    async stopAnomaly(anomalyId) {
        try {
            const response = await this.client.post(`/api/anomaly/${anomalyId}/stop`);
            return response.data;
        } catch (error) {
            console.error('Error stopping anomaly:', error);
            throw new Error('Failed to stop anomaly: ' + (error.response?.data?.detail || error.message));
        }
    }

    async stopAllAnomalies() {
        try {
            const response = await this.client.post('/api/anomaly/stop-all');
            return response.data;
        } catch (error) {
            console.error('Error stopping all anomalies:', error);
            throw new Error('Failed to stop all anomalies: ' + (error.response?.data?.detail || error.message));
        }
    }

    async getAnomalyStatus(anomalyId) {
        try {
            const response = await this.client.get(`/api/anomaly/${anomalyId}/status`);
            return response.data;
        } catch (error) {
            console.error('Error getting anomaly status:', error);
            throw new Error('Failed to get anomaly status: ' + (error.response?.data?.detail || error.message));
        }
    }

    getAnomalyTypes() {
        return [
            {
                key: 'cpu_stress',
                name: 'CPU Stress',
                description: 'Simulate high CPU usage'
            },
            {
                key: 'io_bottleneck',
                name: 'I/O Bottleneck',
                description: 'Simulate slow disk I/O operations'
            },
            {
                key: 'network_bottleneck',
                name: 'Network Bottleneck',
                description: 'Simulate network congestion and high latency'
            },
            {
                key: 'memory_leak',
                name: 'Memory Leak',
                description: 'Simulate gradual memory consumption'
            },
            {
                key: 'too_many_indexes',
                name: 'Too Many Indexes',
                description: 'Create unnecessary database indexes'
            }
        ];
    }
}

export const anomalyService = new AnomalyService();