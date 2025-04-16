import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const WS_BASE_URL = import.meta.env.VITE_WS_URL || API_BASE_URL.replace('http://', 'ws://').replace('https://', 'wss://');
const MAX_RETRIES = 3;

// Request tracking for deduplication
const inFlightRequests = new Map();
const requestCache = new Map();
const CACHE_TTL = 10000; // 10 seconds TTL for cached responses

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

        // WebSocket related properties
        this.ws = null;
        this.wsConnected = false;
        this.wsCallbacks = {
            onMessage: () => {},
            onConnect: () => {},
            onDisconnect: () => {},
            onError: () => {}
        };
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectTimeout = null;
    }

    // WebSocket Methods
    connectToWebSocket(callbacks = {}) {
        // Merge provided callbacks with defaults
        this.wsCallbacks = { ...this.wsCallbacks, ...callbacks };
        
        if (this.ws) {
            console.log('WebSocket connection already exists');
            return;
        }

        try {
            this.ws = new WebSocket(`${WS_BASE_URL}/api/ws/anomalies`);
            
            this.ws.onopen = () => {
                console.log('WebSocket connection established');
                this.wsConnected = true;
                this.reconnectAttempts = 0;
                this.wsCallbacks.onConnect();
            };
            
            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.wsCallbacks.onMessage(data);
                } catch (error) {
                    console.error('Failed to parse WebSocket message:', error);
                }
            };
            
            this.ws.onclose = (event) => {
                console.log('WebSocket connection closed:', event.code, event.reason);
                this.wsConnected = false;
                this.ws = null;
                this.wsCallbacks.onDisconnect(event);
                this._attemptReconnect();
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.wsCallbacks.onError(error);
            };
        } catch (error) {
            console.error('Failed to connect to WebSocket:', error);
            this.wsCallbacks.onError(error);
            this._attemptReconnect();
        }
    }

    disconnectFromWebSocket() {
        if (this.ws) {
            console.log('Closing WebSocket connection');
            this.ws.close();
            this.ws = null;
            this.wsConnected = false;
            
            // Clear any pending reconnect attempts
            if (this.reconnectTimeout) {
                clearTimeout(this.reconnectTimeout);
                this.reconnectTimeout = null;
            }
        }
    }

    sendWebSocketMessage(message) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            console.error('WebSocket not connected, cannot send message');
            return false;
        }
        
        try {
            this.ws.send(JSON.stringify(message));
            return true;
        } catch (error) {
            console.error('Failed to send WebSocket message:', error);
            return false;
        }
    }

    refreshAnomalies() {
        return this.sendWebSocketMessage({ action: 'refresh' });
    }

    _attemptReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.log('Maximum reconnect attempts reached');
            return;
        }
        
        const backoffTime = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
        console.log(`Attempting to reconnect in ${backoffTime}ms (attempt ${this.reconnectAttempts + 1}/${this.maxReconnectAttempts})`);
        
        this.reconnectTimeout = setTimeout(() => {
            this.reconnectAttempts++;
            this.connectToWebSocket(this.wsCallbacks);
        }, backoffTime);
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

    // Helper to deduplicate requests
    async _deduplicatedRequest(cacheKey, request) {
        // If there's an in-flight request for this key, return its promise
        if (inFlightRequests.has(cacheKey)) {
            return inFlightRequests.get(cacheKey);
        }
        
        // Check for a valid cached response
        const cachedItem = requestCache.get(cacheKey);
        if (cachedItem && Date.now() - cachedItem.timestamp < CACHE_TTL) {
            return cachedItem.data;
        }
        
        // Create a new request and store it
        const requestPromise = (async () => {
            try {
                const response = await this._retryableRequest(request);
                
                // Store in cache
                requestCache.set(cacheKey, {
                    data: response,
                    timestamp: Date.now()
                });
                
                return response;
            } finally {
                // Remove from in-flight requests when done
                inFlightRequests.delete(cacheKey);
            }
        })();
        
        // Store the in-flight request
        inFlightRequests.set(cacheKey, requestPromise);
        
        return requestPromise;
    }

    async getActiveAnomalies() {
        console.log('Fetching active anomalies...');
        try {
            // Add timestamp to prevent caching
            const timestamp = new Date().getTime();
            const cacheKey = 'getActiveAnomalies';
            
            const response = await this._deduplicatedRequest(cacheKey, () => 
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
            
            // Check if the response has an 'anomalies' property that is an array
            if (response && Array.isArray(response.anomalies)) {
                return response.anomalies;
            }
            
            // Log a warning if the structure is unexpected
            console.warn('Response structure unexpected or missing anomalies array:', response);
            return []; // Return empty array if anomalies property is not a valid array
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

    async stopAnomaly(typeOrId, experimentName = null) {
        let payload = {
            collect_training_data: false // Default, not usually needed for stop
        };

        if (typeof typeOrId === 'string') {
            // Assume it's the anomaly ID (which corresponds to experiment_name in backend)
            payload.experiment_name = typeOrId;
            // Optionally pass experimentName as well if provided, backend prioritizes experiment_name
            if (experimentName) {
                payload.type = experimentName; // If experimentName is actually the type
            }
            console.log(`Stopping anomaly by ID: ${typeOrId}`);
        } else if (typeof typeOrId === 'object' && typeOrId !== null) {
            // Handle object input { type: '...', name: '...' }
            payload.type = typeOrId.type;
            payload.experiment_name = typeOrId.name || experimentName; // Use name from object or fallback
            console.log(`Stopping anomaly by type/name: type=${payload.type}, name=${payload.experiment_name}`);
        } else {
            console.error("Invalid input to stopAnomaly:", typeOrId);
            throw new Error("Invalid input: Must provide anomaly ID string or type/name object.");
        }

        return this._retryableRequest(() =>
            this.client.post('/api/anomaly/clear', payload)
        );
    }

    async stopAllAnomalies() {
        console.log('Stopping all anomalies...');
        try {
            // First try using the dedicated endpoint
            try {
                const response = await this.client.post('/api/anomaly/stop-all');
                console.log('Used dedicated endpoint to stop all anomalies:', response.data);
                return response.data;
            } catch (endpointError) {
                console.warn('Failed to use dedicated endpoint to stop anomalies, falling back to individual stops:', endpointError);
                // If the new endpoint fails, fall back to stopping each anomaly individually
            }
            
            // Fallback: Get active anomalies and stop them one by one
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
                    // Use both type and name for more precise deletion
                    const result = await this.stopAnomaly(anomaly.type, anomaly.name);
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

    async getCompoundAnomalies() {
        return this._retryableRequest(() =>
            this.client.get('/api/anomaly/compound')
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

    async getMetricRankings(node, modelName = null) {
        return this._retryableRequest(() => 
            this.client.get('/api/models/metrics_ranks', {
                params: {
                    node: node,
                    model_name: modelName
                },
            })
        );
    }

    async getMetricSummary(node, languages = ["Chinese", "English"]) {
        return this._retryableRequest(() => 
            this.client.get('/api/models/metrics_summary', {
                params: {
                    node: node,
                    languages: languages
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

    async getAvailableAnomalyTypes() {
        try {
            // Endpoint now returns List[AnomalyTypeInfo]
            const response = await this.client.get('/api/anomaly/types'); 
            return response;
        } catch (error) {
            console.error('Error fetching available anomaly types:', error);
            // Return default types as fallback (now needs description)
            return [
              { type: 'cpu_stress', description: '(Fallback) CPU Stress' },
              { type: 'network_bottleneck', description: '(Fallback) Network Bottleneck' },
              { type: 'io_bottleneck', description: '(Fallback) IO Bottleneck' },
            ];
        }
    }
}

/**
 * Creates and configures an EventSource for real-time anomaly updates
 * @returns {EventSource} EventSource object connected to the anomaly stream endpoint
 */
const createAnomalyEventSource = () => {
  const API_URL = import.meta.env.VITE_API_URL || '';
  const url = `${API_URL}/api/anomaly/stream`;
  
  try {
    const eventSource = new EventSource(url);
    console.log('Created EventSource connection to anomaly stream');
    return eventSource;
  } catch (error) {
    console.error('Failed to create EventSource connection:', error);
    throw error;
  }
};

export const anomalyService = new AnomalyService();
export { createAnomalyEventSource };