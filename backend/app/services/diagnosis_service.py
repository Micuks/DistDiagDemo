import sys
from pathlib import Path
import os
import logging
import numpy as np
from typing import List, Dict, Any
from datetime import datetime
from scipy.stats import pearsonr
from xgboost import XGBClassifier
import joblib
from app.services.metrics_service import metrics_service
import json
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import glob
import traceback

# Add project root and dist_diagnosis root to Python path
project_root = Path(__file__).parent.parent.parent.parent
dist_diagnosis_root = project_root / 'dist_diagnosis'
sys.path.extend([str(project_root), str(dist_diagnosis_root)])

logger = logging.getLogger(__name__)

class WPRNNode:
    """PageRank implementation for N nodes"""
    def __init__(self, w, node_num):
        self.w = w
        self.node_num = node_num
        self.d = 0.85

    def page_rank(self, iter_num):
        pr = [1.0 / self.node_num] * self.node_num
        
        for _ in range(iter_num):
            next_pr = [(1 - self.d) / self.node_num] * self.node_num
            
            for i in range(self.node_num):
                for j in range(self.node_num):
                    if i != j:
                        next_pr[i] += self.d * pr[j] * self.w[j][i]
            
            pr = next_pr
            
        return pr

class DistDiagnosis:
    """Integrated version of DistDiagnosis for anomaly detection"""
    def __init__(self, num_classifier=1, classes_for_each_node=None, node_num=1):
        self.num_classifier = num_classifier
        self.node_num = node_num
        self.classes = classes_for_each_node if classes_for_each_node else [
            "cpu_stress",
            "io_bottleneck", 
            "network_bottleneck",
            "cache_bottleneck",
            "too_many_indexes"
        ]
        # Enable GPU acceleration with CUDA
        self.clf_list = [XGBClassifier(
            objective='binary:logistic',
            tree_method='gpu_hist',  # Use GPU for histogram calculation
            gpu_id=0,               # Use first GPU
            predictor='gpu_predictor'  # Use GPU for prediction
        ) for _ in range(num_classifier)]
        self.time_window = 5  # Default 5-minute window
        logger.info(f"Initialized {len(self.clf_list)} classifiers for {len(self.classes)} classes")

    def _get_time_window_values(self, metrics: Dict[str, Any], metric_name: str) -> List[float]:
        """Extract time window values for a metric"""
        try:
            values = []
            if not metrics:
                return [0.0] * self.time_window

            # Check if metric exists directly in metrics
            if metric_name in metrics:
                metric_data = metrics[metric_name]
                
                # Handle case where metric_data is a direct float value
                if isinstance(metric_data, (int, float)):
                    return [float(metric_data)] * self.time_window
                    
                # Handle case where metric_data is a dict with 'latest' key
                elif isinstance(metric_data, dict) and 'latest' in metric_data:
                    return [float(metric_data['latest'])] * self.time_window
                    
                # Handle case where metric_data is a list of dictionaries with 'timestamp' and 'value' keys
                elif isinstance(metric_data, list) and metric_data and isinstance(metric_data[0], dict) and 'timestamp' in metric_data[0] and 'value' in metric_data[0]:
                    # Sort by timestamp (newest first) and take up to time_window values
                    sorted_data = sorted(metric_data, key=lambda x: x['timestamp'], reverse=True)[:self.time_window]
                    values = [float(item['value']) for item in sorted_data]
                    # Pad with zeros if we don't have enough values
                    while len(values) < self.time_window:
                        values.append(0.0)
                    return values
                else:
                    logger.warning(f"Unexpected metric data type for {metric_name}: {type(metric_data)}")
                    return [0.0] * self.time_window
            else:
                # Metric not found in the metrics dictionary
                logger.warning(f"Metric {metric_name} not found in metrics")
                return [0.0] * self.time_window

            # Pad with zeros if we don't have enough values (this is now unreachable but kept for safety)
            while len(values) < self.time_window:
                values.append(0.0)

            return values[:self.time_window]  # Ensure we only return time_window values
        except Exception as e:
            logger.error(f"Error getting time window values for {metric_name}: {str(e)}")
            return [0.0] * self.time_window

    def _calculate_time_features(self, values: List[float]) -> List[float]:
        """Calculate statistical features over time window values"""
        try:
            # Convert to numpy array for vectorized operations
            values_array = np.array(values, dtype=np.float64)
            
            # Calculate features in a vectorized way
            features = [
                np.mean(values_array),                         # mean
                np.std(values_array) if len(values_array) > 1 else 0.0,  # std dev
                np.max(values_array) if values_array.size > 0 else 0.0,  # max
                np.min(values_array) if values_array.size > 0 else 0.0,  # min
                np.median(values_array) if values_array.size > 0 else 0.0 # median
            ]
                
            return features
        except Exception as e:
            logger.error(f"Error calculating time features: {str(e)}")
            return [0.0] * 5  # Return zeros for all features

    def _process_metrics(self, metrics: Dict[str, Any]) -> np.ndarray:
        """Process metrics into feature vector using time window statistics"""
        try:
            # Pre-allocate the feature vector with the correct size
            # Each metric produces 5 features, and we're processing multiple metrics from different categories
            num_metrics = (
                3 +  # CPU metrics
                5 +  # Memory metrics
                9 +  # IO metrics
                7 +  # Network metrics
                8    # Transaction metrics
            )
            feature_vector = np.zeros(num_metrics * 5)  # 5 features per metric
            feature_idx = 0
            
            # CPU metrics
            cpu_metrics = metrics.get('cpu', {})
            for metric in ['cpu usage', 'worker time', 'cpu time']:
                values = self._get_time_window_values(cpu_metrics, metric)
                features = self._calculate_time_features(values)
                feature_vector[feature_idx:feature_idx+len(features)] = features
                feature_idx += len(features)
            
            # Memory metrics
            memory_metrics = metrics.get('memory', {})
            for metric in ['total memstore used', 'active memstore used', 'memstore limit', 
                         'memory usage', 'observer memory hold size']:
                values = self._get_time_window_values(memory_metrics, metric)
                features = self._calculate_time_features(values)
                feature_vector[feature_idx:feature_idx+len(features)] = features
                feature_idx += len(features)
            
            # IO metrics
            io_metrics = metrics.get('io', {})
            for metric in ['palf write io count to disk', 'palf write size to disk',
                         'io read count', 'io write count', 'io read delay', 'io write delay',
                         'data micro block cache hit', 'index micro block cache hit', 'row cache hit']:
                values = self._get_time_window_values(io_metrics, metric)
                features = self._calculate_time_features(values)
                feature_vector[feature_idx:feature_idx+len(features)] = features
                feature_idx += len(features)
            
            # Network metrics
            network_metrics = metrics.get('network', {})
            for metric in ['rpc packet in', 'rpc packet in bytes', 'rpc packet out',
                         'rpc packet out bytes', 'rpc net delay', 'mysql packet in',
                         'mysql packet out']:
                values = self._get_time_window_values(network_metrics, metric)
                features = self._calculate_time_features(values)
                feature_vector[feature_idx:feature_idx+len(features)] = features
                feature_idx += len(features)
            
            # Transaction metrics
            trans_metrics = metrics.get('transactions', {})
            for metric in ['trans commit count', 'trans timeout count', 'sql update count',
                         'sql select count', 'active sessions', 'sql fail count',
                         'request queue time', 'DB time']:
                values = self._get_time_window_values(trans_metrics, metric)
                features = self._calculate_time_features(values)
                feature_vector[feature_idx:feature_idx+len(features)] = features
                feature_idx += len(features)
                
            return feature_vector
        except Exception as e:
            logger.error(f"Error processing metrics: {str(e)}")
            return None

    def calculate_score_for_node(self, node_metrics):
        """Calculate PageRank scores for nodes based on metric correlations"""
        w = [[0]*self.node_num for _ in range(self.node_num)]
        for i in range(self.node_num):
            for j in range(self.node_num):
                if i != j and j > i:
                    w[i][j] = self.calculate_rank(node_metrics[i], node_metrics[j])
                elif j < i:
                    w[i][j] = w[j][i]

        pr = WPRNNode(w, self.node_num)
        return pr.page_rank(10)

    def calculate_rank(self, node_x, node_y):
        """Calculate correlation between two nodes' metrics"""
        x_array = np.array(node_x)
        y_array = np.array(node_y)
        
        # Ensure x_array and y_array are 2D
        if x_array.ndim == 1:
            x_array = x_array.reshape(-1, 1)
        if y_array.ndim == 1:
            y_array = y_array.reshape(-1, 1)

        rvalue = []
        
        for i in range(len(x_array[0])):
            x_a = x_array[:, i]
            y_a = y_array[:, i]
            try:
                # Check if either array is constant
                if np.std(x_a) == 0 or np.std(y_a) == 0:
                    logger.debug(f"Skipping constant array in feature {i}.")
                    continue
                else:
                    # Convert numpy arrays to lists for safe logging
                    x_preview = x_a.tolist()[:3] if len(x_a) > 3 else x_a.tolist()
                    y_preview = y_a.tolist()[:3] if len(y_a) > 3 else y_a.tolist()
                    logger.debug(f"Calculating correlation for feature {i} - x shape: {x_a.shape}, y shape: {y_a.shape}")
                    logger.debug(f"Sample values - x: {x_preview}..., y: {y_preview}...")
                
                # Calculate correlation only if arrays have valid data
                if len(x_a) > 1 and len(y_a) > 1:
                    coef, _ = pearsonr(x_a, y_a)
                    if -1 <= coef <= 1:  # Valid correlation coefficient
                        rvalue.append(abs(coef))
                else:
                    logger.debug(f"Skipping correlation for feature {i} - insufficient data points")
            except Exception as e:
                logger.warning(f"Error calculating correlation for feature {i}: {str(e)}")
                continue

        # Return average correlation or 0 if no valid correlations
        if rvalue:
            avg_correlation = sum(rvalue) / len(rvalue)
            logger.debug(f"Average correlation across {len(rvalue)} features: {avg_correlation:.4f}")
            return avg_correlation
        else:
            logger.debug("No valid correlations calculated")
            return 0.0

    def diagnose(self, node_metrics: dict) -> dict:
        """
        Diagnose anomalies from node metrics
        
        Args:
            node_metrics: Dict of node metrics in format:
                {
                    'node_name': {
                        'metric_name': {
                            'latest': value,
                            ...
                        },
                        ...
                    },
                    ...
                }
                
        Returns:
            Dict containing:
                - scores: Dict of anomaly scores per node
                - anomalies: List of detected anomalies with types
                - propagation_graph: Dict showing anomaly propagation
        """
        try:
            # Process metrics into feature vectors
            features = []
            node_names = []
            node_metrics_list = []  # Store processed metrics for correlation
            
            for node, metrics in node_metrics.items():
                feature_vector = self._process_metrics(metrics)
                if feature_vector is not None:
                    features.append(feature_vector)
                    node_names.append(node)
                    node_metrics_list.append(feature_vector)

            if not features:
                logger.warning("No valid features extracted from metrics")
                return self._empty_result(node_metrics)

            # Calculate node importance scores using PageRank
            node_scores = self.calculate_score_for_node(node_metrics_list)
            
            # Get predictions and combine with node scores
            results = {}
            anomalies = []
            propagation_graph = {}

            # check node_names
            for node_idx, node in enumerate(node_names):
                node_results = {}
                feature = np.array([features[node_idx]])
                
                # Get classifier predictions
                for clf_idx, clf in enumerate(self.clf_list):
                    try:
                        # Ensure we don't go out of bounds with the classifier index
                        if clf_idx >= len(self.classes):
                            logger.warning(f"Classifier index {clf_idx} exceeds available classes ({len(self.classes)})")
                            continue
                            
                        # Make prediction and get probabilities
                        prediction = clf.predict_proba(feature)
                        if prediction is None or len(prediction) == 0:
                            logger.warning(f"Empty prediction from classifier {clf_idx} for node {node}")
                            continue
                            
                        proba = prediction[0]
                        
                        # Safe access to probability values
                        if len(proba) > 1:
                            # Binary classification case - use probability of positive class
                            score = float(proba[1])
                        else:
                            # Single class case - use the only probability available
                            score = float(proba[0])
                            
                        # Weight the score by node importance
                        weighted_score = score * node_scores[node_idx]
                        anomaly_type = self.classes[clf_idx]
                        node_results[anomaly_type] = weighted_score
                    except Exception as e:
                        logger.error(f"Error predicting with classifier {clf_idx}: {str(e)}")
                        logger.debug(f"Exception details: {traceback.format_exc()}")
                        continue
                
                results[node] = node_results

                # Find highest scoring anomaly
                if node_results:
                    max_score = max(node_results.values())
                    max_type = max(node_results.items(), key=lambda x: x[1])[0]
                    
                    if max_score > 0.15:  # Threshold aligned with reference implementation
                        anomalies.append({
                            'node': node,
                            'type': max_type,
                            'score': max_score,
                            'timestamp': datetime.now().isoformat()
                        })

            # Calculate propagation graph based on correlations
            for i, node1 in enumerate(node_names):
                propagation_graph[node1] = []
                for j, node2 in enumerate(node_names):
                    if i != j:
                        correlation = self.calculate_rank(node_metrics_list[i], node_metrics_list[j])
                        if correlation > 0.3:  # Correlation threshold from reference
                            propagation_graph[node1].append({
                                'target': node2,
                                'correlation': correlation
                            })

            # Sort anomalies by score
            anomalies.sort(key=lambda x: x['score'], reverse=True)
            
            # After calculating all scores
            all_possible_anomalies = []
            for node_idx, node in enumerate(node_names):
                for anomaly_type, score in results[node].items():
                    all_possible_anomalies.append({
                        'node': node,
                        'type': anomaly_type,
                        'score': score,
                        'timestamp': datetime.now().isoformat()
                    })
            
            # Sort all possibilities by score
            all_possible_anomalies.sort(key=lambda x: x['score'], reverse=True)
            
            return {
                'scores': {node: max(node_results.values()) if node_results else 0.0 for node, node_results in results.items()},
                'anomalies': anomalies,
                'propagation_graph': propagation_graph,
                'all_ranked_anomalies': all_possible_anomalies
            }

        except Exception as e:
            logger.error(f"Error in diagnose: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self._empty_result(node_metrics)

    def _empty_result(self, node_metrics):
        """Return empty result structure"""
        return {
            'scores': {node: 0.0 for node in node_metrics.keys()},
            'anomalies': [],
            'propagation_graph': {},
            'all_ranked_anomalies': []
        }

    def train(self, x_train, y_train):
        """Train the model with collected data"""
        try:
            logger.debug(f"Original training data shapes - X: {x_train.shape}, y: {y_train.shape}")
            logger.debug(f"X data sample: {x_train[0] if len(x_train) > 0 else 'empty'}")
            logger.debug(f"y data sample: {y_train[0] if len(y_train) > 0 else 'empty'}")
            
            # Decompose training data
            start_time = time.time()
            train_x = self.decompose_train_x(x_train)
            train_y = self.decompose_train_y(y_train)
            logger.info(f"Decomposition completed in {time.time() - start_time:.2f} seconds")
            
            logger.debug(f"Decomposed training data shapes - X: {train_x.shape}, y: {train_y.shape}")
            logger.debug(f"Decomposed X sample: {train_x[0] if len(train_x) > 0 else 'empty'}")
            logger.debug(f"Decomposed y sample: {train_y[0] if len(train_y) > 0 else 'empty'}")
            
            if len(train_x) == 0 or len(train_y) == 0:
                raise ValueError("Empty training data after decomposition")

            # Ensure we have enough classifiers for the number of classes
            while len(self.clf_list) < train_y.shape[1]:
                logger.info(f"Adding classifier {len(self.clf_list)} to match training data dimensions")
                self.clf_list.append(XGBClassifier(
                    objective='binary:logistic',
                    tree_method='gpu_hist',
                    gpu_id=0,
                    predictor='gpu_predictor',
                    n_jobs=-1  # Use all available CPU cores for parallel processing
                ))
            
            # Use parallel processing for training multiple classifiers
            from concurrent.futures import ThreadPoolExecutor
            
            def train_classifier(i):
                current_clf = self.clf_list[i]
                current_labels = train_y[:, i].reshape(-1, 1)
                
                logger.debug(f"Training classifier {i} with {len(train_x)} samples")
                logger.debug(f"Labels for classifier {i}: shape={current_labels.shape}, unique values={np.unique(current_labels)}")
                current_clf.fit(train_x, current_labels.ravel())  # Use ravel() to flatten labels
                return i
            
            # Train classifiers in parallel
            start_time = time.time()
            with ThreadPoolExecutor() as executor:
                # Submit all training tasks
                futures = [executor.submit(train_classifier, i) 
                          for i in range(min(len(self.clf_list), train_y.shape[1]))]
                
                # Wait for all tasks to complete
                for future in futures:
                    future.result()
            
            logger.info(f"Successfully trained all classifiers in {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def save_model(self, model_dir):
        """Save trained models"""
        os.makedirs(model_dir, exist_ok=True)
        for i, clf in enumerate(self.clf_list):
            model_path = os.path.join(model_dir, f'classifier_{i}.joblib')
            joblib.dump(clf, model_path)

    def load_model(self, model_dir):
        """Load trained models"""
        self.clf_list = []
        for i in range(self.num_classifier):
            model_path = os.path.join(model_dir, f'classifier_{i}.joblib')
            if os.path.exists(model_path):
                clf = joblib.load(model_path)
                self.clf_list.append(clf)
            else:
                raise FileNotFoundError(f"Model file not found: {model_path}")

    def decompose_train_x(self, composed_train_x):
        """Decompose training features"""
        try:
            logger.debug(f"Decomposing X - input type: {type(composed_train_x)}, shape/len: {composed_train_x.shape if isinstance(composed_train_x, np.ndarray) else len(composed_train_x)}")
            
            if isinstance(composed_train_x, np.ndarray):
                if len(composed_train_x.shape) == 1:
                    result = composed_train_x.reshape(1, -1)
                    logger.debug(f"Reshaped 1D array to shape: {result.shape}")
                    return result
                logger.debug(f"Using numpy array directly with shape: {composed_train_x.shape}")
                return composed_train_x
            
            # Pre-allocate the array if we can determine the size
            if isinstance(composed_train_x, list) and all(isinstance(x, np.ndarray) for x in composed_train_x):
                # Get the shape of the first element to determine dimensions
                first_shape = composed_train_x[0].shape if isinstance(composed_train_x[0], np.ndarray) else None
                if isinstance(composed_train_x[0], list) and composed_train_x[0] and isinstance(composed_train_x[0][0], np.ndarray):
                    # We have a list of lists of arrays
                    total_samples = sum(len(case) for case in composed_train_x)
                    sample_shape = composed_train_x[0][0].shape
                    if total_samples > 0 and all(len(sample_shape) == 1 for case in composed_train_x for sample in case if isinstance(sample, np.ndarray)):
                        # Pre-allocate the result array
                        feature_dim = sample_shape[0]
                        result = np.zeros((total_samples, feature_dim))
                        idx = 0
                        for case in composed_train_x:
                            for node_x in case:
                                if isinstance(node_x, np.ndarray) and len(node_x.shape) == 1:
                                    result[idx] = node_x
                                    idx += 1
                        logger.debug(f"Pre-allocated and filled result array with shape: {result.shape}")
                        return result
            
            # Fall back to list append method if pre-allocation not possible
            train_x = []
            for case in composed_train_x:
                if isinstance(case, (list, np.ndarray)):
                    for node_x in case:
                        train_x.append(node_x)
                else:
                    train_x.append(case)
            result = np.array(train_x)
            logger.debug(f"Decomposed X result shape: {result.shape}")
            return result
        except Exception as e:
            logger.error(f"Error decomposing training features: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def decompose_train_y(self, composed_train_y):
        """Decompose training labels"""
        try:
            logger.debug(f"Decomposing y - input type: {type(composed_train_y)}, shape/len: {composed_train_y.shape if isinstance(composed_train_y, np.ndarray) else len(composed_train_y)}")
            
            if isinstance(composed_train_y, np.ndarray):
                if len(composed_train_y.shape) == 1:
                    result = composed_train_y.reshape(1, -1)
                    logger.debug(f"Reshaped 1D array to shape: {result.shape}")
                    return result
                logger.debug(f"Using numpy array directly with shape: {composed_train_y.shape}")
                return composed_train_y
            
            train_y = []
            for case in composed_train_y:
                if isinstance(case, (list, np.ndarray)):
                    for node_label in case:
                        train_y.append(node_label)
                else:
                    train_y.append(case)
            result = np.array(train_y)
            logger.debug(f"Decomposed y result shape: {result.shape}")
            return result
        except Exception as e:
            logger.error(f"Error decomposing training labels: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

class DiagnosisService:
    def __init__(self, model_name=None):
        # Base models directory
        self.models_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            '../../models/'
        ))
        
        # Define anomaly types that can be detected
        self.anomaly_types = [
            "cpu_stress",
            "io_bottleneck", 
            "network_bottleneck",
            "cache_bottleneck",
            "too_many_indexes"
        ]
        
        # Get node count from environment or config
        self.node_count = int(os.getenv('NODE_COUNT', '3'))
        
        # Get time window size from environment or use default
        self.time_window = int(os.getenv('METRICS_TIME_WINDOW', '5'))
        
        # Initialize DistDiagnosis with multi-node support
        self.diagnosis = DistDiagnosis(
            num_classifier=len(self.anomaly_types),
            classes_for_each_node=self.anomaly_types,
            node_num=self.node_count
        )
        self.diagnosis.time_window = self.time_window  # Set time window for metric processing
        
        # Select model to use - either specified or latest available
        available_models = self.get_available_models()
        self.active_model = model_name if model_name and model_name in available_models else (
            available_models[0] if available_models else None
        )
        
        if self.active_model:
            # Set model path based on active model
            self.model_path = os.path.join(self.models_path, self.active_model)
            self.metrics_path = os.path.join(self.model_path, 'metrics')
            os.makedirs(self.metrics_path, exist_ok=True)
            
            # Load the selected model
            try:
                self.diagnosis.load_model(self.model_path)
                logger.info("Loaded model '%s' from %s", self.active_model, self.model_path)
            except Exception as e:
                logger.error("Failed to load model '%s': %s", self.active_model, str(e))
        else:
            logger.warning("No models available to load")
            # Set default paths even if no model is available
            self.model_path = None
            self.metrics_path = os.path.join(self.models_path, 'metrics')
            os.makedirs(self.metrics_path, exist_ok=True)

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the model with collected data"""
        try:
            if len(X) == 0 or len(y) == 0:
                raise ValueError("Empty training data")
                
            # Split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            
            # Start timing
            self.training_start_time = time.time()
            
            # Train model
            self.diagnosis.train(X_train, y_train)
            
            # Generate model name with dist-diagnosis prefix
            model_name = f"dist-diagnosis-model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create new model directory
            new_model_path = os.path.join(self.models_path, model_name)
            os.makedirs(new_model_path, exist_ok=True)
            
            # Create metrics directory for this model
            new_metrics_path = os.path.join(new_model_path, 'metrics')
            os.makedirs(new_metrics_path, exist_ok=True)
            
            # Save model
            self.diagnosis.save_model(new_model_path)
            
            # Update current model paths
            self.active_model = model_name
            self.model_path = new_model_path
            self.metrics_path = new_metrics_path
            
            # Evaluate and store metrics
            metrics = self.evaluate_model(X_test, y_test, model_name)
            
            return {
                "model_name": model_name,
                "metrics": metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to train model: {str(e)}")
            raise

    def analyze_metrics(self, metrics: Dict[str, Any]) -> List[Dict]:
        """Analyze metrics and return anomaly ranks"""
        try:
            if not metrics:
                logger.warning("Empty metrics received")
                return []
                
            # Get diagnosis results using enhanced algorithm
            diagnosis_result = self.diagnosis.diagnose(metrics)
            
            # Return ranked anomalies with propagation information
            return diagnosis_result.get('anomalies', [])
            
        except Exception as e:
            logger.error("Error in analyze_metrics: %s", str(e))
            return []

    def evaluate_model(self, X_test, y_test, model_name):
        """Evaluate model performance and store metrics"""
        try:
            # Get predictions from all classifiers
            all_preds = []
            for i, clf in enumerate(self.diagnosis.clf_list):
                pred = clf.predict(X_test)
                all_preds.append(pred)
            
            # Combine predictions into a multilabel format to match y_test
            y_pred = np.zeros_like(y_test)
            for i, preds in enumerate(all_preds):
                y_pred[:, i] = preds
            
            # Standard performance metrics - using samples average for multilabel
            metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, average='samples')),
                'recall': float(recall_score(y_test, y_pred, average='samples')),
                'f1_score': float(f1_score(y_test, y_pred, average='samples')),
                'training_time': time.time() - self.training_start_time
            }
            
            # Add RCA performance metrics
            rca_metrics = self._evaluate_rca_performance(X_test, y_test)
            metrics.update(rca_metrics)
            
            # Save metrics
            metrics_file = os.path.join(self.metrics_path, f"{model_name}_metrics.json")
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f)
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise

    def _evaluate_rca_performance(self, X_test, y_test):
        """Evaluate RCA-specific performance metrics"""
        try:
            # Start timer for RCA performance measurement
            start_time = time.time()
            
            # Create a synthetic test scenario from test data
            test_metrics = self._create_test_metrics(X_test)
            
            # Get RCA results
            diagnosis_result = self.diagnosis.diagnose(test_metrics)
            
            # Calculate RCA accuracy (how often did the model identify the correct root cause)
            true_root_causes = self._extract_root_causes_from_labels(y_test)
            predicted_root_causes = self._extract_root_causes_from_diagnosis(diagnosis_result)
            
            # Calculate RCA metrics
            rca_accuracy = self._calculate_rca_accuracy(true_root_causes, predicted_root_causes)
            rca_time = time.time() - start_time
            
            return {
                'rca_accuracy': float(rca_accuracy),
                'rca_time': float(rca_time),
                'rca_precision': float(self._calculate_rca_precision(true_root_causes, predicted_root_causes)),
                'propagation_accuracy': float(self._evaluate_propagation_graph(diagnosis_result.get('propagation_graph', {})))
            }
            
        except Exception as e:
            logger.error(f"Error evaluating RCA performance: {str(e)}")
            return {
                'rca_accuracy': 0.0,
                'rca_time': 0.0,
                'rca_precision': 0.0,
                'propagation_accuracy': 0.0
            }

    def _create_test_metrics(self, X_test):
        """Create synthetic test metrics from test feature vectors"""
        test_metrics = {}
        
        # We'll use the actual number of samples, but limit to first 10 for performance
        max_samples = min(len(X_test), 10)
        
        # Create metrics for each test node
        for i, features in enumerate(X_test[:max_samples]):
            node_name = f"test-node-{i}"
            
            # Create sample feature value index tracker
            feature_idx = 0
            
            # Initialize metrics structure that matches expected format
            test_metrics[node_name] = {
                'cpu': {},
                'memory': {},
                'io': {},
                'network': {},
                'transactions': {}
            }
            
            # Add CPU metrics (3 metrics)
            cpu_metrics = ['cpu usage', 'worker time', 'cpu time']
            for metric in cpu_metrics:
                base_value = abs(float(features[feature_idx % len(features)]))
                test_metrics[node_name]['cpu'][metric] = {
                    'latest': base_value
                }
                feature_idx += 1
            
            # Add Memory metrics (5 metrics)
            memory_metrics = [
                'total memstore used', 'active memstore used', 'memstore limit', 
                'memory usage', 'observer memory hold size'
            ]
            for metric in memory_metrics:
                base_value = abs(float(features[feature_idx % len(features)]))
                test_metrics[node_name]['memory'][metric] = {
                    'latest': base_value
                }
                feature_idx += 1
            
            # Add IO metrics (9 metrics)
            io_metrics = [
                'palf write io count to disk', 'palf write size to disk',
                'io read count', 'io write count', 'io read delay', 'io write delay',
                'data micro block cache hit', 'index micro block cache hit', 'row cache hit'
            ]
            for metric in io_metrics:
                base_value = abs(float(features[feature_idx % len(features)]))
                test_metrics[node_name]['io'][metric] = {
                    'latest': base_value
                }
                feature_idx += 1
            
            # Add Network metrics (7 metrics)
            network_metrics = [
                'rpc packet in', 'rpc packet in bytes', 'rpc packet out',
                'rpc packet out bytes', 'rpc net delay', 'mysql packet in',
                'mysql packet out'
            ]
            for metric in network_metrics:
                base_value = abs(float(features[feature_idx % len(features)]))
                test_metrics[node_name]['network'][metric] = {
                    'latest': base_value
                }
                feature_idx += 1
            
            # Add Transaction metrics (8 metrics)
            transaction_metrics = [
                'trans commit count', 'trans timeout count', 'sql update count',
                'sql select count', 'active sessions', 'sql fail count',
                'request queue time', 'DB time'
            ]
            for metric in transaction_metrics:
                base_value = abs(float(features[feature_idx % len(features)]))
                test_metrics[node_name]['transactions'][metric] = {
                    'latest': base_value
                }
                feature_idx += 1
        
        logger.debug(f"Created synthetic test metrics for {len(test_metrics)} nodes")
        return test_metrics
        
    def _extract_root_causes_from_labels(self, y_test):
        """Extract root causes from test labels"""
        root_causes = []
        for i, label_vector in enumerate(y_test):
            if np.any(label_vector):  # If any anomaly is present
                # Find the index of highest value (most significant anomaly type)
                anomaly_type_idx = np.argmax(label_vector)
                root_causes.append({
                    'node_idx': i,
                    'type_idx': anomaly_type_idx,
                    'type': self.anomaly_types[anomaly_type_idx] if anomaly_type_idx < len(self.anomaly_types) else 'unknown'
                })
        return root_causes
        
    def _calculate_rca_accuracy(self, true_root_causes, predicted_root_causes):
        """Calculate accuracy of root cause identification"""
        if not true_root_causes or not predicted_root_causes:
            return 0.0
            
        # Consider only top predictions (equal to number of true causes)
        top_predictions = predicted_root_causes[:len(true_root_causes)]
        
        # Count matches (both node and type match)
        matches = 0
        for true_cause in true_root_causes:
            for pred_cause in top_predictions:
                if (true_cause['node_idx'] == pred_cause['node_idx'] and
                    true_cause['type_idx'] == pred_cause['type_idx']):
                    matches += 1
                    break
                    
        return matches / len(true_root_causes)
        
    def _calculate_rca_precision(self, true_root_causes, predicted_root_causes):
        """Calculate precision of root cause identification"""
        if not predicted_root_causes:
            return 0.0
            
        # Consider only top predictions (equal to number of true causes)
        top_predictions = predicted_root_causes[:len(true_root_causes)] if true_root_causes else predicted_root_causes[:1]
        
        # Count correct predictions
        correct = 0
        for pred_cause in top_predictions:
            for true_cause in true_root_causes:
                if (true_cause['node_idx'] == pred_cause['node_idx'] and
                    true_cause['type_idx'] == pred_cause['type_idx']):
                    correct += 1
                    break
                    
        return correct / len(top_predictions)
        
    def _evaluate_propagation_graph(self, propagation_graph):
        """Evaluate accuracy of the propagation graph"""
        # In a real implementation, you would compare against ground truth
        # This is a placeholder implementation
        if not propagation_graph:
            return 0.0
            
        # Since we don't have ground truth for the propagation graph in this example,
        # we'll return a synthetic metric based on graph connectivity
        total_connections = 0
        strong_connections = 0  # Connections with correlation > 0.5
        
        for node, connections in propagation_graph.items():
            total_connections += len(connections)
            for conn in connections:
                if conn.get('correlation', 0) > 0.5:
                    strong_connections += 1
                    
        # Return ratio of strong connections to total
        return strong_connections / total_connections if total_connections > 0 else 0.0

    def get_available_models(self):
        """Get list of available models"""
        try:
            models = []
            # Check main model directory
            if os.path.exists(self.models_path):
                for entry in os.listdir(self.models_path):
                    if os.path.isdir(os.path.join(self.models_path, entry)):
                        # Check if any classifier files exist in this directory
                        classifier_files = glob.glob(os.path.join(self.models_path, entry, "classifier_*.joblib"))
                        if classifier_files:
                            models.append(entry)
                            logger.debug(f"Found model directory '{entry}' with {len(classifier_files)} classifiers")
            
            logger.info(f"Found {len(models)} available models")
            return sorted(models, reverse=True)  # Sort by newest first
            
        except Exception as e:
            logger.error(f"Error getting available models: {str(e)}")
            return []

    def switch_model(self, model_name):
        """Switch to using a different model"""
        try:
            logger.info(f"Attempting to switch to model: {model_name}")
            available_models = self.get_available_models()
            
            if model_name not in available_models:
                logger.error(f"Model '{model_name}' not found in available models: {available_models}")
                return False
                
            # Full path verification
            model_path = os.path.join(self.models_path, model_name)
            if not os.path.exists(model_path):
                logger.error(f"Model directory not found: {model_path}")
                return False
                
            logger.info(f"Loading model from: {model_path}")
            self.diagnosis.load_model(model_path)
            
            # Verify classifier loading
            if not self.diagnosis.clf_list:
                logger.error("No classifiers loaded after model switch")
                return False
                
            logger.info(f"Successfully loaded {len(self.diagnosis.clf_list)} classifiers")
            self.active_model = model_name
            return True
            
        except Exception as e:
            logger.error(f"Model switch failed: {str(e)}")
            return False

# Create singleton instance
diagnosis_service = DiagnosisService()