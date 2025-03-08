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
        self.clf_list = [XGBClassifier(objective='binary:logistic') for _ in range(num_classifier)]
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
                    
                # Handle case where metric_data is a dict of timestamps
                elif isinstance(metric_data, dict):
                    timestamps = sorted(metric_data.keys(), reverse=True)[:self.time_window]
                    for ts in timestamps:
                        if isinstance(metric_data[ts], dict) and 'latest' in metric_data[ts]:
                            values.append(float(metric_data[ts]['latest']))
                        elif isinstance(metric_data[ts], (int, float)):
                            values.append(float(metric_data[ts]))
                        else:
                            values.append(0.0)
            
            # If metrics is a dict of timestamps containing metric_name
            elif isinstance(next(iter(metrics.values()), {}), dict):
                timestamps = sorted(metrics.keys(), reverse=True)[:self.time_window]
                for ts in timestamps:
                    ts_data = metrics[ts]
                    if metric_name in ts_data:
                        if isinstance(ts_data[metric_name], dict) and 'latest' in ts_data[metric_name]:
                            values.append(float(ts_data[metric_name]['latest']))
                        elif isinstance(ts_data[metric_name], (int, float)):
                            values.append(float(ts_data[metric_name]))
                        else:
                            values.append(0.0)
                    else:
                        values.append(0.0)

            # Pad with zeros if we don't have enough values
            while len(values) < self.time_window:
                values.append(0.0)

            return values[:self.time_window]  # Ensure we only return time_window values
        except Exception as e:
            logger.error(f"Error getting time window values for {metric_name}: {str(e)}")
            return [0.0] * self.time_window

    def _calculate_time_features(self, values: List[float]) -> List[float]:
        """Calculate statistical features from time window values"""
        try:
            values = np.array(values)
            features = []
            
            # Basic statistics
            features.append(np.mean(values))  # Mean
            features.append(np.std(values))   # Standard deviation
            features.append(np.max(values))   # Maximum
            
            # Trend (slope of linear fit)
            if len(values) > 1:
                x = np.arange(len(values))
                slope = np.polyfit(x, values, 1)[0]
                features.append(slope)
            else:
                features.append(0.0)
            
            # Rate of change (using last two points)
            if len(values) > 1:
                rate = values[-1] - values[-2]
                features.append(rate)
            else:
                features.append(0.0)
                
            return features
        except Exception as e:
            logger.error(f"Error calculating time features: {str(e)}")
            return [0.0] * 5  # Return zeros for all features

    def _process_metrics(self, metrics: Dict[str, Any]) -> np.ndarray:
        """Process metrics into feature vector using time window statistics"""
        try:
            feature_vector = []
            
            # CPU metrics
            cpu_metrics = metrics.get('cpu', {})
            for metric in ['cpu usage', 'worker time', 'cpu time']:
                values = self._get_time_window_values(cpu_metrics, metric)
                feature_vector.extend(self._calculate_time_features(values))
            
            # Memory metrics
            memory_metrics = metrics.get('memory', {})
            for metric in ['total memstore used', 'active memstore used', 'memstore limit', 
                         'memory usage', 'observer memory hold size']:
                values = self._get_time_window_values(memory_metrics, metric)
                feature_vector.extend(self._calculate_time_features(values))
            
            # IO metrics
            io_metrics = metrics.get('io', {})
            for metric in ['palf write io count to disk', 'palf write size to disk',
                         'io read count', 'io write count', 'io read delay', 'io write delay',
                         'data micro block cache hit', 'index micro block cache hit', 'row cache hit']:
                values = self._get_time_window_values(io_metrics, metric)
                feature_vector.extend(self._calculate_time_features(values))
            
            # Network metrics
            network_metrics = metrics.get('network', {})
            for metric in ['rpc packet in', 'rpc packet in bytes', 'rpc packet out',
                         'rpc packet out bytes', 'rpc net delay', 'mysql packet in',
                         'mysql packet out']:
                values = self._get_time_window_values(network_metrics, metric)
                feature_vector.extend(self._calculate_time_features(values))
            
            # Transaction metrics
            trans_metrics = metrics.get('transactions', {})
            for metric in ['trans commit count', 'trans timeout count', 'sql update count',
                         'sql select count', 'active sessions', 'sql fail count',
                         'request queue time', 'DB time']:
                values = self._get_time_window_values(trans_metrics, metric)
                feature_vector.extend(self._calculate_time_features(values))
            
            return np.array(feature_vector)
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
                    logger.debug(f"Skipping constant array in feature {i}")
                    continue
                    
                coef, _ = pearsonr(x_a, y_a)
                if -1 <= coef <= 1:
                    rvalue.append(abs(coef))
            except Exception as e:
                logger.warning(f"Error calculating correlation: {str(e)}")
                continue

        return sum(rvalue) / len(rvalue) if rvalue else 0

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

            for node_idx, node in enumerate(node_names):
                node_results = {}
                feature = np.array([features[node_idx]])
                
                # Get classifier predictions
                for clf_idx, clf in enumerate(self.clf_list):
                    try:
                        proba = clf.predict_proba(feature)[0]
                        score = float(proba[1]) if len(proba) > 1 else float(proba[0])
                        # Weight the score by node importance
                        weighted_score = score * node_scores[node_idx]
                        anomaly_type = self.classes[clf_idx]
                        node_results[anomaly_type] = weighted_score
                    except Exception as e:
                        logger.error(f"Error predicting with classifier {clf_idx}: {str(e)}")
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
            
            return {
                'scores': {node: max(node_results.values()) if node_results else 0.0 for node, node_results in results.items()},
                'anomalies': anomalies,
                'propagation_graph': propagation_graph
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
            'propagation_graph': {}
        }

    def train(self, x_train, y_train):
        """Train the model with collected data"""
        try:
            logger.debug(f"Original training data shapes - X: {x_train.shape}, y: {y_train.shape}")
            logger.debug(f"X data sample: {x_train[0] if len(x_train) > 0 else 'empty'}")
            logger.debug(f"y data sample: {y_train[0] if len(y_train) > 0 else 'empty'}")
            
            # Decompose training data
            train_x = self.decompose_train_x(x_train)
            train_y = self.decompose_train_y(y_train)
            
            logger.debug(f"Decomposed training data shapes - X: {train_x.shape}, y: {train_y.shape}")
            logger.debug(f"Decomposed X sample: {train_x[0] if len(train_x) > 0 else 'empty'}")
            logger.debug(f"Decomposed y sample: {train_y[0] if len(train_y) > 0 else 'empty'}")
            
            if len(train_x) == 0 or len(train_y) == 0:
                raise ValueError("Empty training data after decomposition")

            # Ensure we have enough classifiers for the number of classes
            while len(self.clf_list) < train_y.shape[1]:
                logger.info(f"Adding classifier {len(self.clf_list)} to match training data dimensions")
                self.clf_list.append(XGBClassifier(objective='binary:logistic'))

            for i in range(min(len(self.clf_list), train_y.shape[1])):
                current_clf = self.clf_list[i]
                current_labels = train_y[:, i].reshape(-1, 1)
                
                logger.debug(f"Training classifier {i} with {len(train_x)} samples")
                logger.debug(f"Labels for classifier {i}: shape={current_labels.shape}, unique values={np.unique(current_labels)}")
                current_clf.fit(train_x, current_labels.ravel())  # Use ravel() to flatten labels
                
            logger.info("Successfully trained all classifiers")
            
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
            for clf in self.diagnosis.clf_list:
                pred = clf.predict(X_test)
                all_preds.append(pred)
            
            # Use majority voting for final prediction
            y_pred = np.round(np.mean(all_preds, axis=0))
            
            # Standard performance metrics
            metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, average='weighted')),
                'recall': float(recall_score(y_test, y_pred, average='weighted')),
                'f1_score': float(f1_score(y_test, y_pred, average='weighted')),
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
        # This is a simplified implementation - in real scenarios,
        # you would need a more sophisticated approach to recreate realistic metrics
        test_metrics = {}
        
        # Create metrics for each test node (assuming first 10 samples for demonstration)
        for i, features in enumerate(X_test[:10]):
            node_name = f"test-node-{i}"
            
            # Recreate metrics structure from feature vector
            test_metrics[node_name] = {
                'cpu': {
                    'util': {'latest': features[0]},
                    'system': {'latest': features[1]},
                    'user': {'latest': features[2]},
                    'wait': {'latest': features[3]}
                },
                'memory': {
                    'used': {'latest': features[4]},
                    'free': {'latest': features[5]},
                    'cach': {'latest': features[6]}
                },
                'io': {
                    'aggregated': {
                        'rs': {'latest': features[7]},
                        'ws': {'latest': features[8]},
                        'util': {'latest': features[9]}
                    }
                },
                'network': {
                    'bytin': {'latest': features[10]},
                    'bytout': {'latest': features[11]},
                    'pktin': {'latest': features[12]}
                }
            }
            
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
        
    def _extract_root_causes_from_diagnosis(self, diagnosis_result):
        """Extract predicted root causes from diagnosis result"""
        root_causes = []
        
        # Extract from anomalies ranked by score
        for anomaly in diagnosis_result.get('anomalies', []):
            node = anomaly.get('node', '')
            anomaly_type = anomaly.get('type', '')
            
            # Map type name to index
            type_idx = self.anomaly_types.index(anomaly_type) if anomaly_type in self.anomaly_types else -1
            
            # Extract node index from node name
            node_idx = -1
            if node.startswith('test-node-'):
                try:
                    node_idx = int(node.split('-')[-1])
                except ValueError:
                    pass
                    
            root_causes.append({
                'node_idx': node_idx,
                'type_idx': type_idx,
                'type': anomaly_type
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
            available_models = self.get_available_models()
            
            if model_name not in available_models:
                logger.error(f"Model '{model_name}' not found. Available models: {available_models}")
                return False
                
            # Set new active model
            self.active_model = model_name
            self.model_path = os.path.join(self.models_path, model_name)
            self.metrics_path = os.path.join(self.model_path, 'metrics')
            
            # Load the model
            self.diagnosis.load_model(self.model_path)
            logger.info(f"Switched to model '{model_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error switching to model '{model_name}': {str(e)}")
            return False

# Create singleton instance
diagnosis_service = DiagnosisService()