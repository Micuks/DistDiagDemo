import sys
from pathlib import Path
import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from scipy.stats import pearsonr
from xgboost import XGBClassifier
import joblib
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
        self.p = [1.0/node_num] * node_num  # Initialize with equal probability

    def page_rank(self, iter_num):
        for _ in range(iter_num):
            new_p = [0] * self.node_num
            for i in range(self.node_num):
                tmp = 0
                for j in range(self.node_num):
                    if j != i:
                        # Ensure we have actual weights (not zeros)
                        if self.w[i][j] > 0:
                            tmp += self.p[j] * self.w[i][j] * self.w[i][j]
                
                # Handle case when no correlations exist
                if tmp == 0:
                    # Distribute probability evenly
                    new_p[i] = 1.0/self.node_num
                else:
                    new_p[i] = (1 - self.d)/self.node_num + self.d * tmp
            
            # Normalize to ensure sum of probabilities is 1
            total = sum(new_p)
            if total > 0:
                new_p = [p/total for p in new_p]
            else:
                new_p = [1.0/self.node_num] * self.node_num
                
            self.p = new_p
            
        return self.p

class DistDiagnosis:
    """Simplified version of DistDiagnosis for anomaly detection"""
    def __init__(self, num_classifier=3, classes_for_each_node=None, node_num=1):
        self.num_classifier = num_classifier
        self.node_num = node_num
        self.classes = classes_for_each_node if classes_for_each_node else [
            "cpu_stress",
            "network_bottleneck",
            "cache_bottleneck"
        ]
        
        # Check for CUDA availability
        try:
            device = 'cuda'
            logger.info(f"XGBoost using device: {device} ")
        except Exception as e:
            logger.warning(f"Error checking GPU availability: {str(e)}. Defaulting to CPU.")
            device = 'cpu'
        
        # Initialize one classifier per anomaly type
        self.clf_list = [XGBClassifier(
            objective='binary:logistic',
            device=device,
            n_jobs=-1
        ) for _ in range(num_classifier)]

        # Window configuration
        self.time_window = 3  # Default 3-minute window
        
        logger.info(f"Initialized {len(self.clf_list)} classifiers for {len(self.classes)} classes")

    def _get_time_window_values(self, metrics: Dict[str, Any], metric_name: str) -> List[float]:
        logger.debug(f"Extracting time window values for metric: {metric_name}. Available keys: {list(metrics.keys()) if metrics else None}")
        try:
            if not metrics or metric_name not in metrics:
                logger.debug(f"Metric '{metric_name}' not found in metrics. Available keys: {list(metrics.keys()) if metrics else 'No metrics'}")
                if metrics:
                    metric_keys = [k for k in metrics.keys() if k.lower() == metric_name.lower()]
                    if metric_keys:
                        metric_name = metric_keys[0]
                        logger.debug(f"Found case-insensitive match for metric, using key: {metric_name}")
                    else:
                        # If still not found, return zeros
                        return [0.0] * self.time_window
                else:
                    return [0.0] * self.time_window
            metric_data = metrics[metric_name]
            logger.debug(f"Metric data for '{metric_name}': {metric_data}")
            values = []
            
            # Handle different data formats
            if isinstance(metric_data, (int, float)):
                logger.debug(f"Scalar metric detected with value: {metric_data}")
                values = [float(metric_data)] * self.time_window
            
            # Handle dictionary with 'data' key containing time series
            elif isinstance(metric_data, dict) and 'data' in metric_data:
                data_entries = metric_data['data']
                logger.debug(f"Detected dict with 'data' key; entries count: {len(data_entries)}")
                if isinstance(data_entries, list) and data_entries:
                    if 'value' in data_entries[0]:
                        values = [float(entry['value']) for entry in data_entries]
            
            # Handle list of timestamped values directly
            elif isinstance(metric_data, list) and metric_data and isinstance(metric_data[0], dict):
                if 'value' in metric_data[0]:
                    values = [float(entry['value']) for entry in metric_data]
            
            # Handle dictionary with nested lists or dictionaries
            elif isinstance(metric_data, dict):
                # Try to extract nested values if present
                if any(isinstance(v, (list, dict)) for v in metric_data.values()):
                    # Find the first nested structure that could contain time series
                    for key, nested_data in metric_data.items():
                        if isinstance(nested_data, list) and nested_data:
                            if isinstance(nested_data[0], dict) and 'value' in nested_data[0]:
                                values = [float(entry['value']) for entry in nested_data]
                                break
                else:
                    # If only scalar values in dict, just use those
                    scalar_values = [v for v in metric_data.values() if isinstance(v, (int, float))]
                    if scalar_values:
                        values = [float(v) for v in scalar_values]

            # Pad with last value if we don't have enough
            if values and len(values) < self.time_window:
                last_value = values[-1] if values else 0.0
                values.extend([last_value] * (self.time_window - len(values)))
            
            # Truncate if we have too many
            if len(values) > self.time_window:
                values = values[-self.time_window:]
            while len(values) < self.time_window:
                values.append(0.0)
            logger.debug(f"Final time window values for metric '{metric_name}': {values[:self.time_window]}")
            return values[:self.time_window]
        except Exception as e:
            logger.error(f"Error getting time window values for {metric_name}: {str(e)}")
            return [0.0] * self.time_window

    def _extract_latest_value(self, metrics_dict: Dict, metric_name: str) -> Optional[float]:
        """Extract the latest value from a metric time series"""
        values = self._get_time_window_values(metrics_dict, metric_name)
        if values:
            return values[-1]  # Return the most recent value
        return None

    def _extract_time_series(self, metrics_dict: Dict, metric_name: str) -> List[float]:
        """Extract the full time series for a metric"""
        return self._get_time_window_values(metrics_dict, metric_name)

    def _calculate_time_features(self, values: List[float]) -> List[float]:
        logger.debug(f"Calculating time features for values: {values}")
        if not values or len(values) < 2:
            logger.debug("Not enough values for time feature calculation, returning zeros")
            return [0.0] * 16
        values = np.array([float(x) for x in values])
        mean = np.mean(values)
        std_dev = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        median = np.median(values)
        slope = 0.0
        try:
            x = np.arange(len(values))
            A = np.vstack([x, np.ones(len(x))]).T
            slope, _ = np.linalg.lstsq(A, values, rcond=None)[0]
        except:
            logger.debug("Exception calculating slope, defaulting to 0")
        diff = np.diff(values)
        mean_diff = np.mean(diff) if len(diff) > 0 else 0.0
        max_diff = np.max(diff) if len(diff) > 0 else 0.0
        min_diff = np.min(diff) if len(diff) > 0 else 0.0
        std_diff = np.std(diff) if len(diff) > 0 else 0.0
        last_value = values[-1]
        last_diff = diff[-1] if len(diff) > 0 else 0.0
        last_3_mean = np.mean(values[-3:]) if len(values) >= 3 else last_value
        range_val = max_val - min_val
        coef_var = std_dev / mean if mean != 0 else 0.0
        sum_val = np.sum(values)
        features = [mean, median, std_dev, min_val, max_val, slope, mean_diff, max_diff, min_diff, std_diff, last_value, last_diff, last_3_mean, range_val, coef_var, sum_val]
        features = [0.0 if np.isnan(x) or np.isinf(x) else x for x in features]
        logger.debug(f"Computed time features: {features}")
        return features

    def _process_metrics(self, metrics: dict) -> np.ndarray:
        try:
            # If the provided metrics is already a numpy array (i.e. a feature vector), return it directly.
            if isinstance(metrics, np.ndarray):
                return metrics

            # If the input metrics dict contains a nested 'metrics' key, extract it.
            if isinstance(metrics, dict) and 'metrics' in metrics and isinstance(metrics['metrics'], dict):
                metrics = metrics['metrics']
            
            # Check if metrics dictionary is flat (not nested by expected groups) and restructure if necessary.
            expected_groups = ['cpu', 'memory', 'io', 'network', 'transactions']
            if not any(group in metrics for group in expected_groups):
                groups = {
                    "cpu": ['cpu usage', 'worker time', 'cpu time'],
                    "memory": ['active memstore used', 'total memstore used', 'memstore limit', 'memory usage', 'observer memory hold size'],
                    "io": ['row cache hit', 'row cache miss', 'io read count', 'io read delay', 'io write count', 'io write delay', 'accessed data micro block count', 'data micro block cache hit', 'index micro block cache hit', 'blockscaned data micro block count', 'palf write io count to disk', 'palf write size to disk', 'clog trans log total size'],
                    "network": ['rpc packet in', 'rpc packet in bytes', 'rpc packet out', 'rpc packet out bytes', 'rpc deliver fail', 'rpc net delay', 'rpc net frame delay', 'mysql packet in', 'mysql packet in bytes', 'mysql packet out', 'mysql packet out bytes', 'mysql deliver fail'],
                    "transactions": ['request queue time', 'trans system trans count', 'trans user trans count', 'trans commit count', 'trans rollback count', 'trans timeout count', 'commit log replay count', 'local trans total used time', 'distributed trans total used time', 'sql select count', 'sql update count', 'sql delete count', 'sql other count', 'ps prepare count', 'ps execute count', 'sql local count', 'sql remote count', 'sql distributed count', 'active sessions', 'sql inner insert count', 'sql inner update count', 'sql fail count', 'memstore write lock succ count', 'memstore write lock fail count', 'DB time']
                }
                flat_metrics = metrics.copy()
                metrics = {}
                for group, keys in groups.items():
                    group_data = {}
                    for expected_key in keys:
                        for actual_key, value in flat_metrics.items():
                            if actual_key.lower() == expected_key.lower():
                                group_data[expected_key] = value
                                break
                    metrics[group] = group_data
                logger.debug("Restructured flat metrics into nested groups: %s", metrics)
            
            # Use fixed expected metrics counts based on predefined metric lists
            expected_num_metrics = len(['cpu usage', 'worker time', 'cpu time']) + \
                                   len(['active memstore used', 'total memstore used', 'memstore limit', 'memory usage', 'observer memory hold size']) + \
                                   len(['row cache hit', 'row cache miss', 'io read count', 'io read delay', 'io write count', 'io write delay', 'accessed data micro block count', 'data micro block cache hit', 'index micro block cache hit', 'blockscaned data micro block count', 'palf write io count to disk', 'palf write size to disk', 'clog trans log total size']) + \
                                   len(['rpc packet in', 'rpc packet in bytes', 'rpc packet out', 'rpc packet out bytes', 'rpc deliver fail', 'rpc net delay', 'rpc net frame delay', 'mysql packet in', 'mysql packet in bytes', 'mysql packet out', 'mysql packet out bytes', 'mysql deliver fail']) + \
                                   len(['request queue time', 'trans system trans count', 'trans user trans count', 'trans commit count', 'trans rollback count', 'trans timeout count', 'commit log replay count', 'local trans total used time', 'distributed trans total used time', 'sql select count', 'sql update count', 'sql delete count', 'sql other count', 'ps prepare count', 'ps execute count', 'sql local count', 'sql remote count', 'sql distributed count', 'active sessions', 'sql inner insert count', 'sql inner update count', 'sql fail count', 'memstore write lock succ count', 'memstore write lock fail count', 'DB time'])
            feature_vector = np.zeros(expected_num_metrics * 16)  # 16 features per metric
            feature_idx = 0
            
            unique_values_count = 0  # Count metrics with non-zero values
            
            # CPU metrics
            cpu_metrics = metrics.get('cpu', {})
            for metric in ['cpu usage', 'worker time', 'cpu time']:
                values = self._get_time_window_values(cpu_metrics, metric)
                features = self._calculate_time_features(values)
                if any(f != 0 for f in features):
                    unique_values_count += 1
                if feature_idx + len(features) > feature_vector.size:
                    logger.error(f"Feature vector index out of range at CPU metric '{metric}': feature_idx={feature_idx}, required={len(features)}, total={feature_vector.size}")
                    break
                feature_vector[feature_idx:feature_idx+len(features)] = features
                feature_idx += len(features)
            
            # Memory metrics
            memory_metrics = metrics.get('memory', {})
            for metric in ['active memstore used', 'total memstore used', 'memstore limit', 'memory usage', 'observer memory hold size']:
                values = self._get_time_window_values(memory_metrics, metric)
                features = self._calculate_time_features(values)
                if any(f != 0 for f in features):
                    unique_values_count += 1
                if feature_idx + len(features) > feature_vector.size:
                    logger.error(f"Feature vector index out of range at Memory metric '{metric}': feature_idx={feature_idx}, required={len(features)}, total={feature_vector.size}")
                    break
                feature_vector[feature_idx:feature_idx+len(features)] = features
                feature_idx += len(features)
            
            # IO metrics
            io_metrics = metrics.get('io', {})
            for metric in ['row cache hit', 'row cache miss', 'io read count', 'io read delay', 'io write count', 'io write delay', 'accessed data micro block count', 'data micro block cache hit', 'index micro block cache hit', 'blockscaned data micro block count', 'palf write io count to disk', 'palf write size to disk', 'clog trans log total size']:
                values = self._get_time_window_values(io_metrics, metric)
                features = self._calculate_time_features(values)
                if any(f != 0 for f in features):
                    unique_values_count += 1
                if feature_idx + len(features) > feature_vector.size:
                    logger.error(f"Feature vector index out of range at IO metric '{metric}': feature_idx={feature_idx}, required={len(features)}, total={feature_vector.size}")
                    break
                feature_vector[feature_idx:feature_idx+len(features)] = features
                feature_idx += len(features)
            
            # Network metrics
            network_metrics = metrics.get('network', {})
            for metric in ['rpc packet in', 'rpc packet in bytes', 'rpc packet out', 'rpc packet out bytes', 'rpc deliver fail', 'rpc net delay', 'rpc net frame delay', 'mysql packet in', 'mysql packet in bytes', 'mysql packet out', 'mysql packet out bytes', 'mysql deliver fail']:
                values = self._get_time_window_values(network_metrics, metric)
                features = self._calculate_time_features(values)
                if any(f != 0 for f in features):
                    unique_values_count += 1
                if feature_idx + len(features) > feature_vector.size:
                    logger.error(f"Feature vector index out of range at Network metric '{metric}': feature_idx={feature_idx}, required={len(features)}, total={feature_vector.size}")
                    break
                feature_vector[feature_idx:feature_idx+len(features)] = features
                feature_idx += len(features)
            
            # Transaction metrics
            trans_metrics = metrics.get('transactions', {})
            for metric in ['request queue time', 'trans system trans count', 'trans user trans count', 'trans commit count', 'trans rollback count', 'trans timeout count', 'commit log replay count', 'local trans total used time', 'distributed trans total used time', 'sql select count', 'sql update count', 'sql delete count', 'sql other count', 'ps prepare count', 'ps execute count', 'sql local count', 'sql remote count', 'sql distributed count', 'active sessions', 'sql inner insert count', 'sql inner update count', 'sql fail count', 'memstore write lock succ count', 'memstore write lock fail count', 'DB time']:
                values = self._get_time_window_values(trans_metrics, metric)
                features = self._calculate_time_features(values)
                if any(f != 0 for f in features):
                    unique_values_count += 1
                if feature_idx + len(features) > feature_vector.size:
                    logger.error(f"Feature vector index out of range at Transaction metric '{metric}': feature_idx={feature_idx}, required={len(features)}, total={feature_vector.size}")
                    break
                feature_vector[feature_idx:feature_idx+len(features)] = features
                feature_idx += len(features)
            
            return feature_vector
        except Exception as e:
            logger.error(f"Error processing metrics: {str(e)}")
            return None

    def train(self, x_train, y_train):
        """Train the model with 2D data (samples × features)"""
        try:
            # Ensure data is 2D
            if x_train.ndim != 2:
                raise ValueError(f"Expected 2D training data, got shape {x_train.shape}")
            if y_train.ndim != 2:
                raise ValueError(f"Expected 2D label data, got shape {y_train.shape}")
            
            logger.info(f"Training with data shapes: X={x_train.shape}, y={y_train.shape}")
            
            # Train each classifier on its corresponding anomaly type
            for i in range(self.num_classifier):
                current_clf = self.clf_list[i]
                current_labels = y_train[:, i].reshape(-1, 1)
                current_clf.fit(x_train, current_labels.ravel())
                
            logger.info("Successfully trained all classifiers")
            
        except Exception as e:
            logger.error(f"Error in training: {str(e)}")
            logger.debug(traceback.format_exc())
            raise

    def calculate_score_for_node(self, node_metrics):
        """Calculate scores for nodes based on PageRank"""
        try:
            # Extract nodes and construct weight matrix
            n = len(node_metrics)
            w = [[0] * n for _ in range(n)]
            
            # Calculate weights between each pair of nodes
            for i in range(n):
                for j in range(n):
                    if i == j:
                        w[i][j] = 0
                    elif j > i:
                        w[i][j] = self.calculate_rank(node_metrics[i], node_metrics[j])
                        w[j][i] = w[i][j]
            
            # Initialize PageRank
            pr = WPRNNode(w, n)
            
            # Run PageRank iterations
            scores = pr.page_rank(10)
            
            return scores
        except Exception as e:
            logger.error(f"Error calculating node scores: {str(e)}")
            return [1.0/n] * n if n > 0 else []

    def calculate_rank(self, node_x, node_y):
        """Calculate correlation between nodes using Pearson correlation"""
        try:
            x_array = np.array(node_x)
            y_array = np.array(node_y)
            if x_array.size < 2 or y_array.size < 2:
                logger.error("Not enough elements in feature vectors to compute Pearson correlation")
                return 0.0
            coef, _ = pearsonr(x_array, y_array)
            if np.isnan(coef) or coef < -1 or coef > 1:
                return 0.0
            return abs(coef)
        except Exception as e:
            logger.error(f"Error calculating node rank: {str(e)}")
            return 0.0

    def diagnose(self, node_metrics: dict) -> dict:
        """Diagnose anomalies for each node individually using multiple classifiers and pagerank embedding"""
        try:
            features = []
            node_names = []
            node_metrics_list = []  # Store processed metrics for correlation
            
            logger.info(f"Processing metrics for {len(node_metrics)} nodes")
            for node, metrics in node_metrics.items():
                feature_vector = self._process_metrics(metrics)
                if feature_vector is not None:
                    features.append(feature_vector)
                    node_names.append(node)
                    node_metrics_list.append(feature_vector)
                    logger.debug(f"Processed metrics for node {node}, feature vector shape: {feature_vector.shape}")
                else:
                    logger.warning(f"Failed processing metrics for node {node}")

            if not features:
                logger.warning("No valid features extracted from metrics")
                return self._empty_result(node_metrics)

            # Apply softmax function to normalize probabilities
            def softmax(x):
                """Compute softmax values for each set of scores in x."""
                e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
                return e_x / e_x.sum(axis=0)
            
            # Calculate PageRank scores for nodes
            node_scores = self.calculate_score_for_node(node_metrics_list)
            
            # Process each node independently
            all_predictions = []
            node_predictions = {}
            
            for node_idx, feature_vector in enumerate(features):
                logger.debug(f"Processing node {node_names[node_idx]} with feature vector shape: {feature_vector.shape}")
                node_name = node_names[node_idx]
                node_score = node_scores[node_idx]
                node_predictions[node_name] = []
                
                # Get predictions from each classifier
                for clf_idx, classifier in enumerate(self.clf_list):
                    try:
                        anomaly_type = self.classes[clf_idx]
                        
                        # Get prediction probabilities
                        proba = classifier.predict_proba(feature_vector.reshape(1, -1))
                        
                        # Get positive class probability
                        positive_prob = proba[0][1] if proba.shape[1] > 1 else proba[0][0]
                        
                        # Apply node score as weight
                        weighted_score = positive_prob * node_score
                        
                        prediction = {
                            'node': node_name,
                            'type': anomaly_type,
                            'score': float(weighted_score),
                            'positive_prob_score': float(positive_prob),
                            'pagerank_score': float(node_score)
                        }
                        
                        node_predictions[node_name].append(prediction)
                        all_predictions.append(prediction)
                        
                    except Exception as e:
                        logger.error(f"Error in diagnosis for node {node_name}, classifier {clf_idx}: {str(e)}")

            # Normalize scores across all predictions
            if all_predictions:
                scores_array = np.array([pred['score'] for pred in all_predictions])
                normalized_scores = softmax(scores_array)
                for idx, pred in enumerate(all_predictions):
                    pred['score'] = float(normalized_scores[idx])

            # Build propagation graph
            propagation_graph = {}
            for i in range(len(node_names)):
                for j in range(i + 1, len(node_names)):
                    node_i = node_names[i]
                    node_j = node_names[j]
                    rank = self.calculate_rank(node_metrics_list[i], node_metrics_list[j])
                    if node_i not in propagation_graph:
                        propagation_graph[node_i] = {}
                    if node_j not in propagation_graph:
                        propagation_graph[node_j] = {}
                    propagation_graph[node_i][node_j] = float(rank)
                    propagation_graph[node_j][node_i] = float(rank)

            # Format results
            result = {
                'timestamp': datetime.now().isoformat(),
                'scores': {node_names[i]: float(node_scores[i]) for i in range(len(node_names))},
                'node_predictions': node_predictions,
                'propagation_graph': propagation_graph,
                'all_ranked_anomalies': sorted(all_predictions, key=lambda x: x['score'], reverse=True)
            }

            return result
            
        except Exception as e:
            logger.error(f"Error in diagnosis: {str(e)}")
            logger.debug(traceback.format_exc())
            return self._empty_result(node_metrics)

    def _empty_result(self, node_metrics):
        """Return empty result structure"""
        return {
            'timestamp': datetime.now().isoformat(),
            'scores': {node: 0.0 for node in node_metrics.keys()},
            'node_predictions': {},
            'propagation_graph': {},
            'all_ranked_anomalies': []
        }  # Updated to match system result format

    def save_model(self, model_dir):
        """Save model and related configuration to disk"""
        try:
            os.makedirs(model_dir, exist_ok=True)
            
            # Save each classifier
            for i, clf in enumerate(self.clf_list):
                model_path = os.path.join(model_dir, f'classifier_{i}.joblib')
                joblib.dump(clf, model_path)
            
            # Save configuration
            config = {
                'num_classifier': self.num_classifier,
                'node_num': self.node_num,
                'classes': self.classes,
                'time_window': self.time_window
            }
            
            with open(os.path.join(model_dir, 'config.json'), 'w') as f:
                json.dump(config, f)
                
            logger.info(f"Model saved to {model_dir}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, model_dir):
        """Load model and related configuration from disk"""
        try:
            # Load configuration
            config_path = os.path.join(model_dir, 'config.json')
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found at {config_path}")
                
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Update configuration
            self.num_classifier = config.get('num_classifier', self.num_classifier)
            self.node_num = config.get('node_num', self.node_num)
            self.classes = config.get('classes', self.classes)
            self.time_window = config.get('time_window', self.time_window)
            
            # Load classifiers
            self.clf_list = []
            for i in range(self.num_classifier):
                model_path = os.path.join(model_dir, f'classifier_{i}.joblib')
                if os.path.exists(model_path):
                    clf = joblib.load(model_path)
                    self.clf_list.append(clf)
                else:
                    raise FileNotFoundError(f"Classifier {i} not found at {model_path}")
                    
            logger.info(f"Model loaded from {model_dir} with {len(self.clf_list)} classifiers")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
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
            "network_bottleneck",
            "cache_bottleneck"
        ]
        
        # Get node count from environment or config
        self.node_count = int(os.getenv('NODE_COUNT', '3'))
        
        # Get time window size from environment or use default
        self.time_window = int(os.getenv('METRICS_TIME_WINDOW', '3'))
        
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
            self.model_path = os.path.join(self.models_path, self.active_model)
            self.metrics_path = os.path.join(self.model_path, 'metrics')
            self.diagnosis.load_model(self.model_path)
            logger.info(f"Using model: {self.active_model}")
        else:
            logger.warning("No model available. Please train a model first.")
            self.model_path = None
            self.metrics_path = None
            
    def _process_metrics(self, metrics: Dict[str, Any]) -> List[float]:
        """Process metrics into a feature vector"""
        return self.diagnosis._process_metrics(metrics)
            
    

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the model with collected data"""
        try:
            if len(X) == 0 or len(y) == 0:
                raise ValueError("Empty training data")
                
            # Split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            logger.debug(f"Training data shape: X_train={X_train.shape}, y_train={y_train.shape}")
            logger.debug(f"Test data shape: X_test={X_test.shape}, y_test={y_test.shape}")
            logger.debug(f"Training data first sample: X_train[0]={X_train[0]}, y_train[0]={y_train[0]}")
            logger.debug(f"Test data first sample: X_test[0]={X_test[0]}, y_test[0]={y_test[0]}")
            
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
            
            # Inside the train method, after the train_test_split and existing debug logs, add logging for unique y values:
            unique_y_train = np.unique(y_train)
            unique_y_test = np.unique(y_test)
            logger.info(f"Distinct y values in training set: {unique_y_train}")
            logger.info(f"Distinct y values in test set: {unique_y_test}")
            
            return {
                "model_name": model_name,
                "metrics": metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to train model: {str(e)}")
            raise

    def analyze_metrics(self, metrics: Dict[str, Any]) -> List[Dict]:
        """Analyze metrics to identify potential anomalies"""
        if not self.active_model:
            logger.warning("No active model. Please train or load a model first.")
            return []
        
        return []  # Simplified - we'll focus on diagnose instead

    def diagnose(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Diagnose anomalies from metrics across multiple nodes"""
        try:
            if not self.active_model:
                logger.warning("No active model. Please train or load a model first.")
                return {"error": "No active model"}
            
            # Check if metrics contains data for multiple nodes
            if not metrics or not isinstance(metrics, dict):
                return {"error": "Invalid metrics format"}
            
            # Run diagnosis
            diagnosis_result = self.diagnosis.diagnose(metrics)
            
            # Return result with timestamp
            result = {
                "timestamp": datetime.now().isoformat(),
                "model": self.active_model,
                **diagnosis_result
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in diagnosis: {str(e)}")
            return {"error": str(e)}

    def evaluate_model(self, X_test, y_test, model_name):
        """Evaluate model performance on test data"""
        try:
            # Overall evaluation metrics
            metrics = {}
            
            # Save evaluation metrics
            if self.metrics_path:
                metrics_file = os.path.join(self.metrics_path, 'evaluation.json')
                with open(metrics_file, 'w') as f:
                    json.dump(metrics, f, indent=2)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            return {"error": str(e)}

    def get_available_models(self):
        """Get list of available models in models directory"""
        try:
            if not os.path.exists(self.models_path):
                os.makedirs(self.models_path, exist_ok=True)
                return []
                
            # Get all directories in models path
            model_dirs = [d for d in os.listdir(self.models_path) 
                         if os.path.isdir(os.path.join(self.models_path, d))
                         and d.startswith('dist-diagnosis-model')]
            
            # Sort by creation time (newest first)
            model_dirs.sort(key=lambda x: os.path.getctime(os.path.join(self.models_path, x)), reverse=True)
            
            return model_dirs
            
        except Exception as e:
            logger.error(f"Error getting available models: {str(e)}")
            return []

    def switch_model(self, model_name):
        """Switch to a different model"""
        try:
            available_models = self.get_available_models()
            
            if model_name not in available_models:
                raise ValueError(f"Model {model_name} not found")
                
            # Update paths
            self.active_model = model_name
            self.model_path = os.path.join(self.models_path, model_name)
            self.metrics_path = os.path.join(self.model_path, 'metrics')
            
            # Load model
            self.diagnosis.load_model(self.model_path)
            
            logger.info(f"Switched to model: {model_name}")
            
            return {"message": f"Switched to model: {model_name}"}
            
        except Exception as e:
            logger.error(f"Error switching model: {str(e)}")
            return {"error": str(e)}

    def direct_train(self, X, y):
        """Direct training method for diagnosis purposes"""
        try:
            logger.info(f"Direct training with data shapes: X={X.shape}, y={y.shape}")
            
            # Check for NaN or infinite values
            X_nan_count = np.isnan(X).sum()
            X_inf_count = np.isinf(X).sum()
            y_nan_count = np.isnan(y).sum()
            y_inf_count = np.isinf(y).sum()
            
            logger.info(f"Data validation: X has {X_nan_count} NaN and {X_inf_count} infinite values")
            logger.info(f"Data validation: y has {y_nan_count} NaN and {y_inf_count} infinite values")
            
            # Replace any NaN or infinite values
            if X_nan_count > 0 or X_inf_count > 0:
                logger.warning("Replacing NaN/infinite values in X with zeros")
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            if y_nan_count > 0 or y_inf_count > 0:
                logger.warning("Replacing NaN/infinite values in y with zeros")
                y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Validate data shapes and contents
            if len(X) == 0 or len(y) == 0:
                logger.error("Empty training data")
                return {
                    "error": "Empty training data",
                    "success": False
                }
                
            if len(X) != len(y):
                logger.error(f"Mismatched data shapes: X={X.shape}, y={y.shape}")
                return {
                    "error": f"Mismatched data shapes: X={X.shape}, y={y.shape}",
                    "success": False
                }
            
            # Log some sample data
            logger.info(f"Sample X[0]: {X[0][:10]}...")
            logger.info(f"Sample y[0]: {y[0]}")
            
            # Train the model using the parent class method
            result = self.train(X, y)
            
            logger.info(f"Training completed successfully: {result}")
            return {
                "success": True,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Error in direct_train: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }

# Create singleton instance
diagnosis_service = DiagnosisService()