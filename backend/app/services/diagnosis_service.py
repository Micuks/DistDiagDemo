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
        self.classes = classes_for_each_node if classes_for_each_node else [0, 1]
        # Initialize classifiers with proper number based on classes
        self.clf_list = []
        for _ in range(len(self.classes)):
            self.clf_list.append(XGBClassifier(objective='binary:logistic'))
        logger.info(f"Initialized {len(self.clf_list)} classifiers for {len(self.classes)} classes")

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

    def diagnose(self, node_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Diagnose anomalies from node metrics"""
        try:
            # Process metrics into feature vectors
            features = []
            node_names = []
            for node, metrics in node_metrics.items():
                feature_vector = self._process_metrics(metrics)
                if feature_vector is not None:
                    features.append(feature_vector)
                    node_names.append(node)

            if not features:
                logger.warning("No valid features extracted from metrics")
                return {
                    'scores': {},
                    'anomalies': [],
                    'propagation_graph': {}
                }

            logger.debug(f"Processing features for {len(node_names)} nodes")
            
            # Get predictions from each classifier
            results = {}
            for node_idx, node in enumerate(node_names):
                node_results = {}
                feature = np.array([features[node_idx]])  # Reshape for prediction
                
                logger.debug(f"Making predictions for node {node}")
                for clf_idx, clf in enumerate(self.clf_list):
                    try:
                        # Get probability predictions
                        proba = clf.predict_proba(feature)[0]
                        # For binary classification, take the probability of anomaly (class 1)
                        score = float(proba[1]) if len(proba) > 1 else float(proba[0])
                        anomaly_type = self.classes[clf_idx]
                        node_results[anomaly_type] = score
                        logger.debug(f"Node {node}, Type {anomaly_type}: Score {score:.3f}")
                    except Exception as e:
                        logger.error(f"Error predicting with classifier {clf_idx}: {str(e)}")
                        continue
                
                results[node] = node_results

            # Calculate anomaly scores and rank them
            scores = {}
            anomalies = []
            threshold = 0.1  # Lowered threshold for testing
            
            for node, node_results in results.items():
                if not node_results:
                    continue
                    
                # Find the highest scoring anomaly type
                max_score = max(node_results.values())
                max_type = max(node_results.items(), key=lambda x: x[1])[0]
                scores[node] = max_score
                
                logger.debug(f"Node {node} max score: {max_score:.3f} ({max_type})")
                
                if max_score > threshold:
                    anomalies.append({
                        'node': node,
                        'type': max_type,
                        'score': max_score,
                        'timestamp': datetime.now().isoformat()
                    })
                    logger.info(f"Detected {max_type} anomaly on {node} with score {max_score:.3f}")

            # Sort anomalies by score
            anomalies.sort(key=lambda x: x['score'], reverse=True)
            
            logger.info(f"Found {len(anomalies)} anomalies above threshold {threshold}")
            return {
                'scores': scores,
                'anomalies': anomalies,
                'propagation_graph': {}
            }

        except Exception as e:
            logger.error(f"Error in diagnose: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                'scores': {node: 0.0 for node in node_metrics.keys()},
                'anomalies': [],
                'propagation_graph': {}
            }

    def _process_metrics(self, metrics: Dict[str, Any]) -> np.ndarray:
        """Process metrics into feature vector"""
        try:
            feature_vector = []
            
            # CPU metrics
            cpu_metrics = metrics.get('cpu', {})
            feature_vector.extend([
                float(cpu_metrics.get('util', {}).get('latest', 0)),
                float(cpu_metrics.get('system', {}).get('latest', 0)),
                float(cpu_metrics.get('user', {}).get('latest', 0)),
                float(cpu_metrics.get('wait', {}).get('latest', 0))
            ])
            
            # Memory metrics
            memory_metrics = metrics.get('memory', {})
            feature_vector.extend([
                float(memory_metrics.get('used', {}).get('latest', 0)),
                float(memory_metrics.get('free', {}).get('latest', 0)),
                float(memory_metrics.get('cach', {}).get('latest', 0))
            ])
            
            # IO metrics
            io_metrics = metrics.get('io', {}).get('aggregated', {})
            feature_vector.extend([
                float(io_metrics.get('rs', {}).get('latest', 0)),
                float(io_metrics.get('ws', {}).get('latest', 0)),
                float(io_metrics.get('util', {}).get('latest', 0))
            ])
            
            # Network metrics
            network_metrics = metrics.get('network', {})
            feature_vector.extend([
                float(network_metrics.get('bytin', {}).get('latest', 0)),
                float(network_metrics.get('bytout', {}).get('latest', 0)),
                float(network_metrics.get('pktin', {}).get('latest', 0))
            ])
            
            return np.array(feature_vector)
        except Exception as e:
            logger.error(f"Error processing metrics: {str(e)}")
            return None

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
    def __init__(self):
        self.model_path = os.getenv('MODEL_PATH', 'models/anomaly_detector')
        # Define anomaly types that can be detected
        self.anomaly_types = ['cpu', 'memory', 'io', 'network']
        
        # Initialize DistDiagnosis with multi-node support
        self.diagnosis = DistDiagnosis(
            num_classifier=len(self.anomaly_types),
            classes_for_each_node=self.anomaly_types,
            node_num=3  # Default to 3 nodes, can be adjusted based on cluster size
        )
        
        # Load existing model if available
        if os.path.exists(self.model_path):
            try:
                self.diagnosis.load_model(self.model_path)
                logger.info("Loaded existing model from %s", self.model_path)
            except Exception as e:
                logger.error("Failed to load model: %s", str(e))

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the model with collected data"""
        try:
            # Input validation
            if len(X) == 0 or len(y) == 0:
                raise ValueError("Empty training data")
                
            if len(X) != len(y):
                raise ValueError(f"Mismatched data dimensions: X={len(X)}, y={len(y)}")
                
            logger.info(f"Training model with {len(X)} samples")
            logger.debug(f"X shape: {X.shape}, y shape: {y.shape}")
            
            # Ensure data directory exists
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Reshape data if needed
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
            if len(y.shape) == 1:
                y = y.reshape(1, -1)
                
            # Train the model
            self.diagnosis.train(X, y)
            
            # Save the model
            self.diagnosis.save_model(self.model_path)
            logger.info("Model trained and saved successfully")
        except Exception as e:
            logger.error("Failed to train model: %s", str(e))
            raise

    def analyze_metrics(self, metrics: Dict[str, Any]) -> List[Dict]:
        """Analyze metrics and return anomaly ranks"""
        try:
            if not metrics:
                logger.warning("Empty metrics received")
                return []
                
            # Get diagnosis results
            diagnosis_result = self.diagnosis.diagnose(metrics)
            
            # Return ranked anomalies
            return diagnosis_result.get('anomalies', [])
            
        except Exception as e:
            logger.error("Error in analyze_metrics: %s", str(e))
            return []

# Create singleton instance
diagnosis_service = DiagnosisService()