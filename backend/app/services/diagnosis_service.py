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
        self.classes = classes_for_each_node if classes_for_each_node else ['cpu', 'memory', 'io', 'network']
        self.clf_list = [XGBClassifier(objective='binary:logistic') for _ in range(num_classifier)]
        logger.info(f"Initialized {len(self.clf_list)} classifiers for {len(self.classes)} classes")

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

    def diagnose(self, node_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Diagnose anomalies using the original algorithm's mechanics"""
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
                    
                    if max_score > 0.1:  # Configurable threshold
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
                        if correlation > 0.3:  # Configurable threshold
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
    def __init__(self, model_name=None):
        # Base models directory
        self.models_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            '../../models/'
        ))
        
        # Define anomaly types that can be detected
        self.anomaly_types = ['cpu', 'memory', 'io', 'network']
        
        # Get node count from environment or config
        self.node_count = int(os.getenv('NODE_COUNT', '3'))
        
        # Initialize DistDiagnosis with multi-node support
        self.diagnosis = DistDiagnosis(
            num_classifier=len(self.anomaly_types),
            classes_for_each_node=self.anomaly_types,
            node_num=self.node_count
        )
        
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
            
            # Generate model name with timestamp
            model_name = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
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
            model = self.diagnosis.clf_list[0]  # Get primary classifier
            y_pred = model.predict(X_test)
            
            metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, average='weighted')),
                'recall': float(recall_score(y_test, y_pred, average='weighted')),
                'f1_score': float(f1_score(y_test, y_pred, average='weighted')),
                'training_time': time.time() - self.training_start_time
            }
            
            # Save metrics
            metrics_file = os.path.join(self.metrics_path, f"{model_name}_metrics.json")
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f)
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise

    def get_model_performance(self, model_name):
        """Get stored performance metrics for a model"""
        try:
            # If looking for current active model metrics
            if model_name == self.active_model:
                metrics_path = self.metrics_path
            else:
                # Look for metrics in the specific model directory
                model_dir = os.path.join(self.models_path, model_name)
                metrics_path = os.path.join(model_dir, 'metrics')
                
            metrics_file = os.path.join(metrics_path, f"{model_name}_metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file) as f:
                    return json.load(f)
            
            return None
        except Exception as e:
            logger.error(f"Error loading model metrics for {model_name}: {str(e)}")
            return None

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