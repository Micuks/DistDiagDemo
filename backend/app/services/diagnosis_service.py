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
        self.clf_list = []
        for _ in range(num_classifier):
            self.clf_list.append(XGBClassifier(objective='binary:logistic'))
        self.classes = classes_for_each_node if classes_for_each_node else [0, 1]

    def train(self, x_train, y_train):
        """Train the model with collected data"""
        train_x = self.decompose_train_x(x_train)
        train_y = self.decompose_train_y(y_train)

        for i in range(self.num_classifier):
            current_clf = self.clf_list[i]
            current_labels = train_y[:, i].reshape(-1, 1)
            current_clf.fit(train_x, current_labels)

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
                return {
                    'scores': {},
                    'anomalies': [],
                    'propagation_graph': {}
                }

            # Get predictions from each classifier
            results = {}
            for node_idx, node in enumerate(node_names):
                node_results = {}
                for clf_idx, clf in enumerate(self.clf_list):
                    if clf_idx < len(self.classes):
                        proba = clf.predict_proba(np.array([features[node_idx]]))[0]
                        if len(proba) > 1:  # Binary classification
                            score = float(proba[1])
                        else:  # Single class
                            score = float(proba[0])
                        node_results[self.classes[clf_idx]] = score
                results[node] = node_results

            # Calculate anomaly scores
            scores = {}
            anomalies = []
            for node, node_results in results.items():
                max_score = max(node_results.values())
                scores[node] = max_score
                if max_score > 0.7:  # Threshold for anomaly detection
                    anomaly_type = max(node_results.items(), key=lambda x: x[1])[0]
                    anomalies.append({
                        'node': node,
                        'type': anomaly_type,
                        'score': max_score,
                        'timestamp': datetime.now().isoformat()
                    })

            return {
                'scores': scores,
                'anomalies': sorted(anomalies, key=lambda x: x['score'], reverse=True),
                'propagation_graph': {}  # TODO: Implement propagation analysis
            }

        except Exception as e:
            logger.error(f"Error in diagnose: {str(e)}")
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
        train_x = []
        for case in composed_train_x:
            for node_x in case:
                train_x.append(node_x)
        return np.array(train_x)

    def decompose_train_y(self, composed_train_y):
        """Decompose training labels"""
        train_y = []
        for case in composed_train_y:
            for node_label in case:
                train_y.append(node_label)
        return np.array(train_y)

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
            # Ensure data directory exists
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
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
            # Get diagnosis results
            diagnosis_result = self.diagnosis.diagnose(metrics)
            
            # Return ranked anomalies
            return diagnosis_result.get('anomalies', [])
            
        except Exception as e:
            logger.error("Error in analyze_metrics: %s", str(e))
            return []

# Create singleton instance
diagnosis_service = DiagnosisService()