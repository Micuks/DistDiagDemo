import sys
from pathlib import Path

# Add project root and dist_diagnosis root to Python path
project_root = Path(__file__).parent.parent.parent.parent
dist_diagnosis_root = project_root / 'dist_diagnosis'
sys.path.extend([str(project_root), str(dist_diagnosis_root)])

from dist_diagnosis.dist_diagnosis.dist_diagnosis import DistDiagnosis
import numpy as np
from typing import List, Dict
from datetime import datetime
import os
import logging

logger = logging.getLogger(__name__)

class DiagnosisService:
    def __init__(self):
        self.model_path = os.getenv('MODEL_PATH', 'models/anomaly_detector')
        # Define anomaly types that can be detected
        self.anomaly_types = ['cpu', 'io', 'buffer', 'net', 'load']
        # Initialize DistDiagnosis with multi-node support and all anomaly types
        self.diagnosis = DistDiagnosis(
            num_classifier=len(self.anomaly_types), 
            classes_for_each_node=self.anomaly_types,
            node_num=3  # Default to 3 nodes, can be adjusted based on cluster size
        )
        if os.path.exists(self.model_path):
            self.diagnosis.load_model(self.model_path)
        
        # Define metrics needed for each anomaly type
        self.feature_columns = [
            'cpu_util', 'cpu_sys', 'cpu_user', 'cpu_iowait',  # CPU metrics
            'memory_used', 'memory_free', 'memory_cached',     # Memory metrics
            'disk_read', 'disk_write', 'disk_iops',           # IO metrics
            'net_recv', 'net_send', 'net_packets',            # Network metrics
            'active_sessions', 'sql_response_time',           # Database metrics
            'buffer_hit', 'buffer_miss', 'buffer_ratio'       # Buffer metrics
        ]

    async def analyze_metrics(self, metrics: List[Dict]) -> List[Dict]:
        """Analyze metrics and return anomaly ranks"""
        if not metrics:
            return []

        try:
            # Group metrics by node
            nodes = {}
            for metric in metrics:
                node_name = metric.get('node_name', 'default')
                if node_name not in nodes:
                    nodes[node_name] = []
                nodes[node_name].append(metric)

            # Convert metrics to format expected by DistDiagnosis
            node_metrics = {}
            for node_name, node_data in nodes.items():
                node_metrics[node_name] = {
                    m['timestamp'].isoformat(): {
                        col: float(m.get(col, 0)) 
                        for col in self.feature_columns
                        if col in m
                    }
                    for m in node_data
                }

            # Perform diagnosis
            diagnosis_result = self.diagnosis.diagnose(node_metrics)
            
            # Convert diagnosis result to ranked anomalies
            ranked_anomalies = []
            for node, score in diagnosis_result['scores'].items():
                for anomaly in diagnosis_result.get('anomalies', []):
                    if anomaly.startswith(f"{node}."):
                        anomaly_type = anomaly.split('.')[1]
                        ranked_anomalies.append({
                            'node': node,
                            'type': anomaly_type,
                            'score': score,
                            'timestamp': datetime.now().isoformat()
                        })

            # Sort by score descending
            ranked_anomalies.sort(key=lambda x: x['score'], reverse=True)
            return ranked_anomalies

        except Exception as e:
            logger.error(f"Error in analyze_metrics: {str(e)}")
            return []

    def train(self, training_data: List[Dict]):
        """Train the diagnosis model with historical data"""
        try:
            # Convert training data to format expected by DistDiagnosis
            X_train = []  # Features
            y_train = []  # Labels
            
            # Group data by timestamp to create training samples
            timestamp_groups = {}
            for data in training_data:
                ts = data['timestamp'].isoformat()
                if ts not in timestamp_groups:
                    timestamp_groups[ts] = []
                timestamp_groups[ts].append(data)
            
            # Create training samples
            for ts_data in timestamp_groups.values():
                # Create feature vectors for each node
                node_features = []
                node_labels = []
                
                for node_data in ts_data:
                    # Extract features
                    features = [float(node_data.get(col, 0)) for col in self.feature_columns]
                    node_features.append(features)
                    
                    # Extract labels (anomaly types)
                    labels = [1 if node_data.get(f'is_{atype}', False) else 0 
                             for atype in self.anomaly_types]
                    node_labels.append(labels)
                
                X_train.append(node_features)
                y_train.append(node_labels)
            
            # Train the model
            self.diagnosis.train(np.array(X_train), np.array(y_train))
            
            # Save the model
            if self.model_path:
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                self.diagnosis.save_model(self.model_path)
                
        except Exception as e:
            logger.error(f"Error in training: {str(e)}")
            raise