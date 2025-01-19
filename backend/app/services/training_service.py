import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple
import numpy as np
import shutil
import threading

logger = logging.getLogger(__name__)

class TrainingService:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        # Get data directory from environment variable or use default
        self.data_dir = os.getenv('TRAINING_DATA_DIR', 'data/training')
        
        # Ensure absolute path
        if not os.path.isabs(self.data_dir):
            self.data_dir = os.path.abspath(self.data_dir)
            
        # Create directory with full permissions if it doesn't exist
        os.makedirs(self.data_dir, mode=0o777, exist_ok=True)
        logger.info(f"Initialized training data directory at: {self.data_dir}")
        
        self.current_case = None
        self.current_anomaly = None
        self.metrics_buffer = []
        self.normal_state_buffer = []
        self.is_collecting_normal = False
        self.collection_start_time = None
        
        # Add lock for thread safety
        self._lock = threading.Lock()
        
        # Log existing data
        self._log_existing_data()
        
        self._initialized = True
        
    def _log_existing_data(self):
        """Log information about existing training data"""
        if os.path.exists(self.data_dir):
            cases = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
            logger.info(f"Found {len(cases)} existing training cases:")
            for case in cases:
                case_dir = os.path.join(self.data_dir, case)
                metrics_file = os.path.join(case_dir, "metrics.json")
                label_file = os.path.join(case_dir, "label.json")
                
                metrics_size = os.path.getsize(metrics_file) if os.path.exists(metrics_file) else 0
                has_label = os.path.exists(label_file)
                
                logger.info(f"  - {case}:")
                logger.info(f"    Metrics file: {metrics_size} bytes")
                logger.info(f"    Label file: {'Present' if has_label else 'Missing'}")
        
    def start_collection(self, anomaly_type: str, node: str):
        """Start collecting training data for an anomaly case"""
        self.collection_start_time = datetime.now().isoformat()
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.current_case = f"case_{anomaly_type}_{node}_{timestamp}"
        self.current_anomaly = {
            "type": anomaly_type,
            "node": node,
            "start_time": self.collection_start_time
        }
        self.metrics_buffer = []
        logger.info(f"Started collecting training data for {self.current_case} at {self.collection_start_time}")
        
    def start_normal_collection(self):
        """Start collecting normal state data"""
        with self._lock:
            if self.is_collecting_normal:
                logger.warning("Normal state collection already active")
                return
                
            try:
                # Set collection flags and create new case
                self.collection_start_time = datetime.now().isoformat()
                self.is_collecting_normal = True
                self.normal_state_buffer = []
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                self.current_case = f"case_normal_{timestamp}"
                
                # Ensure data directory exists
                case_dir = os.path.join(self.data_dir, self.current_case)
                os.makedirs(os.path.dirname(case_dir), exist_ok=True)
                
                logger.info(f"Started collecting normal state data with case ID: {self.current_case}")
                logger.info(f"Collection start time: {self.collection_start_time}")
                logger.info(f"Data will be saved to: {case_dir}")
                logger.debug(f"Current state - is_collecting_normal: {self.is_collecting_normal}, current_case: {self.current_case}")
            except Exception as e:
                logger.error(f"Failed to start normal collection: {str(e)}")
                self.is_collecting_normal = False
                self.current_case = None
                self.collection_start_time = None
                self.normal_state_buffer = []
                raise
        
    def stop_normal_collection(self):
        """Stop collecting normal state data and save it"""
        with self._lock:
            if not self.is_collecting_normal:
                logger.warning("No active normal state collection")
                return
                
            if not self.normal_state_buffer:
                logger.warning("No metrics collected during normal state collection")
                self.is_collecting_normal = False
                self.current_case = None
                self.collection_start_time = None
                return
                
            case_dir = os.path.join(self.data_dir, self.current_case)
            temp_dir = f"{case_dir}_temp"
            save_successful = False
            
            try:
                # Create data directory if it doesn't exist
                os.makedirs(self.data_dir, exist_ok=True)
                # Create a temporary directory for atomic save
                os.makedirs(temp_dir, exist_ok=True)
                
                # Save metrics data
                metrics_file = os.path.join(temp_dir, "metrics.json")
                with open(metrics_file, "w") as f:
                    json.dump(self.normal_state_buffer, f, indent=2)
                    
                # Save label indicating normal state
                label_file = os.path.join(temp_dir, "label.json")
                with open(label_file, "w") as f:
                    json.dump({
                        "type": "normal",
                        "start_time": self.normal_state_buffer[0]["timestamp"]
                    }, f, indent=2)
                    
                # Verify files were written correctly
                if not os.path.exists(metrics_file) or os.path.getsize(metrics_file) == 0:
                    raise Exception("Failed to write metrics file or file is empty")
                if not os.path.exists(label_file) or os.path.getsize(label_file) == 0:
                    raise Exception("Failed to write label file or file is empty")
                    
                # Atomically move the temporary directory to the final location
                if os.path.exists(case_dir):
                    shutil.rmtree(case_dir)
                shutil.move(temp_dir, case_dir)
                    
                logger.info(f"Successfully saved normal state data:")
                logger.info(f"  - Metrics file: {metrics_file} ({os.path.getsize(metrics_file)} bytes)")
                logger.info(f"  - Label file: {label_file} ({os.path.getsize(label_file)} bytes)")
                save_successful = True
                
            except Exception as e:
                logger.error(f"Failed to save normal state data: {str(e)}")
                if os.path.exists(temp_dir):
                    try:
                        shutil.rmtree(temp_dir)
                    except Exception as cleanup_error:
                        logger.error(f"Failed to clean up temporary directory: {cleanup_error}")
            finally:
                # Always clear the state
                self.is_collecting_normal = False
                self.normal_state_buffer = []
                self.current_case = None
                self.collection_start_time = None
                logger.info("Cleared normal collection state")
                
                # Only raise the exception if we failed to save AND had data to save
                if not save_successful and len(self.normal_state_buffer) > 0:
                    raise Exception("Failed to save normal state data")
        
    def stop_collection(self):
        """Stop collecting training data and save the case"""
        with self._lock:
            if not self.current_case:
                logger.warning("No active collection")
                return
                
            if not self.metrics_buffer:
                logger.warning("No metrics collected")
                self.current_case = None
                self.current_anomaly = None
                self.collection_start_time = None
                return
                
            case_dir = os.path.join(self.data_dir, self.current_case)
            temp_dir = f"{case_dir}_temp"
            
            try:
                # Create data directory if it doesn't exist
                os.makedirs(self.data_dir, exist_ok=True)
                # Create temporary directory
                os.makedirs(temp_dir, exist_ok=True)
                
                # Save metrics data
                metrics_file = os.path.join(temp_dir, "metrics.json")
                with open(metrics_file, "w") as f:
                    json.dump(self.metrics_buffer, f, indent=2)
                    
                # Save anomaly label
                label_file = os.path.join(temp_dir, "label.json")
                with open(label_file, "w") as f:
                    json.dump(self.current_anomaly, f, indent=2)
                    
                # Verify files were written correctly
                if not os.path.exists(metrics_file) or os.path.getsize(metrics_file) == 0:
                    raise Exception("Failed to write metrics file or file is empty")
                if not os.path.exists(label_file) or os.path.getsize(label_file) == 0:
                    raise Exception("Failed to write label file or file is empty")
                    
                # Atomically move the temporary directory to the final location
                if os.path.exists(case_dir):
                    shutil.rmtree(case_dir)
                shutil.move(temp_dir, case_dir)
                
                logger.info(f"Successfully saved training data for {self.current_case}:")
                logger.info(f"  - Metrics file: {metrics_file} ({os.path.getsize(metrics_file)} bytes)")
                logger.info(f"  - Label file: {label_file} ({os.path.getsize(label_file)} bytes)")
            except Exception as e:
                logger.error(f"Failed to save training data: {str(e)}")
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                raise
            finally:
                # Always clear the state
                self.current_case = None
                self.current_anomaly = None
                self.metrics_buffer = []
                self.collection_start_time = None
                logger.info("Cleared collection state")
        
    def add_metrics(self, metrics: Dict[str, Any]):
        """Add metrics to current collection"""
        with self._lock:
            logger.debug(f"Received metrics - State: is_collecting_normal={self.is_collecting_normal}, current_case={self.current_case}")
            
            if not self.current_case:
                logger.warning("No active case - metrics will not be collected")
                logger.debug(f"Collection state: is_collecting_normal={self.is_collecting_normal}, collection_start_time={self.collection_start_time}")
                return
                
            if not self.collection_start_time:
                logger.warning("No collection start time - metrics will not be collected")
                return

            logger.debug(f"Processing metrics for case: {self.current_case}")
            logger.debug(f"Collection mode: {'normal' if self.is_collecting_normal else 'anomaly'}")
            logger.debug(f"Number of nodes in metrics: {len(metrics)}")

            # Process all historical data points
            for node_ip, node_data in metrics.items():
                logger.debug(f"Processing metrics for node: {node_ip}")
                
                # Get all timestamps from CPU data to use as reference
                timestamps = []
                if 'cpu' in node_data and 'util' in node_data['cpu']:
                    timestamps = list(node_data['cpu']['util'].keys())
                    timestamps.remove('latest')  # Remove 'latest' key
                elif 'cpu' in node_data:
                    # Handle psutil format which might only have 'latest'
                    timestamps = ['latest']
                
                if not timestamps:
                    logger.warning(f"No timestamps found in metrics for node {node_ip}")
                    continue

                # Process each timestamp
                for timestamp in timestamps:
                    filtered_metrics = {}
                    try:
                        filtered_node_data = {}
                        for category, category_data in node_data.items():
                            if category == 'io':
                                # Handle IO metrics
                                filtered_node_data[category] = {
                                    'aggregated': {
                                        metric: {
                                            'value': data.get(timestamp, data.get('latest', 0))
                                        }
                                        for metric, data in category_data['aggregated'].items()
                                    }
                                }
                            else:
                                # For other categories
                                filtered_node_data[category] = {
                                    metric: {
                                        'value': data.get(timestamp, data.get('latest', 0))
                                    }
                                    for metric, data in category_data.items()
                                }
                        filtered_metrics[node_ip] = filtered_node_data

                        metric_entry = {
                            "timestamp": timestamp if timestamp != 'latest' else datetime.now().isoformat(),
                            "metrics": filtered_metrics
                        }

                        if self.is_collecting_normal:
                            self.normal_state_buffer.append(metric_entry)
                        else:
                            self.metrics_buffer.append(metric_entry)

                    except Exception as e:
                        logger.error(f"Error processing metrics for timestamp {timestamp}: {str(e)}")
                        continue

            # Log collection progress
            if self.is_collecting_normal:
                logger.info(f"Total normal state data points collected: {len(self.normal_state_buffer)}")
            else:
                logger.info(f"Total anomaly data points collected: {len(self.metrics_buffer)}")
        
    def get_dataset_stats(self) -> Dict[str, int]:
        """Get statistics about the collected dataset"""
        stats = {"normal": 0, "anomaly": 0}
        anomaly_stats = {
            "cpu": 0,
            "memory": 0,
            "io": 0,
            "network": 0,
            "unknown": 0
        }
        
        if not os.path.exists(self.data_dir):
            return stats
            
        for case_dir in os.listdir(self.data_dir):
            case_path = os.path.join(self.data_dir, case_dir)
            if not os.path.isdir(case_path):
                continue
                
            label_file = os.path.join(case_path, "label.json")
            metrics_file = os.path.join(case_path, "metrics.json")
            
            if not os.path.exists(label_file) or not os.path.exists(metrics_file):
                logger.warning(f"Incomplete case directory {case_dir}: missing {'label' if not os.path.exists(label_file) else 'metrics'} file")
                continue
                
            try:
                with open(label_file) as f:
                    label_data = json.load(f)
                with open(metrics_file) as f:
                    metrics_data = json.load(f)
                    
                if label_data.get("type") == "normal":
                    stats["normal"] += len(metrics_data)
                else:
                    stats["anomaly"] += len(metrics_data)
                    # Count specific anomaly types
                    anomaly_type = label_data.get("type", "unknown")
                    base_type = anomaly_type.replace("_stress", "")
                    if base_type in anomaly_stats:
                        anomaly_stats[base_type] += len(metrics_data)
                    else:
                        anomaly_stats["unknown"] += len(metrics_data)
            except Exception as e:
                logger.error(f"Error reading case {case_dir}: {str(e)}")
                
        logger.info(f"Dataset statistics: {stats}")
        logger.info(f"Anomaly type distribution: {anomaly_stats}")
        return stats
        
    def is_dataset_balanced(self, threshold: float = 0.3) -> bool:
        """Check if the dataset is balanced within the threshold"""
        stats = self.get_dataset_stats()
        total = stats["normal"] + stats["anomaly"]
        
        if total == 0:
            logger.warning("No training data available")
            return False
            
        normal_ratio = stats["normal"] / total
        anomaly_ratio = stats["anomaly"] / total
        
        is_balanced = abs(normal_ratio - anomaly_ratio) <= threshold
        logger.info(f"Dataset balance check: normal_ratio={normal_ratio:.2f}, anomaly_ratio={anomaly_ratio:.2f}, is_balanced={is_balanced}")
        return is_balanced

    def get_training_data(self) -> tuple:
        """Get all training data for model training"""
        if not os.path.exists(self.data_dir):
            logger.warning("Training data directory does not exist")
            return [], []
            
        X = []  # Metrics data
        y = []  # Labels
        
        # Load all case directories
        for case_dir in os.listdir(self.data_dir):
            case_path = os.path.join(self.data_dir, case_dir)
            if not os.path.isdir(case_path):
                continue
                
            try:
                # Load metrics
                metrics_file = os.path.join(case_path, "metrics.json")
                label_file = os.path.join(case_path, "label.json")
                
                if not os.path.exists(metrics_file) or not os.path.exists(metrics_file):
                    logger.warning(f"Skipping incomplete case directory {case_dir}")
                    continue
                    
                with open(metrics_file) as f:
                    metrics_data = json.load(f)
                with open(label_file) as f:
                    label_data = json.load(f)
                    
                # Process metrics into features
                features = self._process_metrics(metrics_data)
                if features is not None:
                    X.extend(features)  # Extend instead of append since features is a list of feature vectors
                    # Create label vector (all zeros for normal state)
                    if label_data.get("type") == "normal":
                        y.extend([np.zeros(4) for _ in range(len(features))])  # 4 anomaly types
                    else:
                        y.extend([self._create_label_vector(label_data) for _ in range(len(features))])
                    
                    logger.info(f"Processed {len(features)} samples from {case_dir}")
                else:
                    logger.warning(f"No valid features extracted from {case_dir}")
                    
            except Exception as e:
                logger.error(f"Error loading training data from {case_dir}: {str(e)}")
                continue
                
        logger.info(f"Total samples loaded: {len(X)}")
        return np.array(X), np.array(y)

    def _process_metrics(self, metrics_data: List[Dict]) -> List[np.ndarray]:
        """Process raw metrics into feature vectors"""
        if not metrics_data:
            logger.warning("No metrics data to process")
            return None
            
        # Extract features from metrics
        features = []
        for entry in metrics_data:
            metrics = entry["metrics"]
            
            # Process metrics for each node
            for node_ip, node_metrics in metrics.items():
                feature_vector = []
                
                try:
                    # CPU metrics
                    cpu_metrics = node_metrics.get("cpu", {})
                    if isinstance(cpu_metrics, dict) and "util" in cpu_metrics:
                        feature_vector.extend([
                            float(cpu_metrics.get("util", {}).get("latest", 0)),
                            float(cpu_metrics.get("system", {}).get("latest", 0)),
                            float(cpu_metrics.get("user", {}).get("latest", 0)),
                            float(cpu_metrics.get("wait", {}).get("latest", 0))
                        ])
                    else:
                        # Handle psutil format
                        feature_vector.extend([
                            float(cpu_metrics.get("latest", 0)),
                            0, 0, 0  # No detailed CPU metrics in psutil
                        ])
                    
                    # Memory metrics
                    memory_metrics = node_metrics.get("memory", {})
                    feature_vector.extend([
                        float(memory_metrics.get("used", {}).get("latest", 0)),
                        float(memory_metrics.get("free", {}).get("latest", 0)),
                        float(memory_metrics.get("cach", {}).get("latest", 0))
                    ])
                    
                    # IO metrics
                    io_metrics = node_metrics.get("io", {}).get("aggregated", {})
                    feature_vector.extend([
                        float(io_metrics.get("rs", {}).get("latest", 0)),  # read ops
                        float(io_metrics.get("ws", {}).get("latest", 0)),  # write ops
                        float(io_metrics.get("util", {}).get("latest", 0))  # disk utilization
                    ])
                    
                    # Network metrics
                    network_metrics = node_metrics.get("network", {})
                    feature_vector.extend([
                        float(network_metrics.get("bytin", {}).get("latest", 0)),  # bytes in
                        float(network_metrics.get("bytout", {}).get("latest", 0)), # bytes out
                        float(network_metrics.get("pktin", {}).get("latest", 0))   # packets in
                    ])
                    
                    features.append(np.array(feature_vector))
                except Exception as e:
                    logger.error(f"Error processing metrics for node {node_ip}: {str(e)}")
                    continue
                
        if not features:
            logger.warning("No valid feature vectors extracted from metrics data")
            return None
            
        return features

    def _create_label_vector(self, label_data: Dict) -> np.ndarray:
        """Create a label vector for the anomaly type"""
        # Define the mapping of anomaly types to indices
        # Map both short and full names of anomaly types
        anomaly_types = {
            'cpu': 0,
            'cpu_stress': 0,
            'memory': 1,
            'memory_stress': 1,
            'io': 2,
            'io_stress': 2,
            'network': 3,
            'network_stress': 3
        }
        
        label_vector = np.zeros(4)  # 4 basic types: cpu, memory, io, network
        
        anomaly_type = label_data.get("type")
        if anomaly_type in anomaly_types:
            label_vector[anomaly_types[anomaly_type]] = 1
        else:
            logger.warning(f"Unknown anomaly type: {anomaly_type}")
            
        return label_vector

# Create singleton instance
training_service = TrainingService() 