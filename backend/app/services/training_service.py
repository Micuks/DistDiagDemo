import os
import json
import time
import logging
import threading
import asyncio
import ijson
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Dict, List, Any, Optional
import numpy as np

from async_lru import alru_cache

logger = logging.getLogger(__name__)

def timed_lru_cache(seconds: int, maxsize: int = 128):
    def wrapper_decorator(func):
        func = lru_cache(maxsize=maxsize)(func)
        func.lifetime = seconds
        func.expiration = time.time() + seconds

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            if time.time() >= func.expiration:
                func.cache_clear()
                func.expiration = time.time() + func.lifetime
            return func(*args, **kwargs)

        return wrapped_func
    return wrapper_decorator

class TrainingService:
    _instance = None
    _lock = None  # Will be initialized in __init__
    _stats_cache = None
    _last_stats_update = None
    STATS_CACHE_TTL = 60  # Cache TTL in seconds
    _stats_executor = ThreadPoolExecutor(max_workers=1)
    _stats_lock = asyncio.Lock()
    _request_lock = asyncio.Lock()
    _last_request_time = 0
    _cache_task = None

    def __new__(cls):
        if cls._instance is None:
            with threading.Lock():  # Use threading.Lock only for singleton creation
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        # Initialize asyncio lock and get event loop
        self._lock = asyncio.Lock()
        try:
            self._event_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._event_loop)
        
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
        
        # Add dataset balance tracking
        self.normal_samples = 0
        self.anomaly_samples = 0
        self.balance_threshold = 0.3  # 30% threshold for balance
        
        # Log existing data
        self._log_existing_data()
        
        self._initialized = True
        
        # Start background cache maintenance
        self._cache_task = asyncio.create_task(self._maintain_cache())

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
        
    async def start_collection(self, anomaly_type: str, node: str):
        """Start collecting training data for an anomaly case"""
        async with self._lock:
            # Validate node parameter
            if not node:
                logger.warning("No node specified for anomaly collection. Using default node.")
                node = "default"  # Use a default node if none specified
                
            self.collection_start_time = datetime.now().isoformat()
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            self.current_case = f"case_{anomaly_type}_{node}_{timestamp}"
            self.current_anomaly = {
                "type": anomaly_type,
                "node": node,
                "start_time": self.collection_start_time
            }
            
            # Clear buffers
            self.metrics_buffer = []
            
            # No pre/post collection - directly start anomaly collection
            self.is_collecting_normal = False
            logger.info(f"Started anomaly data collection for {self.current_case} at {self.collection_start_time}")

    async def _clear_collection_state(self):
        """Clear all collection state variables"""
        async with self._lock:
            self.current_case = None
            self.current_anomaly = None
            self.metrics_buffer = []
            self.collection_start_time = None
            self.is_collecting_normal = False
            logger.info("Cleared collection state")

    async def start_normal_collection(self):
        """Start collecting normal state data"""
        async with self._lock:
            # If there's an existing cache task, cancel it first
            if self._cache_task and not self._cache_task.done():
                self._cache_task.cancel()
                try:
                    await self._cache_task
                except asyncio.CancelledError:
                    pass
            
            self.collection_start_time = datetime.now().isoformat()
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            self.current_case = f"case_normal_{timestamp}"
            self.current_anomaly = None  # No anomaly for normal collection
            
            # Clear buffers
            self.metrics_buffer = []
            self.normal_state_buffer = []
            
            # Set collection state
            self.is_collecting_normal = True
            logger.info(f"Started collecting normal state data for {self.current_case}")
            
            # Start background cache maintenance using asyncio.create_task()
            self._cache_task = asyncio.create_task(self._maintain_cache())

    async def stop_normal_collection(self):
        """Stop collecting normal state data"""
        async with self._lock:
            # Cancel the cache maintenance task if it exists
            if self._cache_task and not self._cache_task.done():
                self._cache_task.cancel()
                try:
                    await self._cache_task
                except asyncio.CancelledError:
                    pass
                
            if not self.is_collecting_normal or self.current_anomaly:
                logger.warning("No active normal state collection to stop")
                return
                
            # Save collected normal data
            if self.normal_state_buffer:
                try:
                    self.normal_samples += len(self.normal_state_buffer)
                    await self._save_normal_data(self.normal_state_buffer)
                    self.normal_state_buffer = []
                except Exception as e:
                    logger.error(f"Failed to save normal state data: {str(e)}")
            
            # Clear collection state
            await self._clear_collection_state()

    def stop_collection(self, save_post_data: bool = True):
        """Stop any active data collection.
        This method is non-async for backward compatibility with existing code.
        It creates a task to call the appropriate async stop method.
        """
        if not self.current_case:
            logger.warning("No active collection to stop")
            return
            
        # Determine which stop method to call
        if self.current_anomaly:
            # Create task to stop anomaly collection
            asyncio.create_task(self._stop_anomaly_collection(save_post_data))
        else:
            # Create task to stop normal collection
            asyncio.create_task(self.stop_normal_collection())
            
        logger.info("Initiated stop collection process")
    
    async def _stop_anomaly_collection(self, save_post_data: bool = True):
        """Stop collecting anomaly data"""
        async with self._lock:
            if not self.current_anomaly:
                logger.warning("No active anomaly collection to stop")
                return
                
            # Save collected anomaly data
            if self.metrics_buffer:
                try:
                    # Save the metrics data as an anomaly case
                    case_dir = os.path.join(self.data_dir, self.current_case)
                    os.makedirs(case_dir, exist_ok=True)
                    
                    metrics_file = os.path.join(case_dir, "metrics.json")
                    with open(metrics_file, 'w') as f:
                        json.dump(self.metrics_buffer, f)
                    
                    # Create and save the label file
                    label_file = os.path.join(case_dir, "label.json") 
                    with open(label_file, 'w') as f:
                        json.dump({
                            "type": self.current_anomaly["type"],
                            "node": self.current_anomaly["node"],
                            "start_time": self.collection_start_time,
                            "end_time": datetime.now().isoformat()
                        }, f)
                    
                    self.anomaly_samples += len(self.metrics_buffer)
                    logger.info(f"Saved {len(self.metrics_buffer)} anomaly metrics to {metrics_file}")
                    self.metrics_buffer = []
                    
                    # Invalidate stats cache
                    self.invalidate_stats_cache()
                except Exception as e:
                    logger.error(f"Failed to save anomaly data: {str(e)}")
            
            # Clear collection state
            await self._clear_collection_state()

    async def _save_normal_data(self, data: List[Dict], prefix: str = ""):
        """Save normal state data to disk"""
        if not self.current_case:
            logger.error("No active case when trying to save normal data")
            return
            
        case_dir = os.path.join(self.data_dir, self.current_case)
        os.makedirs(case_dir, exist_ok=True)
        
        # Save metrics data
        metrics_file = os.path.join(case_dir, f"{prefix}metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(data, f)
        
        # Save label file for normal data
        label_file = os.path.join(case_dir, "label.json")
        with open(label_file, 'w') as f:
            json.dump({
                "type": "normal",
                "start_time": self.collection_start_time
            }, f)
        
        logger.info(f"Saved normal state data to {case_dir}")

    async def add_metrics(self, metrics: Dict[str, Any]):
        """Add metrics to current collection"""
        async with self._lock:
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

            # Process metrics based on collection phase
            processed_metrics = self._process_raw_metrics(metrics)
            
            if self.is_collecting_normal:
                self.normal_state_buffer.extend(processed_metrics)
                logger.debug(f"Added {len(processed_metrics)} metrics to normal state buffer")
            else:  # Anomaly collection
                self.metrics_buffer.extend(processed_metrics)
                logger.debug(f"Added {len(processed_metrics)} metrics to anomaly buffer")
        
    def _process_raw_metrics(self, metrics: Dict[str, Any]) -> List[Dict]:
        """Process raw metrics into a list of timestamped metrics"""
        processed_metrics = []
        
        for node_ip, node_data in metrics.items():
            # Handle different metric formats
            timestamps = set()
            
            # Extract timestamps from all metric categories
            for category, category_data in node_data.items():
                if isinstance(category_data, dict):
                    # For dictionary metrics, collect all timestamps
                    if 'util' in category_data:  # CPU telegraf format
                        timestamps.update(category_data['util'].keys())
                    elif 'aggregated' in category_data:  # IO format
                        for metric in category_data['aggregated'].values():
                            timestamps.update(metric.keys())
                    else:  # Other dictionary metrics
                        for metric in category_data.values():
                            if isinstance(metric, dict):
                                timestamps.update(metric.keys())
            
            # If no timestamps found (e.g., all scalar values), use 'latest'
            if not timestamps:
                timestamps = {'latest'}
            
            # Remove 'latest' if it's not the only timestamp
            if len(timestamps) > 1 and 'latest' in timestamps:
                timestamps.remove('latest')
            
            # Convert to sorted list
            timestamps = sorted(timestamps)
            
            # Process each timestamp
            for timestamp in timestamps:
                try:
                    filtered_metrics = {
                        "timestamp": timestamp,
                        "node": node_ip,
                        "metrics": {}
                    }
                    
                    # Process each metric category
                    for category, category_data in node_data.items():
                        filtered_metrics["metrics"][category] = {}
                        
                        if isinstance(category_data, (int, float)):
                            # Handle scalar values directly
                            filtered_metrics["metrics"][category] = category_data
                        elif isinstance(category_data, dict):
                            if category == 'io':
                                # Handle both aggregated and direct IO metrics
                                if 'aggregated' in category_data:
                                    # Traditional telegraf format
                                    filtered_metrics["metrics"][category] = {
                                        'aggregated': {
                                            metric: data.get(timestamp, data.get('latest', 0))
                                            for metric, data in category_data['aggregated'].items()
                                        }
                                    }
                                else:
                                    # Direct metrics format (e.g., psutil)
                                    filtered_metrics["metrics"][category] = {
                                        metric: value.get(timestamp, value.get('latest', 0)) if isinstance(value, dict) else value
                                        for metric, value in category_data.items()
                                    }
                            elif 'util' in category_data:
                                # CPU telegraf format
                                filtered_metrics["metrics"][category] = {
                                    'util': category_data['util'].get(timestamp, category_data['util'].get('latest', 0))
                                }
                            else:
                                # Generic dictionary metrics
                                for metric, data in category_data.items():
                                    if isinstance(data, dict):
                                        filtered_metrics["metrics"][category][metric] = data.get(timestamp, data.get('latest', 0))
                                    else:
                                        filtered_metrics["metrics"][category][metric] = data
                    
                    processed_metrics.append(filtered_metrics)
                except Exception as e:
                    logger.error(f"Error processing metrics for node {node_ip} at timestamp {timestamp}: {str(e)}")
                    logger.debug(f"Raw metrics for failed node: {node_data}")
                    continue
                    
        return processed_metrics
        
    def _should_update_stats_cache(self) -> bool:
        """Check if stats cache needs to be updated"""
        return (
            self._stats_cache is None
            or self._last_stats_update is None
            or time.time() - self._last_stats_update > self.STATS_CACHE_TTL
        )

    @alru_cache(maxsize=1, ttl=60)
    async def get_dataset_stats(self) -> Dict[str, int]:
        """Get dataset statistics with optimized async computation"""
        async with self._request_lock:
            now = time.time()
            
            # Return cached results for frequent requests
            if now - self._last_request_time < 5:  # 5s coalescing window
                if self._stats_cache:
                    return self._stats_cache
            
            self._last_request_time = now

            # Use background task for computation
            try:
                stats = await asyncio.wait_for(
                    self._event_loop.run_in_executor(self._stats_executor, self._compute_dataset_stats),
                    timeout=20  # 20s timeout
                )
                return stats
            except asyncio.TimeoutError:
                logger.warning("Stats computation timed out, returning cached data")
                return self._stats_cache or {
                    "normal": 0,
                    "anomaly": 0,
                    "anomaly_types": {},
                    "total_samples": 0,
                    "normal_ratio": 0,
                    "anomaly_ratio": 0,
                    "is_balanced": False,
                    "error": "Computation timed out"
                }
            except Exception as e:
                logger.error(f"Error computing stats: {str(e)}")
                raise

    def _count_json_entries(self, file_path: str) -> int:
        """Count entries in a JSON array file using memory-efficient parsing"""
        count = 0
        try:
            with open(file_path, 'rb') as f:
                parser = ijson.items(f, 'item')
                for _ in parser:
                    count += 1
            return count
        except Exception as e:
            logger.error(f"Error counting entries in {file_path}: {str(e)}")
            return 0

    def _compute_dataset_stats(self):
        """Get statistics about the collected training data with optimized caching"""
        if not self._should_update_stats_cache():
            logger.debug("Returning cached stats")
            return self._stats_cache

        logger.info("Calculating fresh dataset statistics")
        stats = {
            "normal": 0,
            "anomaly": 0,
            "anomaly_types": defaultdict(int),
            "total_samples": 0,
            "normal_ratio": 0,
            "anomaly_ratio": 0,
            "is_balanced": False
        }
        
        if not os.path.exists(self.data_dir):
            self._stats_cache = stats
            self._last_stats_update = time.time()
            return stats

        # Process in smaller batches to avoid memory issues
        case_dirs = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
        batch_size = 50  # Process 50 cases at a time
        
        for i in range(0, len(case_dirs), batch_size):
            batch = case_dirs[i:i + batch_size]
            for case_dir in batch:
                case_path = os.path.join(self.data_dir, case_dir)
                label_file = os.path.join(case_path, "label.json")
                
                if not os.path.exists(label_file):
                    continue
                    
                try:
                    with open(label_file, 'r') as f:
                        label_data = json.load(f)
                        
                    if label_data.get("type") == "normal":
                        # For normal cases, count all metrics as normal
                        metrics_file = os.path.join(case_path, "metrics.json")
                        if os.path.exists(metrics_file):
                            count = self._count_json_entries(metrics_file)
                            stats["normal"] += count
                            logger.debug(f"Added {count} normal samples from {case_dir}")
                    else:
                        # For anomaly cases:
                        # 1. Count pre_anomaly as normal
                        pre_anomaly_file = os.path.join(case_path, "pre_anomaly_metrics.json")
                        if os.path.exists(pre_anomaly_file):
                            count = self._count_json_entries(pre_anomaly_file)
                            stats["normal"] += count
                            logger.debug(f"Added {count} pre-anomaly normal samples from {case_dir}")
                        
                        # 2. Count metrics.json as anomaly
                        metrics_file = os.path.join(case_path, "metrics.json")
                        if os.path.exists(metrics_file):
                            count = self._count_json_entries(metrics_file)
                            stats["anomaly"] += count
                            anomaly_type = label_data.get("type")
                            if anomaly_type:
                                stats["anomaly_types"][anomaly_type] += count
                            logger.debug(f"Added {count} anomaly samples of type {anomaly_type} from {case_dir}")
                        
                        # 3. Count post_anomaly as normal
                        post_anomaly_file = os.path.join(case_path, "post_anomaly_metrics.json")
                        if os.path.exists(post_anomaly_file):
                            count = self._count_json_entries(post_anomaly_file)
                            stats["normal"] += count
                            logger.debug(f"Added {count} post-anomaly normal samples from {case_dir}")
                            
                except Exception as e:
                    logger.error(f"Error processing case {case_dir}: {str(e)}")
                    continue
        
        stats["total_samples"] = stats["normal"] + stats["anomaly"]
        if stats["total_samples"] > 0:
            stats["normal_ratio"] = stats["normal"] / stats["total_samples"]
            stats["anomaly_ratio"] = stats["anomaly"] / stats["total_samples"]
            # Consider dataset balanced if minority class is at least 30% of total
            min_ratio = min(stats["normal_ratio"], stats["anomaly_ratio"])
            stats["is_balanced"] = min_ratio >= 0.3
        
        self._stats_cache = stats
        self._last_stats_update = time.time()
        return stats

    def invalidate_stats_cache(self):
        """Invalidate the stats cache to force recalculation"""
        self._stats_cache = None
        self._last_stats_update = None
        self.get_dataset_stats.cache_clear()

    def is_dataset_balanced(self, threshold: float = 0.3) -> bool:
        """Check if the dataset is balanced within the threshold"""
        try:
            stats = self.get_dataset_stats()
            if not stats:
                logger.warning("No training data available")
                return False
                
            total = stats.get("normal", 0) + stats.get("anomaly", 0)
            if total == 0:
                logger.warning("No training samples available")
                return False
                
            normal_ratio = stats.get("normal", 0) / total
            anomaly_ratio = stats.get("anomaly", 0) / total
            
            is_balanced = abs(normal_ratio - anomaly_ratio) <= threshold
            logger.info(f"Dataset balance check: normal_ratio={normal_ratio:.2f}, anomaly_ratio={anomaly_ratio:.2f}, is_balanced={is_balanced}")
            return is_balanced
        except Exception as e:
            logger.error(f"Error checking dataset balance: {str(e)}")
            return False

    async def auto_balance_dataset(self):
        """Automatically balance the dataset by collecting required data"""
        logger.info("Starting auto-balance process")
        
        # Get current stats
        stats = await self.get_dataset_stats()
        total = stats["normal"] + stats["anomaly"]
        
        if total == 0:
            # If no data, start with normal collection
            logger.info("No data available. Starting with normal collection")
            await self.start_normal_collection()
            return
            
        # Calculate ratios
        normal_ratio = stats["normal"] / total if total > 0 else 0
        anomaly_ratio = stats["anomaly"] / total if total > 0 else 0
        
        # Check normal vs anomaly balance first
        if abs(normal_ratio - anomaly_ratio) > self.balance_threshold:
            if normal_ratio < anomaly_ratio:
                logger.info("Normal samples underrepresented. Starting normal collection")
                await self.start_normal_collection()
            else:
                # Find least represented anomaly type
                anomaly_types = stats["anomaly_types"]
                min_type = min(anomaly_types.items(), key=lambda x: x[1])[0]
                logger.info(f"Anomaly type {min_type} underrepresented. Starting collection")
                await self.start_collection(min_type, None)  # Let the system choose the node
        else:
            # Balance anomaly types
            anomaly_types = stats["anomaly_types"]
            avg_anomaly_count = stats["anomaly"] / len(anomaly_types)
            
            # Find most underrepresented type
            min_type = min(anomaly_types.items(), key=lambda x: x[1])[0]
            if anomaly_types[min_type] < avg_anomaly_count * 0.7:  # Allow 30% variance
                logger.info(f"Anomaly type {min_type} below average. Starting collection")
                await self.start_collection(min_type, None)
            else:
                logger.info("Dataset is well balanced")

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
                # Load label file first to determine data type
                label_file = os.path.join(case_path, "label.json")
                if not os.path.exists(label_file):
                    logger.warning(f"Missing label file in {case_dir}")
                    continue
                    
                with open(label_file) as f:
                    label_data = json.load(f)
                    
                # Process anomaly data
                metrics_file = os.path.join(case_path, "metrics.json")
                if os.path.exists(metrics_file):
                    with open(metrics_file) as f:
                        metrics_data = json.load(f)
                        features = self._process_metrics(metrics_data)
                        if features is not None:
                            X.extend(features)
                            y.extend([self._create_label_vector(label_data) for _ in range(len(features))])
                            logger.info(f"Processed {len(features)} anomaly samples from {case_dir}")
                            
            except Exception as e:
                logger.error(f"Error loading training data from {case_dir}: {str(e)}")
                continue
                
        if not X or not y:
            logger.warning("No valid training data found")
            return [], []
            
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
            'network_stress': 3,
            'network_delay': 3,  # Add network_delay type
            'disk': 2,  # Map disk to io category
            'disk_stress': 2
        }
        
        label_vector = np.zeros(4)  # 4 basic types: cpu, memory, io, network
        
        anomaly_type = label_data.get("type")
        if anomaly_type in anomaly_types:
            label_vector[anomaly_types[anomaly_type]] = 1
            logger.debug(f"Created label vector for anomaly type {anomaly_type}: {label_vector}")
        else:
            logger.warning(f"Unknown anomaly type: {anomaly_type}")
            
        return label_vector

    async def _maintain_cache(self):
        """Background task to periodically update the dataset stats cache"""
        try:
            while True:
                await self.get_dataset_stats()
                await asyncio.sleep(30)  # Refresh every 30s
        except asyncio.CancelledError:
            # Handle cancellation gracefully
            logger.debug("Cache maintenance task was cancelled")
            raise

# Create singleton instance
training_service = TrainingService() 