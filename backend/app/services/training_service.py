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
        
        # Set auto reset flag from environment variable (default to True)
        self.auto_reset_on_startup = os.getenv('AUTO_RESET_ON_STARTUP', 'true').lower() == 'true'
        
        # Initialize variables
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
        self.anomaly_groups = defaultdict(int)  # Track collected groups per anomaly type
        self.target_groups_per_type = 10  # 10 groups per anomaly type
        self.required_anomaly_types = 16  # 16 different anomaly scenarios
        self.collection_duration = 180  # Collection duration in seconds
        self.anomaly_variations = defaultdict(set)  # Track variations (nodes/severity) per type
        
        # Add experiment types (anomaly scenarios) list
        self.experiment_types = [
            "cpu_stress",
            "io_bottleneck", 
            "network_bottleneck",
            "cache_bottleneck",
            "too_many_indexes"
        ]
        
        # Log existing data
        self._log_existing_data()
        
        # Reset state on startup if configured
        if self.auto_reset_on_startup:
            self.reset_state_on_startup()
            
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
                # Get list of available nodes that haven't been used for this type
                from app.services.k8s_service import k8s_service
                available_nodes = await k8s_service.get_available_nodes()
                used_nodes = self.anomaly_variations[anomaly_type]
                unused_nodes = [n for n in available_nodes if n not in used_nodes]
                
                if not unused_nodes:
                    # If all nodes used, clear history to allow reuse
                    self.anomaly_variations[anomaly_type].clear()
                    unused_nodes = available_nodes
                
                node = unused_nodes[0] if unused_nodes else available_nodes[0]
                logger.info(f"Selected node {node} for anomaly type {anomaly_type}")
                
            # Track this variation
            self.anomaly_variations[anomaly_type].add(node)
            
            # Make sure metrics service is aware we're collecting data
            try:
                from app.services.metrics_service import metrics_service
                metrics_service.toggle_send_to_training(True)
                logger.info("Enabled sending metrics to training service")
            except Exception as e:
                logger.error(f"Failed to enable metrics collection: {e}")
            
            self.collection_start_time = datetime.now().isoformat()
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            self.current_case = f"case_{anomaly_type}_{node}_{timestamp}"
            self.current_anomaly = {
                "type": anomaly_type,
                "node": node,
                "start_time": self.collection_start_time,
                "duration": self.collection_duration
            }
            
            # Ensure the metrics buffer is initialized as an empty list
            self.metrics_buffer = []
            
            # No pre/post collection - directly start anomaly collection
            self.is_collecting_normal = False
            logger.info(f"Started anomaly data collection for {self.current_case} at {self.collection_start_time}")
            logger.info(f"Metrics buffer initialized: {self.metrics_buffer is not None}, length: {len(self.metrics_buffer)}")
            
            # Schedule automatic stop after collection_duration
            asyncio.create_task(self._auto_stop_collection())

    async def _auto_stop_collection(self):
        """Automatically stop collection after the specified duration"""
        try:
            await asyncio.sleep(self.collection_duration)
            if self.current_case:  # Only stop if still collecting
                logger.info(f"Auto-stopping collection after {self.collection_duration} seconds")
                await self.stop_collection(save_post_data=True)
        except Exception as e:
            logger.error(f"Error in auto-stop collection: {str(e)}")

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

    async def _stop_anomaly_collection(self, save_post_data: bool = True):
        """Stop collecting anomaly data"""
        try:
            async with self._lock:
                if not self.current_anomaly:
                    logger.warning("No active anomaly collection to stop")
                    return

                logger.info("Starting anomaly collection cleanup")
                
                # Update group counter when saving anomaly data
                anomaly_type = self.current_anomaly["type"]
                self.anomaly_groups[anomaly_type] += 1
                logger.info(f"Completed group {self.anomaly_groups[anomaly_type]} for {anomaly_type}")
                
                # Move file operations to thread pool
                await asyncio.to_thread(self._save_anomaly_data)
                logger.info("Anomaly data saved successfully")
                
        except Exception as e:
            logger.error(f"Critical error during anomaly stop: {str(e)}", exc_info=True)
        finally:
            # Ensure state cleanup even if errors occur
            await self._clear_collection_state()
            logger.info("Anomaly collection fully stopped")

    def _save_anomaly_data(self):
        """Sync method for saving anomaly data (to be run in thread pool)"""
        try:
            # Create directory even if we don't have metrics
            case_dir = os.path.join(self.data_dir, self.current_case)
            os.makedirs(case_dir, exist_ok=True)
            
            # Save what we have in the metrics buffer
            metrics_count = len(self.metrics_buffer) if self.metrics_buffer else 0
            logger.info(f"Saving anomaly data: {metrics_count} metrics entries")
            
            # Add detailed debug logging
            if self.metrics_buffer:
                logger.debug(f"Metrics buffer contains {metrics_count} entries")
                # Log a sample of the first entry to help diagnose format issues
                if metrics_count > 0:
                    logger.debug(f"Sample metrics entry: {self.metrics_buffer[0]}")
            else:
                logger.debug("Metrics buffer is empty - this could indicate a collection issue")
                
            if not self.metrics_buffer:
                logger.warning("No metrics to save, creating empty metrics file")
                # Create an empty metrics file so we know the collection happened
                metrics_file = os.path.join(case_dir, "metrics.json")
                with open(metrics_file, 'w') as f:
                    json.dump([], f)
            else:
                metrics_file = os.path.join(case_dir, "metrics.json")
                with open(metrics_file, 'w') as f:
                    json.dump(self.metrics_buffer, f)

            # Always save label information
            label_file = os.path.join(case_dir, "label.json") 
            with open(label_file, 'w') as f:
                # Handle compound anomalies with special label format
                if self.current_anomaly.get("type") == "compound" and "subtypes" in self.current_anomaly:
                    json.dump({
                        "type": "compound",
                        "subtypes": self.current_anomaly["subtypes"],
                        "node": self.current_anomaly["node"],
                        "start_time": self.collection_start_time,
                        "end_time": datetime.now().isoformat(),
                        "metrics_count": metrics_count
                    }, f)
                else:
                    # Regular single anomaly
                    json.dump({
                        "type": self.current_anomaly["type"],
                        "node": self.current_anomaly["node"],
                        "start_time": self.collection_start_time,
                        "end_time": datetime.now().isoformat(),
                        "metrics_count": metrics_count
                    }, f)
                
            logger.info(f"Saved anomaly data to {case_dir} (metrics: {metrics_count})")
            
        except Exception as e:
            logger.error(f"Error saving anomaly data: {str(e)}", exc_info=True)

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
            logger.info(f"Received metrics - State: is_collecting_normal={self.is_collecting_normal}, current_case={self.current_case}")
            
            # Debug information about received metrics 
            node_count = len(metrics) if metrics else 0
            if node_count > 0:
                sample_node = next(iter(metrics))
                sample_categories = list(metrics[sample_node].keys())
                logger.debug(f"Received metrics for {node_count} nodes. Sample node: {sample_node}")
                logger.debug(f"Sample node has these metric categories: {sample_categories}")
                
            if not self.current_case:
                logger.warning("No active case - metrics will not be collected")
                logger.info(f"Collection state: is_collecting_normal={self.is_collecting_normal}, collection_start_time={self.collection_start_time}")
                return
                
            if not self.collection_start_time:
                logger.warning("No collection start time - metrics will not be collected")
                return

            logger.info(f"Processing metrics for case: {self.current_case}")
            logger.info(f"Collection mode: {'normal' if self.is_collecting_normal else 'anomaly'}")
            logger.info(f"Number of nodes in metrics: {len(metrics)}")

            # Process metrics based on collection phase
            processed_metrics = self._process_raw_metrics(metrics)
            logger.info(f"Processed {len(processed_metrics)} metrics from raw data")
            
            if self.is_collecting_normal:
                self.normal_state_buffer.extend(processed_metrics)
                logger.info(f"Added {len(processed_metrics)} metrics to normal state buffer, total size: {len(self.normal_state_buffer)}")
            else:  # Anomaly collection
                self.metrics_buffer.extend(processed_metrics)
                logger.info(f"Added {len(processed_metrics)} metrics to anomaly buffer, total size: {len(self.metrics_buffer)}")
        
    def _process_raw_metrics(self, metrics: Dict[str, Any]) -> List[Dict]:
        """Process raw metrics into a list of timestamped metrics"""
        processed_metrics = []
        
        logger.info(f"Processing raw metrics for {len(metrics)} nodes")
        logger.info(f"Sample node IPs: {list(metrics.keys())[:3]}")
        
        for node_ip, node_data in metrics.items():
            # Log sample of node data structure 
            logger.info(f"Node {node_ip} has {len(node_data)} metric categories: {list(node_data.keys())}")
            
            # Handle different metric formats
            timestamps = set()
            
            # Extract timestamps from all metric categories
            for category, category_data in node_data.items():
                if isinstance(category_data, dict):
                    # For dictionary metrics, collect all timestamps
                    if 'util' in category_data:  # CPU telegraf format
                        timestamps.update(category_data['util'].keys())
                        logger.debug(f"CPU format found for {category}, timestamps: {list(category_data['util'].keys())[:3]}")
                    elif 'aggregated' in category_data:  # IO format
                        logger.debug(f"IO format found for {category}, keys: {list(category_data['aggregated'].keys())}")
                        for metric in category_data['aggregated'].values():
                            timestamps.update(metric.keys())
                    else:  # Other dictionary metrics
                        for metric_name, metric in category_data.items():
                            logger.debug(f"Processing metric {category}.{metric_name}, type: {type(metric)}")
                            if isinstance(metric, dict):
                                timestamps.update(metric.keys())
            
            # If no timestamps found (e.g., all scalar values), use 'latest'
            if not timestamps:
                logger.warning(f"No timestamps found for node {node_ip}, using 'latest'")
                timestamps = {'latest'}
            
            logger.info(f"Node {node_ip} has {len(timestamps)} unique timestamps")
            
            # Remove 'latest' if it's not the only timestamp
            if len(timestamps) > 1 and 'latest' in timestamps:
                timestamps.remove('latest')
            
            # Convert to sorted list
            timestamps = sorted(timestamps)
            
            # Process each timestamp
            timestamp_count = 0
            for timestamp in timestamps:
                try:
                    filtered_metrics = {
                        "timestamp": timestamp,
                        "node": node_ip,
                        "metrics": {}
                    }
                    
                    metrics_count = 0
                    
                    # Process each metric category
                    for category, category_data in node_data.items():
                        filtered_metrics["metrics"][category] = {}
                        
                        if isinstance(category_data, dict):
                            # Handle different formats based on category structure
                            if 'util' in category_data:  # CPU telegraf format
                                if timestamp in category_data['util']:
                                    filtered_metrics["metrics"][category]['util'] = category_data['util'][timestamp]
                                    metrics_count += 1
                            elif 'aggregated' in category_data:  # IO format
                                filtered_metrics["metrics"][category]['aggregated'] = {}
                                for io_metric, values in category_data['aggregated'].items():
                                    if timestamp in values:
                                        filtered_metrics["metrics"][category]['aggregated'][io_metric] = values[timestamp]
                                        metrics_count += 1
                            else:  # Other dictionary metrics
                                for metric_name, metric_data in category_data.items():
                                    if isinstance(metric_data, dict) and timestamp in metric_data:
                                        filtered_metrics["metrics"][category][metric_name] = metric_data[timestamp]
                                        metrics_count += 1
                                    elif not isinstance(metric_data, dict):
                                        # For scalar values, always include them
                                        filtered_metrics["metrics"][category][metric_name] = metric_data
                                        metrics_count += 1
                        else:
                            # Scalar category value
                            filtered_metrics["metrics"][category] = category_data
                            metrics_count += 1
                    
                    # Only add if we have at least one metric
                    if metrics_count > 0:
                        processed_metrics.append(filtered_metrics)
                        timestamp_count += 1
                        
                        # Debug log for the first few processed entries
                        if len(processed_metrics) <= 3:
                            logger.debug(f"Processed metrics for node {node_ip}, timestamp {timestamp}, containing {metrics_count} metrics")
                    else:
                        logger.warning(f"No metrics found for timestamp {timestamp} on node {node_ip}")
                except Exception as e:
                    logger.error(f"Error processing metrics for timestamp {timestamp} on node {node_ip}: {str(e)}")
            
            logger.info(f"Processed {timestamp_count} timestamps for node {node_ip}")
        
        logger.info(f"Processed a total of {len(processed_metrics)} metric entries across all nodes")
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

        logger.debug("Calculating fresh dataset statistics")
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
        
        # If there's an active collection, don't start a new one
        if self.current_anomaly or self.is_collecting_normal:
            logger.info("Collection already in progress, skipping auto-balance")
            return {"success": False, "message": "Collection already in progress"}
        
        # Calculate how balanced each anomaly type is
        anomaly_types = stats.get("anomaly_types", {})
        total_anomalies = stats.get("anomaly", 0)
        total_samples = stats.get("total_samples", 0)
        
        # First check if we need more normal samples compared to anomalies
        normal_ratio = stats.get("normal_ratio", 0)
        anomaly_ratio = stats.get("anomaly_ratio", 0)
        
        # If normal samples are significantly underrepresented, collect normal data
        if normal_ratio < 0.3 and anomaly_ratio > 0.7 and not self.is_collecting_normal:
            logger.info(f"Normal samples underrepresented (ratio: {normal_ratio:.2f}), starting normal collection")
            await self.start_normal_collection()
            return {"success": True, "message": "Started normal data collection for balancing"}
        
        # Check if any anomaly type is underrepresented
        if anomaly_types:
            # Calculate the target count for each type (roughly equal distribution)
            min_target = max(10, total_anomalies / (len(self.experiment_types) * 2))  # At least 10 samples per type
            
            # Find underrepresented types
            underrepresented = []
            for anomaly_type in self.experiment_types:
                count = anomaly_types.get(anomaly_type, 0)
                if count < min_target:
                    underrepresented.append((anomaly_type, count))
            
            # Sort by count (most underrepresented first)
            underrepresented.sort(key=lambda x: x[1])
            
            if underrepresented:
                # Pick the most underrepresented type
                anomaly_type, count = underrepresented[0]
                logger.info(f"Collecting data for underrepresented anomaly type: {anomaly_type} (count: {count}, target: {min_target})")
                
                # Auto-select node (None means system will choose)
                await self.start_collection(anomaly_type, None)
                return {"success": True, "message": f"Started collection for underrepresented anomaly type: {anomaly_type}"}
        
        # If no underrepresented types, check if we need to collect variations for existing types
        for anomaly_type in self.experiment_types:
            # Check if this type has enough groups collected
            if anomaly_type in anomaly_types and self.anomaly_groups.get(anomaly_type, 0) < self.target_groups_per_type:
                logger.info(f"Collecting variation {self.anomaly_groups[anomaly_type]+1} for {anomaly_type}")
                await self.start_collection(anomaly_type, None)  # Let the system choose the node
                return {"success": True, "message": f"Started collecting variation for {anomaly_type}"}
        
        logger.info("All required anomaly variations have been collected, dataset is balanced")
        return {"success": True, "message": "Dataset is already well-balanced"}

    def get_training_data(self) -> tuple:
        """Get training data from all cases"""
        logger.info("Loading training data from all cases...")
        X = []
        y = []
        
        # Get list of all case directories first
        case_dirs = []
        for case_dir in os.listdir(self.data_dir):
            case_path = os.path.join(self.data_dir, case_dir)
            if os.path.isdir(case_path):
                label_file = os.path.join(case_path, "label.json")
                metrics_file = os.path.join(case_path, "metrics.json")
                if os.path.exists(label_file) and os.path.exists(metrics_file):
                    case_dirs.append((case_dir, case_path))
        
        total_cases = len(case_dirs)
        logger.info(f"Found {total_cases} valid case directories")
        
        # Use ThreadPoolExecutor for parallel processing
        import time
        
        start_time = time.time()
        processed_cases = 0
        total_samples = 0
        
        # Define a function to process a single case
        def process_case(case_info):
            case_dir, case_path = case_info
            case_X = []
            case_y = []
            
            try:
                # Load label file
                with open(os.path.join(case_path, "label.json")) as f:
                    label_data = json.load(f)
                
                # Process metrics data
                with open(os.path.join(case_path, "metrics.json")) as f:
                    metrics_data = json.load(f)
                    
                features = self._process_metrics(metrics_data)
                if features is not None:
                    case_X.extend(features)
                    # Create labels for each feature
                    for _ in range(len(features)):
                        case_y.append(self._create_label_vector(label_data))
                    
                    return {
                        'case_dir': case_dir,
                        'X': case_X, 
                        'y': case_y, 
                        'count': len(features),
                        'type': label_data.get('type', 'unknown')
                    }
            except Exception as e:
                logger.error(f"Error processing case {case_dir}: {str(e)}")
                return None
        
        # Process cases in parallel with a maximum of 4 worker threads
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all processing tasks
            future_to_case = {executor.submit(process_case, case_info): case_info for case_info in case_dirs}
            
            # Process results as they complete
            for future in future_to_case:
                result = future.result()
                if result:
                    X.extend(result['X'])
                    y.extend(result['y'])
                    processed_cases += 1
                    total_samples += result['count']
                    
                    # Log progress periodically
                    if processed_cases % 10 == 0 or processed_cases == total_cases:
                        elapsed = time.time() - start_time
                        logger.info(f"Processed {processed_cases}/{total_cases} cases ({processed_cases/total_cases*100:.1f}%) in {elapsed:.1f}s - {total_samples} samples")
                    
                    logger.info(f"Processed {result['count']} samples from {result['case_dir']} ({result['type']})")
        
        if not X or not y:
            logger.warning("No valid training data found")
            return [], []
        
        # Convert to numpy arrays
        X_array = np.array(X)
        y_array = np.array(y)
        
        # Log final stats
        elapsed = time.time() - start_time
        logger.info(f"Completed processing {total_samples} samples from {processed_cases} cases in {elapsed:.1f}s ({total_samples/elapsed:.1f} samples/s)")
        logger.info(f"Training data shapes - X: {X_array.shape}, y: {y_array.shape}")
        
        # Balance classes if needed
        logger.info("Balancing dataset classes...")
        
        # Identify normal vs anomaly samples
        # A sample is an anomaly if any of its label values is 1
        is_anomaly = y_array.any(axis=1)
        normal_indices = np.where(~is_anomaly)[0]
        anomaly_indices = np.where(is_anomaly)[0]
        
        logger.info(f"Dataset composition before balancing: {len(normal_indices)} normal, " 
                   f"{len(anomaly_indices)} anomaly samples")
        
        # If there are too many normal samples compared to anomalies, subsample them
        if len(normal_indices) > len(anomaly_indices):
            # Randomly select a subset of normal samples equal to the number of anomaly samples
            np.random.shuffle(normal_indices)
            selected_normal_indices = normal_indices[:len(anomaly_indices)]
            
            # Combine selected normal samples with all anomaly samples
            balanced_indices = np.concatenate([selected_normal_indices, anomaly_indices])
            np.random.shuffle(balanced_indices)  # Shuffle to mix normal and anomaly samples
            
            X_array = X_array[balanced_indices]
            y_array = y_array[balanced_indices]
            
            logger.info(f"Balanced dataset: {len(X_array)} total samples "
                       f"({len(selected_normal_indices)} normal, {len(anomaly_indices)} anomaly)")
        
        return X_array, y_array

    def _process_metrics(self, metrics_data: List[Dict]) -> List[np.ndarray]:
        """Process raw metrics into feature vectors using time window processing from diagnosis service"""
        if not metrics_data:
            logger.warning("No metrics data to process")
            return None
            
        try:
            # Get reference to diagnosis service
            from app.services.diagnosis_service import diagnosis_service
            
            # Extract features from metrics using diagnosis service's processor
            features = []
            
            # Process metrics in chunks to avoid memory issues with large datasets
            chunk_size = 100  # Process 100 metrics at a time
            for i in range(0, len(metrics_data), chunk_size):
                chunk = metrics_data[i:i+chunk_size]
                chunk_features = []
                
                for entry in chunk:
                    try:
                        # Make sure 'metrics' key exists
                        if 'metrics' not in entry:
                            logger.warning(f"Missing 'metrics' key in entry: {entry.keys()}")
                            continue
                            
                        # Process metrics using diagnosis service's time window processing
                        feature_vector = diagnosis_service.diagnosis._process_metrics(entry["metrics"])
                        if feature_vector is not None:
                            chunk_features.append(feature_vector)
                    except Exception as e:
                        logger.error(f"Error processing metrics entry: {str(e)}")
                        import traceback
                        logger.debug(f"Traceback: {traceback.format_exc()}")
                        continue
                
                # Add chunk features to main features list
                features.extend(chunk_features)
                
                # Log progress for large datasets
                if len(metrics_data) > chunk_size and i % (chunk_size * 10) == 0:
                    logger.debug(f"Processed {i}/{len(metrics_data)} metrics entries ({i/len(metrics_data)*100:.1f}%)")
            
            logger.info(f"Successfully processed {len(features)} feature vectors from {len(metrics_data)} metrics entries")
            return features
        except Exception as e:
            logger.error(f"Error in _process_metrics: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def _create_label_vector(self, label_data: Dict) -> np.ndarray:
        """Create a label vector for the anomaly type"""
        # Define the mapping of anomaly types to indices
        # Map both short and full names of anomaly types
        anomaly_types = {
            'cpu': 0,
            'cpu_stress': 0,
            'io': 1,
            'io_bottleneck': 1,
            'network': 2,
            'network_bottleneck': 2,
            'cache': 3,
            'cache_bottleneck': 3,
            'indexes': 4,
            'too_many_indexes': 4,
        }
        
        label_vector = np.zeros(5)  # 5 basic types: cpu, io, network, cache, indexes
        
        # Handle compound anomalies
        if label_data.get("type") == "compound" and "subtypes" in label_data:
            # Set multiple bits for compound anomalies
            for subtype in label_data["subtypes"]:
                if subtype in anomaly_types:
                    label_vector[anomaly_types[subtype]] = 1
            logger.debug(f"Created compound label vector for subtypes {label_data['subtypes']}: {label_vector}")
        else:
            # Special handling for 'normal' samples - return zeros vector
            anomaly_type = label_data.get("type")
            if anomaly_type == "normal":
                return label_vector
                
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

    def reset_state_on_startup(self):
        """Reset the training service state on startup to ensure clean state after restarts."""
        logger.info("Resetting training service state on startup")
        self.current_case = None
        self.current_anomaly = None
        self.metrics_buffer = []
        self.normal_state_buffer = []
        self.is_collecting_normal = False
        self.collection_start_time = None
        self.anomaly_groups.clear()  # Clear group tracking
        self.anomaly_variations.clear()  # Clear variation tracking
        logger.info("Training service state has been reset")

    async def reset_state(self):
        """Reset the training service state asynchronously."""
        async with self._lock:
            logger.info("Resetting training service state")
            self.current_case = None
            self.current_anomaly = None 
            self.metrics_buffer = []
            self.normal_state_buffer = []
            self.is_collecting_normal = False
            self.collection_start_time = None
            logger.info("Training service state has been reset")
            return {"status": "success", "message": "Training state reset successfully"}

    def stop_collection(self, save_post_data: bool = True):
        """Stop any active data collection."""
        if not self.current_case:
            logger.warning("No active collection to stop")
            return

        # Create and store task reference
        stop_task = asyncio.create_task(self._stop_anomaly_collection(save_post_data)
                                        if self.current_anomaly 
                                        else self.stop_normal_collection())
        
        # Add proper error handling
        def callback(fut):
            try:
                fut.result()
            except Exception as e:
                logger.error(f"Stop collection failed: {str(e)}")

        stop_task.add_done_callback(callback)
        logger.info("Initiated stop collection process")

    async def start_compound_collection(self, anomaly_types: list, node: str = None):
        """
        Start collecting training data for compound anomaly scenarios (multiple simultaneous anomalies)
        
        Args:
            anomaly_types: List of anomaly types to simulate simultaneously
            node: Optional target node, if None a node will be selected automatically
        """
        if len(anomaly_types) < 2:
            raise ValueError("Compound collection requires at least 2 anomaly types")
            
        async with self._lock:
            # Validate anomaly types are supported
            for atype in anomaly_types:
                if atype not in self.experiment_types:
                    raise ValueError(f"Unsupported anomaly type: {atype}. Valid types: {self.experiment_types}")
            
            # Make sure metrics service is aware we're collecting data
            try:
                from app.services.metrics_service import metrics_service
                metrics_service.toggle_send_to_training(True)
                logger.info("Enabled sending metrics to training service")
            except Exception as e:
                logger.error(f"Failed to enable metrics collection: {e}")
            
            # Validate node parameter or select appropriate node
            if not node:
                # Get list of available nodes that haven't been used for this combination
                from app.services.k8s_service import k8s_service
                available_nodes = await k8s_service.get_available_nodes()
                
                # Create a compound key for tracking variations
                compound_key = "+".join(sorted(anomaly_types))
                used_nodes = self.anomaly_variations.get(compound_key, set())
                unused_nodes = [n for n in available_nodes if n not in used_nodes]
                
                if not unused_nodes:
                    # If all nodes used, clear history to allow reuse
                    self.anomaly_variations[compound_key] = set()
                    unused_nodes = available_nodes
                
                node = unused_nodes[0] if unused_nodes else available_nodes[0]
                logger.info(f"Selected node {node} for compound anomaly types {anomaly_types}")
                
                # Track this variation
                if compound_key not in self.anomaly_variations:
                    self.anomaly_variations[compound_key] = set()
                self.anomaly_variations[compound_key].add(node)
            
            # Set up collection parameters with longer duration for compound anomalies
            self.collection_start_time = datetime.now().isoformat()
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            
            # Create case name with all anomaly types
            anomaly_str = "_plus_".join(anomaly_types)
            self.current_case = f"case_compound_{anomaly_str}_{node}_{timestamp}"
            
            # Store anomaly data with all types
            self.current_anomaly = {
                "type": "compound",
                "subtypes": anomaly_types,
                "node": node,
                "start_time": self.collection_start_time,
                "duration": self.collection_duration * 1.5  # 50% longer for compound anomalies
            }
            
            # Initialize metrics buffer
            self.metrics_buffer = []
            self.is_collecting_normal = False
            logger.info(f"Started compound anomaly data collection for {self.current_case} at {self.collection_start_time}")
            logger.info(f"Collecting {len(anomaly_types)} simultaneous anomalies: {', '.join(anomaly_types)}")
            
            # Schedule automatic stop after extended collection_duration
            asyncio.create_task(self._auto_stop_collection())

# Create singleton instance
training_service = TrainingService() 