import os
import json
import logging
import concurrent.futures
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import shutil
import threading
from functools import lru_cache, wraps
import time
from concurrent.futures import ThreadPoolExecutor
import asyncio
import functools
from async_lru import alru_cache
import ijson
from collections import defaultdict
import mmap

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
    _lock = threading.Lock()
    _stats_cache = None
    _last_stats_update = None
    STATS_CACHE_TTL = 60  # Cache TTL in seconds
    _stats_executor = ThreadPoolExecutor(max_workers=1)
    _stats_lock = asyncio.Lock()
    _request_lock = asyncio.Lock()
    _last_request_time = 0

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
        
        # Add dataset balance tracking
        self.normal_samples = 0
        self.anomaly_samples = 0
        self.pre_anomaly_buffer = []
        self.post_anomaly_buffer = []
        self.pre_anomaly_duration = 60  # seconds
        self.post_anomaly_duration = 60  # seconds
        self.balance_threshold = 0.3  # 30% threshold for balance
        
        # Add lock for thread safety
        self._lock = threading.Lock()
        
        # Initialize cache
        self._stats_cache = None
        self._last_stats_update = None
        
        # Log existing data
        self._log_existing_data()
        
        self._initialized = True
        
        async def warmup_cache(self):
            async with self._stats_lock:
                if not self.get_dataset_stats.cache_info().hits:
                    await self.get_dataset_stats()
        
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
        
    def start_collection(self, anomaly_type: str, node: str):
        """Start collecting training data for an anomaly case"""
        with self._lock:
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
            self.pre_anomaly_buffer = []
            self.post_anomaly_buffer = []
            
            # Start collecting pre-anomaly normal data
            self.is_collecting_normal = True
            logger.info(f"Started collecting pre-anomaly normal data for {self.current_case} at {self.collection_start_time}")
            
            # Schedule switch to anomaly collection after pre_anomaly_duration
            threading.Timer(self.pre_anomaly_duration, self._switch_to_anomaly_collection).start()
            
            # Start background cache maintenance
            self._cache_task = asyncio.create_task(self._maintain_cache())
        
    def _switch_to_anomaly_collection(self):
        """Switch from pre-anomaly to anomaly data collection"""
        with self._lock:
            if not self.current_case:
                return
                
            # Save pre-anomaly data as normal
            if self.pre_anomaly_buffer:
                self.normal_samples += len(self.pre_anomaly_buffer)
                self._save_normal_data(self.pre_anomaly_buffer, "pre_anomaly")
                
            # Switch to anomaly collection
            self.is_collecting_normal = False
            logger.info(f"Switched to anomaly data collection for {self.current_case}")
        
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

                # Save metadata for normal case
                self._save_case_metadata(
                    case_dir,
                    0,  # No pre-anomaly for normal case
                    len(self.normal_state_buffer),  # All samples are normal
                    0,  # No post-anomaly for normal case
                    "normal"  # Explicitly mark as normal
                )
                    
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
                
            if not self.metrics_buffer and not self.pre_anomaly_buffer:
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
                
                # Start collecting post-anomaly normal data
                self.is_collecting_normal = True
                logger.info("Started collecting post-anomaly normal data")
                
                # Schedule final save after post-anomaly collection
                threading.Timer(self.post_anomaly_duration, self._finalize_collection, args=[temp_dir, case_dir]).start()
                
            except Exception as e:
                logger.error(f"Failed to prepare for post-anomaly collection: {str(e)}")
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                self._clear_collection_state()
                raise
                
    def _save_case_metadata(self, case_path: str, pre_anomaly_count: int, anomaly_count: int, post_anomaly_count: int, anomaly_type: str) -> None:
        """Save case metadata for faster stats computation"""
        is_normal = anomaly_type == "normal"
        metadata = {
            "counts": {
                "normal": pre_anomaly_count + post_anomaly_count if not is_normal else anomaly_count,
                "anomaly": anomaly_count if not is_normal else 0
            },
            "type": "normal" if is_normal else "anomaly",
            "anomaly_type": "normal" if is_normal else anomaly_type.replace("_stress", "").replace("network_delay", "network"),
            "timestamp": datetime.now().isoformat()
        }
        metadata_path = os.path.join(case_path, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)
            logger.debug(f"Saved metadata for case {case_path}: {metadata}")

    def _finalize_collection(self, temp_dir: str, case_dir: str):
        """Finalize collection by saving all data"""
        with self._lock:
            try:
                # Save pre-anomaly data
                if self.pre_anomaly_buffer:
                    pre_anomaly_file = os.path.join(temp_dir, "pre_anomaly_metrics.json")
                    with open(pre_anomaly_file, "w") as f:
                        json.dump(self.pre_anomaly_buffer, f, indent=2)
                    self.normal_samples += len(self.pre_anomaly_buffer)
                    
                # Save anomaly data
                if self.metrics_buffer:
                    metrics_file = os.path.join(temp_dir, "metrics.json")
                    with open(metrics_file, "w") as f:
                        json.dump(self.metrics_buffer, f, indent=2)
                    self.anomaly_samples += len(self.metrics_buffer)
                    
                # Save post-anomaly data
                if self.post_anomaly_buffer:
                    post_anomaly_file = os.path.join(temp_dir, "post_anomaly_metrics.json")
                    with open(post_anomaly_file, "w") as f:
                        json.dump(self.post_anomaly_buffer, f, indent=2)
                    self.normal_samples += len(self.post_anomaly_buffer)
                    
                # Save anomaly label
                label_file = os.path.join(temp_dir, "label.json")
                with open(label_file, "w") as f:
                    json.dump(self.current_anomaly, f, indent=2)
                    
                # Verify files were written correctly
                files_to_check = [
                    ("metrics.json", self.metrics_buffer),
                    ("pre_anomaly_metrics.json", self.pre_anomaly_buffer),
                    ("post_anomaly_metrics.json", self.post_anomaly_buffer),
                    ("label.json", None)
                ]
                
                for filename, data in files_to_check:
                    filepath = os.path.join(temp_dir, filename)
                    if (data and not os.path.exists(filepath)) or \
                       (os.path.exists(filepath) and os.path.getsize(filepath) == 0):
                        raise Exception(f"Failed to write {filename} or file is empty")
                    
                # Atomically move the temporary directory to the final location
                if os.path.exists(case_dir):
                    shutil.rmtree(case_dir)
                shutil.move(temp_dir, case_dir)
                
                # Save metadata for faster stats computation
                self._save_case_metadata(
                    case_dir,
                    len(self.pre_anomaly_buffer),
                    len(self.metrics_buffer),
                    len(self.post_anomaly_buffer),
                    self.current_anomaly["type"] or "unknown"
                )
                
                logger.info(f"Successfully saved all data for {self.current_case}:")
                logger.info(f"  - Pre-anomaly samples: {len(self.pre_anomaly_buffer)}")
                logger.info(f"  - Anomaly samples: {len(self.metrics_buffer)}")
                logger.info(f"  - Post-anomaly samples: {len(self.post_anomaly_buffer)}")
                logger.info(f"  - Total normal samples: {self.normal_samples}")
                logger.info(f"  - Total anomaly samples: {self.anomaly_samples}")
                
            except Exception as e:
                logger.error(f"Failed to save training data: {str(e)}")
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                raise
            finally:
                self._clear_collection_state()
                
    def _clear_collection_state(self):
        """Clear all collection state variables"""
        self.current_case = None
        self.current_anomaly = None
        self.metrics_buffer = []
        self.pre_anomaly_buffer = []
        self.post_anomaly_buffer = []
        self.collection_start_time = None
        self.is_collecting_normal = False
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

            # Process metrics based on collection phase
            processed_metrics = self._process_raw_metrics(metrics)
            
            if self.is_collecting_normal:
                if self.current_anomaly:  # Pre-anomaly collection
                    self.pre_anomaly_buffer.extend(processed_metrics)
                    logger.debug(f"Added {len(processed_metrics)} metrics to pre-anomaly buffer")
                else:  # Normal state collection
                    self.normal_state_buffer.extend(processed_metrics)
                    logger.debug(f"Added {len(processed_metrics)} metrics to normal state buffer")
            else:  # Anomaly collection
                self.metrics_buffer.extend(processed_metrics)
                logger.debug(f"Added {len(processed_metrics)} metrics to anomaly buffer")
        
    def _process_raw_metrics(self, metrics: Dict[str, Any]) -> List[Dict]:
        """Process raw metrics into a list of timestamped metrics"""
        processed_metrics = []
        
        for node_ip, node_data in metrics.items():
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
                try:
                    filtered_metrics = {
                        "timestamp": timestamp,
                        "node": node_ip,
                        "metrics": {}
                    }
                    
                    for category, category_data in node_data.items():
                        if category == 'io':
                            # Handle IO metrics
                            filtered_metrics["metrics"][category] = {
                                'aggregated': {
                                    metric: data.get(timestamp, data.get('latest', 0))
                                    for metric, data in category_data['aggregated'].items()
                                }
                            }
                        else:
                            # For other categories
                            filtered_metrics["metrics"][category] = {
                                metric: data.get(timestamp, data.get('latest', 0))
                                for metric, data in category_data.items()
                            }
                            
                    processed_metrics.append(filtered_metrics)
                except Exception as e:
                    logger.error(f"Error processing metrics for node {node_ip} at timestamp {timestamp}: {str(e)}")
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
                loop = asyncio.get_event_loop()
                stats = await asyncio.wait_for(
                    loop.run_in_executor(self._stats_executor, self._compute_dataset_stats),
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

    def _compute_dataset_stats(self):
        """Get statistics about the collected training data with optimized caching"""
        if not self._should_update_stats_cache():
            logger.debug("Returning cached stats")
            return self._stats_cache

        logger.info("Calculating fresh dataset statistics")
        with self._lock:
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
            case_stats = []

            for i in range(0, len(case_dirs), batch_size):
                batch = case_dirs[i:i + batch_size]
                futures = []

                with ThreadPoolExecutor(max_workers=4) as executor:
                    for case_dir in batch:
                        case_path = os.path.join(self.data_dir, case_dir)
                        futures.append(
                            executor.submit(self._process_case_stats, case_path)
                        )

                    for future in concurrent.futures.as_completed(futures, timeout=10):
                        try:
                            case_stat = future.result()
                            if case_stat:
                                case_stats.append(case_stat)
                        except Exception as e:
                            logger.warning(f"Error processing case batch: {str(e)}")

            # Aggregate stats from all cases
            for case_stat in case_stats:
                stats["normal"] += case_stat["normal"]
                stats["anomaly"] += case_stat["anomaly"]
                for anomaly_type, count in case_stat.get("anomaly_types", {}).items():
                    stats["anomaly_types"][anomaly_type] += count

            # Calculate ratios and balance
            total = stats["normal"] + stats["anomaly"]
            stats["total_samples"] = total
            
            if total > 0:
                stats["normal_ratio"] = stats["normal"] / total
                stats["anomaly_ratio"] = stats["anomaly"] / total
                stats["is_balanced"] = abs(stats["normal_ratio"] - stats["anomaly_ratio"]) <= self.balance_threshold

            # Update cache with timestamp
            stats["last_updated"] = time.time()
            self._stats_cache = stats
            self._last_stats_update = time.time()
            logger.info(f"Updated dataset statistics cache: {stats}")
            return stats

    def _process_case_stats(self, case_path: str) -> Optional[Dict]:
        """Process case statistics with metadata optimization"""
        try:
            # Try to use metadata file first
            metadata_path = os.path.join(case_path, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path) as f:
                    metadata = json.load(f)
                    if metadata.get("type") == "normal":
                        return {
                            "normal": metadata["counts"]["normal"],
                            "anomaly": 0,
                            "anomaly_types": {}
                        }
                    return {
                        "normal": metadata["counts"]["normal"],
                        "anomaly": metadata["counts"]["anomaly"],
                        "anomaly_types": {
                            metadata["anomaly_type"]: metadata["counts"]["anomaly"]
                        }
                    }

            # Fall back to original processing if no metadata
            case_stat = {
                "normal": 0,
                "anomaly": 0,
                "anomaly_types": defaultdict(int)
            }

            # First check if this is a normal case by reading label.json
            label_file = os.path.join(case_path, "label.json")
            is_normal_case = False
            if os.path.exists(label_file):
                with open(label_file) as f:
                    label_data = json.load(f)
                    is_normal_case = label_data.get("type") == "normal"

            # Process metrics files with optimized reading
            for prefix in ["", "pre_anomaly_", "post_anomaly_"]:
                metrics_file = os.path.join(case_path, f"{prefix}metrics.json")
                if not os.path.exists(metrics_file):
                    continue

                # Use line counting for large files
                file_size = os.path.getsize(metrics_file)
                if file_size > 10_000_000:  # 10MB threshold
                    count = 0
                    with open(metrics_file, 'rb') as f:
                        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                        for line in iter(mm.readline, b''):
                            if b'{' in line:
                                count += 1
                        mm.close()
                    count = count // 2  # Adjust for JSON array structure
                else:
                    with open(metrics_file) as f:
                        data = json.load(f)
                        count = len(data) if isinstance(data, list) else 1

                # For normal cases, all samples are normal regardless of prefix
                if is_normal_case or prefix:
                    case_stat["normal"] += count
                else:
                    case_stat["anomaly"] += count

            # Process label data for anomaly type if not a normal case
            if not is_normal_case and os.path.exists(label_file):
                with open(label_file) as f:
                    label_data = json.load(f)
                    anomaly_type = label_data.get("type", "unknown")
                    base_type = anomaly_type.replace("_stress", "")
                    if "network_delay" in base_type:
                        base_type = "network"
                    if base_type in ["cpu", "memory", "io", "network", "unknown"]:
                        case_stat["anomaly_types"][base_type] = case_stat["anomaly"]

            # Save metadata for future use
            if is_normal_case:
                self._save_case_metadata(
                    case_path,
                    0,  # No pre-anomaly for normal case
                    case_stat["normal"],  # All samples are normal
                    0,  # No post-anomaly for normal case
                    "normal"  # Explicitly mark as normal
                )
            else:
                self._save_case_metadata(
                    case_path,
                    case_stat["normal"] // 2,  # Split between pre and post
                    case_stat["anomaly"],
                    case_stat["normal"] // 2,
                    list(case_stat["anomaly_types"].keys())[0] if case_stat["anomaly_types"] else "unknown"
                )

            return case_stat
        except Exception as e:
            logger.error(f"Error processing case stats for {case_path}: {str(e)}")
            return None

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

    def auto_balance_dataset(self):
        """Automatically balance the dataset by collecting required data"""
        logger.info("Starting auto-balance process")
        
        # Get current stats
        stats = self.get_dataset_stats()
        total = stats["normal"] + stats["anomaly"]
        
        if total == 0:
            # If no data, start with normal collection
            logger.info("No data available. Starting with normal collection")
            self.start_normal_collection()
            return
            
        # Calculate ratios
        normal_ratio = stats["normal"] / total if total > 0 else 0
        anomaly_ratio = stats["anomaly"] / total if total > 0 else 0
        
        # Check normal vs anomaly balance first
        if abs(normal_ratio - anomaly_ratio) > self.balance_threshold:
            if normal_ratio < anomaly_ratio:
                logger.info("Normal samples underrepresented. Starting normal collection")
                self.start_normal_collection()
            else:
                # Find least represented anomaly type
                anomaly_types = stats["anomaly_types"]
                min_type = min(anomaly_types.items(), key=lambda x: x[1])[0]
                logger.info(f"Anomaly type {min_type} underrepresented. Starting collection")
                self.start_collection(min_type, None)  # Let the system choose the node
        else:
            # Balance anomaly types
            anomaly_types = stats["anomaly_types"]
            avg_anomaly_count = stats["anomaly"] / len(anomaly_types)
            
            # Find most underrepresented type
            min_type = min(anomaly_types.items(), key=lambda x: x[1])[0]
            if anomaly_types[min_type] < avg_anomaly_count * 0.7:  # Allow 30% variance
                logger.info(f"Anomaly type {min_type} below average. Starting collection")
                self.start_collection(min_type, None)
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
                    
                # Process pre-anomaly data as normal
                pre_anomaly_file = os.path.join(case_path, "pre_anomaly_metrics.json")
                if os.path.exists(pre_anomaly_file):
                    with open(pre_anomaly_file) as f:
                        pre_anomaly_data = json.load(f)
                        features = self._process_metrics(pre_anomaly_data)
                        if features is not None:
                            X.extend(features)
                            y.extend([np.zeros(4) for _ in range(len(features))])  # Normal state
                            logger.info(f"Processed {len(features)} pre-anomaly samples from {case_dir}")
                            
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
                            
                # Process post-anomaly data as normal
                post_anomaly_file = os.path.join(case_path, "post_anomaly_metrics.json")
                if os.path.exists(post_anomaly_file):
                    with open(post_anomaly_file) as f:
                        post_anomaly_data = json.load(f)
                        features = self._process_metrics(post_anomaly_data)
                        if features is not None:
                            X.extend(features)
                            y.extend([np.zeros(4) for _ in range(len(features))])  # Normal state
                            logger.info(f"Processed {len(features)} post-anomaly samples from {case_dir}")
                            
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
        while True:
            await self.get_dataset_stats()
            await asyncio.sleep(30)  # Refresh every 30s

# Create singleton instance
training_service = TrainingService() 