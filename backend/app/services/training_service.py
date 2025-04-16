import glob
import os
import json
import time
import logging
import threading
import asyncio
import traceback
import ijson
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, wraps
from typing import Dict, List, Any, Optional
import numpy as np
from app.services.diagnosis_service import diagnosis_service
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
    _test_mode = False

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
        
        # Start background cache maintenance task only if not in test mode
        self._test_mode = os.environ.get("TRAINING_TEST_MODE", "false").lower() == "true"
        if not self._test_mode:
            self._cache_task = self._event_loop.create_task(self._maintain_cache())
        else:
            self._cache_task = None

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
        """Start collecting training data for a single anomaly case"""
        async with self._lock:
            # Validate node parameter
            if not node:
                # Get list of available nodes that haven't been used for this type
                from app.services.k8s_service import k8s_service
                available_nodes = await k8s_service.get_available_nodes()
                used_nodes = self.anomaly_variations.get(anomaly_type, set())
                unused_nodes = [n for n in available_nodes if n not in used_nodes]

                if not unused_nodes:
                    # If all nodes used, clear history to allow reuse
                    self.anomaly_variations[anomaly_type] = set()
                    unused_nodes = available_nodes

                node = unused_nodes[0] if unused_nodes else (available_nodes[0] if available_nodes else None)

                if not node:
                    logger.error(f"Cannot start collection for {anomaly_type}: No available nodes.")
                    raise ValueError("No available nodes found.")

                logger.info(f"Selected node {node} for anomaly type {anomaly_type}")

            # Track this variation
            if anomaly_type not in self.anomaly_variations:
                self.anomaly_variations[anomaly_type] = set()
            if isinstance(node, list):
                for n in node:
                    self.anomaly_variations[anomaly_type].add(n)
            else:
                self.anomaly_variations[anomaly_type].add(node)
            logger.info(f"Anomaly variations: {self.anomaly_variations}")

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
            logger.info(f"Started single anomaly data collection for {self.current_case} at {self.collection_start_time}")
            logger.info(f"Metrics buffer initialized: {self.metrics_buffer is not None}, length: {len(self.metrics_buffer)}")

            # Schedule automatic stop after collection_duration
            asyncio.create_task(self._auto_stop_collection())

    async def start_compound_collection(self, anomaly_types: List[str], node: str):
        """Start collecting training data for a compound anomaly case"""
        async with self._lock:
            # Validate node parameter
            if not node:
                # Get list of available nodes that haven't been used for this type
                from app.services.k8s_service import k8s_service
                available_nodes = await k8s_service.get_available_nodes()
                # Create a compound key for tracking variations based on the input list
                # Sort the list first for consistency
                compound_key = "+".join(sorted(anomaly_types))
                used_nodes = self.anomaly_variations.get(compound_key, set())
                unused_nodes = [n for n in available_nodes if n not in used_nodes]

                if not unused_nodes:
                    # If all nodes used, clear history for this specific compound type to allow reuse
                    self.anomaly_variations[compound_key] = set()
                    unused_nodes = available_nodes

                node = unused_nodes[0] if unused_nodes else (available_nodes[0] if available_nodes else None)

                if not node:
                    # Use anomaly_types (the list) in the error message
                    logger.error(f"Cannot start collection for compound types {anomaly_types}: No available nodes.")
                    raise ValueError("No available nodes found for compound collection.")

                logger.info(f"Selected node {node} for compound anomaly types {anomaly_types}")

            # If node was specified, track it using the compound key
            else:
                # Sort the list first for consistency
                compound_key = "+".join(sorted(anomaly_types))
                if compound_key not in self.anomaly_variations:
                    self.anomaly_variations[compound_key] = set()
                # Add the selected node(s) to the variation tracking for the compound key
                if isinstance(node, list):
                    for n in node:
                        self.anomaly_variations[compound_key].add(n)
                else:
                    self.anomaly_variations[compound_key].add(node)

            logger.info(f"Anomaly variations tracked: {self.anomaly_variations}")

            # Set up collection parameters with potentially longer duration for compound anomalies
            self.collection_start_time = datetime.now().isoformat()
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

            # Create case name with all anomaly types
            # Sort for consistent naming
            anomaly_str = "_plus_".join(sorted(anomaly_types))
            self.current_case = f"case_compound_{anomaly_str}_{node}_{timestamp}"

            # Store anomaly data with all types
            self.current_anomaly = {
                "type": "compound",
                # Store sorted list
                "subtypes": sorted(anomaly_types),
                "node": node,
                "start_time": self.collection_start_time,
                "duration": self.collection_duration * 1.5  # Optional: 50% longer for compound anomalies
            }

            # Initialize metrics buffer
            self.metrics_buffer = []
            self.is_collecting_normal = False
            logger.info(f"Started compound anomaly data collection for {self.current_case} at {self.collection_start_time}")
            logger.info(f"Collecting {len(anomaly_types)} simultaneous anomalies: {', '.join(sorted(anomaly_types))}")

            # Schedule automatic stop after extended collection_duration
            # Use the duration stored in current_anomaly
            compound_duration = self.current_anomaly["duration"]
            asyncio.create_task(self._auto_stop_collection_with_duration(compound_duration))

    async def _auto_stop_collection(self):
        """Automatically stop collection after the specified duration"""
        try:
            await asyncio.sleep(self.collection_duration)
            if self.current_case:  # Only stop if still collecting
                logger.info(f"Auto-stopping collection after {self.collection_duration} seconds")
                # Use await here to ensure full completion of the stop operation
                await self.stop_collection(save_post_data=True)
                # Log completion for debugging
                logger.info("Auto-stop collection completed successfully")
        except Exception as e:
            logger.error(f"Error in auto-stop collection: {str(e)}")
            # Ensure state is cleaned up even on error
            await self._clear_collection_state()

    async def _auto_stop_collection_with_duration(self, duration: int):
        """Automatically stop collection after the specified duration"""
        try:
            await asyncio.sleep(duration)
            if self.current_case:  # Only stop if still collecting
                logger.info(f"Auto-stopping collection after {duration} seconds")
                # Use await here to ensure full completion of the stop operation
                await self.stop_collection(save_post_data=True)
                # Log completion for debugging
                logger.info("Auto-stop collection completed successfully")
        except Exception as e:
            logger.error(f"Error in auto-stop collection: {str(e)}")
            # Ensure state is cleaned up even on error
            await self._clear_collection_state()

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
            if self.is_collecting_normal:
                from app.services.k8s_service import k8s_service
                active_anomalies = await k8s_service.get_active_anomalies()
                if active_anomalies:
                    logger.info("Active anomalies detected during normal collection; switching to anomaly collection")
                    await self.stop_normal_collection()
                    active_exp = active_anomalies[0]
                    parts = active_exp.split('-')
                    anomaly_type = f"{parts[1]}_{parts[2]}" if len(parts) >= 3 else (parts[1] if len(parts) > 1 else active_exp)
                    await self.start_collection(anomaly_type, None)
                    return
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
        """Process raw SQL-based metrics snapshot data into a list containing one entry per node."""
        processed_metrics = []
        
        # Use the current time as the snapshot timestamp for all nodes
        snapshot_timestamp = datetime.now().isoformat()
        logger.info(f"Processing raw SQL-based metrics for {len(metrics)} nodes")
        logger.info(f"Sample node IPs: {list(metrics.keys())[:3]}")
        
        for node_ip, node_data in metrics.items():
            logger.info(f"Node {node_ip} has {len(node_data)} metric categories: {list(node_data.keys())}")
            entry = {
                "timestamp": snapshot_timestamp,
                "node": node_ip,
                "metrics": {}
            }
            
            # For each metric category, since SQL-based collection returns a snapshot,
            # simply copy the scalar metric values into the processed entry.
            for category, category_data in node_data.items():
                entry["metrics"][category] = category_data
            
            processed_metrics.append(entry)
            logger.info(f"Processed metrics for node {node_ip} at {snapshot_timestamp}")
        
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
                        # For anomaly cases (single or compound):
                        # 1. Count pre_anomaly as normal (if exists)
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
                            # Handle compound vs single type for stats breakdown
                            anomaly_type = label_data.get("type")
                            if anomaly_type == "compound":
                                # For compound, maybe log under a special key or distribute count?
                                # For simplicity, let's log under 'compound' for now.
                                stats["anomaly_types"]["compound"] += count
                                # Optionally, could increment counts for each subtype involved:
                                # for subtype in label_data.get("subtypes", []):
                                #     stats["anomaly_types"][subtype] += count / len(label_data["subtypes"]) # Distribute count
                            elif anomaly_type:
                                stats["anomaly_types"][anomaly_type] += count
                            logger.debug(f"Added {count} anomaly samples of type {anomaly_type} from {case_dir}")
                        
                        # 3. Count post_anomaly as normal (if exists)
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
        logger.info("Starting auto-balance process")
        from app.services.k8s_service import k8s_service
        active_anomalies = await k8s_service.get_active_anomalies()
        if active_anomalies:
            if self.is_collecting_normal:
                logger.info("Active anomalies detected while normal collection in progress; stopping normal collection.")
                await self.stop_normal_collection()
            if not self.current_anomaly:
                active_exp = active_anomalies[0]
                parts = active_exp.split('-')
                anomaly_type = f"{parts[1]}_{parts[2]}" if len(parts) >= 3 else (parts[1] if len(parts) > 1 else active_exp)
                logger.info(f"Active anomaly {active_exp} detected; starting anomaly collection for type: {anomaly_type}")
                await self.start_collection(anomaly_type, None)
                return {"success": True, "message": f"Started anomaly collection for active anomaly type: {anomaly_type}"}

        # If there's an active collection, don't start a new one
        if self.current_anomaly or self.is_collecting_normal:
            logger.info("Collection already in progress, skipping auto-balance")
            return {"success": False, "message": "Collection already in progress"}
        
        # Calculate how balanced each anomaly type is
        stats = await self.get_dataset_stats()
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
            if anomaly_types and self.anomaly_groups.get(anomaly_type, 0) < self.target_groups_per_type:
                logger.info(f"Collecting variation {self.anomaly_groups[anomaly_type]+1} for {anomaly_type}")
                await self.start_collection(anomaly_type, None)  # Let the system choose the node
                return {"success": True, "message": f"Started collecting variation for {anomaly_type}"}

        logger.info("All required anomaly variations have been collected, dataset is already well-balanced")
        return {"success": True, "message": "Dataset is already well-balanced"}

    def get_training_data(self) -> tuple:
        """Process all collected datasets into 2D training format (samples × features)"""
        try:
            # Scan all case directories
            case_directories = []
            for dataset_dir in glob.glob(os.path.join(self.data_dir, "case_*")):
                if os.path.isdir(dataset_dir):
                    case_directories.append(dataset_dir)
            
            logger.info(f"Found {len(case_directories)} case directories for training")
            if not case_directories:
                return np.array([]), np.array([])
            
            # Process all cases in parallel for better performance
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(self._process_case, case_directories))
            
            # Filter out empty or failed results
            valid_results = [r for r in results if r is not None and r[0] is not None and r[1] is not None]
            if not valid_results:
                logger.warning("No valid training data could be processed")
                return np.array([]), np.array([])
            
            # Stack all samples into 2D arrays
            X_all = []
            y_all = []
            
            for X_case, y_case in valid_results:
                X_all.append(X_case)
                y_all.append(y_case)
            
            if not X_all:
                logger.warning("No samples to stack")
                return np.array([]), np.array([])
            
            # Stack all samples into 2D arrays
            X_array = np.vstack(X_all)
            y_array = np.vstack(y_all)
            
            # Handle NaN and infinite values
            if np.any(np.isnan(X_array)) or np.any(np.isinf(X_array)):
                logger.warning("Replacing NaN/infinite values in features with zeros")
                X_array = np.nan_to_num(X_array, nan=0.0, posinf=0.0, neginf=0.0)
            
            if np.any(np.isnan(y_array)) or np.any(np.isinf(y_array)):
                logger.warning("Replacing NaN/infinite values in labels with zeros")
                y_array = np.nan_to_num(y_array, nan=0.0, posinf=0.0, neginf=0.0)
            
            logger.info(f"Final training data shapes: X={X_array.shape}, y={y_array.shape}")
            return X_array, y_array
                
        except Exception as e:
            logger.error(f"Error processing training data: {str(e)}")
            logger.debug(traceback.format_exc())
            return np.array([]), np.array([])
    
    def _process_case(self, case_dir: str) -> tuple:
        """Process a single case directory into 2D features and labels (samples × features)"""
        try:
            if(self._test_mode):
                # Ensure there's a running event loop in the current thread
                import asyncio
                try:
                    asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
            case_name = os.path.basename(case_dir)
            possible_timestamp = case_name.split('_')[-1]
            if not self._is_valid_timestamp(possible_timestamp):
                logger.warning(f"Invalid timestamp in case {case_name}: {possible_timestamp}. Skipping this case.")
                return None
            
            # Get label data
            label_file = os.path.join(case_dir, "label.json")
            if not os.path.exists(label_file):
                logger.warning(f"Label file missing for case {case_name}")
                return None
                
            with open(label_file, 'r') as f:
                label_data = json.load(f)
            
            # Get metrics data
            metrics_file = os.path.join(case_dir, "metrics.json")
            if not os.path.exists(metrics_file):
                logger.warning(f"Metrics file missing for case {case_name}")
                return None
            
            # Read metrics with error handling
            try:
                with open(metrics_file, 'r') as f:
                    metrics_data = json.load(f)
                
                # Validate metrics data format
                if not isinstance(metrics_data, list):
                    logger.warning(f"Invalid metrics data format in {case_name}: {type(metrics_data)}")
                    return None
                
                # Handle empty metrics data
                if len(metrics_data) == 0:
                    logger.warning(f"Empty metrics data in {case_name}")
                    return None
                    
            except json.JSONDecodeError:
                logger.error(f"JSON decode error in metrics file for {case_name}")
                return None
            
            # Get IP to pod name mapping from k8s_service
            from app.services.k8s_service import k8s_service
            ip_to_name = k8s_service.ip_to_name_map
            
            # Helper function to find matching node
            def find_matching_node(target_node, available_nodes):
                # Direct match
                if target_node in available_nodes:
                    return target_node
                
                # Check if any available node is an IP and maps to target_node
                for node in available_nodes:
                    # If node is an IP, check its pod name mapping
                    pod_name = ip_to_name.get(node)
                    if pod_name and (pod_name == target_node or target_node in pod_name or pod_name in target_node):
                        logger.info(f"Matched IP {node} to pod name {pod_name}")
                        return node
                    
                    # Try direct substring matching as fallback
                    if target_node in node or node in target_node:
                        return node
                
                return None

            # Process each metrics entry into 2D arrays
            X_samples = []  # Will hold all feature vectors
            y_samples = []  # Will hold all labels
            
            # Get target node for anomaly
            target_node = label_data.get("node")
            
            # Process each metrics entry
            for entry in metrics_data:
                if not isinstance(entry, dict):
                    continue
                    
                timestamp = entry.get('timestamp')
                node = entry.get('node')
                metrics = entry.get('metrics')
                
                if not (timestamp and node and metrics):
                    continue
                
                # Process metrics for this time point
                processed_metrics = diagnosis_service._process_metrics({'node': node, 'metrics': metrics})
                if processed_metrics is None:
                    continue
                
                # Determine if this node should be labeled as anomalous
                is_anomalous = False
                if target_node:
                    matched_node = find_matching_node(target_node, [node])
                    if matched_node:
                        is_anomalous = True
                
                # Create label vector - zeros for normal nodes, anomaly type for anomalous node
                if is_anomalous:
                    label_vector = self._create_label_vector(label_data)
                else:
                    label_vector = np.zeros(len(self._get_anomaly_types()))
                
                # Add to samples
                X_samples.append(processed_metrics)
                y_samples.append(label_vector)
            
            if not X_samples:
                logger.warning(f"No valid samples extracted from case {case_name}")
                return None
            
            X_array = np.array(X_samples)
            y_array = np.array(y_samples)
            
            logger.info(f"Processed case {case_name}: {len(X_samples)} samples, X shape: {X_array.shape}, y shape: {y_array.shape}")
            return X_array, y_array
            
        except Exception as e:
            logger.error(f"Error processing case {os.path.basename(case_dir)}: {str(e)}")
            logger.debug(traceback.format_exc())
            return None
            
    def _get_anomaly_types(self):
        """Get list of supported anomaly types"""
        return [
            "cpu_stress",
            "network_bottleneck",
            "cache_bottleneck",
        ]

    def _create_label_vector(self, label_data: Dict) -> np.ndarray:
        """Create label vector for anomaly data (handles single and compound types)"""
        # Define indices based on the ORDERED list from _get_anomaly_types()
        anomaly_types_map = {name: idx for idx, name in enumerate(self._get_anomaly_types())}

        # Initialize label vector with zeros (multi-hot encoding)
        label_vector = np.zeros(len(anomaly_types_map))

        # Handle compound anomalies
        if label_data.get("type") == "compound" and "subtypes" in label_data:
            for subtype in label_data["subtypes"]:
                if subtype in anomaly_types_map:
                    label_vector[anomaly_types_map[subtype]] = 1
            logger.debug(f"Created compound label vector for subtypes {label_data['subtypes']}: {label_vector}")
        else:
            # Handle single anomalies
            anomaly_type = label_data.get("type")
            # Special handling for 'normal' samples - return zeros vector
            if anomaly_type == "normal":
                return label_vector

            if anomaly_type in anomaly_types_map:
                label_vector[anomaly_types_map[anomaly_type]] = 1
                logger.debug(f"Created single label vector for anomaly type {anomaly_type}: {label_vector}")
            else:
                logger.warning(f"Unknown anomaly type in label data: {anomaly_type}")

        logger.info(f"Final label vector for case with label data {label_data}: {label_vector}")
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
                logger.info("Collection stop completed successfully")
            except Exception as e:
                logger.error(f"Stop collection failed: {str(e)}")
                # Force clear state on error to prevent stuck state
                asyncio.create_task(self._clear_collection_state())

        stop_task.add_done_callback(callback)
        logger.info("Initiated stop collection process")
        
        # Return the task so it can be awaited if needed
        return stop_task

    async def start_compound_collection(self, anomaly_types: list, node: str = None):
        """
        Start collecting training data for compound anomaly scenarios (multiple simultaneous anomalies)

        Args:
            anomaly_types: List of anomaly types to simulate simultaneously
            node: Optional target node, if None a node will be selected automatically
        """
        if not isinstance(anomaly_types, list) or len(anomaly_types) < 2: # Ensure it's a list with >= 2 types
            raise ValueError("Compound collection requires a list of at least 2 anomaly types")

        async with self._lock:
            # Validate anomaly types are supported (using the defined list)
            supported_types = self._get_anomaly_types()
            for atype in anomaly_types:
                if atype not in supported_types:
                    raise ValueError(f"Unsupported anomaly type: {atype}. Valid types: {supported_types}")

            # Make sure metrics service is aware we're collecting data
            try:
                from app.services.metrics_service import metrics_service
                metrics_service.toggle_send_to_training(True)
                logger.info("Enabled sending metrics to training service for compound collection")
            except Exception as e:
                logger.error(f"Failed to enable metrics collection for compound scenario: {e}")

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
                    # If all nodes used, clear history for this specific compound type to allow reuse
                    self.anomaly_variations[compound_key] = set()
                    unused_nodes = available_nodes

                node = unused_nodes[0] if unused_nodes else (available_nodes[0] if available_nodes else None) # Handle no nodes

                if not node:
                     logger.error(f"Cannot start compound collection for {anomaly_types}: No available nodes.")
                     raise ValueError("No available nodes found for compound collection.")

                logger.info(f"Selected node {node} for compound anomaly types {anomaly_types}")

            # Track this variation using the compound key
            compound_key = "+".join(sorted(anomaly_types)) # Ensure key is consistent
            if compound_key not in self.anomaly_variations:
                self.anomaly_variations[compound_key] = set()

            # Add the selected node(s) to the variation tracking for the compound key
            if isinstance(node, list):
                for n in node:
                    self.anomaly_variations[compound_key].add(n)
            else:
                self.anomaly_variations[compound_key].add(node)
            logger.info(f"Anomaly variations: {self.anomaly_variations}")

            # Set up collection parameters with potentially longer duration for compound anomalies
            self.collection_start_time = datetime.now().isoformat()
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

            # Create case name with all anomaly types
            # Sort for consistent naming
            anomaly_str = "_plus_".join(sorted(anomaly_types))
            self.current_case = f"case_compound_{anomaly_str}_{node}_{timestamp}"

            # Store anomaly data with all types
            self.current_anomaly = {
                "type": "compound",
                # Store sorted list
                "subtypes": sorted(anomaly_types),
                "node": node,
                "start_time": self.collection_start_time,
                "duration": self.collection_duration * 1.5  # Optional: 50% longer for compound anomalies
            }

            # Initialize metrics buffer
            self.metrics_buffer = []
            self.is_collecting_normal = False
            logger.info(f"Started compound anomaly data collection for {self.current_case} at {self.collection_start_time}")
            logger.info(f"Collecting {len(anomaly_types)} simultaneous anomalies: {', '.join(sorted(anomaly_types))}")

            # Schedule automatic stop after extended collection_duration
            # Use the duration stored in current_anomaly
            compound_duration = self.current_anomaly["duration"]
            asyncio.create_task(self._auto_stop_collection_with_duration(compound_duration))

    async def _auto_stop_collection_with_duration(self, duration: float):
        """Automatically stop collection after a specific duration."""
        try:
            await asyncio.sleep(duration)
            if self.current_case:  # Only stop if still collecting this case
                logger.info(f"Auto-stopping collection after {duration} seconds for case {self.current_case}")
                await self.stop_collection(save_post_data=True)
                logger.info(f"Auto-stop collection completed successfully for {self.current_case}")
        except Exception as e:
            logger.error(f"Error in auto-stop collection with duration: {str(e)}")
            # Ensure state is cleaned up even on error
            await self._clear_collection_state()

    def test_training_data(self):
        """Test function to diagnose training data issues and manually train the model"""
        try:
            # Import the diagnosis service
            from app.services.diagnosis_service import diagnosis_service
            
            logger.info("Starting manual training data test")
            
            # Get training data
            X, y = self.get_training_data()
            
            # Check if we have any data
            if len(X) == 0 or len(y) == 0:
                logger.error("No training data available")
                return {
                    "success": False,
                    "error": "No training data available",
                    "X_shape": (0, 0),
                    "y_shape": (0, 0)
                }
            
            # Log data shapes
            logger.info(f"Training data shapes: X={X.shape}, y={y.shape}")
            
            # Check for NaN or infinity values
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
            
            # Verify diagnosis service is available
            if diagnosis_service is None:
                logger.error("Diagnosis service not available")
                return {
                    "success": False,
                    "error": "Diagnosis service not available",
                    "X_shape": X.shape,
                    "y_shape": y.shape
                }
            
            # Train the model
            logger.info("Starting model training with validated data")
            try:
                result = diagnosis_service.train(X, y)
                logger.info(f"Training completed successfully: {result.get('model_name', 'unnamed model')}")
                return {
                    "success": True,
                    "result": result,
                    "X_shape": X.shape,
                    "y_shape": y.shape
                }
            except Exception as e:
                logger.error(f"Error during model training: {str(e)}")
                logger.error(traceback.format_exc())
                return {
                    "success": False,
                    "error": str(e),
                    "X_shape": X.shape,
                    "y_shape": y.shape
                }
                
        except Exception as e:
            logger.error(f"Error in test_training_data: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }

    def _is_valid_timestamp(self, ts: str) -> bool:
        from datetime import datetime
        try:
            # Try ISO format
            datetime.fromisoformat(ts)
            return True
        except ValueError:
            try:
                # Try YYYYMMDDHHMMSS format
                datetime.strptime(ts, "%Y%m%d%H%M%S")
                return True
            except ValueError:
                return False

# Create singleton instance
training_service = TrainingService() 