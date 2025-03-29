import os
import json
import psutil
import time
import asyncio
import subprocess
import warnings
from typing import Dict, Any, List
from datetime import datetime
import threading
import logging
from functools import lru_cache, wraps
import pymysql
from ..services.training_service import training_service
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from config.dist_diagnosis_config import NODE_CONFIG

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

class MetricsCollectionMethod:
    SQL = "sql"
    OBDIAG = "obdiag"
    PSUTIL = "psutil"

class MetricsService:
    _instance = None
    _lock = threading.Lock()
    _metrics_cache = {}
    _last_collection = None
    CACHE_TTL = 30  # Cache time to live in seconds

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize the metrics service with OB connection details"""
        self.metrics = {}
        self.metrics_history = {}  # Store time series data
        self.timestamp = None
        self.collection_active = False
        self.collection_thread = None
        self.collection_interval = 5  # seconds
        self.event_loop = None
        self._metrics_cache = {}
        self._last_collection = None
        self.MAX_HISTORY_POINTS = 720  # 1 hour of data at 5s intervals
        self._previous_counter_values = {}  # Track previous values for counter metrics
        
        # Collection method configuration
        self.collection_method = os.getenv('METRICS_COLLECTION_METHOD', MetricsCollectionMethod.SQL)
        
        # Whether to send metrics to training service
        self.send_to_training = os.getenv('SEND_METRICS_TO_TRAINING', 'true').lower() == 'true'
        
        # Whether to check for workload state before sending to training
        self.check_workload_active = os.getenv('CHECK_WORKLOAD_ACTIVE', 'true').lower() == 'true'
        
        # Workload activity flag - indicates whether a workload is currently running
        self.workload_active = False
        
        if self.collection_method == MetricsCollectionMethod.OBDIAG:
            warnings.warn(
                "Using OBDIAG collection method which is deprecated. Consider switching to SQL method.",
                DeprecationWarning
            )
        
        # Validate required environment variables
        required_env_vars = ['OB_USER', 'OB_PASSWORD', 'OB_PORT']
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        # OB connection config
        try:
            self.ob_config = {
                'user': os.environ['OB_USER'],
                'password': os.environ['OB_PASSWORD'],
                'database': os.getenv('OB_DATABASE', 'oceanbase'),
                'host': os.getenv('OB_HOST', '127.0.0.1'),
                'port': int(os.environ['OB_PORT'])
            }
            logger.info(f"Initialized OB connection config with host={self.ob_config['host']}, port={self.ob_config['port']}, user={self.ob_config['user']}")
        except (KeyError, ValueError) as e:
            logger.error(f"Failed to initialize OB connection config: {e}")
            raise
        
        # Essential metrics for anomaly detection
        self.essential_metrics = {
            'cpu': [
                'cpu usage',
                'worker time',
                'cpu time',
            ],
            'memory': [
                'total memstore used',
                'active memstore used',
                'memstore limit',
                'memory usage',
                'observer memory hold size',
            ],
            'io': [
                'palf write io count to disk',
                'palf write size to disk',
                'clog trans log total size',
                'io read count',
                'io write count',
                'io read delay',
                'io write delay',
                'blockscaned data micro block count',
                'accessed data micro block count',
                'data micro block cache hit',
                'index micro block cache hit',
                'row cache miss',
                'row cache hit',
            ],
            'network': [
                'rpc packet in',
                'rpc packet in bytes',
                'rpc packet out',
                'rpc packet out bytes',
                'rpc deliver fail',
                'rpc net delay',
                'rpc net frame delay',
                'mysql packet in',
                'mysql packet in bytes',
                'mysql packet out',
                'mysql packet out bytes',
                'mysql deliver fail',
            ],
            'transactions': [
                'trans commit count',
                'trans timeout count',
                'sql update count',
                'active sessions',
                'sql select count',
                'trans system trans count',
                'trans rollback count',
                'sql fail count',
                'request queue time',
                'trans user trans count',
                'commit log replay count',
                'sql delete count',
                'sql other count',
                'ps prepare count',
                'ps execute count',
                'sql inner insert count',
                'sql inner update count',
                'memstore write lock succ count',
                'memstore write lock fail count',
                'DB time',
                'local trans total used time',
                'distributed trans total used time',
                'sql local count',
                'sql remote count',
                'sql distributed count'
            ]
        }
        
        # Get counter metrics
        self._counter_metrics = set()
        for metrics in self.essential_metrics.values():
            for metric in metrics:
                if any(kw in metric.lower() for kw in ['delay', 'count', 'time', 'total', 'hit', 'miss', 'packet', 'sessions']):
                    # Skip if it contains 'memstore' and 'total', skip 'io read count'
                    if not ('total memstore used' in metric.lower() or 'request queue time' in metric.lower()):
                        self._counter_metrics.add(metric.strip().lower())

        # Create reverse mapping from metric name to category
        self.metric_to_category = {}
        for category, metrics in self.essential_metrics.items():
            for metric in metrics:
                self.metric_to_category[metric.strip().lower()] = category
        
        logger.info(f"Initialized {len(self.metric_to_category)} essential metrics across {len(self.essential_metrics)} categories")

        # Start metrics collection
        self.start_collection()

    def set_collection_method(self, method: str):
        """Set the metrics collection method."""
        if method not in [MetricsCollectionMethod.SQL, MetricsCollectionMethod.OBDIAG, MetricsCollectionMethod.PSUTIL]:
            raise ValueError(f"Invalid collection method: {method}")
        
        if method == MetricsCollectionMethod.OBDIAG:
            warnings.warn(
                "Switching to OBDIAG collection method which is deprecated. Consider using SQL method.",
                DeprecationWarning
            )
            
        self.collection_method = method
        logger.info(f"Switched metrics collection method to: {method}")

    def start_collection(self):
        """Start collecting metrics in a separate thread."""
        if self.collection_thread and self.collection_thread.is_alive():
            logger.warning("Metrics collection already running")
            return

        self.collection_active = True
        self.collection_thread = threading.Thread(target=self._collect_metrics_loop)
        self.collection_thread.start()
        logger.info("Started metrics collection")

    def stop_collection(self):
        """Stop collecting metrics."""
        logger.info("Stopping metrics collection")
        self.collection_active = False
        if self.collection_thread:
            self.collection_thread.join()
            self.collection_thread = None
        logger.info("Stopped metrics collection")

    def _collect_metrics_loop(self):
        """Continuously collect metrics while collection is active."""
        logger.info("Starting metrics collection loop")
        
        # Create event loop for this thread
        self.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.event_loop)
        
        # Get reference to the training service event loop
        training_loop = getattr(training_service, '_event_loop', None)
        
        while self.collection_active:
            try:
                if self.collection_method == MetricsCollectionMethod.SQL:
                    metrics = self.event_loop.run_until_complete(self._collect_sql_metrics_only())
                    
                    # Only send metrics to training if configured to do so and either:
                    # 1. Not checking workload state, or
                    # 2. Workload is active
                    # 3. OR training service indicates it's collecting data
                    
                    # Check if training service is actively collecting
                    is_training_collecting = (
                        hasattr(training_service, 'current_case') and 
                        training_service.current_case is not None
                    )
                    
                    should_send_to_training = (
                        self.send_to_training and 
                        (
                            not self.check_workload_active or 
                            self.workload_active or
                            is_training_collecting  # Always send if training service is collecting
                        )
                    )
                    
                    # Log the decision factors
                    logger.debug(f"Metrics collection: send_to_training={self.send_to_training}, check_workload_active={self.check_workload_active}, workload_active={self.workload_active}, is_training_collecting={is_training_collecting}")
                    
                    # Add detailed debug logging about the metrics
                    if metrics:
                        logger.debug(f"Collected metrics from {len(metrics)} nodes.")
                        if len(metrics) > 0:
                            first_node = next(iter(metrics))
                            logger.debug(f"First node metrics categories: {list(metrics[first_node].keys())}")
                            # Sample a metric from each category
                            for category, data in metrics[first_node].items():
                                if isinstance(data, dict) and len(data) > 0:
                                    sample_metric = next(iter(data))
                                    logger.debug(f"Sample {category} metric: {sample_metric}")
                    
                    # If metrics were collected and should be sent, send to training service
                    if metrics and training_loop and should_send_to_training:
                        logger.info(f"Dispatching metrics to training service: {len(metrics)} nodes of data (non-blocking)")
                        future = asyncio.run_coroutine_threadsafe(
                            training_service.add_metrics(metrics), 
                            training_loop
                        )
                        future.add_done_callback(lambda fut: logger.info(f"Successfully sent metrics to training service: {len(metrics)} nodes") if fut.exception() is None else logger.error(f"Failed to send metrics to training service: {fut.exception()}"))
                    elif metrics and not should_send_to_training:
                        logger.debug(f"Not sending metrics to training service due to configuration or workload state: send_to_training={self.send_to_training}, check_workload_active={self.check_workload_active}, workload_active={self.workload_active}")
                    elif not metrics:
                        logger.debug("No metrics collected to send to training service")
                    
                elif self.collection_method == MetricsCollectionMethod.OBDIAG:
                    try:
                        self.event_loop.run_until_complete(self._collect_obdiag_metrics())
                    except Exception as e:
                        logger.error(f"Error collecting metrics with obdiag: {e}")
                        logger.info("Falling back to psutil collection")
                        self.event_loop.run_until_complete(self._collect_psutil_metrics())
                else:
                    self.event_loop.run_until_complete(self._collect_psutil_metrics())
                    
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                if not self.collection_active:
                    break
                time.sleep(self.collection_interval)
                
        # Clean up event loop
        self.event_loop.close()
        self.event_loop = None
        logger.info("Metrics collection loop ended")

    async def _collect_sql_metrics_only(self):
        """Collect metrics using direct SQL queries to OceanBase without sending to training service."""
        timestamp = datetime.now().isoformat()
        current_metrics = {}
        
        # Get list of nodes from config
        nodes = NODE_CONFIG.get('node_list', [])
        
        for node in nodes:
            try:
                # Use full node address including port
                node_ip = node.split(':')[0]
                # Connect using node address directly
                logger.debug(f"Connecting to node {node_ip}")
                conn = pymysql.connect(
                    host=self.ob_config['host'],
                    user=self.ob_config['user'],
                    password=self.ob_config['password'],
                    database=self.ob_config['database'],
                    port=self.ob_config['port']
                )
                
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT name, value 
                        FROM gv$sysstat 
                        WHERE con_id = %s AND svr_ip = %s
                    """, (1, node_ip))
                    
                    # Initialize node metrics structure
                    node_metrics = {
                        'cpu': {},
                        'memory': {},
                        'io': {},
                        'network': {},
                        'transactions': {}
                    }
                    
                    # Categorize metrics using exact name matching
                    metrics_found = set()
                    for name, value in cursor.fetchall():
                        clean_name = name.strip().lower()
                        category = self.metric_to_category.get(clean_name)
                        
                        if category:
                            value = float(value)
                            
                            # Check if this is a counter metric
                            is_counter = clean_name in self._counter_metrics
                            counter_key = (node_ip, category, name)
                            
                            if is_counter:
                                # Get previous value, defaulting to current value for first occurrence
                                prev_value = self._previous_counter_values.get(counter_key, value)
                                # Calculate delta
                                delta = value - prev_value
                                # Store current value for next iteration
                                self._previous_counter_values[counter_key] = value
                                # Use delta as the value, ensure non-negative
                                value = max(delta, 0)
                            
                            node_metrics[category][name] = value
                            metrics_found.add(clean_name)
                            
                            # Store in history
                            if node_ip not in self.metrics_history:
                                self.metrics_history[node_ip] = {}
                            if category not in self.metrics_history[node_ip]:
                                self.metrics_history[node_ip][category] = {}
                            if name not in self.metrics_history[node_ip][category]:
                                self.metrics_history[node_ip][category][name] = []
                                
                            history = self.metrics_history[node_ip][category][name]
                            history.append({"timestamp": timestamp, "value": value})
                            
                            # Keep only last MAX_HISTORY_POINTS
                            if len(history) > self.MAX_HISTORY_POINTS:
                                self.metrics_history[node_ip][category][name] = history[-self.MAX_HISTORY_POINTS:]
                    
                    # Log missing essential metrics
                    missing_metrics = set(self.metric_to_category.keys()) - metrics_found
                    if missing_metrics:
                        logger.warning(f"Missing essential metrics for node {node_ip}: {missing_metrics}")
                    
                    # Add empty placeholders for missing metrics to ensure consistent structure
                    for category in node_metrics:
                        expected_metrics = [m for m, c in self.metric_to_category.items() if c == category]
                        for metric in expected_metrics:
                            if metric not in metrics_found:
                                # Add placeholder with zero value for missing metrics
                                node_metrics[category][metric] = 0.0
                                
                                # Add to history with zero values for consistency
                                if node_ip not in self.metrics_history:
                                    self.metrics_history[node_ip] = {}
                                if category not in self.metrics_history[node_ip]:
                                    self.metrics_history[node_ip][category] = {}
                                if metric not in self.metrics_history[node_ip][category]:
                                    self.metrics_history[node_ip][category][metric] = []
                                    
                                history = self.metrics_history[node_ip][category][metric]
                                history.append({"timestamp": timestamp, "value": 0.0})
                                
                                # Keep only last MAX_HISTORY_POINTS
                                if len(history) > self.MAX_HISTORY_POINTS:
                                    self.metrics_history[node_ip][category][metric] = history[-self.MAX_HISTORY_POINTS:]
                    
                    current_metrics[node_ip] = node_metrics
                
                conn.close()
                
            except Exception as e:
                logger.error(f"Error collecting metrics from {node_ip}: {e}")
                current_metrics[node_ip] = {
                    'cpu': {},
                    'memory': {},
                    'io': {},
                    'network': {},
                    'transactions': {}
                }

        if not current_metrics:
            logger.warning("No metrics were collected from any nodes")
        else:
            logger.debug(f"Successfully collected metrics from nodes: {list(current_metrics.keys())}")
        
        # Add correlation metrics between nodes for compound anomaly detection
        # Calculate node relationships based on metric correlations
        if len(current_metrics) > 1:
            node_correlations = self._calculate_node_correlations(current_metrics)
            
            # Store node correlations in metrics for use in compound anomaly detection
            for node_ip in current_metrics:
                if 'node_relationships' not in current_metrics[node_ip]:
                    current_metrics[node_ip]['node_relationships'] = {}
                current_metrics[node_ip]['node_relationships'] = node_correlations.get(node_ip, {})
        
        self.metrics = current_metrics
        self.timestamp = timestamp
        
        return current_metrics

    def _calculate_node_correlations(self, metrics: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Calculate correlations between nodes based on their metrics.
        This helps identify compound anomalies that affect multiple nodes.
        """
        node_ips = list(metrics.keys())
        correlations = {node: {} for node in node_ips}
        
        # For each pair of nodes
        for i in range(len(node_ips)):
            for j in range(i+1, len(node_ips)):
                node1 = node_ips[i]
                node2 = node_ips[j]
                
                # Calculate correlations for each metric category
                category_correlations = {}
                
                for category in ['cpu', 'memory', 'io', 'network', 'transactions']:
                    node1_metrics = metrics[node1].get(category, {})
                    node2_metrics = metrics[node2].get(category, {})
                    
                    # Get intersection of metrics that both nodes have
                    common_metrics = set(node1_metrics.keys()) & set(node2_metrics.keys())
                    
                    if common_metrics:
                        # Calculate correlation for this category
                        total_corr = 0.0
                        count = 0
                        
                        for metric in common_metrics:
                            val1 = node1_metrics[metric]
                            val2 = node2_metrics[metric]
                            
                            # Simple correlation measure - ratio of smaller to larger value
                            # A value of 1.0 means identical metrics, 0.0 means completely different
                            if val1 == 0 and val2 == 0:
                                # Both zero means perfect correlation for this metric
                                metric_corr = 1.0
                            elif val1 == 0 or val2 == 0:
                                # One zero means no correlation
                                metric_corr = 0.0
                            else:
                                # Compute ratio (smaller / larger)
                                metric_corr = min(val1, val2) / max(val1, val2)
                                
                            total_corr += metric_corr
                            count += 1
                            
                        # Average correlation for this category
                        category_correlations[category] = total_corr / count if count > 0 else 0.0
                    else:
                        # No common metrics in this category
                        category_correlations[category] = 0.0
                
                # Overall correlation is average of category correlations
                overall_corr = sum(category_correlations.values()) / len(category_correlations)
                
                # Store correlations bidirectionally
                correlations[node1][node2] = {
                    'overall': overall_corr,
                    'categories': category_correlations
                }
                
                correlations[node2][node1] = {
                    'overall': overall_corr,
                    'categories': category_correlations
                }
                
        return correlations

    async def _collect_sql_metrics(self):
        """Legacy method that calls _collect_sql_metrics_only but doesn't return the metrics."""
        await self._collect_sql_metrics_only()

    @DeprecationWarning
    async def _collect_obdiag_metrics(self):
        """[DEPRECATED] Collect metrics using obdiag"""
        warnings.warn(
            "Using deprecated obdiag collection method. Consider switching to SQL method.",
            DeprecationWarning
        )
        
        timestamp = datetime.now().isoformat()
        dir_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        base_pack_dir = f"obdiag_gather_pack_{dir_timestamp}"
        obdiag_cmd = os.getenv('OBDIAG_CMD', 'obdiag')
        
        cmd = f"{obdiag_cmd} gather sysstat --store_dir={base_pack_dir}"
        logger.debug(f"Running command: {cmd}")
        process = await asyncio.create_subprocess_shell(
            cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        logger.debug(f"Command output: {stdout.decode()}")

        if process.returncode != 0:
            logger.error(f"Error running obdiag: {stderr.decode()}")
            raise Exception("Failed to collect metrics with obdiag")

        if not os.path.exists(base_pack_dir):
            logger.error(f"Base pack directory {base_pack_dir} does not exist")
            raise Exception("Obdiag output directory not found")

        nested_dirs = [d for d in os.listdir(base_pack_dir) if d.startswith("obdiag_gather_pack_")]
        if not nested_dirs:
            logger.error(f"No nested pack directory found in {base_pack_dir}")
            raise Exception("Nested pack directory not found")

        pack_dir = os.path.join(base_pack_dir, nested_dirs[0])
        logger.debug(f"Using pack directory: {pack_dir}")

        metrics = {}
        for zip_file in os.listdir(pack_dir):
            if zip_file.startswith("sysstat_"):
                node_address = zip_file.split("_")[1]
                logger.debug(f"Processing metrics for node: {node_address}")
                zip_path = os.path.join(pack_dir, zip_file)
                
                extract_dir = pack_dir
                try:
                    # hide unzip output
                    cmd = f"unzip -o {zip_path} -d {extract_dir} > /dev/null 2>&1"
                    logger.debug(f"Extracting metrics data: {cmd}")
                    subprocess.run(cmd, shell=True, check=True)
                    metrics[node_address] = await self._parse_node_metrics(extract_dir)
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to extract metrics data for {node_address}: {e}")
                    continue

        if not metrics:
            logger.warning("No metrics were collected from any nodes")
        else:
            logger.info(f"Successfully collected metrics from nodes: {list(metrics.keys())}")
            try:
                await training_service.add_metrics(metrics)
                logger.debug("Successfully sent metrics to training service")
            except Exception as e:
                logger.error(f"Failed to send metrics to training service: {e}")

        self.metrics = metrics
        self.timestamp = timestamp

        logger.debug(f"Cleaning up directory: {base_pack_dir}")
        subprocess.run(f"rm -rf {base_pack_dir}", shell=True)

    async def _collect_psutil_metrics(self):
        """[DEPRECATED] Collect metrics using psutil as fallback."""
        warnings.warn(
            "Using deprecated psutil collection method. Consider switching to SQL method.",
            DeprecationWarning
        )
        
        metrics = {
            "localhost": {
                "cpu": {
                    "util": self._get_cpu_metrics(),
                },
                "memory": self._get_memory_metrics(),
                "io": {
                    "aggregated": self._get_io_metrics(),
                },
                "network": self._get_network_metrics(),
            }
        }
        
        self.metrics = metrics
        self.timestamp = datetime.now().isoformat()

        try:
            await training_service.add_metrics(metrics)
            logger.debug("Successfully sent psutil metrics to training service")
        except Exception as e:
            logger.error(f"Failed to send psutil metrics to training service: {e}")

    def _get_cpu_metrics(self):
        """Get CPU utilization metrics using psutil."""
        value = psutil.cpu_percent(interval=1)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return {
            "latest": value,
            "min": 0,
            "max": 100,
            "data": [{"timestamp": timestamp, "value": value}]
        }

    def _get_memory_metrics(self):
        """Get memory utilization metrics using psutil."""
        mem = psutil.virtual_memory()
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return {
            "total": {
                "latest": mem.total / (1024 * 1024),  # Convert to MB
                "min": 0,
                "max": mem.total / (1024 * 1024),
                "data": [{"timestamp": timestamp, "value": mem.total / (1024 * 1024)}]
            },
            "used": {
                "latest": mem.used / (1024 * 1024),
                "min": 0,
                "max": mem.total / (1024 * 1024),
                "data": [{"timestamp": timestamp, "value": mem.used / (1024 * 1024)}]
            },
            "util": {
                "latest": mem.percent,
                "min": 0,
                "max": 100,
                "data": [{"timestamp": timestamp, "value": mem.percent}]
            }
        }

    def _get_io_metrics(self):
        """Get disk I/O metrics using psutil."""
        io = psutil.disk_io_counters()
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        if not io:
            return {
                "read_bytes": {
                    "latest": 0, "min": 0, "max": 100,
                    "data": [{"timestamp": timestamp, "value": 0}]
                },
                "write_bytes": {
                    "latest": 0, "min": 0, "max": 100,
                    "data": [{"timestamp": timestamp, "value": 0}]
                },
                "util": {
                    "latest": 0, "min": 0, "max": 100,
                    "data": [{"timestamp": timestamp, "value": 0}]
                }
            }
        
        disk_usage = psutil.disk_usage('/')
        read_bytes = io.read_bytes / (1024 * 1024)  # Convert to MB
        write_bytes = io.write_bytes / (1024 * 1024)
        
        return {
            "read_bytes": {
                "latest": read_bytes,
                "min": 0,
                "max": read_bytes * 1.2,  # 20% headroom
                "data": [{"timestamp": timestamp, "value": read_bytes}]
            },
            "write_bytes": {
                "latest": write_bytes,
                "min": 0,
                "max": write_bytes * 1.2,
                "data": [{"timestamp": timestamp, "value": write_bytes}]
            },
            "util": {
                "latest": disk_usage.percent,
                "min": 0,
                "max": 100,
                "data": [{"timestamp": timestamp, "value": disk_usage.percent}]
            }
        }

    def _get_network_metrics(self):
        """Get network I/O metrics using psutil."""
        net = psutil.net_io_counters()
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        bytes_sent = net.bytes_sent / 1024  # Convert to KB
        bytes_recv = net.bytes_recv / 1024
        
        return {
            "bytes_sent": {
                "latest": bytes_sent,
                "min": 0,
                "max": bytes_sent * 1.2,
                "data": [{"timestamp": timestamp, "value": bytes_sent}]
            },
            "bytes_recv": {
                "latest": bytes_recv,
                "min": 0,
                "max": bytes_recv * 1.2,
                "data": [{"timestamp": timestamp, "value": bytes_recv}]
            },
            "packets_sent": {
                "latest": net.packets_sent,
                "min": 0,
                "max": net.packets_sent * 1.2,
                "data": [{"timestamp": timestamp, "value": net.packets_sent}]
            },
            "packets_recv": {
                "latest": net.packets_recv,
                "min": 0,
                "max": net.packets_recv * 1.2,
                "data": [{"timestamp": timestamp, "value": net.packets_recv}]
            }
        }

    async def _parse_node_metrics(self, node_dir: str) -> Dict[str, Any]:
        """[DEPRECATED] Parse metrics from node directory"""
        logger.debug(f"Parsing metrics from directory: {node_dir}")
        metrics = {}

        node_dirs = [d for d in os.listdir(node_dir) if d.startswith("sysstat_") and not d.endswith(".zip")]
        if not node_dirs:
            logger.error("No node-specific directory found")
            return metrics

        node_specific_dir = os.path.join(node_dir, node_dirs[0])
        logger.debug(f"Using node-specific directory: {node_specific_dir}")

        # Parse CPU metrics
        cpu_file = os.path.join(node_specific_dir, "one_day_cpu_data.txt")
        if os.path.exists(cpu_file):
            with open(cpu_file) as f:
                lines = f.readlines()
                cpu_indices = {
                    "user": 1, "system": 2, "wait": 3,
                    "hirq": 4, "sirq": 5, "util": 6,
                }
                metrics["cpu"] = self._parse_time_series_data(lines, cpu_indices)
        else:
            logger.debug(f"CPU metrics file not found at {cpu_file}")

        # Parse Memory metrics
        mem_file = os.path.join(node_specific_dir, "one_day_mem_data.txt")
        if os.path.exists(mem_file):
            with open(mem_file) as f:
                lines = f.readlines()
                mem_indices = {
                    "free": 1, "used": 2, "buff": 3,
                    "cach": 4, "total": 5, "util": 6,
                }
                metrics["memory"] = self._parse_time_series_data(lines, mem_indices)
        else:
            logger.debug(f"Memory metrics file not found at {mem_file}")

        # Parse IO metrics
        io_file = os.path.join(node_specific_dir, "tsar_io_data.txt")
        if os.path.exists(io_file):
            with open(io_file) as f:
                lines = f.readlines()
                io_indices = {
                    "read_bytes": 1,
                    "write_bytes": 2,
                    "util": 3,
                }
                metrics["io"] = {
                    "aggregated": self._parse_time_series_data(lines, io_indices)
                }
        else:
            logger.debug(f"IO metrics file not found at {io_file}")

        # Parse Network metrics
        net_file = os.path.join(node_specific_dir, "tsar_traffic_data.txt")
        if os.path.exists(net_file):
            with open(net_file) as f:
                lines = f.readlines()
                net_indices = {
                    "bytes_sent": 1,
                    "bytes_recv": 2,
                    "packets_sent": 3,
                    "packets_recv": 4,
                }
                metrics["network"] = self._parse_time_series_data(lines, net_indices)
        else:
            logger.debug(f"Network metrics file not found at {net_file}")

        return metrics

    def _parse_time_series_data(
        self, lines: List[str], metric_indices: Dict[str, int], skip_first_n: int = 0
    ) -> Dict[str, Any]:
        """[DEPRECATED] Parse time series data from lines into metrics"""
        metrics = {}
        for metric_name, idx in metric_indices.items():
            data_points = []
            latest = 0.0
            
            for line in lines[skip_first_n:]:
                if not line.strip() or line.startswith(("#", "Time", "MAX", "MEAN", "MIN", "categories")):
                    continue
                    
                parts = line.split()
                if len(parts) > idx:
                    try:
                        timestamp = parts[0]
                        value = float(parts[idx])
                        data_points.append({"timestamp": timestamp, "value": value})
                        latest = value
                    except (ValueError, IndexError):
                        continue
            
            metrics[metric_name] = {
                "data": data_points[-60:],  # Keep last 60 points
                "latest": latest
            }
            
        return metrics

    @timed_lru_cache(seconds=30, maxsize=1)
    def get_last_metric_point(self) -> Dict[str, Any]:
        """Get the last collected metrics point with fluctuation analysis."""
        if not self.metrics_history:
            return {"metrics": {}, "timestamp": None}

        processed_metrics = {}
        FLUCTUATION_Z_SCORE = 2.0  # 2 standard deviations
        HISTORY_WINDOW = 60  # Last 60 data points (5 minutes of data)
        MIN_HISTORY = 5  # Require at least 5 data points for analysis
        
        for node_ip in self.metrics_history:
            processed_metrics[node_ip] = {}
            
            for category in self.metrics_history[node_ip]:
                processed_metrics[node_ip][category] = {}
                
                for metric_name, history in self.metrics_history[node_ip][category].items():
                    # Ensure history is a list; if not, wrap it as a one-element list.
                    if not isinstance(history, list):
                        history = [{"timestamp": self.timestamp if self.timestamp else "", "value": history}]
                    
                    if len(history) < MIN_HISTORY:
                        processed_metrics[node_ip][category][metric_name] = history
                        continue
                    try:
                        values = [h['value'] for h in history[-HISTORY_WINDOW:-1]]  # Use all but the last point.
                        mean = sum(values) / len(values)
                        # Calculate variance in a numerically stable way
                        squared_diff_sum = sum((x - mean) ** 2 for x in values)
                        std_dev = (squared_diff_sum / len(values)) ** 0.5 if len(values) > 1 else 0
                    except Exception as e:
                        logger.error(f"Error calculating stats for {metric_name}: {e}")
                        processed_metrics[node_ip][category][metric_name] = history
                        continue
                    
                    current_value = history[-1]['value']
                    if abs(std_dev) < 1e-9:
                        z_score = 0.0
                        pct_change = 0.0
                    else:
                        rel_std_dev = std_dev / abs(mean) if abs(mean) > 1e-9 else 0
                        if rel_std_dev < 1e-4:
                            is_monotonic = all(history[i]['value'] <= history[i+1]['value'] for i in range(len(history)-2, len(history)-min(5, len(history)-1), -1))
                            if is_monotonic and mean > 1e6:
                                z_score = min((current_value - mean) / std_dev, 1.0)
                            else:
                                z_score = (current_value - mean) / std_dev
                        else:
                            z_score = (current_value - mean) / std_dev
                        pct_change = (current_value - mean) / mean if abs(mean) > 1e-9 else 0.0
                    
                    processed_history = history.copy()
                    processed_history[-1] = {
                        **history[-1],
                        'z_score': round(z_score, 2),
                        'pct_change': round(pct_change, 4),
                        'has_fluctuation': abs(z_score) >= FLUCTUATION_Z_SCORE and (abs(pct_change) > 0.001 or (abs(current_value - mean) > max(1.0, mean * 0.01)))
                    }
                    processed_metrics[node_ip][category][metric_name] = processed_history
        
        last_metric_point = {}
        for node_ip in processed_metrics:
            last_metric_point[node_ip] = {}
            for category in processed_metrics[node_ip]:
                last_metric_point[node_ip][category] = {}
                for metric_name in processed_metrics[node_ip][category]:
                    last_metric_point[node_ip][category][metric_name] = processed_metrics[node_ip][category][metric_name][-1]
        
        return {
            "metrics": last_metric_point,
            "timestamp": self.timestamp
        }

    @timed_lru_cache(seconds=30, maxsize=1)
    def get_detailed_metrics(self, node_address: str = None, category: str = None) -> Dict[str, Any]:
        """Get detailed time series metrics for specific node and/or category."""
        if not self.metrics_history:
            return {}
            
        if node_address and node_address not in self.metrics_history:
            return {}
            
        if node_address:
            metrics = self.metrics_history[node_address]
            if category:
                return {category: metrics.get(category, {})}
            return metrics
            
        if category:
            return {
                node: node_metrics.get(category, {})
                for node, node_metrics in self.metrics_history.items()
            }
        
        # Add node relationship data to metrics when returning all metrics
        result = self.metrics_history.copy()
        
        # If we have current metrics with node_relationships, include them
        if self.metrics:
            for node in result:
                if node in self.metrics and 'node_relationships' in self.metrics[node]:
                    result[node]['node_relationships'] = self.metrics[node]['node_relationships']
            
        return result
        
    def _dict_to_hashable(self, d):
        """Convert a dictionary to a hashable tuple of (key, value) pairs sorted by key."""
        return tuple(sorted((k, v) for k, v in d.items()))
        
    def get_selected_detailed_metrics(self, node_address: str, selected_metrics: Dict[str, str]) -> Dict[str, Any]:
        """Get detailed time series metrics for specific node, but only for selected metrics in each category.
        
        Args:
            node_address: IP address of the node
            selected_metrics: Dictionary mapping category names to metric names
            
        Returns:
            Dictionary with categories as keys and dictionaries of selected metrics as values
        """
        # Convert selected_metrics to a hashable format and call the cached implementation
        hashable_metrics = self._dict_to_hashable(selected_metrics)
        return self._get_selected_detailed_metrics_cached(node_address, hashable_metrics)
    
    @timed_lru_cache(seconds=30, maxsize=100)
    def _get_selected_detailed_metrics_cached(self, node_address: str, hashable_metrics) -> Dict[str, Any]:
        """Implementation of get_selected_detailed_metrics with cacheable parameters."""
        if not self.metrics_history or node_address not in self.metrics_history:
            return {"error": f"No metrics history for node {node_address}"}
            
        node_metrics = self.metrics_history[node_address]
        result = {}
        
        # Convert hashable metrics tuple back to a dict for processing
        selected_metrics = dict(hashable_metrics)
        
        # For each category in selected_metrics, get only the specified metric
        for category, metric_name in selected_metrics.items():
            if category not in node_metrics:
                # Skip categories that don't exist
                continue
                
            category_metrics = node_metrics[category]
            if metric_name not in category_metrics:
                # Skip metrics that don't exist
                continue
                
            # Initialize category in result if it doesn't exist
            if category not in result:
                result[category] = {}
                
            # Add only the selected metric for this category
            result[category][metric_name] = category_metrics[metric_name]
            
        return result

    def set_workload_active(self, active: bool):
        """Set whether a workload is currently active"""
        self.workload_active = active
        logger.info(f"Workload active status set to: {active}")
        
    def toggle_send_to_training(self, enabled: bool):
        """Toggle whether metrics should be sent to training"""
        self.send_to_training = enabled
        logger.info(f"Send to training set to: {enabled}")
        
    def set_check_workload_active(self, check: bool):
        """Set whether to check for active workload before sending to training"""
        self.check_workload_active = check
        logger.info(f"Check workload active before training set to: {check}")
        
    def _send_metrics_to_training(self, metrics: Dict[str, Any]) -> None:
        """Send metrics to training service if configured to do so."""
        # Only send metrics if enabled and either not checking workload or workload is active
        if (self.send_to_training and 
            (not self.check_workload_active or self.workload_active)):
            
            try:
                from app.services.training_service import training_service
                asyncio.create_task(training_service.add_metrics(metrics))
            except Exception as e:
                logger.error(f"Failed to send metrics to training: {str(e)}")
        else:
            logger.info(
                f"Not sending metrics to training service due to configuration or workload state: "
                f"send_to_training={self.send_to_training}, "
                f"check_workload_active={self.check_workload_active}, "
                f"workload_active={self.workload_active}"
            )

# Create a singleton instance
metrics_service = MetricsService()
