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
        
        # Collection method configuration
        self.collection_method = os.getenv('METRICS_COLLECTION_METHOD', MetricsCollectionMethod.SQL)
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
                'active cpu time',
            ],
            'memory': [
                'total memstore used',
                'active memstore used',
                'memstore limit',
                'memory limit',
                'memory hold',
                'observer memory hold size'
            ],
            'io': [
                'palf write io count to disk',
                'palf write size to disk',
                'clog trans log total size',
                'io read count',
                'io write count',
                'io read delay',
                'io write delay'
            ],
            'network': [
                'rpc packet in bytes',
                'rpc packet out bytes',
                'rpc net delay',
                'rpc net frame delay',
                'mysql packet in bytes',
                'mysql packet out bytes',
                'mysql deliver fail',
                'rpc deliver fail'
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
            ]
        }

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
        
        while self.collection_active:
            try:
                if self.collection_method == MetricsCollectionMethod.SQL:
                    self.event_loop.run_until_complete(self._collect_sql_metrics())
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

    async def _collect_sql_metrics(self):
        """Collect metrics using direct SQL queries to OceanBase."""
        timestamp = datetime.now().isoformat()
        current_metrics = {}
        
        # Get list of nodes from config
        nodes = NODE_CONFIG.get('node_list', [])
        
        for node in nodes:
            try:
                # Use full node address including port
                node_ip = node.split(':')[0]
                # Connect using node address directly
                logger.info(f"Connecting to node {node_ip}")
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
            logger.info(f"Successfully collected metrics from nodes: {list(current_metrics.keys())}")
            try:
                await training_service.add_metrics(current_metrics)
                logger.info("Successfully sent metrics to training service")
            except Exception as e:
                logger.error(f"Failed to send metrics to training service: {e}")

        self.metrics = current_metrics
        self.timestamp = timestamp

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
                if not line.strip() or line.startswith(("#", "Time", "MAX", "MEAN", "MIN")):
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
    def get_metrics(self) -> Dict[str, Any]:
        """Get the latest collected metrics with history."""
        if not self.metrics:
            return {"metrics": {}, "timestamp": None}
            
        return {
            "metrics": self.metrics_history,
            "timestamp": self.timestamp
        }

    @timed_lru_cache(seconds=30, maxsize=1)
    def get_system_metrics(self) -> Dict:
        """Get aggregated system-wide metrics."""
        try:
            metrics = {
                'cpu_usage': 0.0,
                'memory_usage': 0.0,
                'disk_usage': 0.0,
                'network_io': 0.0,
                'transactions': 0.0
            }
            
            if not self.metrics:
                return metrics
                
            node_count = len(self.metrics)
            if node_count == 0:
                return metrics
                
            for node_address, node_data in self.metrics.items():
                # CPU usage from active CPU time and total time
                cpu_metrics = node_data.get('cpu', {})
                active_time = cpu_metrics.get('active cpu time', 0)
                total_time = cpu_metrics.get('cpu_total_time', 1)  # Avoid div by zero
                metrics['cpu_usage'] += (active_time / total_time) * 100 if total_time > 0 else 0
                
                # Memory usage percentage
                mem_metrics = node_data.get('memory', {})
                used = mem_metrics.get('total_memstore_used', 0)
                limit = mem_metrics.get('memstore_limit', 1)  # Avoid div by zero
                metrics['memory_usage'] += (used / limit) * 100 if limit > 0 else 0
                
                # IO operations per second
                io_metrics = node_data.get('io', {})
                metrics['disk_usage'] += (
                    io_metrics.get('io_read_count', 0) +
                    io_metrics.get('io_write_count', 0)
                )
                
                # Network bytes per second
                net_metrics = node_data.get('network', {})
                metrics['network_io'] += (
                    net_metrics.get('rpc packet in bytes', 0) +
                    net_metrics.get('rpc packet out bytes', 0) +
                    net_metrics.get('mysql packet in bytes', 0) +
                    net_metrics.get('mysql packet out bytes', 0)
                )
                
                # Transaction rate
                trans_metrics = node_data.get('transactions', {})
                metrics['transactions'] += trans_metrics.get('trans commit count', 0)
            
            # Average across nodes
            for key in metrics:
                metrics[key] = round(metrics[key] / node_count, 2)
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating system metrics: {e}")
            return {
                'cpu_usage': 0.0,
                'memory_usage': 0.0,
                'disk_usage': 0.0,
                'network_io': 0.0,
                'transactions': 0.0
            }

    @timed_lru_cache(seconds=30, maxsize=100)
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
            
        return self.metrics_history

# Create a singleton instance
metrics_service = MetricsService()
