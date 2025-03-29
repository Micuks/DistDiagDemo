import logging
import psutil
import subprocess
import time
import os
import mysql.connector
import threading
import re
from typing import Dict, List, Optional
from ..schemas.workload import WorkloadType, WorkloadMetrics, Task, WorkloadStatus
import pymysql
from enum import Enum
from dotenv import load_dotenv
from datetime import datetime
import uuid
from kubernetes import client, config

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env'))

logger = logging.getLogger(__name__)

class WorkloadService:
    def __init__(self):
        self.active_workloads: Dict[str, subprocess.Popen] = {}
        self._last_io_stats = None
        self._last_io_time = None
        self.tasks: Dict[str, Task] = {}  # Store tasks in memory
        # Database connection parameters
        self.db_host = os.getenv('OB_HOST', 'localhost')
        self.db_port = int(os.getenv('OB_PORT', '3306'))
        self.db_user = os.getenv('OB_USER', 'root')
        self.db_password = os.getenv('OB_PASSWORD', '')
        self.db_name = os.getenv('OB_NAME', 'sbtest')
        self.tpcc_db = os.getenv('TPCC_DB', 'tpcc')
        self.tpch_db = os.getenv('TPCH_DB', 'tpch')
        self.workload_outputs = {}
        # Initialize with default zones, will be updated with actual zones
        self.ob_zones = []
        # Node to port mapping
        self.available_nodes = self._fetch_available_nodes()
        self.node_port_mapping = self._build_node_port_mapping()
        # Fetch zones from database during initialization
        self._fetch_zones()
        logger.info("WorkloadService initialized")
        logger.info(f"Database connection: host={self.db_host}, port={self.db_port}, user={self.db_user}")

    def _build_node_port_mapping(self):
        """Build node to port mapping from available nodes"""
        # Map each available node to a port
        # Use a dictionary to track assigned ports
        used_ports = {}
        base_port = 22811
        port_map = {}

        # First try to assign based on node number if it follows the pattern obcluster-X-*
        for node in self.available_nodes:
            # Check if node matches the pattern obcluster-X-* where X is a number
            import re
            node_match = re.match(r'obcluster-(\d+)-.*', node)
            if node_match:
                # Use the node number as index
                node_index = int(node_match.group(1)) - 1
                port = base_port + node_index * 10
                # If port is already used, assign it sequentially
                while port in used_ports.values():
                    port += 10
                port_map[node] = port
                used_ports[node] = port
            else:
                # For nodes that don't match pattern, assign sequentially
                for i in range(9):  # Maximum 9 ports (22811, 22821, ..., 22891)
                    port = base_port + i * 10
                    if port not in used_ports.values():
                        port_map[node] = port
                        used_ports[node] = port
                        break

        # Add fallback entries for node1, node2, etc. for backward compatibility
        for i, node_name in enumerate(['node1', 'node2', 'node3']):
            if node_name not in port_map:
                port = base_port + i * 10
                # Make sure not to override existing ports
                while port in used_ports.values():
                    port += 10
                port_map[node_name] = port

        logger.info(f"Built node port mapping: {port_map}")
        return port_map
        
    def _fetch_zones(self) -> List[str]:
        """Fetch OceanBase zones from database"""
        try:
            logger.info("Fetching OceanBase zones from database")
            conn = mysql.connector.connect(
                host=self.db_host,
                port=self.db_port,
                user=self.db_user,
                password=self.db_password
            )
            cursor = conn.cursor()
            
            # Query zones from DBA_OB_ZONES view
            cursor.execute("SELECT zone FROM oceanbase.DBA_OB_ZONES")
            zones = [row[0] for row in cursor.fetchall()]
            
            if zones:
                logger.info(f"Found OceanBase zones: {zones}")
                self.ob_zones = zones
            else:
                logger.warning("No zones found, falling back to default zones")
                self.ob_zones = ['zone1', 'zone2', 'zone3']
                
            return self.ob_zones
        except Exception as e:
            logger.exception(f"Error fetching zones: {str(e)}")
            # Fall back to default zones
            self.ob_zones = ['zone1', 'zone2', 'zone3']
            return self.ob_zones
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()

    def create_task(self, name: str, workload_type: WorkloadType, workload_config: Dict, anomalies: List[Dict] = None) -> Task:
        """Create a new task"""
        logger.info(f"Creating task with name: {name}, type: {workload_type}")
        task_id = str(uuid.uuid4())
        task = Task(
            id=task_id,
            name=name,
            workload_id=f"{workload_type.value}_{workload_config.get('threads', 1)}",
            workload_type=workload_type,
            workload_config=workload_config,
            anomalies=anomalies or [],
            start_time=datetime.utcnow(),
            status=WorkloadStatus.RUNNING
        )
        self.tasks[task_id] = task
        logger.info(f"Created task {task_id}, total tasks: {len(self.tasks)}")
        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID"""
        return self.tasks.get(task_id)

    def get_all_tasks(self) -> List[Task]:
        """Get all tasks"""
        return list(self.tasks.values())

    def update_task_status(self, task_id: str, status: WorkloadStatus, error_message: Optional[str] = None) -> Optional[Task]:
        """Update task status"""
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        task.status = status
        if error_message:
            task.error_message = error_message
        if status in [WorkloadStatus.STOPPED, WorkloadStatus.ERROR]:
            task.end_time = datetime.utcnow()
        
        return task

    def get_active_tasks(self) -> List[Task]:
        """Get all active tasks"""
        logger.info(f"Getting active tasks, total tasks: {len(self.tasks)}")
        for task_id, task in self.tasks.items():
            logger.info(f"Task {task_id}: {task.name}, status: {task.status}, type: {type(task.status)}")
        
        active_tasks = [task for task in self.tasks.values() if task.status == WorkloadStatus.RUNNING]
        logger.info(f"Found {len(active_tasks)} active tasks")
        return active_tasks

    def cleanup_old_tasks(self, hours_threshold: int = 12) -> int:
        """Remove stopped tasks older than the specified threshold (default 12 hours)
        
        Returns:
            int: Number of tasks that were removed
        """
        current_time = datetime.utcnow()
        removed_count = 0
        
        # Make a copy of keys since we'll be modifying the dictionary
        task_ids = list(self.tasks.keys())
        
        for task_id in task_ids:
            task = self.tasks[task_id]
            
            # Skip if task is still running
            if task.status == WorkloadStatus.RUNNING:
                continue
                
            # If task has end_time (it was properly stopped), check if it's old enough to remove
            if task.end_time:
                time_diff = current_time - task.end_time
                hours_diff = time_diff.total_seconds() / 3600
                
                if hours_diff >= hours_threshold:
                    del self.tasks[task_id]
                    removed_count += 1
                    logger.info(f"Removed old task {task_id} ({task.name}) - stopped {hours_diff:.1f} hours ago")
            
            # If task has no end_time but is marked as stopped, it might be in an inconsistent state
            # In this case, we'll also remove it if it's old enough based on start_time
            elif task.status in [WorkloadStatus.STOPPED, WorkloadStatus.ERROR]:
                time_diff = current_time - task.start_time
                hours_diff = time_diff.total_seconds() / 3600
                
                if hours_diff >= hours_threshold:
                    del self.tasks[task_id]
                    removed_count += 1
                    logger.info(f"Removed inconsistent task {task_id} ({task.name}) - started {hours_diff:.1f} hours ago but never properly ended")
        
        return removed_count

    def _check_tables_exist(self, port: int = None) -> bool:
        """Check if sysbench tables already exist"""
        try:
            conn = mysql.connector.connect(
                host=self.db_host,
                port=port or self.db_port,
                user=self.db_user,
                password=self.db_password,
                database=self.db_name
            )
            cursor = conn.cursor()
            
            # Check if sbtest1 exists (as a representative table)
            cursor.execute("SHOW TABLES LIKE 'sbtest1'")
            exists = cursor.fetchone() is not None
            
            return exists
        except Exception as e:
            logger.exception(f"Error checking tables on port {port or self.db_port}: {str(e)}")
            return False
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()

    def prepare_database(self, workload_type: str) -> bool:
        """Prepare the database for sysbench workload"""
        try:
            logger.info(f"Preparing database for {workload_type} workload")
            
            # Create the database on proxy node only
            if not self._create_database_on_node(self.db_port):
                logger.error(f"Failed to create database on proxy node")
                return False
            
            # Check if tables already exist
            if self._check_tables_exist():
                logger.info("Tables already exist, skipping preparation")
                return True
            
            # Build sysbench prepare command with default database connection
            cmd = (
                f"sysbench oltp_read_write "
                f"--mysql-host={self.db_host} "
                f"--mysql-port={self.db_port} "
                f"--mysql-user={self.db_user} "
                f"--mysql-password={self.db_password} "
                f"--mysql-db={self.db_name} "
                f"--tables=10 "
                f"--table-size=100000 "
                f"--threads=4 "
                "prepare"
            )
            
            logger.info(f"Executing command: {cmd}")
            process = subprocess.Popen(
                cmd.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            stdout, stderr = process.communicate()
            
            if stdout:
                logger.info(f"Command output: {stdout}")
            if stderr:
                logger.error(f"Command error output: {stderr}")
            
            if process.returncode == 0:
                logger.info("Database preparation completed successfully")
            else:
                logger.error(f"Database preparation failed with return code {process.returncode}")
            
            return process.returncode == 0
        except Exception as e:
            logger.exception("Error preparing database")
            return False

    def _create_database_on_node(self, port: int) -> bool:
        """Create the database on a specific node"""
        try:
            node_type = "proxy node" if port == self.db_port else f"node with port {port}"
            logger.info(f"Creating database {self.db_name} on {node_type}")
            conn = mysql.connector.connect(
                host=self.db_host,
                port=port,
                user=self.db_user,
                password=self.db_password
            )
            cursor = conn.cursor()
            
            # Make sure we have the latest zones
            if not self.ob_zones:
                self._fetch_zones()
            
            # Set open_cursors parameter for all zones
            for zone in self.ob_zones:
                alter_cmd = f"ALTER SYSTEM SET open_cursors=65535 ZONE='{zone}'"
                logger.info(f"Executing: {alter_cmd}")
                cursor.execute(alter_cmd)
                conn.commit()
            
            # Create database if not exists
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.db_name}")
            conn.commit()
            
            logger.info(f"Database {self.db_name} created successfully on {node_type}")
            return True
        except Exception as e:
            logger.exception(f"Error creating database on port {port}: {str(e)}")
            return False
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()

    def _parse_sysbench_metrics(self, line: str) -> Optional[Dict]:
        """Parse sysbench output line for metrics"""
        try:
            # Example line: [ 10s ] thds: 4 tps: 142.91 qps: 2858.21 (r/w/o: 2000.74/571.64/285.82) lat (ms,95%): 45.79 err/s: 0.00 reconn/s: 0.00
            metrics = {}
            
            # Extract TPS
            tps_match = re.search(r'tps: (\d+\.\d+)', line)
            if tps_match:
                metrics['tps'] = float(tps_match.group(1))
            
            # Extract QPS
            qps_match = re.search(r'qps: (\d+\.\d+)', line)
            if qps_match:
                metrics['qps'] = float(qps_match.group(1))
            
            # Extract latency
            lat_match = re.search(r'lat \(ms,95%\): (\d+\.\d+)', line)
            if lat_match:
                metrics['latency_ms'] = float(lat_match.group(1))
            
            # Extract errors
            err_match = re.search(r'err/s: (\d+\.\d+)', line)
            if err_match:
                metrics['errors'] = int(float(err_match.group(1)))
            
            return metrics if metrics else None
        except Exception as e:
            logger.error(f"Error parsing sysbench metrics: {str(e)}")
            return None

    def _parse_tpcc_metrics(self, line: str) -> Optional[Dict[str, float]]:
        """Parse TPCC output line and extract metrics"""
        try:
            metrics = {}
            
            # Parse TpmC (transactions per minute) if present
            if 'TpmC' in line:
                match = re.search(r'TpmC: (\d+\.?\d*)', line)
                if match:
                    metrics['tpmC'] = float(match.group(1))
            
            # Parse individual transaction types
            for tx_type in ['NEW_ORDER', 'PAYMENT', 'DELIVERY', 'STOCK_LEVEL', 'ORDER_STATUS']:
                if tx_type in line:
                    # Look for count and latency
                    count_match = re.search(rf'{tx_type}\s+(\d+)', line)
                    latency_match = re.search(rf'{tx_type}.*?(\d+\.?\d*)\s*ms', line)
                    
                    if count_match:
                        metrics[f'{tx_type.lower()}_count'] = int(count_match.group(1))
                    if latency_match:
                        metrics[f'{tx_type.lower()}_latency'] = float(latency_match.group(1))
            
            return metrics if metrics else None
            
        except Exception as e:
            logger.warning(f"Failed to parse TPCC metrics from line: {line}", exc_info=True)
            return None

    def _parse_tpch_metrics(self, line: str) -> Optional[Dict]:
        """Parse TPCH output line for metrics"""
        try:
            metrics = {}
            # Example: Query 1 completed - Time: 10.5s
            query_match = re.search(r'Query (\d+) completed - Time: (\d+\.\d+)s', line)
            if query_match:
                query_num = int(query_match.group(1))
                execution_time = float(query_match.group(2))
                metrics[f'q{query_num}_time'] = execution_time
            return metrics if metrics else None
        except Exception as e:
            logger.error(f"Error parsing TPCH metrics: {str(e)}")
            return None

    def prepare_tpcc(self) -> bool:
        """Prepare the database for TPCC workload"""
        try:
            logger.info("Preparing database for TPCC workload")
            
            # Prepare TPC-C only on proxy node
            node_success = self._prepare_tpcc_on_node(self.db_port)
            
            if not node_success:
                logger.error(f"Failed to prepare TPC-C database on proxy node")
                return False
            
            logger.info("TPC-C database preparation completed successfully")
            return True
        except Exception as e:
            logger.exception("Error preparing TPCC database")
            return False
        
    def _prepare_tpcc_on_node(self, port: int) -> bool:
        """Prepare the TPCC database on a specific node"""
        try:
            node_type = "proxy node" if port == self.db_port else f"node with port {port}"
            logger.info(f"Preparing TPC-C database on {node_type}")
            
            # Create TPCC database if not exists
            conn = mysql.connector.connect(
                host=self.db_host,
                port=port,
                user=self.db_user,
                password=self.db_password
            )
            cursor = conn.cursor()
            cursor.execute(f"DROP DATABASE IF EXISTS {self.tpcc_db}")
            cursor.execute(f"CREATE DATABASE {self.tpcc_db}")
            conn.commit()
            cursor.close()
            conn.close()

            # Get the absolute path to the tpcc-mysql directory
            tpcc_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "tpcc-mysql")
            
            # Create tables
            create_tables_cmd = (
                f"mysql "
                f"-h {self.db_host} "
                f"-P {port} "
                f"-u {self.db_user} "
                f"-p{self.db_password} "
                f"{self.tpcc_db} "
                f"< {os.path.join(tpcc_dir, 'create_table.sql')}"
            )
            logger.info(f"Creating tables with command: {create_tables_cmd}")
            if os.system(create_tables_cmd) != 0:
                logger.error(f"Failed to create tables on {node_type}")
                return False

            # Load data
            load_data_cmd = (
                f"{os.path.join(tpcc_dir, 'tpcc_load')} "
                f"-h {self.db_host} "
                f"-P {port} "
                f"-d {self.tpcc_db} "
                f"-u {self.db_user} "
                f"-p {self.db_password} "
                f"-w 10"  # 10 warehouses
            )
            logger.info(f"Loading data with command: {load_data_cmd}")
            if os.system(load_data_cmd) != 0:
                logger.error(f"Failed to load data on {node_type}")
                return False

            # Add foreign keys and indexes
            add_indexes_cmd = (
                f"mysql "
                f"-h {self.db_host} "
                f"-P {port} "
                f"-u {self.db_user} "
                f"-p{self.db_password} "
                f"{self.tpcc_db} "
                f"< {os.path.join(tpcc_dir, 'add_fkey_idx.sql')}"
            )
            logger.info(f"Adding indexes with command: {add_indexes_cmd}")
            if os.system(add_indexes_cmd) != 0:
                logger.error(f"Failed to add indexes on {node_type}")
                return False

            logger.info(f"TPC-C database preparation completed successfully on {node_type}")
            return True
        except Exception as e:
            logger.exception(f"Error preparing TPCC database on {node_type}: {str(e)}")
            return False

    def prepare_tpch(self) -> bool:
        """Prepare the database for TPCH workload"""
        try:
            logger.info("Preparing database for TPCH workload")
            
            # Prepare TPC-H only on proxy node
            node_success = self._prepare_tpch_on_node(self.db_port)
            
            if not node_success:
                logger.error(f"Failed to prepare TPC-H database on proxy node")
                return False
            
            logger.info("TPC-H database preparation completed successfully")
            return True
        except Exception as e:
            logger.exception("Error preparing TPCH database")
            return False

    def _prepare_tpch_on_node(self, port: int) -> bool:
        """Prepare the TPCH database on a specific node"""
        try:
            node_type = "proxy node" if port == self.db_port else f"node with port {port}"
            logger.info(f"Preparing TPC-H database on {node_type}")
            
            # Create TPCH database if not exists
            conn = mysql.connector.connect(
                host=self.db_host,
                port=port,
                user=self.db_user,
                password=self.db_password
            )
            cursor = conn.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.tpch_db}")
            conn.commit()
            cursor.close()
            conn.close()

            # Run TPCH schema creation and data loading
            cmd = (
                f"tpch-mysql "
                f"--host={self.db_host} "
                f"--port={port} "
                f"--user={self.db_user} "
                f"--password={self.db_password} "
                f"--database={self.tpch_db} "
                f"--scale-factor=1 "  # 1GB default size
                "prepare"
            )
            process = subprocess.run(cmd.split(), capture_output=True, text=True)
            success = process.returncode == 0
            
            if success:
                logger.info(f"TPC-H database preparation completed successfully on {node_type}")
            else:
                logger.error(f"TPC-H database preparation failed on {node_type}")
            
            return success
        except Exception as e:
            logger.exception(f"Error preparing TPCH database on {node_type}: {str(e)}")
            return False

    def _read_workload_output(self, workload_id: str, process: subprocess.Popen):
        """Read and store workload output"""
        try:
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    if workload_id not in self.workload_outputs:
                        self.workload_outputs[workload_id] = []
                    self.workload_outputs[workload_id].append(line.strip())
                    logger.debug(f"Workload {workload_id} output: {line.strip()}")
        except Exception as e:
            logger.error(f"Error reading workload output: {str(e)}")

    def start_workload(self, workload_type: WorkloadType, num_threads: int = 4, options: Dict = None, task_name: str = None) -> bool:
        """Start a new workload with custom options"""
        try:
            workload_id = f"{workload_type.value}_{num_threads}"
            
            if workload_id in self.active_workloads:
                logger.warning(f"Workload {workload_id} is already running")
                return False
            
            logger.info(f"Starting workload: type={workload_type.value}, threads={num_threads}, options={options}, task_name={task_name}")
            
            # Create task if name is provided
            task = None
            if task_name:
                logger.info(f"Creating task with name: {task_name}")
                # Create a proper WorkloadConfig object instead of just a dictionary
                workload_config = {
                    "workload_type": workload_type,
                    "num_threads": num_threads,
                    "options": options
                }
                task = self.create_task(
                    name=task_name,
                    workload_type=workload_type,
                    workload_config=workload_config,
                    anomalies=[]  # Anomalies will be added later if needed
                )
                logger.info(f"Task created with ID: {task.id}")
            else:
                logger.warning("No task_name provided, skipping task creation")
            
            # Always use the default database connection parameters from __init__
            logger.info("Using default database connection from obproxy")
            success = self._start_workload_on_node(
                workload_type, 
                num_threads, 
                options, 
                task, 
                workload_id,
                self.db_host, 
                self.db_port
            )
            
            if success and task:
                self.update_task_status(task.id, WorkloadStatus.RUNNING)
            
            return success
            
        except Exception as e:
            logger.exception(f"Error starting workload: {str(e)}")
            if task:
                self.update_task_status(task.id, WorkloadStatus.ERROR, str(e))
            return False
    
    def _start_workload_on_node(self, workload_type: WorkloadType, num_threads: int, options: Dict, task, workload_id: str, host: str, port: int) -> bool:
        """Start a workload on a specific node"""
        try:
            cmd = None
            if workload_type == WorkloadType.SYSBENCH:
                # Build sysbench command with custom options
                cmd = (
                    f"sysbench oltp_read_write "
                    f"--db-driver=mysql "
                    f"--mysql-host={host} "
                    f"--mysql-port={port} "
                    f"--mysql-user={self.db_user} "
                    f"--mysql-password={self.db_password} "
                    f"--mysql-db={self.db_name} "
                    f"--tables={options.get('tables', 10)} "
                    f"--table-size={options.get('tableSize', 100000)} "
                    f"--threads={num_threads} "
                    f"--time=0 "
                    f"--report-interval={options.get('reportInterval', 10)} "
                    f"--rand-type={options.get('randType', 'uniform')} "
                    "run"
                )
            elif workload_type == WorkloadType.TPCC:
                # Get the absolute path to the tpcc-mysql directory
                tpcc_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "tpcc-mysql")
                
                # Validate TPCC directory exists
                if not os.path.exists(tpcc_dir):
                    logger.error(f"TPCC directory not found: {tpcc_dir}")
                    return False
                
                # Validate tpcc_start script exists and is executable
                tpcc_script = os.path.join(tpcc_dir, 'tpcc_start')
                if not os.path.exists(tpcc_script):
                    logger.error(f"TPCC start script not found: {tpcc_script}")
                    return False
                if not os.access(tpcc_script, os.X_OK):
                    logger.error(f"TPCC start script is not executable: {tpcc_script}")
                    return False
                
                # Check if TPCC database exists
                try:
                    with pymysql.connect(
                        host=host,
                        port=port,
                        user=self.db_user,
                        password=self.db_password
                    ) as conn:
                        with conn.cursor() as cursor:
                            cursor.execute("SHOW DATABASES LIKE %s", (self.tpcc_db,))
                            if not cursor.fetchone():
                                logger.error(f"TPCC database {self.tpcc_db} does not exist")
                                return False
                except Exception as e:
                    logger.error(f"Failed to check TPCC database on {host}:{port}: {str(e)}")
                    return False
                
                # Build TPCC command with custom options
                cmd = (
                    f"{tpcc_script} "
                    f"-h {host} "
                    f"-P {port} "
                    f"-d {self.tpcc_db} "
                    f"-u {self.db_user} "
                    f"-p {self.db_password} "
                    f"-w {options.get('warehouses', 10)} "  # Number of warehouses
                    f"-c {num_threads} "  # Number of connections
                    f"-r {options.get('warmupTime', 10)} "  # Warmup time in seconds
                    f"-l {options.get('runningTime', 60) * 60} "  # Run time in seconds
                    f"-i {options.get('reportInterval', 10)} "  # Report interval
                    "run"
                )
            elif workload_type == WorkloadType.TPCH:
                cmd = (
                    f"tpch-mysql "
                    f"--host={host} "
                    f"--port={port} "
                    f"--user={self.db_user} "
                    f"--password={self.db_password} "
                    f"--database={self.tpch_db} "
                    f"--scale-factor=1 "
                    f"--threads={num_threads} "
                    f"--time=0 "
                    f"--report-interval={options.get('reportInterval', 10)} "
                    "run"
                )
            
            if not cmd:
                logger.error(f"Unsupported workload type: {workload_type}")
                return False

            logger.debug(f"Executing command: {cmd}")
            
            process = subprocess.Popen(
                cmd.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
                universal_newlines=True,
                cwd=os.path.dirname(cmd.split()[0]) if workload_type == WorkloadType.TPCC else None
            )
            
            # Wait briefly to check if the process starts successfully
            time.sleep(1)
            if process.poll() is not None:
                _, stderr = process.communicate()
                logger.error(f"Workload process failed to start. Return code: {process.returncode}")
                if stderr:
                    logger.error(f"Stderr output: {stderr}")
                return False
            
            self.active_workloads[workload_id] = process
            
            # Start a thread to read the output
            output_thread = threading.Thread(
                target=self._read_workload_output,
                args=(workload_id, process)
            )
            output_thread.daemon = True
            output_thread.start()
            
            return True
        except Exception as e:
            logger.exception(f"Error starting workload on specific node: {str(e)}")
            return False

    def stop_workload(self, workload_id: str) -> bool:
        """Stop a running workload by ID"""
        try:
            # First check for direct match
            if workload_id in self.active_workloads:
                process = self.active_workloads[workload_id]
                process.terminate()
                process.wait(timeout=5)
                
                del self.active_workloads[workload_id]
                if workload_id in self.workload_outputs:
                    del self.workload_outputs[workload_id]
                
                # Try to find the associated task and update its status
                associated_task = None
                for task in self.get_active_tasks():
                    if task.workload_id == workload_id:
                        associated_task = task
                        break
                
                # Update task status if workload was successfully stopped
                if associated_task:
                    self.update_task_status(associated_task.id, WorkloadStatus.STOPPED)
                
                return True
            
            # If not found directly, check for node-specific workloads
            # Format: original_workload_id_nodename
            matching_workloads = [wid for wid in self.active_workloads.keys() if wid.startswith(f"{workload_id}_")]
            
            if not matching_workloads:
                logger.warning(f"Workload {workload_id} is not running")
                return False
            
            # Stop all matching workloads
            for wid in matching_workloads:
                process = self.active_workloads[wid]
                process.terminate()
                process.wait(timeout=5)
                
                del self.active_workloads[wid]
                if wid in self.workload_outputs:
                    del self.workload_outputs[wid]
            
            # Update associated task status
            for task in self.tasks.values():
                if task.workload_id == workload_id:
                    self.update_task_status(task.id, WorkloadStatus.STOPPED)
                    break
            
            return True
            
        except Exception as e:
            logger.exception(f"Error stopping workload: {str(e)}")
            return False

    def get_workload_status(self, workload_id: str) -> Dict:
        """Get the status of a workload"""
        try:
            # Direct match
            if workload_id in self.active_workloads:
                process = self.active_workloads[workload_id]
                is_running = process.poll() is None
                
                return {
                    "status": "running" if is_running else "stopped",
                    "output": self.workload_outputs.get(workload_id, [])
                }
            
            # Check for node-specific workloads
            # Format: original_workload_id_nodename
            matching_workloads = [wid for wid in self.active_workloads.keys() if wid.startswith(f"{workload_id}_")]
            
            if matching_workloads:
                # Combine output from all matching workloads
                combined_output = []
                all_running = True
                
                for wid in matching_workloads:
                    process = self.active_workloads[wid]
                    is_running = process.poll() is None
                    all_running = all_running and is_running
                    combined_output.extend(self.workload_outputs.get(wid, []))
                
                return {
                    "status": "running" if all_running else "partially_running",
                    "output": combined_output
                }
            
            return {
                "status": "not_found",
                "output": []
            }
            
        except Exception as e:
            logger.exception(f"Error getting workload status: {str(e)}")
            return {
                "status": "error",
                "output": []
            }

    def _fetch_available_nodes(self) -> List[str]:
        """Get list of available nodes for running workloads"""
        try:
            # First try to use the kubernetes API directly
            try:
                # Load kubernetes configuration
                if os.getenv('KUBERNETES_SERVICE_HOST'):
                    config.load_incluster_config()
                else:
                    config.load_kube_config()
                
                # Create API client
                core_api = client.CoreV1Api()
                namespace = os.getenv('OCEANBASE_NAMESPACE', 'oceanbase')
                
                # List pods
                pods = core_api.list_namespaced_pod(
                    namespace=namespace, 
                    label_selector="ref-obcluster"
                )
                
                # Extract pod names
                pod_names = []
                for pod in pods.items:
                    if pod.metadata.name.startswith("obcluster-"):
                        pod_names.append(pod.metadata.name)
                
                if pod_names:
                    logger.info(f"Found {len(pod_names)} pods using direct Kubernetes API")
                    return pod_names
            except Exception as direct_api_error:
                logger.warning(f"Failed to get nodes using direct Kubernetes API: {str(direct_api_error)}")
                
            # If direct API access failed, try using k8s_service as a fallback
            from backend.app.services.k8s_service import k8s_service
            try:
                # Try to use a synchronous method from k8s_service if available
                if hasattr(k8s_service, '_fetch_available_nodes_sync'):
                    nodes = k8s_service._fetch_available_nodes_sync()
                    if nodes:
                        logger.info(f"Found {len(nodes)} nodes using k8s_service sync method")
                        return nodes
            except Exception as k8s_service_error:
                logger.warning(f"Failed to get nodes using k8s_service sync method: {str(k8s_service_error)}")
                
            # Fallback to environment variable
            nodes_str = os.getenv('AVAILABLE_NODES', 'node1,node2,node3')
            nodes = [node.strip() for node in nodes_str.split(',')]
            logger.info(f"Using default nodes from environment: {nodes}")
            return nodes
        except Exception as e:
            logger.error(f"Error getting available nodes: {str(e)}")
            nodes_str = os.getenv('AVAILABLE_NODES', 'node1,node2,node3')
            return [node.strip() for node in nodes_str.split(',')]

    def get_available_nodes(self) -> List[str]:
        """Get list of available nodes for running workloads"""
        if not self.available_nodes:
            self._fetch_available_nodes()
        return self.available_nodes

    def get_active_workloads(self) -> List[Dict]:
        """Get list of active workloads with system metrics"""
        try:
            # Clean up finished processes
            finished_workloads = [
                wid for wid, proc in self.active_workloads.items()
                if proc.poll() is not None
            ]
            for wid in finished_workloads:
                logger.info(f"Removing finished workload: {wid}")
                del self.active_workloads[wid]
                
                # Update associated task status
                for task in self.tasks.values():
                    if task.workload_id == wid:
                        self.update_task_status(task.id, WorkloadStatus.STOPPED)
                        break

            active_workloads = []
            for workload_id, process in self.active_workloads.items():
                # Parse node from workload_id if present
                # Format could be either "type_threads" or "type_threads_nodename"
                parts = workload_id.split('_')
                if len(parts) >= 3:  # Has node info
                    workload_type = parts[0]
                    threads = parts[1]
                    node = '_'.join(parts[2:])  # Rejoin in case node name has underscores
                else:
                    workload_type = parts[0]
                    threads = parts[1]
                    node = None
                
                # Find associated task
                task = next((t for t in self.tasks.values() if t.workload_id == workload_id), None)
                
                workload_info = {
                    'id': workload_id,
                    'type': workload_type,
                    'threads': int(threads),
                    'pid': process.pid,
                    'start_time': task.start_time if task else datetime.utcnow(),
                    'status': task.status if task else WorkloadStatus.RUNNING
                }
                
                # Add node information if available
                if node:
                    workload_info['node'] = node
                
                active_workloads.append(workload_info)
            
            logger.debug(f"Active workloads: {active_workloads}")
            return active_workloads
            
        except Exception as e:
            logger.exception("Error getting active workloads")
            return []

    def get_system_metrics(self) -> Dict:
        """Get current system metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            current_time = time.time()
            
            if self._last_io_stats and self._last_io_time:
                time_delta = current_time - self._last_io_time
                read_bytes_delta = disk_io.read_bytes - self._last_io_stats.read_bytes
                write_bytes_delta = disk_io.write_bytes - self._last_io_stats.write_bytes
                
                # Calculate I/O rate in MB/s
                io_rate = (read_bytes_delta + write_bytes_delta) / (1024 * 1024 * time_delta)
                
                # Convert to percentage (assuming 100MB/s as 100%)
                MAX_IO_RATE = 100  # MB/s
                io_percent = min(100.0, (io_rate / MAX_IO_RATE) * 100)
            else:
                io_percent = 0.0
            
            self._last_io_stats = disk_io
            self._last_io_time = current_time
            
            metrics = {
                'cpu_usage': round(cpu_percent, 1),
                'memory_usage': round(memory_percent, 1),
                'disk_usage': round(io_percent, 1)
            }
            
            logger.debug(f"System metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.exception("Error getting system metrics")
            return {
                'cpu_usage': 0.0,
                'memory_usage': 0.0,
                'disk_usage': 0.0
            }
            
    def stop_all_workloads(self) -> bool:
        """Stop all active workloads"""
        try:
            if not self.active_workloads:
                logger.info("No active workloads to stop")
                return True
            
            logger.info(f"Stopping all {len(self.active_workloads)} active workloads")
            success = True
            
            # Make a copy of keys since we'll be modifying the dictionary
            workload_ids = list(self.active_workloads.keys())
            
            for workload_id in workload_ids:
                try:
                    process = self.active_workloads[workload_id]
                    process.terminate()
                    process.wait(timeout=5)
                    
                    del self.active_workloads[workload_id]
                    if workload_id in self.workload_outputs:
                        del self.workload_outputs[workload_id]
                        
                    # Update associated task status
                    for task in self.tasks.values():
                        if task.workload_id == workload_id:
                            self.update_task_status(task.id, WorkloadStatus.STOPPED)
                            break
                        
                    logger.info(f"Stopped workload: {workload_id}")
                except Exception as e:
                    logger.error(f"Failed to stop workload {workload_id}: {str(e)}")
                    success = False
                
            # Update status of all active tasks to STOPPED
            for task in self.get_active_tasks():
                self.update_task_status(task.id, WorkloadStatus.STOPPED)
            
            return success
        except Exception as e:
            logger.exception("Error stopping all workloads")
            return False

# Singleton instance of WorkloadService
workload_service = WorkloadService()