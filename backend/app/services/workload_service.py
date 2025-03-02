import logging
import psutil
import subprocess
import time
import os
import mysql.connector
import threading
import re
from typing import Dict, List, Optional
from ..schemas.workload import WorkloadType, WorkloadMetrics
import pymysql

logger = logging.getLogger(__name__)

class WorkloadService:
    def __init__(self):
        self.active_workloads: Dict[str, subprocess.Popen] = {}
        self.workload_metrics: Dict[str, Dict] = {}
        self._last_io_stats = None
        self._last_io_time = None
        # Database connection parameters
        self.db_host = os.getenv('OB_HOST', '127.0.0.1')
        self.db_port = int(os.getenv('OB_PORT', '22883'))
        self.db_user = os.getenv('OB_USER', 'root@sys')
        self.db_password = os.getenv('OB_PASSWORD', 'root_password')
        self.db_name = 'sbtest'
        self.tpcc_db = 'tpcc'
        self.tpch_db = 'tpch'
        logger.info("WorkloadService initialized")
        logger.info(f"Database connection: host={self.db_host}, port={self.db_port}, user={self.db_user}")

    def _create_database(self):
        """Create the sysbench database if it doesn't exist"""
        try:
            logger.info(f"Attempting to create database {self.db_name}")
            conn = mysql.connector.connect(
                host=self.db_host,
                port=self.db_port,
                user=self.db_user,
                password=self.db_password
            )
            cursor = conn.cursor()
            
            # Set open_cursors parameter for all zones
            for zone in ['zone1', 'zone2', 'zone3']:
                alter_cmd = f"ALTER SYSTEM SET open_cursors=65535 ZONE='{zone}'"
                logger.info(f"Executing: {alter_cmd}")
                cursor.execute(alter_cmd)
                conn.commit()
            
            # Create database if not exists
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.db_name}")
            conn.commit()
            
            logger.info(f"Database {self.db_name} created successfully")
            return True
        except Exception as e:
            logger.exception(f"Error creating database: {str(e)}")
            return False
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()

    def _check_tables_exist(self) -> bool:
        """Check if sysbench tables already exist"""
        try:
            conn = mysql.connector.connect(
                host=self.db_host,
                port=self.db_port,
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
            logger.exception(f"Error checking tables: {str(e)}")
            return False
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()

    def prepare_database(self):
        """Prepare the database for sysbench workload"""
        try:
            logger.info("Preparing database for sysbench workload")
            
            # First create the database
            if not self._create_database():
                logger.error("Failed to create database")
                return False
            
            # Check if tables already exist
            if self._check_tables_exist():
                logger.info("Sysbench tables already exist, skipping preparation")
                return True
            
            # Build sysbench prepare command with database connection parameters
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
                # Try to get more information about the failure
                try:
                    conn = mysql.connector.connect(
                        host=self.db_host,
                        port=self.db_port,
                        user=self.db_user,
                        password=self.db_password,
                        database=self.db_name,
                    )
                    logger.info("Database connection test successful")
                    conn.close()
                except Exception as e:
                    logger.error(f"Database connection test failed: {str(e)}")
            
            return process.returncode == 0
        except Exception as e:
            logger.exception("Error preparing database")
            return False

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
            # Create TPCC database if not exists
            conn = mysql.connector.connect(
                host=self.db_host,
                port=self.db_port,
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
                f"-P {self.db_port} "
                f"-u {self.db_user} "
                f"-p{self.db_password} "
                f"{self.tpcc_db} "
                f"< {os.path.join(tpcc_dir, 'create_table.sql')}"
            )
            logger.info(f"Creating tables with command: {create_tables_cmd}")
            if os.system(create_tables_cmd) != 0:
                logger.error("Failed to create tables")
                return False

            # Load data
            load_data_cmd = (
                f"{os.path.join(tpcc_dir, 'tpcc_load')} "
                f"-h {self.db_host} "
                f"-P {self.db_port} "
                f"-d {self.tpcc_db} "
                f"-u {self.db_user} "
                f"-p {self.db_password} "
                f"-w 10"  # 10 warehouses
            )
            logger.info(f"Loading data with command: {load_data_cmd}")
            if os.system(load_data_cmd) != 0:
                logger.error("Failed to load data")
                return False

            # Add foreign keys and indexes
            add_indexes_cmd = (
                f"mysql "
                f"-h {self.db_host} "
                f"-P {self.db_port} "
                f"-u {self.db_user} "
                f"-p{self.db_password} "
                f"{self.tpcc_db} "
                f"< {os.path.join(tpcc_dir, 'add_fkey_idx.sql')}"
            )
            logger.info(f"Adding indexes with command: {add_indexes_cmd}")
            if os.system(add_indexes_cmd) != 0:
                logger.error("Failed to add indexes")
                return False

            logger.info("TPC-C database preparation completed successfully")
            return True
        except Exception as e:
            logger.exception("Error preparing TPCC database")
            return False

    def prepare_tpch(self) -> bool:
        """Prepare the database for TPCH workload"""
        try:
            logger.info("Preparing database for TPCH workload")
            # Create TPCH database if not exists
            conn = mysql.connector.connect(
                host=self.db_host,
                port=self.db_port,
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
                f"--port={self.db_port} "
                f"--user={self.db_user} "
                f"--password={self.db_password} "
                f"--database={self.tpch_db} "
                f"--scale-factor=1 "  # 1GB default size
                "prepare"
            )
            process = subprocess.run(cmd.split(), capture_output=True, text=True)
            return process.returncode == 0
        except Exception as e:
            logger.exception("Error preparing TPCH database")
            return False

    def _read_workload_output(self, workload_id: str, process: subprocess.Popen):
        """Read workload output in a separate thread and update metrics"""
        try:
            start_time = time.time()
            last_output_time = start_time
            metrics_received = False
            
            while process.poll() is None:
                # Check if we haven't received output for too long
                if time.time() - last_output_time > 30:  # 30 seconds timeout
                    logger.warning(f"Workload {workload_id} has not produced output for 30 seconds")
                    if not metrics_received:
                        logger.error(f"Workload {workload_id} never produced any metrics - may have failed to start properly")
                        process.terminate()
                        break
                
                line = process.stdout.readline()
                if not line:
                    continue
                    
                line = line.strip()
                if not line:
                    continue
                
                last_output_time = time.time()
                logger.debug(f"Workload {workload_id} output: {line}")
                
                metrics = None
                if 'sysbench' in workload_id and '[ ' in line and ' ]' in line and 'tps:' in line:
                    metrics = self._parse_sysbench_metrics(line)
                elif 'tpcc' in workload_id:
                    # More lenient TPCC parsing - look for any performance metrics
                    if any(x in line for x in ['TpmC', 'NEW_ORDER', 'PAYMENT', 'DELIVERY']):
                        metrics = self._parse_tpcc_metrics(line)
                elif 'tpch' in workload_id and 'Query' in line and 'completed' in line:
                    metrics = self._parse_tpch_metrics(line)

                if metrics:
                    metrics_received = True
                    if workload_id not in self.workload_metrics:
                        self.workload_metrics[workload_id] = {}
                    self.workload_metrics[workload_id].update(metrics)
                    logger.debug(f"Updated metrics for workload {workload_id}: {metrics}")
            
            # Process has ended - get the return code and stderr
            return_code = process.poll()
            _, stderr = process.communicate()
            runtime = time.time() - start_time
            
            if return_code != 0:
                logger.error(f"Workload {workload_id} failed with return code {return_code}")
                if stderr:
                    logger.error(f"Stderr output: {stderr}")
            else:
                logger.info(f"Workload {workload_id} completed successfully after running for {runtime:.1f} seconds")
            
        except Exception as e:
            logger.exception(f"Error reading workload {workload_id} output")
        finally:
            # Clean up the process if it's still running
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
            
            if workload_id in self.workload_metrics:
                del self.workload_metrics[workload_id]
            if workload_id in self.active_workloads:
                del self.active_workloads[workload_id]
            
            logger.info(f"Workload {workload_id} output reader thread finished")

    def start_workload(self, workload_type: WorkloadType, num_threads: int = 4) -> bool:
        """Start a new workload"""
        try:
            workload_id = f"{workload_type.value}_{num_threads}"
            
            if workload_id in self.active_workloads:
                logger.warning(f"Workload {workload_id} is already running")
                return False
            
            logger.info(f"Starting workload: type={workload_type.value}, threads={num_threads}")
            
            cmd = None
            if workload_type == WorkloadType.SYSBENCH:
                cmd = (
                    f"sysbench oltp_read_write "
                    f"--db-driver=mysql "
                    f"--mysql-host={self.db_host} "
                    f"--mysql-port={self.db_port} "
                    f"--mysql-user={self.db_user} "
                    f"--mysql-password={self.db_password} "
                    f"--mysql-db={self.db_name} "
                    f"--tables=10 "
                    f"--table-size=100000 "
                    f"--threads={num_threads} "
                    f"--time=0 "
                    f"--report-interval=10 "
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
                        host=self.db_host,
                        port=self.db_port,
                        user=self.db_user,
                        password=self.db_password
                    ) as conn:
                        with conn.cursor() as cursor:
                            cursor.execute("SHOW DATABASES LIKE %s", (self.tpcc_db,))
                            if not cursor.fetchone():
                                logger.error(f"TPCC database {self.tpcc_db} does not exist")
                                return False
                except Exception as e:
                    logger.error(f"Failed to check TPCC database: {str(e)}")
                    return False
                
                cmd = (
                    f"{tpcc_script} "
                    f"-h {self.db_host} "
                    f"-P {self.db_port} "
                    f"-d {self.tpcc_db} "
                    f"-u {self.db_user} "
                    f"-p {self.db_password} "
                    f"-w 10 "  # 10 warehouses
                    f"-c {num_threads} "  # Number of connections
                    f"-r 10 "  # Warmup time in seconds
                    f"-l {12*60*60}"  # Run time in seconds, 0 means infinite
                    "run"
                )
            elif workload_type == WorkloadType.TPCH:
                cmd = (
                    f"tpch-mysql "
                    f"--host={self.db_host} "
                    f"--port={self.db_port} "
                    f"--user={self.db_user} "
                    f"--password={self.db_password} "
                    f"--database={self.tpch_db} "
                    f"--scale-factor=1 "
                    f"--threads={num_threads} "
                    f"--time=0 "
                    f"--report-interval=10 "
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
            logger.exception(f"Error starting workload: {str(e)}")
            return False

    def stop_workload(self, workload_id: str) -> bool:
        """Stop a specific workload"""
        try:
            if workload_id not in self.active_workloads:
                logger.warning(f"Workload {workload_id} not found in active workloads")
                # It might have already been stopped by the reader thread, so return True
                return True
            
            logger.info(f"Stopping workload: {workload_id}")
            process = self.active_workloads[workload_id]
            
            if process.poll() is None:  # Process is still running
                process.terminate()
                try:
                    process.wait(timeout=5)  # Wait up to 5 seconds for graceful termination
                except subprocess.TimeoutExpired:
                    logger.warning(f"Workload {workload_id} did not terminate gracefully, forcing kill")
                    process.kill()
            
            # Read any remaining output
            stdout, stderr = process.communicate()
            if stdout:
                logger.info(f"Final output from workload {workload_id}: {stdout}")
            if stderr:
                logger.warning(f"Final error output from workload {workload_id}: {stderr}")
            
            # Check if the workload is still in active_workloads before removing it
            # It might have been removed by the reader thread
            if workload_id in self.active_workloads:
                del self.active_workloads[workload_id]
            
            logger.info(f"Workload {workload_id} stopped successfully")
            return True
            
        except Exception as e:
            logger.exception(f"Error stopping workload {workload_id}")
            return False

    def stop_all_workloads(self) -> bool:
        """Stop all active workloads"""
        try:
            logger.info("Stopping all workloads")
            # Make a copy of the workload IDs to avoid dictionary changed during iteration
            workload_ids = list(self.active_workloads.keys())
            
            if not workload_ids:
                logger.info("No active workloads to stop")
                return True
                
            # Try to stop each workload
            stop_errors = 0
            for workload_id in workload_ids:
                try:
                    self.stop_workload(workload_id)
                except Exception as e:
                    logger.error(f"Error stopping workload {workload_id}: {str(e)}")
                    stop_errors += 1
            
            # After attempting to stop all workloads, check if any are still active
            remaining_workloads = list(self.active_workloads.keys())
            if not remaining_workloads:
                logger.info("All workloads stopped successfully")
                return True
            else:
                logger.warning(f"Failed to stop {len(remaining_workloads)} workloads: {remaining_workloads}")
                return False
                    
        except Exception as e:
            logger.exception("Error stopping all workloads")
            return False

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
                if wid in self.workload_metrics:
                    del self.workload_metrics[wid]

            system_metrics = self.get_system_metrics()
            
            active_workloads = []
            for workload_id, process in self.active_workloads.items():
                workload_type, threads = workload_id.split('_')
                
                # Combine system metrics with workload-specific metrics
                metrics = {**system_metrics}
                if workload_id in self.workload_metrics:
                    metrics.update(self.workload_metrics[workload_id])
                
                active_workloads.append({
                    'id': workload_id,
                    'type': workload_type,
                    'threads': int(threads),
                    'pid': process.pid,
                    'metrics': metrics
                })
            
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