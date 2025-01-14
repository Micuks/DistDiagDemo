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

logger = logging.getLogger(__name__)

class WorkloadService:
    def __init__(self):
        self.active_workloads: Dict[str, subprocess.Popen] = {}
        self.workload_metrics: Dict[str, Dict] = {}  # Store workload-specific metrics
        self._last_io_stats = None
        self._last_io_time = None
        # Database connection parameters
        self.db_host = os.getenv('OB_HOST', '127.0.0.1')
        self.db_port = int(os.getenv('OB_PORT', '22883'))
        self.db_user = os.getenv('OB_USER', 'root@sys')
        self.db_password = os.getenv('OB_PASSWORD', 'root_password')
        self.db_name = 'sbtest'
        logger.info("WorkloadService initialized")
        logger.info(f"Database connection: host={self.db_host}, port={self.db_port}, user={self.db_user}, db={self.db_name}")

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
                f"--threads=1 "
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

    def _read_workload_output(self, workload_id: str, process: subprocess.Popen):
        """Read workload output in a separate thread and update metrics"""
        try:
            while process.poll() is None:  # While process is running
                line = process.stdout.readline()
                if not line:
                    break
                    
                line = line.strip()
                if not line:
                    continue
                
                # Log the output for debugging
                logger.debug(f"Workload {workload_id} output: {line}")
                
                # Parse metrics if this is a metrics line
                if '[ ' in line and ' ]' in line and 'tps:' in line:
                    metrics = self._parse_sysbench_metrics(line)
                    if metrics:
                        # Update workload metrics
                        if workload_id not in self.workload_metrics:
                            self.workload_metrics[workload_id] = {}
                        self.workload_metrics[workload_id].update(metrics)
                        logger.debug(f"Updated metrics for workload {workload_id}: {metrics}")
            
            # Process has finished
            logger.info(f"Workload {workload_id} output reader thread finished")
            
        except Exception as e:
            logger.exception(f"Error reading workload {workload_id} output")
        finally:
            # Clean up metrics when the workload is done
            if workload_id in self.workload_metrics:
                del self.workload_metrics[workload_id]

    def start_workload(self, workload_type: WorkloadType, num_threads: int = 1) -> bool:
        """Start a new workload"""
        try:
            workload_id = f"{workload_type.value}_{num_threads}"
            
            if workload_id in self.active_workloads:
                logger.warning(f"Workload {workload_id} is already running")
                return False
            
            logger.info(f"Starting workload: type={workload_type.value}, threads={num_threads}")
            
            if workload_type == WorkloadType.SYSBENCH:
                # Build sysbench run command with database connection parameters
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
                    f"--time=0 "  # Run indefinitely
                    f"--report-interval=10 "  # Report every 10 seconds
                    "run"
                )
                
                logger.debug(f"Executing command: {cmd}")
                
                process = subprocess.Popen(
                    cmd.split(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=1,
                    universal_newlines=True
                )
                
                self.active_workloads[workload_id] = process
                
                # Start a thread to read the process output
                output_thread = threading.Thread(
                    target=self._read_workload_output,
                    args=(workload_id, process),
                    daemon=True
                )
                output_thread.start()
                
                logger.info(f"Workload {workload_id} started successfully with PID {process.pid}")
                return True
            else:
                logger.error(f"Unsupported workload type: {workload_type}")
                return False

        except Exception as e:
            logger.exception(f"Error starting workload {workload_type}")
            return False

    def stop_workload(self, workload_id: str) -> bool:
        """Stop a specific workload"""
        try:
            if workload_id not in self.active_workloads:
                logger.warning(f"Workload {workload_id} not found in active workloads")
                return False
            
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
            workload_ids = list(self.active_workloads.keys())
            
            success = True
            for workload_id in workload_ids:
                if not self.stop_workload(workload_id):
                    success = False
                    
            if success:
                logger.info("All workloads stopped successfully")
            else:
                logger.warning("Some workloads failed to stop")
                
            return success
            
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