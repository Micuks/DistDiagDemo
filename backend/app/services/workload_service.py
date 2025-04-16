import logging
import psutil
import subprocess
import time
import os
import mysql.connector
import threading
import re
import redis
import json
from typing import Dict, List, Optional, Tuple
from ..schemas.tasks import WorkloadType
import pymysql
from enum import Enum
from dotenv import load_dotenv
from datetime import datetime, timezone
import uuid

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env'))

logger = logging.getLogger(__name__)

# Define keys for Redis
WORKLOAD_PROCESS_KEY_PREFIX = "workload_process:"
ACTIVE_WORKLOAD_RUNS_SET_KEY = "active_workload_runs"

class WorkloadService:
    def __init__(self):
        self.active_procs: Dict[str, subprocess.Popen] = {}
        self._last_io_stats = None
        self._last_io_time = None
        self.db_host = os.getenv('OB_HOST', 'localhost')
        self.db_port = int(os.getenv('OB_PORT', '3306'))
        self.db_user = os.getenv('OB_USER', 'root')
        self.db_password = os.getenv('OB_PASSWORD', 'root_password')
        self.db_name = os.getenv('OB_NAME', 'sbtest')
        self.tpcc_db = os.getenv('TPCC_DB', 'tpcc')
        self.tpch_db = os.getenv('TPCH_DB', 'tpch')
        self.redis_host = os.getenv('REDIS_HOST', 'localhost')
        self.redis_port = int(os.getenv('REDIS_PORT', 6379))
        self.redis_db = int(os.getenv('REDIS_DB_WORKLOADS', 2))
        self.redis_client = None
        self._connect_redis()
        self.ob_zones = []
        self._fetch_zones()
        logger.info("WorkloadService initialized")
        if self.redis_client:
            logger.info(f"WorkloadService connected to Redis DB {self.redis_db}. Checking for orphaned processes...")
        else:
            logger.error("WorkloadService failed to connect to Redis. Process persistence/cleanup disabled.")

    def _connect_redis(self):
        """Connects to the Redis instance for workload process tracking."""
        try:
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                decode_responses=True
            )
            self.redis_client.ping() # Test connection
        except redis.exceptions.ConnectionError as e:
            logger.error(f"Failed to connect to Redis for WorkloadService (DB {self.redis_db}): {e}")
            self.redis_client = None

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

    async def start_workload(
        self,
        workload_type: WorkloadType,
        run_id: str,
        config: Dict,
        num_threads: int = 4
    ) -> bool:
        """Starts a workload process, identified by run_id, and tracks it in Redis."""
        if not self.redis_client:
            logger.error("Redis client not available. Cannot start workload.")
            return False

        # Check if run_id already exists in Redis (should not happen ideally)
        if self.redis_client.exists(f"{WORKLOAD_PROCESS_KEY_PREFIX}{run_id}"):
            logger.warning(f"Workload run_id {run_id} already exists in Redis. Aborting start.")
            return False

        # Check if run_id is already in active_procs (in-memory check)
        if run_id in self.active_procs:
            logger.warning(f"Workload run_id {run_id} is already present in active_procs. Aborting start.")
            return False

        # Extract necessary config (adjust based on actual config structure)
        # Ensure num_threads is taken from config if present
        threads = config.get('num_threads', config.get('threads', num_threads))
        options = config

        logger.info(f"Attempting to start workload: run_id={run_id}, type={workload_type.value}, threads={threads}, options={options}")

        # Always use the default database connection parameters from __init__
        logger.info(f"Using database connection: host={self.db_host}, port={self.db_port}")
        host = self.db_host
        port = self.db_port

        cmd_str = None
        cmd_list = []
        cwd = None

        try:
            if workload_type == WorkloadType.SYSBENCH:
                # Build sysbench command with custom options from config
                cmd_list = [
                    "sysbench", "oltp_read_write",
                    "--db-driver=mysql",
                    f"--mysql-host={host}",
                    f"--mysql-port={port}",
                    f"--mysql-user={self.db_user}",
                    f"--mysql-password={self.db_password}",
                    f"--mysql-db={self.db_name}",
                    f"--tables={options.get('tables', 10)}",
                    f"--table-size={options.get('tableSize', 100000)}",
                    f"--threads={threads}",
                    f"--time={options.get('time', 0)}",
                    f"--report-interval={options.get('reportInterval', 10)}",
                    f"--rand-type={options.get('randType', 'uniform')}",
                    "run"
                ]
                cmd_str = " ".join(map(str, cmd_list))
            elif workload_type == WorkloadType.TPCC:
                # Get the absolute path to the tpcc-mysql directory
                tpcc_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "tpcc-mysql")
                tpcc_script = os.path.join(tpcc_dir, 'tpcc_start')

                if not os.path.exists(tpcc_script) or not os.access(tpcc_script, os.X_OK):
                    logger.error(f"TPCC start script not found or not executable: {tpcc_script}")
                    return False
                # Check if TPCC database exists (optional, might rely on prepare step)
                # ... (database check logic could be kept or removed) ...

                cmd_list = [
                    tpcc_script,
                    f"-h", host,
                    f"-P", str(port),
                    f"-d", self.tpcc_db,
                    f"-u", self.db_user,
                    f"-p", self.db_password,
                    f"-w", str(options.get('warehouses', 10)),
                    f"-c", str(threads),
                    f"-r", str(options.get('warmupTime', 10)),
                    f"-l", str(options.get('runningTime', 60) * 10),
                    f"-i", str(options.get('reportInterval', 10)),
                    "run"
                ]
                cmd_str = " ".join(cmd_list)
                cwd = tpcc_dir

            elif workload_type == WorkloadType.TPCH:
                # Check if tpch-mysql command exists (basic check)
                if subprocess.run(["which", "tpch-mysql"], capture_output=True).returncode != 0:
                    logger.error("tpch-mysql command not found in PATH.")
                    return False

                cmd_list = [
                    "tpch-mysql",
                    f"--host={host}",
                    f"--port={port}",
                    f"--user={self.db_user}",
                    f"--password={self.db_password}",
                    f"--database={self.tpch_db}",
                    f"--scale-factor=1",
                    f"--threads={threads}",
                    f"--time={options.get('time', 0)}",
                    f"--report-interval={options.get('reportInterval', 10)}",
                    "run"
                ]
                cmd_str = " ".join(map(str, cmd_list))

            if not cmd_list:
                logger.error(f"Unsupported workload type: {workload_type}")
                return False

            logger.debug(f"Executing command for run_id {run_id}: {cmd_str}")

            # Start the process
            process = subprocess.Popen(
                cmd_list,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
                universal_newlines=True,
                cwd=cwd
            )

            # Wait briefly to check if the process started successfully
            time.sleep(1)
            if process.poll() is not None:
                _, stderr = process.communicate()
                logger.error(f"Workload process for run_id {run_id} failed to start. Return code: {process.returncode}")
                if stderr:
                    logger.error(f"Stderr output: {stderr}")
                return False

            # Store process info in memory and Redis
            self.active_procs[run_id] = process
            pid = process.pid
            start_time_iso = datetime.now(timezone.utc).isoformat()

            process_data = {
                "pid": pid,
                "command": cmd_str,
                "workload_type": workload_type.value,
                "run_id": run_id,
                "start_time": start_time_iso
            }
            self.redis_client.set(f"{WORKLOAD_PROCESS_KEY_PREFIX}{run_id}", json.dumps(process_data))
            self.redis_client.sadd(ACTIVE_WORKLOAD_RUNS_SET_KEY, run_id)

            logger.info(f"Successfully started workload process for run_id {run_id} with PID {pid}")
            return True

        except redis.exceptions.ConnectionError as e:
            logger.error(f"Redis error starting workload {run_id}: {e}")
            return False
        except Exception as e:
            logger.exception(f"Error starting workload {run_id}: {str(e)}")
            # Clean up if process started but Redis failed?
            if run_id in self.active_procs:
                try:
                    self.active_procs[run_id].terminate()
                    self.active_procs[run_id].wait(timeout=2)
                except: pass
                del self.active_procs[run_id]
            return False

    async def stop_workload(self, run_id: str) -> bool:
        """Stop a running workload by run_id and remove tracking from Redis."""
        logger.info(f"Attempting to stop workload run_id: {run_id}")
        if not self.redis_client:
            logger.error("Redis client not available. Cannot stop workload.")
            return False

        process_key = f"{WORKLOAD_PROCESS_KEY_PREFIX}{run_id}"
        process_info_json = self.redis_client.get(process_key)

        if not process_info_json:
            logger.warning(f"Workload run_id {run_id} not found in Redis. Cannot stop.")
            # Also check in-memory procs? Might be inconsistent state.
            if run_id in self.active_procs:
                 logger.warning(f"Run_id {run_id} found in active_procs but not Redis. Attempting termination.")
                 process = self.active_procs.pop(run_id)
                 try:
                     process.terminate()
                     process.wait(timeout=5)
                     logger.info(f"Terminated process for run_id {run_id} found only in memory.")
                     return True # Indicate success as process was likely stopped
                 except Exception as e:
                     logger.error(f"Error terminating process for run_id {run_id} found only in memory: {e}")
                     return False
            return False

        try:
            process_info = json.loads(process_info_json)
            pid = process_info.get('pid')

            # Stop the process using PID
            process_terminated = False
            if pid and psutil.pid_exists(pid):
                try:
                    proc = psutil.Process(pid)
                    proc.terminate() # Send SIGTERM
                    proc.wait(timeout=5) # Wait for termination
                    logger.info(f"Terminated process PID {pid} for run_id {run_id}.")
                    process_terminated = True
                except psutil.NoSuchProcess:
                    logger.warning(f"Process PID {pid} for run_id {run_id} not found, likely already stopped.")
                    process_terminated = True # Consider it stopped
                except psutil.TimeoutExpired:
                    logger.warning(f"Timeout waiting for process PID {pid} (run_id {run_id}) to terminate. Sending SIGKILL.")
                    try:
                        proc.kill()
                        proc.wait(timeout=2)
                        process_terminated = True
                    except Exception as kill_e:
                         logger.error(f"Error sending SIGKILL to PID {pid} (run_id {run_id}): {kill_e}")
                except Exception as e:
                    logger.error(f"Error terminating process PID {pid} for run_id {run_id}: {e}")
            elif pid:
                 logger.warning(f"Process PID {pid} for run_id {run_id} does not exist.")
                 process_terminated = True # Already stopped
            else:
                logger.warning(f"No PID found in Redis for run_id {run_id}. Cannot terminate directly.")
                # Cannot guarantee stop if PID is missing

            # Also remove from in-memory dict if present
            if run_id in self.active_procs:
                del self.active_procs[run_id]

            # Clean up Redis entries regardless of termination success (to avoid stale entries)
            self.redis_client.delete(process_key)
            self.redis_client.srem(ACTIVE_WORKLOAD_RUNS_SET_KEY, run_id)
            logger.info(f"Removed Redis entries for run_id {run_id}.")

            return process_terminated # Return True if process is confirmed (or assumed) stopped

        except redis.exceptions.ConnectionError as e:
            logger.error(f"Redis error stopping workload {run_id}: {e}")
            return False # Cannot confirm state without Redis
        except Exception as e:
            logger.exception(f"Error stopping workload run_id {run_id}: {str(e)}")
            # Attempt cleanup even on error
            try:
                if run_id in self.active_procs: del self.active_procs[run_id]
                self.redis_client.delete(process_key)
                self.redis_client.srem(ACTIVE_WORKLOAD_RUNS_SET_KEY, run_id)
            except: pass
            return False

    def get_active_workload_runs_info(self) -> List[Dict]:
        """Get list of active workload runs tracked in Redis."""
        if not self.redis_client:
            logger.error("Redis client not available. Cannot get active workload runs.")
            return []

        active_runs = []
        run_ids_to_cleanup = [] # Track IDs that need cleanup
        try:
            run_ids = list(self.redis_client.smembers(ACTIVE_WORKLOAD_RUNS_SET_KEY))
            if not run_ids:
                return []

            keys_to_fetch = [f"{WORKLOAD_PROCESS_KEY_PREFIX}{run_id}" for run_id in run_ids]
            # Handle case where mget returns None (e.g., Redis down during call)
            process_data_list = self.redis_client.mget(keys_to_fetch) or []

            valid_run_ids = set() # Track IDs successfully retrieved
            for i, process_json in enumerate(process_data_list):
                # Ensure index exists before accessing run_ids
                if i >= len(run_ids):
                    logger.warning(f"Mismatch between run_ids count ({len(run_ids)}) and mget results ({len(process_data_list)}). Skipping extra results.")
                    break
                run_id = run_ids[i]
                if process_json:
                    try:
                        process_info = json.loads(process_json)
                        pid = process_info.get('pid')
                        # Verify process is still running
                        if pid and psutil.pid_exists(pid):
                            # Correct indentation for append dictionary
                            active_runs.append({
                                'run_id': run_id,
                                'pid': pid,
                                'type': process_info.get('workload_type', 'unknown'),
                                'start_time': process_info.get('start_time')
                                # Add command? Might be too verbose
                            })
                            valid_run_ids.add(run_id)
                        elif pid:
                             # PID exists in info but process doesn't exist
                             logger.warning(f"Process for run_id {run_id} (PID: {pid}) not running, but found in Redis. Marking for cleanup.")
                             run_ids_to_cleanup.append(run_id)
                        else:
                             # No PID in info, incomplete data
                             logger.warning(f"Incomplete data for run_id {run_id} in Redis (missing PID). Marking for cleanup.")
                             run_ids_to_cleanup.append(run_id)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to decode JSON for workload process {run_id}. Marking for cleanup.")
                        run_ids_to_cleanup.append(run_id)
                    except Exception as e:
                         logger.error(f"Error processing workload run {run_id}: {e}. Skipping.")
                else:
                     # Key missing, inconsistent state
                     logger.warning(f"Workload run_id {run_id} in active set but key missing. Marking for cleanup.")
                     run_ids_to_cleanup.append(run_id)

            # Perform cleanup outside the loop
            if run_ids_to_cleanup:
                logger.info(f"Cleaning up {len(run_ids_to_cleanup)} stale/invalid workload run entries from Redis.")
                keys_to_delete = [f"{WORKLOAD_PROCESS_KEY_PREFIX}{rid}" for rid in run_ids_to_cleanup]
                try:
                    if keys_to_delete:
                        self.redis_client.delete(*keys_to_delete)
                    self.redis_client.srem(ACTIVE_WORKLOAD_RUNS_SET_KEY, *run_ids_to_cleanup)
                    # Clean from memory too
                    for rid in run_ids_to_cleanup:
                        if rid in self.active_procs: del self.active_procs[rid]
                except redis.exceptions.ConnectionError as redis_e:
                     logger.error(f"Redis connection error during cleanup: {redis_e}")
                except Exception as cleanup_e:
                    logger.error(f"Error during Redis cleanup of workload runs: {cleanup_e}")


            logger.debug(f"Active workload runs found: {len(active_runs)}")
            return active_runs

        except redis.exceptions.ConnectionError as e:
            logger.error(f"Redis error getting active workload runs: {e}")
            return []
        except Exception as e:
            logger.exception("Error getting active workload runs")
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
            
    async def stop_all_workloads(self) -> bool:
        """Stop all active workloads tracked in Redis."""
        logger.info("Attempting to stop all active workload runs...")
        if not self.redis_client:
            logger.error("Redis client not available. Cannot stop all workloads.")
            return False
        
        success = True
        try:
            run_ids = list(self.redis_client.smembers(ACTIVE_WORKLOAD_RUNS_SET_KEY))
            if not run_ids:
                logger.info("No active workload runs found in Redis to stop.")
                # Also clear in-memory just in case
                self.active_procs.clear()
                return True
            
            logger.info(f"Found {len(run_ids)} active workload runs to stop: {run_ids}")
            
            # Use asyncio.gather if stop_workload becomes async, otherwise loop
            for run_id in run_ids:
                if not await self.stop_workload(run_id):
                    logger.warning(f"Failed to stop workload run_id {run_id} during stop-all.")
                    success = False # Mark overall success as False if any stop fails
            
            # Final check - clear any remaining keys/sets just in case stop_workload missed something
            # Might be overly aggressive, consider if needed
            # remaining_ids = self.redis_client.smembers(ACTIVE_WORKLOAD_RUNS_SET_KEY)
            # if remaining_ids:
            #     logger.warning(f"Redis still contains active runs after stop_all: {remaining_ids}. Force cleaning.")
            #     keys_to_del = [f"{WORKLOAD_PROCESS_KEY_PREFIX}{rid}" for rid in remaining_ids]
            #     if keys_to_del:
            #         self.redis_client.delete(*keys_to_del)
            #     self.redis_client.delete(ACTIVE_WORKLOAD_RUNS_SET_KEY)

            # Clear in-memory store
            self.active_procs.clear()

            logger.info("Finished stop_all_workloads attempt.")
            return success
        except redis.exceptions.ConnectionError as e:
            logger.error(f"Redis error during stop_all_workloads: {e}")
            return False
        except Exception as e:
            logger.exception("Error stopping all workloads")
            return False

# Singleton instance of WorkloadService
workload_service = WorkloadService()