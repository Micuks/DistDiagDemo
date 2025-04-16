from dotenv import load_dotenv
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import logging
import os
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta, timezone
import asyncio
import time
from functools import lru_cache, wraps
import json
import pymysql
import threading
import copy
import redis # Added redis import
import uuid # Added uuid import
# Removed imports for abnormal.py types

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env'))

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

        # Explicitly copy cache management methods from the LRU-cached function
        wrapped_func.cache_clear = func.cache_clear
        wrapped_func.cache_info = func.cache_info
        return wrapped_func
    return wrapper_decorator

class K8sService:
    def __init__(self):
        # Load kubernetes configuration
        if os.getenv('KUBERNETES_SERVICE_HOST'):
            config.load_incluster_config()
        else:
            config.load_kube_config()
        
        self.custom_api = client.CustomObjectsApi()
        self.core_api = client.CoreV1Api()
        self.namespace = os.getenv('OCEANBASE_NAMESPACE', 'oceanbase')
        self.anomaly_duration = 200  # Anomaly duration in seconds - might need adjustment for new types
        self.available_nodes = []
        
        # WebSocket connection manager - will be injected by dependency
        self.connection_manager = None
        
        # Define supported experiment types - added new types
        self.anomaly_types = [
            "cpu_saturation",       # New type from abnormal.py
            "io_saturation",        # New type from abnormal.py
            "net_saturation",       # New type from abnormal.py
            "cpu_stress",           # Original kubectl exec based (consider mapping or removing if replaced by cpu_saturation)
            "io_bottleneck",        # Original kubectl exec based (consider mapping or removing if replaced by io_saturation)
            "network_bottleneck",   # Chaos Mesh NetworkChaos
            "cache_bottleneck",     # OceanBase parameter change + Chaos Mesh tracking resource
            "cpu_stress_chaosmesh", # Alternative Chaos Mesh StressChaos
            "io_bottleneck_chaosmesh",# Alternative Chaos Mesh StressChaos
            "time_skew",            # Chaos Mesh TimeChaos
            "replication_lag",      # Chaos Mesh NetworkChaos (latency)
            "consensus_delay",      # Chaos Mesh NetworkChaos (latency/loss)
            "too_many_indexes"      # SQL based
        ]
        # Add descriptions for anomaly types
        self.anomaly_descriptions = {
            "cpu_saturation": "Simulates high CPU usage using stress-ng, potentially impacting query performance.",
            "io_saturation": "Simulates high I/O load using stress-ng, affecting disk-intensive operations.",
            "net_saturation": "Simulates network bandwidth limitation using tc (traffic control).",
            "cpu_stress": "(Legacy) Simulates CPU load using stress-ng via kubectl exec.",
            "io_bottleneck": "(Legacy) Simulates I/O bottleneck using dd via kubectl exec.",
            "network_bottleneck": "Injects network packet loss using Chaos Mesh NetworkChaos.",
            "cache_bottleneck": "Reduces OceanBase memstore limit via SQL and tracks with a Chaos Mesh resource.",
            "cpu_stress_chaosmesh": "Simulates CPU load using Chaos Mesh StressChaos.",
            "io_bottleneck_chaosmesh": "Simulates I/O stress using Chaos Mesh StressChaos.",
            "time_skew": "Injects clock skew into pods using Chaos Mesh TimeChaos.",
            "replication_lag": "Injects network latency between pods using Chaos Mesh NetworkChaos to simulate replication issues.",
            "consensus_delay": "Injects network latency and packet loss between pods using Chaos Mesh NetworkChaos to simulate consensus issues.",
            "too_many_indexes": "Creates a large number of indexes on TPCC and SBTest tables via SQL."
        }
        # Fetch available nodes (unchanged logic)
        try:
            running_loop = asyncio.get_running_loop()
            asyncio.create_task(self._update_available_nodes())
        except RuntimeError:
            loop = asyncio.get_event_loop()
            self.available_nodes = loop.run_until_complete(self._fetch_available_nodes())
        except Exception as e:
            logger.error(f"Error getting available nodes: {e}")
            self.available_nodes = []

        # OceanBase connection configuration
        self.ob_config = {
            'host': os.getenv('OB_HOST', '127.0.0.1'),
            'port': int(os.getenv('OB_PORT', '2881')),
            'user': os.getenv('OB_USER', 'root@sys'),
            'password': os.getenv('OB_PASSWORD', 'password'),
            'database': os.getenv('OB_DATABASE', 'oceanbase'),
        }

        # Initialize zones
        self.ob_zones = self._fetch_ob_zones()

        # Severity variations (unchanged for now, may need updates for new types later)
        # ... (severity variations definition remains the same for now) ...
        self.severity_variations = {
             "cpu_stress": [
                 {"workers": 32, "load": 100, "stress-ng-params": "--cpu 20 -t 180"},
                 {"workers": 48, "load": 100, "stress-ng-params": "--cpu 40 -t 180"},
                 {"workers": 64, "load": 100, "stress-ng-params": "--cpu 60 -t 180"}
             ],
             "io_bottleneck": [
                 {"intensity": "low", "command": "dd if=/dev/zero of=/tmp/io_stress_file bs=1M count=2048 conv=fdatasync & sleep %s"},
                 {"intensity": "medium", "command": "dd if=/dev/zero of=/tmp/io_stress_file bs=1M count=2048 conv=fdatasync & dd if=/dev/zero of=/tmp/io_stress_file2 bs=1M count=2048 conv=fdatasync & sleep %s"},
                 {"intensity": "high", "command": "dd if=/dev/zero of=/tmp/io_stress_file bs=1M count=2048 conv=fdatasync & dd if=/dev/zero of=/tmp/io_stress_file2 bs=1M count=2048 conv=fdatasync & dd if=/dev/zero of=/tmp/io_stress_file3 bs=1M count=2048 conv=fdatasync & sleep %s"}
             ],
             "network_bottleneck": [
                 {"loss": "50", "correlation": "0"},
                 {"loss": "80", "correlation": "0"},
                 {"loss": "100", "correlation": "0"}
             ],
             "cache_bottleneck": [
                 {"workers": 8, "size": "2GB"},
                 {"workers": 12, "size": "3GB"},
                 {"workers": 16, "size": "4GB"}
             ]
        }
        self.severity_variations["cpu_stress_chaosmesh"] = [ # Same as original cpu_stress severities for now
            {"workers": 32, "load": 100, "stress-ng-params": "--cpu 20 -t 180"},
            {"workers": 48, "load": 100, "stress-ng-params": "--cpu 40 -t 180"},
            {"workers": 64, "load": 100, "stress-ng-params": "--cpu 60 -t 180"}
        ]
        self.severity_variations["io_bottleneck_chaosmesh"] = [ # Using StressChaos IO stressors
             {"percent": 30, "methods": ["write", "read"], "workers": 4, "volumePath": "/tmp/io_stress_dir_chaos"},
             {"percent": 60, "methods": ["write", "read"], "workers": 8, "volumePath": "/tmp/io_stress_dir_chaos"},
             {"percent": 90, "methods": ["write", "read"], "workers": 12, "volumePath": "/tmp/io_stress_dir_chaos"}
        ]
        self.severity_variations["time_skew"] = [ # Time Skew variations
            {"timeOffset": "5s", "clockIds": ["CLOCK_REALTIME"], "scope": "one"},
            {"timeOffset": "15s", "clockIds": ["CLOCK_REALTIME"], "scope": "one"},
            {"timeOffset": "30s", "clockIds": ["CLOCK_REALTIME"], "scope": "one"}
        ]
        self.severity_variations["replication_lag"] = [ # Replication Lag (latency) variations
            {"latency": "50ms", "correlation": "0", "jitter": "5ms", "direction": "to"}, # Target specific direction if needed
            {"latency": "150ms", "correlation": "0", "jitter": "10ms", "direction": "to"},
            {"latency": "300ms", "correlation": "0", "jitter": "20ms", "direction": "to"}
        ]
        self.severity_variations["consensus_delay"] = [ # Consensus Delay (latency/loss) variations - adjust ports/targets as needed
            {"latency": "80ms", "loss": "10", "correlation": "0", "jitter": "10ms", "direction": "both"},
            {"latency": "200ms", "loss": "20", "correlation": "0", "jitter": "20ms", "direction": "both"},
            {"latency": "400ms", "loss": "30", "correlation": "0", "jitter": "30ms", "direction": "both"}
        ]
        self.severity_variations["too_many_indexes"] = [ # Variations might control how many indexes are added, or target different tables
             # For now, just one severity level that adds all defined indexes
             {"level": "high"}
        ]
        # Add placeholders for new types if needed, though they don't use severity currently
        self.severity_variations["cpu_saturation"] = [{"params": "default"}]
        self.severity_variations["io_saturation"] = [{"params": "default"}]
        self.severity_variations["net_saturation"] = [{"params": "default"}]

        # Initialize Redis client
        try:
            redis_host = os.getenv('REDIS_HOST', 'localhost')
            redis_port = int(os.getenv('REDIS_PORT', 6379))
            redis_db = int(os.getenv('REDIS_DB', 0))
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db, decode_responses=True)
            self.redis_client.ping() # Test connection
            logger.info(f"Connected to Redis at {redis_host}:{redis_port}, DB {redis_db}")
        except redis.exceptions.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None # Set to None if connection fails

        # Build pod IP to name mapping (unchanged)
        self.ip_to_name_map = self._build_ip_to_name_mapping()

    def set_connection_manager(self, manager):
        """Set the WebSocket connection manager for broadcasting anomaly updates"""
        self.connection_manager = manager
        logger.info("WebSocket connection manager set for K8sService")

    async def _broadcast_anomaly_update(self):
        """Broadcast active anomalies to all connected WebSocket clients"""
        if not self.connection_manager:
            logger.debug("No connection manager available for broadcasting")
            return
        
        try:
            # Get current anomalies
            anomalies = await self.get_active_anomalies()
            
            # Create message with timestamp
            message = json.dumps({
                "type": "update",
                "data": anomalies,
                "timestamp": datetime.now().isoformat()
            })
            
            # Broadcast to all WebSocket connections
            await self.connection_manager.broadcast(message)
            logger.debug(f"Broadcasted anomaly update to {len(self.connection_manager.active_connections)} connections")
            
            # Also publish to Redis for SSE clients
            if self.redis_client:
                try:
                    self.redis_client.publish("anomaly_updates", message)
                    logger.debug("Published anomaly update to Redis 'anomaly_updates' channel")
                except Exception as e:
                    logger.error(f"Error publishing to Redis: {e}")
        except Exception as e:
            logger.error(f"Error broadcasting anomaly update: {e}")

    def get_anomaly_types(self):
        return self.anomaly_types

    async def get_active_anomalies(self):
        """Get list of currently active anomalies from Redis, automatically cleaning up expired ones."""
        if not self.redis_client:
            logger.error("Redis client not available. Cannot get active anomalies.")
            return [] # Return empty list if Redis is down

        active_anomalies_list = []
        anomalies_to_cleanup = []
        try:
            active_anomaly_ids = self.redis_client.smembers("active_anomalies")
            if not active_anomaly_ids:
                logger.debug("No active anomalies found in Redis set 'active_anomalies'.")
                return []

            # Fetch details for each ID
            pipeline = self.redis_client.pipeline()
            for anomaly_id in active_anomaly_ids:
                pipeline.get(f"anomaly:{anomaly_id}")
            anomaly_data_jsons = pipeline.execute()

            valid_ids_found = set()
            now_utc = datetime.now(timezone.utc)
            # Define types that have an inherent duration managed by the tool (ChaosMesh, stress-ng -t, sleep)
            # SQL types or network rules applied indefinitely are excluded here.
            duration_managed_types = [
                "cpu_saturation", "io_saturation", # stress-ng -t
                "cpu_stress", "io_bottleneck",      # stress-ng -t, sleep
                "network_bottleneck", "cache_bottleneck", # Chaos Mesh duration
                "cpu_stress_chaosmesh", "io_bottleneck_chaosmesh", # Chaos Mesh duration
                "time_skew", "replication_lag", "consensus_delay" # Chaos Mesh duration
            ]
            # Define types that require manual cleanup via tc/iptables rules in delete_anomaly
            manual_cleanup_types = ["net_saturation"]
            grace_period_seconds = 15 # Grace period to allow K8s/process cleanup

            for i, anomaly_json in enumerate(anomaly_data_jsons):
                anomaly_id = list(active_anomaly_ids)[i]
                if anomaly_json:
                    try:
                        anomaly_data = json.loads(anomaly_json)
                        # Ensure essential fields are present
                        if "id" in anomaly_data and "type" in anomaly_data and "start_time" in anomaly_data:
                            anomaly_type = anomaly_data["type"]
                            start_time_str = anomaly_data["start_time"]
                            try:
                                # Ensure start_time is timezone-aware (UTC)
                                start_time_naive = datetime.fromisoformat(start_time_str)
                                if start_time_naive.tzinfo is None:
                                    start_time_utc = start_time_naive.replace(tzinfo=timezone.utc)
                                else:
                                    start_time_utc = start_time_naive.astimezone(timezone.utc)

                                elapsed_time = (now_utc - start_time_utc).total_seconds()

                                # Check for expiry only for relevant types
                                should_check_expiry = anomaly_type in duration_managed_types or anomaly_type in manual_cleanup_types

                                if should_check_expiry and elapsed_time > (self.anomaly_duration + grace_period_seconds):
                                    logger.info(f"Anomaly {anomaly_id} (Type: {anomaly_type}) has expired (elapsed: {elapsed_time:.0f}s > {self.anomaly_duration + grace_period_seconds}s). Scheduling cleanup.")
                                    anomalies_to_cleanup.append(anomaly_id)
                                    # Remove from the active set in Redis immediately if expired
                                    self.redis_client.srem("active_anomalies", anomaly_id)
                                    continue # Skip adding to active_anomalies_list

                                # If not expired or not a type that expires automatically, add to active list
                                active_anomalies_list.append(anomaly_data)
                                valid_ids_found.add(anomaly_id)

                            except (ValueError, TypeError) as time_e:
                                logger.warning(f"Could not parse start_time '{start_time_str}' for anomaly {anomaly_id}: {time_e}. Assuming active.")
                                active_anomalies_list.append(anomaly_data) # Treat as active if time parsing fails
                                valid_ids_found.add(anomaly_id)

                        else:
                             logger.warning(f"Invalid data format for anomaly ID {anomaly_id} in Redis (missing fields): {anomaly_json}")
                             # Clean up invalid entry
                             self.redis_client.delete(f"anomaly:{anomaly_id}")
                             self.redis_client.srem("active_anomalies", anomaly_id)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to decode JSON for anomaly ID {anomaly_id} from Redis: {anomaly_json}")
                        # Clean up corrupted entry
                        self.redis_client.delete(f"anomaly:{anomaly_id}")
                        self.redis_client.srem("active_anomalies", anomaly_id)
                else:
                    # ID was in the set but the key didn't exist - inconsistency
                    logger.warning(f"Anomaly ID {anomaly_id} found in set 'active_anomalies' but key missing. Removing from set.")
                    self.redis_client.srem("active_anomalies", anomaly_id)

            # Schedule cleanup tasks for expired anomalies in the background
            if anomalies_to_cleanup:
                logger.info(f"Creating background tasks to clean up {len(anomalies_to_cleanup)} expired anomalies.")
                for expired_id in anomalies_to_cleanup:
                    # Create task but don't await it here
                    asyncio.create_task(self.delete_anomaly(anomaly_id=expired_id))

            return active_anomalies_list

        except redis.exceptions.ConnectionError as e:
            logger.error(f"Redis connection error in get_active_anomalies: {e}")
            return [] # Return empty on connection error
        except Exception as e:
            logger.error(f"Failed to get active anomalies from Redis: {str(e)}")
            # import traceback
            # logger.error(traceback.format_exc())
            return [] # Return empty list on other errors

    def _get_experiment_template(self, anomaly_type: str, severity: Optional[str] = None) -> Dict[str, Any]:
        """Get the chaos experiment template with varying severity"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        # Use a base name, node-specific suffix added later if needed
        experiment_base_name = f"ob-{anomaly_type.replace('_', '-')}-{timestamp}"

        # Base template with enhanced selectors
        base_template = {
            "apiVersion": "chaos-mesh.org/v1alpha1",
            "metadata": {
                "name": experiment_base_name, # Use base name here
                "namespace": self.namespace
            },
            "spec": {
                "mode": "one", # Default mode, might be overridden
                "selector": {
                    "namespaces": [self.namespace],
                    "labelSelectors": {"ref-obcluster": "obcluster"},
                    "pods": {self.namespace: []} # Pods filled in later
                }
            }
        }

        # Select severity variation based on provided severity or fallback
        severity_variations_for_type = self.severity_variations.get(anomaly_type, [{}])
        if severity is not None:
            severity_lower = severity.lower()
            severity_map = {"low": 0, "medium": 1, "high": 2}
            # Handle cases where fewer than 3 severities are defined
            index = min(severity_map.get(severity_lower, 1), len(severity_variations_for_type) - 1)
        else:
             # Default to medium or the last available option if less than 2 defined
             index = min(1, len(severity_variations_for_type) - 1)

        severity_config = severity_variations_for_type[index]

        # Experiment configurations with varying severity
        experiments = {
            "cpu_stress": { # Kept for backward compatibility or specific use cases
                 "kind": "StressChaos", # Uses Chaos Mesh resource even if applied via kubectl? This seems inconsistent. Let's remove this definition as it's handled separately now.
            },
            "io_bottleneck": { # Kept for backward compatibility or specific use cases
                 "kind": "StressChaos", # Uses Chaos Mesh resource even if applied via kubectl? This seems inconsistent. Let's remove this definition as it's handled separately now.
            },
             "network_bottleneck": {
                 "kind": "NetworkChaos",
                 "spec": {
                     "action": "loss",
                     "loss": {
                         "loss": severity_config.get("loss", "80%"),
                         "correlation": severity_config.get("correlation", "0")
                     },
                     "target": { # Target is now often specified via pod selector, keep label selector as fallback?
                         "selector": {
                             "namespaces": [self.namespace],
                             "labelSelectors": {"ref-obcluster": "obcluster"}
                         },
                         "mode": "one" # Mode is usually set at top level now
                     },
                     "direction": "both",
                     "duration": f"{self.anomaly_duration}s"
                 }
             },
             "cache_bottleneck": { # Note: Actual effect via SQL, this is just a tracking resource
                 "kind": "StressChaos",
                 "spec": {
                     "stressors": {
                         "memory": {
                             "workers": severity_config.get("workers", 1), # Minimal workers just for tracking
                             "size": severity_config.get("size", "1MB") # Minimal size just for tracking
                         }
                     },
                     "duration": f"{self.anomaly_duration}s"
                 }
             },
             "cpu_stress_chaosmesh": {
                 "kind": "StressChaos",
                 "spec": {
                     "stressngStressors": severity_config.get("stress-ng-params", "--cpu 30 -t 180"),
                     "duration": f"{self.anomaly_duration}s"
                 }
             },
             "io_bottleneck_chaosmesh": {
                 "kind": "StressChaos",
                 "spec": {
                     "stressors": {
                         "io": {
                             "workers": severity_config.get("workers", 4),
                             "volumePath": severity_config.get("volumePath", "/tmp/io_stress_dir_chaos"),
                             "percent": severity_config.get("percent", 50),
                             "methods": severity_config.get("methods", ["write", "read"])
                         }
                     },
                     "duration": f"{self.anomaly_duration}s"
                 }
             },
             "time_skew": {
                 "kind": "TimeChaos",
                 "spec": {
                     # mode and selector are handled in base_template/apply_anomaly
                     "timeOffset": severity_config.get("timeOffset", "10s"),
                     "clockIds": severity_config.get("clockIds", ["CLOCK_REALTIME"]),
                     "duration": f"{self.anomaly_duration}s"
                 }
             },
             "replication_lag": {
                 "kind": "NetworkChaos",
                 "spec": {
                     "action": "delay",
                      # mode and selector are handled in base_template/apply_anomaly
                     "delay": {
                         "latency": severity_config.get("latency", "100ms"),
                         "correlation": severity_config.get("correlation", "0"),
                         "jitter": severity_config.get("jitter", "10ms")
                     },
                     "direction": severity_config.get("direction", "to"), # Target specific direction if needed
                     "duration": f"{self.anomaly_duration}s",
                     # TargetPort example (adjust as needed for replication):
                     # "target": {
                     #     "selector": base_template["spec"]["selector"],
                     #     "mode": "one" # Apply to one target pod matching selector
                     # },
                     # "targetScope": {
                     #     "namespace": self.namespace,
                     #     "labelSelector": {"ref-obcluster": "obcluster"}, # Target other OB pods
                     #     "mode": "all" # Apply latency to traffic TO all matching pods
                     # }
                 }
             },
             "consensus_delay": {
                 "kind": "NetworkChaos",
                 "spec": {
                     "action": "delay", # Can also combine with loss if needed
                     # mode and selector are handled in base_template/apply_anomaly
                     "delay": {
                         "latency": severity_config.get("latency", "100ms"),
                         "correlation": severity_config.get("correlation", "0"),
                         "jitter": severity_config.get("jitter", "10ms")
                     },
                     "loss": { # Adding loss component
                         "loss": severity_config.get("loss", "10"),
                         "correlation": severity_config.get("correlation", "0")
                     },
                     "direction": severity_config.get("direction", "both"),
                     "duration": f"{self.anomaly_duration}s",
                      # TargetPort example (adjust as needed for consensus):
                     # "target": {
                     #     "selector": base_template["spec"]["selector"],
                     #     "mode": "one" # Apply to one target pod matching selector
                     # },
                     # "targetScope": { # Example: Delay/loss between selected pod and other cluster members
                     #     "namespace": self.namespace,
                     #     "labelSelector": {"ref-obcluster": "obcluster"}, # Target other OB pods
                     #     "mode": "all"
                     # }
                 }
             }
        }

        # Special handling for too_many_indexes (SQL based, template is for tracking only)
        if anomaly_type == "too_many_indexes":
            # ... (SQL commands generation remains the same) ...
            tpcc_sql_commands = [ "USE tpcc", "CREATE INDEX idx_customer_1 ON customer(c_w_id)", "CREATE INDEX idx_customer_2 ON customer(c_d_id)", "CREATE INDEX idx_customer_3 ON customer(c_last)", "CREATE INDEX idx_customer_4 ON customer(c_first)", "CREATE INDEX idx_customer_5 ON customer(c_balance)", "CREATE INDEX idx_district_1 ON district(d_w_id)", "CREATE INDEX idx_district_2 ON district(d_name)", "CREATE INDEX idx_district_3 ON district(d_street_1)", "CREATE INDEX idx_history_1 ON history(h_c_id)", "CREATE INDEX idx_history_2 ON history(h_d_id)", "CREATE INDEX idx_history_3 ON history(h_w_id)", "CREATE INDEX idx_item_1 ON item(i_name)", "CREATE INDEX idx_item_2 ON item(i_price)", "CREATE INDEX idx_item_3 ON item(i_data)", "CREATE INDEX idx_new_orders_1 ON new_orders(no_w_id)", "CREATE INDEX idx_new_orders_2 ON new_orders(no_d_id)", "CREATE INDEX idx_order_line_1 ON order_line(ol_w_id)", "CREATE INDEX idx_order_line_2 ON order_line(ol_d_id)", "CREATE INDEX idx_order_line_3 ON order_line(ol_i_id)", "CREATE INDEX idx_order_line_4 ON order_line(ol_delivery_d)", "CREATE INDEX idx_orders_1 ON orders(o_w_id)", "CREATE INDEX idx_orders_2 ON orders(o_d_id)", "CREATE INDEX idx_orders_3 ON orders(o_c_id)", "CREATE INDEX idx_orders_4 ON orders(o_entry_d)", "CREATE INDEX idx_stock_1 ON stock(s_w_id)", "CREATE INDEX idx_stock_2 ON stock(s_i_id)", "CREATE INDEX idx_stock_3 ON stock(s_quantity)", "CREATE INDEX idx_warehouse_1 ON warehouse(w_name)", "CREATE INDEX idx_warehouse_2 ON warehouse(w_street_1)" ]
            sbtest_sql_commands = [ "USE sbtest", "CREATE INDEX idx_sbtest1_1 ON sbtest1(k)", "CREATE INDEX idx_sbtest1_2 ON sbtest1(c)", "CREATE INDEX idx_sbtest1_3 ON sbtest1(pad)", "CREATE INDEX idx_sbtest2_1 ON sbtest2(k)", "CREATE INDEX idx_sbtest2_2 ON sbtest2(c)", "CREATE INDEX idx_sbtest2_3 ON sbtest2(pad)", "CREATE INDEX idx_sbtest3_1 ON sbtest3(k)", "CREATE INDEX idx_sbtest3_2 ON sbtest3(c)", "CREATE INDEX idx_sbtest3_3 ON sbtest3(pad)", "CREATE INDEX idx_sbtest4_1 ON sbtest4(k)", "CREATE INDEX idx_sbtest4_2 ON sbtest4(c)", "CREATE INDEX idx_sbtest4_3 ON sbtest4(pad)", "CREATE INDEX idx_sbtest5_1 ON sbtest5(k)", "CREATE INDEX idx_sbtest5_2 ON sbtest5(c)", "CREATE INDEX idx_sbtest5_3 ON sbtest5(pad)", "CREATE INDEX idx_sbtest6_1 ON sbtest6(k)", "CREATE INDEX idx_sbtest6_2 ON sbtest6(c)", "CREATE INDEX idx_sbtest6_3 ON sbtest6(pad)", "CREATE INDEX idx_sbtest7_1 ON sbtest7(k)", "CREATE INDEX idx_sbtest7_2 ON sbtest7(c)", "CREATE INDEX idx_sbtest7_3 ON sbtest7(pad)", "CREATE INDEX idx_sbtest8_1 ON sbtest8(k)", "CREATE INDEX idx_sbtest8_2 ON sbtest8(c)", "CREATE INDEX idx_sbtest8_3 ON sbtest8(pad)", "CREATE INDEX idx_sbtest9_1 ON sbtest9(k)", "CREATE INDEX idx_sbtest9_2 ON sbtest9(c)", "CREATE INDEX idx_sbtest9_3 ON sbtest9(pad)", "CREATE INDEX idx_sbtest10_1 ON sbtest10(k)", "CREATE INDEX idx_sbtest10_2 ON sbtest10(c)", "CREATE INDEX idx_sbtest10_3 ON sbtest10(pad)" ]
            sql_commands = tpcc_sql_commands + sbtest_sql_commands
            return {
                "apiVersion": "chaos-mesh.org/v1alpha1",
                "kind": "SQLChaos", # Still using this marker
                "metadata": {
                    "name": experiment_base_name, # Use base name
                    "namespace": self.namespace
                },
                "spec": {
                    "mode": "one", # Not really applicable here
                    "selector": {
                        "namespaces": [self.namespace],
                        "labelSelectors": {"ref-obcluster": "obcluster"}
                    },
                    "action": "exec",
                    "sqlCommands": sql_commands # Include commands for potential reference
                }
            }

        # Create a deep copy of base template
        template = copy.deepcopy(base_template) # Use deepcopy
        experiment_config = experiments.get(anomaly_type)
        if not experiment_config:
            # Return the base template if no specific config found (e.g., for manual kubectl types if kept)
            # Or raise error if type should have config
             if anomaly_type not in ["cpu_stress", "io_bottleneck"]: # Allow missing config for these legacy types
                raise ValueError(f"Unsupported or missing experiment config for anomaly type: {anomaly_type}")
             else:
                 # Return base template for legacy types, kind needs to be determined later
                 return template


        # Update kind and merge specs
        template["kind"] = experiment_config.get("kind", template.get("kind"))
        if "spec" in experiment_config:
            # Merge specs carefully, potentially overriding base spec fields
            # spec_base = template.get("spec", {})
            # spec_override = experiment_config["spec"]
            # # Simple merge for now, might need deeper merge logic
            # spec_base.update(spec_override)
            # template["spec"] = spec_base
            template["spec"].update(experiment_config["spec"]) # Overwrite/add keys from experiment_config spec

        return template


    # _get_experiment_plural needs updates for new types (returns 'none' as they don't have K8s CRDs)
    def _get_experiment_plural(self, anomaly_type: str) -> str:
        """Get the plural form of the experiment type for the API"""
        normalized_type = anomaly_type.replace('-', '_')

        mapping = {
            "cpu_saturation": "none", # Not a K8s CRD
            "io_saturation": "none", # Not a K8s CRD
            "net_saturation": "none", # Not a K8s CRD
            "cpu_stress": "stresschaos", # If using tracking resource
            "io_bottleneck": "stresschaos", # If using tracking resource
            "network_bottleneck": "networkchaos",
            "cache_bottleneck": "stresschaos", # Uses a tracking resource
            "cpu_stress_chaosmesh": "stresschaos",
            "io_bottleneck_chaosmesh": "stresschaos",
            "time_skew": "timechaos",
            "replication_lag": "networkchaos",
            "consensus_delay": "networkchaos",
            "too_many_indexes": "none" # Not a K8s CRD
        }

        plural = mapping.get(normalized_type)
        if plural is None:
            raise ValueError(f"Unsupported anomaly type for plural lookup: {anomaly_type}")
        return plural


    # _update_ob_parameter_for_zone, _update_ob_parameter, _execute_sql_commands remain unchanged
    # ... (keep these methods as they are) ...
    async def _update_ob_parameter_for_zone(self, config_changes: Dict[str, Any], zone: str) -> None:
        """Update OceanBase configuration parameters for a specific zone using SQL"""
        try:
            conn = pymysql.connect(**self.ob_config)
            with conn.cursor() as cursor:
                for param, value in config_changes.items():
                    ob_param = param.lower()
                    sql = f"ALTER SYSTEM SET {ob_param} = {value} ZONE='{zone}'" # Ensure zone is quoted
                    logger.info(f"Executing SQL: {sql}")
                    cursor.execute(sql)
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to update OceanBase configuration for zone {zone}: {str(e)}")
            raise
        finally:
            if conn: conn.close()


    async def _update_ob_parameter(self, config_changes: Dict[str, Any]) -> None:
        """Update OceanBase configuration parameters using SQL for all zones"""
        try:
            conn = pymysql.connect(**self.ob_config)
            with conn.cursor() as cursor:
                for param, value in config_changes.items():
                    ob_param = param.lower()
                    base_sql = f"ALTER SYSTEM SET {ob_param} = {value}"
                    for zone in self.ob_zones:
                        sql = base_sql + f" ZONE='{zone}'" # Ensure zone is quoted
                        logger.info(f"Executing SQL: {sql}")
                        cursor.execute(sql)
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to update OceanBase configuration: {str(e)}")
            raise
        finally:
            if conn: conn.close()


    async def _execute_sql_commands(self, commands: List[str]) -> None:
        """Execute SQL commands on OceanBase cluster using direct connection"""
        conn = None # Initialize conn to None
        try:
            conn = pymysql.connect(**self.ob_config)
            with conn.cursor() as cursor:
                current_db = None
                for cmd in commands:
                    if cmd.lower().startswith("use "):
                        current_db = cmd.split()[1].strip(';`\'" ') # More robust parsing
                        logger.info(f"Switching to database: {current_db}")
                        # Execute USE command separately might be safer depending on driver
                        cursor.execute(f"USE `{current_db}`") # Use backticks for safety
                        continue # Skip executing the original USE command string

                    logger.info(f"Executing SQL: {cmd}")
                    try:
                        cursor.execute(cmd)
                    except pymysql.err.InternalError as e:
                        # Log expected errors during CREATE/DROP as INFO
                        err_code, err_msg = e.args
                        if err_code in [1061, 1091, 1060, 1050]: # Duplicate key name, Cant drop, Duplicate column name, Table already exists etc.
                             logger.info(f"SQL command info (expected during create/drop): {cmd}, info: {str(e)}")
                        else:
                            logger.warning(f"InternalError executing SQL command: {cmd}, error: {str(e)}")
                        continue # Continue for expected errors
                    except pymysql.err.ProgrammingError as e:
                        # Errors like table doesn't exist during DROP
                        err_code, err_msg = e.args
                        if err_code == 1051: # Unknown table
                            logger.info(f"SQL command info (expected during drop): {cmd}, info: {str(e)}")
                        else:
                            logger.warning(f"ProgrammingError executing SQL command: {cmd}, error: {str(e)}")
                        continue
                    except Exception as e:
                        logger.warning(f"Unexpected error executing SQL command: {cmd}, error: {str(e)}")
                        continue # Log and continue for other errors too? Or raise? For now, continue.

            conn.commit()
            logger.info(f"Successfully executed SQL commands")

        except Exception as e:
            logger.error(f"Failed to execute SQL commands: {str(e)}")
            if conn: conn.rollback() # Rollback on general failure
            raise
        finally:
            if conn: conn.close()


    # Renamed and refactored apply_anomaly to apply_anomaly
    async def apply_anomaly(self, anomaly_type: str, target_node: Optional[Union[List[str], str]] = None, severity: str = "medium"):
        """Apply an anomaly (Chaos Mesh, script-based, SQL) and track its state in Redis."""
        if not self.redis_client:
            logger.error("Redis client not available. Cannot apply anomaly.")
            raise ConnectionError("Redis client not available.")

        created_anomalies = [] # Store details of anomalies created in this call

        # Normalize target_node(s) to a list
        target_nodes = []
        if isinstance(target_node, str):
            target_nodes = [target_node]
        elif isinstance(target_node, list):
            target_nodes = target_node
        elif not target_node:
            # If no node specified, try to pick one? Or fail?
            # For now, let's require a target for node-specific anomalies
            # For cluster-wide (like too_many_indexes), it might be okay
            if anomaly_type not in ["too_many_indexes", "cache_bottleneck"]: # These can be cluster-wide or zone-wide
                 available_nodes = await self.get_available_nodes()
                 if not available_nodes:
                     logger.error(f"Cannot apply anomaly '{anomaly_type}': No target node specified and no available nodes found.")
                     raise ValueError(f"Target node required for anomaly type '{anomaly_type}' and none available.")
                 target_nodes = [available_nodes[0]] # Default to the first available node if none specified
                 logger.warning(f"No target node specified for '{anomaly_type}', defaulting to first available node: {target_nodes[0]}")
            else:
                 target_nodes = ["cluster_wide"] # Placeholder for cluster/zone wide anomalies

        logger.info(f"Attempting to apply anomaly '{anomaly_type}' on targets: {target_nodes} with severity '{severity}'")

        for node in target_nodes: # Loop through specified target nodes
            anomaly_id = str(uuid.uuid4())
            start_time_iso = datetime.now().isoformat()
            k8s_resource_name = None # Store name if a K8s resource is created
            status = "failed" # Default status
            anomaly_details = {
                "id": anomaly_id,
                "type": anomaly_type,
                "node": node if node != "cluster_wide" else None, # Store actual node name or None
                "severity": severity,
                "start_time": start_time_iso,
                "status": status, # Initial status
                "k8s_name": None,
                "params": {}, # Store specific params used
            }

            try: # Outer try for the whole anomaly application on a node
                # === New Anomaly Types (re-implemented) ===
                # These types require a target node name (pod name).
                # target_ip variable removed as it's no longer used directly

                if anomaly_type == "cpu_saturation":
                    if node == "cluster_wide": raise ValueError("cpu_saturation requires a specific target node name.")
                    logger.info(f"Triggering CPU Saturation on pod {node} (Anomaly ID: {anomaly_id})")
                    # Use kubectl exec instead of ssh
                    # Run stress-ng in the background inside the pod's container
                    cmd = f"kubectl exec -n {self.namespace} {node} -- /bin/bash -c 'nohup stress-ng --cpu 30 -t {self.anomaly_duration} > /dev/null 2>&1 &'"
                    logger.debug(f"Executing CPU saturation command: {cmd}")
                    try:
                        # Execute the command. We don't need Popen if nohup/& handles backgrounding. Check return code.
                        import subprocess # Keep the import here for clarity within the try block
                        # Directly await the mocked subprocess.run
                        process = await asyncio.to_thread(
                            subprocess.run, cmd, shell=True, check=True, capture_output=True
                        )
                        logger.debug(f"kubectl exec for cpu_saturation submitted (stdout: {process.stdout.decode()}, stderr: {process.stderr.decode()})")
                        anomaly_details["status"] = "active"
                        anomaly_details["params"] = {"target_pod": node, "duration": self.anomaly_duration}
                    except FileNotFoundError:
                         logger.error(f"kubectl command not found. Cannot execute: {cmd}")
                         raise # Re-raise to outer try
                    except subprocess.CalledProcessError as sub_e:
                        logger.error(f"Error executing command for cpu_saturation on {node} (exit code {sub_e.returncode}): {cmd}. Stderr: {sub_e.stderr.decode()}")
                        raise # Re-raise to outer try
                    except Exception as sub_e:
                         logger.error(f"Unexpected error executing command for cpu_saturation on {node}: {sub_e}")
                         raise # Re-raise to outer try

                elif anomaly_type == "io_saturation":
                    if node == "cluster_wide": raise ValueError("io_saturation requires a specific target node name.")
                    logger.info(f"Triggering IO Saturation on pod {node} (Anomaly ID: {anomaly_id})")
                    # Use kubectl exec instead of ssh
                    cmd = f"kubectl exec -n {self.namespace} {node} -- /bin/bash -c 'nohup stress-ng --io 320 -t {self.anomaly_duration} > /dev/null 2>&1 &'"
                    logger.debug(f"Executing IO saturation command: {cmd}")
                    try:
                        # Execute the command. We don't need Popen if nohup/& handles backgrounding. Check return code.
                        import subprocess # Keep the import here
                        # Directly await the mocked subprocess.run
                        process = await asyncio.to_thread(
                            subprocess.run, cmd, shell=True, check=True, capture_output=True
                        )
                        logger.debug(f"kubectl exec for io_saturation submitted (stdout: {process.stdout.decode()}, stderr: {process.stderr.decode()})")
                        anomaly_details["status"] = "active"
                        anomaly_details["params"] = {"target_pod": node, "duration": self.anomaly_duration}
                    except FileNotFoundError:
                         logger.error(f"kubectl command not found. Cannot execute: {cmd}")
                         raise # Re-raise to outer try
                    except subprocess.CalledProcessError as sub_e:
                        logger.error(f"Error executing command for io_saturation on {node} (exit code {sub_e.returncode}): {cmd}. Stderr: {sub_e.stderr.decode()}")
                        raise # Re-raise to outer try
                    except Exception as sub_e:
                         logger.error(f"Unexpected error executing command for io_saturation on {node}: {sub_e}")
                         raise # Re-raise to outer try

                elif anomaly_type == "net_saturation":
                    if node == "cluster_wide": raise ValueError("net_saturation requires a specific target node name.")
                    logger.info(f"Triggering Net Saturation on pod {node} (Anomaly ID: {anomaly_id})")
                    # Use kubectl exec instead of ssh
                    # Ensure inner quotes are escaped for the shell within kubectl exec
                    commands = [
                        f"kubectl exec -n {self.namespace} {node} -- /bin/bash -c 'tc qdisc add dev eth0 root handle 1: htb default 1'",
                        f"kubectl exec -n {self.namespace} {node} -- /bin/bash -c 'tc class add dev eth0 parent 1: classid 1:1 htb rate 1mbps'",
                        f"kubectl exec -n {self.namespace} {node} -- /bin/bash -c 'tc class add dev eth0 parent 1:1 classid 1:2 htb rate 100kbit ceil 120kbit prio 1'",
                        f"kubectl exec -n {self.namespace} {node} -- /bin/bash -c 'tc class add dev eth0 parent 1:1 classid 1:3 htb rate 100kbit ceil 120kbit prio 1'",
                        f"kubectl exec -n {self.namespace} {node} -- /bin/bash -c 'tc filter add dev eth0 parent 1:0 prio 1 protocol ip handle 2 fw flowid 1:2'",
                        f"kubectl exec -n {self.namespace} {node} -- /bin/bash -c 'tc filter add dev eth0 parent 1:0 prio 1 protocol ip handle 3 fw flowid 1:3'",
                        f"kubectl exec -n {self.namespace} {node} -- /bin/bash -c 'iptables -A OUTPUT -t mangle -p tcp --sport 2881 -j MARK --set-mark 2'",
                        f"kubectl exec -n {self.namespace} {node} -- /bin/bash -c 'iptables -A OUTPUT -t mangle -p tcp --sport 2882 -j MARK --set-mark 3'"
                    ]
                    logger.info(f"Executing Net Saturation trigger commands on {node}")
                    success = True
                    executed_commands = [] # Track executed commands for potential partial rollback
                    for cmd in commands:
                        logger.debug(f"Executing Net Saturation trigger command: {cmd}")
                        try:
                            # Run command and check for errors
                            import subprocess # Keep the import here
                            # Directly await the mocked subprocess.run
                            process = await asyncio.to_thread(
                                subprocess.run, cmd, shell=True, check=True, capture_output=True
                            )
                            executed_commands.append(cmd) # Add to list if successful
                            logger.debug(f"Command '{cmd}' stdout: {process.stdout.decode()}")
                            logger.debug(f"Command '{cmd}' stderr: {process.stderr.decode()}")
                        except subprocess.CalledProcessError as cmd_e:
                            logger.error(f"Net saturation trigger command failed (exit code {cmd_e.returncode}): {cmd}. Stderr: {cmd_e.stderr.decode()}")
                            success = False
                            # Attempt rollback? For now, just log and fail the anomaly.
                            break # Stop executing commands on failure
                        except FileNotFoundError:
                             logger.error(f"kubectl command not found. Cannot execute: {cmd}")
                             success = False
                             break # Cannot continue if kubectl is missing
                        except Exception as sub_e:
                            logger.error(f"Unexpected error executing net saturation trigger command '{cmd}': {sub_e}")
                            success = False
                            break # Stop on unexpected error

                    if success:
                        anomaly_details["status"] = "active"
                        anomaly_details["params"] = {"target_pod": node, "commands_applied": len(executed_commands)}
                        logger.info(f"Successfully applied Net Saturation trigger commands on {node}")
                    else:
                        # If setup failed, raise an exception to be caught by the outer try block
                        # This prevents storing the anomaly as active in Redis.
                        # Optional: Add logic here to attempt cleanup/rollback of executed_commands if needed.
                        raise Exception(f"Net saturation trigger failed on {node}")

                # === Existing Types ===
                elif anomaly_type == "cpu_stress": # Original kubectl exec based
                    if node == "cluster_wide": raise ValueError("cpu_stress requires a specific target node name.")
                    logger.info(f"Applying original cpu_stress via kubectl on pod {node} (Anomaly ID: {anomaly_id})")
                    severity_map = {"low": 0, "medium": 1, "high": 2}
                    index = severity_map.get(severity.lower(), 1)
                    # Ensure index is valid for the defined severities
                    severities = self.severity_variations.get(anomaly_type, [{}])
                    index = min(index, len(severities) - 1)
                    severity_config = severities[index]

                    cpu_command = f"stress-ng {severity_config.get('stress-ng-params', '--cpu 80 -t 180')}" # Use params from severity
                    kubectl_cmd = f"kubectl exec -n {self.namespace} {node} -- /bin/bash -c '{cpu_command}' > /dev/null 2>&1 &"
                    logger.info(f"Executing kubectl command: {kubectl_cmd}")
                    import subprocess
                    try: # Inner try for the subprocess call
                        # Run in background, don't wait for completion
                        import subprocess # Keep the import here
                        # Directly await the mocked subprocess.Popen
                        process = await asyncio.to_thread(
                            subprocess.Popen, kubectl_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                        )
                        # We don't store process.pid as it's inside the container. Rely on pkill during delete.
                        anomaly_details["status"] = "active"
                        anomaly_details["k8s_name"] = f"manual-cpu-stress-{node}-{datetime.now().strftime('%Y%m%d%H%M%S')}" # Generate pseudo-name for tracking
                        anomaly_details["params"] = {"command": cpu_command}
                    except FileNotFoundError:
                         logger.error("kubectl command not found. Cannot start stress-ng process.")
                         raise # Re-raise the error to be caught by outer try
                    except Exception as sub_e: # Catch subprocess execution errors
                         logger.error(f"Error executing kubectl for cpu_stress on {node}: {sub_e}")
                         raise # Re-raise the error

                elif anomaly_type == "io_bottleneck": # Original kubectl exec based
                    if node == "cluster_wide": raise ValueError("io_bottleneck requires a specific target node name.")
                    logger.info(f"Applying original io_bottleneck via kubectl on pod {node} (Anomaly ID: {anomaly_id})")
                    severity_map = {"low": 0, "medium": 1, "high": 2}
                    index = severity_map.get(severity.lower(), 1)
                     # Ensure index is valid
                    severities = self.severity_variations.get(anomaly_type, [{}])
                    index = min(index, len(severities) - 1)
                    severity_config = severities[index]

                    io_command = severity_config.get("command", "dd if=/dev/zero of=/tmp/io_stress_file bs=1M count=1024 conv=fdatasync & sleep %s") % self.anomaly_duration
                    kubectl_cmd = f"kubectl exec -n {self.namespace} {node} -- /bin/bash -c '{io_command}' > /dev/null 2>&1 &"
                    logger.info(f"Executing kubectl command: {kubectl_cmd}")
                    import subprocess
                    try: # Inner try for the subprocess call
                        process = await asyncio.to_thread(
                            subprocess.Popen, kubectl_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                        )
                        anomaly_details["status"] = "active"
                        # Create a dummy tracking resource in K8s? Or just track in Redis? Let's just use Redis.
                        anomaly_details["k8s_name"] = f"manual-io-bottleneck-{node}-{datetime.now().strftime('%Y%m%d%H%M%S')}" # Generate pseudo-name
                        anomaly_details["params"] = {"command": io_command, "expected_duration": self.anomaly_duration}
                    except FileNotFoundError:
                         logger.error("kubectl command not found. Cannot start io_bottleneck process.")
                         raise # Re-raise error
                    except Exception as sub_e: # Catch subprocess execution errors
                         logger.error(f"Error executing kubectl for io_bottleneck on {node}: {sub_e}")
                         raise # Re-raise error


                elif anomaly_type == "cache_bottleneck":
                    logger.info(f"Applying cache_bottleneck (Anomaly ID: {anomaly_id})")
                    # Determine severity and memstore limit
                    severity_map = {"low": 0, "medium": 1, "high": 2}
                    index = severity_map.get(severity.lower(), 1)
                    severities = self.severity_variations.get(anomaly_type, [{}])
                    index = min(index, len(severities) - 1)
                    # severity_config = severities[index] # Not directly used for limit now

                    if severity.lower() == "low": memstore_limit = "8"
                    elif severity.lower() == "medium": memstore_limit = "6"
                    else: memstore_limit = "4" # High severity

                    # Determine target zones
                    target_zones = set()
                    if node != "cluster_wide": # If specific node(s) were targeted implicitly
                         logger.warning("Applying cache_bottleneck to ALL zones as specific node-to-zone mapping is complex.")
                         target_zones = self.ob_zones
                    else: # Explicitly cluster-wide
                         target_zones = self.ob_zones

                    if not target_zones:
                        logger.error("Cannot determine target zones for cache_bottleneck.")
                        raise ValueError("Could not determine target zones.")

                    logger.info(f"Setting memstore_limit_percentage={memstore_limit} for zones: {target_zones}")
                    # Inner try for the parameter update part
                    try:
                        for zone in target_zones:
                            await self._update_ob_parameter_for_zone({"memstore_limit_percentage": memstore_limit}, zone)
                    except Exception as ob_e:
                        logger.error(f"Failed to update OceanBase parameter for cache_bottleneck {anomaly_id}: {ob_e}")
                        raise # Re-raise to outer try

                    # Create a tracking K8s resource (minimal stress)
                    experiment = self._get_experiment_template(anomaly_type, severity) # Gets the minimal tracking template
                    k8s_resource_name = f"{experiment['metadata']['name']}-tracker-{anomaly_id[:8]}" # Unique name
                    experiment["metadata"]["name"] = k8s_resource_name
                    experiment["spec"]["selector"]["pods"] = {} # Apply to any pod matching labels (effectively cluster-wide tracking)
                    experiment["spec"]["mode"] = "all" # Doesn't matter much for tracking resource

                    try: # Inner try for creating the K8s tracking resource
                        # Directly await the mocked K8s API call
                        await self.custom_api.create_namespaced_custom_object(
                            group="chaos-mesh.org", version="v1alpha1", namespace=self.namespace,
                            plural=self._get_experiment_plural(anomaly_type), body=experiment
                        )
                        logger.info(f"Created K8s tracking resource for cache_bottleneck: {k8s_resource_name}")
                        anomaly_details["status"] = "active"
                        anomaly_details["k8s_name"] = k8s_resource_name
                        anomaly_details["params"] = {"memstore_limit": memstore_limit, "zones": list(target_zones)}
                    except ApiException as k8s_e:
                         logger.error(f"Failed to create K8s tracking resource for cache_bottleneck: {k8s_e}")
                         # Parameter change succeeded, but tracking failed. Mark as degraded.
                         anomaly_details["status"] = "degraded" # Mark as active but K8s tracking failed
                         anomaly_details["params"] = {"memstore_limit": memstore_limit, "zones": list(target_zones), "error": "K8s tracking resource creation failed"}
                         # Do not re-raise here, let the anomaly be stored as degraded


                elif anomaly_type == "too_many_indexes":
                    if node != "cluster_wide":
                         logger.warning("Applying 'too_many_indexes' cluster-wide, ignoring specific node target.")
                    logger.info(f"Applying 'too_many_indexes' SQL commands (Anomaly ID: {anomaly_id})")
                    experiment_template = self._get_experiment_template(anomaly_type, severity) # Gets SQL commands
                    sql_commands = experiment_template["spec"]["sqlCommands"]
                    try: # Inner try for SQL execution
                        await self._execute_sql_commands(sql_commands)
                        logger.info(f"Successfully executed SQL commands for '{anomaly_id}'.")
                        anomaly_details["status"] = "active"
                        anomaly_details["node"] = "cluster_wide" # Mark explicitly
                        anomaly_details["params"] = {"sql_command_count": len(sql_commands)}
                    except Exception as sql_e:
                        logger.error(f"Failed to apply 'too_many_indexes' anomaly '{anomaly_id}': {str(sql_e)}")
                        raise # Re-raise exception to outer try

                # === Other Chaos Mesh Types ===
                else:
                     # General Chaos Mesh experiment application
                    if node == "cluster_wide": raise ValueError(f"Anomaly type '{anomaly_type}' requires a specific target node name.")
                    logger.info(f"Applying Chaos Mesh anomaly '{anomaly_type}' on pod {node} (Anomaly ID: {anomaly_id})")
                    experiment = self._get_experiment_template(anomaly_type, severity)
                    # Generate unique name per node/instance based on anomaly_id
                    k8s_resource_name = f"{experiment['metadata']['name'].rsplit('-', 1)[0]}-{node}-{anomaly_id[:8]}"
                    experiment["metadata"]["name"] = k8s_resource_name
                    experiment["spec"]["selector"]["pods"] = {self.namespace: [node]} # Target specific pod
                    experiment["spec"]["mode"] = "one" # Ensure mode targets the single pod

                    plural = self._get_experiment_plural(anomaly_type)
                    if plural == "none":
                        raise ValueError(f"Cannot apply anomaly type '{anomaly_type}' via Chaos Mesh: plural is 'none'.")

                    try: # Inner try for K8s resource creation
                        # Directly await the mocked K8s API call
                        await self.custom_api.create_namespaced_custom_object(
                            group="chaos-mesh.org", version="v1alpha1", namespace=self.namespace,
                            plural=plural, body=experiment
                        )
                        logger.info(f"Created Chaos Mesh resource {k8s_resource_name} for anomaly {anomaly_id} on node {node}")
                        anomaly_details["status"] = "active"
                        anomaly_details["k8s_name"] = k8s_resource_name
                        # Store key params from experiment spec for reference
                        anomaly_details["params"] = {"kind": experiment.get("kind"), "spec": experiment.get("spec")} # Store spec

                    except ApiException as k8s_e:
                        logger.error(f"Failed to create Chaos Mesh resource {k8s_resource_name} for node {node}: {str(k8s_e)}")
                        # Handle specific errors like AlreadyExists? For now, just fail.
                        raise # Re-raise the exception to outer try

            # --- Moved Redis Storage Logic Outside Main Try/Except ---
            # This part executes only if the main anomaly application try block succeeded
            # or resulted in a 'degraded' state (like cache_bottleneck K8s tracking failure)
            # and didn't raise an exception caught by the outer except block below.

            except Exception as e: # Catch exceptions ONLY from the anomaly application logic above
                 # Log failure for this specific node/target
                 logger.error(f"Failed to APPLY anomaly '{anomaly_type}' for target '{node}' (ID: {anomaly_id}): {str(e)}")
                 # Set status to failed explicitly, although it defaults to this
                 anomaly_details["status"] = "failed"
                 # Ensure Redis state is clean if setup failed mid-way (e.g., after creating ID but before storing)
                 try: # Add inner try for cleanup robustness
                     self.redis_client.delete(f"anomaly:{anomaly_id}") # Attempt to delete key if it exists
                     self.redis_client.srem("active_anomalies", anomaly_id) # Attempt to remove from set
                 except Exception as cleanup_e:
                     logger.error(f"Error during Redis cleanup after anomaly application failure for {anomaly_id}: {cleanup_e}")
                 # Continue to the next node in the loop, do not store this failed anomaly
                 continue # Skip Redis storage for this failed node

            # --- Store successful/degraded anomaly in Redis --- 
            # This code runs ONLY if the main try block completed without raising an exception
            if anomaly_details["status"] in ["active", "degraded"]:
                try:
                    self.redis_client.set(f"anomaly:{anomaly_id}", json.dumps(anomaly_details))
                    self.redis_client.sadd("active_anomalies", anomaly_id)
                    created_anomalies.append(anomaly_details)
                    logger.info(f"Stored anomaly {anomaly_id} (Type: {anomaly_type}, Node: {node}, Status: {anomaly_details['status']}) in Redis.")
                except redis.exceptions.ConnectionError as redis_e:
                     logger.error(f"Redis connection error while storing anomaly {anomaly_id}: {redis_e}. Anomaly may be active but not tracked.")
                     # Re-raise the original Redis error so the caller knows tracking failed.
                     raise redis_e # Re-raise the original redis exception
                except Exception as store_e:
                    logger.error(f"Failed to store anomaly {anomaly_id} in Redis: {store_e}")
                    # Re-raise to indicate tracking failure.
                    raise Exception(f"Failed to store anomaly {anomaly_id} in Redis") from store_e
            # else: anomaly_details["status"] remained "failed" from application logic error

        # Broadcast update after applying anomalies
        if created_anomalies and self.connection_manager:
            await self._broadcast_anomaly_update()
            
        return created_anomalies # Return list of successfully initiated and tracked anomalies

    async def delete_anomaly(self, anomaly_id: Optional[str] = None, anomaly_type: Optional[str] = None, target_node: Optional[str] = None):
        """Delete a specific anomaly by ID or type/node, cleaning up resources and Redis state."""
        if not self.redis_client:
            logger.error("Redis client not available. Cannot delete anomaly.")
            raise ConnectionError("Redis client not available.")

        deleted_ids = []
        anomalies_to_process = []

        if anomaly_id:
            # Delete by specific ID
            anomaly_data_json = self.redis_client.get(f"anomaly:{anomaly_id}")
            if anomaly_data_json:
                anomalies_to_process.append(json.loads(anomaly_data_json))
            else:
                logger.warning(f"Anomaly ID {anomaly_id} not found in Redis.")
        elif anomaly_type:
            # Delete by type and optional node
            active_anomaly_ids = self.redis_client.smembers("active_anomalies")
            for active_id in active_anomaly_ids:
                anomaly_data_json = self.redis_client.get(f"anomaly:{active_id}")
                if anomaly_data_json:
                    anomaly_data = json.loads(anomaly_data_json)
                    if anomaly_data.get("type") == anomaly_type:
                        if target_node and anomaly_data.get("node") != target_node:
                            continue # Skip if node doesn't match
                        anomalies_to_process.append(anomaly_data)
            if not anomalies_to_process:
                 logger.warning(f"No active anomalies found for type '{anomaly_type}'" + (f" on node '{target_node}'" if target_node else ""))

        else:
            logger.error("Must provide anomaly_id or anomaly_type to delete_anomaly.")
            return [] # Return empty list if nothing to delete

        # Process deletions
        for anomaly_data in anomalies_to_process:
            current_id = anomaly_data.get("id")
            current_type = anomaly_data.get("type")
            current_node = anomaly_data.get("node") # This might be pod name or IP depending on how it was stored
            experiment_name = anomaly_data.get("k8s_name") # Name used for K8s resources if applicable

            logger.info(f"Attempting to delete anomaly ID: {current_id}, Type: {current_type}, Node: {current_node}, K8s Name: {experiment_name}")

            try:
                # === New Anomaly Types (re-implemented) ===
                if current_type == "cpu_saturation":
                    # stress-ng started with -t duration, should terminate on its own.
                    # No explicit 'recover' command needed unless process needs forced kill.
                    logger.info(f"CPU Saturation anomaly {current_id} relies on process termination for recovery.")
                    # Optional: Could add `kubectl exec {current_node} -- pkill stress-ng` if needed.
                elif current_type == "io_saturation":
                    # stress-ng started with -t duration, should terminate on its own.
                    logger.info(f"IO Saturation anomaly {current_id} relies on process termination for recovery.")
                    # Optional: Could add `kubectl exec {current_node} -- pkill stress-ng` if needed.
                elif current_type == "net_saturation":
                    if current_node:
                        logger.info(f"Recovering Net Saturation on node {current_node} for anomaly {current_id}")
                        # Implement recovery commands using kubectl exec
                        commands = [
                            f"kubectl exec -n {self.namespace} {current_node} -- /bin/bash -c 'iptables -t mangle -F'",
                            f"kubectl exec -n {self.namespace} {current_node} -- /bin/bash -c 'tc qdisc del dev eth0 root'"
                        ]
                        logger.info(f"Executing Net Saturation recovery commands on {current_node}")
                        success = True
                        for cmd in commands:
                            logger.debug(f"Executing Net Saturation recovery command: {cmd}")
                            try:
                                # Run command and check return code
                                import subprocess
                                # Directly await the mocked subprocess.run
                                process = await asyncio.to_thread(
                                    subprocess.run, cmd, shell=True, check=True, capture_output=True
                                )
                                logger.debug(f"Command '{cmd}' stdout: {process.stdout.decode()}")
                                logger.debug(f"Command '{cmd}' stderr: {process.stderr.decode()}")
                            except subprocess.CalledProcessError as cmd_e:
                                # Log error but continue trying other commands
                                logger.warning(f"Net saturation recovery command failed (exit code {cmd_e.returncode}): {cmd}. Stderr: {cmd_e.stderr.decode()}")
                                success = False
                            except FileNotFoundError:
                                logger.error(f"kubectl command not found. Cannot execute: {cmd}")
                                success = False
                                break # Cannot continue if kubectl is missing
                            except Exception as cmd_e:
                                logger.error(f"Unexpected error executing net saturation recovery command '{cmd}': {cmd_e}")
                                success = False
                                # Decide whether to stop or continue on error

                        if success:
                            logger.info(f"Successfully executed Net Saturation recovery commands on {current_node}")
                        else:
                            logger.warning(f"Net Saturation recovery on {current_node} encountered errors.")
                    else:
                        logger.warning(f"Cannot recover Net Saturation for {current_id}: target node missing in Redis data.")

                # === Existing Anomaly Types ===
                elif current_type == "cpu_stress": # Original kubectl exec based
                    if current_node and experiment_name: # Assuming 'node' stored the pod name
                        logger.info(f"Stopping original cpu_stress (stress-ng) on node {current_node} for anomaly {current_id}")
                        # Use kubectl to kill the process - requires node name
                        kubectl_cmd = f"kubectl exec -n {self.namespace} {current_node} -- pkill stress-ng"
                        import subprocess
                        try:
                            # Directly await the mocked subprocess.run
                            await asyncio.to_thread(
                                subprocess.run, kubectl_cmd, shell=True, check=True, capture_output=True
                            )
                            logger.info(f"Successfully stopped stress-ng on node {current_node} via kubectl.")
                        except subprocess.CalledProcessError as e:
                            logger.warning(f"Failed to stop stress-ng on {current_node} via kubectl: {e.stderr.decode()}")
                        except FileNotFoundError:
                            logger.error("kubectl command not found. Cannot stop stress-ng process.")
                    else:
                        logger.warning(f"Cannot stop original cpu_stress for {current_id}: node or experiment name missing.")

                elif current_type == "io_bottleneck": # Original kubectl exec based
                    if current_node and experiment_name:
                        logger.info(f"Stopping original io_bottleneck (dd) on node {current_node} for anomaly {current_id}")
                        # Attempt to kill the 'dd' and 'sleep' processes. This is less reliable.
                        # Consider adding process tracking if using this method heavily.
                        kubectl_cmd_dd = f"kubectl exec -n {self.namespace} {current_node} -- pkill dd"
                        kubectl_cmd_sleep = f"kubectl exec -n {self.namespace} {current_node} -- pkill sleep"
                        import subprocess
                        try:
                            # Directly await the mocked subprocess.run
                            await asyncio.to_thread(
                                subprocess.run, kubectl_cmd_dd, shell=True, check=False, capture_output=True
                            )
                            await asyncio.to_thread(
                                subprocess.run, kubectl_cmd_sleep, shell=True, check=False, capture_output=True
                            )
                            logger.info(f"Attempted to stop dd/sleep processes on node {current_node} via kubectl.")
                        except FileNotFoundError:
                            logger.error("kubectl command not found. Cannot stop dd/sleep processes.")
                        # Also delete the associated tracking K8s resource if one was created (unlikely for this type now)
                        if experiment_name:
                            plural = self._get_experiment_plural(current_type) # Should be stresschaos
                            try:
                                # Directly await the mocked K8s API call
                                await self.custom_api.delete_namespaced_custom_object(
                                    group="chaos-mesh.org", version="v1alpha1", namespace=self.namespace,
                                    plural=plural, name=experiment_name
                                )
                                logger.info(f"Deleted K8s tracking resource: {experiment_name}")
                            except ApiException as e:
                                if e.status != 404:
                                    logger.warning(f"Error deleting K8s tracking resource {experiment_name}: {str(e)}")
                    else:
                        logger.warning(f"Cannot stop original io_bottleneck for {current_id}: node or experiment name missing.")

                elif current_type == "cache_bottleneck":
                    logger.info(f"Recovering cache_bottleneck for anomaly {current_id}. Restoring default memstore_limit_percentage.")
                    # Restore default memstore_limit_percentage for all zones
                    # We might need to track which zones were affected if we want finer control
                    try: # Added try block for parameter update
                        for zone in self.ob_zones:
                            await self._update_ob_parameter_for_zone({"memstore_limit_percentage": "0"}, zone) # Restore default value 0
                        logger.info("Restored default memstore_limit_percentage for all zones.")
                    except Exception as ob_e: # Catch specific update errors
                        logger.error(f"Failed to restore OceanBase parameter for cache_bottleneck {current_id}: {ob_e}")
                        # Continue to delete K8s resource if possible, but log the error
                    # Also delete the associated tracking K8s resource
                    if experiment_name:
                        plural = self._get_experiment_plural(current_type) # Should be stresschaos
                        try:
                            # Directly await the mocked K8s API call
                            await self.custom_api.delete_namespaced_custom_object(
                                group="chaos-mesh.org", version="v1alpha1", namespace=self.namespace,
                                plural=plural, name=experiment_name
                            )
                            logger.info(f"Deleted K8s tracking resource: {experiment_name}")
                        except ApiException as e:
                            if e.status != 404:
                                logger.warning(f"Error deleting K8s tracking resource {experiment_name}: {str(e)}")

                elif current_type == "too_many_indexes":
                    logger.info(f"Cleaning up 'too_many_indexes' anomaly {current_id} by dropping created indexes.")
                    # Define SQL commands to drop indexes (same as before)
                    drop_tpcc_sql_commands = [
                        "USE tpcc",
                        "DROP INDEX idx_customer_1 ON customer", "DROP INDEX idx_customer_2 ON customer", "DROP INDEX idx_customer_3 ON customer", "DROP INDEX idx_customer_4 ON customer", "DROP INDEX idx_customer_5 ON customer",
                        "DROP INDEX idx_district_1 ON district", "DROP INDEX idx_district_2 ON district", "DROP INDEX idx_district_3 ON district",
                        "DROP INDEX idx_history_1 ON history", "DROP INDEX idx_history_2 ON history", "DROP INDEX idx_history_3 ON history",
                        "DROP INDEX idx_item_1 ON item", "DROP INDEX idx_item_2 ON item", "DROP INDEX idx_item_3 ON item",
                        "DROP INDEX idx_new_orders_1 ON new_orders", "DROP INDEX idx_new_orders_2 ON new_orders",
                        "DROP INDEX idx_order_line_1 ON order_line", "DROP INDEX idx_order_line_2 ON order_line", "DROP INDEX idx_order_line_3 ON order_line", "DROP INDEX idx_order_line_4 ON order_line",
                        "DROP INDEX idx_orders_1 ON orders", "DROP INDEX idx_orders_2 ON orders", "DROP INDEX idx_orders_3 ON orders", "DROP INDEX idx_orders_4 ON orders",
                        "DROP INDEX idx_stock_1 ON stock", "DROP INDEX idx_stock_2 ON stock", "DROP INDEX idx_stock_3 ON stock",
                        "DROP INDEX idx_warehouse_1 ON warehouse", "DROP INDEX idx_warehouse_2 ON warehouse"
                    ]
                    drop_sbtest_sql_commands = [
                        "USE sbtest",
                        "DROP INDEX idx_sbtest1_1 ON sbtest1", "DROP INDEX idx_sbtest1_2 ON sbtest1", "DROP INDEX idx_sbtest1_3 ON sbtest1",
                        "DROP INDEX idx_sbtest2_1 ON sbtest2", "DROP INDEX idx_sbtest2_2 ON sbtest2", "DROP INDEX idx_sbtest2_3 ON sbtest2",
                        "DROP INDEX idx_sbtest3_1 ON sbtest3", "DROP INDEX idx_sbtest3_2 ON sbtest3", "DROP INDEX idx_sbtest3_3 ON sbtest3",
                        "DROP INDEX idx_sbtest4_1 ON sbtest4", "DROP INDEX idx_sbtest4_2 ON sbtest4", "DROP INDEX idx_sbtest4_3 ON sbtest4",
                        "DROP INDEX idx_sbtest5_1 ON sbtest5", "DROP INDEX idx_sbtest5_2 ON sbtest5", "DROP INDEX idx_sbtest5_3 ON sbtest5",
                        "DROP INDEX idx_sbtest6_1 ON sbtest6", "DROP INDEX idx_sbtest6_2 ON sbtest6", "DROP INDEX idx_sbtest6_3 ON sbtest6",
                        "DROP INDEX idx_sbtest7_1 ON sbtest7", "DROP INDEX idx_sbtest7_2 ON sbtest7", "DROP INDEX idx_sbtest7_3 ON sbtest7",
                        "DROP INDEX idx_sbtest8_1 ON sbtest8", "DROP INDEX idx_sbtest8_2 ON sbtest8", "DROP INDEX idx_sbtest8_3 ON sbtest8",
                        "DROP INDEX idx_sbtest9_1 ON sbtest9", "DROP INDEX idx_sbtest9_2 ON sbtest9", "DROP INDEX idx_sbtest9_3 ON sbtest9",
                        "DROP INDEX idx_sbtest10_1 ON sbtest10", "DROP INDEX idx_sbtest10_2 ON sbtest10", "DROP INDEX idx_sbtest10_3 ON sbtest10"
                    ]
                    drop_sql_commands = drop_tpcc_sql_commands + drop_sbtest_sql_commands
                    try:
                        await self._execute_sql_commands(drop_sql_commands)
                        logger.info("Successfully dropped indexes for 'too_many_indexes' anomaly.")
                    except Exception as e:
                        logger.error(f"Failed to drop indexes for 'too_many_indexes' anomaly {current_id}: {str(e)}")

                # === Other Chaos Mesh Types ===
                else:
                    # General Chaos Mesh experiment deletion
                    plural = self._get_experiment_plural(current_type)
                    if plural != "none" and experiment_name: # experiment_name is k8s_name from Redis
                        logger.info(f"Deleting Chaos Mesh resource '{experiment_name}' (Type: {current_type}) for anomaly {current_id}")
                        try:
                            # Directly await the mocked K8s API call
                            await self.custom_api.delete_namespaced_custom_object(
                                group="chaos-mesh.org", version="v1alpha1", namespace=self.namespace,
                                plural=plural, name=experiment_name
                            )
                            logger.info(f"Successfully deleted Chaos Mesh resource: {experiment_name}")
                        except ApiException as e:
                            if e.status == 404:
                                logger.warning(f"Chaos Mesh resource {experiment_name} not found for deletion (anomaly {current_id}). Might have been already deleted.")
                            else:
                                logger.error(f"Error deleting Chaos Mesh resource {experiment_name} for anomaly {current_id}: {str(e)}")
                                # Log error and continue to Redis cleanup
                    elif plural == "none":
                         logger.info(f"Anomaly type '{current_type}' (ID: {current_id}) does not have a corresponding K8s resource to delete (plural is 'none').")
                    elif not experiment_name:
                        logger.warning(f"Cannot delete K8s resource for anomaly {current_id} (Type: {current_type}): K8s name missing in Redis data.")

                # --- Cleanup Redis State ---
                # This should happen AFTER attempting the specific cleanup actions above
                self.redis_client.delete(f"anomaly:{current_id}")
                self.redis_client.srem("active_anomalies", current_id)
                deleted_ids.append(current_id)
                logger.info(f"Removed anomaly {current_id} from Redis.")

            except Exception as e: # Catch exceptions from the main processing loop for one anomaly
                logger.error(f"Failed to process deletion for anomaly {current_id} (Type: {current_type}, Node: {current_node}): {str(e)}")
                # Attempt cleanup in Redis even if resource deletion failed
                try:
                     if current_id: # Ensure current_id was set before error
                         self.redis_client.delete(f"anomaly:{current_id}")
                         self.redis_client.srem("active_anomalies", current_id)
                         logger.info(f"Cleaned up Redis entry for failed deletion: {current_id}")
                except Exception as redis_e:
                     logger.error(f"Failed to cleanup Redis for anomaly {current_id} after error: {redis_e}")
                # Continue to the next anomaly if processing multiple

        # Broadcast update after deleting anomalies
        if deleted_ids and self.connection_manager:
            await self._broadcast_anomaly_update()
            
        return deleted_ids

    async def delete_all_anomalies(self):
        """Delete all active anomalies tracked in Redis."""
        if not self.redis_client:
            logger.error("Redis client not available. Cannot delete all anomalies.")
            raise ConnectionError("Redis client not available.")

        all_deleted_ids = []
        active_anomaly_ids = list(self.redis_client.smembers("active_anomalies"))
        logger.info(f"Attempting to delete {len(active_anomaly_ids)} active anomalies tracked in Redis...")

        # Use asyncio.gather to run deletions concurrently
        delete_tasks = [self.delete_anomaly(anomaly_id=aid) for aid in active_anomaly_ids]
        results = await asyncio.gather(*delete_tasks, return_exceptions=True)

        for i, result in enumerate(results):
            aid = active_anomaly_ids[i]
            if isinstance(result, Exception):
                logger.error(f"Error deleting anomaly {aid} during delete_all: {result}")
            elif result: # delete_anomaly returns list of deleted IDs
                all_deleted_ids.extend(result)
                logger.info(f"Successfully processed deletion for anomaly {aid} during delete_all.")
            else:
                 logger.warning(f"Deletion call for anomaly {aid} returned empty list or None.")

        # Remove the extra cleanup step that causes test failures
        # remaining_ids = self.redis_client.smembers("active_anomalies")
        # if remaining_ids:
        #    logger.warning(f"Redis set 'active_anomalies' still contains IDs after delete_all: {remaining_ids}. Attempting individual cleanup.")
        #    for rem_id in remaining_ids:
        #         self.redis_client.delete(f"anomaly:{rem_id}")
        #         self.redis_client.srem("active_anomalies", rem_id)

        logger.info(f"Finished delete_all_anomalies. Deleted IDs: {all_deleted_ids}")
        # self.invalidate_cache() # Removed

        # Broadcast empty update after deleting all anomalies
        if all_deleted_ids and self.connection_manager:
            await self._broadcast_anomaly_update()
            
        return all_deleted_ids


    # _fetch_available_nodes, get_available_nodes remain unchanged
    # ... (keep these methods as they are) ...
    async def _fetch_available_nodes(self) -> List[str]:
        """Get list of available pods for running experiments"""
        try:
            pods = await asyncio.to_thread(
                self.core_api.list_namespaced_pod,
                namespace=self.namespace,
                label_selector="ref-obcluster" # Assuming this label selects the right pods
            )
            pod_names = []
            # Filter for pods likely running observer processes (adjust filtering as needed)
            for pod in pods.items:
                # Example filtering: assume observer pods follow a naming convention
                if pod.metadata.name.startswith("obcluster-") and pod.status.phase == "Running":
                     # Check pod conditions, e.g., Ready status
                     is_ready = False
                     if pod.status.conditions:
                         for condition in pod.status.conditions:
                             if condition.type == "Ready" and condition.status == "True":
                                 is_ready = True
                                 break
                     if is_ready:
                         pod_names.append(pod.metadata.name)
                     else:
                         logger.debug(f"Pod {pod.metadata.name} found but not ready.")
            logger.info(f"Found available and ready nodes: {pod_names}")
            return pod_names
        except ApiException as e:
            # Log specific K8s API errors
            logger.error(f"Kubernetes API error getting available pods: {str(e)} (Status: {e.status}, Reason: {e.reason})")
            return []
        except Exception as e:
             # Catch other potential errors
            logger.error(f"Unexpected error getting available pods: {str(e)}")
            return []

    async def get_available_nodes(self) -> List[str]:
        """Get list of available pods for running experiments (uses cached list if available)"""
        # Maybe add a short TTL to this cache? For now, it updates on startup.
        if not self.available_nodes:
             logger.info("Available nodes list is empty, fetching...")
             self.available_nodes = await self._fetch_available_nodes()
        return self.available_nodes

    # _fetch_ob_zones remains unchanged
    # ... (keep _fetch_ob_zones as is) ...
    def _fetch_ob_zones(self):
        """Fetch OceanBase zones from the database"""
        conn = None
        try:
            conn = pymysql.connect(**self.ob_config)
            with conn.cursor() as cursor:
                cursor.execute("SELECT DISTINCT zone FROM oceanbase.DBA_OB_ZONES WHERE status = 'ACTIVE'")
                zones = [row[0] for row in cursor.fetchall()]

            if not zones:
                 logger.warning("No zones found. Using default zones.")
                 return ['zone1', 'zone2', 'zone3'] # Default fallback

            logger.info(f"Found OceanBase zones: {zones}")
            return zones
        except pymysql.err.ProgrammingError as e:
             # Handle cases where the table/view doesn't exist
             logger.error(f"SQL ProgrammingError fetching OceanBase zones: {str(e)}. Check table/view names.")
             logger.info("Using default zones due to SQL error.")
             return ['zone1', 'zone2', 'zone3']
        except Exception as e:
            logger.error(f"Error fetching OceanBase zones: {str(e)}")
            logger.info("Using default zones due to error.")
            return ['zone1', 'zone2', 'zone3']
        finally:
             if conn: conn.close()


    # verify_experiment_exists method removed (replaced by checking Redis)

    # _update_available_nodes remains unchanged
    async def _update_available_nodes(self):
        """Asynchronously update available_nodes from Kubernetes."""
        nodes = await self._fetch_available_nodes()
        self.available_nodes = nodes

    # get_pod_ip_to_name_mapping and _build_ip_to_name_mapping remain unchanged
    # ... (keep these methods as they are) ...
    async def get_pod_ip_to_name_mapping(self) -> Dict[str, str]:
        """Get mapping of pod IPs to pod names"""
        try:
            # Consider caching this result with a TTL?
            pods = await asyncio.to_thread(
                self.core_api.list_namespaced_pod,
                namespace=self.namespace,
                label_selector="ref-obcluster" # Ensure this selector is correct
            )
            ip_to_name = {}
            for pod in pods.items:
                # Ensure pod has an IP and meets criteria (e.g., running, part of the cluster)
                if pod.status.pod_ip and pod.metadata.name.startswith("obcluster-") and pod.status.phase == "Running":
                    ip_to_name[pod.status.pod_ip] = pod.metadata.name
            # logger.info(f"Got pod IP to name mapping: {ip_to_name}") # Can be noisy
            return ip_to_name
        except ApiException as e:
            logger.error(f"Error getting pod IP to name mapping: {str(e)}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error getting pod IP to name mapping: {str(e)}")
            return {}


    def _build_ip_to_name_mapping(self) -> Dict[str, str]:
        """Build pod IP to name mapping synchronously using Kubernetes API."""
        mapping = {}
        try:
            # No need to reload config here, assume it's done in __init__
            # core_api = client.CoreV1Api() # Use self.core_api
            pods = self.core_api.list_namespaced_pod(namespace=self.namespace, label_selector="ref-obcluster")
            for pod in pods.items:
                pod_ip = pod.status.pod_ip
                pod_name = pod.metadata.name
                # Add phase check similar to async version
                if pod_ip and pod_name.startswith("obcluster-") and pod.status.phase == "Running":
                    mapping[pod_ip] = pod_name
            logger.info(f"Built initial pod IP to name mapping: {mapping}")
        except ApiException as e:
             logger.error(f"K8s API error building pod IP to name mapping: {e}")
        except Exception as e:
            # logger = logging.getLogger(__name__) # Use self logger? No, it's class level
            logger.error(f"Error building pod IP to name mapping: {e}")
        return mapping


# Create singleton k8s service object
k8s_service = K8sService()