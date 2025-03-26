from dotenv import load_dotenv
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import logging
import os
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import asyncio
import time
from functools import lru_cache, wraps
from async_lru import alru_cache
import json
import pymysql
import threading

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
        self.active_anomalies = {}  # Track active anomalies with timestamps
        self._experiment_cache = {}  # Cache for experiment status
        self.CACHE_TTL = 5  # Cache TTL in seconds
        self._last_cache_update = None
        self._last_request_time = 0  # This will force a refresh on next request
        self.collection_duration = 180  # Collection duration in seconds
        self.experiment_types = ["cpu_stress", "io_bottleneck", "network_bottleneck", "cache_bottleneck", "too_many_indexes"]
        self.available_nodes = []
        try:
            self.available_nodes = loop.run_until_complete(self._fetch_available_nodes())
        except RuntimeError:
            # If there's no running event loop
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                self.available_nodes = new_loop.run_until_complete(self._fetch_available_nodes())
            finally:
                new_loop.close()
        except Exception as e:
            logger.error(f"Error getting available nodes: {e}")
            self.available_nodes = []
        # Force immediate refresh of active anomalies on startup
        self._last_cache_update = 0
        # Create event loop for initialization if none exists
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Use synchronous methods to avoid async in __init__
            # We'll schedule the initial cache refresh to happen soon after startup
            threading.Thread(target=self._initialize_active_anomalies).start()
        except Exception as e:
            logger.error(f"Error setting up anomaly state initialization: {e}")
        
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

        # Add severity variations for experiments
        self.severity_variations = {
            "cpu_stress": [
                {"workers": 16, "load": 50},
                {"workers": 24, "load": 75},
                {"workers": 32, "load": 100}
            ],
            "io_bottleneck": [
                {"delay": "500ms", "percent": 75},
                {"delay": "1000ms", "percent": 85},
                {"delay": "2000ms", "percent": 100}
            ],
            "network_bottleneck": [
                {"latency": "1000ms", "correlation": "75"},
                {"latency": "2000ms", "correlation": "85"},
                {"latency": "3000ms", "correlation": "100"}
            ],
            "cache_bottleneck": [
                {"workers": 4, "size": "1GB"},
                {"workers": 6, "size": "1.5GB"},
                {"workers": 8, "size": "2GB"}
            ]
        }

    def _should_update_cache(self) -> bool:
        """Check if cache needs to be updated"""
        return (
            self._last_cache_update == 0 or  # Force update if explicitly set to 0
            not self._last_cache_update or
            time.time() - self._last_cache_update > self.CACHE_TTL
        )

    @alru_cache(maxsize=100, ttl=5)
    async def verify_experiment_deleted(self, anomaly_type: str) -> bool:
        """Verify that an experiment has been fully deleted with caching"""
        try:
            if anomaly_type in self._experiment_cache:
                cache_entry = self._experiment_cache[anomaly_type]
                if time.time() - cache_entry['timestamp'] < self.CACHE_TTL:
                    return cache_entry['deleted']

            plural = self._get_experiment_plural(anomaly_type)
            name = f"ob-{anomaly_type.replace('_', '-')}"
            
            try:
                self.custom_api.get_namespaced_custom_object(
                    group="chaos-mesh.org",
                    version="v1alpha1",
                    namespace=self.namespace,
                    plural=plural,
                    name=name
                )
                # Cache the result
                self._experiment_cache[anomaly_type] = {
                    'deleted': False,
                    'timestamp': time.time()
                }
                return False  # If we get here, the object still exists
            except ApiException as e:
                if e.status == 404:  # Not found means it's deleted
                    # Cache the result
                    self._experiment_cache[anomaly_type] = {
                        'deleted': True,
                        'timestamp': time.time()
                    }
                    return True
                raise e
        except Exception as e:
            logger.error(f"Error verifying experiment deletion: {str(e)}")
            return False

    async def wait_for_deletion(self, anomaly_type: str, timeout: int = 30):
        """Wait for an experiment to be fully deleted with exponential backoff"""
        # Increase timeout for disk_stress experiments
        if anomaly_type == "disk_stress":
            timeout = 120  # 2 minutes for disk stress

        start_time = datetime.now()
        retry_delay = 2
        max_retries = timeout // retry_delay

        for _ in range(max_retries):
            if await self.verify_experiment_deleted(anomaly_type):
                logger.info(f"Successfully verified deletion of {anomaly_type} experiment")
                return True
            
            # Exponential backoff with a cap
            retry_delay = min(retry_delay * 1.5, 10)
            logger.debug(f"Experiment {anomaly_type} still exists, waiting {retry_delay} seconds...")
            await asyncio.sleep(retry_delay)

        logger.error(f"Timeout waiting for {anomaly_type} experiment deletion after {timeout} seconds")
        return False

    async def delete_chaos_experiment(self, anomaly_type: str, experiment_name: str = None):
        """Delete a specific chaos mesh experiment with optimized retries"""
        try:
            # Special handling for cache bottleneck
            if anomaly_type == "cache_bottleneck":
                # Restore default memstore_limit_percentage
                await self._update_ob_parameter({
                    "memstore_limit_percentage": "0"  # Restore default value 0
                })
                if experiment_name:
                    if experiment_name in self.active_anomalies:
                        del self.active_anomalies[experiment_name]
                    return [experiment_name]
                return []
            
            # Special handling for too_many_indexes
            elif anomaly_type == "too_many_indexes":
                # Generate a unique experiment name for tracking
                experiment_name = f"ob-too-many-indexes-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                
                # Read SQL commands from template
                template = self._get_experiment_template(anomaly_type)
                sql_commands = template["spec"].get("sqlCommands", [])
                
                # Execute SQL commands directly
                await self._execute_sql_commands(sql_commands)
                
                # If no specific target nodes were provided, use a default
                if not target_nodes:
                    target_nodes = ["obcluster"]
                
                # Create tracking records for each target node
                for node in target_nodes:
                    node_experiment_name = f"{experiment_name}-{node.replace('.', '-')}"
                    
                    # Track the experiment with SQL commands included
                    self.active_anomalies[node_experiment_name] = {
                        "start_time": datetime.now().isoformat(),
                        "status": "active",
                        "type": anomaly_type,
                        "name": node_experiment_name,
                        "target": "obcluster",
                        "node": node,
                        "sql_commands": sql_commands
                    }
                    
                    logger.info(f"Created {anomaly_type} experiment {node_experiment_name} for node {node}")
                
                # Invalidate cache
                self.invalidate_cache()
                self._last_cache_update = 0
                
                return

            # For other types, proceed with chaos mesh experiment deletion
            plural = self._get_experiment_plural(anomaly_type)
            deleted_experiments = []
            
            # If no specific experiment name is provided, delete all experiments of this type
            if experiment_name is None:
                # List all experiments of this type
                experiments = await asyncio.to_thread(
                    self.custom_api.list_namespaced_custom_object,
                    group="chaos-mesh.org",
                    version="v1alpha1",
                    namespace=self.namespace,
                    plural=plural
                )
                
                # Filter experiments by type prefix
                type_prefix = f"ob-{anomaly_type.replace('_', '-')}"
                matching_experiments = [
                    exp for exp in experiments.get("items", [])
                    if exp["metadata"]["name"].startswith(type_prefix)
                ]
                
                # Delete each matching experiment
                for exp in matching_experiments:
                    exp_name = exp["metadata"]["name"]
                    try:
                        await asyncio.to_thread(
                            self.custom_api.delete_namespaced_custom_object,
                            group="chaos-mesh.org",
                            version="v1alpha1",
                            namespace=self.namespace,
                            plural=plural,
                            name=exp_name
                        )
                        # Remove from active anomalies tracking
                        if exp_name in self.active_anomalies:
                            del self.active_anomalies[exp_name]
                        deleted_experiments.append(exp_name)
                        logger.info(f"Deleted {anomaly_type} experiment {exp_name}")
                    except ApiException as e:
                        if e.status != 404:  # Ignore 404 Not Found errors
                            logger.warning(f"Error deleting experiment {exp_name}: {str(e)}")
            else:
                # Delete specific experiment
                try:
                    await asyncio.to_thread(
                        self.custom_api.delete_namespaced_custom_object,
                        group="chaos-mesh.org",
                        version="v1alpha1",
                        namespace=self.namespace,
                        plural=plural,
                        name=experiment_name
                    )
                    # Remove from active anomalies tracking
                    if experiment_name in self.active_anomalies:
                        del self.active_anomalies[experiment_name]
                    deleted_experiments.append(experiment_name)
                    logger.info(f"Deleted experiment {experiment_name}")
                except ApiException as e:
                    if e.status != 404:  # Ignore 404 Not Found errors
                        raise e
            
            # Additional check: find and remove any other experiments of this type
            # from active_anomalies that might have been missed
            type_prefix = f"ob-{anomaly_type.replace('_', '-')}"
            for name in list(self.active_anomalies.keys()):
                if name.startswith(type_prefix) or self.active_anomalies[name].get("type") == anomaly_type:
                    # Double check if it's really gone from Kubernetes
                    try:
                        await asyncio.to_thread(
                            self.custom_api.get_namespaced_custom_object,
                            group="chaos-mesh.org",
                            version="v1alpha1",
                            namespace=self.namespace,
                            plural=plural,
                            name=name
                        )
                        # If we get here, it still exists, strange but let's not remove it
                        logger.warning(f"Experiment {name} still exists after deletion attempt")
                    except ApiException as e:
                        if e.status == 404:  # Not found, good - it's gone
                            if name in self.active_anomalies:
                                del self.active_anomalies[name]
                                if name not in deleted_experiments:
                                    deleted_experiments.append(name)
                                    logger.info(f"Removed {name} from tracking after verified deletion")
                    except Exception as e:
                        logger.error(f"Error verifying deletion of {name}: {str(e)}")
            
            # Invalidate the cache to force refresh
            self.invalidate_cache()
            self._last_cache_update = 0  # Force refresh on next request
                
            logger.info(f"Completed deletion of {anomaly_type} experiments")
            return deleted_experiments
        except Exception as e:
            logger.error(f"Failed to delete chaos experiment: {str(e)}")
            raise

    async def delete_all_chaos_experiments(self):
        """Delete all running chaos mesh experiments in parallel"""
        try:
            # Clear active anomalies tracking and cache
            self.active_anomalies = {}
            self._experiment_cache = {}
            
            # Delete all types of chaos experiments in parallel
            experiment_types = self.experiment_types
            tasks = [self.delete_chaos_experiment(exp_type) for exp_type in experiment_types]
            await asyncio.gather(*tasks)
            
            logger.info(f"Deleted all chaos experiments in namespace {self.namespace}")
        except Exception as e:
            logger.error(f"Failed to delete chaos experiments: {str(e)}")
            raise Exception(f"Failed to delete chaos experiments: {str(e)}")

    @alru_cache(maxsize=1, ttl=1)
    async def get_active_anomalies(self):
        """Get list of currently active anomalies with caching"""
        try:
            # Always force update if _last_cache_update is 0
            if self._last_cache_update == 0 or self._should_update_cache():
                # Sync with Kubernetes to get current active experiments
                experiment_types = self.experiment_types
                current_experiments = []
                found_experiment_names = set()  # Track experiments found in k8s
                
                # First log the currently tracked anomalies (as a fallback)
                logger.debug(f"Currently tracked anomalies: {list(self.active_anomalies.keys())}")
                
                # Get the actual state from Kubernetes
                for exp_type in experiment_types:
                    plural = self._get_experiment_plural(exp_type)
                    try:
                        experiments = await asyncio.to_thread(
                            self.custom_api.list_namespaced_custom_object,
                            group="chaos-mesh.org",
                            version="v1alpha1",
                            namespace=self.namespace,
                            plural=plural
                        )
                        
                        logger.debug(f"Found {len(experiments.get('items', []))} {exp_type} experiments in Kubernetes")
                        
                        for exp in experiments.get('items', []):
                            # Get the phase, default to empty string if not found
                            phase = exp.get('status', {}).get('phase', '').lower()
                            # logger.debug(f"Experiment {exp['metadata']['name']} has phase: '{phase}'")
                            
                            # Be more flexible with phase detection - consider any experiment without a failed/deleted status as active
                            if phase and phase not in ['failed', 'finished', 'deleted', 'completed', 'terminating']:
                                name = exp['metadata']['name']
                                found_experiment_names.add(name)  # Add to set of found experiments
                                
                                # Use existing data if available, otherwise create new
                                if name in self.active_anomalies:
                                    current_experiments.append(self.active_anomalies[name])
                                else:
                                    start_time = exp['metadata'].get('creationTimestamp', datetime.now().isoformat())
                                    target = exp['spec']['selector']['labelSelectors'].get('ref-obcluster', 'unknown')
                                    current_experiments.append({
                                        "name": name,
                                        "type": exp_type,
                                        "start_time": start_time,
                                        "status": "active",
                                        "target": target,
                                        "node": target
                                    })
                    except ApiException as e:
                        if e.status != 404:  # Ignore 404 Not Found errors
                            logger.error(f"Error listing {exp_type} experiments: {str(e)}")
                        continue
                    except Exception as e:
                        logger.error(f"Unexpected error listing {exp_type} experiments: {str(e)}")
                        continue

                # Update active_anomalies with current experiments
                new_active_anomalies = {}
                
                # Only keep experiments actually found in Kubernetes
                for exp in current_experiments:
                    if exp['name'] in found_experiment_names:
                        new_active_anomalies[exp['name']] = exp
                
                # If we found nothing in Kubernetes but have local records, 
                # do one final verification of each locally tracked anomaly
                if not new_active_anomalies and self.active_anomalies:
                    logger.info(f"No anomalies found in Kubernetes but have {len(self.active_anomalies)} local records. Verifying each.")
                    for name, anomaly in self.active_anomalies.items():
                        try:
                            # Try to fetch each anomaly directly to verify it exists
                            anomaly_type = anomaly.get('type')
                            if not anomaly_type:
                                continue
                                
                            plural = self._get_experiment_plural(anomaly_type)
                            try:
                                await asyncio.to_thread(
                                    self.custom_api.get_namespaced_custom_object,
                                    group="chaos-mesh.org",
                                    version="v1alpha1",
                                    namespace=self.namespace,
                                    plural=plural,
                                    name=name
                                )
                                # If we get here, the anomaly still exists
                                new_active_anomalies[name] = anomaly
                                logger.info(f"Verified {name} still exists in Kubernetes")
                            except ApiException as e:
                                if e.status == 404:
                                    # Anomaly doesn't exist anymore
                                    logger.info(f"Anomaly {name} no longer exists in Kubernetes")
                                    continue
                                else:
                                    # Some other error, assume it exists
                                    new_active_anomalies[name] = anomaly
                                    logger.warning(f"Error verifying {name}, assuming it exists: {str(e)}")
                        except Exception as e:
                            logger.error(f"Error verifying anomaly {name}: {str(e)}")
                
                # Replace active_anomalies with the current state
                if new_active_anomalies or not self.active_anomalies:
                    # Only update if we found something or we had nothing before
                    self.active_anomalies = new_active_anomalies
                
                logger.info(f"Active anomalies: {list(self.active_anomalies.keys())}")
                
                self._last_cache_update = time.time()
                return list(self.active_anomalies.values())
            
            logger.debug(f"Using cached active anomalies: {list(self.active_anomalies.keys())}")
            return list(self.active_anomalies.values())
        except Exception as e:
            logger.error(f"Failed to get active anomalies: {str(e)}")
            # If anything fails, return the locally tracked anomalies as a fallback
            return list(self.active_anomalies.values())

    def _get_experiment_template(self, anomaly_type: str) -> Dict[str, Any]:
        """Get the chaos experiment template with varying severity"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        experiment_name = f"ob-{anomaly_type.replace('_', '-')}-{timestamp}"
        
        # Base template with enhanced selectors
        base_template = {
            "apiVersion": "chaos-mesh.org/v1alpha1",
            "metadata": {
                "name": experiment_name,
                "namespace": self.namespace
            },
            "spec": {
                "mode": "all",
                "selector": {
                    "namespaces": [self.namespace],
                    "labelSelectors": {"ref-obcluster": "obcluster"}
                }
            }
        }

        # Get severity variation based on previous experiments
        severity_index = len(self.active_anomalies) % len(self.severity_variations.get(anomaly_type, [{"default": "default"}]))
        severity = self.severity_variations.get(anomaly_type, [{}])[severity_index]

        # Experiment configurations with varying severity
        experiments = {
            "cpu_stress": {
                "kind": "StressChaos",
                "spec": {
                    "stressors": {
                        "cpu": {
                            "workers": severity.get("workers", 32),
                            "load": severity.get("load", 100)
                        }
                    },
                    "duration": f"{self.collection_duration}s"
                }
            },
            "io_bottleneck": {
                "kind": "IOChaos",
                "spec": {
                    "action": "latency",
                    "delay": severity.get("delay", "1000ms"),
                    "path": "/home/admin/",
                    "percent": severity.get("percent", 100),
                    "methods": ["write", "read"],
                    "duration": f"{self.collection_duration}s"
                }
            },
            "network_bottleneck": {
                "kind": "NetworkChaos",
                "spec": {
                    "action": "delay",
                    "delay": {
                        "latency": severity.get("latency", "2000ms"),
                        "correlation": severity.get("correlation", "100"),
                        "jitter": "0ms"
                    },
                    "target": {
                        "selector": {
                            "namespaces": [self.namespace],
                            "labelSelectors": {"ref-obcluster": "obcluster"}
                        },
                        "mode": "all"
                    },
                    "direction": "both",
                    "duration": f"{self.collection_duration}s"
                }
            },
            "cache_bottleneck": {
                "kind": "StressChaos",
                "spec": {
                    "stressors": {
                        "memory": {
                            "workers": severity.get("workers", 8),
                            "size": severity.get("size", "2GB")
                        }
                    },
                    "duration": f"{self.collection_duration}s"
                }
            }
        }

        # Special handling for too_many_indexes
        if anomaly_type == "too_many_indexes":
            # This will be handled differently as it requires SQL execution
            # Create indexes on both tpcc and sbtest databases
            tpcc_sql_commands = [
                "USE tpcc",
                # Create multiple indexes on customer table
                "CREATE INDEX idx_customer_1 ON customer(c_w_id)",
                "CREATE INDEX idx_customer_2 ON customer(c_d_id)",
                "CREATE INDEX idx_customer_3 ON customer(c_last)",
                "CREATE INDEX idx_customer_4 ON customer(c_first)",
                "CREATE INDEX idx_customer_5 ON customer(c_balance)",
                # Create multiple indexes on district table
                "CREATE INDEX idx_district_1 ON district(d_w_id)",
                "CREATE INDEX idx_district_2 ON district(d_name)",
                "CREATE INDEX idx_district_3 ON district(d_street_1)",
                # Create multiple indexes on history table
                "CREATE INDEX idx_history_1 ON history(h_c_id)",
                "CREATE INDEX idx_history_2 ON history(h_d_id)",
                "CREATE INDEX idx_history_3 ON history(h_w_id)",
                # Create multiple indexes on item table
                "CREATE INDEX idx_item_1 ON item(i_name)",
                "CREATE INDEX idx_item_2 ON item(i_price)",
                "CREATE INDEX idx_item_3 ON item(i_data)",
                # Create multiple indexes on new_orders table
                "CREATE INDEX idx_new_orders_1 ON new_orders(no_w_id)",
                "CREATE INDEX idx_new_orders_2 ON new_orders(no_d_id)",
                # Create multiple indexes on order_line table
                "CREATE INDEX idx_order_line_1 ON order_line(ol_w_id)",
                "CREATE INDEX idx_order_line_2 ON order_line(ol_d_id)",
                "CREATE INDEX idx_order_line_3 ON order_line(ol_i_id)",
                "CREATE INDEX idx_order_line_4 ON order_line(ol_delivery_d)",
                # Create multiple indexes on orders table
                "CREATE INDEX idx_orders_1 ON orders(o_w_id)",
                "CREATE INDEX idx_orders_2 ON orders(o_d_id)",
                "CREATE INDEX idx_orders_3 ON orders(o_c_id)",
                "CREATE INDEX idx_orders_4 ON orders(o_entry_d)",
                # Create multiple indexes on stock table
                "CREATE INDEX idx_stock_1 ON stock(s_w_id)",
                "CREATE INDEX idx_stock_2 ON stock(s_i_id)",
                "CREATE INDEX idx_stock_3 ON stock(s_quantity)",
                # Create multiple indexes on warehouse table
                "CREATE INDEX idx_warehouse_1 ON warehouse(w_name)",
                "CREATE INDEX idx_warehouse_2 ON warehouse(w_street_1)"
            ]
            
            sbtest_sql_commands = [
                "USE sbtest",
                # Create multiple indexes for each sbtest table
                "CREATE INDEX idx_sbtest1_1 ON sbtest1(k)",
                "CREATE INDEX idx_sbtest1_2 ON sbtest1(c)",
                "CREATE INDEX idx_sbtest1_3 ON sbtest1(pad)",
                "CREATE INDEX idx_sbtest2_1 ON sbtest2(k)",
                "CREATE INDEX idx_sbtest2_2 ON sbtest2(c)",
                "CREATE INDEX idx_sbtest2_3 ON sbtest2(pad)",
                "CREATE INDEX idx_sbtest3_1 ON sbtest3(k)",
                "CREATE INDEX idx_sbtest3_2 ON sbtest3(c)",
                "CREATE INDEX idx_sbtest3_3 ON sbtest3(pad)",
                "CREATE INDEX idx_sbtest4_1 ON sbtest4(k)",
                "CREATE INDEX idx_sbtest4_2 ON sbtest4(c)",
                "CREATE INDEX idx_sbtest4_3 ON sbtest4(pad)",
                "CREATE INDEX idx_sbtest5_1 ON sbtest5(k)",
                "CREATE INDEX idx_sbtest5_2 ON sbtest5(c)",
                "CREATE INDEX idx_sbtest5_3 ON sbtest5(pad)",
                "CREATE INDEX idx_sbtest6_1 ON sbtest6(k)",
                "CREATE INDEX idx_sbtest6_2 ON sbtest6(c)",
                "CREATE INDEX idx_sbtest6_3 ON sbtest6(pad)",
                "CREATE INDEX idx_sbtest7_1 ON sbtest7(k)",
                "CREATE INDEX idx_sbtest7_2 ON sbtest7(c)",
                "CREATE INDEX idx_sbtest7_3 ON sbtest7(pad)",
                "CREATE INDEX idx_sbtest8_1 ON sbtest8(k)",
                "CREATE INDEX idx_sbtest8_2 ON sbtest8(c)",
                "CREATE INDEX idx_sbtest8_3 ON sbtest8(pad)",
                "CREATE INDEX idx_sbtest9_1 ON sbtest9(k)",
                "CREATE INDEX idx_sbtest9_2 ON sbtest9(c)",
                "CREATE INDEX idx_sbtest9_3 ON sbtest9(pad)",
                "CREATE INDEX idx_sbtest10_1 ON sbtest10(k)",
                "CREATE INDEX idx_sbtest10_2 ON sbtest10(c)",
                "CREATE INDEX idx_sbtest10_3 ON sbtest10(pad)"
            ]
            
            # Combine all SQL commands
            sql_commands = tpcc_sql_commands + sbtest_sql_commands
            
            # Return a template that includes the SQL commands
            # This is not an actual Chaos Mesh resource type but used for tracking
            return {
                "apiVersion": "chaos-mesh.org/v1alpha1",
                "kind": "SQLChaos",  # Not a real Chaos Mesh type, just for internal tracking
                "metadata": {
                    "name": experiment_name,
                    "namespace": self.namespace
                },
                "spec": {
                    "mode": "all",
                    "selector": {
                        "namespaces": [self.namespace],
                        "labelSelectors": {"ref-obcluster": "obcluster"}
                    },
                    "action": "exec",
                    "sqlCommands": sql_commands
                }
            }

        # Create a deep copy of base template
        template = base_template.copy()
        experiment_config = experiments.get(anomaly_type)
        if not experiment_config:
            raise ValueError(f"Unsupported anomaly type: {anomaly_type}")
        
        # Update non-spec fields
        for key, value in experiment_config.items():
            if key != "spec":
                template[key] = value
        
        # Merge specs while preserving base spec fields
        if "spec" in experiment_config:
            template["spec"].update(experiment_config["spec"])
        
        return template

    def _get_experiment_plural(self, anomaly_type: str) -> str:
        """Get the plural form of the experiment type for the API"""
        # Normalize the anomaly type (kebab-case to snake_case)
        normalized_type = anomaly_type.replace('-', '_')
        
        if normalized_type == "cpu_stress":
            return "stresschaos"
        elif normalized_type == "io_bottleneck":
            return "iochaos"
        elif normalized_type == "network_bottleneck":
            return "networkchaos"
        elif normalized_type == "cache_bottleneck":
            return "stresschaos"
        elif normalized_type == "too_many_indexes":
            # Not actually used since we don't create a real Chaos Mesh resource for this
            return "none"
        else:
            raise ValueError(f"Unsupported anomaly type: {anomaly_type}")

    async def _update_ob_parameter(self, config_changes: Dict[str, Any]) -> None:
        """Update OceanBase configuration parameters using SQL"""
        try:
            conn = pymysql.connect(**self.ob_config)
            with conn.cursor() as cursor:
                for param, value in config_changes.items():
                    # Convert parameter name to OceanBase format
                    ob_param = param.lower()
                    base_sql = f"ALTER SYSTEM SET {ob_param} = {value}"
                    for zone in self.ob_zones:
                        sql = base_sql + f" ZONE={zone}"
                        logger.info(f"Executing SQL: {sql}")
                        cursor.execute(sql)
            
            conn.commit()
            conn.close()
            logger.info(f"Updated OceanBase configuration: {config_changes}")
            
        except Exception as e:
            logger.error(f"Failed to update OceanBase configuration: {str(e)}")
            raise

    async def _execute_sql_commands(self, commands: List[str]) -> None:
        """Execute SQL commands on OceanBase cluster using direct connection"""
        try:
            conn = pymysql.connect(**self.ob_config)
            with conn.cursor() as cursor:
                current_db = None
                for cmd in commands:
                    # Check if command is for changing database
                    if cmd.lower().startswith("use "):
                        current_db = cmd.split()[1].strip()
                        logger.info(f"Switching to database: {current_db}")
                    
                    logger.info(f"Executing SQL: {cmd}")
                    try:
                        cursor.execute(cmd)
                    except pymysql.err.OperationalError as e:
                        # Handle errors but continue with other commands
                        logger.warning(f"Error executing SQL command: {cmd}, error: {str(e)}")
                        # If it's a duplicate index error, continue
                        if "Duplicate key name" in str(e) or "already exists" in str(e):
                            continue
                        # For other errors, continue but log them
                        continue
                    except Exception as e:
                        logger.warning(f"Unexpected error executing SQL command: {cmd}, error: {str(e)}")
                        continue
            
            conn.commit()
            conn.close()
            
            logger.info(f"Successfully executed SQL commands")
            
        except Exception as e:
            logger.error(f"Failed to execute SQL commands: {str(e)}")
            raise

    async def apply_chaos_experiment(self, anomaly_type: str, target_node: Optional[Union[List[str], str]] = None):
        """Apply a chaos mesh experiment based on the anomaly type with optimized retries"""
        try:
            # Normalize target_node to a list if it's a string
            if isinstance(target_node, str):
                target_nodes = [target_node]
            elif isinstance(target_node, list):
                target_nodes = target_node
            else:
                target_nodes = []
                
            # Special handling for cache bottleneck
            if anomaly_type == "cache_bottleneck":
                # Update memstore_limit_percentage
                await self._update_ob_parameter({
                    "memstore_limit_percentage": "20"  # Reduce from default 0
                })
                # Create a tracking experiment without actual chaos mesh resource
                experiment = {
                    "apiVersion": "chaos-mesh.org/v1alpha1",
                    "kind": "StressChaos",
                    "metadata": {
                        "name": f"ob-cache-stress-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        "namespace": self.namespace
                    },
                    "spec": {
                        "mode": "all",
                        "selector": {
                            "namespaces": [self.namespace],
                            "labelSelectors": {"ref-obcluster": "obcluster"}
                        }
                    }
                }
            
            # Special handling for too_many_indexes
            elif anomaly_type == "too_many_indexes":
                # Generate a unique experiment name for tracking
                experiment_name = f"ob-too-many-indexes-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                
                # Read SQL commands from template
                template = self._get_experiment_template(anomaly_type)
                sql_commands = template["spec"].get("sqlCommands", [])
                
                # Execute SQL commands directly
                await self._execute_sql_commands(sql_commands)
                
                # If no specific target nodes were provided, use a default
                if not target_nodes:
                    target_nodes = ["obcluster"]
                
                # Create tracking records for each target node
                for node in target_nodes:
                    node_experiment_name = f"{experiment_name}-{node.replace('.', '-')}"
                    
                    # Track the experiment with SQL commands included
                    self.active_anomalies[node_experiment_name] = {
                        "start_time": datetime.now().isoformat(),
                        "status": "active",
                        "type": anomaly_type,
                        "name": node_experiment_name,
                        "target": "obcluster",
                        "node": node,
                        "sql_commands": sql_commands
                    }
                    
                    logger.info(f"Created {anomaly_type} experiment {node_experiment_name} for node {node}")
                
                # Invalidate cache
                self.invalidate_cache()
                self._last_cache_update = 0
                
                return
            else:
                experiment = self._get_experiment_template(anomaly_type)

            experiment_name = experiment["metadata"]["name"]
            
            # Track the active anomaly
            target = experiment["spec"]["selector"]["labelSelectors"].get("ref-obcluster", "obcluster")
            
            # If no specific target nodes were provided, use the default target
            if not target_nodes:
                target_nodes = [target]
            
            # Create an experiment for each target node
            for node in target_nodes:
                # Make a copy of the experiment definition for this node
                node_experiment = experiment.copy()
                node_experiment_name = f"{experiment_name}-{node.replace('.', '-')}"
                node_experiment["metadata"]["name"] = node_experiment_name
                
                # Update selector to target specific pod if needed
                if anomaly_type not in ["cache_bottleneck", "too_many_indexes"]:
                    # Adjust selector to target the specific pod
                    if "pod-selector" not in node_experiment["spec"]["selector"]:
                        node_experiment["spec"]["selector"]["podNames"] = [node]
                
                # Only create chaos mesh experiment for non-special cases
                if anomaly_type not in ["cache_bottleneck", "too_many_indexes"]:
                    # Create experiment with retry logic
                    max_retries = 5
                    retry_delay = 2
                    
                    for attempt in range(max_retries):
                        try:
                            await asyncio.to_thread(
                                self.custom_api.create_namespaced_custom_object,
                                group="chaos-mesh.org",
                                version="v1alpha1",
                                namespace=self.namespace,
                                plural=self._get_experiment_plural(anomaly_type),
                                body=node_experiment
                            )
                            break
                        except ApiException as e:
                            if attempt < max_retries - 1 and ("AlreadyExists" in str(e) or "is being deleted" in str(e)):
                                logger.info(f"Experiment creation failed, retrying in {retry_delay} seconds...")
                                await asyncio.sleep(retry_delay)
                                retry_delay *= 2
                                continue
                            raise e
                
                # Track this specific anomaly
                self.active_anomalies[node_experiment_name] = {
                    "start_time": datetime.now().isoformat(),
                    "status": "active",
                    "type": anomaly_type,
                    "name": node_experiment_name,
                    "target": target,
                    "node": node
                }
                
                logger.info(f"Created {anomaly_type} experiment {node_experiment_name} on node {node}")
            
            # Invalidate cache
            self.invalidate_cache()
            self._last_cache_update = 0
            
        except Exception as e:
            logger.error(f"Failed to create chaos experiment: {str(e)}")
            raise

    def invalidate_cache(self):
        """Invalidate all caches"""
        self._experiment_cache = {}
        self._last_cache_update = None
        
        # For alru_cache decorated methods, we'll force cache invalidation
        # by updating the last request time
        self._last_request_time = 0  # This will force a refresh on next request

    def _initialize_active_anomalies(self):
        """Initialize active anomalies from Kubernetes"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Use synchronous methods to avoid async in _initialize_active_anomalies
            # We'll schedule the initial cache refresh to happen soon after startup
            threading.Thread(target=self._initialize_active_anomalies_sync).start()
        except Exception as e:
            logger.error(f"Error setting up anomaly state initialization: {e}")

    def _initialize_active_anomalies_sync(self):
        """Synchronously initialize active anomalies"""
        try:
            # Wait a bit for the service to fully initialize
            time.sleep(2)
            
            # Create new active_anomalies dictionary
            new_active_anomalies = {}
            
            # Check each experiment type
            for exp_type in self.experiment_types:
                plural = self._get_experiment_plural(exp_type)
                try:
                    experiments = self.custom_api.list_namespaced_custom_object(
                        group="chaos-mesh.org",
                        version="v1alpha1",
                        namespace=self.namespace,
                        plural=plural
                    )
                    
                    if "items" in experiments:
                        for exp in experiments["items"]:
                            name = exp['metadata']['name']
                            start_time = exp['metadata'].get('creationTimestamp', datetime.now().isoformat())
                            target = exp['spec']['selector']['labelSelectors'].get('ref-obcluster', 'unknown')
                            new_active_anomalies[name] = {
                                "name": name,
                                "type": exp_type,
                                "start_time": start_time,
                                "status": "active",
                                "target": target,
                                "node": target
                            }
                            
                except Exception as e:
                    logger.error(f"Error listing {exp_type} experiments during initialization: {str(e)}")
                    continue
            
            # Update local active_anomalies
            self.active_anomalies = new_active_anomalies
            
            logger.info(f"Initialized active anomalies on startup: {list(self.active_anomalies.keys())}")
        except Exception as e:
            logger.error(f"Error initializing active anomalies: {str(e)}")

    async def _fetch_available_nodes(self) -> List[str]:
        """Get list of available pods for running experiments"""
        try:
            pods = await asyncio.to_thread(
                self.core_api.list_namespaced_pod,
                namespace=self.namespace,
                label_selector="ref-obcluster"
            )
            pod_names = []
            for pod in pods.items:
                if pod.metadata.name.startswith("obcluster-"):
                    pod_names.append(pod.metadata.name)
            return pod_names
        except ApiException as e:
            logger.error(f"Error getting available pods: {str(e)}")
            return []
        
    async def get_available_nodes(self) -> List[str]:
        """Get list of available pods for running experiments"""
        if not self.available_nodes:
            return await self._fetch_available_nodes()
        return self.available_nodes
    
    def _fetch_ob_zones(self):
        """Fetch OceanBase zones from the database"""
        try:
            conn = pymysql.connect(**self.ob_config)
            with conn.cursor() as cursor:
                cursor.execute("SELECT zone FROM oceanbase.DBA_OB_ZONES")
                zones = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            if not zones:
                logger.warning("No zones found in database, using default zones")
                return ['zone1', 'zone2', 'zone3']
                
            logger.info(f"Found OceanBase zones: {zones}")
            return zones
        except Exception as e:
            logger.error(f"Error fetching OceanBase zones: {str(e)}")
            logger.info("Using default zones due to error")
            return ['zone1', 'zone2', 'zone3']

# Create singleton k8s service object
k8s_service = K8sService()