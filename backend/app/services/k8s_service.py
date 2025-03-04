from kubernetes import client, config
from kubernetes.client.rest import ApiException
import logging
import os
from typing import Dict, Any, List
from datetime import datetime
import asyncio
import time
from functools import lru_cache, wraps
from async_lru import alru_cache
import json
import pymysql

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
        self.namespace = os.getenv('OCEANBASE_NAMESPACE', 'oceanbase')
        self.active_anomalies = {}  # Track active anomalies with timestamps
        self._experiment_cache = {}  # Cache for experiment status
        self.CACHE_TTL = 5  # Cache TTL in seconds
        self._last_cache_update = None
        self._last_request_time = 0  # This will force a refresh on next request
        self.experiment_types = ["cpu_stress", "io_bottleneck", "network_bottleneck", "cache_bottleneck", "too_many_indexes"]
        
        # OceanBase connection configuration
        self.ob_config = {
            'host': os.getenv('OB_HOST', '127.0.0.1'),
            'port': int(os.getenv('OB_PORT', '2881')),
            'user': os.getenv('OB_USER', 'root@sys'),
            'password': os.getenv('OB_PASSWORD', 'password'),
            'database': os.getenv('OB_DATABASE', 'oceanbase'),
            'zones': ['zone1','zone2','zone3']
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
            if anomaly_type == "cache_stress":
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
                # Drop created indexes
                if "sql_commands" in self.active_anomalies:
                    drop_commands = []
                    for cmd in self.active_anomalies["sql_commands"]:
                        if cmd.startswith("create index"):
                            # Extract index name and table
                            parts = cmd.split()
                            index_name = parts[2]
                            table_name = parts[4].split('(')[0]
                            drop_commands.append(f"drop index {index_name} on {table_name}")
                    
                    if drop_commands:
                        await self._execute_sql_commands(drop_commands)
                    self.active_anomalies.pop("sql_commands", None)
                
                if experiment_name:
                    if experiment_name in self.active_anomalies:
                        del self.active_anomalies[experiment_name]
                    return [experiment_name]
                return []

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
                            logger.debug(f"Experiment {exp['metadata']['name']} has phase: '{phase}'")
                            
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
        """Get the chaos experiment template optimized for metric impact"""
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

        # Experiment configurations targeting critical metrics
        experiments = {
            "cpu_stress": {
                "kind": "StressChaos",
                "spec": {
                    "stressors": {
                        "cpu": {
                            "workers": 32,  # Increased to match external process contention
                            "load": 100
                        }
                    },
                    "duration": "300s"
                }
            },
            "io_stress": {  # Renamed from disk_stress for clarity
                "kind": "IOChaos",
                "spec": {
                    "action": "latency",
                    "delay": "1000ms",  # Increased latency for more impact
                    "path": "/home/admin/oceanbase/store",  # OceanBase data path
                    "percent": 100,
                    "methods": ["write", "read"],  # Both read and write operations
                    "duration": "300s"
                }
            },
            "network_delay": {
                "kind": "NetworkChaos",
                "spec": {
                    "action": "delay",
                    "delay": {
                        "latency": "2000ms",  # Increased for more noticeable impact
                        "correlation": "100",
                        "jitter": "0ms"  # Removed jitter for consistent delay
                    },
                    "target": {
                        "selector": {
                            "namespaces": [self.namespace],
                            "labelSelectors": {"ref-obcluster": "obcluster"}
                        },
                        "mode": "all"
                    },
                    "direction": "both"  # Affect both inbound and outbound traffic
                }
            },
            "cache_stress": {  # New type for cache bottleneck
                "kind": "StressChaos",
                "spec": {
                    "stressors": {
                        "memory": {
                            "workers": 8,
                            "size": "2GB"  # Large memory allocation to trigger cache pressure
                        }
                    },
                    "duration": "300s"
                }
            }
        }

        # Special handling for too_many_indexes
        if anomaly_type == "too_many_indexes":
            # This will be handled differently as it requires SQL execution
            return {
                "apiVersion": "chaos-mesh.org/v1alpha1",
                "kind": "SQLChaos",  # Custom type for SQL operations
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
                    "sqlCommands": [
                        "create index toomany1 on bmsql_config(cfg_value) local",
                        "create index toomany2 on bmsql_config(cfg_name) local",
                        "create index toomany3 on bmsql_customer(c_w_id) local",
                        "create index toomany4 on bmsql_customer(c_d_id) local",
                        "create index toomany5 on bmsql_customer(c_id) local"
                    ]
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

    @timed_lru_cache(seconds=60, maxsize=10)
    def _get_experiment_plural(self, anomaly_type: str) -> str:
        """Get the plural form of the experiment type for the API with caching"""
        # Normalize the anomaly type (kebab-case to snake_case)
        normalized_type = anomaly_type.replace('-', '_')
        
        if normalized_type == "cpu_stress":
            return "stresschaos"
        elif normalized_type == "io_stress":
            return "iochaos"
        elif normalized_type == "network_delay":
            return "networkchaos"
        elif normalized_type == "cache_stress":
            return "stresschaos"
        elif normalized_type == "too_many_indexes":
            return "sqlchaos"  # Custom type for SQL operations
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
                    for zone in self.ob_config['zones']:
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
                for cmd in commands:
                    logger.info(f"Executing SQL: {cmd}")
                    cursor.execute(cmd)
            
            conn.commit()
            conn.close()
            
            # Track the SQL commands for cleanup
            if "sql_commands" not in self.active_anomalies:
                self.active_anomalies["sql_commands"] = []
            self.active_anomalies["sql_commands"].extend(commands)
            
            logger.info(f"Successfully executed {len(commands)} SQL commands")
            
        except Exception as e:
            logger.error(f"Failed to execute SQL commands: {str(e)}")
            raise

    async def apply_chaos_experiment(self, anomaly_type: str):
        """Apply a chaos mesh experiment based on the anomaly type with optimized retries"""
        try:
            # Special handling for cache bottleneck
            if anomaly_type == "cache_stress":
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
                # Read SQL commands from template
                experiment = self._get_experiment_template(anomaly_type)
                sql_commands = experiment["spec"].get("sqlCommands", [])
                await self._execute_sql_commands(sql_commands)
                # Create a tracking experiment
                experiment = {
                    "apiVersion": "chaos-mesh.org/v1alpha1",
                    "kind": "SQLChaos",
                    "metadata": {
                        "name": f"ob-too-many-indexes-{datetime.now().strftime('%Y%m%d%H%M%S')}",
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
            else:
                experiment = self._get_experiment_template(anomaly_type)

            experiment_name = experiment["metadata"]["name"]
            
            # Only create chaos mesh experiment for non-special cases
            if anomaly_type not in ["cache_stress", "too_many_indexes"]:
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
                            body=experiment
                        )
                        break
                    except ApiException as e:
                        if attempt < max_retries - 1 and ("AlreadyExists" in str(e) or "is being deleted" in str(e)):
                            logger.info(f"Experiment creation failed, retrying in {retry_delay} seconds...")
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 2
                            continue
                        raise e
            
            # Track the active anomaly
            target = experiment["spec"]["selector"]["labelSelectors"].get("ref-obcluster", "obcluster")
            self.active_anomalies[experiment_name] = {
                "start_time": datetime.now().isoformat(),
                "status": "active",
                "type": anomaly_type,
                "name": experiment_name,
                "target": target,
                "node": target
            }
            
            logger.info(f"Created {anomaly_type} experiment {experiment_name}")
            
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
        # Only clear the timed_lru_cache as it supports cache_clear
        self._get_experiment_plural.cache_clear()
        
        # For alru_cache decorated methods, we'll force cache invalidation
        # by updating the last request time
        self._last_request_time = 0  # This will force a refresh on next request