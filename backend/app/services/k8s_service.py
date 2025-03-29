from dotenv import load_dotenv
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import logging
import os
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import asyncio
import time
from functools import lru_cache, wraps
from async_lru import alru_cache
import json
import pymysql
import threading
import copy

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
        self.available_nodes = []
        # Define supported experiment types
        self.experiment_types = ["cpu_stress", "io_bottleneck", "network_bottleneck", "cache_bottleneck"]
        try:
            running_loop = asyncio.get_running_loop()
            # If a running loop exists, schedule an asynchronous update
            asyncio.create_task(self._update_available_nodes())
        except RuntimeError:
            # No running loop, so block synchronously
            loop = asyncio.get_event_loop()
            self.available_nodes = loop.run_until_complete(self._fetch_available_nodes())
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
                {"workers": 32, "load": 50},
                {"workers": 48, "load": 75},
                {"workers": 64, "load": 100}
            ],
            "io_bottleneck": [
                {"intensity": "low", "command": "dd if=/dev/zero of=/tmp/io_stress_file bs=1M count=2048 conv=fdatasync & sleep %s"},
                {"intensity": "medium", "command": "dd if=/dev/zero of=/tmp/io_stress_file bs=1M count=2048 conv=fdatasync & dd if=/dev/zero of=/tmp/io_stress_file2 bs=1M count=2048 conv=fdatasync & sleep %s"},
                {"intensity": "high", "command": "dd if=/dev/zero of=/tmp/io_stress_file bs=1M count=2048 conv=fdatasync & dd if=/dev/zero of=/tmp/io_stress_file2 bs=1M count=2048 conv=fdatasync & dd if=/dev/zero of=/tmp/io_stress_file3 bs=1M count=2048 conv=fdatasync & sleep %s"}
            ],
            "network_bottleneck": [
                {"latency": "1000ms", "correlation": "85"},
                {"latency": "2000ms", "correlation": "95"},
                {"latency": "3000ms", "correlation": "100"}
            ],
            "cache_bottleneck": [
                {"workers": 8, "size": "2GB"},
                {"workers": 12, "size": "3GB"},
                {"workers": 16, "size": "4GB"}
            ]
        }

        # Build pod IP to name mapping synchronously
        self.ip_to_name_map = self._build_ip_to_name_mapping()

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
        """Delete a chaos experiment"""
        deleted_experiments = []
        
        try:
            # Special handling for io_bottleneck - clean up properly
            if anomaly_type == "io_bottleneck":
                # If experiment_name is provided, only delete that experiment
                if experiment_name:
                    # Find and remove from active_anomalies
                    if experiment_name in self.active_anomalies:
                        # Kill any running processes associated with this experiment
                        process_id = self.active_anomalies[experiment_name].get('process_id')
                        if process_id:
                            try:
                                import os
                                import signal
                                # Kill the parent process (kubectl)
                                os.kill(process_id, signal.SIGKILL)
                            except Exception as e:
                                logger.warning(f"Failed to kill process {process_id}: {str(e)}")
                                
                        # Clean up files on the target node
                        node = self.active_anomalies[experiment_name].get('node')
                        if node:
                            cleanup_cmd = f"kubectl exec -n {self.namespace} {node} -- /bin/bash -c 'pkill -f \"dd if=/dev/urandom\" 2>/dev/null; pkill -f \"yes > /dev/null\" 2>/dev/null; rm -rf /tmp/io_stress/*' || true"
                            logger.info(f"Running cleanup: {cleanup_cmd}")
                            import os
                            os.system(cleanup_cmd)
                            
                        # Remove from active anomalies
                        del self.active_anomalies[experiment_name]
                        deleted_experiments.append(experiment_name)
                else:
                    # Delete all IO bottleneck experiments
                    io_experiments = [name for name, anomaly in self.active_anomalies.items() 
                                     if anomaly.get('type') == 'io_bottleneck']
                    
                    for exp_name in io_experiments:
                        # Kill any running processes
                        process_id = self.active_anomalies[exp_name].get('process_id')
                        if process_id:
                            try:
                                import os
                                import signal
                                os.kill(process_id, signal.SIGKILL)
                            except Exception as e:
                                logger.warning(f"Failed to kill process {process_id}: {str(e)}")
                                
                        # Clean up files on the target node
                        node = self.active_anomalies[exp_name].get('node')
                        if node:
                            cleanup_cmd = f"kubectl exec -n {self.namespace} {node} -- /bin/bash -c 'pkill -f \"dd if=/dev/urandom\" 2>/dev/null; pkill -f \"yes > /dev/null\" 2>/dev/null; rm -rf /tmp/io_stress/*' || true"
                            logger.info(f"Running cleanup: {cleanup_cmd}")
                            import os
                            os.system(cleanup_cmd)
                            
                        # Remove from active anomalies
                        del self.active_anomalies[exp_name]
                        deleted_experiments.append(exp_name)
                        
                # Also delete any K8s tracking resources
                plural = self._get_experiment_plural(anomaly_type)
                
                try:
                    if experiment_name and 'ob-io-bottleneck' in experiment_name:
                        # Extract the base name without node suffix
                        base_name = experiment_name.split('-', 3)[-1]
                        k8s_name = f"ob-io-bottleneck-{base_name}"
                        
                        await asyncio.to_thread(
                            self.custom_api.delete_namespaced_custom_object,
                            group="chaos-mesh.org",
                            version="v1alpha1",
                            namespace=self.namespace,
                            plural=plural,
                            name=k8s_name
                        )
                        logger.info(f"Deleted K8s tracking resource: {k8s_name}")
                    else:
                        # List all io bottleneck experiments
                        experiments = await asyncio.to_thread(
                            self.custom_api.list_namespaced_custom_object,
                            group="chaos-mesh.org",
                            version="v1alpha1",
                            namespace=self.namespace,
                            plural=plural
                        )
                        
                        # Delete all io bottleneck experiments
                        for exp in experiments.get('items', []):
                            name = exp['metadata']['name']
                            if 'io-bottleneck' in name:
                                await asyncio.to_thread(
                                    self.custom_api.delete_namespaced_custom_object,
                                    group="chaos-mesh.org",
                                    version="v1alpha1",
                                    namespace=self.namespace,
                                    plural=plural,
                                    name=name
                                )
                                logger.info(f"Deleted K8s tracking resource: {name}")
                except Exception as e:
                    logger.warning(f"Error deleting K8s tracking resources: {str(e)}")
                    
                # Return deleted experiments
                return deleted_experiments
            
            # Special handling for cache bottleneck
            elif anomaly_type == "cache_bottleneck":
                # Restore default memstore_limit_percentage for all zones
                for zone in self.ob_zones:
                    await self._update_ob_parameter_for_zone({
                        "memstore_limit_percentage": "0"  # Restore default value 0
                    }, zone)
                
                # Since we now create an actual Kubernetes resource for cache_bottleneck,
                # we need to delete it like other experiment types
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
                    type_prefix = f"ob-cache-stress"
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
                
                # Invalidate cache to get fresh data
                self.invalidate_cache()
                self._last_cache_update = 0
                
                return deleted_experiments
            
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
                            # Extract phase from status; if not available, use desiredPhase from experiment field.
                            phase = exp.get('status', {}).get('phase')
                            if phase is None:
                                phase = exp.get('status', {}).get('experiment', {}).get('desiredPhase', '')
                            phase_str = str(phase).lower() if phase else ""

                            # Skip experiments that indicate completion or failure
                            if phase_str in ['failed', 'finished', 'deleted', 'completed', 'terminating', 'stop']:
                                continue

                            name = exp['metadata']['name']
                            found_experiment_names.add(name)  # Add to set of found experiments

                            # Use existing data if available, otherwise create new
                            if name in self.active_anomalies:
                                current_experiments.append(self.active_anomalies[name])
                            else:
                                start_time = exp['metadata'].get('creationTimestamp', datetime.now().isoformat())
                                target_label = exp['spec']['selector']['labelSelectors'].get('ref-obcluster', 'unknown')
                                target_pods = exp['spec']['selector']['pods'].get(self.namespace, [])
                                current_experiments.append({
                                    "name": name,
                                    "type": exp_type,
                                    "start_time": start_time,
                                    "status": "active",
                                    "target": target_label,
                                    "node": target_pods
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
                            # Get the anomaly type
                            anomaly_type = anomaly.get('type')
                            if not anomaly_type:
                                continue
                                
                            # Special handling for io_bottleneck anomalies
                            if anomaly_type == 'io_bottleneck':
                                # Check if it's still within expected duration
                                start_time = datetime.fromisoformat(anomaly.get('start_time', ''))
                                now = datetime.now()
                                elapsed_seconds = (now - start_time).total_seconds()
                                expected_duration = anomaly.get('expected_duration', self.collection_duration)
                                
                                if elapsed_seconds < expected_duration:
                                    # Still within expected duration, keep it
                                    new_active_anomalies[name] = anomaly
                                    logger.info(f"Verified {name} still exists in our tracking")
                                else:
                                    logger.info(f"Anomaly {name} expired (duration {elapsed_seconds}s > expected {expected_duration}s)")
                                continue
                                
                            # Standard verification through Kubernetes API for other anomaly types
                            plural = self._get_experiment_plural(anomaly_type)
                            try:
                                response = await asyncio.to_thread(
                                    self.custom_api.get_namespaced_custom_object,
                                    group="chaos-mesh.org",
                                    version="v1alpha1",
                                    namespace=self.namespace,
                                    plural=plural,
                                    name=name
                                )
                                phase = response.get('status', {}).get('phase')
                                if phase is None:
                                    phase = response.get('status', {}).get('experiment', {}).get('desiredPhase', '')
                                phase_str = str(phase).lower() if phase else ""
                                if phase_str in ['failed', 'finished', 'deleted', 'completed', 'terminating', 'stop']:
                                    logger.info(f"Anomaly {name} is marked as {phase_str} and will be removed from tracking")
                                else:
                                    new_active_anomalies[name] = anomaly
                                    logger.info(f"Verified {name} still exists in Kubernetes")
                            except ApiException as e:
                                if e.status == 404:  # Not found, good - it's gone
                                    logger.info(f"Anomaly {name} no longer exists in Kubernetes")
                                else:
                                    new_active_anomalies[name] = anomaly
                                    logger.warning(f"Error verifying {name}, assuming it exists: {str(e)}")
                        except Exception as e:
                            logger.error(f"Error verifying anomaly {name}: {str(e)}")
                
                # Replace active_anomalies with the current state
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

    def _get_experiment_template(self, anomaly_type: str, severity: Optional[str] = None) -> Dict[str, Any]:
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
                "mode": "one",
                "selector": {
                    "namespaces": [self.namespace],
                    "labelSelectors": {"ref-obcluster": "obcluster"},
                    "pods": {self.namespace: []}
                }
            }
        }

        # Select severity variation based on provided severity or fallback
        if severity is not None:
            severity_lower = severity.lower()
            severity_map = {"low": 0, "medium": 1, "high": 2}
            index = severity_map.get(severity_lower, 1)
        else:
            index = len(self.active_anomalies) % len(self.severity_variations.get(anomaly_type, [{"default": "default"}]))

        severity_config = self.severity_variations.get(anomaly_type, [{}])[index]

        # Experiment configurations with varying severity
        experiments = {
            "cpu_stress": {
                "kind": "StressChaos",
                "spec": {
                    "stressors": {
                        "cpu": {
                            "workers": severity_config.get("workers", 32),
                            "load": severity_config.get("load", 100)
                        }
                    },
                    "duration": f"{self.collection_duration}s"
                }
            },
            "io_bottleneck": {
                "kind": "StressChaos",
                "spec": {
                    "stressors": {
                        "exec": {
                            "command": [
                                "/bin/bash",
                                "-c",
                                severity_config.get("command", "dd if=/dev/zero of=/tmp/io_stress_file bs=1M count=1024 conv=fdatasync & sleep %s") % self.collection_duration
                            ]
                        }
                    },
                    "duration": f"{self.collection_duration}s"
                }
            },
            "network_bottleneck": {
                "kind": "NetworkChaos",
                "spec": {
                    "action": "delay",
                    "delay": {
                        "latency": severity_config.get("latency", "2000ms"),
                        "correlation": severity_config.get("correlation", "100"),
                        "jitter": "0ms"
                    },
                    "target": {
                        "selector": {
                            "namespaces": [self.namespace],
                            "labelSelectors": {"ref-obcluster": "obcluster"}
                        },
                        "mode": "one"
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
                            "workers": severity_config.get("workers", 8),
                            "size": severity_config.get("size", "2GB")
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
                    "mode": "one",
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
            return "stresschaos"
        elif normalized_type == "network_bottleneck":
            return "networkchaos"
        elif normalized_type == "cache_bottleneck":
            return "stresschaos"
        elif normalized_type == "too_many_indexes":
            return "none"
        else:
            raise ValueError(f"Unsupported anomaly type: {anomaly_type}")

    async def _update_ob_parameter_for_zone(self, config_changes: Dict[str, Any], zone: str) -> None:
        """Update OceanBase configuration parameters for a specific zone using SQL"""
        try:
            conn = pymysql.connect(**self.ob_config)
            with conn.cursor() as cursor:
                for param, value in config_changes.items():
                    # Convert parameter name to OceanBase format
                    ob_param = param.lower()
                    sql = f"ALTER SYSTEM SET {ob_param} = {value} ZONE={zone}"
                    logger.info(f"Executing SQL: {sql}")
                    cursor.execute(sql)
            
            conn.commit()
            conn.close()
            logger.info(f"Updated OceanBase configuration for zone {zone}: {config_changes}")
            
        except Exception as e:
            logger.error(f"Failed to update OceanBase configuration for zone {zone}: {str(e)}")
            raise

    async def _update_ob_parameter(self, config_changes: Dict[str, Any]) -> None:
        """Update OceanBase configuration parameters using SQL for all zones"""
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

    async def apply_chaos_experiment(self, anomaly_type: str, target_node: Optional[Union[List[str], str]] = None, severity: str = "medium"):
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
                severity_map = {"low": 0, "medium": 1, "high": 2}
                index = severity_map.get(severity.lower(), 1)
                severity_config = self.severity_variations.get(anomaly_type, [{}])[index]
                
                memory_workers = severity_config.get("workers", 8)
                if memory_workers <= 4:  # Lite severity
                    memstore_limit = "9"
                elif memory_workers <= 6:  # Medium severity
                    memstore_limit = "6"
                else:  # Severe, must be > 0.01 else database will crash
                    memstore_limit = "3"
                
                target_zones = set()
                for node in target_nodes:
                    node_parts = node.split('-')
                    for i, part in enumerate(node_parts):
                        if part.startswith('zone') and i > 0:
                            target_zones.add(part)
                if not target_zones:
                    target_zones = self.ob_zones
                for zone in target_zones:
                    await self._update_ob_parameter_for_zone({"memstore_limit_percentage": memstore_limit}, zone)
                
                # Create a tracking experiment with an actual chaos mesh resource
                experiment = {
                    "apiVersion": "chaos-mesh.org/v1alpha1",
                    "kind": "StressChaos",
                    "metadata": {
                        "name": f"ob-cache-bottleneck-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        "namespace": self.namespace
                    },
                    "spec": {
                        "mode": "one",
                        "selector": {
                            "namespaces": [self.namespace],
                            "labelSelectors": {"ref-obcluster": "obcluster"},
                            "pods": {self.namespace: target_nodes if target_nodes else []}
                        },
                        "stressors": {
                            "memory": {
                                "workers": 1,
                                "size": "100MB"  # Very small stress just to keep the resource alive
                            }
                        },
                        "duration": f"{self.collection_duration}s"
                    }
                }
            
            # Special handling for io_bottleneck - direct execution for more control
            elif anomaly_type == "io_bottleneck":
                severity_map = {"low": 0, "medium": 1, "high": 2}
                index = severity_map.get(severity.lower(), 1)
                severity_config = self.severity_variations.get(anomaly_type, [{}])[index]
                # Generate a unique experiment name for tracking
                experiment_name = f"ob-io-bottleneck-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                
                # Create tracking experiment for Kubernetes
                experiment = {
                    "apiVersion": "chaos-mesh.org/v1alpha1",
                    "kind": "StressChaos",
                    "metadata": {
                        "name": experiment_name,
                        "namespace": self.namespace
                    },
                    "spec": {
                        "mode": "one",
                        "selector": {
                            "namespaces": [self.namespace],
                            "labelSelectors": {"ref-obcluster": "obcluster"},
                            "pods": {self.namespace: target_nodes if target_nodes else []}
                        },
                        "stressors": {
                            "memory": {
                                "workers": 1,
                                "size": "10MB"  # Minimal resource just for tracking
                            }
                        },
                        "duration": f"{self.collection_duration}s"
                    }
                }
                
                # Create the tracking resource in Kubernetes
                try:
                    await asyncio.to_thread(
                        self.custom_api.create_namespaced_custom_object,
                        group="chaos-mesh.org",
                        version="v1alpha1",
                        namespace=self.namespace,
                        plural=self._get_experiment_plural(anomaly_type),
                        body=experiment
                    )
                    logger.info(f"Created tracking resource for IO bottleneck: {experiment_name}")
                except Exception as e:
                    logger.warning(f"Failed to create tracking resource for IO bottleneck: {str(e)}")
                
                # For each target node, execute IO stress command directly on the pod 
                for node in target_nodes:
                    node_experiment_name = f"{experiment_name}-{node.replace('.', '-')}"
                    
                    try:
                        logger.info(f"Running IO stress command on node {node}")
                        io_command = severity_config.get("command") % self.collection_duration
                        
                        kubectl_cmd = f"kubectl exec -n {self.namespace} {node} -- /bin/bash -c '{io_command}' > /dev/null 2>&1 &"
                        logger.info(f"Executing kubectl command: {kubectl_cmd}")
                        import subprocess
                        process = subprocess.Popen(kubectl_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        
                        # Track this specific anomaly
                        self.active_anomalies[node_experiment_name] = {
                            "start_time": datetime.now().isoformat(),
                            "status": "active",
                            "type": anomaly_type,
                            "name": node_experiment_name,
                            "target": "obcluster",
                            "node": node,
                            "command": io_command,
                            "process_id": process.pid,
                            "expected_duration": self.collection_duration
                        }
                        
                        logger.info(f"Created {anomaly_type} experiment {node_experiment_name} on node {node}")
                    except Exception as e:
                        logger.error(f"Error running IO stress on node {node}: {str(e)}")
            
                # The following block for too_many_indexes remains unchanged
                experiment_name = f"ob-too-many-indexes-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                template = self._get_experiment_template(anomaly_type)
                sql_commands = template["spec"].get("sqlCommands", [])
                
                await self._execute_sql_commands(sql_commands)
                
                if not target_nodes:
                    target_nodes = ["obcluster"]
                
                for node in target_nodes:
                    node_experiment_name = f"{experiment_name}-{node.replace('.', '-')}"
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
                
                self.invalidate_cache()
                self._last_cache_update = 0
                
                return
            else:
                experiment = self._get_experiment_template(anomaly_type, severity)
            
            experiment_name = experiment["metadata"]["name"]
            target = experiment["spec"]["selector"]["labelSelectors"].get("ref-obcluster", "obcluster")
            
            if not target_nodes:
                target_nodes = [target]

            for node in target_nodes:
                node_experiment = copy.deepcopy(experiment)
                node_experiment_name = f"{experiment_name}-{node.replace('.', '-')}"
                node_experiment["metadata"]["name"] = node_experiment_name
                node_experiment["spec"]["mode"] = "one"
                node_experiment["spec"]["selector"]["pods"] = {self.namespace: [node]}
                
                try:
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
                except Exception as e:
                    logger.error(f"Error creating experiment for {node}: {str(e)}")
                
                if anomaly_type != "io_bottleneck" or not target_nodes:
                    self.active_anomalies[node_experiment_name] = {
                        "start_time": datetime.now().isoformat(),
                        "status": "active",
                        "type": anomaly_type,
                        "name": node_experiment_name,
                        "target": target,
                        "node": node
                    }
                    logger.info(f"Created {anomaly_type} experiment {node_experiment_name} on node {node}")
            
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

    async def verify_experiment_exists(self, experiment_name: str, anomaly_type: str = None) -> bool:
        """Verify if a specific experiment exists, with special handling for io_bottleneck"""
        try:
            # Special handling for IO bottleneck - we track it ourselves
            if anomaly_type == "io_bottleneck" or (
                not anomaly_type and experiment_name and "io-bottleneck" in experiment_name
            ):
                # If it's in our active_anomalies dict, check if it's within expected duration
                if experiment_name in self.active_anomalies:
                    anomaly = self.active_anomalies[experiment_name]
                    start_time = datetime.fromisoformat(anomaly.get('start_time', ''))
                    now = datetime.now()
                    elapsed_seconds = (now - start_time).total_seconds()
                    expected_duration = anomaly.get('expected_duration', self.collection_duration)
                    
                    # If within expected duration, consider it active
                    return elapsed_seconds < expected_duration
                
                return False  # Not in our tracking
            
            # For all other anomaly types, check Kubernetes
            if not anomaly_type:
                # Try to determine type from experiment name
                for exp_type in self.experiment_types:
                    if exp_type.replace("_", "-") in experiment_name:
                        anomaly_type = exp_type
                        break
            
            if not anomaly_type:
                logger.warning(f"Cannot verify existence of {experiment_name} without knowing its type")
                return False
            
            # Check if it exists in Kubernetes
            try:
                await asyncio.to_thread(
                    self.custom_api.get_namespaced_custom_object,
                    group="chaos-mesh.org",
                    version="v1alpha1",
                    namespace=self.namespace,
                    plural=self._get_experiment_plural(anomaly_type),
                    name=experiment_name
                )
                return True
            except ApiException as e:
                if e.status == 404:
                    return False
                raise e
                
        except Exception as e:
            logger.error(f"Error verifying experiment existence: {str(e)}")
            return False

    async def _update_available_nodes(self):
        """Asynchronously update available_nodes from Kubernetes."""
        nodes = await self._fetch_available_nodes()
        self.available_nodes = nodes

    async def get_pod_ip_to_name_mapping(self) -> Dict[str, str]:
        """Get mapping of pod IPs to pod names"""
        try:
            pods = await asyncio.to_thread(
                self.core_api.list_namespaced_pod,
                namespace=self.namespace,
                label_selector="ref-obcluster"
            )
            ip_to_name = {}
            for pod in pods.items:
                if pod.status.pod_ip and pod.metadata.name.startswith("obcluster-"):
                    ip_to_name[pod.status.pod_ip] = pod.metadata.name
            logger.info(f"Got pod IP to name mapping: {ip_to_name}")
            return ip_to_name
        except ApiException as e:
            logger.error(f"Error getting pod IP to name mapping: {str(e)}")
            return {}

    def _build_ip_to_name_mapping(self) -> Dict[str, str]:
        """Build pod IP to name mapping synchronously using Kubernetes API."""
        mapping = {}
        try:
            from kubernetes import config, client
            import os
            if os.getenv('KUBERNETES_SERVICE_HOST'):
                config.load_incluster_config()
            else:
                config.load_kube_config()
            core_api = client.CoreV1Api()
            namespace = os.getenv('OCEANBASE_NAMESPACE', 'oceanbase')
            pods = core_api.list_namespaced_pod(namespace=namespace, label_selector="ref-obcluster")
            for pod in pods.items:
                pod_ip = pod.status.pod_ip
                pod_name = pod.metadata.name
                if pod_ip and pod_name.startswith("obcluster-"):
                    mapping[pod_ip] = pod_name
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error building pod IP to name mapping: {e}")
        return mapping

# Create singleton k8s service object
k8s_service = K8sService()