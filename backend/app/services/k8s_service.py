from kubernetes import client, config
from kubernetes.client.rest import ApiException
import logging
import os
from typing import Dict, Any
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

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

    async def verify_experiment_deleted(self, anomaly_type: str) -> bool:
        """Verify that an experiment has been fully deleted"""
        try:
            plural = self._get_experiment_plural(anomaly_type)
            name = f"ob-{anomaly_type.replace('_', '-')}"
            
            self.custom_api.get_namespaced_custom_object(
                group="chaos-mesh.org",
                version="v1alpha1",
                namespace=self.namespace,
                plural=plural,
                name=name
            )
            return False  # If we get here, the object still exists
        except ApiException as e:
            if e.status == 404:  # Not found means it's deleted
                return True
            raise e

    async def wait_for_deletion(self, anomaly_type: str, timeout: int = 30):
        """Wait for an experiment to be fully deleted"""
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

    async def delete_chaos_experiment(self, anomaly_type: str):
        """Delete a specific chaos mesh experiment"""
        try:
            plural = self._get_experiment_plural(anomaly_type)
            name = f"ob-{anomaly_type.replace('_', '-')}"
            
            try:
                # For disk stress, try to get the experiment first to check its status
                if anomaly_type == "disk_stress":
                    try:
                        experiment = self.custom_api.get_namespaced_custom_object(
                            group="chaos-mesh.org",
                            version="v1alpha1",
                            namespace=self.namespace,
                            plural=plural,
                            name=name
                        )
                        logger.info(f"Found existing {anomaly_type} experiment, status: {experiment.get('status', {})}")
                    except ApiException as e:
                        if e.status != 404:  # Log any error other than Not Found
                            logger.warning(f"Error checking {anomaly_type} experiment status: {e}")

                self.custom_api.delete_namespaced_custom_object(
                    group="chaos-mesh.org",
                    version="v1alpha1",
                    namespace=self.namespace,
                    plural=plural,
                    name=name
                )
                # Wait for the deletion to complete
                if not await self.wait_for_deletion(anomaly_type):
                    if anomaly_type == "disk_stress":
                        logger.warning("Disk stress experiment deletion timed out, but continuing...")
                    else:
                        raise Exception(f"Timeout waiting for {anomaly_type} experiment deletion")
            except ApiException as e:
                if e.status != 404:  # Ignore 404 Not Found errors
                    raise e
            
            # Remove from active anomalies tracking
            if anomaly_type in self.active_anomalies:
                del self.active_anomalies[anomaly_type]
            
            logger.info(f"Deleted {anomaly_type} experiment in namespace {self.namespace}")
        except Exception as e:
            logger.error(f"Failed to delete chaos experiment: {str(e)}")
            raise Exception(f"Failed to delete chaos experiment: {str(e)}")

    async def delete_all_chaos_experiments(self):
        """Delete all running chaos mesh experiments"""
        try:
            # Clear active anomalies tracking
            self.active_anomalies = {}
            
            # Delete all types of chaos experiments
            experiment_types = ["cpu_stress", "memory_stress", "network_delay", "disk_stress"]
            for exp_type in experiment_types:
                await self.delete_chaos_experiment(exp_type)
            
            logger.info(f"Deleted all chaos experiments in namespace {self.namespace}")
        except Exception as e:
            logger.error(f"Failed to delete chaos experiments: {str(e)}")
            raise Exception(f"Failed to delete chaos experiments: {str(e)}")

    async def get_active_anomalies(self):
        """Get list of currently active anomalies"""
        return list(self.active_anomalies.values())

    def _get_experiment_template(self, anomaly_type: str) -> Dict[str, Any]:
        """Get the appropriate chaos mesh experiment template"""
        if anomaly_type == "cpu_stress":
            return {
                "apiVersion": "chaos-mesh.org/v1alpha1",
                "kind": "StressChaos",
                "metadata": {
                    "name": "ob-cpu-stress",
                    "namespace": self.namespace
                },
                "spec": {
                    "mode": "one",
                    "selector": {
                        "namespaces": [self.namespace],
                        "labelSelectors": {
                            "ref-obcluster": "obcluster"
                        }
                    },
                    "stressors": {
                        "cpu": {
                            "workers": 2,
                            "load": 90
                        }
                    },
                    "duration": "10m"
                }
            }
        elif anomaly_type == "memory_stress":
            return {
                "apiVersion": "chaos-mesh.org/v1alpha1",
                "kind": "StressChaos",
                "metadata": {
                    "name": "ob-memory-stress",
                    "namespace": self.namespace
                },
                "spec": {
                    "mode": "one",
                    "selector": {
                        "namespaces": [self.namespace],
                        "labelSelectors": {
                            "ref-obcluster": "obcluster"
                        }
                    },
                    "stressors": {
                        "memory": {
                            "workers": 2,
                            "size": "256MB"
                        }
                    },
                    "duration": "10m"
                }
            }
        elif anomaly_type == "network_delay":
            return {
                "apiVersion": "chaos-mesh.org/v1alpha1",
                "kind": "NetworkChaos",
                "metadata": {
                    "name": "ob-network-delay",
                    "namespace": self.namespace
                },
                "spec": {
                    "action": "delay",
                    "mode": "one",
                    "selector": {
                        "namespaces": [self.namespace],
                        "labelSelectors": {
                            "ref-obcluster": "obcluster"
                        }
                    },
                    "delay": {
                        "latency": "100ms",
                        "correlation": "100",
                        "jitter": "0ms"
                    },
                    "duration": "10m"
                }
            }
        elif anomaly_type == "disk_stress":
            return {
                "apiVersion": "chaos-mesh.org/v1alpha1",
                "kind": "IOChaos",
                "metadata": {
                    "name": "ob-disk-stress",
                    "namespace": self.namespace
                },
                "spec": {
                    "action": "latency",
                    "mode": "one",
                    "selector": {
                        "namespaces": [self.namespace],
                        "labelSelectors": {
                            "ref-obcluster": "obcluster"
                        }
                    },
                    "delay": "100ms",
                    "path": "/home/admin/oceanbase/store",
                    "percent": 100,
                    "duration": "10m"
                }
            }
        else:
            raise ValueError(f"Unsupported anomaly type: {anomaly_type}")

    def _get_experiment_plural(self, anomaly_type: str) -> str:
        """Get the plural form of the experiment type for the API"""
        if anomaly_type in ["cpu_stress", "memory_stress"]:
            return "stresschaos"
        elif anomaly_type == "network_delay":
            return "networkchaos"
        elif anomaly_type == "disk_stress":
            return "iochaos"
        else:
            raise ValueError(f"Unsupported anomaly type: {anomaly_type}") 

    async def apply_chaos_experiment(self, anomaly_type: str):
        """Apply a chaos mesh experiment based on the anomaly type"""
        experiment = self._get_experiment_template(anomaly_type)
        
        try:
            # Delete any existing experiment of the same type first
            await self.delete_chaos_experiment(anomaly_type)
            
            # Create new experiment with retry logic
            max_retries = 5  # Increased from 3
            retry_delay = 2  # Increased initial delay
            
            for attempt in range(max_retries):
                try:
                    # Verify again that the old experiment is gone
                    if not await self.verify_experiment_deleted(anomaly_type):
                        logger.info(f"Experiment still exists, waiting {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                        continue

                    self.custom_api.create_namespaced_custom_object(
                        group="chaos-mesh.org",
                        version="v1alpha1",
                        namespace=self.namespace,
                        plural=self._get_experiment_plural(anomaly_type),
                        body=experiment
                    )
                    break
                except ApiException as e:
                    if attempt < max_retries - 1 and ("AlreadyExists" in str(e) or "is being deleted" in str(e)):
                        logger.info(f"Experiment still being deleted, retrying in {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    raise e
            
            # Track the active anomaly
            self.active_anomalies[anomaly_type] = {
                "start_time": datetime.now().isoformat(),
                "status": "active",
                "type": anomaly_type,
                "target": experiment["spec"]["selector"]["labelSelectors"]["ref-obcluster"]
            }
            logger.info(f"Created {anomaly_type} experiment in namespace {self.namespace}")
        except Exception as e:
            logger.error(f"Failed to create chaos experiment: {str(e)}")
            raise Exception(f"Failed to create chaos experiment: {str(e)}") 