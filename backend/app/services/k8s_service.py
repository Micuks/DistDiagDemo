from kubernetes import client, config
from kubernetes.client.rest import ApiException
import logging
import os
from typing import Dict, Any
from datetime import datetime

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

    async def apply_chaos_experiment(self, anomaly_type: str):
        """Apply a chaos mesh experiment based on the anomaly type"""
        experiment = self._get_experiment_template(anomaly_type)
        
        try:
            # Delete any existing experiments first
            await self.delete_all_chaos_experiments()
            
            # Create new experiment
            self.custom_api.create_namespaced_custom_object(
                group="chaos-mesh.org",
                version="v1alpha1",
                namespace=self.namespace,
                plural=self._get_experiment_plural(anomaly_type),
                body=experiment
            )
            # Track the active anomaly
            self.active_anomalies[anomaly_type] = {
                "start_time": datetime.now().isoformat(),
                "status": "active",
                "type": anomaly_type,
                "target": experiment["spec"]["selector"]["labelSelectors"]["ref-obcluster"]
            }
            logger.info(f"Created {anomaly_type} experiment in namespace {self.namespace}")
        except ApiException as e:
            logger.error(f"Failed to create chaos experiment: {str(e)}")
            raise Exception(f"Failed to create chaos experiment: {str(e)}")

    async def delete_all_chaos_experiments(self):
        """Delete all running chaos mesh experiments"""
        try:
            # Clear active anomalies tracking
            self.active_anomalies = {}
            
            # Delete CPU/Memory stress experiments
            self.custom_api.delete_collection_namespaced_custom_object(
                group="chaos-mesh.org",
                version="v1alpha1",
                namespace=self.namespace,
                plural="stresschaos"
            )
            
            # Delete network chaos experiments
            self.custom_api.delete_collection_namespaced_custom_object(
                group="chaos-mesh.org",
                version="v1alpha1",
                namespace=self.namespace,
                plural="networkchaos"
            )
            
            # Delete IO chaos experiments
            self.custom_api.delete_collection_namespaced_custom_object(
                group="chaos-mesh.org",
                version="v1alpha1",
                namespace=self.namespace,
                plural="iochaos"
            )
            
            logger.info(f"Deleted all chaos experiments in namespace {self.namespace}")
        except ApiException as e:
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