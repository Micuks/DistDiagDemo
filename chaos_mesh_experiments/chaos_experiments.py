from kubernetes import client, config
from kubernetes.client import ApiClient
import yaml
import time

class ChaosMeshExperiment:
    def __init__(self, namespace="chaos-mesh"):
        """Initialize the Chaos Mesh experiment manager"""
        config.load_kube_config()
        self.api_client = ApiClient()
        self.custom_api = client.CustomObjectsApi()
        self.namespace = namespace
        self.api_versions = {}  # Store API versions for different kinds
        self.plural_names = {}  # Store plural names for different kinds
        
        # Verify Chaos Mesh installation
        self._verify_chaos_mesh()

    def _verify_chaos_mesh(self):
        """Verify that Chaos Mesh CRDs are installed"""
        try:
            api = client.ApiextensionsV1Api()
            crds = api.list_custom_resource_definition()
            chaos_crds = [crd for crd in crds.items if crd.spec.group == "chaos-mesh.org"]
            
            if not chaos_crds:
                raise RuntimeError(
                    "Chaos Mesh CRDs not found. Please ensure Chaos Mesh is properly installed in your cluster.\n"
                    "Installation guide: https://chaos-mesh.org/docs/production-installation-using-helm/"
                )
            
            # Print available CRDs for debugging
            print("Available Chaos Mesh CRDs:")
            for crd in chaos_crds:
                print(f"- {crd.spec.names.kind}: {crd.spec.group}/{crd.spec.versions[0].name}")
                # Store both API version and plural name
                self.api_versions[crd.spec.names.kind] = crd.spec.versions[0].name
                self.plural_names[crd.spec.names.kind] = crd.spec.names.plural

        except Exception as e:
            raise RuntimeError(f"Failed to verify Chaos Mesh installation: {str(e)}")

    def apply_chaos(self, chaos_manifest):
        """Apply a Chaos Mesh experiment"""
        try:
            kind = chaos_manifest["kind"]
            # Get the correct plural name from CRD
            plural = self.plural_names.get(kind)
            if not plural:
                raise ValueError(f"Unknown chaos kind: {kind}")
            
            version = self.api_versions.get(kind, "v1alpha1")
            
            print(f"Applying {kind} with API version {version}")
            self.custom_api.create_namespaced_custom_object(
                group="chaos-mesh.org",
                version=version,
                namespace=self.namespace,
                plural=plural,
                body=chaos_manifest
            )
            print(f"Applied {chaos_manifest['kind']} successfully")
        except client.rest.ApiException as e:
            if e.status == 404:
                print(f"Error: Chaos Mesh API endpoint not found. Please verify Chaos Mesh is installed properly.")
                print(f"Details: {e.reason}")
                print(f"Attempted to access: chaos-mesh.org/{version} with plural '{plural}'")
            else:
                print(f"API error applying chaos: {e.status} - {e.reason}")
        except Exception as e:
            print(f"Error applying chaos: {str(e)}")

    def delete_chaos(self, name, kind):
        """Delete a Chaos Mesh experiment"""
        try:
            # Get the correct plural name from CRD
            plural = self.plural_names.get(kind)
            if not plural:
                raise ValueError(f"Unknown chaos kind: {kind}")
            
            version = self.api_versions.get(kind, "v1alpha1")
            
            print(f"Deleting {kind} with API version {version}")
            self.custom_api.delete_namespaced_custom_object(
                group="chaos-mesh.org",
                version=version,
                namespace=self.namespace,
                plural=plural,
                name=name
            )
            print(f"Deleted {kind} {name} successfully")
        except client.rest.ApiException as e:
            if e.status == 404:
                print(f"Error: Chaos experiment {kind}/{name} not found")
                print(f"Details: {e.reason}")
            else:
                print(f"API error deleting chaos: {e.status} - {e.reason}")
        except Exception as e:
            print(f"Error deleting chaos: {str(e)}")

class CPUSaturation:
    def __init__(self, namespace="oceanbase"):
        self.chaos = ChaosMeshExperiment(namespace)

    def trigger(self, duration="180s"):
        cpu_chaos = {
            "apiVersion": "chaos-mesh.org/v1alpha1",
            "kind": "StressChaos",
            "metadata": {
                "name": "ob-cpu-stress",
                "namespace": self.chaos.namespace
            },
            "spec": {
                "mode": "all",
                "selector": {
                    "namespaces": [self.chaos.namespace],
                    "labelSelectors": {
                        "app.kubernetes.io/instance": "obcluster"
                    }
                },
                "stressors": {
                    "cpu": {
                        "workers": 2,
                        "load": 100
                    }
                },
                "duration": duration
            }
        }
        self.chaos.apply_chaos(cpu_chaos)

    def recover(self):
        self.chaos.delete_chaos("ob-cpu-stress", "StressChaos")

class IOSaturation:
    def __init__(self, namespace="oceanbase"):
        self.chaos = ChaosMeshExperiment(namespace)

    def trigger(self, duration="180s"):
        io_chaos = {
            "apiVersion": "chaos-mesh.org/v1alpha1",
            "kind": "IOChaos",
            "metadata": {
                "name": "ob-io-stress",
                "namespace": self.chaos.namespace
            },
            "spec": {
                "action": "latency",
                "mode": "all",
                "selector": {
                    "namespaces": [self.chaos.namespace],
                    "labelSelectors": {
                        "app.kubernetes.io/instance": "obcluster"
                    }
                },
                "delay": "100ms",
                "path": "/home/admin/oceanbase/store",
                "percent": 100,
                "duration": duration
            }
        }
        self.chaos.apply_chaos(io_chaos)

    def recover(self):
        self.chaos.delete_chaos("ob-io-stress", "IOChaos")

class NetSaturation:
    def __init__(self, namespace="oceanbase"):
        self.chaos = ChaosMeshExperiment(namespace)

    def trigger(self, duration="180s"):
        network_chaos = {
            "apiVersion": "chaos-mesh.org/v1alpha1",
            "kind": "NetworkChaos",
            "metadata": {
                "name": "ob-network-delay",
                "namespace": self.chaos.namespace
            },
            "spec": {
                "action": "bandwidth",
                "mode": "all",
                "selector": {
                    "namespaces": [self.chaos.namespace],
                    "labelSelectors": {
                        "app.kubernetes.io/instance": "obcluster"
                    }
                },
                "bandwidth": {
                    "rate": "1mbps",
                    "limit": 1048576,
                    "buffer": 10000
                },
                "duration": duration,
                "direction": "both"
            }
        }
        self.chaos.apply_chaos(network_chaos)

    def recover(self):
        self.chaos.delete_chaos("ob-network-delay", "NetworkChaos")

if __name__ == "__main__":
    # Example usage
    cpu_stress = CPUSaturation()
    io_stress = IOSaturation()
    net_stress = NetSaturation()

    # Trigger CPU stress
    cpu_stress.trigger()
    time.sleep(10)
    cpu_stress.recover()

    # Trigger IO stress
    io_stress.trigger()
    time.sleep(10)
    io_stress.recover()

    # Trigger Network stress
    net_stress.trigger()
    time.sleep(10)
    net_stress.recover() 