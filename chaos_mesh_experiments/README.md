# OceanBase Chaos Mesh Experiments

This directory contains Python scripts to generate various chaos experiments for OceanBase using Chaos Mesh in Kubernetes.

## Prerequisites

1. A running Kubernetes cluster (kind)
2. Chaos Mesh installed in the cluster
3. OceanBase running in the cluster
4. Python 3.7+

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure Chaos Mesh is installed in your cluster:
```bash
# Install Chaos Mesh using Helm
helm repo add chaos-mesh https://charts.chaos-mesh.org
helm repo update
helm install chaos-mesh chaos-mesh/chaos-mesh --namespace=chaos-testing --create-namespace
```

## Available Chaos Experiments

1. CPU Saturation
   - Simulates high CPU usage on OceanBase pods
   - Uses StressChaos to generate CPU load

2. I/O Saturation
   - Simulates I/O pressure on OceanBase pods
   - Uses IOChaos to add latency to I/O operations

3. Network Saturation
   - Simulates network bandwidth limitations
   - Uses NetworkChaos to limit bandwidth to 1Mbps

## Usage

```python
from chaos_experiments import CPUSaturation, IOSaturation, NetSaturation

# Initialize experiments
cpu_stress = CPUSaturation(namespace="oceanbase")
io_stress = IOSaturation(namespace="oceanbase")
net_stress = NetSaturation(namespace="oceanbase")

# Trigger CPU stress
cpu_stress.trigger(duration="180s")  # Runs for 3 minutes
cpu_stress.recover()  # Removes the chaos experiment

# Trigger I/O stress
io_stress.trigger(duration="180s")
io_stress.recover()

# Trigger Network stress
net_stress.trigger(duration="180s")
net_stress.recover()
```

## Notes

1. Make sure to adjust the namespace in the experiment initialization to match your OceanBase deployment
2. The default duration for all experiments is 180 seconds
3. Always call the `recover()` method after you're done with the experiment
4. Adjust the selector labels in the code to match your OceanBase pod labels 