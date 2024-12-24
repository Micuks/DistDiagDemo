#!/bin/bash

# Check if Kind is installed
if ! command -v kind &> /dev/null; then
    echo "Kind is not installed. Installing Kind..."
    [ $(uname -m) = x86_64 ] && curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.25.0/kind-linux-amd64
    [ $(uname -m) = aarch64 ] && curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.25.0/kind-linux-arm64
    chmod +x ./kind
    sudo mv ./kind /usr/local/bin/kind
fi

# Get host IP address
HOST_IP=$(ip -4 addr show docker0 | grep -Po 'inet \K[\d.]+')
if [ -z "$HOST_IP" ]; then
    HOST_IP=$(ip route get 1 | awk '{print $7;exit}')
fi

# Create Kind configuration with 3 worker nodes
cat << EOF > kind-config.yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: distdiag-cluster
networking:
  apiServerAddress: "${HOST_IP}"
nodes:
- role: control-plane
  kubeadmConfigPatches:
  - |
    kind: InitConfiguration
    nodeRegistration:
      kubeletExtraArgs:
        node-labels: "ingress-ready=true"
        eviction-hard: "memory.available<5%"
- role: worker
  kubeadmConfigPatches:
  - |
    kind: JoinConfiguration
    nodeRegistration:
      kubeletExtraArgs:
        eviction-hard: "memory.available<5%"
- role: worker
  kubeadmConfigPatches:
  - |
    kind: JoinConfiguration
    nodeRegistration:
      kubeletExtraArgs:
        eviction-hard: "memory.available<5%"
- role: worker
  kubeadmConfigPatches:
  - |
    kind: JoinConfiguration
    nodeRegistration:
      kubeletExtraArgs:
        eviction-hard: "memory.available<5%"
EOF

# Create data directory for persistence
mkdir -p data

# Pull the Kind node image manually
echo "Pulling Kind node image..."
docker pull kindest/node:v1.31.2

# Create the cluster
echo "Creating Kind cluster..."
kind create cluster --config kind-config.yaml --image kindest/node:v1.31.2

# Wait for cluster to be ready
echo "Waiting for cluster to be ready..."
kubectl wait --for=condition=ready node --all --timeout=300s

echo "Local Kubernetes cluster is ready!"
echo "You can now run './scripts/deploy-k8s.sh dev' to deploy OceanBase"

# Print cluster information
echo -e "\nCluster Information:"
echo "---------------------"
kubectl get nodes -o wide 
