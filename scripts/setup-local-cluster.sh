#!/bin/bash

# Check and use system proxy settings
if [ -n "$HTTP_PROXY" ] || [ -n "$HTTPS_PROXY" ]; then
    echo "Using system proxy settings:"
    echo "HTTP_PROXY: $HTTP_PROXY"
    echo "HTTPS_PROXY: $HTTPS_PROXY"
fi

# Check if Kind is installed
if ! command -v kind &> /dev/null; then
    echo "Kind is not installed. Installing Kind..."
    if [ -n "$HTTPS_PROXY" ]; then
        [ $(uname -m) = x86_64 ] && curl -x "$HTTPS_PROXY" -Lo ./kind https://kind.sigs.k8s.io/dl/v0.25.0/kind-linux-amd64
        [ $(uname -m) = aarch64 ] && curl -x "$HTTPS_PROXY" -Lo ./kind https://kind.sigs.k8s.io/dl/v0.25.0/kind-linux-arm64
    else
        [ $(uname -m) = x86_64 ] && curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.25.0/kind-linux-amd64
        [ $(uname -m) = aarch64 ] && curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.25.0/kind-linux-arm64
    fi
    chmod +x ./kind
    sudo mv ./kind /usr/local/bin/kind
fi

# Create Kind configuration with 3 worker nodes and proxy settings
cat << EOF > kind-config.yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: distdiag-cluster
nodes:
- role: control-plane
  extraPortMappings:
  - containerPort: 2881
    hostPort: 2881
    protocol: TCP
  - containerPort: 2882
    hostPort: 2882
    protocol: TCP
  - containerPort: 7474
    hostPort: 7474
    protocol: TCP
  extraMounts:
  - hostPath: ./data
    containerPath: /data
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

# Create the cluster with proxy settings
echo "Creating Kind cluster..."
kind create cluster --config kind-config.yaml --image kindest/node:v1.31.2

# Wait for cluster to be ready
echo "Waiting for cluster to be ready..."
kubectl wait --for=condition=ready node --all --timeout=300s

# Configure proxy for kubectl if needed
if [ -n "$HTTP_PROXY" ] || [ -n "$HTTPS_PROXY" ]; then
    kubectl config set-cluster kind-distdiag-cluster --proxy-url="$HTTPS_PROXY"
fi

echo "Local Kubernetes cluster is ready!"
echo "You can now run './scripts/deploy-k8s.sh dev' to deploy OceanBase"

# Print cluster information
echo -e "\nCluster Information:"
echo "---------------------"
kubectl get nodes -o wide
echo -e "\nProxy Settings:"
echo "HTTP_PROXY: $HTTP_PROXY"
echo "HTTPS_PROXY: $HTTPS_PROXY" 