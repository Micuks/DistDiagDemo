#!/bin/bash

# Set environment variables
ENVIRONMENT=${1:-dev}  # Default to dev if not specified
NAMESPACE=oceanbase

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for required commands
if ! command_exists kubectl; then
    echo "Error: kubectl is not installed"
    exit 1
fi

# Check if OB operator is installed
if ! kubectl get crd obclusters.oceanbase.oceanbase.com > /dev/null 2>&1; then
    echo "OceanBase operator not found. Installing operator..."
    ./scripts/install-ob-operator.sh
fi

# Deploy OceanBase cluster
echo "Deploying OceanBase cluster to $ENVIRONMENT environment..."

# Apply Kustomize overlay
kubectl apply -k k8s/overlays/$ENVIRONMENT

# Wait for OBCluster to be ready
echo "Waiting for OceanBase cluster to be ready..."
kubectl wait --for=condition=ready obcluster/obcluster -n $NAMESPACE --timeout=600s

# Get deployment status
echo "Deployment status:"
kubectl get obcluster -n $NAMESPACE
kubectl get pods -n $NAMESPACE
kubectl get services -n $NAMESPACE

echo "OceanBase cluster deployment complete!"

# Print connection information
OCEANBASE_HOST=$(kubectl get service oceanbase -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
if [ -z "$OCEANBASE_HOST" ]; then
    OCEANBASE_HOST=$(kubectl get service oceanbase -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
fi

echo "
Connection Information:
----------------------
Host: $OCEANBASE_HOST
SQL Port: 2881
RPC Port: 2882
Username: root
Password: password123 (from secret)

To connect using mysql client:
mysql -h $OCEANBASE_HOST -P 2881 -uroot -ppassword123
" 