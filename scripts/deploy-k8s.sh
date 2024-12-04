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

if ! command_exists kustomize; then
    echo "Error: kustomize is not installed"
    exit 1
fi

if ! command_exists helm; then
    echo "Error: helm is not installed"
    exit 1
fi

# Create namespace if it doesn't exist
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Check and install prerequisites
echo "Checking prerequisites..."

# Install cert-manager if not present
if ! kubectl get crd certificates.cert-manager.io > /dev/null 2>&1; then
    echo "Installing cert-manager..."
    kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
    # Wait for cert-manager to be ready
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/instance=cert-manager -n cert-manager --timeout=300s
fi

# Install OB operator using Helm if not present
if ! kubectl get crd obclusters.oceanbase.oceanbase.com > /dev/null 2>&1; then
    echo "Installing OB operator..."
    helm repo add ob-operator https://oceanbase.github.io/ob-operator/
    helm repo update
    helm install ob-operator ob-operator/ob-operator --namespace oceanbase-system --create-namespace
    # Wait for operator to be ready
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=ob-operator -n oceanbase-system --timeout=300s
fi

# Apply secrets
echo "Applying secrets..."
kubectl apply -f k8s/base/secrets.yaml

# Deploy OceanBase cluster based on environment
echo "Deploying OceanBase cluster for $ENVIRONMENT environment..."

if [ "$ENVIRONMENT" = "dev" ]; then
    echo "Applying development environment configuration..."
    kubectl apply -k k8s/base
else
    echo "Applying production environment configuration..."
    kubectl apply -k k8s/overlays/prod
fi

# Wait for OBCluster to be ready
echo "Waiting for OceanBase cluster to be ready..."
kubectl wait --for=condition=ready obcluster/obcluster -n $NAMESPACE --timeout=600s

# Print deployment status
echo "Deployment status:"
kubectl get obcluster -n $NAMESPACE
kubectl get pods -n $NAMESPACE
kubectl get services -n $NAMESPACE

echo "OceanBase cluster deployment complete!"

# Print connection information
OBSERVER_POD=$(kubectl get pods -n $NAMESPACE -l app=observer -o jsonpath='{.items[0].metadata.name}')
OBPROXY_POD=$(kubectl get pods -n $NAMESPACE -l app=obproxy -o jsonpath='{.items[0].metadata.name}')

echo "
Connection Information:
----------------------
To connect via OBProxy (recommended):
kubectl port-forward $OBPROXY_POD -n $NAMESPACE 2883:2883

Then connect using:
mysql -h127.0.0.1 -P2883 -uroot@sys

To connect directly to observer:
kubectl port-forward $OBSERVER_POD -n $NAMESPACE 2881:2881

Then connect using:
mysql -h127.0.0.1 -P2881 -uroot@sys
" 