#!/bin/bash

# Check if helm is installed
if ! command -v helm &> /dev/null; then
    echo "Helm is not installed. Installing Helm..."
    if [ -n "$HTTPS_PROXY" ]; then
        curl -x "$HTTPS_PROXY" -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
    else
        curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
    fi
fi

# Install cert-manager
echo "Installing cert-manager..."
helm repo add jetstack https://charts.jetstack.io
helm repo update
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.9.1/cert-manager.crds.yaml
helm install \
  cert-manager jetstack/cert-manager \
  --namespace cert-manager \
  --create-namespace \
  --version v1.9.1 \
  --wait

# Wait for cert-manager CRDs to be ready
echo "Waiting for cert-manager CRDs to be established..."
kubectl wait --for=condition=established --timeout=60s crd/certificates.cert-manager.io
kubectl wait --for=condition=established --timeout=60s crd/issuers.cert-manager.io
kubectl wait --for=condition=established --timeout=60s crd/clusterissuers.cert-manager.io

# Wait for cert-manager pods
echo "Waiting for cert-manager to be ready..."
kubectl -n cert-manager wait --for=condition=ready pod -l app.kubernetes.io/instance=cert-manager --timeout=300s

# Add OceanBase Helm repository
echo "Adding OceanBase Helm repository..."
helm repo add oceanbase https://oceanbase.github.io/ob-operator
helm repo update

# Install local-path-provisioner for local storage
echo "Installing local-path storage class..."
kubectl apply -f https://raw.githubusercontent.com/rancher/local-path-provisioner/master/deploy/local-path-storage.yaml

# Clean up existing OceanBase Operator installation
echo "Cleaning up existing OceanBase Operator installation..."
helm uninstall ob-operator -n oceanbase-system --ignore-not-found
kubectl delete namespace oceanbase-system --ignore-not-found

# Wait for cleanup
echo "Waiting for OceanBase Operator cleanup to complete..."
sleep 10

# Install OceanBase Operator
echo "Installing OceanBase Operator..."
helm repo add ob-operator https://oceanbase.github.io/ob-operator/
helm repo update
helm install ob-operator ob-operator/ob-operator \
  --namespace=oceanbase-system \
  --create-namespace \
  --wait

# Wait for operator to be ready
echo "Waiting for operator to be ready..."
kubectl -n oceanbase-system wait --for=condition=ready pod -l app.kubernetes.io/name=ob-operator --timeout=300s

echo "OceanBase Operator installation complete!"
echo "You can now deploy the OceanBase cluster using:"
echo "./scripts/deploy-k8s.sh dev" 