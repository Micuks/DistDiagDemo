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

# Add OceanBase Helm repository
echo "Adding OceanBase Helm repository..."
helm repo add oceanbase https://oceanbase.github.io/ob-operator
helm repo update

# Install local-path-provisioner for local storage
echo "Installing local-path storage class..."
kubectl apply -f https://raw.githubusercontent.com/rancher/local-path-provisioner/master/deploy/local-path-storage.yaml

# Install OceanBase Operator
echo "Installing OceanBase Operator..."
helm install ob-operator oceanbase/ob-operator \
    --namespace oceanbase \
    --create-namespace \
    --set image.repository=oceanbase/ob-operator \
    --set image.tag=4.2.1-100000092023101717

# Wait for operator to be ready
echo "Waiting for operator to be ready..."
kubectl -n oceanbase wait --for=condition=ready pod -l app.kubernetes.io/name=ob-operator --timeout=300s

echo "OceanBase Operator installation complete!"
echo "You can now deploy the OceanBase cluster using:"
echo "./scripts/deploy-k8s.sh dev" 