#!/bin/bash

# Install cert-manager
echo "Installing cert-manager..."
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.16.2/cert-manager.yaml

# Wait for cert-manager CRDs to be ready
echo "Waiting for cert-manager CRDs to be established..."
kubectl wait --for=condition=established --timeout=60s crd/certificates.cert-manager.io
kubectl wait --for=condition=established --timeout=60s crd/issuers.cert-manager.io
kubectl wait --for=condition=established --timeout=60s crd/clusterissuers.cert-manager.io

# Wait for cert-manager pods
echo "Waiting for cert-manager to be ready..."
kubectl -n cert-manager wait --for=condition=ready pod -l app=cert-manager --timeout=300s

# Install local-path-provisioner for local storage
echo "Installing local-path storage class..."
kubectl apply -f https://raw.githubusercontent.com/rancher/local-path-provisioner/master/deploy/local-path-storage.yaml

# Clean up existing OceanBase Operator installation
echo "Cleaning up existing OceanBase Operator installation..."
kubectl delete namespace oceanbase-system --ignore-not-found

# Wait for cleanup
echo "Waiting for OceanBase Operator cleanup to complete..."
sleep 10

# Install OceanBase Operator
echo "Installing OceanBase Operator..."
kubectl apply -f https://raw.githubusercontent.com/oceanbase/ob-operator/2.2.0_release/deploy/operator.yaml

# Wait for operator to be ready
echo "Waiting for operator to be ready..."
kubectl -n oceanbase-system wait --for=condition=ready pod -l app.kubernetes.io/name=ob-operator --timeout=300s

echo "OceanBase Operator installation complete!"
echo "You can now deploy the OceanBase cluster using:"
echo "./scripts/deploy-k8s.sh dev" 