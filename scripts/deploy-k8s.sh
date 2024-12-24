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
    helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager \
  --create-namespace \
  --version v1.16.2 \
  --wait \
  --timeout 10m0s \
  --set installCRDs=true
    # Wait for cert-manager to be ready
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/instance=cert-manager -n cert-manager --timeout=300s
fi

# Install OB operator using Helm if not present
if ! kubectl get crd obclusters.oceanbase.oceanbase.com > /dev/null 2>&1; then
    echo "Installing OB operator..."
    helm repo add ob-operator https://oceanbase.github.io/ob-operator/
    helm repo update
    helm install ob-operator ob-operator/ob-operator --namespace oceanbase-system --create-namespace --version=2.2.0
    # Wait for operator to be ready
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=ob-operator -n oceanbase-system --timeout=300s
fi

# Install local-path-provisioner if not present
if ! kubectl get storageclass local-path > /dev/null 2>&1; then
    echo "Installing local-path-provisioner..."
    kubectl apply -f https://raw.githubusercontent.com/rancher/local-path-provisioner/v0.0.24/deploy/local-path-storage.yaml
fi

# Create secrets
echo "Creating secrets..."
# Use `root_password` and `proxyro_password` as passwords
ROOT_PASSWORD="root_password"
PROXYRO_PASSWORD="proxyro_password"

kubectl create secret -n $NAMESPACE generic root-password --from-literal=password=$ROOT_PASSWORD --dry-run=client -o yaml | kubectl apply -f -
kubectl create secret -n $NAMESPACE generic proxyro-password --from-literal=password=$PROXYRO_PASSWORD --dry-run=client -o yaml | kubectl apply -f -

# Deploy OceanBase cluster
echo "Deploying OceanBase cluster..."
kubectl apply -f k8s/base/obcluster.yaml

# Wait for OBCluster to be ready
echo "Waiting for OceanBase cluster to be ready..."
kubectl wait --for=condition=ready obcluster/obcluster -n $NAMESPACE --timeout=1200s

# Get RS_LIST by querying OceanBase cluster
echo "Getting RS_LIST from OceanBase cluster..."

# Wait for the cluster to be ready before querying
kubectl wait --for=condition=ready obcluster/obcluster -n $NAMESPACE --timeout=1200s

# Forward port to connect to OceanBase
POD=$(kubectl get pods -n $NAMESPACE -l app=observer -o jsonpath='{.items[0].metadata.name}')
kubectl port-forward -n $NAMESPACE $POD 2881:2881 &
PF_PID=$!

# Wait for port-forward to be established
sleep 5

# Query RS_LIST from OceanBase
RS_LIST=$(mysql -h127.0.0.1 -P2881 -uroot@sys -p$ROOT_PASSWORD -N -e "SELECT GROUP_CONCAT(CONCAT(SVR_IP, ':', SQL_PORT) SEPARATOR ';') AS RS_LIST FROM oceanbase.DBA_OB_SERVERS;")

# Kill port-forward
kill $PF_PID

if [ -z "$RS_LIST" ]; then
    echo "Failed to get RS_LIST from OceanBase cluster"
    exit 1
fi

echo "Retrieved RS_LIST: $RS_LIST"

# Update OBProxy configuration with RS_LIST
echo "Deploying OBProxy..."
sed "s/\${RS_LIST}/$RS_LIST/" k8s/base/obproxy.yaml | kubectl apply -f -

# Wait for OBProxy to be ready
echo "Waiting for OBProxy to be ready..."
kubectl wait --for=condition=ready deployment/obproxy -n $NAMESPACE --timeout=300s

# Print deployment status
echo "Deployment status:"
kubectl get obcluster -n $NAMESPACE
kubectl get pods -n $NAMESPACE
kubectl get services -n $NAMESPACE

echo "OceanBase cluster deployment complete!"

# Install Kubernetes Dashboard
echo "Installing Kubernetes Dashboard..."
kubectl apply -f https://raw.githubusercontent.com/kubernetes/dashboard/v2.7.0/aio/deploy/recommended.yaml

# Create dashboard admin user and get token
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ServiceAccount
metadata:
  name: admin-user
  namespace: kubernetes-dashboard
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: admin-user
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: cluster-admin
subjects:
- kind: ServiceAccount
  name: admin-user
  namespace: kubernetes-dashboard
EOF

# Wait for dashboard to be ready
kubectl wait --for=condition=ready pod -l k8s-app=kubernetes-dashboard -n kubernetes-dashboard --timeout=300s

# Get Dashboard token
DASHBOARD_TOKEN=$(kubectl get -n default secret oceanbase-dashboard-user-credentials -o jsonpath='{.data.admin}' | base64 -d)

# Start port forwarding for services
echo "Starting port forwards..."
bash scripts/manage-ports.sh start

# Print connection information and credentials
echo "
Connection Information:
----------------------
Root Password: $ROOT_PASSWORD
ProxyRO Password: $PROXYRO_PASSWORD

OBProxy is now accessible at localhost:2883
Kubernetes Dashboard is now accessible at https://localhost:8443

To connect via OBProxy (already forwarded):
mysql -h127.0.0.1 -P2883 -uroot@sys#obcluster -p$ROOT_PASSWORD

To connect directly to observer:
kubectl port-forward $(kubectl get pods -n $NAMESPACE -l app=observer -o jsonpath='{.items[0].metadata.name}') -n $NAMESPACE 2881:2881

Then connect using:
mysql -h127.0.0.1 -P2881 -uroot@sys -p$ROOT_PASSWORD

To access Kubernetes Dashboard:
1. Open https://localhost:8443 in your browser
2. Use the following token to login:
$DASHBOARD_TOKEN

To manage port forwards:
- Start: scripts/manage-ports.sh start
- Stop:  scripts/manage-ports.sh stop
" 