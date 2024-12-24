# URL for cert-manager manifest
CERT_MANAGER_YAML="https://github.com/cert-manager/cert-manager/releases/download/v1.16.2/cert-manager.yaml"

# Namespace where cert-manager is deployed
NAMESPACE="cert-manager"

# Function to check the status of cert-manager pods
check_cert_manager_status() {
    # Get the status of all pods in the cert-manager namespace
    RUNNING_COUNT=$(kubectl get pods -n $NAMESPACE --no-headers 2>/dev/null | grep -c "Running")
    TOTAL_COUNT=$(kubectl get pods -n $NAMESPACE --no-headers 2>/dev/null | wc -l)

    # If all pods are in the Running state, return success
    if [[ "$RUNNING_COUNT" -eq "$TOTAL_COUNT" && "$TOTAL_COUNT" -gt 0 ]]; then
        return 0
    else
        return 1
    fi
}

# Start the loop
while true; do
    echo "Deleting cert-manager resources..."
    kubectl delete -f $CERT_MANAGER_YAML

    echo "Reapplying cert-manager manifest..."
    kubectl apply -f $CERT_MANAGER_YAML

    echo "Waiting for cert-manager pods to become Running..."
    sleep 10  # Wait for pods to initialize

    if check_cert_manager_status; then
        echo "All cert-manager pods are Running!"
        break
    else
        echo "Cert-manager pods are not Running yet. Retrying..."
    fi
done