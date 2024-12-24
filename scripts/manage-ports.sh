#!/bin/bash

NAMESPACE=oceanbase
PID_FILE="/tmp/ob_port_forwards.pid"

start_forwards() {
    echo "Starting port forwards..."
    
    # Forward OBProxy
    nohup kubectl port-forward svc/svc-obproxy -n $NAMESPACE 2883:2883 > /dev/null 2>&1 &
    echo $! >> $PID_FILE
    
    # Forward Kubernetes Dashboard
    nohup kubectl port-forward -n default service/kubernetes-dashboard 18081:80 > /dev/null 2>&1 &
    echo $! >> $PID_FILE
    
    echo "Port forwards started. PIDs saved in $PID_FILE"
    echo "To stop all forwards, run: $0 stop"
}

stop_forwards() {
    if [ -f "$PID_FILE" ]; then
        echo "Stopping port forwards..."
        while read pid; do
            if ps -p $pid > /dev/null; then
                kill $pid
                echo "Stopped process $pid"
            fi
        done < $PID_FILE
        rm $PID_FILE
        echo "All port forwards stopped"
    else
        echo "No port forwards found"
    fi
}

case "$1" in
    start)
        start_forwards
        ;;
    stop)
        stop_forwards
        ;;
    *)
        echo "Usage: $0 {start|stop}"
        exit 1
        ;;
esac 