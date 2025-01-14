# DistDiagDemo Backend

This is the backend service for the DistDiagDemo project, providing APIs for anomaly detection, metrics collection, and workload management for OceanBase clusters.

## Features

- Real-time metrics collection from OceanBase clusters
- Database and tenant-level metrics monitoring
- Anomaly detection and diagnosis
- Workload generation and management using sysbench
- Prometheus integration for metrics storage

## Prerequisites

- Python 3.8+
- sysbench
- Docker (for containerized deployment)
- Access to an OceanBase cluster
- Prometheus instance for metrics collection

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd backend
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install sysbench:
```bash
apt-get update && apt-get install -y sysbench
```

4. Set up environment variables (create a `.env` file):
```env
OB_HOST=obproxy
OB_PORT=2881
OB_USER=root
OB_PASSWORD=your_password
PROMETHEUS_URL=http://prometheus:9090
```

## API Endpoints

### Metrics API

#### Get Database Metrics
```
GET /api/metrics/database
Response: {
    "metrics": [
        {
            "timestamp": "2024-01-10T12:00:00",
            "qps": 1000.5,
            "tps": 500.2,
            "active_sessions": 50,
            "sql_response_time": 0.1,
            "disk_io_bytes": 1024000,
            "disk_iops": 1000,
            "memory_usage": 8192,
            "cache_hit_ratio": 95.5,
            "slow_queries": 5,
            "deadlocks": 0,
            "replication_lag": 0,
            "connection_count": 100
        }
    ]
}
```

#### Get Tenant Metrics
```
GET /api/metrics/tenant?tenant_name=tenant1
Response: {
    "metrics": [
        {
            "timestamp": "2024-01-10T12:00:00",
            "tenant": "tenant1",
            "cpu_percent": 45.5,
            "memory_used": 1024.5,
            "disk_used": 10240.0,
            "iops": 1000,
            "session_count": 50,
            "active_session_count": 10
        }
    ]
}
```

### Workload API

#### Prepare Database
```
POST /api/workload/prepare
Response: {
    "workload_id": "prepare",
    "status": "Database prepared successfully"
}
```

#### Start Workload
```
POST /api/workload/start
Request: {
    "workload_type": "oltp_read_write",  // or "oltp_read_only", "oltp_write_only"
    "threads": 4  // 1-64 threads
}
Response: {
    "workload_id": "oltp_read_write_20240110_120000",
    "status": "Workload started successfully"
}
```

#### Stop Workload
```
POST /api/workload/{workload_id}/stop
Response: {
    "workload_id": "oltp_read_write_20240110_120000",
    "status": "Workload stopped successfully"
}
```

#### Stop All Workloads
```
POST /api/workload/stop-all
Response: {
    "workload_id": "all",
    "status": "All workloads stopped successfully"
}
```

#### Get Active Workloads
```
GET /api/workload/active
Response: {
    "workloads": [
        {
            "workload_id": "oltp_read_write_20240110_120000",
            "running": true
        }
    ]
}
```

### Anomaly API

#### Start Anomaly
```
POST /api/anomaly/start
Request: {
    "type": "cpu_stress"  // or "memory_stress", "network_delay", "disk_stress"
}
Response: {
    "status": "success",
    "message": "Started cpu_stress anomaly"
}
```

#### Stop Anomaly
```
POST /api/anomaly/stop
Response: {
    "status": "success",
    "message": "Stopped all anomalies"
}
```

## Development

1. Start the development server:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

2. Access the API documentation:
```
http://localhost:8000/docs
```

## Docker Deployment

1. Build the Docker image:
```bash
docker build -t distdiagdemo-backend .
```

2. Run the container:
```bash
docker run -d \
  -p 8000:8000 \
  -e OB_HOST=obproxy \
  -e OB_PORT=2881 \
  -e OB_USER=root \
  -e OB_PASSWORD=your_password \
  --name distdiagdemo-backend \
  distdiagdemo-backend
```

## Testing

Run the test suite:
```bash
pytest
```

## Monitoring

The service exposes Prometheus metrics at `/metrics` endpoint for monitoring the backend service itself.

## License
