# DistDiagDemo Backend

This is the backend service for the DistDiagDemo project, providing APIs for anomaly detection, metrics collection, and workload management for OceanBase clusters.

## Features

- Real-time metrics collection from OceanBase clusters
- Database and tenant-level metrics monitoring
- Anomaly detection and diagnosis
- Workload generation and management using sysbench
- Prometheus integration for metrics storage

## Backend Architecture

The backend is built using FastAPI and follows a modular architecture:

```
backend/
├── app/
│   ├── api/                 # API endpoints
│   │   ├── anomaly.py      # Anomaly control endpoints
│   │   ├── metrics.py      # Metrics collection endpoints
│   │   └── workload.py     # Workload management endpoints
│   ├── core/               # Core functionality
│   │   ├── config.py       # Configuration management
│   │   └── logging.py      # Logging setup
│   ├── schemas/            # Pydantic models
│   │   ├── anomaly.py      # Anomaly-related schemas
│   │   ├── metrics.py      # Metrics-related schemas
│   │   └── workload.py     # Workload-related schemas
│   ├── services/           # Business logic
│   │   ├── anomaly_service.py    # Anomaly detection/injection
│   │   ├── diagnosis_service.py  # Root cause analysis
│   │   ├── k8s_service.py        # Kubernetes operations
│   │   ├── metrics_service.py    # Metrics collection
│   │   └── workload_service.py   # Workload management
│   └── main.py            # Application entry point
├── tests/                 # Test cases
├── requirements.txt       # Python dependencies
└── README.md             # Documentation
```

### Component Interactions

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   FastAPI   │     │  Services   │     │  External   │
│  Endpoints  │     │   Layer     │     │  Systems    │
├─────────────┤     ├─────────────┤     ├─────────────┤
│ /anomaly    │────▶│ Anomaly     │────▶│ Kubernetes  │
│ /metrics    │────▶│ Metrics     │────▶│ OceanBase   │
│ /workload   │────▶│ Workload    │────▶│ Tsar/Obdiag │
└─────────────┘     └─────────────┘     └─────────────┘
```

### Key Components

1. **API Layer** (`app/api/`)
   - RESTful endpoints using FastAPI
   - Request/response handling
   - Input validation
   - Error handling

2. **Service Layer** (`app/services/`)
   - `workload_service.py`: Manages database workloads (Sysbench, TPC-C, TPC-H)
   - `metrics_service.py`: Collects and processes system metrics
   - `diagnosis_service.py`: Performs root cause analysis
   - `k8s_service.py`: Handles Kubernetes operations
   - `anomaly_service.py`: Controls anomaly injection

3. **Schema Layer** (`app/schemas/`)
   - Data models using Pydantic
   - Request/response validation
   - Type checking
   - Documentation generation

4. **Core Layer** (`app/core/`)
   - Application configuration
   - Logging setup
   - Common utilities

### Data Flow

1. **Workload Management**:
   ```
   Client → API → WorkloadService → Database
     ↳ Prepare database
     ↳ Start workload
     ↳ Monitor metrics
     ↳ Stop workload
   ```

2. **Anomaly Detection**:
   ```
   MetricsService → Metrics Collection → DiagnosisService
     ↳ Real-time monitoring
     ↳ Anomaly detection
     ↳ Root cause analysis
     ↳ Alert generation
   ```

3. **Metrics Collection**:
   ```
   OceanBase/System → MetricsService → Prometheus
     ↳ Performance metrics
     ↳ Resource usage
     ↳ Query statistics
     ↳ System health
   ```

## Prerequisites

- Python 3.8+
- Docker (for containerized deployment)
- Access to an OceanBase cluster
- Prometheus instance for metrics collection

## Workload Tools Setup

The backend supports three types of workloads: Sysbench OLTP, TPC-C, and TPC-H. Here's how to set up each:

### 1. Sysbench

On Ubuntu/Debian:
```bash
curl -s https://packagecloud.io/install/repositories/akopytov/sysbench/script.deb.sh | sudo bash
sudo apt -y install sysbench
```

On CentOS/RHEL:
```bash
curl -s https://packagecloud.io/install/repositories/akopytov/sysbench/script.rpm.sh | sudo bash
sudo yum -y install sysbench
```

### 2. TPC-C (using tpcc-mysql)

First, install required dependencies:
```bash
# On Ubuntu/Debian
sudo apt-get install -y libmysqlclient-dev libmysqlclient21 build-essential

# On CentOS/RHEL
sudo yum install -y mysql-devel gcc make
```

Then build tpcc-mysql:
```bash
# Clone the repository
git clone https://github.com/Percona-Lab/tpcc-mysql.git

# Build the binaries
cd tpcc-mysql/src
make

# Move binaries to the main directory
mv tpcc_load tpcc_start ..
cd ..
chmod +x tpcc_load tpcc_start
```

Note: Make sure `mysql_config` is available in your PATH. You can verify this by running:
```bash
which mysql_config
```

### 3. TPC-H

TPC-H support is coming soon.

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd backend
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables (create a `.env` file):
```env
OB_HOST=obproxy
OB_PORT=2881
OB_USER=root
OB_PASSWORD=your_password
PROMETHEUS_URL=http://prometheus:9090
```

## Running the Service

1. Start the development server:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

2. Access the API documentation:
```
http://localhost:8000/docs
```

## Using Workloads

### Sysbench OLTP
1. Prepare the database:
```bash
curl -X POST "http://localhost:8000/api/workload/prepare" -H "Content-Type: application/json" -d '{"type": "sysbench"}'
```

2. Start the workload:
```bash
curl -X POST "http://localhost:8000/api/workload/start" -H "Content-Type: application/json" -d '{"type": "sysbench", "threads": 4}'
```

### TPC-C
1. Prepare the database:
```bash
curl -X POST "http://localhost:8000/api/workload/prepare" -H "Content-Type: application/json" -d '{"type": "tpcc"}'
```

2. Start the workload:
```bash
curl -X POST "http://localhost:8000/api/workload/start" -H "Content-Type: application/json" -d '{"type": "tpcc", "threads": 4}'
```

### TPC-H (Coming Soon)
1. Prepare the database:
```bash
curl -X POST "http://localhost:8000/api/workload/prepare" -H "Content-Type: application/json" -d '{"type": "tpch"}'
```

2. Start the workload:
```bash
curl -X POST "http://localhost:8000/api/workload/start" -H "Content-Type: application/json" -d '{"type": "tpch", "threads": 1}'
```

## Monitoring

The service exposes Prometheus metrics at `/metrics` endpoint for monitoring the backend service itself.

## Testing

Run the test suite:
```bash
pytest
```

## License
