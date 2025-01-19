# DistDiagDemo

A distributed database anomaly detection and diagnosis system.

## Overview

DistDiagDemo is a comprehensive diagnostic and workload management system for distributed databases. It provides real-time monitoring, anomaly detection, and root cause analysis capabilities.

## Features

- Real-time metrics monitoring and visualization
- Anomaly injection for testing and validation
- Workload management and performance optimization
  - Multiple workload types support:
    - Sysbench OLTP workloads
    - TPC-C (OLTP benchmark)
    - TPC-H (OLAP benchmark)
  - Configurable workload parameters
  - Real-time workload monitoring
  - Automated database preparation
- Automated anomaly detection and diagnosis using DistDiagnosis
- Support for both `obdiag` and `psutil` metrics collection

## Environment Setup

### Prerequisites

- Python 3.8+
- Node.js 16+
- Docker (for running OceanBase and Chaos Mesh)

### Backend Setup

1. Install Python dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. Configure the `obdiag` command (if using):
   ```bash
   export OBDIAG_CMD='your_obdiag_command'  # Default is 'obdiag'
   ```

3. Start the backend server:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8001
   ```

### Frontend Setup

1. Install Node.js dependencies:
   ```bash
   cd frontend
   npm install
   ```

2. Start the frontend development server:
   ```bash
   npm run dev
   ```

## Training and Using DistDiagnosis

DistDiagnosis is an integrated anomaly detection and diagnosis system that combines machine learning with distributed system knowledge. It uses XGBoost classifiers for anomaly detection and PageRank-based analysis for root cause identification.

### Data Collection

1. Normal State Data Collection:
   - Use the "Start Normal Collection" button to collect metrics during normal operations
   - The system will collect metrics data under various conditions:
     - Different times of day
     - Different workload intensities
     - Different database operations (OLTP/OLAP)
   - Click "Stop Normal Collection" when sufficient normal state data is gathered
   - This baseline data helps the model accurately identify anomalies

2. Anomaly Data Collection:
   - Enable "Training Data Collection" switch before injecting anomalies
   - The system will collect:
     - Normal state metrics before anomaly injection
     - Metrics during the anomaly period
     - Normal state metrics after anomaly clearance
   - Collect data for different types of anomalies:
     - CPU Stress: High CPU utilization scenarios
     - Memory Stress: Memory pressure situations
     - IO Stress: Disk I/O bottlenecks
     - Network Delay: Network latency issues

3. Dataset Balance:
   - The system tracks the balance between normal and anomaly data
   - Dataset statistics are displayed in real-time:
     - Total number of samples
     - Number of normal state samples
     - Number of anomaly samples
   - A visual progress bar shows the ratio of normal to anomaly data
   - The system considers a dataset balanced when:
     - The difference between normal and anomaly ratios is within 30%
     - Example: 40% normal state data and 60% anomaly data is considered balanced
   - The balance indicator shows:
     - Green: Dataset is balanced
     - Red: Dataset needs more data in either category

4. Best Practices:
   - Collect normal state data under diverse operating conditions
   - For each anomaly type:
     - Allow sufficient time for normal state collection before injection
     - Keep the anomaly active long enough to gather representative data
     - Continue collecting normal state data after anomaly clearance
   - Monitor the dataset balance and collect additional data as needed
   - Aim for a roughly equal distribution between normal and anomaly states

### Training the Model

1. After collecting sufficient data, train the model using the API:
   ```bash
   curl -X POST http://localhost:8001/api/anomaly/train
   ```

2. The training process:
   - Processes collected metrics into feature vectors
   - Labels data based on injected anomalies
   - Trains XGBoost classifiers for each anomaly type
   - Saves the trained model for future use

### Using the System

1. Real-time Monitoring:
   - The system automatically collects metrics using either `obdiag` or `psutil`
   - Metrics are displayed in the Metrics Panel with interactive charts

2. Anomaly Detection:
   - The trained model continuously analyzes incoming metrics
   - Detected anomalies are displayed in the Ranks Panel
   - Each anomaly includes:
     - Type (CPU, Memory, IO, Network)
     - Node where it was detected
     - Confidence score
     - Timestamp

3. Root Cause Analysis:
   - The system uses PageRank-based analysis to identify root causes
   - Correlations between nodes are considered
   - Results are ranked by confidence and impact

### Model Performance

The system's accuracy depends on:
- Quality and quantity of training data
- Coverage of different anomaly scenarios
- Regular retraining with new data

For best results:
1. Collect training data under various conditions
2. Include both normal and anomalous states
3. Retrain periodically with new data
4. Validate detection accuracy with known anomalies

## Project Structure

```
DistDiagDemo/
├── backend/              # FastAPI backend service
│   ├── app/             # Application code
│   ├── tests/           # Test cases
│   └── README.md        # Backend documentation
├── frontend/            # React frontend application
│   ├── src/            # Source code
│   ├── public/         # Static assets
│   └── README.md       # Frontend documentation
├── k8s/                # Kubernetes deployment files
│   ├── base/           # Base configurations
│   └── overlays/       # Environment-specific configs
├── docs/               # Documentation
└── README.md           # Main documentation
```

## Quick Start

1. Start the backend server
2. Launch the frontend application
3. Access the web interface at `http://localhost:3000`
4. Use the Anomaly Control Panel to inject test anomalies
5. Monitor results in the Metrics and Ranks panels

## Documentation

- [Backend Documentation](backend/README.md)
- [Frontend Documentation](frontend/README.md)
- [API Documentation](http://localhost:8000/docs) (when backend is running)
- [Deployment Guide](docs/deployment.md)

## Development

### Backend Development
- FastAPI for REST API
- Pydantic for data validation
- SQLAlchemy for database operations
- Prometheus integration for metrics

### Frontend Development
- React with Vite
- Material-UI components
- Recharts for visualization
- Real-time updates with WebSocket

## License

MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

- [OceanBase](https://github.com/oceanbase/oceanbase) - The distributed database system
- [Sysbench](https://github.com/akopytov/sysbench) - Benchmark tool
- [TPC-C MySQL](https://github.com/Percona-Lab/tpcc-mysql) - TPC-C implementation
- [FastAPI](https://fastapi.tiangolo.com/) - Backend framework
- [React](https://reactjs.org/) - Frontend framework
