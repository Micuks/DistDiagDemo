# DBPecker

A distributed database anomaly detection and diagnosis system.

## Overview

DBPecker is a comprehensive diagnostic and workload management system for distributed databases. It provides real-time monitoring, anomaly detection, and root cause analysis capabilities.

## Features

- Real-time metrics monitoring and visualization
  - Statistical fluctuation detection with visual indicators
  - Intelligent formatting of metric values with appropriate units
  - Smart display of technical metrics with tooltips
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
- Advanced statistical analysis for detecting metric fluctuations
- Delta-based processing for cumulative metrics (delay metrics)

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
DBPecker/
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

## Recent Enhancements

### Enhanced Metrics Visualization

- **Fluctuation Detection and Display**
  - Visual indicators for metrics showing significant statistical fluctuations
  - Z-score and percentage change badges showing magnitude and direction of changes
  - Yellow highlighting for metrics exceeding statistically significant thresholds

- **Improved Value Formatting**
  - Thousands separators for all numeric values (e.g., 1,234,567)
  - Dynamic unit selection based on magnitude (KB → MB → GB)
  - Type-specific formatting for time, memory, and network metrics
  - Tooltips showing complete names for long metrics

- **UI Improvements**
  - Proper capitalization of technical terms (CPU, IO, MySQL, etc.)
  - Two-line display for long metric names with ellipsis
  - Responsive metric cards with consistent sizing

### Backend Improvements

- **Statistical Analysis Enhancements**
  - More accurate fluctuation detection using combined z-score and percentage change
  - Minimum history requirements to ensure statistical validity
  - Improved numerical stability for edge cases (near-zero variance)

- **Metric Processing Upgrades**
  - Delta calculation for cumulative counter metrics (CPU, IO, network)
  - Proper handling of microsecond-based delay metrics
  - Enhanced error handling and logging

- **Code Optimization**
  - Streamlined codebase with removal of redundant functions
  - More robust type checking and error prevention
  - Performance improvements for large metric datasets

## Supported Anomaly Types

DBPecker supports multiple types of anomalies that can be injected into the system for testing, training, and demonstration purposes. These anomalies are designed to simulate common issues encountered in distributed database environments.

### Resource Contention Anomalies

- **CPU Stress**
  - Simulates high CPU utilization scenarios
  - Uses StressChaos to generate CPU load with 32 worker threads at 100% load
  - Duration: 300 seconds

- **I/O Bottleneck**
  - Simulates disk I/O pressure on database nodes
  - Uses IOChaos to add 1000ms latency to all read/write operations
  - Targets OceanBase data storage path

- **Network Bottleneck**
  - Simulates network latency issues between database nodes
  - Uses NetworkChaos to inject 2000ms latency on all connections
  - Affects both inbound and outbound traffic

- **Cache Bottleneck**
  - Simulates memory pressure affecting database cache performance
  - Uses StressChaos to allocate 2GB of memory across 8 worker threads
  - Additionally adjusts OceanBase memstore_limit_percentage parameter to 20%

### Database-Specific Anomalies

- **Too Many Indexes**
  - Simulates performance degradation caused by excessive index creation
  - Creates multiple redundant indexes on both tpcc and sbtest databases
  - Implemented using direct SQL command execution rather than Chaos Mesh
  - Supports two database workloads:
    - TPC-C: Creates 34 indexes across 9 tables including customer, district, history, item, new_orders, order_line, orders, stock, and warehouse
    - Sysbench: Creates 30 indexes across 10 sbtest tables (sbtest1-sbtest10)
  - Proper cleanup mechanism to drop all created indexes when the anomaly is removed

Each anomaly can be injected through the Anomaly Control Panel in the web interface or via the API. For advanced scenarios, multiple anomalies can be combined to simulate complex failure patterns.

The anomaly system is designed to be extensible, allowing new anomaly types to be added with minimal code changes. All anomalies include proper tracking and cleanup mechanisms to ensure the system returns to a normal state after testing.

# DistDiagDemo

A distributed database diagnosis demonstration platform.

## Control Panel Refactoring - Completed Features

### Overview
The control panel has been refactored to provide a modern UI/UX with improved workflow for configuring and executing workloads and anomalies on distributed database nodes.

### Backend Enhancements
1. **Workload Service**
   - Added support for multiple configurable options for different workload types:
     - Sysbench: table size, number of tables, report interval, random type
     - TPC-C: warehouses, warmup time, running time, report interval
     - TPC-H: report interval and scale factor
   - Implemented node selection capabilities for targeting specific database nodes
   - Added detailed status reporting for running workloads

2. **Anomaly Service**
   - Created a comprehensive service for anomaly injection and monitoring
   - Implemented severity controls (1-10 scale) for fine-tuning anomaly intensity
   - Added support for targeting specific nodes for anomaly injection
   - Implemented automatic cleanup functionality for timed anomalies

3. **API Endpoints**
   - Added new endpoints for node discovery
   - Updated workload and anomaly endpoints to support enhanced configuration options
   - Improved error handling and response formatting

### Frontend Improvements
1. **Step-by-Step Workflow**
   - Implemented a three-step process: workload configuration, anomaly configuration, review & execution
   - Added clear navigation between steps with validation
   - Provided comprehensive summary view before execution

2. **Enhanced Configuration Options**
   - Created dynamic form controls that adapt based on workload type
   - Implemented node selection dropdown populated with available nodes
   - Added severity controls for anomalies with visual indicators

3. **React Hooks**
   - Created custom hooks (useWorkload, useAnomaly) for managing state and API interactions
   - Implemented data fetching with error handling and loading states
   - Added real-time status updates for active workloads and anomalies

### UI Components
1. **WorkloadConfig**
   - Basic configuration: workload type, threads, target node, database preparation
   - Advanced options: dynamic form based on selected workload type
   - Form validation with clear error messages

2. **AnomalyConfig**
   - Form for configuring new anomalies with type, target node, and severity
   - List view of configured anomalies with tags for type and severity
   - Delete functionality for removing configured anomalies

3. **ExecutionSummary**
   - Detailed view of configured workload and anomalies
   - Visual indicators for configuration options
   - Execute button with loading state and validation

### Technical Improvements
1. **Error Handling**
   - Comprehensive error handling throughout the stack
   - User-friendly error messages with suggestions
   - Automatic retry mechanisms for transient failures

2. **Performance Optimizations**
   - Efficient data fetching with cache control
   - Minimized re-renders using useCallback and useMemo
   - Optimized API response formats

## Usage
1. Navigate to the control panel
2. Configure your workload in step 1
3. Configure anomalies to inject in step 2
4. Review your configuration and execute in step 3
5. Monitor metrics and diagnoses in the dashboard

## Future Enhancements
- Support for custom workload scripts
- Saved configuration templates
- Batch execution of multiple scenarios
- Enhanced visualization of execution results
