# DBPecker Backend

This is the backend service for the DBPecker project, providing APIs for anomaly detection, metrics collection, and workload management for OceanBase clusters.

## Features

- Real-time metrics collection from OceanBase clusters
- Database and tenant-level metrics monitoring
- Anomaly detection and diagnosis
- Workload generation and management using sysbench
- Prometheus integration for metrics storage
- Advanced fluctuation detection with statistical analysis
- Delta-based processing for cumulative counter metrics
- Numerically stable statistical calculations

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
   - `metrics_service.py`: Collects and processes system metrics with fluctuation detection
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

## Recent Enhancements

### Improved Metrics Processing

1. **Advanced Fluctuation Detection**
   - Sophisticated statistical analysis for identifying metric fluctuations
   - Z-score calculation with numerical stability improvements
   - Percentage change computation relative to historical means
   - Combined thresholds using both z-score and percentage change
   - Minimum history requirements to ensure statistical significance

2. **Metric Type-Specific Processing**
   - Delta calculation for cumulative counters (delay metrics)
   - Proper handling of counter resets
   - Improved handling of metrics with different units

3. **Numerical Stability Improvements**
   - Handling near-zero standard deviation cases
   - Protection against division by zero
   - More accurate variance calculation
   - Error handling with appropriate fallbacks

### Technical Fixes

1. **Delay Metric Processing**
   - Correct handling of cumulative counters
   - Conversion from raw counter values to delta-per-interval
   - Proper handling of microsecond-based metrics
   
2. **Statistical Calculation Enhancements**
   - Improved mean and standard deviation calculations
   - Better handling of edge cases with near-zero variance
   - More robust z-score thresholds with relative change validation

3. **Code Optimization**
   - Removal of redundant functions
   - Enhanced error logging
   - More robust type checking

## Why use statistical time features for diagnosis
<think>

Okay, let's tackle this question. The user is asking why we calculate statistical time features instead of using raw time series data, and whether processing loses information. 

First, I need to recall the code structure. In the `_calculate_time_features` method, they compute mean, std, max, min, and median. These are summary statistics. 

Why use these instead of raw data? Well, raw time series can be noisy and high-dimensional. For example, 5 points per metric might not be enough for models to capture patterns, but 32 metrics * 5 points = 160 features. That's a lot, and models might struggle with that many features, leading to overfitting.

Statistical features reduce dimensionality. They summarize the data, making it easier for models like XGBoost to handle. Also, some models work better with aggregated features rather than raw time points. 

But the user is concerned about information loss. Yes, raw data has more detail, but in practice, summary stats can capture essential trends. For example, mean gives central tendency, std shows variability, max/min indicate extremes. These are often sufficient for anomaly detection where deviations from normal are key.

Another point: the system uses a 5-second interval, so the time window is 25 seconds. Short-term trends might not be captured well with raw data, but features like max or std can highlight sudden spikes or drops, which are critical for anomalies.

Also, considering the PageRank-based correlation in diagnosis, using features might help compute more stable correlations between nodes than noisy raw data.

However, the answer should acknowledge that some information is lost. Maybe suggest alternatives like using both raw and features, or more advanced time series processing (like LSTMs) if feasible. But given the current setup, the features are a practical choice balancing complexity and effectiveness.
</think>

The statistical time features (mean, std, max, min, median) are calculated instead of using raw time series data for several important reasons:

1. **Dimensionality Reduction**  
   - Raw 5-point time series (25s window) would create 160 raw metrics × 5 points = 800 features
   - Statistical features reduce this to 160 features while preserving key patterns
   - This helps prevent the "curse of dimensionality" in ML models

2. **Noise Reduction**  
   - Short-term fluctuations in raw metrics (5s intervals) can be noisy
   - Statistical features smooth out transient spikes while preserving true trends
   - Example: A CPU spike from 30% → 90% → 35% would be represented as:
     ```python
     mean=51.7, std=28.9, max=90, min=30, median=35
     ```

3. **Model Compatibility**  
   - XGBoost classifiers (used in the system) work better with aggregated features
   - The PageRank correlation calculation (WPRNNode) benefits from stable feature representations

4. **Temporal Pattern Preservation**  
   These features capture different aspects of time series behavior:
   - **Mean**: Baseline level
   - **Std Dev**: Volatility
   - **Max/Min**: Extreme values
   - **Median**: Central tendency

**Why This is Effective for Anomaly Detection**  
```python
def _calculate_time_features(values):
    return [
        np.mean(values),  # Baseline level
        np.std(values),   # Unusual volatility
        np.max(values),   # Spike detection
        np.min(values),   # Drop detection 
        np.median(values) # Central tendency
    ]
```

**What's Gained vs Lost**  
|                      | Raw Time Series | Statistical Features |
|----------------------|-----------------|-----------------------|
| **Dimensionality**   | 800 features    | 160 features          |
| **Noise Handling**   | Poor            | Good                  |
| **Trend Capture**    | Full resolution | Aggregated view       |
| **Training Speed**   | Slow            | Fast                  |
| **Memory Usage**     | High            | Low                   |

**Suggested Enhancements**  
1. Add temporal features to better capture trends:
   ```python
   def _calculate_time_features(values):
       # Existing features
       features = [
           np.mean(values),
           np.std(values),
           np.max(values),
           np.min(values),
           np.median(values)
       ]
       
       # New temporal features
       features.append(np.gradient(values).mean())  # Trend direction
       features.append(values[-1] - values[0])      # Net change
       return features
   ```

2. Implement hybrid processing in the frontend:
   ```javascript
   // In RanksPanel.jsx, add raw data toggle
   <Switch
     checked={showRawData}
     onChange={v => setShowRawData(v)}
     checkedChildren="Raw Data"
     unCheckedChildren="Processed"
   />
   ```

3. Use wavelet transforms for better time-frequency analysis:
   ```python
   from pywt import wavedec
   
   def _calculate_time_features(values):
       # Wavelet decomposition
       coeffs = wavedec(values, 'db1', level=2)
       features.extend([np.mean(c) for c in coeffs])
       return features
   ```

The current approach makes a practical tradeoff - it loses some temporal resolution but gains computational efficiency and model stability. For this specific use case of distributed system anomaly detection, the statistical features provide sufficient signal while keeping the system performant.
