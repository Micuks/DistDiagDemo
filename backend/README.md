# Backend - DistDiagDemo

A FastAPI-based backend for the Distributed Database Diagnosis Demo application. This service manages anomaly injection into OceanBase clusters using Chaos Mesh, collects metrics via Prometheus, and performs anomaly detection using machine learning models.

## Features

- Chaos Mesh integration for anomaly injection
- Prometheus metrics collection
- Machine learning-based anomaly detection
- RESTful API endpoints
- Kubernetes integration

## Prerequisites

- Python 3.8 or higher
- Kubernetes cluster with:
  - OceanBase deployed
  - Chaos Mesh installed
  - Prometheus set up
- Access to Kubernetes cluster (kubeconfig)

## Installation

1. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the backend directory:
```env
PROMETHEUS_URL=http://prometheus:9090
METRICS_WINDOW_MINUTES=30
MODEL_PATH=models/anomaly_detector
CHAOS_MESH_NAMESPACE=chaos-testing
```

## Development

To start the development server:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

## API Documentation

Once the server is running, you can access:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Project Structure

```
backend/
├── app/
│   ├── api/              # API endpoints
│   │   └── anomaly.py
│   ├── schemas/          # Pydantic models
│   │   └── anomaly.py
│   ├── services/         # Business logic
│   │   ├── k8s_service.py
│   │   ├── metrics_service.py
│   │   └── diagnosis_service.py
│   └── main.py          # FastAPI application
├── models/              # ML models
└── requirements.txt     # Project dependencies
```

## API Endpoints

- `POST /api/anomaly/start` - Start an anomaly experiment
  - Body: `{"type": "cpu_stress"}`
  - Available types: cpu_stress, memory_stress, network_delay, disk_stress

- `POST /api/anomaly/stop` - Stop all running anomalies

- `GET /api/metrics` - Get system metrics
  - Returns CPU, memory, and network metrics

- `GET /api/anomaly/ranks` - Get anomaly detection results
  - Returns anomaly ranks over time

## Chaos Mesh Experiments

The backend supports the following Chaos Mesh experiments:

1. CPU Stress:
   - Stresses CPU with 100% load on one core

2. Memory Stress:
   - Consumes 256MB of memory

3. Network Delay:
   - Adds 100ms latency to network traffic

4. Disk Stress:
   - Adds 100ms latency to I/O operations

## Machine Learning Model

The anomaly detection uses a TensorFlow model with:
- Input features: CPU, Memory, Network metrics
- Output: Anomaly rank (0-1)
- Fallback to distance-based detection if model unavailable

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Troubleshooting

1. Kubernetes Connection Issues:
   - Ensure kubeconfig is properly configured
   - Check if KUBERNETES_SERVICE_HOST environment variable is set in cluster

2. Prometheus Connection Issues:
   - Verify PROMETHEUS_URL is correct
   - Check if Prometheus is accessible from the backend

3. Model Loading Issues:
   - Ensure model files exist in MODEL_PATH
   - Check if TensorFlow is properly installed 