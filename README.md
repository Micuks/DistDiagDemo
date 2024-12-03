# DistDiagDemo: Distributed Database Anomaly Detection

A demonstration application for detecting and diagnosing anomalies in distributed database systems, inspired by the GRANO methodology. This project showcases advanced techniques in distributed anomaly detection, root cause analysis, and visualization.

## Features

- Real-time anomaly detection using machine learning models
- Graph-based root cause analysis with Neo4j
- Interactive visualization dashboard
- Multi-model anomaly detection (XGBoost + LSTM)
- Real-time metric collection and processing
- Automated notification system

## Project Structure

```
.
├── anomaly_graph/          # Graph-based analysis components
├── backend/               # FastAPI backend service
├── config/               # Configuration files
├── data/                 # Data storage and processing
├── detection_layer/      # ML models and detection logic
├── docker/              # Docker configuration
├── docs/                # Documentation
├── frontend/            # React frontend application
├── models/              # Trained model artifacts
├── notifications/       # Alert and notification system
├── scripts/            # Utility scripts
└── tests/              # Test suites
```

## Prerequisites

- Docker and Docker Compose
- Python 3.8+
- Node.js 16+
- Neo4j 4.4+

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/DistDiagDemo.git
   cd DistDiagDemo
   ```

2. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configurations
   ```

3. Start the services:
   ```bash
   docker-compose up --build
   ```

4. Access the applications:
   - Frontend Dashboard: http://localhost:3000
   - API Documentation: http://localhost:8000/docs
   - Neo4j Browser: http://localhost:7474

## Development Setup

1. Backend Development:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   pip install -r requirements.txt
   uvicorn app.main:app --reload
   ```

2. Frontend Development:
   ```bash
   cd frontend
   npm install
   npm start
   ```

## Documentation

Detailed documentation is available in the `docs/` directory:
- [Architecture Overview](docs/architecture.md)
- [API Documentation](docs/api.md)
- [Development Guide](docs/development.md)
- [Deployment Guide](docs/deployment.md)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
