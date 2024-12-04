# DistDiagDemo: Distributed Database Anomaly Detection

A demonstration application for detecting and diagnosing anomalies in distributed database systems, inspired by the GRANO methodology. This project showcases advanced techniques in distributed anomaly detection, root cause analysis, and visualization.

## Features

- Real-time anomaly detection using machine learning models
- Graph-based root cause analysis with Neo4j
- Interactive visualization dashboard
- Multi-model anomaly detection (XGBoost + LSTM)
- Real-time metric collection and processing
- Automated notification system
- Kubernetes-based OceanBase cluster deployment

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
├── k8s/                 # Kubernetes manifests
│   ├── base/           # Base configurations
│   └── overlays/       # Environment-specific overlays
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
- Kubernetes tools (for K8s deployment)
  - kubectl
  - kind (for local deployment)

## Deployment Options

### 1. Docker Compose (Development)

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
   - Frontend Dashboard: http://localhost:13000
   - API Documentation: http://localhost:18000/docs
   - Neo4j Browser: http://localhost:7474

### 2. Kubernetes Deployment

#### Local Single-Server Deployment

1. Set up local Kubernetes cluster using Kind:
   ```bash
   ./scripts/setup-local-cluster.sh
   ```

2. Deploy OceanBase in development mode (single node):
   ```bash
   ./scripts/deploy-k8s.sh dev
   ```

3. Verify the deployment:
   ```bash
   kubectl -n oceanbase get pods
   kubectl -n oceanbase get services
   ```

#### Production Multi-Node Deployment

1. Ensure you have a Kubernetes cluster with sufficient resources:
   - Minimum 3 worker nodes
   - Each node: 16GB RAM, 4 CPU cores, 100GB storage
   - Container runtime (Docker/containerd)

2. Deploy OceanBase in production mode (3 nodes):
   ```bash
   ./scripts/deploy-k8s.sh prod
   ```

3. Verify the deployment:
   ```bash
   kubectl -n oceanbase get pods
   kubectl -n oceanbase get services
   ```

4. Check cluster status:
   ```bash
   kubectl -n oceanbase exec -it oceanbase-0 -- mysql -h127.0.0.1 -P2881 -uroot -ppassword123 -e "SELECT * FROM oceanbase.gv\$observer;"
   ```

#### Kubernetes Deployment Architecture

The production deployment consists of:

1. **Controller Node**
   - Manages cluster coordination
   - Handles metadata operations
   - Runs on the first pod (oceanbase-0)

2. **Data Nodes** (3 replicas)
   - Store and process data
   - Handle SQL queries
   - Automatic failover support
   - Distributed across pods (oceanbase-[0,1,2])

3. **Services**
   - Headless service for inter-node communication
   - LoadBalancer service for external access
   - NodePort service for local development

4. **Storage**
   - Persistent volumes for data storage
   - Automatic volume provisioning
   - Data replication across nodes

5. **Configuration**
   - ConfigMaps for OceanBase settings
   - Secrets for sensitive data
   - Environment-specific overlays

#### Resource Requirements

| Environment | Nodes | CPU/Node | RAM/Node | Storage/Node |
|-------------|-------|----------|----------|--------------|
| Development | 1     | 2 cores  | 4GB      | 20GB        |
| Production  | 3     | 4 cores  | 16GB     | 100GB       |

#### Accessing the Cluster

1. Get the cluster endpoint:
   ```bash
   kubectl -n oceanbase get service oceanbase -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
   ```

2. Connect using MySQL client:
   ```bash
   mysql -h<CLUSTER_IP> -P2881 -uroot -ppassword123
   ```

3. Monitor the cluster:
   ```bash
   # Check pod status
   kubectl -n oceanbase get pods -o wide

   # View pod logs
   kubectl -n oceanbase logs -f oceanbase-0

   # Get cluster status
   kubectl -n oceanbase exec -it oceanbase-0 -- mysql -h127.0.0.1 -P2881 -uroot -ppassword123 \
     -e "SELECT zone,svr_ip,svr_port,status FROM oceanbase.gv\$observer;"
   ```

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
