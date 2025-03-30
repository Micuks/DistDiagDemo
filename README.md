# DBPecker

Distributed Database Anomaly Detection & Diagnosis Platform

## Overview

DBPecker is a sophisticated platform designed to monitor, detect, and diagnose anomalies in distributed database environments. With real-time metrics visualization, automated anomaly detection, and streamlined diagnosis workflows, DBPecker helps administrators maintain optimal performance and stability in their databases.

## Setup

### Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.8 or higher
- Node.js 16 or higher (with pnpm installed)
- Kubernetes and kubectl for cluster management
- ob-operator and Chaos Mesh (refer to their official manuals for installation)
- Toda and stress-ng for pod resource stress testing (refer to their respective official documentation)

### Step 0: Install ob-operator and Chaos Mesh

Follow the official documentation to install these components:

- [ob-operator Documentation](https://github.com/oceanbase/ob-operator)  <!-- Replace with actual link -->
- [Chaos Mesh Documentation](https://chaos-mesh.org/docs/)

### Step 1: Bring Up the OceanBase Cluster

1. Create a Kubernetes cluster configured with 3 nodes.
2. Deploy the OceanBase cluster:
   - Apply the cluster manifest by running:
     ```bash
     kubectl apply -f k8s/obcluster.yaml
     ```
3. Configure the OceanBase proxy:
   - Open the file `k8s/obproxy.yaml`.
   - Modify the `RS_LIST` field to include the actual IP:port for each of the 3 nodes.
4. Apply the proxy configuration:
   ```bash
   kubectl apply -f k8s/obproxy.yaml
   ```

### Step 2: Launch the Backend

1. Navigate to the backend directory.
2. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the backend server with auto-reload enabled:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
   ```

### Step 3: Launch the Frontend

1. Navigate to the frontend directory.
2. Install node dependencies using pnpm:
   ```bash
   pnpm install
   ```
3. Start the frontend development server:
   ```bash
   pnpm run dev
   ```
4. Open your web browser and go to [http://localhost:3000](http://localhost:3000) to access the application.

## Usage

Once the system is up and running, you can:

- Monitor real-time metrics of your OceanBase cluster.
- Configure and inject anomalies to test and validate system behavior.
- Access detailed anomaly diagnostics and root cause analysis through an intuitive control panel.

Use the interface to manage workloads, observe system performance, and ensure your distributed database environment remains robust.

## Additional Information

For more details, please refer to the following resources:

- Official ob-operator documentation
- Official Chaos Mesh documentation

## License

This project is licensed under the MIT License. See the LICENSE file for details.
