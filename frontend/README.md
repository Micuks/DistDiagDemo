# DistDiagDemo Frontend

A React-based frontend application for monitoring and diagnosing OceanBase clusters. This application provides real-time metrics visualization, anomaly detection, and workload management capabilities.

## Features

- Real-time metrics dashboard
- Database and tenant-level monitoring
- Workload management interface
- Anomaly detection visualization
- Interactive charts and graphs
- Dark/light theme support

## Prerequisites

- Node.js 16+
- npm or yarn
- Access to the DistDiagDemo backend service

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd frontend
```

2. Install dependencies:
```bash
npm install
# or
yarn install
```

3. Create a `.env` file:
```env
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
```

## Development

Start the development server:
```bash
npm run dev
# or
yarn dev
```

The application will be available at `http://localhost:5173`.

## Building for Production

Build the application:
```bash
npm run build
# or
yarn build
```

## Features Overview

### Metrics Dashboard

The metrics dashboard provides real-time visualization of:

- Database Metrics:
  - QPS (Queries Per Second)
  - TPS (Transactions Per Second)
  - Active Sessions
  - SQL Response Time
  - Disk I/O
  - Memory Usage
  - Cache Hit Ratio
  - Slow Queries
  - Deadlocks
  - Replication Lag
  - Connection Count

- Tenant Metrics:
  - CPU Usage
  - Memory Usage
  - Disk Usage
  - IOPS
  - Session Count
  - Active Sessions

### Workload Management

The workload management interface allows you to:

1. Prepare the database for benchmarking:
```typescript
await api.post('/api/workload/prepare');
```

2. Start a new workload:
```typescript
await api.post('/api/workload/start', {
  workload_type: 'oltp_read_write',
  threads: 4
});
```

3. Monitor active workloads:
```typescript
const workloads = await api.get('/api/workload/active');
```

4. Stop workloads:
```typescript
await api.post(`/api/workload/${workloadId}/stop`);
```

### Anomaly Detection

The anomaly detection interface provides:

1. Real-time anomaly detection visualization
2. Historical anomaly data
3. Root cause analysis
4. Anomaly injection controls for testing

## Project Structure

```
frontend/
├── src/
│   ├── components/        # React components
│   │   ├── Dashboard/
│   │   ├── Metrics/
│   │   ├── Workload/
│   │   └── Anomaly/
│   ├── services/         # API services
│   ├── hooks/           # Custom React hooks
│   ├── utils/           # Utility functions
│   ├── types/           # TypeScript types
│   └── App.tsx         # Main application
├── public/             # Static assets
└── index.html         # HTML template
```

## Component Usage

### MetricsDashboard

```jsx
import { MetricsDashboard } from './components/Dashboard';

function App() {
  return (
    <MetricsDashboard 
      refreshInterval={5000}
      showTenantMetrics={true}
    />
  );
}
```

### WorkloadManager

```jsx
import { WorkloadManager } from './components/Workload';

function App() {
  return (
    <WorkloadManager
      onWorkloadStart={(workload) => console.log('Started:', workload)}
      onWorkloadStop={(workloadId) => console.log('Stopped:', workloadId)}
    />
  );
}
```

### AnomalyDetector

```jsx
import { AnomalyDetector } from './components/Anomaly';

function App() {
  return (
    <AnomalyDetector
      refreshInterval={10000}
      threshold={0.8}
      onAnomalyDetected={(anomaly) => console.log('Detected:', anomaly)}
    />
  );
}
```

## API Integration

The frontend communicates with the backend using the following services:

```typescript
// services/api.ts
export const api = {
  metrics: {
    getDatabase: () => fetch('/api/metrics/database'),
    getTenant: (tenant?: string) => fetch(`/api/metrics/tenant?tenant_name=${tenant}`),
  },
  workload: {
    prepare: () => fetch('/api/workload/prepare', { method: 'POST' }),
    start: (params: WorkloadParams) => fetch('/api/workload/start', {
      method: 'POST',
      body: JSON.stringify(params),
    }),
    stop: (workloadId: string) => fetch(`/api/workload/${workloadId}/stop`, {
      method: 'POST',
    }),
  },
  anomaly: {
    start: (type: string) => fetch('/api/anomaly/start', {
      method: 'POST',
      body: JSON.stringify({ type }),
    }),
    stop: () => fetch('/api/anomaly/stop', { method: 'POST' }),
  },
};
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

[Your License] 