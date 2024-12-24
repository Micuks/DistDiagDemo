# Frontend - DistDiagDemo

A React-based frontend for the Distributed Database Diagnosis Demo application. This interface allows users to inject anomalies into an OceanBase cluster and monitor system metrics and anomaly detection results in real-time.

## Features

- Anomaly injection control panel
- Real-time system metrics visualization
- Anomaly rank monitoring
- Dark theme UI for better visibility
- Responsive design

## Prerequisites

- Node.js (v14 or higher)
- npm or yarn
- Access to the backend API

## Installation

1. Install dependencies:
```bash
npm install
```

2. Create a `.env` file in the frontend directory:
```env
REACT_APP_API_URL=http://localhost:8000  # Replace with your backend API URL
```

## Development

To start the development server:

```bash
npm start
```

The application will be available at `http://localhost:3000`.

## Building for Production

To create a production build:

```bash
npm run build
```

The build artifacts will be stored in the `build/` directory.

## Project Structure

```
frontend/
├── src/
│   ├── components/        # React components
│   │   └── AnomalyDashboard.js
│   ├── services/         # API services
│   │   └── anomalyService.js
│   └── App.js           # Main application component
├── public/              # Static files
└── package.json         # Project dependencies
```

## API Integration

The frontend communicates with the backend through the following endpoints:

- `POST /api/anomaly/start` - Start an anomaly experiment
- `POST /api/anomaly/stop` - Stop running anomalies
- `GET /api/metrics` - Get system metrics
- `GET /api/anomaly/ranks` - Get anomaly detection results

## Available Anomaly Types

- CPU Stress
- Memory Stress
- Network Delay
- Disk Stress

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request 