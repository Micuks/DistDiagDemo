# DistDiagDemo Frontend

A modern React-based frontend for the DistDiagDemo project, providing an intuitive interface for database workload management, anomaly detection, and system monitoring.

## Features

- Real-time system metrics visualization
- Interactive workload management (Sysbench, TPC-C, TPC-H)
- Anomaly detection and monitoring
- Dynamic charts and graphs using Recharts
- Material-UI based responsive design
- Fluctuation-aware metrics visualization with visual indicators
- Enhanced formatting for numeric values with thousands separators
- Intelligent display of units based on metric values
- Tooltip-enabled metric name display for long names
- Smart handling of technical abbreviations (CPU, IO, MySQL, etc.)

## Frontend Architecture

The frontend is built using React with Vite and follows a component-based architecture:

```
frontend/
├── src/
│   ├── components/           # React components
│   │   ├── AnomalyDashboard.jsx    # Main dashboard component
│   │   ├── MetricsChart.jsx        # Metrics visualization
│   │   ├── WorkloadControl.jsx     # Workload management
│   │   └── common/                 # Reusable components
│   │       ├── Button.jsx
│   │       ├── Chart.jsx
│   │       └── Alert.jsx
│   ├── services/             # API integration
│   │   ├── anomalyService.js      # Anomaly-related API calls
│   │   ├── metricsService.js      # Metrics-related API calls
│   │   └── workloadService.js     # Workload-related API calls
│   ├── styles/               # Styling
│   │   ├── theme.js              # MUI theme configuration
│   │   └── global.css           # Global styles
│   ├── utils/                # Utility functions
│   │   ├── formatters.js         # Data formatting
│   │   └── constants.js          # Constants
│   ├── index.jsx             # Application entry point
│   └── index.html            # HTML template
├── public/                   # Static assets
├── vite.config.js           # Vite configuration
└── package.json             # Dependencies and scripts
```

### Component Architecture

```
┌──────────────────────────────────────┐
│            AnomalyDashboard          │
├──────────────────┬──────────────────┤
│   MetricsChart   │  WorkloadControl │
├──────────────────┴──────────────────┤
│        Common Components            │
└──────────────────────────────────────┘
```

### Data Flow

```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   React UI   │ ←──→ │   Services   │ ←──→ │ Backend API  │
└──────────────┘      └──────────────┘      └──────────────┘
       ↑                     ↑                     ↑
       │                     │                     │
    User Input        API Integration        Data Processing
```

### Key Components

1. **AnomalyDashboard** (`components/AnomalyDashboard.jsx`)
   - Main application container
   - State management for workloads and metrics
   - Real-time data updates
   - Error handling

2. **Services** (`services/`)
   - RESTful API integration
   - WebSocket connections for real-time updates
   - Error handling and retry logic
   - Response transformation

3. **Charts and Visualization** (`components/MetricsChart.jsx`)
   - Real-time metrics plotting
   - Interactive data visualization
   - Time-series data handling
   - Custom tooltips and legends

4. **Workload Management** (`components/WorkloadControl.jsx`)
   - Workload type selection
   - Parameter configuration
   - Start/stop controls
   - Status monitoring

### State Management

```
Component State (React Hooks)
├── Workload State
│   ├── Active workloads
│   ├── Workload parameters
│   └── Status and metrics
├── Anomaly State
│   ├── Detection status
│   ├── Anomaly types
│   └── Alert history
└── System Metrics
    ├── Performance data
    ├── Resource usage
    └── Historical trends
```

## Prerequisites

- Node.js 16+
- pnpm (recommended) or npm
- Modern web browser

## Installation

1. Install dependencies:
```bash
pnpm install
```

2. Set up environment variables (create `.env`):
```env
VITE_API_URL=http://localhost:8000
```

## Development

1. Start the development server:
```bash
pnpm dev
```

2. Build for production:
```bash
pnpm build
```

3. Preview production build:
```bash
pnpm preview
```

## Project Structure Details

### Components

1. **AnomalyDashboard**
   - Main dashboard layout
   - Real-time metrics display
   - Workload management interface
   - Anomaly detection controls

2. **MetricsChart**
   - Line charts for system metrics
   - CPU, memory, I/O visualization
   - Custom tooltips and legends
   - Time range selection
   - Historical trend comparison

3. **MetricsPanel**
   - Fluctuation-aware metric tiles with visual indicators
   - Responsive grid layout with intelligent overflow handling
   - Statistical significance indicators (z-scores, percentage changes)
   - Interactive tooltips showing full metric names
   - Visual highlighting for anomalous metrics

4. **WorkloadControl**
   - Workload type selection
   - Parameter configuration
   - Start/stop functionality
   - Status monitoring

### Services

1. **anomalyService.js**
   - Start/stop anomaly injection
   - Fetch anomaly detection results
   - Real-time monitoring

2. **metricsService.js**
   - System metrics collection
   - Performance data aggregation
   - Historical data retrieval

3. **workloadService.js**
   - Workload management
   - Database preparation
   - Status monitoring
   - Results collection

## Recent Enhancements

### Metrics Visualization Improvements

1. **Fluctuation Detection**
   - Visual indicators for metrics experiencing significant fluctuations
   - Yellow background and left border highlight unstable metrics
   - Z-score and percentage change badges show magnitude and direction
   
2. **Formatting Enhancements**
   - Thousands separators for all numeric values (e.g., 1,234,567)
   - Dynamic unit selection based on value magnitude (KB → MB → GB)
   - Proper handling of different metric types:
     - Time values: nanoseconds → seconds (properly scaled)
     - Memory: bytes → MB/GB (with appropriate conversions)
     - Delay metrics: microseconds → milliseconds (fixed from previous incorrect nanosecond assumption)
     - Count metrics: whole numbers with thousands separators

3. **UI/UX Improvements**
   - Tooltips for long metric names
   - Proper capitalization of technical terms (CPU, IO, MySQL, etc.)
   - Two-line display for metric names with proper truncation
   - Responsive layout ensuring consistent metric tile sizes

### Technical Fixes

1. **Unit Conversion Accuracy**
   - Corrected delay metric handling from nanoseconds to microseconds
   - Improved conversion logic for all metric types
   
2. **Object Type Handling**
   - Enhanced `getMetricSuffix` to handle both string and object metric formats
   - Fixed "metric.includes is not a function" error

3. **Display Optimizations**
   - Text overflow handling with tooltips
   - Consistent vertical alignment
   - Improved contrast and readability

## Styling

The project uses Material-UI (MUI) with a custom theme for consistent styling:

```javascript
// theme.js
const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
  // ... other theme configurations
});
```

## Testing

Run the test suite:
```bash
pnpm test
```

## License

MIT License
