import React, { useState, useEffect } from 'react';
import {
  Container,
  Paper,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
  Typography,
  Snackbar,
  Alert,
  Divider,
  Stack,
  Chip,
  Box,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  CircularProgress,
  LinearProgress,
} from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { anomalyService } from '../services/anomalyService';
import { workloadService } from '../services/workloadService';

const AnomalyDashboard = () => {
  const [selectedAnomaly, setSelectedAnomaly] = useState('');
  const [metrics, setMetrics] = useState([]);
  const [anomalyRanks, setAnomalyRanks] = useState([]);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [activeWorkloads, setActiveWorkloads] = useState([]);
  const [workloadLoading, setWorkloadLoading] = useState(false);
  const [systemMetrics, setSystemMetrics] = useState({});
  const [activeAnomalies, setActiveAnomalies] = useState([]);

  const anomalyOptions = [
    { id: 'cpu_stress', name: 'CPU Stress' },
    { id: 'memory_stress', name: 'Memory Stress' },
    { id: 'network_delay', name: 'Network Delay' },
    { id: 'disk_stress', name: 'Disk Stress' },
  ];

  const workloadOptions = [
    { id: 'sysbench', name: 'Sysbench OLTP' },
    { id: 'tpcc', name: 'TPC-C' },
    { id: 'tpch', name: 'TPC-H' }
  ];

  const handleAnomalyChange = (event) => {
    setSelectedAnomaly(event.target.value);
  };

  const isAnomalyActive = (anomalyType) => {
    return activeAnomalies.some(anomaly => anomaly.type === anomalyType);
  };

  const handleAnomalyToggle = async (anomalyType) => {
    try {
      setLoading(true);
      if (isAnomalyActive(anomalyType)) {
        await anomalyService.stopAnomaly(anomalyType);
      } else {
        await anomalyService.startAnomaly(anomalyType);
      }
      const anomalies = await anomalyService.getActiveAnomalies();
      setActiveAnomalies(anomalies);
    } catch (err) {
      setError(err.message || 'Failed to toggle anomaly');
    } finally {
      setLoading(false);
    }
  };

  const handleStartWorkload = async (workloadType) => {
    try {
      setWorkloadLoading(true);
      await workloadService.startWorkload(workloadType);
      await fetchActiveWorkloads();
    } catch (err) {
      setError(err.message || 'Failed to start workload');
    } finally {
      setWorkloadLoading(false);
    }
  };

  const handlePrepareDatabase = async (workloadType) => {
    try {
      setWorkloadLoading(true);
      await workloadService.prepareDatabase(workloadType);
      setError('Database prepared successfully');
    } catch (err) {
      setError(err.message || 'Failed to prepare database');
    } finally {
      setWorkloadLoading(false);
    }
  };

  const handleStopWorkload = async (workloadId) => {
    try {
      setWorkloadLoading(true);
      await workloadService.stopWorkload(workloadId);
      await fetchActiveWorkloads();
    } catch (err) {
      setError(err.message || 'Failed to stop workload');
    } finally {
      setWorkloadLoading(false);
    }
  };

  const handleStopAllWorkloads = async () => {
    try {
      setWorkloadLoading(true);
      await workloadService.stopAllWorkloads();
      await fetchActiveWorkloads();
    } catch (err) {
      setError(err.message || 'Failed to stop all workloads');
    } finally {
      setWorkloadLoading(false);
    }
  };

  const handleStopAllAnomalies = async () => {
    try {
      setLoading(true);
      await anomalyService.stopAllAnomalies();
      const anomalies = await anomalyService.getActiveAnomalies();
      setActiveAnomalies(anomalies);
    } catch (err) {
      setError(err.message || 'Failed to stop all anomalies');
    } finally {
      setLoading(false);
    }
  };

  const fetchActiveWorkloads = async () => {
    try {
      const data = await workloadService.getActiveWorkloads();
      setActiveWorkloads(data.workloads || []);
      setSystemMetrics(data.systemMetrics || {
        cpu_usage: 0,
        memory_usage: 0,
        disk_usage: 0
      });
    } catch (err) {
      console.error('Failed to fetch active workloads:', err);
      setError(err.message);
    }
  };

  useEffect(() => {
    let intervalId;

    const fetchData = async () => {
      try {
        const [metricsData, ranksData, anomaliesData] = await Promise.all([
          anomalyService.getMetrics(),
          anomalyService.getAnomalyRanks(),
          anomalyService.getActiveAnomalies()
        ]);
        
        setMetrics(metricsData);
        setAnomalyRanks(ranksData);
        setActiveAnomalies(anomaliesData);
        await fetchActiveWorkloads();
      } catch (err) {
        setError(err.message || 'Failed to fetch data');
      }
    };

    fetchData();
    intervalId = setInterval(fetchData, 5000);

    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, []);

  return (
    <Container maxWidth="lg">
      <Grid container spacing={3}>
        {/* Anomaly Control Panel */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2.5 }}>
            <Typography variant="h6" gutterBottom>Anomaly Control</Typography>
            <Grid container spacing={2}>
              {anomalyOptions.map((option) => (
                <Grid item xs={12} sm={6} md={3} key={option.id}>
                  <Button
                    variant="contained"
                    color={isAnomalyActive(option.id) ? "error" : "primary"}
                    onClick={() => handleAnomalyToggle(option.id)}
                    fullWidth
                    disabled={loading}
                  >
                    {loading && option.id === selectedAnomaly ? 'Processing...' : 
                     isAnomalyActive(option.id) ? `Stop ${option.name}` : `Start ${option.name}`}
                  </Button>
                </Grid>
              ))}
              {activeAnomalies.length > 0 && (
                <Grid item xs={12}>
                  <Button
                    variant="contained"
                    color="error"
                    onClick={handleStopAllAnomalies}
                    fullWidth
                    disabled={loading}
                  >
                    Stop All Anomalies
                  </Button>
                </Grid>
              )}
            </Grid>

            {/* Active Anomalies Display */}
            {activeAnomalies.length > 0 && (
              <Box sx={{ mt: 2 }}>
                <Typography variant="subtitle2" gutterBottom>Active Anomalies:</Typography>
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Type</TableCell>
                        <TableCell>Target</TableCell>
                        <TableCell>Start Time</TableCell>
                        <TableCell>Status</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {activeAnomalies.map((anomaly, index) => (
                        <TableRow key={index}>
                          <TableCell>{anomaly.type}</TableCell>
                          <TableCell>{anomaly.target}</TableCell>
                          <TableCell>{new Date(anomaly.start_time).toLocaleString()}</TableCell>
                          <TableCell>
                            <Chip
                              label={anomaly.status}
                              color={anomaly.status === 'active' ? 'success' : 'default'}
                              size="small"
                            />
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Box>
            )}
          </Paper>
        </Grid>

        {/* Workload Control Panel */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2.5 }}>
            <Typography variant="h6" gutterBottom>Workload Control</Typography>
            
            {/* Workload Buttons */}
            <Grid container spacing={2} sx={{ mb: 3 }}>
              {workloadOptions.map((workload) => (
                <Grid item xs={12} sm={4} key={workload.id}>
                  <Stack spacing={1}>
                    <Typography variant="subtitle2">{workload.name}</Typography>
                    <Button
                      variant="contained"
                      color="primary"
                      onClick={() => handlePrepareDatabase(workload.id)}
                      disabled={workloadLoading}
                      fullWidth
                      size="small"
                    >
                      Prepare Database
                    </Button>
                    <Button
                      variant="contained"
                      color="success"
                      onClick={() => handleStartWorkload(workload.id)}
                      disabled={workloadLoading}
                      fullWidth
                    >
                      Start Workload
                    </Button>
                  </Stack>
                </Grid>
              ))}
            </Grid>

            {/* Stop All Workloads Button */}
            <Box sx={{ mb: 3 }}>
              <Button
                variant="contained"
                color="error"
                onClick={handleStopAllWorkloads}
                disabled={workloadLoading || activeWorkloads.length === 0}
                fullWidth
              >
                Stop All Workloads
              </Button>
            </Box>
            
            {/* System Metrics */}
            {Object.keys(systemMetrics).length > 0 && (
              <Box sx={{ mb: 2 }}>
                <Typography variant="subtitle2" gutterBottom>System Load:</Typography>
                <Grid container spacing={2}>
                  <Grid item xs={4}>
                    <Typography variant="body2" color="textSecondary">CPU Usage</Typography>
                    <LinearProgress 
                      variant="determinate" 
                      value={systemMetrics.cpu_usage} 
                      sx={{ mt: 1 }}
                    />
                    <Typography variant="body2" align="right">
                      {(systemMetrics.cpu_usage || 0).toFixed(1)}%
                    </Typography>
                  </Grid>
                  <Grid item xs={4}>
                    <Typography variant="body2" color="textSecondary">Memory Usage</Typography>
                    <LinearProgress 
                      variant="determinate" 
                      value={systemMetrics.memory_usage} 
                      sx={{ mt: 1 }}
                    />
                    <Typography variant="body2" align="right">
                      {(systemMetrics.memory_usage || 0).toFixed(1)}%
                    </Typography>
                  </Grid>
                  <Grid item xs={4}>
                    <Typography variant="body2" color="textSecondary">Disk I/O</Typography>
                    <LinearProgress 
                      variant="determinate" 
                      value={systemMetrics.disk_usage} 
                      sx={{ mt: 1 }}
                    />
                    <Typography variant="body2" align="right">
                      {(systemMetrics.disk_usage || 0).toFixed(1)}%
                    </Typography>
                  </Grid>
                </Grid>
              </Box>
            )}

            <Divider sx={{ my: 2 }} />
            
            {/* Active Workloads Table */}
            {activeWorkloads.length > 0 && (
              <Grid item xs={12}>
                <Typography variant="subtitle2" gutterBottom>Active Workloads:</Typography>
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Type</TableCell>
                        <TableCell>Status</TableCell>
                        <TableCell>Actions</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {activeWorkloads.map((workload) => (
                        <TableRow key={workload.id || `workload-${workload.type}-${workload.threads}`}>
                          <TableCell>
                            {workload.type}
                            <Typography variant="caption" display="block" color="textSecondary">
                              {workload.threads} threads
                            </Typography>
                          </TableCell>
                          <TableCell>
                            <Chip 
                              label={workloadService.formatWorkloadStatus(workload)}
                              color={workloadService.getStatusColor(workload.status)}
                              size="small"
                            />
                          </TableCell>
                          <TableCell>
                            <Button
                              size="small"
                              color="error"
                              onClick={() => handleStopWorkload(workload.id)}
                              disabled={workloadLoading || workload.status === 'stopping'}
                            >
                              Stop
                            </Button>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Grid>
            )}
          </Paper>
        </Grid>

        {/* Metrics Chart */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2.5 }}>
            <Typography variant="h6" gutterBottom>System Metrics</Typography>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={metrics}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="timestamp"
                  tickFormatter={(timestamp) => new Date(timestamp).toLocaleTimeString()}
                />
                <YAxis />
                <Tooltip 
                  labelFormatter={(timestamp) => new Date(timestamp).toLocaleString()}
                />
                <Legend />
                <Line type="monotone" dataKey="cpu" stroke="#8884d8" name="CPU Usage" />
                <Line type="monotone" dataKey="memory" stroke="#82ca9d" name="Memory Usage" />
                <Line type="monotone" dataKey="network" stroke="#ffc658" name="Network Usage" />
              </LineChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* Anomaly Ranks Chart */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2.5 }}>
            <Typography variant="h6" gutterBottom>Anomaly Ranks</Typography>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={anomalyRanks}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="timestamp"
                  tickFormatter={(timestamp) => new Date(timestamp).toLocaleTimeString()}
                />
                <YAxis domain={[0, 1]} />
                <Tooltip 
                  labelFormatter={(timestamp) => new Date(timestamp).toLocaleString()}
                />
                <Legend />
                <Line type="monotone" dataKey="score" stroke="#ff7300" name="Anomaly Score" />
              </LineChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
      </Grid>

      <Snackbar 
        open={!!error} 
        autoHideDuration={6000} 
        onClose={() => setError('')}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert 
          onClose={() => setError('')} 
          severity="error" 
          variant="filled"
          sx={{ width: '100%' }}
        >
          {error}
        </Alert>
      </Snackbar>
    </Container>
  );
};

export default AnomalyDashboard; 