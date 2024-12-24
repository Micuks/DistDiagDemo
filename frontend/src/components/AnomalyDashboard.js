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
} from '@material-ui/core';
import { Alert } from '@material-ui/lab';
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

const AnomalyDashboard = () => {
  const [selectedAnomaly, setSelectedAnomaly] = useState('');
  const [isAnomalyActive, setIsAnomalyActive] = useState(false);
  const [metrics, setMetrics] = useState([]);
  const [anomalyRanks, setAnomalyRanks] = useState([]);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const anomalyOptions = [
    { id: 'cpu_stress', name: 'CPU Stress' },
    { id: 'memory_stress', name: 'Memory Stress' },
    { id: 'network_delay', name: 'Network Delay' },
    { id: 'disk_stress', name: 'Disk Stress' },
  ];

  const handleAnomalyChange = (event) => {
    setSelectedAnomaly(event.target.value);
  };

  const handleAnomalyToggle = async () => {
    try {
      setLoading(true);
      if (isAnomalyActive) {
        await anomalyService.stopAnomaly();
      } else {
        await anomalyService.startAnomaly(selectedAnomaly);
      }
      setIsAnomalyActive(!isAnomalyActive);
    } catch (err) {
      setError(err.message || 'Failed to toggle anomaly');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    let intervalId;

    const fetchData = async () => {
      try {
        const [metricsData, ranksData] = await Promise.all([
          anomalyService.getMetrics(),
          anomalyService.getAnomalyRanks()
        ]);
        
        setMetrics(metricsData);
        setAnomalyRanks(ranksData);
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
        <Grid item xs={12}>
          <Paper style={{ padding: 20 }}>
            <Grid container spacing={2} alignItems="center">
              <Grid item xs={8}>
                <FormControl fullWidth>
                  <InputLabel>Select Anomaly</InputLabel>
                  <Select
                    value={selectedAnomaly}
                    onChange={handleAnomalyChange}
                    disabled={loading}
                  >
                    {anomalyOptions.map((option) => (
                      <MenuItem key={option.id} value={option.id}>
                        {option.name}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={4}>
                <Button
                  variant="contained"
                  color={isAnomalyActive ? "secondary" : "primary"}
                  onClick={handleAnomalyToggle}
                  fullWidth
                  disabled={!selectedAnomaly || loading}
                >
                  {loading ? 'Processing...' : isAnomalyActive ? 'Stop Anomaly' : 'Start Anomaly'}
                </Button>
              </Grid>
            </Grid>
          </Paper>
        </Grid>

        <Grid item xs={12}>
          <Paper style={{ padding: 20 }}>
            <Typography variant="h6">System Metrics</Typography>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={metrics}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="timestamp" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="cpu" stroke="#8884d8" name="CPU Usage" />
                <Line type="monotone" dataKey="memory" stroke="#82ca9d" name="Memory Usage" />
                <Line type="monotone" dataKey="network" stroke="#ffc658" name="Network Usage" />
              </LineChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        <Grid item xs={12}>
          <Paper style={{ padding: 20 }}>
            <Typography variant="h6">Anomaly Ranks</Typography>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={anomalyRanks}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="timestamp" />
                <YAxis domain={[0, 1]} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="rank" stroke="#ff7300" name="Anomaly Rank" />
              </LineChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
      </Grid>

      <Snackbar 
        open={!!error} 
        autoHideDuration={6000} 
        onClose={() => setError('')}
      >
        <Alert onClose={() => setError('')} severity="error">
          {error}
        </Alert>
      </Snackbar>
    </Container>
  );
};

export default AnomalyDashboard; 