import React from 'react';
import { ThemeProvider, createTheme, CssBaseline } from '@material-ui/core';
import AnomalyDashboard from './components/AnomalyDashboard';

const theme = createTheme({
  palette: {
    type: 'dark',
    primary: {
      main: '#90caf9',
    },
    secondary: {
      main: '#f48fb1',
    },
    background: {
      default: '#303030',
      paper: '#424242',
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <div style={{ padding: '24px 0' }}>
        <AnomalyDashboard />
      </div>
    </ThemeProvider>
  );
}

export default App; 