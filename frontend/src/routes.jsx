import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import MetricsPanel from './components/MetricsPanel';
import RanksPanel from './components/RanksPanel';
import ControlPanel from './components/ControlPanel';
import ModelTrainingPage from './pages/ModelTrainingPage';
import Dashboard from './pages/Dashboard';

const AppRoutes = () => {
    return (
        <Routes>
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/metrics" element={<MetricsPanel />} />
            <Route path="/ranks" element={<RanksPanel />} />
            <Route path="/control" element={<ControlPanel />} />
            <Route path="/training" element={<ModelTrainingPage />} />
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
        </Routes>
    );
};

export default AppRoutes; 