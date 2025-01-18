import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import MetricsPanel from './components/MetricsPanel';
import RanksPanel from './components/RanksPanel';
import ControlPanel from './components/ControlPanel';

const AppRoutes = () => {
    return (
        <Routes>
            <Route path="/control" element={<ControlPanel />} />
            <Route path="/metrics" element={<MetricsPanel />} />
            <Route path="/ranks" element={<RanksPanel />} />
            <Route path="/" element={<Navigate to="/control" replace />} />
        </Routes>
    );
};

export default AppRoutes; 