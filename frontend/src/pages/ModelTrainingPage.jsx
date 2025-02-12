import React, { useState, useEffect } from 'react';
import { Card, Spin } from 'antd';
import ModelTrainingPanel from '../components/ModelTrainingPanel';
import { useAnomalyData } from '../hooks/useAnomalyData';

const ModelTrainingPage = () => {
    const { refetch: refetchAnomalies } = useAnomalyData();
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        // Prefetch required data when entering the page
        Promise.all([refetchAnomalies()])
            .finally(() => setLoading(false));
    }, []);

    return (
        <Card title="Model Training Management">
            {loading ? <Spin tip="Loading training environment..." /> : <ModelTrainingPanel />}
        </Card>
    );
};

export default ModelTrainingPage; 