import React, { useState, useEffect } from 'react';
import { Card, Table, Spin, Alert } from 'antd';
import { anomalyService } from '../services/anomalyService';

const ModelPerformanceView = ({ modelName }) => {
    const [loading, setLoading] = useState(true);
    const [performance, setPerformance] = useState(null);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchPerformance = async () => {
            try {
                setLoading(true);
                setError(null);
                const data = await anomalyService.getModelPerformance(modelName);
                setPerformance(data);
            } catch (err) {
                setError('Failed to load model performance metrics');
                console.error(err);
            } finally {
                setLoading(false);
            }
        };

        if (modelName) {
            fetchPerformance();
        }
    }, [modelName]);

    if (loading) {
        return <Spin tip="Loading performance metrics..." />;
    }

    if (error) {
        return <Alert message={error} type="error" showIcon />;
    }

    if (!performance) {
        return <Alert message="No performance data available" type="info" showIcon />;
    }

    const columns = [
        {
            title: 'Metric',
            dataIndex: 'metric',
            key: 'metric',
        },
        {
            title: 'Value',
            dataIndex: 'value',
            key: 'value',
            render: (value) => typeof value === 'number' ? value.toFixed(4) : value,
        },
    ];

    const metrics = Object.entries(performance.metrics || {}).map(([key, value]) => ({
        key,
        metric: key.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' '),
        value,
    }));

    return (
        <div>
            <Table 
                dataSource={metrics} 
                columns={columns} 
                pagination={false}
                size="small"
            />
            {performance.training_time && (
                <Alert
                    message={`Training Time: ${performance.training_time.toFixed(2)}s`}
                    type="info"
                    showIcon
                    style={{ marginTop: 16 }}
                />
            )}
        </div>
    );
};

export default ModelPerformanceView; 