import React, { useState, useEffect } from 'react';
import { Card, Table, Spin, Alert, Row, Col, Statistic } from 'antd';
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

    if (loading) return <Spin tip="Loading performance metrics..." />;
    if (error) return <Alert message={error} type="error" showIcon />;
    if (!performance) return <Alert message="No performance data available" type="info" showIcon />;

    const metrics = [
        { key: 'accuracy', label: 'Accuracy' },
        { key: 'precision', label: 'Precision' },
        { key: 'recall', label: 'Recall' },
        { key: 'f1_score', label: 'F1 Score' }
    ];

    return (
        <Card title={`Model Performance: ${modelName}`} size="small">
            <Row gutter={[16, 16]}>
                {metrics.map(({ key, label }) => (
                    <Col span={6} key={key}>
                        <Statistic
                            title={label}
                            value={performance[key]}
                            precision={4}
                        />
                    </Col>
                ))}
            </Row>
            {performance.training_time && (
                <Alert
                    message={`Training Time: ${performance.training_time.toFixed(2)}s`}
                    type="info"
                    showIcon
                    style={{ marginTop: 16 }}
                />
            )}
        </Card>
    );
};

export default ModelPerformanceView; 