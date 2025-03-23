import React, { useState, useEffect } from 'react';
import { Card, Spin, Alert, Row, Col, Statistic, Progress, Divider, Typography, Tooltip } from 'antd';
import { CheckCircleFilled, ClockCircleFilled, LineChartOutlined, PercentageOutlined, AimOutlined, TrophyOutlined } from '@ant-design/icons';
import { trainingService } from '../services/trainingService';

const { Title, Text } = Typography;

const ModelPerformanceView = ({ modelName }) => {
    const [loading, setLoading] = useState(true);
    const [performance, setPerformance] = useState(null);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchPerformance = async () => {
            try {
                setLoading(true);
                setError(null);
                
                const data = await trainingService.getModelPerformance(modelName);
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
        } else {
            setLoading(false);
            setError('No model selected');
        }
    }, [modelName]);

    if (loading) return <Spin tip="Loading performance metrics..." />;
    if (error) return <Alert message={error} type="error" showIcon />;
    if (!performance) return <Alert message="No performance data available" type="info" showIcon />;

    // Helper function to determine color based on metric value
    const getColorForMetric = (key, value) => {
        if (key === 'training_time') return 'blue'; // Training time doesn't have thresholds
        
        if (value > 0.9) return '#3f8600'; // Excellent - Green
        if (value > 0.7) return '#108ee9'; // Good - Blue
        if (value > 0.5) return '#faad14'; // Fair - Yellow/Orange
        return '#cf1322'; // Poor - Red
    };

    // Helper function to get the appropriate icon for each metric
    const getIconForMetric = (key) => {
        switch (key) {
            case 'accuracy': return <CheckCircleFilled />;
            case 'precision': return <AimOutlined />;
            case 'recall': return <PercentageOutlined />;
            case 'f1_score': return <TrophyOutlined />;
            case 'training_time': return <ClockCircleFilled />;
            default: return <LineChartOutlined />;
        }
    };

    // Helper function to get description for each metric
    const getDescriptionForMetric = (key) => {
        switch (key) {
            case 'accuracy': return 'Overall correct predictions';
            case 'precision': return 'True positives / (True positives + False positives)';
            case 'recall': return 'True positives / (True positives + False negatives)';
            case 'f1_score': return 'Harmonic mean of precision and recall';
            case 'training_time': return 'Time taken to train the model';
            default: return '';
        }
    };

    const metrics = [
        { key: 'accuracy', label: 'Accuracy' },
        { key: 'precision', label: 'Precision' },
        { key: 'recall', label: 'Recall' },
        { key: 'f1_score', label: 'F1 Score' }
    ];

    return (
        <Card 
            className="model-performance-card"
            bordered={false}
            style={{ boxShadow: '0 1px 2px rgba(0,0,0,0.1)' }}
        >
            <Title level={4} style={{ marginBottom: 16 }}>
                Model Performance Metrics
            </Title>
            
            <Row gutter={[16, 16]}>
                {metrics.map(({ key, label }) => {
                    const value = performance[key];
                    const color = getColorForMetric(key, value);
                    
                    return (
                        <Col xs={24} sm={12} md={6} key={key}>
                            <Tooltip title={getDescriptionForMetric(key)}>
                                <Card bordered={false} size="small" style={{ borderLeft: `4px solid ${color}` }}>
                                    <Statistic
                                        title={
                                            <Text strong style={{ fontSize: 16 }}>
                                                {getIconForMetric(key)} {label}
                                            </Text>
                                        }
                                        value={value}
                                        precision={4}
                                        valueStyle={{ color, fontSize: 24 }}
                                        suffix={key !== 'training_time' && <Progress 
                                            type="circle" 
                                            percent={value * 100} 
                                            width={30} 
                                            format={() => ''} 
                                            strokeColor={color}
                                            style={{ marginLeft: 8 }}
                                        />}
                                    />
                                </Card>
                            </Tooltip>
                        </Col>
                    );
                })}
            </Row>
            
            <Divider style={{ margin: '16px 0' }} />
            
            <Row>
                <Col span={24}>
                    <Card bordered={false} size="small" style={{ background: '#f5f5f5' }}>
                        <Statistic
                            title={
                                <Text strong style={{ fontSize: 16 }}>
                                    <ClockCircleFilled /> Training Time
                                </Text>
                            }
                            value={performance.training_time}
                            precision={2}
                            suffix="seconds"
                            valueStyle={{ color: '#1890ff', fontSize: 22 }}
                        />
                    </Card>
                </Col>
            </Row>
        </Card>
    );
};

export default ModelPerformanceView; 