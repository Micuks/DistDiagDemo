import React, { useState, useEffect } from 'react';
import { Card, Button, Space, Row, Col, Switch, Divider, Statistic, Progress, Alert, Spin, message } from 'antd';
import { anomalyService } from '../services/anomalyService';

const ModelTrainingPanel = () => {
    const [loading, setLoading] = useState(false);
    const [collectTrainingData, setCollectTrainingData] = useState(false);
    const [isCollectingNormal, setIsCollectingNormal] = useState(false);
    const [trainingStats, setTrainingStats] = useState(null);
    const [isAutoBalancing, setIsAutoBalancing] = useState(false);

    const handleNormalCollection = async () => {
        try {
            setLoading(true);
            await anomalyService.toggleNormalCollection(!isCollectingNormal);
            setIsCollectingNormal(!isCollectingNormal);
            message.success(`${!isCollectingNormal ? 'Started' : 'Stopped'} collecting normal data`);
        } catch (error) {
            message.error('Failed to toggle normal data collection');
            console.error(error);
        } finally {
            setLoading(false);
        }
    };

    const handleAutoBalance = async () => {
        try {
            setLoading(true);
            await anomalyService.toggleAutoBalance(!isAutoBalancing);
            setIsAutoBalancing(!isAutoBalancing);
            message.success(`${!isAutoBalancing ? 'Enabled' : 'Disabled'} auto-balancing`);
        } catch (error) {
            message.error('Failed to toggle auto-balancing');
            console.error(error);
        } finally {
            setLoading(false);
        }
    };

    const fetchTrainingStats = async () => {
        try {
            const response = await anomalyService.getTrainingStats();
            setTrainingStats(response.stats);  // Store just the stats object
        } catch (error) {
            console.error('Failed to fetch training stats:', error);
        }
    };

    useEffect(() => {
        fetchTrainingStats();
        const interval = setInterval(fetchTrainingStats, 5000);
        return () => clearInterval(interval);
    }, []);

    return (
        <div>
            <Row gutter={[16, 16]}>
                <Col span={24}>
                    <Card title="Training Data Collection">
                        <Space direction="vertical" style={{ width: '100%' }}>
                            <div>
                                <Switch
                                    checked={isCollectingNormal}
                                    onChange={handleNormalCollection}
                                    loading={loading}
                                />
                                <span style={{ marginLeft: 8 }}>Collect Normal Data</span>
                            </div>
                            <div>
                                <Switch
                                    checked={isAutoBalancing}
                                    onChange={handleAutoBalance}
                                    loading={loading}
                                />
                                <span style={{ marginLeft: 8 }}>Auto-balance Training Data</span>
                            </div>
                        </Space>
                    </Card>
                </Col>
            </Row>

            <Divider />

            <Row gutter={[16, 16]}>
                <Col span={24}>
                    <Card title="Training Statistics">
                        {trainingStats ? (
                            <>
                                <Row gutter={[16, 16]}>
                                    <Col span={8}>
                                        <Statistic
                                            title="Normal Samples"
                                            value={trainingStats.normal}
                                        />
                                    </Col>
                                    <Col span={8}>
                                        <Statistic
                                            title="Anomaly Samples"
                                            value={trainingStats.anomaly}
                                        />
                                    </Col>
                                    <Col span={8}>
                                        <Statistic
                                            title="Total Samples"
                                            value={trainingStats.total_samples}
                                        />
                                    </Col>
                                </Row>
                                <Divider />
                                <Progress
                                    percent={Math.round(trainingStats.normal_ratio * 100)}
                                    success={{ percent: Math.round(trainingStats.anomaly_ratio * 100) }}
                                    format={() => 'Data Distribution'}
                                />
                                {trainingStats.is_balanced ? (
                                    <Alert
                                        message="Data is well-balanced"
                                        type="success"
                                        showIcon
                                        style={{ marginTop: 16 }}
                                    />
                                ) : (
                                    <Alert
                                        message="Data is imbalanced"
                                        description="Consider enabling auto-balance or collecting more data"
                                        type="warning"
                                        showIcon
                                        style={{ marginTop: 16 }}
                                    />
                                )}
                            </>
                        ) : (
                            <Spin tip="Loading training statistics..." />
                        )}
                    </Card>
                </Col>
            </Row>
        </div>
    );
};

export default ModelTrainingPanel; 