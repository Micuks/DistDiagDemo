import React, { useState, useEffect } from 'react';
import { Card, Button, Space, Row, Col, Switch, Divider, Statistic, Progress, Alert, Spin, message } from 'antd';
import { anomalyService } from '../services/anomalyService';
import { useAnomalyData } from '../hooks/useAnomalyData';

const ModelTrainingPanel = () => {
    const [loading, setLoading] = useState(false);
    const [collectTrainingData, setCollectTrainingData] = useState(false);
    const [isCollectingNormal, setIsCollectingNormal] = useState(false);
    const [trainingStats, setTrainingStats] = useState(null);
    const [isAutoBalancing, setIsAutoBalancing] = useState(false);
    const { data: activeAnomalies = [] } = useAnomalyData();
    const [collectionStatus, setCollectionStatus] = useState({
        isCollecting: false,
        currentType: null
    });

    // Fetch collection status periodically
    useEffect(() => {
        const fetchStatus = async () => {
            try {
                const status = await anomalyService.getCollectionStatus();
                setCollectionStatus({
                    isCollecting: status.is_collecting_normal || status.is_collecting_anomaly,
                    currentType: status.current_type
                });
            } catch (error) {
                console.error('Failed to fetch collection status:', error);
            }
        };
        
        fetchStatus();
        const interval = setInterval(fetchStatus, 5000);
        return () => clearInterval(interval);
    }, []);

    // Fetch training stats periodically
    useEffect(() => {
        fetchTrainingStats();
        const interval = setInterval(fetchTrainingStats, 5000);
        return () => clearInterval(interval);
    }, []);

    const handleCollectionToggle = async () => {
        try {
            setLoading(true);
            if (collectionStatus.isCollecting) {
                // Stop collection based on current type
                if (collectionStatus.currentType === 'normal') {
                    await anomalyService.stopNormalCollection();
                } else {
                    // For anomaly, stop collection on the active anomaly
                    const activeAnomaly = activeAnomalies[0];
                    if (activeAnomaly) {
                        await anomalyService.stopAnomaly(activeAnomaly.type);
                    }
                }
            } else {
                // Start collection based on presence of anomalies
                if (activeAnomalies.length > 0) {
                    const activeAnomaly = activeAnomalies[0];
                    await anomalyService.startAnomaly(activeAnomaly.type, {
                        node: activeAnomaly.node,
                        collect_training_data: true
                    });
                } else {
                    await anomalyService.startNormalCollection();
                }
            }
            
            // Refresh status after toggle
            const newStatus = await anomalyService.getCollectionStatus();
            setCollectionStatus({
                isCollecting: newStatus.is_collecting_normal || newStatus.is_collecting_anomaly,
                currentType: newStatus.current_type
            });
            message.success(`${collectionStatus.isCollecting ? 'Stopped' : 'Started'} data collection`);
        } catch (error) {
            message.error('Failed to toggle data collection');
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

    return (
        <div>
            <Row gutter={[16, 16]}>
                <Col span={24}>
                    <Card title="Training Data Collection">
                        <Space direction="vertical" style={{ width: '100%' }}>
                            <div>
                                <Switch
                                    checked={collectionStatus.isCollecting}
                                    onChange={handleCollectionToggle}
                                    loading={loading}
                                    disabled={loading}
                                />
                                <span style={{ marginLeft: 8 }}>
                                    {collectionStatus.isCollecting 
                                        ? `Collecting ${collectionStatus.currentType} data` 
                                        : 'Start data collection'}
                                </span>
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