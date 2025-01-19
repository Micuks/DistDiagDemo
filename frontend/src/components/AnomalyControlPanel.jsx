import React, { useState, useEffect } from 'react';
import { Card, Button, Space, Table, message, Row, Col, Switch, Divider, Statistic, Progress } from 'antd';
import { anomalyService } from '../services/anomalyService';

const AnomalyControlPanel = () => {
    const [activeAnomalies, setActiveAnomalies] = useState([]);
    const [loading, setLoading] = useState(false);
    const [collectTrainingData, setCollectTrainingData] = useState(false);
    const [isCollectingNormal, setIsCollectingNormal] = useState(false);
    const [trainingStats, setTrainingStats] = useState(null);

    const anomalyOptions = [
        { id: 'cpu_stress', name: 'CPU Stress' },
        { id: 'memory_stress', name: 'Memory Stress' },
        { id: 'network_delay', name: 'Network Delay' },
        { id: 'disk_stress', name: 'Disk Stress' },
    ];

    const columns = [
        {
            title: 'Type',
            dataIndex: 'type',
            key: 'type',
        },
        {
            title: 'Target',
            dataIndex: 'target',
            key: 'target',
        },
        {
            title: 'Start Time',
            dataIndex: 'start_time',
            key: 'start_time',
        },
        {
            title: 'Status',
            dataIndex: 'status',
            key: 'status',
        },
        {
            title: 'Action',
            key: 'action',
            render: (_, record) => (
                <Button 
                    type="primary" 
                    danger 
                    onClick={() => handleAnomalyToggle(record.type)}
                    loading={loading}
                >
                    Clear
                </Button>
            ),
        },
    ];

    const isAnomalyActive = (anomalyType) => {
        return activeAnomalies.some(anomaly => anomaly.type === anomalyType);
    };

    const handleAnomalyToggle = async (anomalyType) => {
        try {
            setLoading(true);
            if (isAnomalyActive(anomalyType)) {
                await anomalyService.stopAnomaly(anomalyType, collectTrainingData);
                message.success(`Cleared ${anomalyType}`);
            } else {
                await anomalyService.startAnomaly(anomalyType, collectTrainingData);
                message.success(`Injected ${anomalyType}`);
            }
            const anomalies = await anomalyService.getActiveAnomalies();
            setActiveAnomalies(anomalies);
        } catch (err) {
            message.error(err.message || 'Failed to toggle anomaly');
        } finally {
            setLoading(false);
        }
    };

    const handleStopAllAnomalies = async () => {
        try {
            setLoading(true);
            await anomalyService.stopAllAnomalies();
            const anomalies = await anomalyService.getActiveAnomalies();
            setActiveAnomalies(anomalies);
            message.success('All anomalies cleared');
        } catch (err) {
            message.error(err.message || 'Failed to clear all anomalies');
        } finally {
            setLoading(false);
        }
    };

    const handleNormalCollection = async () => {
        setLoading(true);
        try {
            if (!isCollectingNormal) {
                await anomalyService.startNormalCollection();
                message.success('Started collecting normal state data');
            } else {
                await anomalyService.stopNormalCollection();
                message.success('Stopped collecting normal state data');
                await fetchTrainingStats();
            }
            setIsCollectingNormal(!isCollectingNormal);
        } catch (error) {
            message.error('Failed to manage normal state collection');
        }
        setLoading(false);
    };

    useEffect(() => {
        const fetchAnomalies = async () => {
            try {
                const anomalies = await anomalyService.getActiveAnomalies();
                setActiveAnomalies(anomalies);
            } catch (err) {
                message.error(err.message || 'Failed to fetch active anomalies');
            }
        };

        fetchAnomalies();
        const intervalId = setInterval(fetchAnomalies, 5000);

        return () => clearInterval(intervalId);
    }, []);

    useEffect(() => {
        fetchTrainingStats();
    }, []);

    const fetchTrainingStats = async () => {
        try {
            const stats = await anomalyService.getTrainingStats();
            setTrainingStats(stats);
        } catch (error) {
            console.error('Error fetching training stats:', error);
        }
    };

    return (
        <Card title="Anomaly Control Panel">
            <Space direction="vertical" style={{ width: '100%' }}>
                <Row gutter={16}>
                    <Col span={12}>
                        <Switch
                            checked={collectTrainingData}
                            onChange={setCollectTrainingData}
                            checkedChildren="Training Data Collection ON"
                            unCheckedChildren="Training Data Collection OFF"
                        />
                    </Col>
                    <Col span={12}>
                        <Button
                            type={isCollectingNormal ? 'danger' : 'primary'}
                            onClick={handleNormalCollection}
                            loading={loading}
                        >
                            {isCollectingNormal ? 'Stop Normal Collection' : 'Start Normal Collection'}
                        </Button>
                    </Col>
                </Row>

                {trainingStats && (
                    <>
                        <Divider>Training Dataset Statistics</Divider>
                        <Row gutter={16}>
                            <Col span={8}>
                                <Statistic title="Total Samples" value={trainingStats.total_samples} />
                            </Col>
                            <Col span={8}>
                                <Statistic title="Normal Samples" value={trainingStats.stats.normal} />
                            </Col>
                            <Col span={8}>
                                <Statistic title="Anomaly Samples" value={trainingStats.stats.anomaly} />
                            </Col>
                        </Row>
                        <Row style={{ marginTop: '16px' }}>
                            <Col span={24}>
                                <div>Dataset Balance</div>
                                <Progress
                                    percent={Math.round(trainingStats.normal_ratio * 100)}
                                    success={{ percent: Math.round(trainingStats.anomaly_ratio * 100) }}
                                    format={() => `${Math.round(trainingStats.normal_ratio * 100)}% Normal / ${Math.round(trainingStats.anomaly_ratio * 100)}% Anomaly`}
                                    status={trainingStats.is_balanced ? 'success' : 'exception'}
                                />
                            </Col>
                        </Row>

                        <Divider>Model Training</Divider>
                        <Row gutter={[16, 16]}>
                            <Col span={24}>
                                <Space>
                                    <Button
                                        type="primary"
                                        onClick={async () => {
                                            try {
                                                setLoading(true);
                                                await anomalyService.trainModel();
                                                message.success('Model trained successfully');
                                            } catch (error) {
                                                message.error('Failed to train model: ' + error.message);
                                            } finally {
                                                setLoading(false);
                                            }
                                        }}
                                        loading={loading}
                                        disabled={!trainingStats?.is_balanced}
                                    >
                                        Train Model
                                    </Button>
                                    {!trainingStats?.is_balanced && (
                                        <span style={{ color: '#ff4d4f' }}>
                                            Dataset must be balanced before training
                                        </span>
                                    )}
                                </Space>
                            </Col>
                        </Row>
                    </>
                )}

                <Row gutter={[16, 16]}>
                    {anomalyOptions.map((option) => (
                        <Col span={6} key={option.id}>
                            <Button
                                type={isAnomalyActive(option.id) ? "primary" : "default"}
                                danger={isAnomalyActive(option.id)}
                                onClick={() => handleAnomalyToggle(option.id)}
                                loading={loading}
                                style={{ width: '100%' }}
                            >
                                {isAnomalyActive(option.id) ? `Clear ${option.name}` : `Inject ${option.name}`}
                            </Button>
                        </Col>
                    ))}
                </Row>

                {activeAnomalies.length > 0 && (
                    <>
                        <Button 
                            type="primary" 
                            danger 
                            onClick={handleStopAllAnomalies}
                            loading={loading}
                            style={{ marginTop: 16 }}
                        >
                            Clear All Anomalies
                        </Button>
                        <Table 
                            columns={columns} 
                            dataSource={activeAnomalies} 
                            rowKey="type"
                            size="small"
                            style={{ marginTop: 16 }}
                        />
                    </>
                )}
            </Space>
        </Card>
    );
};

export default AnomalyControlPanel; 