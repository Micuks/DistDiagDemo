import React, { useState, useEffect } from 'react';
import { Card, Button, Space, Table, message, Row, Col, Switch, Divider, Statistic, Progress, Alert, Tag } from 'antd';
import { anomalyService } from '../services/anomalyService';

const AnomalyControlPanel = () => {
    const [activeAnomalies, setActiveAnomalies] = useState([]);
    const [loading, setLoading] = useState(false);
    const [collectTrainingData, setCollectTrainingData] = useState(false);
    const [isCollectingNormal, setIsCollectingNormal] = useState(false);
    const [trainingStats, setTrainingStats] = useState(null);
    const [isAutoBalancing, setIsAutoBalancing] = useState(false);

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
            render: (type) => {
                const option = anomalyOptions.find(opt => opt.id === type);
                return option ? option.name : type;
            }
        },
        {
            title: 'Target Node',
            dataIndex: 'node',
            key: 'node',
            render: (node) => node || 'Default'
        },
        {
            title: 'Start Time',
            dataIndex: 'start_time',
            key: 'start_time',
            render: (time) => new Date(time).toLocaleString()
        },
        {
            title: 'Status',
            dataIndex: 'status',
            key: 'status',
            render: (status) => (
                <Tag color={status === 'active' ? 'green' : 'red'}>
                    {status.toUpperCase()}
                </Tag>
            )
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
                const response = await anomalyService.startAnomaly(anomalyType, collectTrainingData);
                message.success(`${response.message}`);  // Use server response message
            }
            const anomalies = await anomalyService.getActiveAnomalies();
            setActiveAnomalies(anomalies);
            // Refresh training stats after anomaly state change
            if (collectTrainingData) {
                await fetchTrainingStats();
            }
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

    const handleAutoBalance = async () => {
        try {
            setIsAutoBalancing(true);
            await anomalyService.autoBalanceDataset();
            message.success('Auto-balance process started');
            
            // Start polling for updates
            const pollInterval = setInterval(async () => {
                const stats = await anomalyService.getTrainingStats();
                setTrainingStats(stats);
                if (stats.is_balanced) {
                    clearInterval(pollInterval);
                    setIsAutoBalancing(false);
                    message.success('Dataset is now balanced');
                }
            }, 5000);
            
            // Stop polling after 30 minutes
            setTimeout(() => {
                clearInterval(pollInterval);
                setIsAutoBalancing(false);
            }, 30 * 60 * 1000);
            
        } catch (error) {
            message.error('Failed to start auto-balance: ' + error.message);
            setIsAutoBalancing(false);
        }
    };

    useEffect(() => {
        const fetchAnomalies = async () => {
            try {
                const startTime = Date.now();
                const anomalies = await anomalyService.getActiveAnomalies();
                console.log("fetching active anomalies costs time: ", Date.now() - startTime);
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
            const startTime = Date.now();
            const stats = await anomalyService.getTrainingStats();
            console.log("fetching training stats costs time: ", Date.now() - startTime);
            setTrainingStats(stats);
        } catch (error) {
            console.error('Error fetching training stats:', error);
        }
    };

    return (
        <Card title="Anomaly Control Panel">
            <Space direction="vertical" style={{ width: '100%' }}>
                <Row gutter={16}>
                    <Col span={8}>
                        <Switch
                            checked={collectTrainingData}
                            onChange={(checked) => {
                                setCollectTrainingData(checked);
                                if (checked) {
                                    message.info('Training data will be collected for anomaly injections');
                                }
                            }}
                            checkedChildren="Training Data Collection ON"
                            unCheckedChildren="Training Data Collection OFF"
                        />
                    </Col>
                    <Col span={8}>
                        <Button
                            type={isCollectingNormal ? 'danger' : 'primary'}
                            onClick={handleNormalCollection}
                            loading={loading}
                        >
                            {isCollectingNormal ? 'Stop Normal Collection' : 'Start Normal Collection'}
                        </Button>
                    </Col>
                    <Col span={8}>
                        <Button
                            type="primary"
                            onClick={handleAutoBalance}
                            loading={isAutoBalancing}
                            disabled={isCollectingNormal}
                        >
                            {isAutoBalancing ? 'Auto-Balancing...' : 'Auto-Balance Dataset'}
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

                        <Row style={{ marginTop: '16px' }}>
                            <Col span={24}>
                                <div>Anomaly Type Distribution</div>
                                {Object.entries(trainingStats.stats.anomaly_types).map(([type, count]) => (
                                    <div key={type} style={{ marginBottom: '8px' }}>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                                            <span>{type.toUpperCase()}</span>
                                            <span>{count} samples</span>
                                        </div>
                                        <Progress
                                            percent={Math.round((count / trainingStats.stats.anomaly) * 100)}
                                            size="small"
                                            status={count > 0 ? 'active' : 'exception'}
                                        />
                                    </div>
                                ))}
                            </Col>
                        </Row>

                        <Divider>Model Training</Divider>
                        <Row gutter={[16, 16]}>
                            <Col span={24}>
                                <Space direction="vertical" style={{ width: '100%' }}>
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
                                        <Alert
                                            message="Dataset Not Balanced"
                                            description={
                                                <div>
                                                    <p>The dataset must be balanced before training. Current status:</p>
                                                    <ul>
                                                        <li>Normal data: {Math.round(trainingStats.normal_ratio * 100)}%</li>
                                                        <li>Anomaly data: {Math.round(trainingStats.anomaly_ratio * 100)}%</li>
                                                        <li>Required: Difference should be within 30%</li>
                                                    </ul>
                                                </div>
                                            }
                                            type="warning"
                                            showIcon
                                        />
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