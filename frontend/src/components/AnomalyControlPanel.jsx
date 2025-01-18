import React, { useState, useEffect } from 'react';
import { Card, Button, Space, Table, message, Row, Col } from 'antd';
import { anomalyService } from '../services/anomalyService';

const AnomalyControlPanel = () => {
    const [activeAnomalies, setActiveAnomalies] = useState([]);
    const [loading, setLoading] = useState(false);

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
            dataIndex: 'startTime',
            key: 'startTime',
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
                    Stop
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
                await anomalyService.stopAnomaly(anomalyType);
                message.success(`Stopped ${anomalyType}`);
            } else {
                await anomalyService.startAnomaly(anomalyType);
                message.success(`Started ${anomalyType}`);
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
            message.success('All anomalies stopped');
        } catch (err) {
            message.error(err.message || 'Failed to stop all anomalies');
        } finally {
            setLoading(false);
        }
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

    return (
        <Card title="Anomaly Control" style={{ marginBottom: 16 }}>
            <Space direction="vertical" style={{ width: '100%' }}>
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
                                {isAnomalyActive(option.id) ? `Stop ${option.name}` : `Start ${option.name}`}
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
                            Stop All Anomalies
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