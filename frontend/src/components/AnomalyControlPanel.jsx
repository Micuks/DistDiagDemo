import React, { useState } from 'react';
import { Card, Button, Space, Table, message, Row, Col, Tag, Spin, Alert } from 'antd';
import { anomalyService } from '../services/anomalyService';
import { useAnomalyData } from '../hooks/useAnomalyData';

const AnomalyControlPanel = () => {
    const { data: activeAnomalies = [], error, isLoading, refetch } = useAnomalyData();
    const [loading, setLoading] = useState(false);

    const anomalyOptions = [
        { id: 'cpu_stress', name: 'CPU Stress' },
        { id: 'memory_stress', name: 'Memory Stress' },
        { id: 'network_delay', name: 'Network Delay' },
        { id: 'disk_stress', name: 'Disk Stress(currently unavailable)' },
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
            title: 'Actions',
            key: 'actions',
            render: (_, record) => (
                <Button 
                    type="primary" 
                    danger 
                    onClick={() => handleAnomalyToggle(record.type)}
                    loading={loading}
                >
                    Stop
                </Button>
            )
        }
    ];

    const isAnomalyActive = (anomalyType) => {
        return Array.isArray(activeAnomalies) && activeAnomalies.some(a => a.type === anomalyType && a.status === 'active');
    };

    const handleAnomalyToggle = async (anomalyType) => {
        let isActive = isAnomalyActive(anomalyType);
        try {
            setLoading(true);
            if (isActive) {
                await anomalyService.stopAnomaly(anomalyType);
                message.success(`Stopped ${anomalyType} anomaly`);
            } else {
                await anomalyService.startAnomaly(anomalyType);
                message.success(`Started ${anomalyType} anomaly`);
            }
            refetch();
        } catch (error) {
            message.error(`Failed to ${isActive ? 'stop' : 'start'} anomaly: ${error.message}`);
            console.error(error);
        } finally {
            setLoading(false);
        }
    };

    const handleStopAllAnomalies = async () => {
        try {
            setLoading(true);
            await anomalyService.stopAllAnomalies();
            message.success('Stopped all anomalies');
            refetch();
        } catch (error) {
            message.error('Failed to stop all anomalies');
            console.error(error);
        } finally {
            setLoading(false);
        }
    };

    if (error) {
        return (
            <Alert
                message="Error"
                description="Failed to load active anomalies"
                type="error"
                showIcon
            />
        );
    }

    return (
        <div>
            <Row gutter={[16, 16]}>
                <Col span={24}>
                    <Card title="Anomaly Control">
                        <Space direction="vertical" style={{ width: '100%' }}>
                            <Space wrap>
                                {anomalyOptions.map(option => (
                                    <Button
                                        key={option.id}
                                        type={isAnomalyActive(option.id) ? 'primary' : 'default'}
                                        onClick={() => handleAnomalyToggle(option.id)}
                                        loading={loading}
                                    >
                                        {option.name}
                                    </Button>
                                ))}
                                <Button
                                    type="primary"
                                    danger
                                    onClick={handleStopAllAnomalies}
                                    loading={loading}
                                >
                                    Stop All
                                </Button>
                            </Space>
                            {isLoading ? (
                                <Spin tip="Loading anomalies..." />
                            ) : (
                                <Table
                                    columns={columns}
                                    dataSource={activeAnomalies || []}
                                    rowKey="type"
                                    pagination={false}
                                />
                            )}
                        </Space>
                    </Card>
                </Col>
            </Row>
        </div>
    );
};

export default AnomalyControlPanel; 