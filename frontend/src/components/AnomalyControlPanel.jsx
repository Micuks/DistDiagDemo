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
        { id: 'disk_stress', name: 'Disk Stress' },
        { id: 'too_many_indexes', name: 'Too Many Indexes' }
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

    const isAnomalyActive = (anomalyId) => {
        return Array.isArray(activeAnomalies) && activeAnomalies.some(a => 
            a && a.type === anomalyId && a.status === 'active'
        );
    };

    const handleAnomalyToggle = async (anomalyId) => {
        try {
            setLoading(true);
            if (isAnomalyActive(anomalyId)) {
                await anomalyService.stopAnomaly(anomalyId);
                message.success(`Anomaly ${anomalyId} stopped`);
            } else {
                await anomalyService.startAnomaly(anomalyId);
                message.success(`Anomaly ${anomalyId} started`);
            }
            await refetch();
        } catch (err) {
            message.error(`Failed to toggle anomaly: ${err.message}`);
        } finally {
            setLoading(false);
        }
    };

    const handleStopAllAnomalies = async () => {
        try {
            setLoading(true);
            message.loading({ content: 'Stopping all anomalies...', key: 'stopAll' });
            
            const result = await anomalyService.stopAllAnomalies();
            
            // Force an immediate refetch regardless of the result
            await refetch();
            
            // Check if any anomalies remain
            const currentAnomalies = await anomalyService.getActiveAnomalies();
            
            if (currentAnomalies && currentAnomalies.length > 0) {
                // Some anomalies still exist
                message.warning({ 
                    content: `Stopped some anomalies but ${currentAnomalies.length} remain active. Try again or stop individually.`, 
                    key: 'stopAll', 
                    duration: 5 
                });
                console.warn('Anomalies still active after stopAll:', currentAnomalies);
            } else {
                // All anomalies successfully stopped
                message.success({ 
                    content: 'Successfully stopped all anomalies', 
                    key: 'stopAll' 
                });
            }
            
            // Set a timer to do one more refetch after a short delay
            // This handles cases where deletions are still being processed in the backend
            setTimeout(() => {
                refetch();
            }, 2000);
        } catch (error) {
            console.error('Failed to stop all anomalies:', error);
            message.error({ 
                content: `Failed to stop all anomalies: ${error.message}`, 
                key: 'stopAll', 
                duration: 5 
            });
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
                                    rowKey="name"
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