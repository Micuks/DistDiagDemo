import React, { useState, useEffect } from 'react';
import { Card, Button, Space, Table, message, Row, Col, Tag, Spin, Alert } from 'antd';
import { anomalyService } from '../services/anomalyService';
import { useAnomalyData } from '../hooks/useAnomalyData';

const AnomalyControlPanel = () => {
    const { data: activeAnomalies = [], error, isLoading, refetch } = useAnomalyData();
    const [loading, setLoading] = useState(false);
    const [tableLoading, setTableLoading] = useState(true);
    const [lastStableData, setLastStableData] = useState([]);
    
    // Use stable data for table to prevent flickering
    useEffect(() => {
        // Only update the lastStableData when we have data and aren't in a loading state
        // This keeps the table populated with previous data during refreshes
        if (activeAnomalies && activeAnomalies.length > 0) {
            setLastStableData(activeAnomalies);
        }
        
        // Initialize tableLoading as true only on first load
        // After first load complete, set to false
        if (!isLoading && tableLoading) {
            setTableLoading(false);
        }
    }, [activeAnomalies, isLoading]);

    const anomalyOptions = [
        { id: 'cpu_stress', name: 'CPU Stress' },
        { id: 'io_bottleneck', name: 'I/O Bottleneck' },
        { id: 'network_bottleneck', name: 'Network Bottleneck' },
        { id: 'cache_bottleneck', name: 'Cache Bottleneck' },
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
                    {String(status).toUpperCase()}
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

    // Determine which data to show - use lastStableData during refreshes to prevent flickering
    const tableData = (isLoading && lastStableData.length > 0) ? lastStableData : (activeAnomalies || []);

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
                            
                            {/* Show initial loading spinner only during first load when table is empty */}
                            {tableLoading ? (
                                <div style={{ textAlign: 'center', padding: '20px' }}>
                                    <Spin tip="Loading anomalies..." />
                                </div>
                            ) : (
                                <div style={{ position: 'relative' }}>
                                    {/* Overlay a light loading indicator during refreshes */}
                                    {isLoading && !tableLoading && (
                                        <div style={{
                                            position: 'absolute',
                                            top: '8px',
                                            right: '8px',
                                            zIndex: 1,
                                            background: 'rgba(255, 255, 255, 0.7)',
                                            padding: '5px 10px',
                                            borderRadius: '4px',
                                            display: 'flex',
                                            alignItems: 'center',
                                            boxShadow: '0 2px 8px rgba(0, 0, 0, 0.15)'
                                        }}>
                                            <Spin size="small" style={{ marginRight: '8px' }} />
                                            <span>Refreshing...</span>
                                        </div>
                                    )}
                                    
                                    <Table
                                        columns={columns}
                                        dataSource={tableData}
                                        rowKey="name"
                                        pagination={false}
                                        locale={{ 
                                            emptyText: 'No active anomalies' 
                                        }}
                                    />
                                </div>
                            )}
                        </Space>
                    </Card>
                </Col>
            </Row>
        </div>
    );
};

export default AnomalyControlPanel; 