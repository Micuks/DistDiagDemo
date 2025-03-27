import React, { useState, useEffect } from 'react';
import { Card, Button, Space, Table, message, Row, Col, Tag, Spin, Alert } from 'antd';
import { anomalyService } from '../services/anomalyService';
import { useAnomalyData } from '../hooks/useAnomalyData';

const AnomalyControlPanel = () => {
    const { data: activeAnomalies = [], error, isLoading, refetch } = useAnomalyData();
    const [loading, setLoading] = useState(false);
    const [tableLoading, setTableLoading] = useState(true);
    const [lastStableData, setLastStableData] = useState([]);
    const [groupedAnomalies, setGroupedAnomalies] = useState([]);
    
    // Group anomalies by type
    useEffect(() => {
        // Group anomalies of the same type together
        const groupAnomalies = (anomalies) => {
            if (!anomalies || anomalies.length === 0) return [];
            
            const groups = {};
            
            anomalies.forEach(anomaly => {
                const key = anomaly.type;
                
                if (!groups[key]) {
                    // Create a new group
                    groups[key] = {
                        ...anomaly,
                        nodes: [anomaly.node],
                        names: [anomaly.name],
                        original_entries: [anomaly]
                    };
                } else {
                    // Add to existing group
                    if (!groups[key].nodes.includes(anomaly.node)) {
                        groups[key].nodes.push(anomaly.node);
                    }
                    if (!groups[key].names.includes(anomaly.name)) {
                        groups[key].names.push(anomaly.name);
                    }
                    groups[key].original_entries.push(anomaly);
                }
            });
            
            return Object.values(groups);
        };
        
        const grouped = groupAnomalies(activeAnomalies);
        setGroupedAnomalies(grouped);
    }, [activeAnomalies]);
    
    // Use stable data for table to prevent flickering
    useEffect(() => {
        // Only update the lastStableData when we have data and aren't in a loading state
        // This keeps the table populated with previous data during refreshes
        if (groupedAnomalies && groupedAnomalies.length > 0) {
            setLastStableData(groupedAnomalies);
        } else if (activeAnomalies && activeAnomalies.length > 0) {
            // Fallback to ungrouped if grouping fails
            setLastStableData(activeAnomalies);
        }
        
        // Initialize tableLoading as true only on first load
        // After first load complete, set to false
        if (!isLoading && tableLoading) {
            setTableLoading(false);
        }
    }, [groupedAnomalies, activeAnomalies, isLoading, tableLoading]);

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
            title: 'Target Node(s)',
            dataIndex: 'nodes',
            key: 'nodes',
            render: (nodes) => {
                if (Array.isArray(nodes)) {
                    if (nodes.length === 1) {
                        return nodes[0] || 'Default';
                    }
                    return (
                        <span>
                            {nodes[0]} (+{nodes.length - 1} more)
                            <div style={{ fontSize: '11px', marginTop: '2px' }}>
                                {nodes.slice(1, 4).map(node => node).join(', ')}
                                {nodes.length > 4 ? '...' : ''}
                            </div>
                        </span>
                    );
                }
                return nodes || 'Default';
            }
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
                    onClick={() => handleAnomalyToggle(record.type, record.names)}
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

    const handleAnomalyToggle = async (anomalyId, anomalyNames) => {
        try {
            setLoading(true);
            if (isAnomalyActive(anomalyId)) {
                // If it's a grouped entry with multiple names, stop each one
                if (Array.isArray(anomalyNames) && anomalyNames.length > 1) {
                    message.loading(`Stopping all ${anomalyId} anomalies...`, 1);
                    
                    for (const name of anomalyNames) {
                        try {
                            await anomalyService.stopAnomaly(anomalyId, name);
                        } catch (innerError) {
                            console.error(`Failed to stop anomaly ${name}:`, innerError);
                            // Continue with others even if one fails
                        }
                    }
                    
                    message.success(`All ${anomalyId} anomalies stopped`);
                } else {
                    // Just stop by type if no names array
                    await anomalyService.stopAnomaly(anomalyId);
                    message.success(`Anomaly ${anomalyId} stopped`);
                }
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
    const tableData = (isLoading && lastStableData.length > 0) ? lastStableData : (groupedAnomalies || []);

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
                                        rowKey="type"
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