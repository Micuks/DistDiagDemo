import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { Card, Button, Space, Table, message, Row, Col, Tag, Spin, Alert, Modal, Tooltip, Select, Switch, Typography } from 'antd';
import { anomalyService } from '../services/anomalyService';
import { WarningOutlined, InfoCircleOutlined, StopOutlined, ReloadOutlined, CheckCircleOutlined, LaptopOutlined, UnorderedListOutlined, AppstoreOutlined } from '@ant-design/icons';
import { useAnomalyStreamData } from '../hooks/useAnomalyStreamData';
import { useAnomalyData } from '../hooks/useAnomalyData';
import formatDistanceToNow from 'date-fns/formatDistanceToNow';

const AnomalyControlPanel = () => {
    const { 
        data: activeAnomalies = [], 
        isLoading,
        isConnected: isStreamConnected,
        refetch: refetchAnomalies 
    } = useAnomalyStreamData();
    const [loading, setLoading] = useState(false);
    const [tableLoading, setTableLoading] = useState(true);
    const [lastStableData, setLastStableData] = useState([]);
    const [groupedAnomalies, setGroupedAnomalies] = useState([]);
    const [anomalies, setAnomalies] = useState([]);
    
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

    // Update anomaly status handling
    useEffect(() => {
        // Set anomalies from the streaming data
        setAnomalies(activeAnomalies || []);
    }, [activeAnomalies]);

    const anomalyTypes = [
        { id: 'cpu_stress', name: 'CPU Stress' },
        { id: 'network_bottleneck', name: 'Network Bottleneck' },
        { id: 'cache_bottleneck', name: 'Cache Bottleneck' }
    ];

    const columns = [
        {
            title: 'Type',
            dataIndex: 'type',
            key: 'type',
            render: (type) => {
                const option = anomalyTypes.find(opt => opt.id === type);
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

    const handleAnomalyToggle = async (anomalyType) => {
        setLoading(true);
        try {
            const isActive = isAnomalyActive(anomalyType);
            
            if (isActive) {
                await Promise.all(
                    activeAnomalies
                        .filter(anomaly => anomaly.type === anomalyType)
                        .map(anomaly => anomalyService.deleteAnomaly(anomaly.id))
                );
                message.success(`${anomalyType} anomaly stopped`);
            } else {
                await anomalyService.injectAnomaly(anomalyType);
                message.success(`${anomalyType} anomaly started`);
            }
            
            // Use the refetch from the new hook
            refetchAnomalies();
        } catch (error) {
            console.error(`Error toggling ${anomalyType} anomaly:`, error);
            message.error(`Failed to toggle ${anomalyType} anomaly`);
        } finally {
            setLoading(false);
        }
    };

    const handleStopAllAnomalies = async () => {
        setLoading(true);
        try {
            await anomalyService.stopAnomalyCollection();
            message.success('All anomalies stopped');
            
            // Use the refetch from the new hook
            refetchAnomalies();
        } catch (error) {
            console.error('Error stopping all anomalies:', error);
            message.error('Failed to stop all anomalies');
        } finally {
            setLoading(false);
        }
    };

    const handleRefresh = () => {
        refetchAnomalies();
        message.info('Refreshing anomaly data...');
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

    const renderConnectionStatus = () => {
        return (
            <Tooltip title={isStreamConnected ? "Real-time updates connected" : "Using polling fallback"}>
                <Tag color={isStreamConnected ? "green" : "orange"}>
                    {isStreamConnected ? <CheckCircleOutlined /> : <ReloadOutlined spin />}
                    {isStreamConnected ? " Live" : " Polling"}
                </Tag>
            </Tooltip>
        );
    };

    return (
        <div>
            <Row gutter={[16, 16]}>
                <Col span={24}>
                    <Card 
                        title={
                            <Space>
                                Anomaly Control
                                {renderConnectionStatus()}
                            </Space>
                        }
                        extra={
                            <Button 
                                icon={<ReloadOutlined />} 
                                onClick={handleRefresh} 
                                loading={isLoading}
                                size="small"
                            >
                                Refresh
                            </Button>
                        }
                    >
                        <Space direction="vertical" style={{ width: '100%' }}>
                            <Space wrap>
                                {anomalyTypes.map(option => (
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
                                    icon={<StopOutlined />}
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