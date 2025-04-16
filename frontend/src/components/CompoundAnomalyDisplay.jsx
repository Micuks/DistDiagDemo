import React, { useState, useEffect } from 'react';
import { Card, Table, Typography, Tag, Spin, Empty, Alert, Tabs, Collapse } from 'antd';
import { WarningOutlined, CheckCircleOutlined, SyncOutlined } from '@ant-design/icons';
import { anomalyService } from '../services/anomalyService';
import { FiAlertTriangle } from 'react-icons/fi';

const { Title, Text } = Typography;
const { Panel } = Collapse;

const CompoundAnomalyDisplay = () => {
    const [compoundData, setCompoundData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [refreshKey, setRefreshKey] = useState(0);

    // Function to fetch compound anomaly data
    const fetchCompoundAnomalies = async () => {
        try {
            setLoading(true);
            const data = await anomalyService.getCompoundAnomalies();
            setCompoundData(data);
            setError(null);
        } catch (err) {
            console.error('Error fetching compound anomalies:', err);
            setError('Failed to fetch anomaly data. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    // Load data on component mount and when refreshKey changes
    useEffect(() => {
        fetchCompoundAnomalies();
        
        // Set up polling interval
        const intervalId = setInterval(() => {
            fetchCompoundAnomalies();
        }, 10000); // Refresh every 10 seconds
        
        // Clean up interval on component unmount
        return () => clearInterval(intervalId);
    }, [refreshKey]);

    // Manual refresh handler
    const handleRefresh = () => {
        setRefreshKey(prevKey => prevKey + 1);
    };

    // Helper to get color based on anomaly type
    const getAnomalyTypeColor = (type) => {
        switch (type) {
            case 'cpu_stress':
                return 'volcano';
            case 'network_bottleneck':
                return 'geekblue';
            case 'cache_bottleneck':
                return 'purple';
                return 'orange';
            default:
                return 'default';
        }
    };

    // Helper to format anomaly type name
    const formatAnomalyType = (type) => {
        switch (type) {
            case 'cpu_stress':
                return 'CPU Stress';
            case 'network_bottleneck':
                return 'Network Bottleneck';
            case 'cache_bottleneck':
                return 'Cache Bottleneck';
                return 'Too Many Indexes';
            default:
                return type;
        }
    };

    const getAnomalyIcon = (type) => {
        switch (type) {
            case 'cpu_stress':
                return <DesktopOutlined />;
            case 'network_bottleneck':
                return <WifiOutlined />;
            case 'cache_bottleneck':
                return <DatabaseOutlined />;
            default:
                return <ExclamationCircleOutlined />;
        }
    };

    const getAnomalyLabel = (type) => {
        switch (type) {
            case 'cpu_stress':
                return 'CPU Stress';
            case 'network_bottleneck':
                return 'Network Bottleneck';
            case 'cache_bottleneck':
                return 'Cache Bottleneck';
            default:
                return type;
        }
    };

    // Render compound anomaly panels
    const renderCompoundAnomalies = () => {
        if (!compoundData || !compoundData.compound_anomalies) {
            return <Empty description="No compound anomalies detected" />;
        }

        const compoundAnomalies = compoundData.compound_anomalies;
        const nodes = Object.keys(compoundAnomalies);
        
        if (nodes.length === 0) {
            return <Empty description="No compound anomalies detected" />;
        }

        return (
            <Collapse>
                {nodes.map(node => (
                    <Panel 
                        header={
                            <span>
                                <WarningOutlined style={{ color: 'red', marginRight: 8 }} />
                                <Text strong>{node}</Text>
                                <Text type="secondary" style={{ marginLeft: 8 }}>
                                    ({compoundAnomalies[node].length} anomalies)
                                </Text>
                            </span>
                        } 
                        key={node}
                    >
                        <Table 
                            dataSource={compoundAnomalies[node]}
                            rowKey={(record) => `${record.type}-${record.score}`}
                            pagination={false}
                            size="small"
                            columns={[
                                {
                                    title: 'Anomaly Type',
                                    dataIndex: 'type',
                                    key: 'type',
                                    render: (type) => (
                                        <Tag color={getAnomalyTypeColor(type)}>
                                            {formatAnomalyType(type)}
                                        </Tag>
                                    ),
                                },
                                {
                                    title: 'Score',
                                    dataIndex: 'score',
                                    key: 'score',
                                    render: (score) => (
                                        <Text>{score.toFixed(2)}</Text>
                                    ),
                                },
                                {
                                    title: 'Timestamp',
                                    dataIndex: 'timestamp',
                                    key: 'timestamp',
                                    render: (timestamp) => {
                                        const date = new Date(timestamp);
                                        return <Text>{date.toLocaleTimeString()}</Text>;
                                    },
                                }
                            ]}
                        />
                    </Panel>
                ))}
            </Collapse>
        );
    };

    // Render normal anomalies
    const renderNormalAnomalies = () => {
        if (!compoundData || !compoundData.anomalies || compoundData.anomalies.length === 0) {
            return <Empty description="No anomalies detected" />;
        }

        return (
            <Table 
                dataSource={compoundData.anomalies}
                rowKey={(record) => `${record.node}-${record.type}`}
                pagination={false}
                columns={[
                    {
                        title: 'Node',
                        dataIndex: 'node',
                        key: 'node',
                    },
                    {
                        title: 'Anomaly Type',
                        dataIndex: 'type',
                        key: 'type',
                        render: (type) => (
                            <Tag color={getAnomalyTypeColor(type)}>
                                {formatAnomalyType(type)}
                            </Tag>
                        ),
                    },
                    {
                        title: 'Score',
                        dataIndex: 'score',
                        key: 'score',
                        render: (score) => (
                            <Text>{score.toFixed(2)}</Text>
                        ),
                    },
                    {
                        title: 'Timestamp',
                        dataIndex: 'timestamp',
                        key: 'timestamp',
                        render: (timestamp) => {
                            const date = new Date(timestamp);
                            return <Text>{date.toLocaleTimeString()}</Text>;
                        },
                    }
                ]}
            />
        );
    };

    return (
        <Card 
            title={
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Title level={4}>Anomaly Detection Results</Title>
                    <SyncOutlined 
                        spin={loading} 
                        onClick={handleRefresh}
                        style={{ fontSize: '18px', cursor: 'pointer' }} 
                    />
                </div>
            }
        >
            {error && (
                <Alert 
                    message="Error"
                    description={error}
                    type="error"
                    showIcon
                    style={{ marginBottom: 16 }}
                />
            )}
            
            {loading ? (
                <div style={{ textAlign: 'center', padding: '20px' }}>
                    <Spin size="large" />
                    <div style={{ marginTop: 16 }}>Loading anomaly data...</div>
                </div>
            ) : (
                <Tabs 
                    defaultActiveKey="compound"
                    items={[
                        {
                            key: 'compound',
                            label: (
                                <span>
                                    <WarningOutlined />
                                    Compound Anomalies
                                </span>
                            ),
                            children: (
                                <>
                                    <Alert
                                        message="Compound Anomaly Detection"
                                        description="Compound anomalies occur when multiple issues affect the same node simultaneously. These are often more serious and require prioritized attention."
                                        type="info"
                                        showIcon
                                        style={{ marginBottom: 16 }}
                                    />
                                    {renderCompoundAnomalies()}
                                </>
                            )
                        },
                        {
                            key: 'single',
                            label: (
                                <span>
                                    <WarningOutlined />
                                    Single Anomalies
                                </span>
                            ),
                            children: renderNormalAnomalies()
                        }
                    ]}
                />
            )}
        </Card>
    );
};

export default CompoundAnomalyDisplay; 