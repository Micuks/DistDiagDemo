import React, { useState, useEffect } from 'react';
import { Space, Card, Row, Col, Statistic, Spin } from 'antd';
import { fetchMetrics, startMetricsCollection, stopMetricsCollection } from '../services/metricsService';

const MetricsPanel = () => {
    const [metrics, setMetrics] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const startCollection = async () => {
            try {
                await startMetricsCollection();
            } catch (error) {
                console.error('Error starting metrics collection:', error);
            }
        };

        const fetchData = async () => {
            try {
                const data = await fetchMetrics();
                console.log('Received metrics data:', data);
                setMetrics(data);
            } catch (error) {
                console.error('Error fetching metrics:', error);
            } finally {
                setLoading(false);
            }
        };

        // Start collection and fetch data initially
        startCollection();
        fetchData();

        // Then fetch every 5 seconds instead of 30
        const interval = setInterval(fetchData, 5000);

        // Cleanup: stop collection when component unmounts
        return () => {
            clearInterval(interval);
            stopMetricsCollection().catch(error => {
                console.error('Error stopping metrics collection:', error);
            });
        };
    }, []);

    const renderNodeMetrics = (nodeIp, nodeData) => {
        const { cpu, memory, io, network } = nodeData;

        return (
            <Card title={`Node: ${nodeIp}`} key={nodeIp} style={{ marginBottom: 16 }}>
                <Row gutter={16}>
                    {/* CPU Metrics */}
                    {cpu && (
                        <Col span={6}>
                            <Card title="CPU Usage">
                                {Object.entries(cpu).map(([metric, data]) => (
                                    <Statistic 
                                        key={metric}
                                        title={metric} 
                                        value={data.latest} 
                                        suffix="%" 
                                    />
                                ))}
                            </Card>
                        </Col>
                    )}

                    {/* Memory Metrics */}
                    {memory && (
                        <Col span={6}>
                            <Card title="Memory Usage">
                                {Object.entries(memory).map(([metric, data]) => (
                                    <Statistic 
                                        key={metric}
                                        title={metric} 
                                        value={data.latest} 
                                        suffix={metric === 'util' ? '%' : 'MB'} 
                                    />
                                ))}
                            </Card>
                        </Col>
                    )}

                    {/* IO Metrics */}
                    {io && (
                        <Col span={6}>
                            <Card title="Disk I/O">
                                <Card type="inner" title="Aggregated I/O">
                                    {Object.entries(io.aggregated).map(([metric, data]) => (
                                        <Statistic 
                                            key={metric}
                                            title={metric} 
                                            value={data.latest} 
                                            suffix={metric === 'util' ? '%' : ''} 
                                        />
                                    ))}
                                </Card>
                            </Card>
                        </Col>
                    )}

                    {/* Network Metrics */}
                    {network && (
                        <Col span={6}>
                            <Card title="Network">
                                {Object.entries(network).map(([metric, data]) => (
                                    <Statistic 
                                        key={metric}
                                        title={metric} 
                                        value={data.latest} 
                                        suffix={metric.includes('byt') ? 'B/s' : 'pkt/s'} 
                                    />
                                ))}
                            </Card>
                        </Col>
                    )}
                </Row>
            </Card>
        );
    };

    return (
        <Space direction="vertical" style={{ width: '100%' }}>
            <Card title="System Metrics" style={{ marginBottom: 16 }}>
                {loading ? (
                    <div style={{ textAlign: 'center', padding: '20px' }}>
                        <Spin size="large" />
                    </div>
                ) : (
                    <>
                        <div style={{ marginBottom: 16 }}>
                            Last Updated: {metrics?.timestamp ? new Date(metrics.timestamp).toLocaleString() : 'N/A'}
                        </div>
                        {metrics?.metrics && Object.entries(metrics.metrics).map(([nodeIp, nodeData]) => 
                            renderNodeMetrics(nodeIp, nodeData)
                        )}
                    </>
                )}
            </Card>
        </Space>
    );
};

export default MetricsPanel; 