import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Statistic, Table, Spin, Button } from 'antd';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import { fetchMetrics, startMetricsCollection, stopMetricsCollection } from '../services/metricsService';

const MetricsPanel = () => {
    const [metrics, setMetrics] = useState(null);
    const [loading, setLoading] = useState(true);
    const [collecting, setCollecting] = useState(false);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const data = await fetchMetrics();
                setMetrics(data);
            } catch (error) {
                console.error('Error fetching metrics:', error);
            } finally {
                setLoading(false);
            }
        };

        // Fetch initially and then every 30 seconds
        fetchData();
        const interval = setInterval(fetchData, 30000);

        return () => clearInterval(interval);
    }, []);

    const handleStartCollection = async () => {
        try {
            await startMetricsCollection();
            setCollecting(true);
        } catch (error) {
            console.error('Error starting collection:', error);
        }
    };

    const handleStopCollection = async () => {
        try {
            await stopMetricsCollection();
            setCollecting(false);
        } catch (error) {
            console.error('Error stopping collection:', error);
        }
    };

    if (loading) {
        return <Spin size="large" />;
    }

    const renderNodeMetrics = (nodeIp, nodeData) => {
        const { cpu, memory, io, network } = nodeData;

        return (
            <Card title={`Node: ${nodeIp}`} key={nodeIp} style={{ marginBottom: 16 }}>
                <Row gutter={16}>
                    {/* CPU Metrics */}
                    {cpu && (
                        <Col span={6}>
                            <Card title="CPU Usage">
                                <Statistic title="User" value={cpu.user.toFixed(2)} suffix="%" />
                                <Statistic title="System" value={cpu.system.toFixed(2)} suffix="%" />
                                <Statistic title="IO Wait" value={cpu.iowait.toFixed(2)} suffix="%" />
                                <Statistic title="Idle" value={cpu.idle.toFixed(2)} suffix="%" />
                            </Card>
                        </Col>
                    )}

                    {/* Memory Metrics */}
                    {memory && (
                        <Col span={6}>
                            <Card title="Memory Usage">
                                <Statistic 
                                    title="Used" 
                                    value={(memory.used / 1024).toFixed(2)} 
                                    suffix="GB" 
                                />
                                <Statistic 
                                    title="Free" 
                                    value={(memory.free / 1024).toFixed(2)} 
                                    suffix="GB" 
                                />
                                <Statistic 
                                    title="Buffer" 
                                    value={(memory.buffer / 1024).toFixed(2)} 
                                    suffix="GB" 
                                />
                                <Statistic 
                                    title="Cache" 
                                    value={(memory.cache / 1024).toFixed(2)} 
                                    suffix="GB" 
                                />
                            </Card>
                        </Col>
                    )}

                    {/* IO Metrics */}
                    {io && (
                        <Col span={6}>
                            <Card title="Disk I/O">
                                {/* Aggregated Metrics */}
                                <Card type="inner" title="Overall I/O">
                                    <Statistic 
                                        title="Read" 
                                        value={(io.rsecs * 512 / 1024 / 1024).toFixed(2)} 
                                        suffix="MB/s" 
                                    />
                                    <Statistic 
                                        title="Write" 
                                        value={(io.wsecs * 512 / 1024 / 1024).toFixed(2)} 
                                        suffix="MB/s" 
                                    />
                                    <Statistic 
                                        title="Read IOPS" 
                                        value={io.rs.toFixed(1)} 
                                    />
                                    <Statistic 
                                        title="Write IOPS" 
                                        value={io.ws.toFixed(1)} 
                                    />
                                    <Statistic 
                                        title="Utilization" 
                                        value={io.util.toFixed(1)} 
                                        suffix="%" 
                                    />
                                    <Statistic 
                                        title="Avg Wait" 
                                        value={io.await.toFixed(2)} 
                                        suffix="ms" 
                                    />
                                </Card>
                                
                                {/* Per-device Metrics */}
                                {io.devices && Object.entries(io.devices).map(([device, metrics]) => (
                                    <Card 
                                        type="inner" 
                                        title={`Device: ${device}`} 
                                        key={device}
                                        style={{ marginTop: 16 }}
                                    >
                                        <Statistic 
                                            title="Read" 
                                            value={(metrics.rsecs * 512 / 1024 / 1024).toFixed(2)} 
                                            suffix="MB/s" 
                                        />
                                        <Statistic 
                                            title="Write" 
                                            value={(metrics.wsecs * 512 / 1024 / 1024).toFixed(2)} 
                                            suffix="MB/s" 
                                        />
                                        <Statistic 
                                            title="Utilization" 
                                            value={metrics.util.toFixed(1)} 
                                            suffix="%" 
                                        />
                                    </Card>
                                ))}
                            </Card>
                        </Col>
                    )}

                    {/* Network Metrics */}
                    {network && (
                        <Col span={6}>
                            <Card title="Network">
                                <Statistic 
                                    title="RX" 
                                    value={(network.rx_bytes / 1024 / 1024).toFixed(2)} 
                                    suffix="MB/s" 
                                />
                                <Statistic 
                                    title="TX" 
                                    value={(network.tx_bytes / 1024 / 1024).toFixed(2)} 
                                    suffix="MB/s" 
                                />
                                <Statistic title="RX Packets" value={network.rx_packets} />
                                <Statistic title="TX Packets" value={network.tx_packets} />
                            </Card>
                        </Col>
                    )}
                </Row>
            </Card>
        );
    };

    return (
        <div>
            <div style={{ marginBottom: 16 }}>
                <Button 
                    type="primary" 
                    onClick={collecting ? handleStopCollection : handleStartCollection}
                    style={{ marginRight: 8 }}
                >
                    {collecting ? 'Stop Collection' : 'Start Collection'}
                </Button>
                Last Updated: {metrics?.timestamp || 'N/A'}
            </div>
            
            {metrics?.metrics && Object.entries(metrics.metrics).map(([nodeIp, nodeData]) => 
                renderNodeMetrics(nodeIp, nodeData)
            )}
        </div>
    );
};

export default MetricsPanel; 