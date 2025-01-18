import React, { useState, useEffect, useRef } from 'react';
import { Space, Card, Row, Col, Statistic, Spin, Select, Modal, Button } from 'antd';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { fetchMetrics, startMetricsCollection, stopMetricsCollection } from '../services/metricsService';

const { Option } = Select;

const MetricsChart = ({ data, title, dataKey, suffix = '' }) => (
    <div style={{ width: '100%', height: 200 }}>
        <ResponsiveContainer>
            <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="timestamp" />
                <YAxis />
                <Tooltip formatter={(value) => `${value}${suffix}`} />
                <Legend />
                <Line type="monotone" dataKey="value" name={title} stroke="#8884d8" dot={false} />
            </LineChart>
        </ResponsiveContainer>
    </div>
);

const MetricsPanel = () => {
    const [metrics, setMetrics] = useState(null);
    const [loading, setLoading] = useState(true);
    const [chartModal, setChartModal] = useState({ visible: false, title: '', data: [], suffix: '' });
    const nodeRefs = useRef({});

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
                setMetrics(data);
            } catch (error) {
                console.error('Error fetching metrics:', error);
            } finally {
                setLoading(false);
            }
        };

        startCollection();
        fetchData();
        const interval = setInterval(fetchData, 5000);

        return () => {
            clearInterval(interval);
            stopMetricsCollection().catch(error => {
                console.error('Error stopping metrics collection:', error);
            });
        };
    }, []);

    const scrollToNode = (nodeIp) => {
        nodeRefs.current[nodeIp]?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    };

    const handleMetricClick = (title, data, suffix = '') => {
        setChartModal({
            visible: true,
            title,
            data,
            suffix
        });
    };

    const renderMetricStatistic = (metric, data, category, suffix = '') => (
        <Col span={8} key={metric}>
            <div 
                onClick={() => handleMetricClick(`${category} - ${metric}`, data.data, suffix)}
                style={{ cursor: 'pointer' }}
            >
                <Statistic 
                    title={metric} 
                    value={data.latest} 
                    suffix={suffix}
                />
            </div>
        </Col>
    );

    const renderNodeMetrics = (nodeIp, nodeData) => {
        const { cpu, memory, io, network, tcp_udp, swap } = nodeData;

        return (
            <Card 
                ref={el => nodeRefs.current[nodeIp] = el}
                title={`Node: ${nodeIp}`} 
                key={nodeIp} 
                style={{ marginBottom: 16 }}
                collapsible
                defaultActiveKey={['1']}
            >
                <Row gutter={[16, 16]}>
                    {/* CPU Metrics */}
                    {cpu && (
                        <Col span={12}>
                            <Card title="CPU Usage">
                                <Row gutter={[16, 16]}>
                                    <Col span={24}>
                                        <MetricsChart 
                                            data={cpu.util.data} 
                                            title="CPU Utilization" 
                                            suffix="%" 
                                        />
                                    </Col>
                                    {Object.entries(cpu).map(([metric, data]) => 
                                        renderMetricStatistic(metric, data, 'CPU', '%')
                                    )}
                                </Row>
                            </Card>
                        </Col>
                    )}

                    {/* Memory Metrics */}
                    {memory && (
                        <Col span={12}>
                            <Card title="Memory Usage">
                                <Row gutter={[16, 16]}>
                                    <Col span={24}>
                                        <MetricsChart 
                                            data={memory.util.data} 
                                            title="Memory Utilization" 
                                            suffix="%" 
                                        />
                                    </Col>
                                    {Object.entries(memory).map(([metric, data]) => 
                                        renderMetricStatistic(
                                            metric, 
                                            data, 
                                            'Memory',
                                            metric === 'util' ? '%' : 'MB'
                                        )
                                    )}
                                </Row>
                            </Card>
                        </Col>
                    )}

                    {/* Swap Metrics */}
                    {swap && (
                        <Col span={12}>
                            <Card title="Swap Usage">
                                <Row gutter={[16, 16]}>
                                    <Col span={24}>
                                        <MetricsChart 
                                            data={swap.util.data} 
                                            title="Swap Utilization" 
                                            suffix="%" 
                                        />
                                    </Col>
                                    {Object.entries(swap).map(([metric, data]) => 
                                        renderMetricStatistic(
                                            metric, 
                                            data, 
                                            'Swap',
                                            metric === 'util' ? '%' : 
                                            metric.includes('rate') ? '/s' : 'MB'
                                        )
                                    )}
                                </Row>
                            </Card>
                        </Col>
                    )}

                    {/* IO Metrics */}
                    {io && (
                        <Col span={12}>
                            <Card title="Disk I/O">
                                <Row gutter={[16, 16]}>
                                    <Col span={24}>
                                        <MetricsChart 
                                            data={io.aggregated.util.data} 
                                            title="I/O Utilization" 
                                            suffix="%" 
                                        />
                                    </Col>
                                    {Object.entries(io.aggregated).map(([metric, data]) => 
                                        renderMetricStatistic(
                                            metric, 
                                            data, 
                                            'I/O',
                                            metric === 'util' ? '%' : ''
                                        )
                                    )}
                                </Row>
                            </Card>
                        </Col>
                    )}

                    {/* Network Metrics */}
                    {network && (
                        <Col span={12}>
                            <Card title="Network">
                                <Row gutter={[16, 16]}>
                                    <Col span={24}>
                                        <MetricsChart 
                                            data={network.bytin.data} 
                                            title="Network In" 
                                            suffix=" B/s" 
                                        />
                                    </Col>
                                    {Object.entries(network).map(([metric, data]) => 
                                        renderMetricStatistic(
                                            metric, 
                                            data, 
                                            'Network',
                                            metric.includes('byt') ? 'B/s' : 'pkt/s'
                                        )
                                    )}
                                </Row>
                            </Card>
                        </Col>
                    )}

                    {/* TCP/UDP Metrics */}
                    {tcp_udp && (
                        <Col span={12}>
                            <Card title="TCP/UDP">
                                <Row gutter={[16, 16]}>
                                    <Col span={24}>
                                        <MetricsChart 
                                            data={tcp_udp.active.data} 
                                            title="TCP Active Connections" 
                                        />
                                    </Col>
                                    {Object.entries(tcp_udp).map(([metric, data]) => 
                                        renderMetricStatistic(metric, data, 'TCP/UDP')
                                    )}
                                </Row>
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
                        <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
                            <Col>
                                <Space>
                                    <span>Quick Navigate:</span>
                                    {metrics?.metrics && Object.keys(metrics.metrics).map(nodeIp => (
                                        <Button 
                                            key={nodeIp} 
                                            onClick={() => scrollToNode(nodeIp)}
                                            size="small"
                                        >
                                            {nodeIp}
                                        </Button>
                                    ))}
                                </Space>
                            </Col>
                            <Col>
                                Last Updated: {metrics?.timestamp ? new Date(metrics.timestamp).toLocaleString() : 'N/A'}
                            </Col>
                        </Row>
                        {metrics?.metrics && Object.entries(metrics.metrics).map(([nodeIp, nodeData]) => 
                            renderNodeMetrics(nodeIp, nodeData)
                        )}
                    </>
                )}
            </Card>

            <Modal
                title={chartModal.title}
                open={chartModal.visible}
                onCancel={() => setChartModal({ ...chartModal, visible: false })}
                footer={null}
                width={800}
            >
                <MetricsChart 
                    data={chartModal.data} 
                    title={chartModal.title}
                    suffix={chartModal.suffix}
                />
            </Modal>
        </Space>
    );
};

export default MetricsPanel; 