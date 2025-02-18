import React, { useState, useEffect, useRef } from 'react';
import { Space, Card, Row, Col, Statistic, Spin, Select, Modal, Button } from 'antd';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { fetchMetrics, fetchDetailedMetrics } from '../services/metricsService';
import { anomalyService } from '../services/anomalyService';
import { message } from 'antd';

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
    const [chartData, setChartData] = useState({});
    const [chartModal, setChartModal] = useState({ visible: false, title: '', data: [], suffix: '', loading: false });
    const nodeRefs = useRef({});

    const fetchChartData = async (nodeIp, category) => {
        try {
            const detailedData = await fetchDetailedMetrics(nodeIp, category);
            setChartData(prev => ({
                ...prev,
                [`${nodeIp}_${category}`]: detailedData.metrics[nodeIp][category]
            }));
        } catch (error) {
            console.error(`Error fetching ${category} chart data for ${nodeIp}:`, error);
        }
    };

    useEffect(() => {
        const fetchData = async () => {
            try {
                const data = await fetchMetrics();
                setMetrics(data);

                // Fetch detailed data for displayed charts
                if (data?.metrics) {
                    Object.entries(data.metrics).forEach(([nodeIp, nodeData]) => {
                        // Batch fetch all chart data
                        const categories = ['cpu', 'memory', 'swap', 'io', 'network', 'tcp_udp'];
                        Promise.all(categories.map(category => 
                            fetchChartData(nodeIp, category)
                        )).catch(error => {
                            console.error('Error batch fetching chart data:', error);
                        });
                    });
                }
            } catch (error) {
                console.error('Error fetching metrics:', error);
            } finally {
                setLoading(false);
            }
        };

        fetchData();
        const interval = setInterval(fetchData, 10000);

        return () => {
            clearInterval(interval);
        };
    }, []);

    const handleMetricClick = async (nodeIp, category, metric, title, suffix = '') => {
        setChartModal({
            visible: true,
            title,
            data: [],
            suffix,
            loading: true
        });

        try {
            // Use cached data if available
            const cachedData = chartData[`${nodeIp}_${category}`];
            if (cachedData && cachedData[metric]) {
                setChartModal(prev => ({
                    ...prev,
                    data: cachedData[metric].data,
                    loading: false
                }));
            } else {
                const detailedData = await fetchDetailedMetrics(nodeIp, category);
                const metricData = detailedData.metrics[nodeIp][category][metric].data;
                setChartModal(prev => ({
                    ...prev,
                    data: metricData,
                    loading: false
                }));
            }
        } catch (error) {
            console.error('Error fetching detailed metrics:', error);
            message.error('Failed to load detailed metrics');
            setChartModal(prev => ({
                ...prev,
                loading: false
            }));
        }
    };

    const renderChart = (nodeIp, category, metric, title, suffix = '') => {
        const data = chartData[`${nodeIp}_${category}`]?.[metric]?.data;
        return data ? (
            <MetricsChart 
                data={data}
                title={title}
                suffix={suffix}
            />
        ) : (
            <div style={{ textAlign: 'center', padding: '20px' }}>
                <Spin size="large" />
            </div>
        );
    };

    const scrollToNode = (nodeIp) => {
        nodeRefs.current[nodeIp]?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    };

    const renderMetricStatistic = (nodeIp, metric, data, category, suffix = '') => (
        <Col span={8} key={metric}>
            <div 
                onClick={() => handleMetricClick(nodeIp, category, metric, `${category} - ${metric}`, suffix)}
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
                                        {renderChart(nodeIp, 'cpu', 'util', 'CPU Utilization', '%')}
                                    </Col>
                                    {Object.entries(cpu).map(([metric, data]) => 
                                        renderMetricStatistic(nodeIp, metric, data, 'cpu', '%')
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
                                        {renderChart(nodeIp, 'memory', 'util', 'Memory Utilization', '%')}
                                    </Col>
                                    {Object.entries(memory).map(([metric, data]) => 
                                        renderMetricStatistic(
                                            nodeIp,
                                            metric, 
                                            data, 
                                            'memory',
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
                                        {renderChart(nodeIp, 'swap', 'util', 'Swap Utilization', '%')}
                                    </Col>
                                    {Object.entries(swap).map(([metric, data]) => 
                                        renderMetricStatistic(
                                            nodeIp,
                                            metric, 
                                            data, 
                                            'swap',
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
                                        {renderChart(nodeIp, 'io', 'util', 'I/O Utilization', '%')}
                                    </Col>
                                    {Object.entries(io.aggregated).map(([metric, data]) => 
                                        renderMetricStatistic(
                                            nodeIp,
                                            metric, 
                                            data, 
                                            'io',
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
                                        {renderChart(nodeIp, 'network', 'bytin', 'Network In', 'B/s')}
                                    </Col>
                                    {Object.entries(network).map(([metric, data]) => 
                                        renderMetricStatistic(
                                            nodeIp,
                                            metric, 
                                            data, 
                                            'network',
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
                                        {renderChart(nodeIp, 'tcp_udp', 'active', 'TCP Active Connections')}
                                    </Col>
                                    {Object.entries(tcp_udp).map(([metric, data]) => 
                                        renderMetricStatistic(
                                            nodeIp,
                                            metric, 
                                            data, 
                                            'tcp_udp'
                                        )
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
                onCancel={() => setChartModal({ visible: false, title: '', data: [], suffix: '', loading: false })}
                footer={null}
                width={800}
            >
                {chartModal.loading ? (
                    <div style={{ textAlign: 'center', padding: '20px' }}>
                        <Spin size="large" />
                    </div>
                ) : (
                    <MetricsChart 
                        data={chartModal.data} 
                        title={chartModal.title}
                        suffix={chartModal.suffix}
                    />
                )}
            </Modal>
        </Space>
    );
};

export default MetricsPanel; 