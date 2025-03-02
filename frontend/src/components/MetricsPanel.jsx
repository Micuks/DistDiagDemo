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

const formatValue = (value, category, metric) => {
    if (typeof value !== 'number') return '0';
    
    // Handle time values (convert from ns)
    if (metric.includes('time')) {
        return `${(value / 1e9).toFixed(2)}`;
    }
    
    // Handle delay values (convert from ns to ms)
    if (metric.includes('delay')) {
        return `${(value / 1e6).toFixed(2)}`;
    }
    
    // Handle memory values (convert to MB)
    if (category === 'memory' && (metric.includes('memstore') || metric.includes('memory'))) {
        return `${(value / (1024 * 1024)).toFixed(2)}`;
    }
    
    // Handle network bytes (convert to MB)
    if (category === 'network' && metric.includes('bytes')) {
        const mbValue = value / (1024 * 1024);
        if (mbValue >= 1000) {
            return `${(mbValue / 1024).toFixed(2)}`;  // Convert to GB
        }
        return `${mbValue.toFixed(2)}`;
    }

    // Handle disk write size (convert to MB)
    if (metric.includes('write size') || metric.includes('log total size')) {
        const mbValue = value / (1024 * 1024);
        if (mbValue >= 1000) {
            return `${(mbValue / 1024).toFixed(2)}`;  // Convert to GB
        }
        return `${mbValue.toFixed(2)}`;
    }
    
    // Handle percentages
    if (metric === 'cpu usage' || metric.includes('util')) {
        return value.toFixed(1);
    }
    
    // Handle count metrics
    if (metric.includes('count')) {
        return Math.round(value).toLocaleString();
    }
    
    return value.toFixed(2);
};

const MetricsPanel = () => {
    const [metrics, setMetrics] = useState(null);
    const [loading, setLoading] = useState(true);
    const [chartData, setChartData] = useState({});
    const [chartModal, setChartModal] = useState({ visible: false, title: '', data: [], suffix: '', loading: false });
    const nodeRefs = useRef({});
    const [selectedMetrics, setSelectedMetrics] = useState({
        cpu: 'cpu usage',
        memory: 'total memstore used',
        io: 'io read count',
        network: 'rpc net delay',
        transactions: 'trans commit count'
    });

    const fetchChartData = async (nodeIp, category) => {
        try {
            const detailedData = await fetchDetailedMetrics(nodeIp, category);
            if (detailedData?.metrics?.[nodeIp]?.[category]) {
                setChartData(prev => ({
                    ...prev,
                    [`${nodeIp}_${category}`]: detailedData.metrics[nodeIp][category]
                }));
            }
        } catch (error) {
            console.error(`Error fetching ${category} chart data for ${nodeIp}:`, error);
        }
    };

    useEffect(() => {
        const fetchData = async () => {
            try {
                const data = await fetchMetrics();
                setMetrics(data || { metrics: {}, timestamp: null });

                // Fetch detailed data for displayed charts
                if (data?.metrics) {
                    Object.entries(data.metrics).forEach(([nodeIp, nodeData]) => {
                        // Updated categories to match backend
                        const categories = ['cpu', 'memory', 'io', 'network', 'transactions'];
                        Promise.all(categories.map(category => 
                            fetchChartData(nodeIp, category)
                        )).catch(error => {
                            console.error('Error batch fetching chart data:', error);
                        });
                    });
                }
            } catch (error) {
                console.error('Error fetching metrics:', error);
                setMetrics({ metrics: {}, timestamp: null });
            } finally {
                setLoading(false);
            }
        };

        fetchData();
        const interval = setInterval(fetchData, 10000);
        return () => clearInterval(interval);
    }, []);

    const handleMetricClick = (nodeIp, category, metric) => {
        setSelectedMetrics(prev => ({
            ...prev,
            [category]: metric
        }));
        
        // Scroll to the relevant chart
        if (nodeRefs.current[nodeIp]) {
            nodeRefs.current[nodeIp].scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    };

    const formatChartValue = (value, category, metric) => {
        if (category === 'memory' && (metric.includes('memstore') || metric.includes('memory'))) {
            return (value / (1024 * 1024)).toFixed(2);
        }
        if (metric.includes('time') && category === 'cpu') {
            return (value / 1e9).toFixed(2);
        }
        if (category === 'network' && metric.includes('bytes')) {
            return (value / 1024).toFixed(2);
        }
        if (metric.includes('delay')) {
            return (value / 1e6).toFixed(2);
        }
        if (metric === 'cpu usage' || metric.includes('util')) {
            return parseFloat(value).toFixed(1);
        }
        if (metric.includes('count')) {
            return Math.round(value);
        }
        return value.toFixed(2);
    };

    const getMetricSuffix = (category, metric) => {
        if (category === 'memory') return 'MB';
        if (metric.includes('time') && category === 'cpu') return 's';
        if (category === 'network' && metric.includes('bytes')) {
            const mbValue = Number(metric.value) / (1024 * 1024);
            return mbValue >= 1000 ? 'GB' : 'MB';
        }
        if (metric.includes('write size') || metric.includes('log total size')) {
            const mbValue = Number(metric.value) / (1024 * 1024);
            return mbValue >= 1000 ? 'GB' : 'MB';
        }
        if (metric.includes('delay')) return 'ms';
        if (metric === 'cpu usage' || metric.includes('util')) return '%';
        if (metric.includes('count')) return '';
        return '';
    };

    const renderTimeSeriesChart = (data, category, metric) => {
        if (!data || data.length === 0) {
            return <div style={{ textAlign: 'center', padding: '20px' }}>No data available</div>;
        }

        const suffix = getMetricSuffix(category, metric);
        const maxValue = Math.max(...data.map(d => d.value));
        
        return (
            <div style={{ width: '100%', height: 200, marginBottom: 16 }}>
                <ResponsiveContainer>
                    <LineChart data={data} margin={{ left: 5, right: 30, top: 10, bottom: 25 }}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis 
                            dataKey="timestamp" 
                            tickFormatter={(ts) => new Date(ts).toLocaleTimeString()}
                        />
                        <YAxis 
                            width={85}
                            tickFormatter={(value) => formatValue(value, category, metric)}
                            domain={[0, 'auto']}
                            label={{ 
                                value: `(${suffix})`,
                                position: 'bottom',
                                offset: 15,
                                style: { 
                                    textAnchor: 'middle',
                                    fontSize: 12,
                                    fill: '#666'
                                }
                            }}
                            tick={{
                                fontSize: 12,
                                fill: '#666'
                            }}
                        />
                        <Tooltip 
                            formatter={(value) => [`${formatValue(value, category, metric)} ${suffix}`, metric]}
                            labelFormatter={(ts) => new Date(ts).toLocaleString()}
                        />
                        <Line 
                            type="monotone" 
                            dataKey="value" 
                            name={metric} 
                            stroke="#8884d8" 
                            dot={false}
                        />
                    </LineChart>
                </ResponsiveContainer>
            </div>
        );
    };

    const renderMetricStatistic = (nodeIp, metric, values, category) => {
        const latestEntry = Array.isArray(values) ? values[values.length - 1] : null;
        const value = latestEntry?.value ?? 0;
        const suffix = getMetricSuffix(category, metric);
        const formattedValue = `${formatValue(Number(value), category, metric)} ${suffix}`;
        const isSelected = selectedMetrics[category] === metric;
        
        return (
            <Col span={8} key={metric}>
                <div 
                    onClick={() => setSelectedMetrics(prev => ({ ...prev, [category]: metric }))}
                    style={{ 
                        cursor: 'pointer',
                        padding: '8px',
                        borderRadius: '4px',
                        backgroundColor: isSelected ? '#e6f7ff' : 'transparent',
                        border: isSelected ? '1px solid #91d5ff' : '1px solid transparent',
                        transition: 'all 0.3s ease'
                    }}
                    onMouseEnter={(e) => {
                        e.currentTarget.style.backgroundColor = isSelected ? '#e6f7ff' : '#f5f5f5';
                    }}
                    onMouseLeave={(e) => {
                        e.currentTarget.style.backgroundColor = isSelected ? '#e6f7ff' : 'transparent';
                    }}
                >
                    <Statistic 
                        title={metric} 
                        value={formattedValue}
                        valueStyle={{ 
                            fontSize: '16px',
                            color: isSelected ? '#1890ff' : 'rgba(0, 0, 0, 0.85)'
                        }}
                    />
                </div>
            </Col>
        );
    };

    const renderMetricsCard = (title, metrics = {}, nodeIp, category) => {
        const selectedMetric = selectedMetrics[category];
        const chartData = metrics[selectedMetric];

        return (
            <Card title={title} size="small">
                {renderTimeSeriesChart(chartData, category, selectedMetric)}
                <Row gutter={[16, 16]}>
                    {Object.entries(metrics).map(([metric, values]) => 
                        renderMetricStatistic(nodeIp, metric, values, category)
                    )}
                </Row>
            </Card>
        );
    };

    const renderNodeMetrics = (nodeIp, nodeData = {}) => {
        const { cpu, memory, io, network, transactions } = nodeData;

        return (
            <Card 
                title={`Node: ${nodeIp}`} 
                key={nodeIp} 
                style={{ marginBottom: 16 }}
            >
                <Row gutter={[16, 16]}>
                    {/* CPU Metrics */}
                    {cpu && (
                        <Col span={12}>
                            {renderMetricsCard('CPU Usage', cpu, nodeIp, 'cpu')}
                        </Col>
                    )}

                    {/* Memory Metrics */}
                    {memory && (
                        <Col span={12}>
                            {renderMetricsCard('Memory Usage', memory, nodeIp, 'memory')}
                        </Col>
                    )}

                    {/* IO Metrics */}
                    {io && (
                        <Col span={12}>
                            {renderMetricsCard('Disk I/O', io, nodeIp, 'io')}
                        </Col>
                    )}

                    {/* Network Metrics */}
                    {network && (
                        <Col span={12}>
                            {renderMetricsCard('Network', network, nodeIp, 'network')}
                        </Col>
                    )}

                    {/* Transaction Metrics */}
                    {transactions && (
                        <Col span={12}>
                            {renderMetricsCard('Transactions', transactions, nodeIp, 'transactions')}
                        </Col>
                    )}
                </Row>
            </Card>
        );
    };

    if (loading) {
        return (
            <div style={{ textAlign: 'center', padding: '50px' }}>
                <Spin size="large" />
            </div>
        );
    }

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
                                Last Updated: {metrics?.timestamp ? new Date(metrics.timestamp).toLocaleString() : 'N/A'}
                            </Col>
                        </Row>
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