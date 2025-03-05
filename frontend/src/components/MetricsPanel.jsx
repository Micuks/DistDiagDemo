import React, { useState, useEffect, useRef } from 'react';
import { Space, Card, Row, Col, Statistic, Spin, Select, Modal, Button, Tooltip } from 'antd';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Legend, ResponsiveContainer } from 'recharts';
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
    if (typeof metric === 'string' ? metric.includes('time') : metric.name.includes('time')) {
        return `${(value / 1e9).toFixed(2).toLocaleString()}`;
    }
    
    // Handle delay values (convert from µs to ms)
    if (typeof metric === 'string' ? metric.includes('delay') : metric.name.includes('delay')) {
        return `${(value / 1000).toFixed(2).toLocaleString()}`;
    }
    
    // Handle memory values (convert to MB)
    if (category === 'memory' && (
        typeof metric === 'string' ? 
        (metric.includes('memstore') || metric.includes('memory')) : 
        (metric.name.includes('memstore') || metric.name.includes('memory'))
    )) {
        return `${(value / (1024 * 1024)).toFixed(2).toLocaleString()}`;
    }
    
    // Handle network bytes (convert to MB/GB)
    if (category === 'network' && (typeof metric === 'string' ? metric.includes('bytes') : metric.name.includes('bytes'))) {
        const mbValue = value / (1024 * 1024);
        if (mbValue >= 1000) {
            return `${(mbValue / 1024).toFixed(2).toLocaleString()}`;  // Convert to GB
        }
        return `${mbValue.toFixed(2).toLocaleString()}`;
    }

    // Handle disk write size (convert to MB/GB)
    if ((typeof metric === 'string' ? 
        (metric.includes('write size') || metric.includes('log total size')) :
        (metric.name.includes('write size') || metric.name.includes('log total size'))
    )) {
        const mbValue = value / (1024 * 1024);
        if (mbValue >= 1000) {
            return `${(mbValue / 1024).toFixed(2).toLocaleString()}`;  // Convert to GB
        }
        return `${mbValue.toFixed(2).toLocaleString()}`;
    }
    
    // Handle percentages
    if ((typeof metric === 'string' ? metric === 'cpu usage' || metric.includes('util') : 
        metric.name === 'cpu usage' || metric.name.includes('util'))) {
        return value.toFixed(1);
    }
    
    // Handle count metrics with thousands separators
    if (typeof metric === 'string' ? metric.includes('count') : metric.name.includes('count')) {
        return Math.round(value).toLocaleString();
    }
    
    // Default formatting with thousands separators
    return value.toLocaleString(undefined, {
        maximumFractionDigits: 0,
        useGrouping: true
    });
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
            return (value / (1024 * 1024)).toFixed(2).toLocaleString();
        }
        if (metric.includes('time') && category === 'cpu') {
            return (value / 1e9).toFixed(2).toLocaleString();
        }
        if (category === 'network' && metric.includes('bytes')) {
            return (value / 1024).toFixed(2).toLocaleString();
        }
        if (metric.includes('delay')) {
            return (value / 1000).toFixed(2).toLocaleString();
        }
        if (metric === 'cpu usage' || metric.includes('util')) {
            return parseFloat(value).toFixed(1);
        }
        if (metric.includes('count')) {
            return Math.round(value).toLocaleString();
        }
        return value.toLocaleString(undefined, {
            maximumFractionDigits: 0,
            useGrouping: true
        });
    };

    const getMetricSuffix = (category, metric) => {
        // Handle both string and object metric formats
        const metricName = typeof metric === 'string' ? metric : metric.name;
        const metricValue = typeof metric === 'object' ? metric.value : 0;

        if (category === 'memory') return 'MB';
        if (metricName.includes('time') && category === 'cpu') return 's';
        if (category === 'network' && metricName.includes('bytes')) {
            const mbValue = metricValue / (1024 * 1024);
            return mbValue >= 1000 ? 'GB' : 'MB';
        }
        if (metricName.includes('write size') || metricName.includes('log total size')) {
            const mbValue = metricValue / (1024 * 1024);
            return mbValue >= 1000 ? 'GB' : 'MB';
        }
        if (metricName.includes('delay')) {
            const msValue = metricValue / 1000;  // Convert µs to ms
            return msValue >= 1000 ? 's' : 'ms';
        }
        if (metricName === 'cpu usage' || metricName.includes('util')) return '%';
        if (metricName.includes('count')) return '';
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

    const renderMetricItem = (nodeIp, category, metricName, values) => {
        const latestEntry = Array.isArray(values) ? values[values.length - 1] : null;
        if (!latestEntry) return null;

        // Detect monotonically increasing counter metrics
        const isMonotonicCounter = 
            Array.isArray(values) && 
            values.length > 2 && 
            // Check if values are monotonically increasing
            values.slice(-5).every((v, i, arr) => i === 0 || v.value >= arr[i-1].value) &&
            // And check if it's a large number (likely a counter)
            values[values.length - 1].value > 1e6;
            
        // For monotonic counters, if no fluctuation is detected, don't highlight
        const hasFluctuation = isMonotonicCounter && Math.abs(latestEntry.z_score || 0) < 2.0 
            ? false 
            : (latestEntry?.has_fluctuation || false);
            
        const formattedValue = formatValue(latestEntry.value, category, metricName);
        const suffix = getMetricSuffix(category, metricName);
        const isSelected = selectedMetrics[category] === metricName;
        
        // Clean up metric name for display
        const displayName = metricName
            .replace(/_/g, ' ') // Replace underscores with spaces
            .replace(/\b\w/g, l => l.toUpperCase()) // Capitalize first letter of each word
            // Fix common tech abbreviations
            .replace(/\bCpu\b/g, 'CPU')
            .replace(/\bIo\b/g, 'IO')
            .replace(/\bRpc\b/g, 'RPC')
            .replace(/\bSql\b/g, 'SQL') 
            .replace(/\bMysql\b/g, 'MySQL')
            .replace(/\bIops\b/g, 'IOPS')
            .replace(/\bDb\b/g, 'DB')
            .replace(/\bApi\b/g, 'API')
            .replace(/\bUuid\b/g, 'UUID')
            .replace(/\bTps\b/g, 'TPS')
            .replace(/\bQps\b/g, 'QPS')
            .replace(/\bIp\b/g, 'IP')
            .replace(/\bTcp\b/g, 'TCP')
            .replace(/\bUdp\b/g, 'UDP')
            .replace(/\bHttp\b/g, 'HTTP')
            .replace(/\bHttps\b/g, 'HTTPS')
            .replace(/\bDns\b/g, 'DNS')
            .replace(/\bOs\b/g, 'OS');

        return (
            <Col span={8} key={metricName}>
                <div 
                    onClick={() => setSelectedMetrics(prev => ({ ...prev, [category]: metricName }))}
                    style={{ 
                        cursor: 'pointer',
                        padding: '8px',
                        borderRadius: '4px',
                        backgroundColor: hasFluctuation ? '#fffbe6' : isSelected ? '#e6f7ff' : 'transparent',
                        borderLeft: hasFluctuation ? '3px solid #ffd666' : isSelected ? '1px solid #91d5ff' : '1px solid transparent',
                        boxShadow: hasFluctuation ? '0 2px 8px rgba(255, 214, 102, 0.1)' : 'none',
                        transition: 'all 0.3s ease',
                        height: '100%',
                        display: 'flex',
                        flexDirection: 'column',
                        justifyContent: 'space-between'
                    }}
                    onMouseEnter={(e) => {
                        if (!hasFluctuation) {
                            e.currentTarget.style.backgroundColor = isSelected ? '#e6f7ff' : '#f5f5f5';
                        }
                    }}
                    onMouseLeave={(e) => {
                        if (!hasFluctuation) {
                            e.currentTarget.style.backgroundColor = isSelected ? '#e6f7ff' : 'transparent';
                        }
                    }}
                >
                    <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: '4px' }}>
                        <span style={{ 
                            fontSize: '16px',
                            fontWeight: '500',
                            color: isSelected ? '#1890ff' : 'rgba(0, 0, 0, 0.8)'
                        }}>
                            {formattedValue} {suffix}
                        </span>
                        {hasFluctuation && (
                            <div style={getBadgeStyle(latestEntry)}>
                                {getFluctuationText(latestEntry)}
                            </div>
                        )}
                    </div>
                    
                    <Tooltip title={displayName} placement="bottomLeft">
                        <div style={{ 
                            fontSize: '13px', 
                            color: 'rgba(0, 0, 0, 0.6)', 
                            lineHeight: '1.2',
                            textOverflow: 'ellipsis',
                            overflow: 'hidden',
                            display: '-webkit-box',
                            WebkitBoxOrient: 'vertical',
                            WebkitLineClamp: 2,
                            height: '31px',
                            wordBreak: 'break-word'
                        }}>
                            {displayName}
                        </div>
                    </Tooltip>
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
                        renderMetricItem(nodeIp, category, metric, values)
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

    // Helper functions for fluctuation display
    const getFluctuationStyle = (metric) => ({
        backgroundColor: metric.has_fluctuation ? '#fffbe6' : 'inherit',
        borderLeft: metric.has_fluctuation ? '3px solid #ffd666' : 'none',
        boxShadow: metric.has_fluctuation ? '0 2px 8px rgba(255, 214, 102, 0.1)' : 'none',
        padding: '12px',
        marginBottom: '8px',
        borderRadius: '4px',
        transition: 'all 0.3s ease'
    });

    const getBadgeStyle = (metric) => ({
        color: metric.pct_change > 0 ? '#389e0d' : '#cf1322',
        marginLeft: 8,
        fontWeight: 600,
        fontSize: '0.9em',
        padding: '2px 6px',
        borderRadius: '3px',
        backgroundColor: metric.pct_change > 0 ? 'rgba(56, 158, 13, 0.1)' : 'rgba(207, 19, 34, 0.1)'
    });

    const getFluctuationText = (metric) => {
        // For constant rate changes in large values, show the absolute change
        if (metric.pct_change === 0 && metric.value > 1e6 && Math.abs(metric.z_score) < 2.0) {
            // Don't show any fluctuation text for monotonically increasing counters with normal z-scores
            return '';
        }
        
        // If pct_change is very small but z-score is significant, display as <1% instead of 0%
        let pct = Math.abs(Math.round(metric.pct_change * 100));
        if (pct === 0 && Math.abs(metric.z_score) >= 2.0) {
            pct = '<1';
        }
        const direction = metric.pct_change > 0 ? '↑' : '↓';
        const z = metric.z_score.toFixed(1);
        return `${direction}${pct}% (z=${z})`;
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