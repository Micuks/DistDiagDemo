import React, { useState, useEffect } from 'react';
import { Card, Table, Space, Tag, Typography } from 'antd';
import { anomalyService } from '../services/anomalyService';

const { Text } = Typography;

const RanksPanel = () => {
    const [loading, setLoading] = useState(false);
    const [anomalyRanks, setAnomalyRanks] = useState([]);

    const columns = [
        {
            title: 'Timestamp',
            dataIndex: 'timestamp',
            key: 'timestamp',
            render: (timestamp) => new Date(timestamp).toLocaleString(),
        },
        {
            title: 'Node',
            dataIndex: 'node',
            key: 'node',
        },
        {
            title: 'Type',
            dataIndex: 'type',
            key: 'type',
            render: (type) => {
                let color = 'default';
                switch (type) {
                    case 'cpu':
                        color = 'red';
                        break;
                    case 'memory':
                        color = 'blue';
                        break;
                    case 'io':
                        color = 'orange';
                        break;
                    case 'network':
                        color = 'green';
                        break;
                    default:
                        color = 'default';
                }
                return <Tag color={color}>{type.toUpperCase()}</Tag>;
            },
        },
        {
            title: 'Confidence',
            dataIndex: 'score',
            key: 'score',
            render: (score) => {
                const percent = Math.round(score * 100);
                let color = '#52c41a';  // Green for high confidence
                if (percent < 50) {
                    color = '#ff4d4f';  // Red for low confidence
                } else if (percent < 75) {
                    color = '#faad14';  // Yellow for medium confidence
                }
                return <Text style={{ color }}>{percent}%</Text>;
            },
            sorter: (a, b) => a.score - b.score,
            defaultSortOrder: 'descend',
        },
    ];

    const fetchAnomalyRanks = async () => {
        try {
            setLoading(true);
            const ranks = await anomalyService.getAnomalyRanks();
            setAnomalyRanks(ranks);
        } catch (error) {
            console.error('Error fetching anomaly ranks:', error);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchAnomalyRanks();
        const interval = setInterval(fetchAnomalyRanks, 5000);
        return () => clearInterval(interval);
    }, []);

    return (
        <Card title="Anomaly Detection Results">
            <Table
                columns={columns}
                dataSource={anomalyRanks}
                rowKey={(record) => `${record.node}-${record.timestamp}-${record.type}`}
                loading={loading}
                pagination={{ pageSize: 10 }}
            />
        </Card>
    );
};

export default RanksPanel; 