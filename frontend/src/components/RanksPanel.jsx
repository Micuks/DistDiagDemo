import React, { useState, useEffect } from 'react';
import { Card, Table, Spin } from 'antd';
import { anomalyService } from '../services/anomalyService';

const RanksPanel = () => {
    const [ranks, setRanks] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const data = await anomalyService.getAnomalyRanks();
                setRanks(data);
            } catch (error) {
                console.error('Error fetching anomaly ranks:', error);
            } finally {
                setLoading(false);
            }
        };

        // Fetch initially and then every 30 seconds
        fetchData();
        const interval = setInterval(fetchData, 30000);

        return () => clearInterval(interval);
    }, []);

    const columns = [
        {
            title: 'Timestamp',
            dataIndex: 'timestamp',
            key: 'timestamp',
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
        },
        {
            title: 'Score',
            dataIndex: 'score',
            key: 'score',
            render: (score) => score.toFixed(3),
        },
    ];

    if (loading) {
        return <Spin size="large" />;
    }

    return (
        <Card title="Anomaly Ranks">
            <Table
                dataSource={ranks}
                columns={columns}
                rowKey={(record) => `${record.timestamp}-${record.node}-${record.type}`}
            />
        </Card>
    );
};

export default RanksPanel; 