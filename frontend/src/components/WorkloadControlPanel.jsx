import React, { useState, useEffect } from 'react';
import { Card, Button, Space, Table, message, Row, Col, Progress, Select } from 'antd';
import { workloadService } from '../services/workloadService';

const WorkloadControlPanel = () => {
    const [activeWorkloads, setActiveWorkloads] = useState([]);
    const [loading, setLoading] = useState(false);
    const [selectedWorkload, setSelectedWorkload] = useState(null);
    const [systemMetrics, setSystemMetrics] = useState({
        cpu_usage: 0,
        memory_usage: 0,
        disk_usage: 0
    });

    const workloadOptions = [
        { value: 'sysbench', label: 'Sysbench OLTP(not found on idmg-monitor)' },
        { value: 'tpcc', label: 'TPC-C' },
        // { value: 'tpch', label: 'TPC-H(not implemented)' }
    ];

    const columns = [
        {
            title: 'ID',
            dataIndex: 'id',
            key: 'id',
        },
        {
            title: 'Type',
            dataIndex: 'type',
            key: 'type',
        },
        {
            title: 'Start Time',
            dataIndex: 'startTime',
            key: 'startTime',
        },
        {
            title: 'Status',
            dataIndex: 'status',
            key: 'status',
        },
        {
            title: 'Action',
            key: 'action',
            render: (_, record) => (
                <Button
                    type="primary"
                    danger
                    onClick={() => handleStopWorkload(record.id)}
                    loading={loading}
                >
                    Stop
                </Button>
            ),
        },
    ];

    const handleStartWorkload = async (workloadType, threads = 4) => {
        if (!workloadType) {
            message.warning('Please select a workload type');
            return;
        }
        try {
            setLoading(true);
            await workloadService.startWorkload(workloadType, threads);
            message.success(`Started ${workloadType} workload`);
            await fetchActiveWorkloads();
        } catch (err) {
            message.error(err.message || 'Failed to start workload');
        } finally {
            setLoading(false);
        }
    };

    const handlePrepareDatabase = async (workloadType) => {
        if (!workloadType) {
            message.warning('Please select a workload type');
            return;
        }
        try {
            setLoading(true);
            await workloadService.prepareDatabase(workloadType);
            message.success('Database prepared successfully');
        } catch (err) {
            message.error(err.message || 'Failed to prepare database');
        } finally {
            setLoading(false);
        }
    };

    const handleStopWorkload = async (workloadId) => {
        try {
            setLoading(true);
            await workloadService.stopWorkload(workloadId);
            message.success('Workload stopped successfully');
            await fetchActiveWorkloads();
        } catch (err) {
            message.error(err.message || 'Failed to stop workload');
        } finally {
            setLoading(false);
        }
    };

    const handleStopAllWorkloads = async () => {
        try {
            setLoading(true);
            await workloadService.stopAllWorkloads();
            message.success('All workloads stopped');
            await fetchActiveWorkloads();
        } catch (err) {
            message.error(err.message || 'Failed to stop all workloads');
        } finally {
            setLoading(false);
        }
    };

    const fetchActiveWorkloads = async () => {
        try {
            const data = await workloadService.getActiveWorkloads();
            setActiveWorkloads(data.workloads || []);
            setSystemMetrics(data.systemMetrics || {
                cpu_usage: 0,
                memory_usage: 0,
                disk_usage: 0
            });
        } catch (err) {
            message.error(err.message || 'Failed to fetch active workloads');
        }
    };

    useEffect(() => {
        fetchActiveWorkloads();
        const intervalId = setInterval(fetchActiveWorkloads, 5000);
        return () => clearInterval(intervalId);
    }, []);

    return (
        <Card title="Workload Control" style={{ marginBottom: 16 }}>
            <Space direction="vertical" style={{ width: '100%' }}>
                <Row gutter={[16, 16]} align="middle">
                    <Col span={12}>
                        <Select
                            placeholder="Select a workload"
                            style={{ width: '100%' }}
                            options={workloadOptions}
                            value={selectedWorkload}
                            onChange={setSelectedWorkload}
                        />
                    </Col>
                    <Col span={6}>
                        <Button
                            type="primary"
                            onClick={() => handleStartWorkload(selectedWorkload)}
                            loading={loading}
                            style={{ width: '100%' }}
                        >
                            Start Workload
                        </Button>
                    </Col>
                    <Col span={6}>
                        <Button
                            onClick={() => handlePrepareDatabase(selectedWorkload)}
                            loading={loading}
                            style={{ width: '100%' }}
                        >
                            Prepare Database
                        </Button>
                    </Col>
                </Row>

                {activeWorkloads.length > 0 && (
                    <Card title="System Metrics" size="small" style={{ marginTop: 16 }}>
                        <Row gutter={16}>
                            <Col span={8}>
                                <div>CPU Usage</div>
                                <Progress percent={Math.round(systemMetrics.cpu_usage)} status="active" />
                            </Col>
                            <Col span={8}>
                                <div>Memory Usage</div>
                                <Progress percent={Math.round(systemMetrics.memory_usage)} status="active" />
                            </Col>
                            <Col span={8}>
                                <div>Disk Usage</div>
                                <Progress percent={Math.round(systemMetrics.disk_usage)} status="active" />
                            </Col>
                        </Row>
                    </Card>
                )}
                {activeWorkloads.length > 0 && (
                    <>
                        <Button
                            type="primary"
                            danger
                            onClick={handleStopAllWorkloads}
                            loading={loading}
                            style={{ marginTop: 16 }}
                        >
                            Stop All Workloads
                        </Button>
                        <Table
                            columns={columns}
                            dataSource={activeWorkloads}
                            rowKey="id"
                            size="small"
                            style={{ marginTop: 16 }}
                        />
                    </>
                )}
            </Space>
        </Card>
    );
};

export default WorkloadControlPanel; 