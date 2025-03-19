import React from 'react';
import { Card, Space, Typography, Button, Descriptions, Tag, Spin, Input, Form } from 'antd';
import { PlayCircleOutlined } from '@ant-design/icons';

const { Title, Text } = Typography;

const ExecutionSummary = ({ workloadConfig, anomalyConfig, onExecute, isExecuting, taskName, onTaskNameChange }) => {
    const getWorkloadTypeName = (type) => {
        switch (type) {
            case 'sysbench':
                return 'Sysbench OLTP';
            case 'tpcc':
                return 'TPC-C';
            case 'tpch':
                return 'TPC-H';
            default:
                return type;
        }
    };

    const getAnomalyTypeName = (type) => {
        switch (type) {
            case 'cpu_stress':
                return 'CPU Stress';
            case 'io_bottleneck':
                return 'I/O Bottleneck';
            case 'network_bottleneck':
                return 'Network Bottleneck';
            case 'cache_bottleneck':
                return 'Cache Bottleneck';
            case 'too_many_indexes':
                return 'Too Many Indexes';
            default:
                return type;
        }
    };

    return (
        <Space direction="vertical" style={{ width: '100%' }} size="large">
            <Card>
                <Title level={4}>Configuration Summary</Title>
                <Space direction="vertical" style={{ width: '100%' }} size="large">
                    <Form layout="vertical">
                        <Form.Item 
                            label="Task Name" 
                            required 
                            tooltip="Provide a name for this execution task"
                        >
                            <Input 
                                placeholder="Enter a name for this task" 
                                value={taskName} 
        defaultValue={"Untitled Task"}
                                onChange={(e) => onTaskNameChange(e.target.value)}
                            />
                        </Form.Item>
                    </Form>

                    <Descriptions title="Workload Configuration" bordered>
                        <Descriptions.Item label="Type">
                            <Tag color="blue">{getWorkloadTypeName(workloadConfig?.type)}</Tag>
                        </Descriptions.Item>
                        <Descriptions.Item label="Target Node">
                            <Tag color="green">{workloadConfig?.node?.join(', ')}</Tag>
                        </Descriptions.Item>
                        <Descriptions.Item label="Threads">
                            {workloadConfig?.threads}
                        </Descriptions.Item>
                        <Descriptions.Item label="Prepare Database">
                            {workloadConfig?.prepareDatabase ? 'Yes' : 'No'}
                        </Descriptions.Item>
                        {workloadConfig?.options && Object.entries(workloadConfig.options).map(([key, value]) => (
                            <Descriptions.Item key={key} label={key.replace(/([A-Z])/g, ' $1').trim()}>
                                {value}
                            </Descriptions.Item>
                        ))}
                    </Descriptions>

                    <Descriptions title="Anomaly Configuration" bordered>
                        {anomalyConfig?.anomalies?.length > 0 ? (
                            anomalyConfig.anomalies.map((anomaly, index) => (
                                <React.Fragment key={anomaly.id}>
                                    <Descriptions.Item label={`Anomaly ${index + 1} Type`}>
                                        <Tag color="red">{getAnomalyTypeName(anomaly.type)}</Tag>
                                    </Descriptions.Item>
                                    <Descriptions.Item label={`Anomaly ${index + 1} Node`}>
                                        <Tag color="orange">{anomaly.node.join(', ')}</Tag>
                                    </Descriptions.Item>
                                    <Descriptions.Item label={`Anomaly ${index + 1} Severity`}>
                                        <Tag color={
                                            anomaly.severity === 'high' ? 'red' :
                                            anomaly.severity === 'medium' ? 'orange' : 'green'
                                        }>
                                            {anomaly.severity.toUpperCase()}
                                        </Tag>
                                    </Descriptions.Item>
                                </React.Fragment>
                            ))
                        ) : (
                            <Descriptions.Item label="Scenario">
                                <Tag color="green">Normal Execution (No Anomalies)</Tag>
                            </Descriptions.Item>
                        )}
                    </Descriptions>
                </Space>
            </Card>

            <Card>
                <Space direction="vertical" style={{ width: '100%' }} size="middle">
                    <Text type="secondary">
                        Review your configuration above. Click Execute to start the task.
                    </Text>
                    <Button
                        type="primary"
                        icon={<PlayCircleOutlined />}
                        onClick={onExecute}
                        loading={isExecuting}
                        disabled={!workloadConfig || !taskName}
                    >
                        Execute Configuration
                    </Button>
                </Space>
            </Card>
        </Space>
    );
};

export default ExecutionSummary; 