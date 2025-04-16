import React, { useEffect } from "react";
import {
    Card,
    Space,
    Typography,
    Button,
    Descriptions,
    Tag,
    Spin,
    Input,
    Form,
    Alert,
} from "antd";
import { PlayCircleOutlined } from "@ant-design/icons";

const { Title, Text } = Typography;

const ExecutionSummary = ({
    workloadConfig,
    anomalyConfig,
    onExecute,
    isExecuting,
    taskName,
    onTaskNameChange,
}) => {
    const getWorkloadTypeName = (type) => {
        switch (type) {
            case "sysbench":
                return "Sysbench OLTP";
            case "tpcc":
                return "TPC-C";
            case "tpch":
                return "TPC-H";
            default:
                return type;
        }
    };

    const getAnomalyTypeName = (type) => {
        switch (type) {
            case "cpu_stress":
                return "CPU Stress";
            case "network_bottleneck":
                return "Network Bottleneck";
            case "cache_bottleneck":
                return "Cache Bottleneck";
            case "too_many_indexes":
                return "Too Many Indexes";
            default:
                return type;
        }
    };
    
    // Updated default task name generation
    useEffect(() => {
        // Generate name only if workloadConfig is present
        if (workloadConfig && workloadConfig.type) {
            const workloadName = getWorkloadTypeName(workloadConfig.type);
            let defaultName = `${workloadName}`;

            if (anomalyConfig?.anomalies?.length > 0) {
                const anomalyTypes = anomalyConfig.anomalies.map(a =>
                    getAnomalyTypeName(a.type).replace(/\s+/g, '')
                );
                defaultName += ` with ${anomalyTypes.join('+')}`;
            } else {
                defaultName += " Normal Run"; // Changed default suffix
            }

            defaultName += ` (${new Date().toISOString().split('T')[0]})`;

            // Set the default name only if taskName is currently empty
            if (!taskName) {
                onTaskNameChange(defaultName);
            }
        }
        // Trigger only when workload or anomaly config changes, not taskName
    }, [workloadConfig, anomalyConfig, onTaskNameChange]);

    // Format option label for better readability
    const formatOptionLabel = (key) => {
        // Convert camelCase to Title Case with spaces
        return key
            .replace(/([A-Z])/g, " $1")
            .replace(/^./, (str) => str.toUpperCase())
            .trim();
    };

    // Format option value for display
    const formatOptionValue = (key, value) => {
        // Handle special cases
        if (key === 'reportInterval') {
            return `${value} seconds`;
        } else if (key === 'warmupTime') {
            return `${value} seconds`;
        } else if (key === 'runningTime') {
            return `${value} minutes`;
        } else if (key === 'randType') {
            return value.charAt(0).toUpperCase() + value.slice(1);
        } else if (typeof value === 'boolean') {
            return value ? 'Yes' : 'No';
        } else {
            return value;
        }
    };

    return (
        <Space direction="vertical" style={{ width: "100%" }} size="large">
            <Card>
                <Title level={4}>Configuration Summary</Title>
                <Space direction="vertical" style={{ width: "100%" }} size="large">
                    <Form layout="vertical">
                        <Form.Item
                            label="Task Name"
                            required
                            tooltip="Provide a name for this execution task"
                        >
                            <Input
                                placeholder="Enter a name for this task"
                                value={taskName}
                                onChange={(e) => onTaskNameChange(e.target.value)}
                            />
                        </Form.Item>
                    </Form>

                    {/* Always display workload config if available */}
                    {workloadConfig ? (
                        <Descriptions title="Workload Configuration" bordered>
                            <Descriptions.Item label="Type">
                                <Tag color="blue">
                                    {getWorkloadTypeName(workloadConfig?.type)}
                                </Tag>
                            </Descriptions.Item>
                            <Descriptions.Item label="Threads">
                                {workloadConfig?.threads}
                            </Descriptions.Item>
                            
                            {/* Advanced Options Section */}
                            {workloadConfig?.options && (
                                <Descriptions.Item label="Advanced Options" span={3}>
                                    <Descriptions size="small" bordered column={1}>
                                        {Object.entries(workloadConfig.options)
                                            .filter(([key]) => key !== 'node')
                                            .map(([key, value]) => (
                                                <Descriptions.Item
                                                    key={key}
                                                    label={formatOptionLabel(key)}
                                                >
                                                    {formatOptionValue(key, value)}
                                                </Descriptions.Item>
                                            ))}
                                    </Descriptions>
                                </Descriptions.Item>
                            )}
                        </Descriptions>
                    ) : (
                        <Text type="secondary">Workload not yet configured.</Text>
                    )}

                    <Descriptions title="Anomaly Configuration" bordered>
                        {anomalyConfig?.anomalies?.length > 0 ? (
                            <>
                                <Descriptions.Item label="Anomaly Types" span={3}>
                                    {anomalyConfig.anomalies.map((a) => (
                                        <Tag color="red" key={a.id}>
                                            {getAnomalyTypeName(a.type)} on {a.node.join(", ")}
                                        </Tag>
                                    ))}
                                </Descriptions.Item>
                            </>
                        ) : (
                            <Descriptions.Item label="Anomalies" span={3}>
                                <Text type="secondary">None</Text>
                            </Descriptions.Item>
                        )}
                    </Descriptions>
                </Space>
            </Card>

            <Card>
                <Space direction="vertical" style={{ width: "100%" }} size="middle">
                    <Text type="secondary">
                        Review your configuration above. Click Execute Task button in the main control panel area to start.
                    </Text>
                </Space>
            </Card>
        </Space>
    );
};

export default ExecutionSummary;
