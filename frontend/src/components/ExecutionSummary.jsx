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
    skipWorkload = false,
    activeWorkloads = [],
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
    
    // Generate default task name based on workload type and anomalies
    useEffect(() => {
        if (workloadConfig || (skipWorkload && activeWorkloads.length > 0)) {
            // Get workload type - either from config or active workload
            const workloadType = workloadConfig?.type || activeWorkloads[0]?.type;
            const workloadName = getWorkloadTypeName(workloadType);
            
            let defaultName = `${workloadName}`;
            
            if (anomalyConfig?.anomalies?.length > 0) {
                const anomalyTypes = anomalyConfig.anomalies.map(a => 
                    getAnomalyTypeName(a.type).replace(/\s+/g, '')
                );
                defaultName += ` with ${anomalyTypes.join('+')}`;
            } else {
                defaultName += " Normal";
            }
            
            defaultName += ` (${new Date().toISOString().split('T')[0]})`;
            
            // In anomaly-only mode, add an "Anomaly" prefix
            if (skipWorkload) {
                defaultName = `Anomaly: ${defaultName}`;
            }
            
            onTaskNameChange(defaultName);
        }
    }, [workloadConfig, anomalyConfig, skipWorkload, activeWorkloads]);

    // Get active workload for displaying in anomaly-only mode
    const activeWorkload = skipWorkload && activeWorkloads.length > 0 ? activeWorkloads[0] : null;

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
                {skipWorkload && (
                    <Alert
                        message="Anomaly-Only Mode"
                        description="You are adding anomalies to an existing workload. No new workload will be started."
                        type="info"
                        showIcon
                        style={{ marginBottom: 16 }}
                    />
                )}
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

                    {/* Display workload config or active workload info */}
                    {skipWorkload ? (
                        <Descriptions title="Existing Workload" bordered>
                            <Descriptions.Item label="Type">
                                <Tag color="blue">
                                    {getWorkloadTypeName(activeWorkload?.type)}
                                </Tag>
                            </Descriptions.Item>
                            <Descriptions.Item label="Workload ID">
                                {activeWorkload?.id}
                            </Descriptions.Item>
                            <Descriptions.Item label="Threads">
                                {activeWorkload?.threads}
                            </Descriptions.Item>
                            {activeWorkload?.metrics && (
                                <>
                                    <Descriptions.Item label="CPU Usage">
                                        {activeWorkload.metrics.cpu_usage}%
                                    </Descriptions.Item>
                                    <Descriptions.Item label="Memory Usage">
                                        {activeWorkload.metrics.memory_usage}%
                                    </Descriptions.Item>
                                    {activeWorkload.metrics.tps && (
                                        <Descriptions.Item label="TPS">
                                            {activeWorkload.metrics.tps}
                                        </Descriptions.Item>
                                    )}
                                </>
                            )}
                        </Descriptions>
                    ) : (
                        workloadConfig && (
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
                        )
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
                        Review your configuration above. Click Execute to start the task.
                    </Text>
                    <Button
                        type="primary"
                        icon={<PlayCircleOutlined />}
                        onClick={onExecute}
                        loading={isExecuting}
                        disabled={(!workloadConfig && !skipWorkload) || !taskName || (skipWorkload && anomalyConfig?.anomalies?.length === 0)}
                        size="large"
                    >
                        {isExecuting ? 
                            "Starting Execution..." : 
                            skipWorkload ? "Inject Anomalies" : "Start Execution"
                        }
                    </Button>
                </Space>
            </Card>
        </Space>
    );
};

export default ExecutionSummary;
