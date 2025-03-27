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
            case "io_bottleneck":
                return "I/O Bottleneck";
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
                                <Descriptions.Item label="Prepare Database">
                                    {workloadConfig?.prepareDatabase ? "Yes" : "No"}
                                </Descriptions.Item>
                                {workloadConfig?.options &&
                                    Object.entries(workloadConfig.options)
                                        .filter(([key]) => key !== 'node')
                                        .map(([key, value]) => (
                                            <Descriptions.Item
                                                key={key}
                                                label={key.replace(/([A-Z])/g, " $1").trim()}
                                            >
                                                {value}
                                            </Descriptions.Item>
                                        ))}
                                <Descriptions.Item label="Target Node">
                                    <Tag color="green">{workloadConfig?.options?.node?.join(", ")}</Tag>
                                </Descriptions.Item>
                            </Descriptions>
                        )
                    )}

                    <Descriptions title="Anomaly Configuration" bordered>
                        {anomalyConfig?.anomalies?.length > 0 ? (
                            anomalyConfig.anomalies.map((anomaly, index) => (
                                <React.Fragment key={anomaly.id}>
                                    <Descriptions.Item
                                        label={`Anomaly ${index + 1} Type`}
                                    >
                                        <Tag color="red">
                                            {getAnomalyTypeName(anomaly.type)}
                                        </Tag>
                                    </Descriptions.Item>
                                    <Descriptions.Item
                                        label={`Anomaly ${index + 1} Node`}
                                    >
                                        <Tag color="orange">
                                            {anomaly.node.join(", ")}
                                        </Tag>
                                    </Descriptions.Item>
                                    <Descriptions.Item
                                        label={`Anomaly ${index + 1} Severity`}
                                    >
                                        <Tag
                                            color={
                                                anomaly.severity === "high"
                                                    ? "red"
                                                    : anomaly.severity === "medium"
                                                    ? "orange"
                                                    : "green"
                                            }
                                        >
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
