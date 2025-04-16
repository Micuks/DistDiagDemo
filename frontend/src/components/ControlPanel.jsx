import React, { useState, useEffect } from 'react';
import { Steps, Card, Space, Typography, Button, message, Alert, Divider, Row, Col, Form, Input, Select, Table, Tabs, Tag } from 'antd';
import { ControlOutlined, ApiOutlined, LineChartOutlined, ExperimentOutlined, ArrowRightOutlined, HistoryOutlined, PlayCircleOutlined } from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import { useTask } from '../hooks/useTask';
import WorkloadConfig from './WorkloadConfig';
import AnomalyConfig from './AnomalyConfig';
import ExecutionSummary from './ExecutionSummary';
import ExecutionDashboard from './ExecutionDashboard';

const { Title, Text, Paragraph } = Typography;
const { Option } = Select;

const ControlPanel = () => {
    const [currentStep, setCurrentStep] = useState(0);
    const [workloadConfig, setWorkloadConfig] = useState(null);
    const [anomalyConfig, setAnomalyConfig] = useState({ anomalies: [] });
    const [isExecuted, setIsExecuted] = useState(false);
    const [taskName, setTaskName] = useState('');
    const [showDashboard, setShowDashboard] = useState(false);
    const [activeTabKey, setActiveTabKey] = useState('1');
    const [skipWorkload, setSkipWorkload] = useState(false);
    const navigate = useNavigate();

    const { tasks, createTask, loading: taskLoading, error: taskError, fetchTasks } = useTask();

    const activeWorkloads = tasks.filter(task => ['running', 'pending', 'stopping'].includes(task.status));

    const toggleWorkloadSkip = () => {
        setSkipWorkload(!skipWorkload);
        if (!skipWorkload) {
            setWorkloadConfig(null);
        }
    };

    useEffect(() => {
        localStorage.removeItem('onExecutionDashboard');
        const hasActiveTasks = tasks.some(task => ['running', 'pending', 'stopping'].includes(task.status));
        setShowDashboard(hasActiveTasks);
    }, [tasks]);

    const handleTaskNameChange = (name) => {
        setTaskName(name);
    };

    const handleExecute = async () => {
        if (!workloadConfig || !workloadConfig.type || !workloadConfig.threads) {
            message.error('Please complete workload configuration (Type and Threads required)');
            setCurrentStep(0);
            return;
        }
        if (!taskName.trim()) {
            message.error('Please provide a task name');
            setCurrentStep(2);
            return;
        }

        const taskCreateData = {
            name: taskName.trim(),
            workload_type: workloadConfig.type,
            workload_config: {
                 num_threads: workloadConfig.threads,
                 ...(workloadConfig.options || {})
            },
            anomalies: anomalyConfig.anomalies || [],
        };

        console.log("Creating task with data:", taskCreateData);

        try {
            const createdTask = await createTask(taskCreateData);
            if (createdTask) {
                setIsExecuted(true);
                setShowDashboard(true);
                setActiveTabKey('1');
            } else {
                 console.error("Task creation returned undefined/null or failed silently.");
            }
        } catch (error) {
            console.error("Error during task creation caught in ControlPanel:", error);
        }
    };

    const handleNewExecution = () => {
        setCurrentStep(0);
        setWorkloadConfig(null);
        setAnomalyConfig({ anomalies: [] });
        setTaskName('');
        setShowDashboard(false);
        setActiveTabKey('1');
        setIsExecuted(false);
    };

    const handleReset = () => {
        setIsExecuted(false);
        setCurrentStep(0);
        setWorkloadConfig(null);
        setAnomalyConfig({ anomalies: [] });
        setTaskName('');
        setShowDashboard(false);
        fetchTasks();
    };

    const formatAnomalyType = (type) => {
        if (!type) return '';
        return type
            .split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    };

    const historyColumns = [
        {
            title: 'Task Name',
            dataIndex: 'name',
            key: 'name',
            ellipsis: true,
        },
        {
            title: 'Workload',
            dataIndex: 'workload_type',
            key: 'workload_type',
            render: (type, record) => `${type.toUpperCase()} (${record.workload_config?.num_threads || 'N/A'} threads)`,
        },
        {
            title: 'Anomalies',
            dataIndex: 'anomalies',
            key: 'anomalies',
            render: (anomalies) => {
                if (!anomalies || anomalies.length === 0) return 'None';
                return (
                    <Space direction="vertical" size="small">
                        {anomalies.map((anomaly, index) => (
                            <Tag key={index} color="red">
                                {formatAnomalyType(anomaly.type)}
                                {anomaly.target ? ` on ${anomaly.target}` : ''}
                                {anomaly.severity ? ` (${anomaly.severity})` : ''}
                            </Tag>
                        ))}
                    </Space>
                );
            },
            ellipsis: true,
        },
        {
            title: 'Start Time',
            dataIndex: 'start_time',
            key: 'start_time',
            render: (time) => time ? new Date(time).toLocaleString() : '-',
            sorter: (a, b) => new Date(a.start_time) - new Date(b.start_time),
            defaultSortOrder: 'descend',
        },
        {
            title: 'End Time',
            dataIndex: 'end_time',
            key: 'end_time',
            render: (time) => time ? new Date(time).toLocaleString() : '-',
        },
        {
            title: 'Status',
            dataIndex: 'status',
            key: 'status',
            filters: [
                { text: 'Stopped', value: 'stopped' },
                { text: 'Error', value: 'error' },
            ],
            onFilter: (value, record) => record.status === value,
            render: (status) => {
                let color = 'default';
                if (status === 'stopped') color = 'success';
                else if (status === 'error') color = 'error';
                else if (status === 'running') color = 'processing';
                else if (status === 'pending') color = 'warning';
                else if (status === 'stopping') color = 'warning';

                return <Tag color={color}>{status ? status.toUpperCase() : 'UNKNOWN'}</Tag>;
            },
        },
        {
             title: 'Error Message',
             dataIndex: 'error_message',
             key: 'error_message',
             ellipsis: true,
             render: (msg) => msg || '-'
        }
    ];

    const renderNextStep = () => {
        return (
            <Card style={{ marginTop: '24px' }}>
                <Title level={5}>What's Next?</Title>
                <Paragraph>
                    After setting up your workload and anomalies, you have several options:
                </Paragraph>
                <Row gutter={[16, 16]}>
                    <Col span={8}>
                        <Button type="primary"
                            icon={<ApiOutlined />}
                            onClick={() => navigate('/training')}
                            block>
                            Collect Training Data
                        </Button>
                        <Text type="secondary"
                            style={{ display: 'block', marginTop: '8px' }}>
                            Gather data to train machine learning models
                            for anomaly detection
                        </Text>
                    </Col>
                    <Col span={8}>
                        <Button icon={<LineChartOutlined />}
                            onClick={() => navigate('/metrics')}
                            block>
                            View System Metrics
                        </Button>
                        <Text type="secondary"
                            style={{ display: 'block', marginTop: '8px' }}>
                            Monitor real-time system performance metrics
                        </Text>
                    </Col>
                    <Col span={8}>
                        <Button icon={<ExperimentOutlined />}
                            onClick={() => navigate('/ranks')}
                            block>
                            View RCA Results
                        </Button>
                        <Text type="secondary"
                            style={{ display: 'block', marginTop: '8px' }}>
                            Review root cause analysis from different models
                        </Text>
                    </Col>
                </Row>
            </Card>
        );
    };

    const renderContent = () => {
        if (activeTabKey === '2') {
            const historyTasks = tasks.filter(task => ['stopped', 'error'].includes(task.status));
            return (
                <Card>
                    <Space direction="vertical" style={{ width: '100%' }} size="large">
                        <Title level={4}>Task History</Title>
                        <Button onClick={fetchTasks} loading={taskLoading} style={{marginBottom: 16}}>Refresh History</Button>
                        <Table
                            dataSource={historyTasks}
                            columns={historyColumns}
                            rowKey="id"
                            loading={taskLoading}
                            pagination={{ pageSize: 10, showSizeChanger: true }}
                            scroll={{ x: 'max-content' }}
                        />
                    </Space>
                </Card>
            );
        }

        return (
            <>
                <Steps current={currentStep}
                    items={
                        steps.map((step, index) => ({
                            title: step.title,
                            description: step.description,
                            icon: step.icon
                        }))
                    }
                    style={{ marginBottom: 24 }}
                />

                <Card>{steps[currentStep].content}</Card>

                <Space style={{ marginTop: 24, justifyContent: 'space-between', width: '100%' }}>
                    <span>
                        {currentStep > 0 && (
                            <Button onClick={() => setCurrentStep(currentStep - 1)}>
                                Previous Step
                            </Button>
                        )}
                    </span>
                    <span>
                        {currentStep < steps.length - 1 && (
                            <Button type="primary"
                                onClick={() => setCurrentStep(currentStep + 1)}
                                disabled={currentStep === 0 && (!workloadConfig || !workloadConfig.type)}
                            >
                                {currentStep === 0 ? "Configure Anomalies →" : "Review Configuration →"}
                            </Button>
                        )}
                        {currentStep === steps.length - 1 && (
                             <Button
                                type="primary"
                                icon={<PlayCircleOutlined />}
                                onClick={handleExecute}
                                loading={taskLoading}
                                disabled={!workloadConfig || !taskName}
                                size="large"
                             >
                                Execute Task
                             </Button>
                        )}
                    </span>
                </Space>
            </>
        );
    };

    const steps = [{
            title: 'Configure Workload',
            description: 'Select and configure database workload',
            content: (
                <>
                    <Alert message="Step 1: Configure your workload"
                        description={
                            activeWorkloads.length > 0 ? (
                                <Space direction="vertical">
                                    <Text>There {activeWorkloads.length > 1 ? 'are' : 'is'} active workload{activeWorkloads.length > 1 ? 's' : ''} running. You can either configure a new workload or add anomalies to {activeWorkloads.length > 1 ? 'an' : 'the'} existing workload.</Text>
                                    <Button type={skipWorkload ? "primary" : "default"}
                                        onClick={toggleWorkloadSkip}>
                                        {skipWorkload ? "Configure New Workload" : "Use Existing Workload"}
                                    </Button>
                                </Space>
                            ) : (
                                "Select the type of database workload you want to run, number of threads."
                            )
                        }
                        type="info"
                        showIcon style={{ marginBottom: 16 }}
                    /> 
                    {!skipWorkload && (
                        <WorkloadConfig onConfigChange={setWorkloadConfig}
                            initialConfig={workloadConfig}
                        />
                    )}
                    {skipWorkload && (
                        <Card>
                            <Title level={4}>Using Existing Workload</Title>
                            <div>
                                <Text>You'll be adding anomalies to one of these active workloads:</Text>
                                <ul>
                                    {activeWorkloads.map(workload => (
                                        <li key={workload.id}>
                                            <Text strong>{workload.workload_type?.toUpperCase()}</Text> - Threads: {workload.workload_config?.num_threads || 'N/A'}
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        </Card>
                    )}
                </>
            ),
            icon: <ControlOutlined />
        },
        {
            title: 'Configure Anomalies',
            description: 'Select and configure anomalies to inject (optional)',
            content: (
                <>
                    <Alert message="Step 2: Configure anomalies (Optional)"
                        description={
                            skipWorkload ?
                            "Select the types of anomalies you want to inject into the existing workload." : "Select the types of anomalies you want to inject into the system. This is optional - you can run a normal workload without anomalies."
                        }
                        type="info"
                        showIcon style={{ marginBottom: 16 }}
                    />
                    <AnomalyConfig onConfigChange={setAnomalyConfig}
                        initialConfig={anomalyConfig}
                    />
                </>
            ),
            icon: <ControlOutlined />
        },
        {
            title: 'Review & Execute',
            description: 'Review configuration and start execution',
            content: (
                <>
                    <Alert message="Step 3: Review configuration" description="Review your configuration settings and provide a name for this task. Click 'Execute Task' button below to start." type="info" showIcon style={{ marginBottom: 16 }} />
                    <ExecutionSummary
                        workloadConfig={workloadConfig}
                        anomalyConfig={anomalyConfig}
                        taskName={taskName}
                        onTaskNameChange={handleTaskNameChange}
                    />
                </>
            ),
            icon: <ArrowRightOutlined />
        },
    ];

    if (showDashboard) {
        return (
            <>
                <ExecutionDashboard workloadConfig={workloadConfig}
                    anomalyConfig={anomalyConfig}
                    onReset={handleReset}
                    onNewExecution={handleNewExecution}
                />
                {renderNextStep()}
            </>
        );
    }

    return (
        <Card>
            <Space direction="vertical"
                style={{ width: '100%' }}
                size="large">
                <Row gutter={[16, 16]}
                    align="middle">
                    <Col>
                        <ControlOutlined style={{ fontSize: 32, marginRight: 8 }} />
                    </Col>
                    <Col>
                        <Title level={3}
                            style={{ margin: 0 }}>Control Panel</Title>
                        <Text type="secondary">
                            Configure and execute workload and anomaly scenarios
                        </Text>
                    </Col>
                </Row>

                <Divider />

                <Alert message="Workflow Guide"
                    description={
                        <Text>
                            Use the steps below to configure a Task (Workload + Optional Anomalies), then review and execute.
                            Use the Task History tab to view past executions.
                        </Text>
                    }
                    type="success"
                    showIcon style={{ marginBottom: 16 }}
                />
                
                <Tabs 
                    activeKey={activeTabKey} 
                    onChange={(key) => setActiveTabKey(key)}
                    items={[
                        {
                            key: '1',
                            label: 'Configure Task'
                        },
                        {
                            key: '2',
                            label: <span><HistoryOutlined /> Task History</span>
                        }
                    ]}
                />

                {renderContent()}
            </Space>
        </Card>
    );
};

export default ControlPanel;