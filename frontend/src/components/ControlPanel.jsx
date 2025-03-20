import React, { useState, useEffect } from 'react';
import { Steps, Card, Space, Typography, Button, message, Alert, Divider, Row, Col, Form, Input, Select } from 'antd';
import { ControlOutlined, ApiOutlined, LineChartOutlined, ExperimentOutlined, ArrowRightOutlined } from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import { useAnomalyData } from '../hooks/useAnomalyData';
import { workloadService } from '../services/workloadService';
import { anomalyService } from '../services/anomalyService';
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
    const [isExecuting, setIsExecuting] = useState(false);
    const [isExecuted, setIsExecuted] = useState(false);
    const [taskName, setTaskName] = useState('');
    const { data: activeAnomalies = [], refetch: refetchAnomalies } = useAnomalyData();
    const [showDashboard, setShowDashboard] = useState(false);
    const [activeWorkloads, setActiveWorkloads] = useState([]);
    const [skipWorkload, setSkipWorkload] = useState(false);
    const navigate = useNavigate();
    const [loading, setLoading] = useState(false);

    // Check for active workloads when component mounts
    useEffect(() => {
        localStorage.removeItem('onExecutionDashboard');
        const checkActiveTasks = async () => {
            try {
                // Use the new task API to check for active tasks
                const activeTasks = await workloadService.getActiveTasks();
                console.log("Active tasks: ", activeTasks);
                const activeWorkloads = await workloadService.getActiveWorkloads();
                console.log("Active workloads: ", activeWorkloads);
                
                // Set active workloads
                setActiveWorkloads(activeWorkloads);
                
                // If there are active tasks, show the dashboard
                if (activeTasks.length > 0) {
                    setShowDashboard(true);
                }
            } catch (error) {
                console.error('Failed to check active tasks:', error);
            }
        };
        
        checkActiveTasks();
    }, []);

    const handleTaskNameChange = (name) => {
        setTaskName(name);
    };

    const handleExecute = async () => {
        // Check if we have anomalies configured
        const hasAnomalies = anomalyConfig && 
                            anomalyConfig.anomalies && 
                            anomalyConfig.anomalies.length > 0;
        
        // Check if we're running in anomaly-only mode (with existing workloads)
        const isAnomalyOnlyMode = skipWorkload && activeWorkloads.length > 0;

        // In anomaly-only mode, we don't need workload config
        if (!isAnomalyOnlyMode && !workloadConfig) {
            message.error('Please complete workload configuration');
            return;
        }

        // Always require a task name
        if (!taskName.trim()) {
            message.error('Please provide a task name');
            return;
        }

        // In normal mode (not anomaly-only), check if same type workload is running
        if (!isAnomalyOnlyMode && activeWorkloads.length > 0) {
            const sameTypeWorkloads = activeWorkloads.filter(
                workload => workload.type === workloadConfig.type && workload.status === 'running'
            );
            
            if (sameTypeWorkloads.length > 0) {
                // If a workload of same type is running, ask if user wants to just add anomalies
                const activeWorkloadId = sameTypeWorkloads[0].id;
                message.info(`A ${workloadConfig.type} workload is already running. Will add anomalies to the existing workload.`);
                setSkipWorkload(true);
            }
        }

        // If no anomalies are configured in anomaly-only mode, there's nothing to do
        if (isAnomalyOnlyMode && !hasAnomalies) {
            message.error('Please add at least one anomaly when adding to an existing workload');
            return;
        }

        setIsExecuting(true);
        try {
            let workloadResponse = null;
            
            // Start the workload if we're not in anomaly-only mode
            if (!isAnomalyOnlyMode) {
                // First prepare the database if needed
                if (workloadConfig.prepareDatabase) {
                    await workloadService.prepareDatabase(workloadConfig.type);
                    message.success('Database prepared successfully');
                }

                // Start the workload
                workloadResponse = await workloadService.startWorkload({
                    type: workloadConfig.type,
                    threads: workloadConfig.threads,
                    options: workloadConfig.options,
                    task_name: taskName.trim()
                });
                
                message.success('Workload started successfully');
            } else {
                // In anomaly-only mode, create a task with only anomalies
                // Use the selected workload's ID
                const activeWorkloadId = activeWorkloads[0].id;
                const activeWorkloadType = activeWorkloads[0].type;
                
                // Create a task without starting a new workload
                workloadResponse = await workloadService.createTask({
                    type: activeWorkloadType,
                    task_name: taskName.trim(),
                    workload_id: activeWorkloadId,
                    anomalies: anomalyConfig.anomalies
                });
                
                message.success('Task created successfully with existing workload');
            }

            // Start anomalies if configured
            if (hasAnomalies) {
                for (const anomaly of anomalyConfig.anomalies) {
                    await anomalyService.startAnomaly(anomaly.type, anomaly.node);
                }
                message.success('Anomalies started successfully');
            } else {
                message.info("Running normal scenario without anomalies");
            }

            // Refresh anomaly data
            await refetchAnomalies();
            
            // Update active workloads
            const updatedWorkloads = await workloadService.getActiveWorkloads();
            setActiveWorkloads(updatedWorkloads);
            
            // Mark as executed to show dashboard
            setIsExecuted(true);
            setShowDashboard(true);
        } catch (error) {
            message.error(`Failed to execute configuration: ${error.message}`);
        } finally {
            setIsExecuting(false);
        }
    };

    const handleNewExecution = () => {
        setCurrentStep(0);
        setWorkloadConfig(null);
        setAnomalyConfig({ anomalies: [] });
        setTaskName('');
        setShowDashboard(false);
        setSkipWorkload(false);
    };

    const handleReset = () => {
        setIsExecuted(false);
        setCurrentStep(0);
        setWorkloadConfig(null);
        setAnomalyConfig({ anomalies: [] });
        setTaskName('');
        setShowDashboard(false);
        setSkipWorkload(false);
    };

    const toggleWorkloadSkip = () => {
        setSkipWorkload(!skipWorkload);
    };

    const renderNextStep = () => {
        return (
            <Card style={{ marginTop: '24px' }}>
                <Title level={5}>What's Next?</Title>
                <Paragraph>
                    After setting up your workload and anomalies, you have several options:
                </Paragraph>
                <Row gutter={[16, 16]}>
                    <Col span={8}>
                        <Button 
                            type="primary" 
                            icon={<ApiOutlined />}
                            onClick={() => navigate('/training')}
                            block
                        >
                            Collect Training Data
                        </Button>
                        <Text type="secondary" style={{ display: 'block', marginTop: '8px' }}>
                            Gather data to train machine learning models for anomaly detection
                        </Text>
                    </Col>
                    <Col span={8}>
                        <Button 
                            icon={<LineChartOutlined />}
                            onClick={() => navigate('/metrics')}
                            block
                        >
                            View System Metrics
                        </Button>
                        <Text type="secondary" style={{ display: 'block', marginTop: '8px' }}>
                            Monitor real-time system performance metrics
                        </Text>
                    </Col>
                    <Col span={8}>
                        <Button 
                            icon={<ExperimentOutlined />}
                            onClick={() => navigate('/ranks')}
                            block
                        >
                            View RCA Results
                        </Button>
                        <Text type="secondary" style={{ display: 'block', marginTop: '8px' }}>
                            Review root cause analysis from different models
                        </Text>
                    </Col>
                </Row>
            </Card>
        );
    };

    const steps = [
        {
            title: 'Configure Workload',
            description: 'Select and configure database workload',
            content: (
                <>
                    <Alert
                        message="Step 1: Configure your workload"
                        description={
                            activeWorkloads.length > 0 ? (
                                <Space direction="vertical">
                                    <Text>There are active workloads running. You can either configure a new workload or add anomalies to an existing workload.</Text>
                                    <Button 
                                        type={skipWorkload ? "primary" : "default"}
                                        onClick={toggleWorkloadSkip}
                                    >
                                        {skipWorkload ? "Configure New Workload" : "Use Existing Workload"}
                                    </Button>
                                </Space>
                            ) : (
                                "Select the type of database workload you want to run, number of threads, and the target node(s)."
                            )
                        }
                        type="info"
                        showIcon
                        style={{ marginBottom: 16 }}
                    />
                    {!skipWorkload && (
                        <WorkloadConfig
                            onConfigChange={setWorkloadConfig}
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
                                            <Text strong>{workload.type.toUpperCase()}</Text> - Threads: {workload.threads}
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
                    <Alert
                        message="Step 2: Configure anomalies (Optional)"
                        description={
                            skipWorkload ? 
                            "Select the types of anomalies you want to inject into the existing workload." :
                            "Select the types of anomalies you want to inject into the system. This is optional - you can run a normal workload without anomalies."
                        }
                        type="info"
                        showIcon
                        style={{ marginBottom: 16 }}
                    />
                    <AnomalyConfig
                        onConfigChange={setAnomalyConfig}
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
                    <Alert
                        message="Step 3: Review and execute"
                        description="Review your configuration settings and provide a name for this task. Click 'Execute' to start the workload."
                        type="info"
                        showIcon
                        style={{ marginBottom: 16 }}
                    />
                    <ExecutionSummary
                        workloadConfig={skipWorkload ? null : workloadConfig}
                        anomalyConfig={anomalyConfig}
                        onExecute={handleExecute}
                        isExecuting={isExecuting}
                        taskName={taskName}
                        onTaskNameChange={handleTaskNameChange}
                        skipWorkload={skipWorkload}
                        activeWorkloads={activeWorkloads}
                    />
                </>
            ),
            icon: <ControlOutlined />
        },
    ];

    if (showDashboard) {
        return (
            <>
                <ExecutionDashboard 
                    workloadConfig={workloadConfig}
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
            <Space direction="vertical" style={{ width: '100%' }} size="large">
                <Row gutter={[16, 16]} align="middle">
                    <Col>
                        <ControlOutlined style={{ fontSize: 32, marginRight: 8 }} />
                    </Col>
                    <Col>
                        <Title level={3} style={{ margin: 0 }}>Control Panel</Title>
                        <Text type="secondary">
                            Configure and execute workload and anomaly scenarios
                        </Text>
                    </Col>
                </Row>

                <Divider />

                <Alert
                    message="Workflow Guide"
                    description={
                        <Text>
                            First you'll configure a workload and optional anomalies, then execute them to either collect 
                            training data, monitor system metrics, or analyze root causes with different models.
                        </Text>
                    }
                    type="success"
                    showIcon
                    style={{ marginBottom: 16 }}
                />

                <Steps
                    current={currentStep}
                    items={steps.map((step, index) => ({
                        title: step.title,
                        description: step.description,
                        icon: step.icon
                    }))}
                    onChange={setCurrentStep}
                    style={{ marginBottom: 24 }}
                />

                <Card>
                    {steps[currentStep].content}
                </Card>

                <Space style={{ marginTop: 24, justifyContent: 'flex-end' }}>
                    {currentStep > 0 && (
                        <Button onClick={() => setCurrentStep(currentStep - 1)}>
                            Previous Step
                        </Button>
                    )}
                    {currentStep < steps.length - 1 && (
                        <Button
                            type="primary"
                            onClick={() => setCurrentStep(currentStep + 1)}
                            disabled={!workloadConfig && currentStep === 0 && !skipWorkload}
                        >
                            {currentStep === 0 ? "Configure Anomalies →" : "Review Configuration →"}
                        </Button>
                    )}
                </Space>
            </Space>
        </Card>
    );
};

export default ControlPanel;