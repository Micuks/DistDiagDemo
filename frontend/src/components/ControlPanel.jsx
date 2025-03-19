import React, { useState } from 'react';
import { Steps, Card, Space, Typography, Button, message, Alert, Divider, Row, Col } from 'antd';
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

const ControlPanel = () => {
    const [currentStep, setCurrentStep] = useState(0);
    const [workloadConfig, setWorkloadConfig] = useState(null);
    const [anomalyConfig, setAnomalyConfig] = useState({ anomalies: [] });
    const [isExecuting, setIsExecuting] = useState(false);
    const [isExecuted, setIsExecuted] = useState(false);
    const [taskName, setTaskName] = useState('');
    const [tasksHistory, setTasksHistory] = useState([]);
    const { data: activeAnomalies = [], refetch: refetchAnomalies } = useAnomalyData();
    const [showDashboard, setShowDashboard] = useState(false);
    const navigate = useNavigate();

    const handleTaskNameChange = (name) => {
        setTaskName(name);
    };

    const handleExecute = async () => {
        if (!workloadConfig) {
            message.error('Please complete workload configuration');
            return;
        }

        if (!taskName.trim()) {
            message.error('Please provide a task name');
            return;
        }

        setIsExecuting(true);
        try {
            // First prepare the database if needed
            if (workloadConfig.prepareDatabase) {
                await workloadService.prepareDatabase(workloadConfig.type);
                message.success('Database prepared successfully');
            }

            // Create task object
            const task = {
                id: Date.now().toString(),
                name: taskName.trim(),
                workload: { ...workloadConfig },
                anomalies: anomalyConfig.anomalies || [],
                startTime: new Date().toISOString(),
                status: 'running'
            };

            // Start the workload
            const workloadResponse = await workloadService.startWorkload(
                workloadConfig.type,
                workloadConfig.threads,
                workloadConfig.options
            );
            
            if (workloadResponse && workloadResponse.id) {
                task.workloadId = workloadResponse.id;
            }
            
            message.success('Workload started successfully');

            // Start anomalies if configured
            if (anomalyConfig && anomalyConfig.anomalies && anomalyConfig.anomalies.length > 0) {
                for (const anomaly of anomalyConfig.anomalies) {
                    await anomalyService.startAnomaly(anomaly.type, anomaly.node);
                }
                message.success('Anomalies started successfully');
            } else {
                message.info("Running normal scenario without anomalies");
            }

            // Refresh anomaly data
            await refetchAnomalies();
            
            // Update tasks history
            setTasksHistory(prev => [...prev, task]);
            
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
    };

    const handleReset = () => {
        setIsExecuted(false);
        setCurrentStep(0);
        setWorkloadConfig(null);
        setAnomalyConfig({ anomalies: [] });
        setTaskName('');
        setShowDashboard(false);
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
                        description="Select the type of database workload you want to run, number of threads, and the target node(s)."
                        type="info"
                        showIcon
                        style={{ marginBottom: 16 }}
                    />
                    <WorkloadConfig
                        onConfigChange={setWorkloadConfig}
                        initialConfig={workloadConfig}
                    />
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
                        description="Select the types of anomalies you want to inject into the system. This is optional - you can run a normal workload without anomalies."
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
                        workloadConfig={workloadConfig}
                        anomalyConfig={anomalyConfig}
                        onExecute={handleExecute}
                        isExecuting={isExecuting}
                        taskName={taskName}
                        onTaskNameChange={handleTaskNameChange}
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
                    tasksHistory={tasksHistory}
                    setTasksHistory={setTasksHistory}
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
                            Previous
                        </Button>
                    )}
                    {currentStep < steps.length - 1 && (
                        <Button
                            type="primary"
                            onClick={() => setCurrentStep(currentStep + 1)}
                            disabled={!workloadConfig && currentStep === 0}
                        >
                            Next
                        </Button>
                    )}
                </Space>
            </Space>
        </Card>
    );
};

export default ControlPanel;