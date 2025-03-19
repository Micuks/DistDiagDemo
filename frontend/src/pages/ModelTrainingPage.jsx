import React, { useState, useEffect } from 'react';
import { Card, Spin, Typography, Row, Col, Alert, Steps, Space, Button, Divider } from 'antd';
import { useNavigate } from 'react-router-dom';
import ModelTrainingPanel from '../components/ModelTrainingPanel';
import { useAnomalyData } from '../hooks/useAnomalyData';
import { DatabaseOutlined, ControlOutlined, LineChartOutlined, ExperimentOutlined, ArrowRightOutlined, ArrowLeftOutlined } from '@ant-design/icons';

const { Title, Text, Paragraph } = Typography;

const ModelTrainingPage = () => {
    const navigate = useNavigate();
    // Don't destructure refetch unless you need to call it explicitly
    const { isLoading: isLoadingAnomalies, data: activeAnomalies = [] } = useAnomalyData();
    const [initialLoading, setInitialLoading] = useState(true);

    useEffect(() => {
        // Set a timeout to avoid flickering for very fast loads
        const timer = setTimeout(() => {
            setInitialLoading(false);
        }, 500);
        
        return () => clearTimeout(timer);
    }, []);

    const workflowSteps = [
        {
            title: 'Setup',
            description: 'Configure workloads and anomalies',
            path: '/control',
            icon: <ControlOutlined />
        },
        {
            title: 'Train',
            description: 'Collect data and train models',
            path: '/training',
            icon: <DatabaseOutlined />
        },
        {
            title: 'Monitor',
            description: 'View system metrics',
            path: '/metrics',
            icon: <LineChartOutlined />
        },
        {
            title: 'Analyze',
            description: 'Review RCA results',
            path: '/ranks',
            icon: <ExperimentOutlined />
        }
    ];

    const renderPageHeader = () => (
        <Row gutter={[16, 16]} align="middle" style={{ marginBottom: 16 }}>
            <Col>
                <DatabaseOutlined style={{ fontSize: 32, marginRight: 8 }} />
            </Col>
            <Col>
                <Title level={2} style={{ margin: 0 }}>Model Training Management</Title>
                <Typography.Text type="secondary">
                    Collect training data and train anomaly detection models
                </Typography.Text>
            </Col>
        </Row>
    );

    const renderWorkflowGuide = () => (
        <Card style={{ marginBottom: 24 }}>
            <Title level={4}>Model Training Workflow</Title>
            <Paragraph>
                This is step 2 of the DBPecker workflow. Here you can collect training data for normal operation and anomalies, 
                then train machine learning models to detect and diagnose anomalies.
            </Paragraph>
            
            <Steps
                current={1}
                items={workflowSteps.map(step => ({
                    title: step.title,
                    description: step.description,
                    icon: step.icon
                }))}
                style={{ marginBottom: 16, marginTop: 16 }}
            />
            
            <Alert
                message="Workflow Guidance"
                description={
                    <Space direction="vertical">
                        <Text>
                            To get the most out of the training process:
                        </Text>
                        <ol>
                            <li>Start by creating a workload in Control Panel (with or without anomalies)</li>
                            <li>Use the Data Collection tab to collect training samples</li>
                            <li>Collect a balanced set of normal and anomaly data</li>
                            <li>Train your model once you have sufficient data</li>
                            <li>Once trained, view model performance and use it for anomaly detection</li>
                        </ol>
                    </Space>
                }
                type="info"
                showIcon
            />
            
            <Divider />
            
            <Row justify="space-between">
                <Col>
                    <Button 
                        type="primary" 
                        icon={<ArrowLeftOutlined />}
                        onClick={() => navigate('/control')}
                    >
                        Back to Control Panel
                    </Button>
                </Col>
                <Col>
                    <Button 
                        type="default"
                        onClick={() => navigate('/metrics')}
                    >
                        View Metrics <ArrowRightOutlined />
                    </Button>
                </Col>
            </Row>
        </Card>
    );

    // Only show loading if it's the initial loading state
    const showLoading = initialLoading && isLoadingAnomalies;

    const renderActiveAnomalyAlert = () => {
        if (activeAnomalies && activeAnomalies.length > 0) {
            return (
                <Alert
                    message="Active Anomalies Detected"
                    description={
                        <Text>
                            There are currently {activeAnomalies.length} active anomalies in the system. 
                            This is a good opportunity to collect anomaly training data.
                        </Text>
                    }
                    type="warning"
                    showIcon
                    style={{ marginBottom: 16 }}
                />
            );
        }
        return null;
    };

    return (
        <Card bordered={false}>
            {renderPageHeader()}
            
            {renderWorkflowGuide()}
            
            {renderActiveAnomalyAlert()}
            
            {showLoading ? (
                <div style={{ padding: 100, textAlign: 'center' }}>
                    <Spin size="large" tip="Loading training environment..." />
                    <div style={{ marginTop: 16 }}>
                        <Typography.Text type="secondary">
                            Preparing machine learning environment and loading data...
                        </Typography.Text>
                    </div>
                </div>
            ) : (
                <ModelTrainingPanel />
            )}
        </Card>
    );
};

export default ModelTrainingPage; 