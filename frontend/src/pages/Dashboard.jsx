import React from 'react';
import { Card, Typography, Row, Col, Steps, Button, Space, Statistic, Divider } from 'antd';
import { useNavigate } from 'react-router-dom';
import { 
    ControlOutlined, 
    ApiOutlined, 
    LineChartOutlined, 
    ExperimentOutlined,
    RocketOutlined, 
    DatabaseOutlined,
    BarChartOutlined,
    BranchesOutlined,
    ThunderboltOutlined
} from '@ant-design/icons';

const { Title, Text, Paragraph } = Typography;

const Dashboard = () => {
    const navigate = useNavigate();

    const workflowSteps = [
        {
            title: 'Setup',
            description: 'Configure workloads and anomalies',
            path: '/control',
            icon: <ControlOutlined />,
            details: 'Define your workload parameters and configure anomalies to inject'
        },
        {
            title: 'Train',
            description: 'Collect data and train models',
            path: '/training',
            icon: <ApiOutlined />,
            details: 'Collect training data and build machine learning models'
        },
        {
            title: 'Monitor',
            description: 'View system metrics',
            path: '/metrics',
            icon: <LineChartOutlined />,
            details: 'Monitor system metrics in real-time to observe performance'
        },
        {
            title: 'Analyze',
            description: 'Review RCA results',
            path: '/ranks',
            icon: <ExperimentOutlined />,
            details: 'Analyze root causes of performance issues with different models'
        }
    ];

    return (
        <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
            <Card bordered={false}>
                <Row gutter={[24, 24]} align="middle">
                    <Col xs={24} md={8}>
                        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
                            <RocketOutlined style={{ fontSize: '64px', color: '#1890ff', marginBottom: '16px' }} />
                            <Title level={2}>DBPecker</Title>
                            <Text type="secondary">Distributed Database Diagnosis Platform</Text>
                        </div>
                    </Col>
                    <Col xs={24} md={16}>
                        <Paragraph>
                            DBPecker helps you diagnose performance issues in distributed database systems by 
                            simulating workloads, injecting anomalies, collecting metrics, and analyzing root causes 
                            through various machine learning models.
                        </Paragraph>
                        <Paragraph>
                            Follow the workflow below to get started, or jump directly to any section if you're 
                            already familiar with the platform.
                        </Paragraph>
                    </Col>
                </Row>
            </Card>

            <Divider orientation="left">
                <Space>
                    <BranchesOutlined />
                    <span>Guided Workflow</span>
                </Space>
            </Divider>

            <Card variant={false} style={{ marginBottom: '24px' }}>
                <Row justify="center">
                    <Col xs={24} md={20} lg={18}>
                        <Steps
                            direction="vertical"
                            current={-1}
                            items={workflowSteps.map((step, index) => ({
                                title: step.title,
                                description: (
                                    <Space direction="vertical" size="small" style={{ width: '100%' }}>
                                        <Text>{step.description}</Text>
                                        <Text type="secondary">{step.details}</Text>
                                        <Button 
                                            type="primary" 
                                            icon={step.icon}
                                            onClick={() => navigate(step.path)}
                                            style={{ marginTop: '8px' }}
                                        >
                                            Go to {step.title}
                                        </Button>
                                    </Space>
                                ),
                                icon: step.icon
                            }))}
                        />
                    </Col>
                </Row>
            </Card>

            <Divider orientation="left">
                <Space>
                    <ThunderboltOutlined />
                    <span>Quick Actions</span>
                </Space>
            </Divider>

            <Row gutter={[16, 16]}>
                <Col xs={24} sm={12} md={6}>
                    <Card 
                        hoverable
                        onClick={() => navigate('/control')}
                        style={{ textAlign: 'center', height: '160px' }}
                    >
                        <ControlOutlined style={{ fontSize: '32px', color: '#1890ff', marginBottom: '8px' }} />
                        <Statistic title="Control Panel" value="Configure" />
                        <Text type="secondary">Set up workloads</Text>
                    </Card>
                </Col>
                <Col xs={24} sm={12} md={6}>
                    <Card 
                        hoverable
                        onClick={() => navigate('/training')}
                        style={{ textAlign: 'center', height: '160px' }}
                    >
                        <DatabaseOutlined style={{ fontSize: '32px', color: '#52c41a', marginBottom: '8px' }} />
                        <Statistic title="Data Collection" value="Train" />
                        <Text type="secondary">Gather training data</Text>
                    </Card>
                </Col>
                <Col xs={24} sm={12} md={6}>
                    <Card 
                        hoverable
                        onClick={() => navigate('/metrics')}
                        style={{ textAlign: 'center', height: '160px' }}
                    >
                        <BarChartOutlined style={{ fontSize: '32px', color: '#faad14', marginBottom: '8px' }} />
                        <Statistic title="System Metrics" value="Monitor" />
                        <Text type="secondary">View performance</Text>
                    </Card>
                </Col>
                <Col xs={24} sm={12} md={6}>
                    <Card 
                        hoverable
                        onClick={() => navigate('/ranks')}
                        style={{ textAlign: 'center', height: '160px' }}
                    >
                        <ExperimentOutlined style={{ fontSize: '32px', color: '#eb2f96', marginBottom: '8px' }} />
                        <Statistic title="Anomaly Ranks" value="Analyze" />
                        <Text type="secondary">Review diagnoses</Text>
                    </Card>
                </Col>
            </Row>
        </div>
    );
};

export default Dashboard; 