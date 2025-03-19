import React, { useState, useEffect } from 'react';
import { Layout, Menu, Steps, Modal, Button, Typography, Tooltip, Drawer, Badge } from 'antd';
import { DashboardOutlined, ExperimentOutlined, LineChartOutlined, 
         ApiOutlined, ControlOutlined, QuestionCircleOutlined,
         RocketOutlined, ArrowRightOutlined } from '@ant-design/icons';
import { BrowserRouter as Router, useNavigate, useLocation } from 'react-router-dom';
import AppRoutes from './routes';

const { Header, Content, Sider } = Layout;
const { Text, Title } = Typography;

const Navigation = () => {
    const navigate = useNavigate();
    const location = useLocation();
    const [workflowModalVisible, setWorkflowModalVisible] = useState(false);
    const [siderCollapsed, setSiderCollapsed] = useState(false);
    const [currentStep, setCurrentStep] = useState(0);
    
    // Check if it's the user's first visit
    useEffect(() => {
        const firstVisit = localStorage.getItem('firstVisit') !== 'false';
        if (firstVisit) {
            setWorkflowModalVisible(true);
            localStorage.setItem('firstVisit', 'false');
        }
        
        // Determine current step based on path
        const pathToStepMap = {
            '/control': 0,
            '/training': 1,
            '/metrics': 2,
            '/ranks': 3
        };
        
        setCurrentStep(pathToStepMap[location.pathname] || 0);
    }, [location.pathname]);

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
            icon: <ApiOutlined />
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

    const menuItems = [
        {
            key: '/control',
            icon: <ControlOutlined />,
            label: (
                <Badge dot={currentStep === 0} offset={[5, 0]}>
                    <Tooltip title="Step 1: Configure workloads and anomalies">
                        Control Panel
                    </Tooltip>
                </Badge>
            )
        },
        {
            key: '/training',
            icon: <ApiOutlined />,
            label: (
                <Badge dot={currentStep === 1} offset={[5, 0]}>
                    <Tooltip title="Step 2: Collect data and train models">
                        Model Training
                    </Tooltip>
                </Badge>
            )
        },
        {
            key: '/metrics',
            icon: <LineChartOutlined />,
            label: (
                <Badge dot={currentStep === 2} offset={[5, 0]}>
                    <Tooltip title="Step 3: View system metrics">
                        System Metrics
                    </Tooltip>
                </Badge>
            )
        },
        {
            key: '/ranks',
            icon: <ExperimentOutlined />,
            label: (
                <Badge dot={currentStep === 3} offset={[5, 0]}>
                    <Tooltip title="Step 4: View root cause analysis">
                        Anomaly Ranks
                    </Tooltip>
                </Badge>
            )
        },
        {
            key: 'workflow',
            icon: <QuestionCircleOutlined />,
            label: (
                <Tooltip title="View workflow guide">
                    Workflow Guide
                </Tooltip>
            )
        }
    ];
    
    const handleMenuClick = ({ key }) => {
        if (key === 'workflow') {
            setWorkflowModalVisible(true);
        } else {
            navigate(key);
        }
    };
    
    const getNextStep = () => {
        const nextStepIndex = (currentStep + 1) % workflowSteps.length;
        return workflowSteps[nextStepIndex];
    };
    
    const nextStep = getNextStep();

    return (
        <>
            <Menu
                theme="dark"
                mode="horizontal"
                selectedKeys={[location.pathname]}
                items={menuItems}
                onClick={handleMenuClick}
            />
            
            <Modal
                title={
                    <div style={{ display: 'flex', alignItems: 'center' }}>
                        <RocketOutlined style={{ fontSize: '24px', marginRight: '10px' }} />
                        <span>DBPecker Workflow Guide</span>
                    </div>
                }
                open={workflowModalVisible}
                onCancel={() => setWorkflowModalVisible(false)}
                footer={[
                    <Button key="close" onClick={() => setWorkflowModalVisible(false)}>
                        Close
                    </Button>,
                    <Button 
                        key="start" 
                        type="primary" 
                        onClick={() => {
                            setWorkflowModalVisible(false);
                            navigate('/control');
                        }}
                    >
                        Start Workflow
                    </Button>
                ]}
                width={700}
            >
                <Steps
                    current={currentStep}
                    items={workflowSteps.map(step => ({
                        title: step.title,
                        description: step.description,
                        icon: step.icon
                    }))}
                />
                
                <div style={{ margin: '20px 0' }}>
                    <Title level={4}>How to use DBPecker:</Title>
                    <ol style={{ fontSize: '14px', lineHeight: '1.8' }}>
                        <li><strong>Control Panel:</strong> Configure your workload and anomaly scenarios</li>
                        <li><strong>Model Training:</strong> Collect training data and train anomaly detection models</li>
                        <li><strong>System Metrics:</strong> View real-time system metrics to monitor performance</li>
                        <li><strong>Anomaly Ranks:</strong> Review root cause analysis from different models</li>
                    </ol>
                    
                    <Text type="secondary">
                        Follow this workflow for the best experience or click on any section to jump directly to it.
                    </Text>
                </div>
            </Modal>
            
            {location.pathname !== '/dashboard' && (
                <div style={{ 
                    position: 'fixed', 
                    bottom: '20px', 
                    right: '20px', 
                    zIndex: 1000,
                    background: '#1890ff',
                    padding: '10px 15px',
                    borderRadius: '4px',
                    boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    color: 'white'
                }} onClick={() => navigate(nextStep.path)}>
                    <Text style={{ color: 'white', marginRight: '10px' }}>
                        Next: {nextStep.title}
                    </Text>
                    <ArrowRightOutlined />
                </div>
            )}
        </>
    );
};

const App = () => {
    return (
        <Router>
            <Layout style={{ minHeight: '100vh' }}>
                <Header style={{ display: 'flex', alignItems: 'center' }}>
                    <div style={{ 
                        color: 'white', 
                        marginRight: '40px', 
                        fontSize: '18px', 
                        fontWeight: 'bold',
                        display: 'flex',
                        alignItems: 'center' 
                    }}>
                        <DashboardOutlined style={{ marginRight: '8px', fontSize: '20px' }} />
                        DBPecker
                    </div>
                    <Navigation />
                </Header>
                <Content style={{ padding: '24px', minHeight: 'calc(100vh - 64px)' }}>
                    <AppRoutes />
                </Content>
            </Layout>
        </Router>
    );
};

export default App; 