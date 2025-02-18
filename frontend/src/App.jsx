import React from 'react';
import { Layout, Menu } from 'antd';
import { BrowserRouter as Router, useNavigate, useLocation } from 'react-router-dom';
import AppRoutes from './routes';

const { Header, Content } = Layout;

const Navigation = () => {
    const navigate = useNavigate();
    const location = useLocation();

    const items = [
        {
            key: '/metrics',
            label: 'System Metrics'
        },
        {
            key: '/ranks',
            label: 'Anomaly Ranks'
        },
        {
            key: '/control',
            label: 'Control Panel'
        },
        {
            key: '/training',
            label: 'Model Training'
        },
    ];

    return (
        <Menu
            theme="dark"
            mode="horizontal"
            selectedKeys={[location.pathname]}
            items={items}
            onClick={({ key }) => navigate(key)}
        />
    );
};

const App = () => {
    return (
        <Router>
            <Layout>
                <Header>
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