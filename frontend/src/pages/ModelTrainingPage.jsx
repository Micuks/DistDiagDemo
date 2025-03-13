import React, { useState, useEffect } from 'react';
import { Card, Spin, Typography, Row, Col } from 'antd';
import ModelTrainingPanel from '../components/ModelTrainingPanel';
import { useAnomalyData } from '../hooks/useAnomalyData';
import { DatabaseOutlined } from '@ant-design/icons';

const { Title } = Typography;

const ModelTrainingPage = () => {
    // Don't destructure refetch unless you need to call it explicitly
    const { isLoading: isLoadingAnomalies } = useAnomalyData();
    const [initialLoading, setInitialLoading] = useState(true);

    useEffect(() => {
        // Set a timeout to avoid flickering for very fast loads
        const timer = setTimeout(() => {
            setInitialLoading(false);
        }, 500);
        
        return () => clearTimeout(timer);
    }, []);

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

    // Only show loading if it's the initial loading state
    const showLoading = initialLoading && isLoadingAnomalies;

    return (
        <Card bordered={false}>
            {renderPageHeader()}
            
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