import React, { useState } from 'react';
import { Card, Typography, Button, Select, Space, Divider, Alert, message, notification } from 'antd';
import { DatabaseOutlined, LockOutlined, ReloadOutlined } from '@ant-design/icons';
import { workloadService } from '../services/workloadService';

const { Title, Text, Paragraph } = Typography;
const { Option } = Select;

const AdminPanel = () => {
    const [selectedWorkloadType, setSelectedWorkloadType] = useState('sysbench');
    const [isLoading, setIsLoading] = useState(false);
    const [operationResult, setOperationResult] = useState(null);

    const handlePrepareDatabase = async() => {
        setIsLoading(true);
        setOperationResult(null);

        try {
            await workloadService.prepareDatabase(selectedWorkloadType);
            message.success(`Database prepared successfully for ${selectedWorkloadType.toUpperCase()}`);
            setOperationResult({
                type: 'success',
                message: `Database for ${selectedWorkloadType.toUpperCase()} prepared successfully.`
            });
        } catch (error) {
            console.error('Failed to prepare database:', error);
            notification.error({
                message: 'Database Preparation Failed',
                description: error.message || 'An unknown error occurred',
                duration: 0
            });
            setOperationResult({
                type: 'error',
                message: `Failed to prepare database: ${error.message || 'Unknown error'}`
            });
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div style={{ maxWidth: '1000px', margin: '0 auto' }}>
            <Card>
                <div style={{ display: 'flex', alignItems: 'center', marginBottom: '20px' }}>
                    <LockOutlined style={{ fontSize: '24px', marginRight: '12px', color: '#1890ff' }} />
                    <Title level={2} style={{ margin: 0 }}>Admin Panel</Title>
                </div>

                <Alert message="Administrator Access"
                    description="This panel contains administrative functions that should only be used by system administrators."
                    type="warning"
                    showIcon style={{ marginBottom: '24px' }}
                />

                <Divider orientation="left">Database Preparation</Divider>

                <Card title={<div><DatabaseOutlined /> Database Preparation</div>}
                    style={{ marginBottom: '20px' }}>
                    <Paragraph>
                        Prepare the database
                        for testing workloads. This operation creates the necessary
                        tables and indexes required
                        for running the selected workload type.
                    </Paragraph>
                    <Paragraph type="warning">
                        <strong>Note:</strong> Database preparation is a resource-intensive operation and should 
                        only be performed when necessary. This operation will connect to the database configured in the server settings.
                    </Paragraph>

                    <Space direction="vertical"
                        style={{ width: '100%' }}>
                        <div style={{ marginBottom: '16px' }}>
                            <Text strong>Workload Type:</Text>
                            <Select
                                value={selectedWorkloadType}
                                onChange={setSelectedWorkloadType}
                                style={{ width: '200px', marginLeft: '12px' }}
                                disabled={isLoading}>
                                <Option value="sysbench">Sysbench OLTP</Option>
                                <Option value="tpcc">TPC-C</Option>
                                {/* <Option value="tpch">TPC-H</Option> */}
                            </Select>
                        </div>

                        <Button type="primary"
                            icon={<ReloadOutlined />}
                            loading={isLoading}
                            onClick={handlePrepareDatabase}>
                            Prepare {selectedWorkloadType.toUpperCase()}
                            Database
                        </Button>

                        {operationResult && (
                            <Alert message={operationResult.type === 'success' ? "Success" : "Error"}
                                description={operationResult.message}
                                type={operationResult.type}
                                showIcon style={{ marginTop: '16px' }}
                            />
                        )}
                    </Space>
                </Card>

                <Divider orientation="left">System Information</Divider>

                <Card title={<div><DatabaseOutlined /> Database Connection</div>}>
                    <Paragraph>
                        The system is configured to connect to the database using the parameters
                        defined in the server environment.
                    </Paragraph>
                    <ul>
                        <li><Text strong>Host:</Text> Configured in OB_HOST environment variable</li>
                        <li><Text strong>Port:</Text> Configured in OB_PORT environment variable</li>
                        <li><Text strong>User:</Text> Configured in OB_USER environment variable</li>
                        <li><Text strong>Database:</Text> Configured in OB_NAME environment variable</li>
                    </ul>
                </Card>
            </Card>
        </div>
    );
};

export default AdminPanel;