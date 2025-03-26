import React, { useState, useEffect } from 'react';
import { Form, Select, InputNumber, Switch, Space, Card, Typography, Divider } from 'antd';
import { workloadService } from '../services/workloadService';

const { Title, Text } = Typography;
const { Option } = Select;

const WorkloadConfig = ({ onConfigChange, initialConfig }) => {
    const [form] = Form.useForm();
    const [workloadType, setWorkloadType] = useState(initialConfig?.type);
    const [availableNodes, setAvailableNodes] = useState([]);

    useEffect(() => {
        // Load available nodes
        const loadNodes = async () => {
            try {
                const nodes = await workloadService.getAvailableNodes();
                setAvailableNodes(nodes);
            } catch (error) {
                console.error('Failed to load nodes:', error);
            }
        };
        loadNodes();
    }, []);

    useEffect(() => {
        if (initialConfig) {
            form.setFieldsValue(initialConfig);
        }
    }, [initialConfig, form]);

    const handleValuesChange = (changedValues, allValues) => {
        if (changedValues.type) {
            setWorkloadType(changedValues.type);
        }
        onConfigChange(allValues);
    };

    const getWorkloadOptions = () => {
        switch (workloadType) {
            case 'sysbench':
                return (
                    <>
                        <Form.Item
                            label="Table Size"
                            name={['options', 'tableSize']}
                            initialValue={100000}
                        >
                            <InputNumber min={1000} max={1000000} />
                        </Form.Item>
                        <Form.Item
                            label="Number of Tables"
                            name={['options', 'tables']}
                            initialValue={10}
                        >
                            <InputNumber min={1} max={100} />
                        </Form.Item>
                        <Form.Item
                            label="Report Interval (seconds)"
                            name={['options', 'reportInterval']}
                            initialValue={10}
                        >
                            <InputNumber min={1} max={3600} />
                        </Form.Item>
                        <Form.Item
                            label="Random Type"
                            name={['options', 'randType']}
                            initialValue="uniform"
                        >
                            <Select>
                                <Option value="uniform">Uniform</Option>
                                <Option value="gaussian">Gaussian</Option>
                                <Option value="special">Special</Option>
                                <Option value="pareto">Pareto</Option>
                            </Select>
                        </Form.Item>
                    </>
                );
            case 'tpcc':
                return (
                    <>
                        <Form.Item
                            label="Number of Warehouses"
                            name={['options', 'warehouses']}
                            initialValue={10}
                        >
                            <InputNumber min={1} max={100} />
                        </Form.Item>
                        <Form.Item
                            label="Warmup Time (seconds)"
                            name={['options', 'warmupTime']}
                            initialValue={10}
                        >
                            <InputNumber min={0} max={3600} />
                        </Form.Item>
                        <Form.Item
                            label="Running Time (minutes)"
                            name={['options', 'runningTime']}
                            initialValue={60}
                        >
                            <InputNumber min={1} max={1440} />
                        </Form.Item>
                        <Form.Item
                            label="Report Interval (seconds)"
                            name={['options', 'reportInterval']}
                            initialValue={10}
                        >
                            <InputNumber min={1} max={3600} />
                        </Form.Item>
                    </>
                );
            default:
                return null;
        }
    };

    return (
        <Form
            form={form}
            layout="vertical"
            onValuesChange={handleValuesChange}
            initialValues={{
                type: workloadType,
                threads: 4,
                prepareDatabase: false,
                options: {}
            }}
        >
            <Space direction="vertical" style={{ width: '100%' }} size="large">
                <Card>
                    <Title level={4}>Basic Configuration</Title>
                    <Form.Item
                        label="Workload Type"
                        name="type"
                        rules={[{ required: true, message: 'Please select a workload type' }]}
                    >
                        <Select>
                            <Option value="sysbench">Sysbench OLTP</Option>
                            <Option value="tpcc">TPC-C</Option>
                            {/* <Option value="tpch">TPC-H</Option> */}
                        </Select>
                    </Form.Item>

                    <Form.Item
                        label="Number of Threads"
                        name="threads"
                        rules={[{ required: true, message: 'Please specify number of threads' }]}
                    >
                        <InputNumber min={1} max={64} />
                    </Form.Item>

                    <Form.Item
                        label="Target Node(s)"
                        name={['options', 'node']}
                        rules={[{ required: true, message: 'Please select at least one target node' }]}
                    >
                        <Select mode="multiple" placeholder="Select one or more nodes">
                            {availableNodes.map(node => (
                                <Option key={node} value={node}>{node}</Option>
                            ))}
                        </Select>
                    </Form.Item>

                    <Form.Item
                        label="Prepare Database"
                        help="* Database only need to be prepared once"
                        name="prepareDatabase"
                        valuePropName="checked"
                    >
                        <Switch />
                    </Form.Item>
                </Card>

                {workloadType && (
                    <Card>
                        <Title level={4}>Advanced Options</Title>
                        <Text type="secondary">
                            Configure specific options for {workloadType.toUpperCase()} workload
                        </Text>
                        <Divider />
                        {getWorkloadOptions()}
                    </Card>
                )}
            </Space>
        </Form>
    );
};

export default WorkloadConfig;