import React, { useState, useEffect } from "react";
import { Form, Select, Button, Card, List, Typography, Alert } from "antd";
import { PlusOutlined, DeleteOutlined } from "@ant-design/icons";
import { anomalyService } from "../services/anomalyService";

const { Option } = Select;
const { Title, Text } = Typography;

const AnomalyConfig = ({ onConfigChange, initialConfig }) => {
  const [form] = Form.useForm(); // Create form instance
  const [anomalies, setAnomalies] = useState(initialConfig?.anomalies || []);
  const [availableNodes, setAvailableNodes] = useState([]); // Initialize as empty array
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchNodes = async () => {
      setLoading(true);
      try {
        const nodes = await anomalyService.getAvailableNodes();
        setAvailableNodes(nodes || []); // Ensure it's an array even if API returns null/undefined
        console.log("Set nodes: ", availableNodes);
      } catch (error) {
        console.error("Failed to fetch nodes:", error);
        setAvailableNodes([]); // Set to empty array on error
      } finally {
        setLoading(false);
      }
    };

    fetchNodes();
  }, []);

  useEffect(() => {
    if (initialConfig) {
      form.setFieldsValue(initialConfig);
      setAnomalies(initialConfig.anomalies || []);
    }
  }, [initialConfig, form]);

  const getSeverityOptions = (anomalyType) => {
    // Your severity options logic here
    return [
      { label: "Low", value: "low" },
      { label: "Medium", value: "medium" },
      { label: "High", value: "high" },
    ];
  };

  // In AnomalyConfig.jsx
  const handleAddAnomaly = async () => {
    try {
      const values = await form.validateFields();

      // Create new anomaly object with unique ID
      const newAnomaly = {
        id: Date.now().toString(), // Simple unique ID
        type: values.type,
        node: values.node,
        severity: values.severity,
      };

      // Update anomalies list
      const updatedAnomalies = [...anomalies, newAnomaly];
      setAnomalies(updatedAnomalies);

      // Notify parent component of the change
      if (onConfigChange) {
        onConfigChange({ anomalies: updatedAnomalies });
      }

      // Reset form after successful submission
      form.resetFields();
    } catch (error) {
      console.error("Validation failed:", error);
    }
  };

  const handleRemoveAnomaly = (id) => {
    const updatedAnomalies = anomalies.filter((anomaly) => anomaly.id !== id);
    setAnomalies(updatedAnomalies);

    // Notify parent component of the change
    if (onConfigChange) {
      onConfigChange({ anomalies: updatedAnomalies });
    }
  };

  return (
    <>
      <Alert
        message="Anomalies are optional"
        description="You can run a normal scenario without injecting any anomalies. Skip this step if you want to run a baseline test."
        type="info"
        showIcon
        style={{ marginBottom: 16 }}
      />
      
      <Card>
        <Title level={4}>Add Anomaly</Title>
        <Form
          form={form} // Connect the form instance here
          layout="inline"
          onFinish={handleAddAnomaly}
        >
          <Form.Item
            name="type"
            rules={[{ required: true, message: "Select anomaly type" }]}
          >
            <Select style={{ width: 200 }} placeholder="Select type">
              <Option value="cpu_stress">CPU Stress</Option>
              <Option value="io_bottleneck">I/O Bottleneck</Option>
              <Option value="network_bottleneck">Network Bottleneck</Option>
              <Option value="cache_bottleneck">Cache Bottleneck</Option>
              <Option value="too_many_indexes">Too Many Indexes</Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="node"
            rules={[{ required: true, message: "Select target node" }]}
          >
            <Select
              style={{ width: 200 }}
              placeholder="Select node"
              mode="multiple"
              loading={loading}
            >
              {availableNodes &&
                availableNodes.map((node) => (
                  <Option key={node} value={node}>
                    {node}
                  </Option>
                ))}
            </Select>
          </Form.Item>

          <Form.Item
            name="severity"
            rules={[{ required: true, message: "Select severity" }]}
          >
            <Select
              style={{ width: 200 }}
              placeholder="Select severity"
              options={getSeverityOptions(form.getFieldValue("type"))}
            />
          </Form.Item>

          <Form.Item>
            <Button
              type="primary"
              icon={<PlusOutlined />}
              onClick={handleAddAnomaly}
            >
              Add
            </Button>
          </Form.Item>
        </Form>
      </Card>

      <Card>
        <Title level={4}>Configured Anomalies</Title>
        {anomalies.length === 0 ? (
          <Text type="secondary">No anomalies configured. This will run as a normal scenario.</Text>
        ) : (
          <List
            dataSource={anomalies}
            renderItem={(anomaly) => (
              <List.Item
                actions={[
                  <Button
                    type="text"
                    danger
                    icon={<DeleteOutlined />}
                    onClick={() => handleRemoveAnomaly(anomaly.id)}
                  />,
                ]}
              >
                <List.Item.Meta
                  title={anomaly.type}
                  description={`Node: ${anomaly.node}, Severity: ${anomaly.severity}`}
                />
              </List.Item>
            )}
          />
        )}
      </Card>
    </>
  );
};

export default AnomalyConfig;
