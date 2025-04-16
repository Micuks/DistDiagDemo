import React, { useState, useEffect } from "react";
import { Form, Select, Button, Card, List, Typography, Alert, Spin, Tooltip } from "antd";
import { PlusOutlined, DeleteOutlined } from "@ant-design/icons";
import { anomalyService } from "../services/anomalyService";

const { Option } = Select;
const { Title, Text } = Typography;

const AnomalyConfig = ({ onConfigChange, initialConfig }) => {
  const [form] = Form.useForm(); // Create form instance
  const [anomalies, setAnomalies] = useState(initialConfig?.anomalies || []);
  const [availableNodes, setAvailableNodes] = useState([]); // Initialize as empty array
  const [availableAnomalyTypes, setAvailableAnomalyTypes] = useState([]); // Will now store objects {type: string, description: string}
  const [loadingNodes, setLoadingNodes] = useState(false);
  const [loadingTypes, setLoadingTypes] = useState(false);

  useEffect(() => {
    const fetchNodes = async () => {
      setLoadingNodes(true);
      try {
        const nodes = await anomalyService.getAvailableNodes();
        setAvailableNodes(nodes || []); // Ensure it's an array even if API returns null/undefined
        console.log("Set nodes: ", availableNodes);
      } catch (error) {
        console.error("Failed to fetch nodes:", error);
        setAvailableNodes([]); // Set to empty array on error
      } finally {
        setLoadingNodes(false);
      }
    };

    const fetchAnomalyTypes = async () => {
      setLoadingTypes(true);
      try {
        const typesWithDescriptions = await anomalyService.getAvailableAnomalyTypes();
        setAvailableAnomalyTypes(typesWithDescriptions || []); // Ensure it's an array
        console.log("Set anomaly types: ", typesWithDescriptions);
      } catch (error) {
        console.error("Failed to fetch anomaly types:", error);
        // Fallback already handled in service
        setAvailableAnomalyTypes([]); 
      } finally {
        setLoadingTypes(false);
      }
    };

    fetchNodes();
    fetchAnomalyTypes();
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

  // Format anomaly type for display
  const formatAnomalyType = (type) => {
    if (!type) return ''; // Add check for undefined type
    const specialCases = {
      cpu: "CPU",
      io: "IO",
      net: "Network",
      chaosmesh: "ChaosMesh"
    };
    return type
      .split('_')
      .map(word => {
        const lowerWord = word.toLowerCase();
        return specialCases[lowerWord] || (lowerWord.charAt(0).toUpperCase() + lowerWord.slice(1));
      })
      .join(' ');
  };

 
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
            <Select 
              style={{ width: 300 }} 
              placeholder="Select anomaly type"
              loading={loadingTypes}
            >
              {availableAnomalyTypes.map(anomalyInfo => (
                <Option key={anomalyInfo.type} value={anomalyInfo.type}>
                  <Tooltip title={anomalyInfo.description} placement="right"> 
                    <span>{formatAnomalyType(anomalyInfo.type)}</span>
                  </Tooltip>
                </Option>
              ))}
            </Select>
          </Form.Item>

          <Form.Item
            name="node"
            rules={[{ required: true, message: "Select target node" }]}
          >
            <Select
              style={{ width: 300 }}
              placeholder="Select node"
              mode="multiple"
              loading={loadingNodes}
              placement="bottomLeft"
              maxTagCount={1}
              maxTagTextLength={15}
              dropdownMatchSelectWidth={false}
              listHeight={250}
              popupMatchSelectWidth={false}
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
              style={{ width: 300 }}
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
                  title={formatAnomalyType(anomaly.type)}
                  description={`Node: ${Array.isArray(anomaly.node) ? anomaly.node.join(', ') : anomaly.node}, Severity: ${anomaly.severity}`}
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
