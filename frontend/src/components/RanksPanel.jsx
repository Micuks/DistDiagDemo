import React, { useState, useEffect } from "react";
import {
  Card,
  Table,
  Space,
  Tag,
  Typography,
  Select,
  Button,
  Alert,
  Spin,
  Row,
  Col,
  Divider,
  Progress,
  Tabs,
  Empty,
} from "antd";
import { anomalyService } from "../services/anomalyService";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from "recharts";

const { Text, Title } = Typography;
const { Option } = Select;
const { TabPane } = Tabs;

const RanksPanel = () => {
  const [loading, setLoading] = useState(false);
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModels, setSelectedModels] = useState([]);
  const [comparisonLoading, setComparisonLoading] = useState(false);
  const [comparisonData, setComparisonData] = useState(null);
  const [comparisonError, setComparisonError] = useState(null);
  const [nodeData, setNodeData] = useState([]);
  const [activeTab, setActiveTab] = useState("1");

  // Columns for the anomaly ranks table
  const columns = [
    {
      title: "Timestamp",
      dataIndex: "timestamp",
      key: "timestamp",
      render: (timestamp) => new Date(timestamp).toLocaleString(),
    },
    {
      title: "Node",
      dataIndex: "node",
      key: "node",
    },
    {
      title: "Type",
      dataIndex: "type",
      key: "type",
      render: (type) => {
        let color = "default";
        switch (type) {
          case "cpu":
            color = "red";
            break;
          case "io":
            color = "orange";
            break;
          case "network":
            color = "green";
            break;
          case "cache":
            color = "blue";
            break;
          case "too many indexes":
            color = "purple";
            break;
          default:
            color = "default";
        }
        return <Tag color={color}>{type.toUpperCase()}</Tag>;
      },
    },
    {
      title: "Confidence",
      dataIndex: "score",
      key: "score",
      render: (score) => {
        const percent = Math.round(score * 100);
        let color = "#52c41a";
        if (percent < 50) {
          color = "#ff4d4f";
        } else if (percent < 75) {
          color = "#faad14";
        }
        return <Text style={{ color }}>{percent}%</Text>;
      },
      sorter: (a, b) => a.score - b.score,
      defaultSortOrder: "descend",
    },
  ];

  // Columns for per-node RCA comparison table
  const initialRcaComparisonColumns = [
    {
      title: "Node",
      dataIndex: "node",
      key: "node",
      fixed: "left",
      width: 120,
    },
    {
      title: "Anomaly Type",
      dataIndex: "anomalyType",
      key: "anomalyType",
      width: 150,
      render: (type) => {
        let color = "default";
        const lowerType = type?.toLowerCase();
        switch (true) {
          case lowerType?.includes('cpu'):
            color = "red";
            break;
          case lowerType?.includes('io'):
            color = "orange";
            break;
          case lowerType?.includes('network'):
            color = "green";
            break;
          case lowerType?.includes('cache'):
            color = "blue";
            break;
          case lowerType?.includes('index'):
            color = "purple";
            break;
          default:
            color = "default";
        }
        return type ? (
          <Tag color={color}>{type.toUpperCase()}</Tag>
        ) : (
          <Text type="secondary">None</Text>
        );
      },
    },
  ];

  const fetchAvailableModels = async () => {
    try {
      const models = await anomalyService.getAvailableModels();
      // Filter out any models with null or undefined names
      const validModels = models.filter(model => model !== null && ((typeof model === 'string' && model !== '') || (typeof model === 'object' && model.name !== '')));
      setAvailableModels(validModels);

      // Default select the first model if available
      if (validModels.length > 0) {
        setSelectedModels([validModels[0]]);
      }
    } catch (error) {
      console.error("Error fetching available models:", error);
    }
  };

  const handleCompareModels = async () => {
    if (selectedModels.length === 0) {
      setComparisonError("Please select at least one model to analyze.");
      return;
    }

    try {
      setComparisonError(null);
      setComparisonLoading(true);
      const response = await anomalyService.compareModels(selectedModels);
      
      // Check for model-specific errors and show them as warnings
      const modelErrors = [];
      selectedModels.forEach(model => {
        if (response[model] && response[model].error) {
          modelErrors.push(`${model}: ${response[model].error}`);
        }
      });
      
      if (modelErrors.length > 0) {
        // Show model errors as a warning
        setComparisonError(`Some models had errors: ${modelErrors.join('; ')}`);
      }
      
      // Response contains the data directly
      setComparisonData(response);

      // Create RCA comparison columns dynamically based on selected models
      const newColumns = [...initialRcaComparisonColumns];

      // Add model-specific columns
      selectedModels.forEach((model, index) => {
        // Always add column even if model has errors - we'll show placeholder data
        const baseColor = `hsl(${(index * 360) / selectedModels.length}, 70%, 50%)`;
        const hasError = response[model] && response[model].error;

        // Root Cause column for this model
        newColumns.push({
          title: (
            <span>
              {model} Root Cause
              {hasError && <Tag color="red" style={{ marginLeft: 5 }}>Error</Tag>}
            </span>
          ),
          dataIndex: `${model}_root_cause`,
          key: `${model}_root_cause`,
          width: 150,
          render: (text) => (
            <Text style={{ 
              color: hasError ? '#999' : baseColor, 
              fontWeight: hasError ? 'normal' : 'bold' 
            }}>
              {text}
            </Text>
          ),
        });

        // Confidence column for this model
        newColumns.push({
          title: (
            <span>
              {model} Confidence
              {hasError && <Tag color="red" style={{ marginLeft: 5 }}>Error</Tag>}
            </span>
          ),
          dataIndex: `${model}_confidence`,
          key: `${model}_confidence`,
          width: 130,
          render: (score) => {
            if (hasError) return <Text type="secondary">N/A</Text>;
            if (!score && score !== 0) return <Text type="secondary">N/A</Text>;

            const percent = Math.round(score * 100);
            let color = "#52c41a";
            if (percent < 50) {
              color = "#ff4d4f";
            } else if (percent < 75) {
              color = "#faad14";
            }
            return (
              <Progress
                percent={percent}
                size="small"
                strokeColor={color}
                format={(percent) => `${percent}%`}
              />
            );
          },
        });
      });

      setRcaComparisonColumns(newColumns);

      // Prepare rows for RCA comparison table
      // First, collect all unique nodes across all models
      const uniqueNodes = new Set();
      
      // Only iterate through models that exist in the response
      selectedModels.forEach(model => {
        if (response[model] && Array.isArray(response[model].ranks)) {
          response[model].ranks.forEach(rank => {
            if (rank && rank.node) {
              uniqueNodes.add(rank.node);
            }
          });
        }
      });

      // If no nodes found, add a default node for displaying model errors
      if (uniqueNodes.size === 0) {
        uniqueNodes.add("No data available");
      }

      // Create a row for each unique node
      const rows = Array.from(uniqueNodes).map(node => {
        const row = { node, anomalyType: getNodeAnomalyType(node) };

        // Add data for each model
        selectedModels.forEach(model => {
          const hasError = response[model] && response[model].error;
          
          if (hasError) {
            // If model has an error, show that in the row
            row[`${model}_root_cause`] = "Error: " + response[model].error;
            row[`${model}_confidence`] = null;
          } else if (response[model]) {
            // Normal case - get data from the model
            row[`${model}_root_cause`] = getModelRootCause(model, node);
            row[`${model}_confidence`] = getModelConfidence(model, node);
          } else {
            // Model not in response at all
            row[`${model}_root_cause`] = "Model data unavailable";
            row[`${model}_confidence`] = null;
          }
        });

        return row;
      });

      setRcaComparisonData(rows);
    } catch (error) {
      console.error("Error comparing models:", error);
      setComparisonError(`Failed to compare models: ${error.message}`);
    } finally {
      setComparisonLoading(false);
    }
  };

  // Helper function to get the anomaly type for a node
  const getNodeAnomalyType = (nodeName) => {
    if (!comparisonData || !nodeName) return "Normal";
    
    // Check each model's results for this node's anomaly type
    for (const model of selectedModels) {
      if (!comparisonData[model] || !comparisonData[model].ranks) continue;
      
      const nodeData = comparisonData[model].ranks.find(
        rank => rank && rank.node === nodeName
      );
      
      if (nodeData && nodeData.type && nodeData.score > 1) {
        return nodeData.type;
      }
    }
    
    return "Normal";
  };

  // Helper function to get the root cause for a node from a specific model
  const getModelRootCause = (model, node) => {
    if (!comparisonData || !comparisonData[model] || !node) return "No data";
    
    // First check if we have direct rank data for this node
    if (comparisonData[model].ranks && Array.isArray(comparisonData[model].ranks)) {
      const rankData = comparisonData[model].ranks.find(
        rank => rank && rank.node === node
      );
      
      if (rankData && rankData.type) {
        return rankData.type;
      }
    }
    
    return "Normal";
  };

  // Helper function to get confidence score for a node from a specific model
  const getModelConfidence = (model, node) => {
    if (!comparisonData || !comparisonData[model] || !node) return null;

    // Check ranks data first for node-specific confidence
    if (comparisonData[model].ranks && Array.isArray(comparisonData[model].ranks)) {
      const rankData = comparisonData[model].ranks.find(
        rank => rank && rank.node === node
      );
      
      if (rankData && rankData.score !== undefined) {
        return rankData.score / 100; // Convert to decimal for consistency
      }
    }
    
    // Fallback to model's overall precision
    return comparisonData[model].rca_precision || 0.5;
  };

  const handleModelSelectionChange = (values) => {
    // Filter out null and undefined values
    const filteredValues = values.filter(
      (value) => value !== null && value !== undefined
    );
    
    // Only update if at least one model is selected
    if (filteredValues.length > 0) {
      setSelectedModels(filteredValues);
    }
  };

  // Initialize state for RCA comparison
  const [rcaComparisonColumns, setRcaComparisonColumns] = useState(
    initialRcaComparisonColumns
  );
  const [rcaComparisonData, setRcaComparisonData] = useState([]);

  useEffect(() => {
    fetchAvailableModels();
  }, []);

  return (
    <Card title="Per-Node RCA Diagnosis Results" bordered={false}>
      {/* Model Selection Controls */}
      <div style={{ marginBottom: 16 }}>
        <Row gutter={16} align="middle">
          <Col span={16}>
            <Select
              mode="multiple"
              style={{ width: '100%' }}
              placeholder="Select models to analyze"
              value={selectedModels}
              onChange={handleModelSelectionChange}
              optionLabelProp="label"
            >
              {availableModels.map((model) => (
                <Option 
                  key={model || `model-${Math.random()}`} 
                  value={model || ""}
                  label={model || "Unknown model"}
                >
                  <div>
                    <Text>{model || "Unknown model"}</Text>
                  </div>
                </Option>
              ))}
            </Select>
          </Col>
          <Col span={8}>
            <Button
              type="primary"
              onClick={handleCompareModels}
              loading={comparisonLoading}
              disabled={selectedModels.length === 0}
            >
              Analyze Models
            </Button>
          </Col>
        </Row>
        {comparisonError && (
          <Alert
            message={comparisonError}
            type="error"
            showIcon
            style={{ marginTop: 16 }}
          />
        )}
      </div>
      
      {/* RCA Comparison Table - Full Width */}
      {comparisonData ? (
        <Table
          columns={rcaComparisonColumns}
          dataSource={rcaComparisonData}
          rowKey="node"
          pagination={false}
          scroll={{ x: "max-content" }}
          style={{ marginTop: 16 }}
          loading={comparisonLoading}
        />
      ) : (
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description="Select one or more models and click Analyze Models to view results"
        />
      )}
    </Card>
  );
};

export default RanksPanel;
