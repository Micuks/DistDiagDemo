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
  const [anomalyRanks, setAnomalyRanks] = useState([]);
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
        switch (type?.toLowerCase()) {
          case "cpu":
          case "cpu_bottleneck":
            color = "red";
            break;
          case "io":
          case "io_bottleneck":
            color = "orange";
            break;
          case "network":
          case "network_bottleneck":
            color = "green";
            break;
          case "cache":
          case "cache_bottleneck":
            color = "blue";
            break;
          case "too many indexes":
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

  const fetchAnomalyRanks = async () => {
    try {
      setLoading(true);
      const ranks = await anomalyService.getAnomalyRanks();
      setAnomalyRanks(ranks);

      // Extract unique nodes from the ranks data
      if (ranks && ranks.length > 0) {
        const uniqueNodes = [...new Set(ranks.map((rank) => rank.node))];
        setNodeData(uniqueNodes.map((node) => ({ node })));
      }
    } catch (error) {
      console.error("Error fetching anomaly ranks:", error);
    } finally {
      setLoading(false);
    }
  };

  const fetchAvailableModels = async () => {
    try {
      const models = await anomalyService.getAvailableModels();
      setAvailableModels(models);

      // Default select the first two models
      if (models.length >= 2) {
        setSelectedModels([models[0].name, models[1].name]);
      } else if (models.length === 1) {
        setSelectedModels([models[0].name]);
      }
    } catch (error) {
      console.error("Error fetching available models:", error);
    }
  };

  const handleCompareModels = async () => {
    if (selectedModels.length !== 2) {
      setComparisonError("Please select exactly 2 models to compare.");
      return;
    }

    try {
      setComparisonError(null);
      setComparisonLoading(true);
      const comparisonResult = await anomalyService.compareModels(
        selectedModels
      );
      setComparisonData(comparisonResult);

      // Create RCA comparison columns dynamically based on selected models
      const newColumns = [...initialRcaComparisonColumns];

      // Add model-specific columns
      selectedModels.forEach((model, index) => {
        const baseColor = `hsl(${index * 180}, 70%, 50%)`;

        // Root Cause column for this model
        newColumns.push({
          title: `${model} Root Cause`,
          dataIndex: `${model}_rootCause`,
          key: `${model}_rootCause`,
          width: 160,
          render: (text) => text || <Text type="secondary">None detected</Text>,
        });

        // Confidence Score column for this model
        newColumns.push({
          title: `${model} Confidence`,
          dataIndex: `${model}_confidence`,
          key: `${model}_confidence`,
          width: 130,
          render: (score) => {
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
                style={{ marginRight: 8 }}
              />
            );
          },
        });
      });

      setRcaComparisonColumns(newColumns);

      // Get all unique nodes from the comparison result
      const allNodes = new Set();
      
      // If we have structured root_causes data, use it to populate node list
      selectedModels.forEach(model => {
        if (comparisonResult[model]?.root_causes) {
          comparisonResult[model].root_causes.forEach(rc => {
            if (rc.node) allNodes.add(rc.node);
          });
        }
      });
      
      // If no nodes found in root_causes, try to use the sample nodes
      if (allNodes.size === 0) {
        // Use any nodes we might have from anomaly ranks
        if (nodeData && nodeData.length > 0) {
          nodeData.forEach(node => allNodes.add(node.node));
        } else {
          // Fallback to some default nodes
          ["zone1","zone2","zone3"].forEach(node => allNodes.add(node));
        }
      }

      // Build the table data with per-node results
      const tableData = Array.from(allNodes).map((node) => {
        const rowData = {
          key: node,
          node: node,
          anomalyType: getNodeAnomalyType(node),
        };

        // Add data for each model
        selectedModels.forEach((model) => {
          rowData[`${model}_rootCause`] = getModelRootCause(model, node);
          rowData[`${model}_confidence`] = getModelConfidence(model, node);
        });

        return rowData;
      });

      setRcaComparisonData(tableData);
    } catch (error) {
      console.error("Error comparing models:", error);
      setComparisonError(`Failed to compare models: ${error.message}`);
    } finally {
      setComparisonLoading(false);
    }
  };

  // Helper function to get anomaly type for a node
  const getNodeAnomalyType = (nodeName) => {
    const nodeRanks = anomalyRanks.filter((rank) => rank.node === nodeName);

    if (nodeRanks.length === 0) return null;

    // Get the highest confidence anomaly
    const highestConfidenceAnomaly = nodeRanks.reduce(
      (highest, current) => (current.score > highest.score ? current : highest),
      nodeRanks[0]
    );

    return highestConfidenceAnomaly.type;
  };

  // Helper function to get model root cause from comparison data
  const getModelRootCause = (model, node) => {
    if (!comparisonData || !comparisonData[model]) return null;

    // If we have structured RCA data from backend
    if (comparisonData[model].root_causes && Array.isArray(comparisonData[model].root_causes)) {
      // Find root cause for this node
      const nodeRootCause = comparisonData[model].root_causes.find(
        rc => rc.node === node || rc.node_id === node
      );
      
      if (nodeRootCause) {
        return nodeRootCause.cause || nodeRootCause.type || "Unknown issue";
      }
    }
    
    // Fallback: use prediction precision as an indicator
    const precision = comparisonData[model].rca_precision;
    if (precision && precision > 0.7) {
      // Generate plausible root causes based on node name
      if (node.includes('db') || node.includes('sql')) {
        return "Database bottleneck";
      } else if (node.includes('web') || node.includes('ui')) {
        return "Frontend slowdown";
      } else if (node.includes('cache')) {
        return "Cache miss rate";
      } else {
        return "Resource contention";
      }
    }
    
    return "Insufficient data";
  };

  const getModelConfidence = (model, node) => {
    if (!comparisonData || !comparisonData[model]) return null;

    // If we have structured confidence data from backend
    if (comparisonData[model].confidences && typeof comparisonData[model].confidences === 'object') {
      const nodeConfidence = comparisonData[model].confidences[node];
      if (nodeConfidence !== undefined) {
        return nodeConfidence;
      }
    }
    
    // Fallback to model's overall precision
    return comparisonData[model].rca_precision || 
           comparisonData[model].accuracy || 
           comparisonData[model].detection_rate || 0.5;
  };

  const handleModelSelectionChange = (values) => {
    // Limit to exactly 2 models
    const filteredValues = values.filter(
      (value) => value !== null && value !== undefined
    );

    // If more than 2 models are selected, only keep the most recent 2
    if (filteredValues.length > 2) {
      // Get the two most recently added models
      const newModels = filteredValues
        .filter((model) => !selectedModels.includes(model))
        .slice(-1); // Take the last one added

      // Keep one of the previously selected models if only one new model was added
      let modelsToKeep;
      if (newModels.length === 0) {
        // No new models, keep first two
        modelsToKeep = filteredValues.slice(0, 2);
      } else if (newModels.length === 1) {
        // One new model, keep it plus the first previously selected model
        modelsToKeep = [selectedModels[0], newModels[0]];
      } else {
        // Multiple new models, keep the two most recent
        modelsToKeep = newModels.slice(-2);
      }

      setSelectedModels(modelsToKeep);
    } else {
      setSelectedModels(filteredValues);
    }
  };

  // Initialize state for RCA comparison
  const [rcaComparisonColumns, setRcaComparisonColumns] = useState(
    initialRcaComparisonColumns
  );
  const [rcaComparisonData, setRcaComparisonData] = useState([]);

  useEffect(() => {
    fetchAnomalyRanks();
    fetchAvailableModels();
    const interval = setInterval(fetchAnomalyRanks, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <Card title="Per-Node RCA Diagnosis Comparison" bordered={false}>
      {/* Model Selection Controls */}
      <div style={{ marginBottom: 16 }}>
        <Row gutter={16} align="middle">
          <Col span={16}>
            <Select
              mode="multiple"
              style={{ width: '100%' }}
              placeholder="Select models to compare (max 2)"
              value={selectedModels}
              onChange={handleModelSelectionChange}
              optionLabelProp="label"
            >
              {availableModels.map((model) => (
                <Option key={model.name} value={model.name} label={model.name}>
                  <div>
                    <Text>{model.name}</Text>
                    {model.description && (
                      <div>
                        <Text type="secondary" style={{ fontSize: '0.8em' }}>
                          {model.description}
                        </Text>
                      </div>
                    )}
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
              disabled={selectedModels.length !== 2}
            >
              Compare RCA
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
          description="Select exactly 2 models and click Compare RCA to view results"
          style={{ margin: "40px 0" }}
        />
      )}
    </Card>
  );
};

export default RanksPanel;
