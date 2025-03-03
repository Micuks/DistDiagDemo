import React, { useState, useEffect } from "react";
import {
  Card,
  Table,
  Space,
  Tag,
  Typography,
  Select,
  Button,
  Tabs,
  Alert,
  Spin,
  Row,
  Col,
  Divider,
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
} from "recharts";

const { Text, Title } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;

const RanksPanel = () => {
  const [loading, setLoading] = useState(false);
  const [anomalyRanks, setAnomalyRanks] = useState([]);
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModels, setSelectedModels] = useState([]);
  const [comparisonLoading, setComparisonLoading] = useState(false);
  const [comparisonData, setComparisonData] = useState(null);
  const [activeTab, setActiveTab] = useState("anomalyRanks");
  const [comparisonError, setComparisonError] = useState(null);

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
          case "memory":
            color = "blue";
            break;
          case "io":
            color = "orange";
            break;
          case "network":
            color = "green";
            break;
          default:
            color = "default";
        }
        return <Tag color={color}> {type.toUpperCase()} </Tag>;
      },
    },
    {
      title: "Confidence",
      dataIndex: "score",
      key: "score",
      render: (score) => {
        const percent = Math.round(score * 100);
        let color = "#52c41a"; // Green for high confidence
        if (percent < 50) {
          color = "#ff4d4f"; // Red for low confidence
        } else if (percent < 75) {
          color = "#faad14"; // Yellow for medium confidence
        }
        return <Text style={{ color }}> {percent} % </Text>;
      },
      sorter: (a, b) => a.score - b.score,
      defaultSortOrder: "descend",
    },
  ];

  const fetchAnomalyRanks = async () => {
    try {
      setLoading(true);
      const ranks = await anomalyService.getAnomalyRanks();
      setAnomalyRanks(ranks);
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
      // Select first two models by default if available
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
    if (selectedModels.length < 1) {
      setComparisonError("Please select at least one model to analyze.");
      return;
    }

    try {
      setComparisonError(null);
      setComparisonLoading(true);
      const comparisonResult = await anomalyService.compareModels(
        selectedModels
      );
      setComparisonData(comparisonResult);
      // Auto-switch to comparison tab
      setActiveTab("modelComparison");
    } catch (error) {
      console.error("Error comparing models:", error);
      setComparisonError("Failed to compare models. Please try again.");
    } finally {
      setComparisonLoading(false);
    }
  };

  const handleModelSelectionChange = (values) => {
    setSelectedModels(values);
  };

  useEffect(() => {
    fetchAnomalyRanks();
    fetchAvailableModels();
    const interval = setInterval(fetchAnomalyRanks, 5000);
    return () => clearInterval(interval);
  }, []);

  // Transform comparison data for chart visualization
  const prepareChartData = () => {
    if (!comparisonData) return [];

    const metrics = ["precision", "recall", "f1_score", "accuracy"];
    const chartData = [];

    metrics.forEach((metric) => {
      const dataPoint = { metric };
      selectedModels.forEach((model) => {
        if (
          comparisonData[model] &&
          typeof comparisonData[model][metric] === "number"
        ) {
          dataPoint[model] = comparisonData[model][metric] * 100; // Convert to percentage
        }
      });
      chartData.push(dataPoint);
    });

    return chartData;
  };

  return (
    <Card>
      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab="Anomaly Ranks" key="anomalyRanks">
          <Table
            columns={columns}
            dataSource={anomalyRanks}
            rowKey={(record) =>
              `${record.node}-${record.timestamp}-${record.type}`
            }
            loading={loading}
            pagination={{ pageSize: 10 }}
          />{" "}
        </TabPane>{" "}
        <TabPane tab="Model Comparison" key="modelComparison">
          <Space direction="vertical" style={{ width: "100%" }}>
            <Card>
              <Row gutter={16}>
                <Col span={16}>
                  <Space direction="vertical" style={{ width: "100%" }}>
                    <Title level={5}> Select Models to Compare: </Title>{" "}
                    <Select
                      mode="multiple"
                      style={{ width: "100%" }}
                      placeholder="Select models to compare"
                      value={selectedModels}
                      onChange={handleModelSelectionChange}
                      optionLabelProp="label"
                    >
                      {" "}
                      {availableModels.map((model) => (
                        <Option
                          value={model.name}
                          key={model.name}
                          label={model.name}
                        >
                          <Space>
                            <span> {model.name} </span>{" "}
                          </Space>{" "}
                        </Option>
                      ))}{" "}
                    </Select>{" "}
                  </Space>{" "}
                </Col>{" "}
                <Col
                  span={8}
                  style={{ display: "flex", alignItems: "flex-end" }}
                >
                  <Button
                    type="primary"
                    onClick={handleCompareModels}
                    loading={comparisonLoading}
                    disabled={selectedModels.length < 1}
                  >
                    Compare Models{" "}
                  </Button>{" "}
                </Col>{" "}
              </Row>
              {comparisonError && (
                <Alert
                  message="Error"
                  description={comparisonError}
                  type="error"
                  showIcon
                  style={{ marginTop: 16 }}
                />
              )}{" "}
            </Card>
            {comparisonLoading ? (
              <div style={{ textAlign: "center", padding: "40px 0" }}>
                <Spin size="large" />
                <div style={{ marginTop: 16 }}>
                  {" "}
                  Comparing model performance...{" "}
                </div>{" "}
              </div>
            ) : comparisonData ? (
              <Card title="RCA Performance Comparison">
                <div style={{ height: 400 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={prepareChartData()}
                      margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="metric" />
                      <YAxis
                        label={{
                          value: "Score (%)",
                          angle: -90,
                          position: "insideLeft",
                        }}
                        domain={[0, 100]}
                      />{" "}
                      <Tooltip formatter={(value) => `${value.toFixed(2)}%`} />{" "}
                      <Legend />{" "}
                      {selectedModels.map((model, index) => (
                        <Bar
                          key={model}
                          dataKey={model}
                          name={model}
                          fill={
                            [
                              "#1890ff",
                              "#52c41a",
                              "#fa8c16",
                              "#f5222d",
                              "#722ed1",
                            ][index % 5]
                          }
                        />
                      ))}{" "}
                    </BarChart>{" "}
                  </ResponsiveContainer>{" "}
                </div>{" "}
                <Divider />
                <Row gutter={16}>
                  {" "}
                  {selectedModels.map(
                    (model) =>
                      comparisonData[model] && (
                        <Col
                          span={Math.floor(24 / selectedModels.length)}
                          key={model}
                        >
                          <Card title={model} size="small">
                            <p>
                              {" "}
                              <strong> Precision: </strong>{" "}
                              {(comparisonData[model].precision * 100).toFixed(
                                2
                              )}
                              %
                            </p>
                            <p>
                              {" "}
                              <strong> Recall: </strong>{" "}
                              {(comparisonData[model].recall * 100).toFixed(2)}%
                            </p>
                            <p>
                              {" "}
                              <strong> F1 Score: </strong>{" "}
                              {(comparisonData[model].f1_score * 100).toFixed(
                                2
                              )}
                              %
                            </p>
                            <p>
                              {" "}
                              <strong> Accuracy: </strong>{" "}
                              {(comparisonData[model].accuracy * 100).toFixed(
                                2
                              )}
                              %
                            </p>
                          </Card>{" "}
                        </Col>
                      )
                  )}{" "}
                </Row>{" "}
              </Card>
            ) : (
              <Card>
                <div style={{ textAlign: "center", padding: "20px" }}>
                  <Text type="secondary">
                    {" "}
                    Select models and click "Compare Models" to see performance
                    metrics{" "}
                  </Text>{" "}
                </div>{" "}
              </Card>
            )}{" "}
          </Space>{" "}
        </TabPane>{" "}
      </Tabs>{" "}
    </Card>
  );
};

export default RanksPanel;
