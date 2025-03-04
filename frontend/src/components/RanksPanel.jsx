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
const { Option } = Select;

const RanksPanel = () => {
  const [loading, setLoading] = useState(false);
  const [anomalyRanks, setAnomalyRanks] = useState([]);
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModels, setSelectedModels] = useState([]);
  const [comparisonLoading, setComparisonLoading] = useState(false);
  const [comparisonData, setComparisonData] = useState(null);
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
      const comparisonResult = await anomalyService.compareModels(selectedModels);
      setComparisonData(comparisonResult);
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

  const prepareChartData = () => {
    if (!comparisonData) return [];

    const metrics = ["precision", "recall", "f1_score", "accuracy"];
    const chartData = [];

    metrics.forEach((metric) => {
      const dataPoint = { metric };
      selectedModels.forEach((model) => {
        if (comparisonData[model] && typeof comparisonData[model][metric] === "number") {
          dataPoint[model] = comparisonData[model][metric] * 100;
        }
      });
      chartData.push(dataPoint);
    });

    return chartData;
  };

  return (
    <Card>
      <Row gutter={[24, 24]}>
        {/* Anomaly Ranks Column */}
        <Col xs={24} md={12}>
          <Card title="Anomaly Ranking" bordered={false}>
            <Table
              columns={columns}
              dataSource={anomalyRanks}
              rowKey={(record) => `${record.node}-${record.timestamp}-${record.type}`}
              loading={loading}
              pagination={{ pageSize: 5 }}
              scroll={{ y: 400 }}
            />
          </Card>
        </Col>

        {/* Model Comparison Column */}
        <Col xs={24} md={12}>
          <Card 
            title="Model Performance Comparison" 
            bordered={false}
            extra={
              <Space>
                <Select
                  mode="multiple"
                  style={{ width: 200 }}
                  placeholder="Select models"
                  value={selectedModels}
                  onChange={handleModelSelectionChange}
                  optionLabelProp="label"
                >
                  {availableModels.map((model) => (
                    <Option value={model.name} key={model.name} label={model.name}>
                      {model.name}
                    </Option>
                  ))}
                </Select>
                <Button
                  type="primary"
                  onClick={handleCompareModels}
                  loading={comparisonLoading}
                  disabled={selectedModels.length < 1}
                >
                  Compare
                </Button>
              </Space>
            }
          >
            {comparisonError && (
              <Alert
                message="Error"
                description={comparisonError}
                type="error"
                showIcon
                style={{ marginBottom: 16 }}
              />
            )}

            {comparisonData ? (
              <>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={prepareChartData()}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="metric" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    {selectedModels.map((model, index) => (
                      <Bar
                        key={model}
                        dataKey={model}
                        fill={`hsl(${(index * 360) / selectedModels.length}, 70%, 50%)`}
                      />
                    ))}
                  </BarChart>
                </ResponsiveContainer>

                <Divider>Detailed Metrics</Divider>
                <Row gutter={[16, 16]}>
                  {selectedModels.map((model) => (
                    <Col span={24} key={model}>
                      <Card size="small" title={model}>
                        <Space direction="vertical" style={{ width: '100%' }}>
                          <Progress
                            percent={Math.round(comparisonData[model].precision * 100)}
                            status="active"
                            format={(percent) => `Precision: ${percent}%`}
                          />
                          <Progress
                            percent={Math.round(comparisonData[model].recall * 100)}
                            status="active"
                            format={(percent) => `Recall: ${percent}%`}
                          />
                          <Progress
                            percent={Math.round(comparisonData[model].f1_score * 100)}
                            status="active"
                            format={(percent) => `F1 Score: ${percent}%`}
                          />
                          <Text>Detection Time: {comparisonData[model].detection_time}s</Text>
                          <Text>RCA Time: {comparisonData[model].rca_time}s</Text>
                        </Space>
                      </Card>
                    </Col>
                  ))}
                </Row>
              </>
            ) : (
              <div style={{ textAlign: 'center', padding: 40 }}>
                <Text type="secondary">Select models and click compare to view performance</Text>
              </div>
            )}
          </Card>
        </Col>
      </Row>
    </Card>
  );
};

export default RanksPanel;
