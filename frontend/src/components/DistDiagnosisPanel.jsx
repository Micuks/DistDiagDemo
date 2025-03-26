import React from "react";
import { SyncOutlined, InfoCircleOutlined } from '@ant-design/icons';
import {
  Card,
  Space,
  Tag,
  Typography,
  Button,
  Spin,
  Row,
  Col,
  Empty,
  Radio,
  Switch,
  List
} from "antd";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  LineChart,
  Line,
  Tooltip
} from "recharts";
import { getRankColor, getModelColor } from '../utils/rankUtils.jsx';

const { Text } = Typography;

const DistDiagnosisPanel = ({ 
  selectedModels = [], 
  comparisonData = {}, 
  comparisonLoading = false,
  timelineView = 'anomalies',
  setTimelineView,
  showRawData = false, 
  setShowRawData,
  proceedToMetricRanking,
  getAnomalyDistribution,
  getTimelineData,
  renderForceGraph,
  renderNodeHeatmap
}) => {
  
  if (selectedModels.length === 0 || !comparisonData || Object.keys(comparisonData).length === 0) {
    if (comparisonLoading) {
      return (
        <div style={{ textAlign: 'center', padding: '40px 0' }}>
          <Spin size="large" />
          <div style={{ marginTop: 16 }}>
            <Text>Analyzing data with selected models...</Text>
          </div>
        </div>
      );
    }
    
    return (
      <Empty
        image={Empty.PRESENTED_IMAGE_SIMPLE}
        description={
          <div>
            <p>No analysis data available</p>
            <p>Select one or more models and click "Run Analysis" to view results</p>
          </div>
        }
      />
    );
  }

  return (
    <div className="comparison-results">
      {selectedModels.map(model => {
        const modelData = comparisonData[model] || {};
        const anomalies = modelData.ranks || [];
        
        return (
          <div key={model} className="model-analysis-section">
            <Card 
              title={
                <div style={{ display: 'flex', alignItems: 'center' }}>
                  <span style={{ marginRight: '10px', color: getModelColor(model) }}>
                    {model}
                  </span>
                  <Tag color={getModelColor(model, true)}>
                    {anomalies.length} {anomalies.length === 1 ? 'anomaly' : 'anomalies'} detected
                  </Tag>
                </div>
              }
              style={{ marginBottom: '16px' }}
            >
              {anomalies.length > 0 ? (
                <List
                  dataSource={anomalies}
                  renderItem={item => (
                    <List.Item 
                      key={`${item.node}-${item.type}`}
                      actions={[
                        <Button 
                          type="primary" 
                          onClick={() => proceedToMetricRanking(item)}
                        >
                          Analyze Metrics
                        </Button>
                      ]}
                    >
                      <List.Item.Meta
                        title={
                          <div>
                            <Tag color={getRankColor(item.type)}>{item.type}</Tag>
                            <span style={{ marginLeft: 8 }}>{item.node}</span>
                          </div>
                        }
                        description={
                          <div>
                            <p>Score: {Math.round(item.score * 1000) / 1000}</p>
                            <p>{item.description}</p>
                            {item.related_metrics && item.related_metrics.length > 0 && (
                              <Tooltip title={
                                <div>
                                  <div style={{ marginBottom: 5 }}>Related metrics:</div>
                                  {item.related_metrics.map((metric, i) => (
                                    <div key={i} style={{ fontSize: 12 }}>
                                      {metric.name}: {metric.value.toFixed(2)}
                                    </div>
                                  ))}
                                </div>
                              }>
                                <Space>
                                  <Text type="secondary">Related metrics:</Text>
                                  <InfoCircleOutlined style={{ color: '#1890ff' }} />
                                </Space>
                              </Tooltip>
                            )}
                          </div>
                        }
                      />
                    </List.Item>
                  )}
                />
              ) : (
                <Empty description="No anomalies detected" />
              )}
            </Card>
          </div>
        );
      })}
      
      {/* Visualization Components */}
      <div className="visualization-components">
        <Row gutter={16}>
          <Col xs={24} lg={12}>
            <Card title="Propagation Graph" className="chart-card">
              {renderForceGraph()}
            </Card>
          </Col>
          
          <Col xs={24} lg={12}>
            <Card title="Anomaly Distribution by Type" className="chart-card">
              <ResponsiveContainer width="100%" height={350}>
                <BarChart data={getAnomalyDistribution()}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="type" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  
                  {selectedModels.map((model, index) => (
                    comparisonData[model] && !comparisonData[model].error && (
                      <Bar
                        key={model}
                        dataKey={model}
                        fill={getModelColor(model, index)}
                      />
                    )
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </Col>
        </Row>
        
        <Row gutter={16} style={{ marginTop: 16 }}>
          <Col span={24}>
            <Card title="Anomaly Timeline" className="chart-card">
              <div style={{ marginBottom: 16 }}>
                <Switch 
                  checkedChildren="Raw Data" 
                  unCheckedChildren="Processed" 
                  onChange={value => setShowRawData(value)} 
                  style={{ marginRight: 16 }}
                />
                <Radio.Group 
                  value={timelineView} 
                  onChange={e => setTimelineView(e.target.value)}
                  buttonStyle="solid"
                >
                  <Radio.Button value="anomalies">Anomalies</Radio.Button>
                  <Radio.Button value="trends">Trends</Radio.Button>
                  <Radio.Button value="features">Features</Radio.Button>
                </Radio.Group>
              </div>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={getTimelineData()}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="timestamp" 
                    tickFormatter={timestamp => {
                      const date = new Date(timestamp);
                      return date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
                    }}
                    label={{ value: 'Time', position: 'insideBottomRight', offset: -5 }}
                  />
                  <YAxis 
                    label={{ 
                      value: timelineView === 'anomalies' ? 'Anomaly Count' : 
                            timelineView === 'trends' ? 'Trend Direction' : 'Feature Importance',
                      angle: -90, 
                      position: 'insideLeft' 
                    }}
                  />
                  <Tooltip 
                    labelFormatter={timestamp => {
                      const date = new Date(timestamp);
                      return date.toLocaleString([], {
                        year: 'numeric',
                        month: 'short',
                        day: 'numeric',
                        hour: '2-digit',
                        minute: '2-digit',
                        second: '2-digit'
                      });
                    }}
                    formatter={(value, name) => {
                      // Format the tooltip value based on the data type
                      if (name.includes('_trend')) {
                        return [`${value > 0 ? '+' : ''}${value}`, 'Trend'];
                      } else if (name.includes('_')) {
                        return [value.toFixed(3), name.split('_')[1]];
                      }
                      return [value, name];
                    }}
                  />
                  <Legend />
                  
                  {selectedModels.map((model, index) => {
                    if (!comparisonData[model] || comparisonData[model].error) {
                      return null;
                    }
                    
                    // Determine which data to show based on the selected view
                    const dataKey = timelineView === 'anomalies' ? model :
                                   timelineView === 'trends' ? `${model}_trend` :
                                   // For features view, show the first feature if available
                                   comparisonData[model].ranks && 
                                   comparisonData[model].ranks[0] && 
                                   comparisonData[model].ranks[0].features ?
                                   `${model}_${Object.keys(comparisonData[model].ranks[0].features)[0]}` :
                                   model;
                    
                    return (
                      <Line
                        key={dataKey}
                        type="monotone"
                        dataKey={dataKey}
                        name={timelineView === 'features' ? `${model} (${dataKey.split('_')[1]})` : model}
                        stroke={getModelColor(model, index)}
                        strokeWidth={2}
                        dot={{ r: 4 }}
                        activeDot={{ r: 8 }}
                        isAnimationActive={true}
                        animationDuration={500}
                      />
                    );
                  })}
                </LineChart>
              </ResponsiveContainer>
            </Card>
          </Col>
        </Row>
        
        <Row gutter={16} style={{ marginTop: 16 }}>
          <Col span={24}>
            <Card title="Node Performance Heatmap" className="chart-card">
              {renderNodeHeatmap()}
            </Card>
          </Col>
        </Row>
      </div>
    </div>
  );
};

export default DistDiagnosisPanel; 