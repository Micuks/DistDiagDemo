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
  List,
  Tooltip
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
  Tooltip as RechartsTooltip
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
                            <p>Score: {Math.round(item.positive_prob_score * 1000) / 1000}</p>
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
            <Card title="Anomaly Distribution by Type" className="chart-card" bordered={true} style={{ boxShadow: '0 1px 4px rgba(0,0,0,0.1)' }}>
              <ResponsiveContainer width="100%" height={350}>
                <BarChart data={getAnomalyDistribution()} barSize={30} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="type" 
                    tick={{ fill: '#333', fontWeight: 'bold' }} 
                    tickFormatter={(value) => value.replace(/_/g, ' ')}
                  />
                  <YAxis 
                    label={{ value: 'Count', angle: -90, position: 'insideLeft', offset: -5 }}
                    tickCount={5}
                  />
                  <Tooltip 
                    formatter={(value, name, props) => [`${value} anomalies`, name]}
                    labelFormatter={(value) => `Type: ${value.replace(/_/g, ' ')}`}
                    cursor={{ fill: 'rgba(0, 0, 0, 0.05)' }}
                    contentStyle={{ border: '1px solid #ccc', borderRadius: '4px', padding: '10px' }}
                  />
                  <Legend verticalAlign="top" wrapperStyle={{ paddingBottom: '15px' }} />
                  
                  {selectedModels.map((model, index) => (
                    comparisonData[model] && !comparisonData[model].error && (
                      <Bar
                        key={model}
                        dataKey={model}
                        name={model.split('_')[0]}
                        fill={getModelColor(model, index)}
                        radius={[4, 4, 0, 0]}
                        animationDuration={1000}
                        label={{ 
                          position: 'top', 
                          fill: '#666',
                          fontSize: 10,
                          formatter: (value) => value > 0 ? value : ''
                        }}
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
            <Card title="Anomaly Timeline" className="chart-card" bordered={true} style={{ boxShadow: '0 1px 4px rgba(0,0,0,0.1)' }}>
              <div style={{ marginBottom: 16, display: 'flex', alignItems: 'center' }}>
                <Switch 
                  checkedChildren="Raw Data" 
                  unCheckedChildren="Processed" 
                  onChange={value => setShowRawData(value)} 
                  style={{ marginRight: 16 }}
                />
                <Tooltip title="Switch between raw metric data and processed anomaly data">
                  <InfoCircleOutlined style={{ marginRight: 16 }} />
                </Tooltip>
                <Radio.Group 
                  value={timelineView} 
                  onChange={e => setTimelineView(e.target.value)}
                  buttonStyle="solid"
                  style={{ marginRight: 'auto' }}
                >
                  <Tooltip title="Shows the number of detected anomalies over time">
                    <Radio.Button value="anomalies">Anomalies</Radio.Button>
                  </Tooltip>
                  <Tooltip title="Shows the trend direction of anomalies (increasing or decreasing)">
                    <Radio.Button value="trends">Trends</Radio.Button>
                  </Tooltip>
                  <Tooltip title="Shows the importance of key metrics related to the anomalies">
                    <Radio.Button value="features">Features</Radio.Button>
                  </Tooltip>
                </Radio.Group>
              </div>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={getTimelineData()} margin={{ top: 5, right: 30, left: 20, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis 
                    dataKey="timestamp" 
                    tickFormatter={timestamp => {
                      const date = new Date(timestamp);
                      return date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
                    }}
                    label={{ value: 'Time', position: 'insideBottomRight', offset: -5 }}
                    stroke="#333"
                  />
                  <YAxis 
                    label={{ 
                      value: timelineView === 'anomalies' ? 'Anomaly Count' : 
                            timelineView === 'trends' ? 'Trend Direction' : 'Feature Importance',
                      angle: -90, 
                      position: 'insideLeft',
                      offset: -10
                    }}
                    stroke="#333"
                    tickLine={true}
                    axisLine={true}
                  />
                  <RechartsTooltip 
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
                    contentStyle={{ 
                      border: '1px solid #ccc', 
                      borderRadius: '4px', 
                      padding: '10px',
                      backgroundColor: 'rgba(255, 255, 255, 0.9)'
                    }}
                  />
                  <Legend verticalAlign="top" wrapperStyle={{ paddingBottom: '8px' }} />
                  
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
                        name={timelineView === 'features' ? `${model} (${dataKey.split('_')[1]})` : model.split('_')[0]}
                        stroke={getModelColor(model, index)}
                        strokeWidth={2}
                        dot={{ r: 4, strokeWidth: 1, fill: 'white' }}
                        activeDot={{ r: 6, stroke: '#666', strokeWidth: 1 }}
                        isAnimationActive={true}
                        animationDuration={800}
                        animationEasing="ease-in-out"
                      />
                    );
                  })}
                </LineChart>
              </ResponsiveContainer>
            </Card>
          </Col>
        </Row>
      </div>
    </div>
  );
};

export default DistDiagnosisPanel; 