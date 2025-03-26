import React from "react";
import { ReloadOutlined } from '@ant-design/icons';
import {
  Card,
  Tag,
  Typography,
  Button,
  Spin,
  Row,
  Col,
  Progress,
  Statistic,
  List
} from "antd";
import {
  PieChart,
  Pie,
  ResponsiveContainer,
  Cell,
  Tooltip,
  Legend
} from "recharts";
import { 
  getRankColor, 
  getCategoryColor, 
  getCategoryColorHex, 
  getScoreStatus, 
  getCategoryDistribution, 
  renderCustomizedLabel, 
  getRecommendedActions 
} from '../utils/rankUtils.jsx';

const { Text } = Typography;

const MetricRooterPanel = ({ 
  selectedAnomaly,
  metricRankings,
  metricRankingLoading,
  proceedToMetricRanking,
  goBackToAnomalySelection
}) => {
  
  // Format the value based on the metric type
  const formatMetricValue = (metric) => {
    // Use latest_value if available, otherwise use value
    const val = metric.latest_value !== undefined ? metric.latest_value : metric.value;
    
    if (val === undefined || val === null) {
      return 'N/A';
    }
    
    // Format large numbers for better readability without adding units
    if (val > 1000000000) {
      return `${(val / 1000000000).toFixed(2)}`;
    } else if (val > 1000000) {
      return `${(val / 1000000).toFixed(2)}`;
    } else if (val > 1000) {
      return `${(val / 1000).toFixed(2)}`;
    }
    
    // Default formatting
    return val.toFixed(2);
  };
  
  // Get significance level text
  const getSignificanceText = (zScore) => {
    if (zScore > 3) return "Critical";
    if (zScore > 2) return "High";
    if (zScore > 1.5) return "Moderate";
    return "Low";
  };
  
  // Get significance tag color
  const getSignificanceColor = (zScore) => {
    if (zScore > 3) return "#f5222d";
    if (zScore > 2) return "#fa541c";
    if (zScore > 1.5) return "#fa8c16";
    return "#52c41a";
  };
  
  return (
    <div className="metric-ranking-analysis">
      <div className="step-header" style={{ marginBottom: '20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <Button 
            type="primary" 
            onClick={goBackToAnomalySelection} 
            style={{ marginRight: '10px' }}
          >
            Back to Anomaly Selection
          </Button>
          <Tag color={getRankColor(selectedAnomaly?.type)} style={{ fontSize: '16px', padding: '5px 10px' }}>
            {selectedAnomaly?.type}
          </Tag>
          <span style={{ marginLeft: '10px', fontSize: '16px', fontWeight: 'bold' }}>
            {selectedAnomaly?.node}
          </span>
        </div>
        <div>
          <Button 
            onClick={() => proceedToMetricRanking(selectedAnomaly)}
            icon={<ReloadOutlined />}
          >
            Refresh Analysis
          </Button>
        </div>
      </div>
      
      {metricRankingLoading ? (
        <div style={{ textAlign: 'center', padding: '40px 0' }}>
          <Spin size="large" />
          <div style={{ marginTop: 16 }}>
            <Text>Analyzing metrics with MetricRooter...</Text>
          </div>
        </div>
      ) : (
        <div>
          <Row gutter={16}>
            <Col xs={24} lg={16}>
              <Card title="Top Metrics by Anomaly Score" className="chart-card">
                <List
                  bordered
                  dataSource={metricRankings.metrics || []}
                  renderItem={metric => {
                    const details = metric.details || {};
                    const primaryFactor = Object.entries(details)
                      .filter(([key, _]) => key !== 'z_score')
                      .sort((a, b) => b[1] - a[1])
                      .slice(0, 2);
                    
                    return (
                      <List.Item 
                        className="metric-item"
                        style={{
                          background: "#f9f9f9",
                          borderRadius: "8px",
                          padding: "16px",
                          marginBottom: "10px"
                        }}
                      >
                        <List.Item.Meta
                          title={
                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                              <div>
                                <span style={{ fontSize: '16px', fontWeight: 'bold' }}>{metric.name}</span>
                                <Tag color={getCategoryColor(metric.category)} style={{ marginLeft: '10px' }}>
                                  {metric.category}
                                </Tag>
                              </div>
                              <Tag 
                                color={getSignificanceColor(metric.z_score)} 
                                style={{ marginLeft: 'auto' }}
                              >
                                {getSignificanceText(metric.z_score)} Anomaly
                              </Tag>
                            </div>
                          }
                          description={
                            <div style={{ marginTop: '10px' }}>
                              <Row gutter={[16, 8]}>
                                <Col span={12}>
                                  <Statistic 
                                    title="Z-score" 
                                    value={metric.z_score} 
                                    precision={2} 
                                    valueStyle={{ color: getSignificanceColor(metric.z_score) }}
                                  />
                                </Col>
                                <Col span={12}>
                                  <Statistic 
                                    title="Current Value" 
                                    value={formatMetricValue(metric)} 
                                    suffix={metric.mean ? ` (mean: ${formatMetricValue({latest_value: metric.mean})})` : ""}
                                    valueStyle={{ fontSize: '16px' }}
                                  />
                                </Col>
                              </Row>
                              
                              <div style={{ marginTop: '10px' }}>
                                {primaryFactor.map(([factor, value]) => (
                                  <div key={factor} style={{ display: 'flex', alignItems: 'center', marginBottom: '8px' }}>
                                    <span style={{ width: '120px', fontSize: '14px' }}>
                                      {factor.charAt(0).toUpperCase() + factor.slice(1)}:
                                    </span>
                                    <Progress 
                                      percent={factor === 'connections' ? 
                                        Math.min(100, value * 100 / 3) : // Show connections count as percentage of 3
                                        Math.min(100, value * 100)} 
                                      size="small" 
                                      status={factor !== 'connections' && value > 0.7 ? "exception" : "normal"}
                                      style={{ flex: 1, marginLeft: '10px' }}
                                      format={() => value.toFixed(2)} // Show the actual value instead of percentage
                                    />
                                  </div>
                                ))}
                              </div>
                            </div>
                          }
                        />
                        <div style={{ width: '100px', textAlign: 'center' }} className="metric-score">
                          <Progress 
                            type="circle" 
                            format={() => metric.score.toFixed(2)}
                            percent={Math.min(100, metric.score * 100)} 
                            width={80}
                            status={getScoreStatus(metric.score)}
                          />
                          <div style={{ marginTop: '5px', fontSize: '12px' }}>
                            Score
                          </div>
                        </div>
                      </List.Item>
                    );
                  }}
                  locale={{ emptyText: 'No anomalous metrics found' }}
                />
              </Card>
            </Col>
            
            <Col xs={24} lg={8}>
              <Row gutter={[0, 16]}>
                <Col xs={24}>
                  <Card 
                    title="MetricRooter Analysis Summary" 
                    className="chart-card"
                  >
                    <div className="summary-content">
                      <p style={{ fontSize: '14px', lineHeight: '1.8' }}>
                        {metricRankings.summary || 'No detailed analysis available.'}
                      </p>
                      
                      {metricRankings.metrics && metricRankings.metrics.length > 0 && (
                        <div className="category-distribution metric-chart-container">
                          <h4>Category Distribution</h4>
                          <ResponsiveContainer width="100%" height={200}>
                            <PieChart>
                              <Pie
                                data={getCategoryDistribution(metricRankings.metrics)}
                                cx="50%"
                                cy="50%"
                                labelLine={true}
                                label={renderCustomizedLabel}
                                outerRadius={80}
                                fill="#8884d8"
                                dataKey="value"
                              >
                                {getCategoryDistribution(metricRankings.metrics).map((entry, index) => (
                                  <Cell key={`cell-${index}`} fill={getCategoryColorHex(entry.name)} />
                                ))}
                              </Pie>
                              <Tooltip formatter={(value, name) => [`${value}%`, name]} />
                              <Legend />
                            </PieChart>
                          </ResponsiveContainer>
                        </div>
                      )}
                    </div>
                  </Card>
                </Col>
                <Col xs={24}>
                  <Card 
                    title="Recommended Actions" 
                    className="chart-card"
                    style={{ background: "#f0f7ff" }}
                  >
                    <ul className="action-list" style={{ paddingLeft: '20px', marginTop: '10px' }}>
                      {getRecommendedActions(metricRankings.metrics).map((action, index) => (
                        <li key={index} className="action-item" style={{ marginBottom: '10px' }}>
                          <Text>{action}</Text>
                        </li>
                      ))}
                    </ul>
                  </Card>
                </Col>
              </Row>
            </Col>
          </Row>
        </div>
      )}
      
      <style jsx>{`
        .metric-item {
          transition: all 0.3s ease;
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
          border: 1px solid #f0f0f0;
        }
        .metric-item:hover {
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
          transform: translateY(-2px);
        }
        .metric-score {
          animation: fadeIn 0.5s ease-out;
        }
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        .action-item {
          padding: 8px 0;
          border-bottom: 1px solid #f0f0f0;
        }
        .action-item:last-child {
          border-bottom: none;
        }
        .metric-chart-container {
          height: 200px;
          margin-top: 20px;
        }
      `}</style>
    </div>
  );
};

export default MetricRooterPanel; 