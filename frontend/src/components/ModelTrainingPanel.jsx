import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Card, Button, Space, Row, Col, Switch, Divider, Statistic, Progress, Alert, Spin, message, Checkbox, Select, Tabs } from 'antd';
import { DatabaseOutlined, ApiOutlined, ExperimentOutlined, LineChartOutlined, InfoCircleOutlined } from '@ant-design/icons';
import { trainingService } from '../services/trainingService';
import { useAnomalyData } from '../hooks/useAnomalyData';
import ModelPerformanceView from './ModelPerformanceView';
import ModelTrainingProgress from './ModelTrainingProgress';

const { TabPane } = Tabs;

const ModelTrainingPanel = () => {
  const [loading, setLoading] = useState(false);
  const [collectionStatus, setCollectionStatus] = useState({
    isCollecting: false,
    currentType: null,
  });
  const [trainingStats, setTrainingStats] = useState(null);
  const [isAutoBalancing, setIsAutoBalancing] = useState(false);
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);
  const { data: activeAnomalies = [] } = useAnomalyData();
  const [isTraining, setIsTraining] = useState(false);
  const [isCollectionToggling, setIsCollectionToggling] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState({ stage: 'idle', progress: 0 });
  const [activeTab, setActiveTab] = useState('1');
  const [trainingCompletionAcknowledged, setTrainingCompletionAcknowledged] = useState(false);
  
  const fetchStatusIntervalRef = useRef(null);
  const fetchStatsIntervalRef = useRef(null);
  const fetchTrainingStatusIntervalRef = useRef(null);

  useEffect(() => {
    fetchModels();
    
    return () => {
      if (fetchStatusIntervalRef.current) clearInterval(fetchStatusIntervalRef.current);
      if (fetchStatsIntervalRef.current) clearInterval(fetchStatsIntervalRef.current);
      if (fetchTrainingStatusIntervalRef.current) clearInterval(fetchTrainingStatusIntervalRef.current);
    };
  }, []);

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const status = await trainingService.getCollectionStatus();
        setCollectionStatus({
          isCollecting: status.is_collecting_normal || status.is_collecting_anomaly || false,
          currentType: status.current_type || null,
        });
      } catch (error) {
        console.error("Failed to fetch collection status:", error);
      }
    };

    fetchStatus();
    
    if (fetchStatusIntervalRef.current) clearInterval(fetchStatusIntervalRef.current);
    fetchStatusIntervalRef.current = setInterval(fetchStatus, 5000);
    
    return () => {
      if (fetchStatusIntervalRef.current) clearInterval(fetchStatusIntervalRef.current);
    };
  }, []);

  useEffect(() => {
    fetchTrainingStats();
    
    if (fetchStatsIntervalRef.current) clearInterval(fetchStatsIntervalRef.current);
    fetchStatsIntervalRef.current = setInterval(fetchTrainingStats, 5000);
    
    return () => {
      if (fetchStatsIntervalRef.current) clearInterval(fetchStatsIntervalRef.current);
    };
  }, []);

  useEffect(() => {
    const fetchTrainingStatus = async () => {
      try {
        const status = await trainingService.getTrainingStatus();
        setTrainingStatus(status);
        
        if (status.stage !== 'idle' && status.stage !== 'completed' && status.stage !== 'failed') {
          setActiveTab('2');
          setIsTraining(true);
          setTrainingCompletionAcknowledged(false);
        } else if (status.stage === 'completed' || status.stage === 'failed') {
          setIsTraining(false);
          
          if (status.stage === 'completed') {
            if (!trainingCompletionAcknowledged) {
              setTrainingCompletionAcknowledged(true);
              message.success("Model training completed successfully!");
              fetchModels();
              fetchTrainingStats();
            }
            
            if (status.stats && Object.keys(status.stats).length > 0) {
              setActiveTab('3');
            }
          }
        }
      } catch (error) {
        console.error("Failed to fetch training status:", error);
      }
    };

    fetchTrainingStatus();
    
    if (fetchTrainingStatusIntervalRef.current) clearInterval(fetchTrainingStatusIntervalRef.current);
    fetchTrainingStatusIntervalRef.current = setInterval(fetchTrainingStatus, 3000);
    
    return () => {
      if (fetchTrainingStatusIntervalRef.current) clearInterval(fetchTrainingStatusIntervalRef.current);
    };
  },[]);

  const fetchModels = async () => {
    try {
      const models = await trainingService.getAvailableModels();
      setAvailableModels(models);
      if (models.length > 0) {
        setSelectedModel(models[0]);
      }
    } catch (error) {
      console.error("Failed to fetch models:", error);
    }
  };

  const handleCollectionToggle = async () => {
    if (isCollectionToggling) return;
    
    try {
      setIsCollectionToggling(true);
      setLoading(true);
      
      let response;
      
      if (collectionStatus.isCollecting) {
        if (collectionStatus.currentType === "normal") {
          response = await trainingService.stopNormalCollection();
        } else {
          response = await trainingService.stopAnomalyCollection(true);
        }
        
        if (response && response.status === 'pending') {
          message.warning(response.message || "No collection in progress to stop");
        } else {
          message.success("Successfully stopped data collection");
        }
      } else {
        if (activeAnomalies.length > 0) {
          const activeAnomaly = activeAnomalies[0];
          response = await trainingService.startAnomalyCollection(
            activeAnomaly.type,
            activeAnomaly.node
          );
          
          if (response && response.status === 'pending') {
            message.warning(response.message || "Collection already in progress");
          } else {
            message.success(`Started data collection for ${activeAnomaly.type}`);
          }
        } else {
          response = await trainingService.startNormalCollection();
          
          if (response && response.status === 'pending') {
            message.warning(response.message || "Normal collection already in progress");
          } else {
            message.success("Started normal data collection");
          }
        }
      }

      const newStatus = await trainingService.getCollectionStatus();
      setCollectionStatus({
        isCollecting: newStatus.is_collecting_normal || newStatus.is_collecting_anomaly || false,
        currentType: newStatus.current_type || null,
      });
    } catch (error) {
      message.error("Failed to toggle data collection");
      console.error(error);
    } finally {
      setLoading(false);
      setIsCollectionToggling(false);
    }
  };

  const handleAutoBalance = async () => {
    if (isAutoBalancing) return;
    
    try {
      setIsAutoBalancing(true);
      setLoading(true);
      
      message.info('Auto-balancing dataset...');
      const response = await trainingService.autoBalanceDataset();
      
      if (response && response.success) {
        message.success('Dataset balanced successfully');
        fetchTrainingStats();
      } else {
        message.error('Failed to balance dataset');
      }
    } catch (error) {
      message.error('Failed to balance dataset');
      console.error(error);
    } finally {
      setLoading(false);
      setIsAutoBalancing(false);
    }
  };

  const fetchTrainingStats = useCallback(async () => {
    try {
      const response = await trainingService.getTrainingStats();
      if (response && response.status === 'success' && response.stats) {
        setTrainingStats(response.stats);
      } else {
      }
    } catch (error) {
      console.error("Error loading training stats", error);
    }
  }, []);

  const handleTrainModel = async () => {
    if (isTraining || trainingStatus.stage !== 'idle') return;
    
    try {
      setIsTraining(true);
      setLoading(true);
      message.info('Initiating model training process...');
      
      setActiveTab('2');
      
      const response = await trainingService.trainModel();
      
      if (response && response.status === 'pending') {
        message.warning(response.message || 'Training is already in progress');
      }
    } catch (error) {
      console.error("Error training model", error);
      message.error(`Failed to start model training: ${error.message}`);
      setIsTraining(false);
    } finally {
      setLoading(false);
    }
  };

  const renderDataCollectionTab = () => {
    return (
      <Row gutter={[16, 16]}>
        <Col span={24}>
          <Card title="Training Data Collection" 
                extra={
                  <Space>
                    <Button 
                      type="primary" 
                      onClick={handleTrainModel}
                      loading={isTraining}
                      disabled={!trainingStats || trainingStats.total_samples < 10 || isTraining}
                    >
                      Train Model
                    </Button>
                  </Space>
                }>
            <Space direction="vertical" style={{ width: "100%" }}>
              <Row gutter={[16, 16]} align="middle">
                <Col span={16}>
                  <Space>
                    <Switch
                      checked={collectionStatus.isCollecting}
                      onChange={handleCollectionToggle}
                      loading={loading}
                      disabled={loading || isTraining}
                    />
                    <span>
                      {collectionStatus.isCollecting
                        ? `Collecting ${collectionStatus.currentType} data`
                        : 'Start data collection'}
                    </span>
                  </Space>
                </Col>
                <Col span={8} style={{ textAlign: 'right' }}>
                  <Space>
                    <Switch
                      checked={isAutoBalancing}
                      onChange={handleAutoBalance}
                      loading={loading}
                      disabled={loading || isTraining}
                    />
                    <span>Auto-balance</span>
                  </Space>
                </Col>
              </Row>
              
              {activeAnomalies.length > 0 && !collectionStatus.isCollecting && (
                <Alert
                  message={`Active anomaly detected: ${activeAnomalies[0].type}`}
                  description="You can collect training data for this anomaly type."
                  type="info"
                  showIcon
                  style={{ marginTop: 16 }}
                />
              )}
              
              {(!trainingStats || trainingStats.total_samples < 10) && (
                <Alert
                  message="Insufficient Training Data"
                  description="You need at least 10 samples to train a model."
                  type="warning"
                  showIcon
                  style={{ marginTop: 16 }}
                />
              )}
            </Space>
          </Card>
        </Col>
        
        <Col span={24}>
          <Card title="Training Statistics">
            {trainingStats ? (
              <>
                <Row gutter={[16, 16]}>
                  <Col span={8}>
                    <Statistic
                      title="Normal Samples"
                      value={trainingStats.normal || 0}
                      valueStyle={{ color: '#3f8600' }}
                      prefix={<DatabaseOutlined />}
                    />
                  </Col>
                  <Col span={8}>
                    <Statistic
                      title="Anomaly Samples"
                      value={trainingStats.anomaly || 0}
                      valueStyle={{ color: '#cf1322' }}
                      prefix={<ExperimentOutlined />}
                    />
                  </Col>
                  <Col span={8}>
                    <Statistic
                      title="Total Samples"
                      value={trainingStats.total_samples || 0}
                      prefix={<DatabaseOutlined />}
                    />
                  </Col>
                </Row>
                <Divider />
                <Progress
                  percent={Math.round((trainingStats.normal_ratio || 0) * 100)}
                  success={{ percent: Math.round((trainingStats.anomaly_ratio || 0) * 100) }}
                  format={() => `${Math.round((trainingStats.normal_ratio || 0) * 100)}% Normal / ${Math.round((trainingStats.anomaly_ratio || 0) * 100)}% Anomaly`}
                />
                <Divider />
                <Row gutter={[16, 16]}>
                  {trainingStats?.anomaly_types && Object.entries(trainingStats.anomaly_types).map(([type, count]) => (
                    <Col span={6} key={type}>
                      <Statistic
                        title={`${type.charAt(0).toUpperCase() + type.slice(1)} Anomalies`}
                        value={count}
                        valueStyle={{ color: '#1890ff' }}
                      />
                    </Col>
                  ))}
                  {(!trainingStats?.anomaly_types || Object.keys(trainingStats?.anomaly_types || {}).length === 0) && (
                    <Col span={24}>
                      <Alert
                        message="No anomaly types collected yet"
                        type="info"
                        showIcon
                      />
                    </Col>
                  )}
                </Row>
                {trainingStats.is_balanced ? (
                  <Alert
                    message="Data is well-balanced"
                    type="success"
                    showIcon
                    style={{ marginTop: 16 }}
                  />
                ) : (
                  <Alert
                    message="Data is imbalanced"
                    description="Consider enabling auto-balance or collecting more data"
                    type="warning"
                    showIcon
                    style={{ marginTop: 16 }}
                  />
                )}
              </>
            ) : (
              <Spin tip="Loading training statistics..." />
            )}
          </Card>
        </Col>
      </Row>
    );
  };

  const renderTrainingProgressTab = () => {
    return (
      <Row gutter={[16, 16]}>
        <Col span={24}>
          <ModelTrainingProgress status={trainingStatus} />
        </Col>
      </Row>
    );
  };

  const renderModelPerformanceTab = () => {
    return (
      <Row gutter={[16, 16]}>
        <Col span={24}>
          <Card 
            title={
              <div style={{ display: 'flex', alignItems: 'center' }}>
                <LineChartOutlined style={{ fontSize: 20, marginRight: 8 }} />
                <span>Model Performance Analysis</span>
              </div>
            }
            bordered={false}
            className="model-selection-card"
          >
            <Row gutter={[16, 16]}>
              <Col span={24} style={{ marginBottom: 16 }}>
                <Select
                  style={{ width: "100%" }}
                  placeholder="Select model to view performance"
                  value={selectedModel}
                  onChange={setSelectedModel}
                  optionLabelProp="label"
                  size="large"
                  bordered
                  dropdownStyle={{ maxHeight: 400 }}
                >
                  {availableModels.map(model => (
                    <Select.Option key={model} value={model} label={model.replace(/\.[^/.]+$/, "")}>
                      <div style={{ display: 'flex', alignItems: 'center' }}>
                        <LineChartOutlined style={{ marginRight: 8 }} />
                        {model.replace(/\.[^/.]+$/, "")}
                      </div>
                    </Select.Option>
                  ))}
                </Select>
              </Col>
              
              <Col span={24}>
                {selectedModel ? (
                  <ModelPerformanceView modelName={selectedModel} />
                ) : (
                  <Alert 
                    message="No Model Selected" 
                    description="Please select a model to view its performance metrics."
                    type="info" 
                    showIcon 
                  />
                )}
              </Col>
            </Row>
          </Card>
        </Col>
      </Row>
    );
  };

  return (
    <>
      <Alert
        message="Optional Training Module"
        description="This module is for training and improving models but is not required during anomaly diagnosis. You can skip this step in the workflow."
        type="info"
        showIcon
        icon={<InfoCircleOutlined />}
        style={{ marginBottom: 16 }}
      />
      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane 
          tab={<span><DatabaseOutlined />Data Collection</span>} 
          key="1"
        >
          {renderDataCollectionTab()}
        </TabPane>
        <TabPane 
          tab={<span><ApiOutlined />Training Process</span>} 
          key="2"
        >
          {renderTrainingProgressTab()}
        </TabPane>
        <TabPane 
          tab={<span><LineChartOutlined />Model Performance</span>} 
          key="3"
        >
          {renderModelPerformanceTab()}
        </TabPane>
      </Tabs>
    </>
  );
};

export default ModelTrainingPanel;
