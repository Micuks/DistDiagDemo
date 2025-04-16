import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Card, Button, Space, Row, Col, Switch, Divider, Statistic, Progress, Alert, Spin, message, Checkbox, Select, Tabs, Timeline, Typography, Modal, Form } from 'antd';
import { DatabaseOutlined, ApiOutlined, ExperimentOutlined, LineChartOutlined, InfoCircleOutlined, SyncOutlined, RiseOutlined, FallOutlined, PlusCircleOutlined } from '@ant-design/icons';
import { trainingService } from '../services/trainingService';
import { useAnomaly } from '../hooks/useAnomaly';
import ModelPerformanceView from './ModelPerformanceView';
import ModelTrainingProgress from './ModelTrainingProgress';

const ModelAdaptationInfo = ({ activeAnomalies, trainingStats, isAutoMonitoring }) => {
  // TODO: Drop the feature
  const [adaptationHistory, setAdaptationHistory] = useState([
    { time: new Date().toISOString(), event: 'System initialized', type: 'info' }
  ]);

  // Add adaptation events to history
  useEffect(() => {
    const now = new Date().toISOString();
    
    // When auto-monitoring is toggled
    if (isAutoMonitoring) {
      setAdaptationHistory(prev => [
        { time: now, event: 'Automatic adaptation enabled', type: 'success' },
        ...prev
      ]);
    }
    
    // When anomalies are detected
    if (activeAnomalies?.length > 0) {
      const anomalyType = activeAnomalies[0].type;
      
      // Check if this anomaly was already logged recently (avoid duplicates)
      const recentEntries = adaptationHistory.slice(0, 3);
      const isDuplicate = recentEntries.some(entry => 
        entry.event.includes(anomalyType) && entry.type === 'warning'
      );
      
      if (!isDuplicate) {
        setAdaptationHistory(prev => [
          { time: now, event: `New anomaly pattern detected: ${anomalyType}`, type: 'warning' },
          ...prev
        ]);
      }
    }
  }, [isAutoMonitoring, activeAnomalies]);
  
  // Add training stats changes to history
  useEffect(() => {
    if (!trainingStats) return;
    
    const now = new Date().toISOString();
    
    // Check for significant changes in class balance
    if (trainingStats.is_balanced && adaptationHistory[0]?.event !== 'Dataset balanced successfully') {
      setAdaptationHistory(prev => [
        { time: now, event: 'Dataset balanced successfully', type: 'success' },
        ...prev
      ]);
    }
    
    // Add new anomaly types when detected
    if (trainingStats.anomaly_types) {
      const anomalyTypes = Object.keys(trainingStats.anomaly_types);
      
      // Get previously seen types from recent history
      const recentEvents = adaptationHistory.slice(0, 10);
      const recentTypes = new Set();
      recentEvents.forEach(event => {
        const match = event.event.match(/New anomaly type collected: (\w+)/);
        if (match) recentTypes.add(match[1]);
      });
      
      // Add new types that weren't seen recently
      anomalyTypes.forEach(type => {
        if (!recentTypes.has(type) && trainingStats.anomaly_types[type] > 0) {
          setAdaptationHistory(prev => [
            { time: now, event: `New anomaly type collected: ${type}`, type: 'info' },
            ...prev
          ]);
        }
      });
    }
  }, [trainingStats]);
  
  // Limit history length
  useEffect(() => {
    if (adaptationHistory.length > 20) {
      setAdaptationHistory(prev => prev.slice(0, 20));
    }
  }, [adaptationHistory]);
  
  return (
    <></>
  )
  // return (
  //   <Card title={
  //     <span>
  //       <SyncOutlined spin={isAutoMonitoring} style={{ marginRight: 8 }} />
  //       Model Adaptation Status
  //     </span>
  //   }>
  //     <Space direction="vertical" style={{ width: '100%' }}>
  //       <Alert
  //         message="Continuous Learning System"
  //         description={
  //           <>
  //             <Typography.Paragraph>
  //               The system {isAutoMonitoring ? 'is actively monitoring' : 'can monitor'} for data drift and class imbalances, 
  //               automatically collecting samples to maintain model accuracy.
  //             </Typography.Paragraph>
  //             <Typography.Paragraph>
  //               <strong>Feature drift adaptation:</strong> Exponential smoothing (α=0.2) preserves critical feature distributions
  //               while adapting to emerging patterns.
  //             </Typography.Paragraph>
  //           </>
  //         }
  //         type="info"
  //         showIcon
  //       />
        
  //       <Divider>Adaptation Timeline</Divider>
        
  //       <Timeline
  //         mode="left"
  //         items={adaptationHistory.map(item => ({
  //           label: new Date(item.time).toLocaleTimeString(),
  //           color: item.type === 'success' ? 'green' : item.type === 'warning' ? 'orange' : 'blue',
  //           children: item.event
  //         }))}
  //       />
  //     </Space>
  //   </Card>
  // );
};

const ModelTrainingPanel = () => {
  const [loading, setLoading] = useState(false);
  const [collectionStatus, setCollectionStatus] = useState({
    isCollecting: false,
    currentType: null,
  });
  const [trainingStats, setTrainingStats] = useState(null);
  const [isAutoBalancing, setIsAutoBalancing] = useState(false);
  const [isAutoMonitoring, setIsAutoMonitoring] = useState(false);
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);
  const { activeAnomalies = [], isLoading: isLoadingAnomalies } = useAnomaly();
  const [isTraining, setIsTraining] = useState(false);
  const [isCollectionToggling, setIsCollectionToggling] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState({ stage: 'idle', progress: 0 });
  const [activeTab, setActiveTab] = useState('1');
  const [trainingCompletionAcknowledged, setTrainingCompletionAcknowledged] = useState(false);

  const fetchStatusIntervalRef = useRef(null);
  const fetchStatsIntervalRef = useRef(null);
  const fetchTrainingStatusIntervalRef = useRef(null);
  const autoMonitoringIntervalRef = useRef(null);

  useEffect(() => {
    fetchModels();
    
    return () => {
      if (fetchStatusIntervalRef.current) clearInterval(fetchStatusIntervalRef.current);
      if (fetchStatsIntervalRef.current) clearInterval(fetchStatsIntervalRef.current);
      if (fetchTrainingStatusIntervalRef.current) clearInterval(fetchTrainingStatusIntervalRef.current);
      if (autoMonitoringIntervalRef.current) clearInterval(autoMonitoringIntervalRef.current);
    };
  }, []);

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const status = await trainingService.getCollectionStatus();
        
        // Add additional validation to detect potential stale state
        if (status.is_collecting_normal || status.is_collecting_anomaly) {
          // Check last update time if provided by backend
          if (status.last_update_time) {
            const lastUpdateTime = new Date(status.last_update_time);
            const currentTime = new Date();
            const timeDifferenceMs = currentTime - lastUpdateTime;
            
            // If last update was more than 5 minutes ago, we might have a stale state
            if (timeDifferenceMs > 5 * 60 * 1000) {
              console.warn("Detected potentially stale collection state, last update:", lastUpdateTime);
              
              // Force refresh collection status by calling process-status endpoint
              const processStatus = await trainingService.getProcessStatus();
              if (!processStatus.anomaly_collection_in_progress && !processStatus.normal_collection_in_progress) {
                console.info("Process status indicates no active collection, resetting local state");
                setCollectionStatus({
                  isCollecting: false,
                  currentType: null,
                });
                return;
              }
            }
          }
        }
        
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

  useEffect(() => {
    if (isAutoMonitoring) {
      const checkAndBalanceDataset = async () => {
        try {
          const stats = await trainingService.getTrainingStats();
          
          if (stats && stats.status === 'success' && stats.stats) {
            if (!stats.stats.is_balanced && !collectionStatus.isCollecting) {
              message.info('Auto-monitoring detected imbalanced dataset, initiating balancing...');
              
              const anomalyTypes = stats.stats.anomaly_types || {};
              const threshold = stats.stats.total_samples * 0.1;
              const underrepresentedTypes = [];
              
              if (activeAnomalies.length > 0) {
                const activeType = activeAnomalies[0].type;
                const count = anomalyTypes[activeType] || 0;
                
                if (count < threshold) {
                  message.info(`Auto-collecting underrepresented active anomaly type: ${activeType}`);
                  await trainingService.startAnomalyCollection(activeType, activeAnomalies[0].node);
                  return;
                }
              }
              
              await trainingService.autoBalanceDataset();
            }
          }
        } catch (error) {
          console.error('Error in auto-monitoring:', error);
        }
      };
      
      checkAndBalanceDataset();
      autoMonitoringIntervalRef.current = setInterval(checkAndBalanceDataset, 30000);
    } else {
      if (autoMonitoringIntervalRef.current) {
        clearInterval(autoMonitoringIntervalRef.current);
        autoMonitoringIntervalRef.current = null;
      }
    }
    
    return () => {
      if (autoMonitoringIntervalRef.current) {
        clearInterval(autoMonitoringIntervalRef.current);
      }
    };
  }, [isAutoMonitoring, activeAnomalies, collectionStatus.isCollecting]);

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
        if (activeAnomalies.length > 1) {
          const types = activeAnomalies.map(a => a.type);
          response = await trainingService.startCompoundCollection(types, null);
          if (response && response.status === 'pending') {
            message.warning(response.message || "Collection already in progress");
          } else {
            message.success(`Started compound data collection for: ${types.join(', ')}`);
          }
        } else if (activeAnomalies.length === 1) {
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

      // Refresh status after action
      await new Promise(resolve => setTimeout(resolve, 1000));
      const newStatus = await trainingService.getCollectionStatus();
      const processStatus = await trainingService.getProcessStatus();
      const isReallyCollecting = processStatus.anomaly_collection_in_progress || processStatus.normal_collection_in_progress;

      setCollectionStatus({
        isCollecting: isReallyCollecting,
        currentType: isReallyCollecting ? newStatus.current_type : null,
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

  const handleAutoMonitoringToggle = () => {
    setIsAutoMonitoring(prev => !prev);
    message.info(isAutoMonitoring 
      ? 'Disabled automatic anomaly monitoring' 
      : 'Enabled automatic anomaly monitoring');
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
    const collectionButtonText = collectionStatus.isCollecting
      ? `Stop Collecting ${collectionStatus.currentType || ''} Data`
      : activeAnomalies.length > 1
      ? `Start Compound Collection (${activeAnomalies.map(a => a.type).join(' + ')})`
      : activeAnomalies.length === 1
      ? `Start Collection (${activeAnomalies[0].type})`
      : 'Start Normal Collection';

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
                      disabled={!trainingStats || trainingStats.total_samples < 10 || isTraining || collectionStatus.isCollecting}
                    >
                      Train Model
                    </Button>
                  </Space>
                }>
            <Space direction="vertical" style={{ width: "100%" }}>
              <Row gutter={[16, 16]} align="middle">
                <Col span={16}>
                  <Space>
                    <Button
                      type={collectionStatus.isCollecting ? "danger" : "primary"}
                      onClick={handleCollectionToggle}
                      loading={loading || isCollectionToggling}
                      disabled={loading || isTraining || isCollectionToggling}
                    >
                      {collectionButtonText}
                    </Button>
                  </Space>
                </Col>
                <Col span={8} style={{ textAlign: 'right' }}>
                  <Space>
                    <Switch
                      checked={isAutoBalancing}
                      onChange={handleAutoBalance}
                      loading={loading}
                      disabled={loading || isTraining || collectionStatus.isCollecting}
                    />
                    <span>Auto-balance</span>
                    <Divider type="vertical" />
                    <Switch
                      checked={isAutoMonitoring}
                      onChange={handleAutoMonitoringToggle}
                      loading={loading}
                      disabled={isTraining}
                    />
                    <span>Auto-monitor</span>
                  </Space>
                </Col>
              </Row>
              
              {activeAnomalies.length > 0 && !collectionStatus.isCollecting && (
                <Alert
                  message={
                    activeAnomalies.length > 1
                    ? `Multiple active anomalies detected: ${activeAnomalies.map(a => a.type).join(', ')}`
                    : `Active anomaly detected: ${activeAnomalies[0].type}`
                  }
                  description={collectionStatus.isCollecting ? "" : "Click the button above to start collecting relevant training data."}
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
          {/* <ModelAdaptationInfo 
            activeAnomalies={activeAnomalies}
            trainingStats={trainingStats}
            isAutoMonitoring={isAutoMonitoring}
          /> */}
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
                    description={
                      <div>
                        Consider enabling auto-balance or collecting more data
                        {isAutoMonitoring && (
                          <strong> - Automatic monitoring is active and will address this imbalance</strong>
                        )}
                      </div>
                    }
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
    <div>
      <Tabs 
        activeKey={activeTab} 
        onChange={setActiveTab}
        items={[
          {
            key: '1',
            label: 'Data Collection',
            icon: <DatabaseOutlined />,
            children: renderDataCollectionTab()
          },
          {
            key: '2',
            label: 'Training Progress',
            icon: <ApiOutlined />,
            children: renderTrainingProgressTab()
          },
          {
            key: '3',
            label: 'Model Performance',
            icon: <LineChartOutlined />,
            children: renderModelPerformanceTab()
          }
        ]}
      />
    </div>
  );
};

export default ModelTrainingPanel;
