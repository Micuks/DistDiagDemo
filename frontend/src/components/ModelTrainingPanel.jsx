import React, { useState, useEffect } from 'react';
import { Card, Button, Space, Row, Col, Switch, Divider, Statistic, Progress, Alert, Spin, message, Checkbox, Select } from 'antd';
import { trainingService } from '../services/trainingService';
import { useAnomalyData } from '../hooks/useAnomalyData';
import ModelPerformanceView from './ModelPerformanceView';

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

  // Fetch available models
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const models = await trainingService.getAvailableModels();
        setAvailableModels(models);
        if (models.length > 0) {
          setSelectedModel(models[0]);
        }
      } catch (error) {
        console.error("Failed to fetch models:", error);
        message.error("Failed to load available models");
      }
    };
    fetchModels();
  }, []);

  // Fetch collection status periodically
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const status = await trainingService.getCollectionStatus();
        // Update the isCollecting property based on either normal or anomaly collection
        setCollectionStatus({
          isCollecting: status.is_collecting_normal || status.is_collecting_anomaly || false,
          currentType: status.current_type || null,
        });
        
        console.log("Collection status updated:", status);
      } catch (error) {
        console.error("Failed to fetch collection status:", error);
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  // Fetch training stats periodically
  useEffect(() => {
    fetchTrainingStats();
    const interval = setInterval(fetchTrainingStats, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleCollectionToggle = async () => {
    if (isCollectionToggling) return;
    
    try {
      setIsCollectionToggling(true);
      setLoading(true);
      
      let response;
      
      if (collectionStatus.isCollecting) {
        // Stop collection based on current type
        if (collectionStatus.currentType === "normal") {
          response = await trainingService.stopNormalCollection();
        } else {
          // We're no longer using postCollect option
          response = await trainingService.stopAnomalyCollection(true);
        }
        
        if (response && response.status === 'pending') {
          message.warning(response.message || "No collection in progress to stop");
        } else {
          message.success("Successfully stopped data collection");
        }
      } else {
        // Start collection based on presence of anomalies
        if (activeAnomalies.length > 0) {
          const activeAnomaly = activeAnomalies[0];
          // Start collection for existing anomaly without injecting new one
          response = await trainingService.startAnomalyCollection(
            activeAnomaly.type,
            activeAnomaly.node
          );
          
          if (response && response.status === 'pending') {
            message.warning(response.message || "Collection already in progress");
          } else {
            console.log("Start anomaly collection response:", response);
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

      // Refresh status after toggle
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

    const fetchTrainingStats = async () => {
        try {
            const response = await trainingService.getTrainingStats();
            if (response && response.status === 'success' && response.stats) {
                setTrainingStats(response.stats);
            } else {
                setTrainingStats(null);
            }
        } catch (error) {
            console.error("Error loading training stats", error);
            setTrainingStats(null);
        }
    };

    const handleTrainModel = async () => {
        // Don't allow multiple simultaneous requests
        if (isTraining) return;
        
        try {
            setIsTraining(true);
            setLoading(true);
            message.info('Training model with collected data...');
            
            const response = await trainingService.trainModel();
            
            if (response && response.status === 'success') {
                message.success('Model trained successfully');
                // Refresh the list of models
                const models = await trainingService.getAvailableModels();
                setAvailableModels(models);
                if (models.length > 0) {
                    setSelectedModel(models[0].name);
                }
            } else if (response && response.status === 'pending') {
                // Handle the case where training is already in progress
                message.warning(response.message || 'Training is already in progress');
            } else {
                message.error('Failed to train model');
            }
        } catch (error) {
            console.error("Error training model", error);
            message.error(`Failed to train model: ${error.message}`);
        } finally {
            setLoading(false);
            setIsTraining(false);
        }
    };

  return (
    <div>
      <Row gutter={[16, 16]}>
        <Col span={24}>
          <Row gutter={[16, 16]}>
            <Col span={24}>
              <Space direction="vertical" style={{ width: "100%" }}>
                <Select
                  style={{ width: "100%" }}
                  placeholder="Select model to view performance"
                  value={selectedModel}
                  onChange={setSelectedModel}
                >
                  {availableModels.map(model => (
                    <Select.Option key={model} value={model}>
                      {model.replace(/\.[^/.]+$/, "")}
                    </Select.Option>
                  ))}
                </Select>
                {/*{selectedModel && (
                  <ModelPerformanceView modelName={selectedModel} />
                )}*/}
              </Space>
            </Col>
          </Row>
          <Divider />
          <Card title="Training Data Collection">
            <Space direction="vertical" style={{ width: "100%" }}>
              <div>
                <Switch
                  checked={collectionStatus.isCollecting}
                  onChange={handleCollectionToggle}
                  loading={loading}
                  disabled={loading}
                />
                <span style={{ marginLeft: 8 }}>
                  {collectionStatus.isCollecting
                    ? `Collecting ${collectionStatus.currentType} data`
                    : 'Start data collection'}
                </span>
              </div>
              {activeAnomalies.length > 0 && !collectionStatus.isCollecting && (
                <div style={{ marginLeft: 32 }}>
                </div>
              )}
              <div>
                <Switch
                  checked={isAutoBalancing}
                  onChange={handleAutoBalance}
                  loading={loading}
                />
                <span style={{ marginLeft: 8 }}>Auto-balance Training Data</span>
              </div>
              <div style={{ marginTop: 16 }}>
                <Button 
                  type="primary" 
                  onClick={handleTrainModel}
                  loading={loading}
                  disabled={!trainingStats || trainingStats.total_samples < 10}
                >
                  Train Model
                </Button>
                {(!trainingStats || trainingStats.total_samples < 10) && (
                  <span style={{ marginLeft: 8, color: 'rgba(0, 0, 0, 0.45)' }}>
                    Need at least 10 samples
                  </span>
                )}
              </div>
            </Space>
          </Card>
        </Col>
      </Row>

      <Divider />
      <Row gutter={[16, 16]}>
        <Col span={24}>
          <Card title="Training Statistics">
            {trainingStats ? (
              <>
                <Row gutter={[16, 16]}>
                  <Col span={8}>
                    <Statistic
                      title="Normal Samples"
                      value={trainingStats.normal || 0}
                    />
                  </Col>
                  <Col span={8}>
                    <Statistic
                      title="Anomaly Samples"
                      value={trainingStats.anomaly || 0}
                    />
                  </Col>
                  <Col span={8}>
                    <Statistic
                      title="Total Samples"
                      value={trainingStats.total_samples || 0}
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
    </div>
  );
};

export default ModelTrainingPanel;
