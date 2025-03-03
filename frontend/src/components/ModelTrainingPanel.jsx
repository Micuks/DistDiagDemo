import React, { useState, useEffect } from "react";
import {
  Card,
  Button,
  Space,
  Row,
  Col,
  Switch,
  Divider,
  Statistic,
  Progress,
  Alert,
  Spin,
  message,
  Checkbox,
  Select,
} from "antd";
import { anomalyService } from "../services/anomalyService";
import { useAnomalyData } from "../hooks/useAnomalyData";
import ModelPerformanceView from "./ModelPerformanceView";

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

  // Fetch available models
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const models = await anomalyService.getAvailableModels();
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
        const status = await anomalyService.getCollectionStatus();
        setCollectionStatus({
          isCollecting: status.isCollecting || false,
          currentType: status.currentType || null,
        });
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
    try {
      setLoading(true);
      if (collectionStatus.isCollecting) {
        // Stop collection based on current type
        if (collectionStatus.currentType === "normal") {
          await anomalyService.stopNormalCollection();
        } else {
          // We're no longer using postCollect option
          await anomalyService.stopAnomalyCollection(true);
        }
      } else {
        // Start collection based on presence of anomalies
        if (activeAnomalies.length > 0) {
          const activeAnomaly = activeAnomalies[0];
          // Start collection for existing anomaly without injecting new one
          await anomalyService.startAnomalyCollection(
            activeAnomaly.type,
            activeAnomaly.node
          );
        } else {
          await anomalyService.startNormalCollection();
        }
      }

      // Refresh status after toggle
      const newStatus = await anomalyService.getCollectionStatus();
      setCollectionStatus({
        isCollecting: newStatus.isCollecting || false,
        currentType: newStatus.currentType || null,
      });
      message.success(
        `${newStatus.isCollecting ? "Started" : "Stopped"} data collection`
      );
    } catch (error) {
      message.error("Failed to toggle data collection");
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const handleAutoBalance = async () => {
    try {
      setLoading(true);
      await anomalyService.toggleAutoBalance(!isAutoBalancing);
      setIsAutoBalancing(!isAutoBalancing);
      message.success(
        `${!isAutoBalancing ? "Enabled" : "Disabled"} auto-balancing`
      );
    } catch (error) {
      message.error("Failed to toggle auto-balancing");
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const fetchTrainingStats = async () => {
    try {
      const response = await anomalyService.getTrainingStats();
      setTrainingStats(
        response.stats || {
          normal: 0,
          anomaly: 0,
          total_samples: 0,
          normal_ratio: 0,
          anomaly_ratio: 0,
          anomaly_types: {},
          is_balanced: false,
        }
      );
    } catch (error) {
      console.error("Failed to fetch training stats:", error);
      setTrainingStats({
        normal: 0,
        anomaly: 0,
        total_samples: 0,
        normal_ratio: 0,
        anomaly_ratio: 0,
        anomaly_types: {},
        is_balanced: false,
      });
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
                  {" "}
                  {availableModels.map((model) => (
                    <Select.Option key={model} value={model}>
                      {" "}
                      {model.replace(/\.[^/.]+$/, "")}{" "}
                    </Select.Option>
                  ))}{" "}
                </Select>{" "}
                {selectedModel && (
                  <ModelPerformanceView modelName={selectedModel} />
                )}{" "}
              </Space>{" "}
            </Col>{" "}
          </Row>{" "}
          <Divider />
          <Card title="Training Data Collection">
            <Space direction="vertical" style={{ width: "100%" }}>
              <div>
                <Switch
                  checked={collectionStatus.isCollecting}
                  onChange={handleCollectionToggle}
                  loading={loading}
                  disabled={loading}
                />{" "}
                <span style={{ marginLeft: 8 }}>
                  {" "}
                  {collectionStatus.isCollecting
                    ? `Collecting ${collectionStatus.currentType} data`
                    : "Start data collection"}{" "}
                </span>{" "}
              </div>{" "}
              <div>
                <Switch
                  checked={isAutoBalancing}
                  onChange={handleAutoBalance}
                  loading={loading}
                />{" "}
                <span style={{ marginLeft: 8 }}>
                  Auto - balance Training Data{" "}
                </span>{" "}
              </div>{" "}
            </Space>{" "}
          </Card>{" "}
        </Col>{" "}
      </Row>{" "}
      <Divider />
      <Row gutter={[16, 16]}>
        <Col span={24}>
          <Card title="Training Statistics">
            {" "}
            {trainingStats ? (
              <>
                <Row gutter={[16, 16]}>
                  <Col span={8}>
                    <Statistic
                      title="Normal Samples"
                      value={trainingStats.normal || 0}
                    />{" "}
                  </Col>{" "}
                  <Col span={8}>
                    <Statistic
                      title="Anomaly Samples"
                      value={trainingStats.anomaly || 0}
                    />{" "}
                  </Col>{" "}
                  <Col span={8}>
                    <Statistic
                      title="Total Samples"
                      value={trainingStats.total_samples || 0}
                    />{" "}
                  </Col>{" "}
                </Row>{" "}
                <Divider />
                <Progress
                  percent={Math.round((trainingStats.normal_ratio || 0) * 100)}
                  success={{
                    percent: Math.round(
                      (trainingStats.anomaly_ratio || 0) * 100
                    ),
                  }}
                  format={() =>
                    `${Math.round(
                      (trainingStats.normal_ratio || 0) * 100
                    )}% Normal / ${Math.round(
                      (trainingStats.anomaly_ratio || 0) * 100
                    )}% Anomaly`
                  }
                />{" "}
                <Divider />
                <Row gutter={[16, 16]}>
                  {" "}
                  {trainingStats.anomaly_types &&
                    Object.entries(trainingStats.anomaly_types).map(
                      ([type, count]) => (
                        <Col span={6} key={type}>
                          <Statistic
                            title={`${
                              type.charAt(0).toUpperCase() + type.slice(1)
                            } Anomalies`}
                            value={count}
                          />{" "}
                        </Col>
                      )
                    )}{" "}
                  {(!trainingStats.anomaly_types ||
                    Object.keys(trainingStats.anomaly_types || {}).length ===
                      0) && (
                    <Col span={24}>
                      <Alert
                        message="No anomaly types collected yet"
                        type="info"
                        showIcon
                      />
                    </Col>
                  )}{" "}
                </Row>{" "}
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
                )}{" "}
              </>
            ) : (
              <Spin tip="Loading training statistics..." />
            )}{" "}
          </Card>{" "}
        </Col>{" "}
      </Row>{" "}
    </div>
  );
};

export default ModelTrainingPanel;
