import React, { useState, useEffect, useCallback, useMemo } from "react";
import { SyncOutlined, SearchOutlined, AlertOutlined, InfoCircleOutlined } from '@ant-design/icons';
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
  Radio,
  Checkbox,
  Slider,
  Statistic,
  Switch,
  message,
  Tooltip
} from "antd";
import { anomalyService } from "../services/anomalyService";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  LineChart,
  Line
} from "recharts";

const { Text, Title } = Typography;
const { Option } = Select;
const { TabPane } = Tabs;

// Helper function to determine color for rank types
const getRankColor = (type) => {
  if (!type) return "default";
  
  const lowerType = type.toLowerCase();
  switch (true) {
    case lowerType.includes('cpu'):
      return "red";
    case lowerType.includes('io'):
      return "orange";
    case lowerType.includes('network'):
      return "green";
    case lowerType.includes('cache'):
      return "blue";
    case lowerType.includes('index'):
      return "purple";
    default:
      return "default";
  }
};

// Add debounce utility at the top level
const debounce = (func, wait) => {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
};

const RanksPanel = () => {
  const [loading, setLoading] = useState(false);
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModels, setSelectedModels] = useState([]);
  const [comparisonLoading, setComparisonLoading] = useState(false);
  const [comparisonData, setComparisonData] = useState(null);
  const [comparisonError, setComparisonError] = useState(null);
  const [nodeData, setNodeData] = useState([]);
  const [activeTab, setActiveTab] = useState("1");
  const [autoAnalysisEnabled, setAutoAnalysisEnabled] = useState(true);
  const [lastAnalyzedAnomalies, setLastAnalyzedAnomalies] = useState([]);
  const [timeRange, setTimeRange] = useState("1h");
  const [filteredNodes, setFilteredNodes] = useState([]);
  const [selectedMetrics, setSelectedMetrics] = useState(["cpu", "memory", "io", "network"]);
  const [chartType, setChartType] = useState("radar");
  const [showConfidenceIntervals, setShowConfidenceIntervals] = useState(true);
  const [sortMethod, setSortMethod] = useState("confidence");
  const [thresholdValue, setThresholdValue] = useState(50);
  const [metricsFluctuations, setMetricsFluctuations] = useState({});
  const [hasFluctuations, setHasFluctuations] = useState(false);
  const [lastAnalysisTime, setLastAnalysisTime] = useState(0);
  const [fluctuationCooldown, setFluctuationCooldown] = useState(false);
  const [showRawData, setShowRawData] = useState(false);
  const [timelineView, setTimelineView] = useState('anomalies');
  const [selectedFeatureModel, setSelectedFeatureModel] = useState(null);
  const [lastRequestTime, setLastRequestTime] = useState(0);
  const [modelAnalysisTimes, setModelAnalysisTimes] = useState({});
  const [modelAdvantages, setModelAdvantages] = useState({});
  const [processingModels, setProcessingModels] = useState(new Set());
  const [rcaComparisonData, setRcaComparisonData] = useState([]);
  const cooldownPeriodMs = 30000; // 30 seconds cooldown
  const REQUEST_THROTTLE_MS = 5000; // Minimum 5 seconds between requests

  // Helper function for model colors - moved after state declarations
  const getModelColor = (model, index) => {
    const colors = [
      '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ];
    
    const modelIndex = selectedModels.indexOf(model);
    return colors[modelIndex % colors.length];
  };

  // Render function for root causes (handles both single and multiple anomalies)
  const renderRootCause = (model, node) => {
    if (!comparisonData || !comparisonData[model]) {
      return <Text type="secondary">No data</Text>;
    }
    
    if (comparisonData[model].processing) {
      return <Spin size="small" />;
    }

    // Check if there was an error
    if (comparisonData[model].error) {
      return <Text type="danger">Error: {comparisonData[model].error}</Text>;
    }
    
    // Try to find anomalies in different data structures
    let anomalies = [];
    
    // First check direct anomalies stored in comparison data
    if (comparisonData[model][`${model}_anomalies`]) {
      anomalies = comparisonData[model][`${model}_anomalies`];
    } 
    // Then check in the ranks array
    else if (comparisonData[model].ranks && Array.isArray(comparisonData[model].ranks)) {
      anomalies = comparisonData[model].ranks.filter(r => r.node === node);
    }
    
    // If we have no anomalies, check the response data directly
    if ((!anomalies || anomalies.length === 0) && node) {
      const modelData = comparisonData[model];
      
      if (modelData && modelData.ranks && Array.isArray(modelData.ranks)) {
        anomalies = modelData.ranks.filter(r => r.node === node);
      }
    }
    
    // Check for root cause in rcaComparisonData if anomalies not found
    if ((!anomalies || anomalies.length === 0) && rcaComparisonData) {
      const nodeData = rcaComparisonData.find(r => r.node === node);
      if (nodeData && nodeData[`${model}_root_cause`] && nodeData[`${model}_root_cause`] !== "Normal") {
        return (
          <Tag color={getRankColor(nodeData[`${model}_root_cause`])}>
            {nodeData[`${model}_root_cause`]}
          </Tag>
        );
      }
    }
    
    if (!anomalies || anomalies.length === 0) {
      return <Text type="secondary">Normal</Text>;
    }

    // Sort by score descending
    const sortedAnomalies = [...anomalies].sort((a, b) => (b.score || 0) - (a.score || 0));
    
    if (sortedAnomalies.length === 1) {
      // Single anomaly case
      const anomaly = sortedAnomalies[0];
      return (
        <Tag color={getRankColor(anomaly.type)}>
          {anomaly.type}
        </Tag>
      );
    } else {
      // Multiple anomalies case - show top 3 with scores
      return (
        <div>
          {sortedAnomalies.slice(0, 3).map((anomaly, idx) => (
            <div key={idx} style={{ marginBottom: idx < 2 ? 4 : 0 }}>
              <Tag color={getRankColor(anomaly.type)}>
                {anomaly.type} ({Math.round((anomaly.score || 0) * 100)}%)
              </Tag>
              {anomaly.related_metrics && anomaly.related_metrics.length > 0 && (
                <Tooltip title={
                  <div>
                    <div style={{ marginBottom: 5 }}>Related metrics:</div>
                    {anomaly.related_metrics.map((metric, i) => (
                      <div key={i} style={{ fontSize: 12 }}>
                        {metric.name}: {metric.value.toFixed(2)}
                      </div>
                    ))}
                  </div>
                }>
                  <InfoCircleOutlined style={{ marginLeft: 4, color: '#1890ff' }} />
                </Tooltip>
              )}
            </div>
          ))}
          {sortedAnomalies.length > 3 && (
            <div style={{ fontSize: 11, color: '#999', marginTop: 2 }}>
              +{sortedAnomalies.length - 3} more anomalies
            </div>
          )}
        </div>
      );
    }
  };

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
        const lowerType = type?.toLowerCase();
        switch (true) {
          case lowerType?.includes('cpu'):
            color = "red";
            break;
          case lowerType?.includes('io'):
            color = "orange";
            break;
          case lowerType?.includes('network'):
            color = "green";
            break;
          case lowerType?.includes('cache'):
            color = "blue";
            break;
          case lowerType?.includes('index'):
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

  const fetchAvailableModels = async () => {
    try {
      setLoading(true);
      const models = await anomalyService.getAvailableModels();
      // Filter out any models with null or undefined names
      const validModels = models.filter(model => model !== null && ((typeof model === 'string' && model !== '') || (typeof model === 'object' && model.name !== '')));
      setAvailableModels(validModels);

      // Default select the first model if available
      if (validModels.length > 0) {
        setSelectedModels([validModels[0]]);
        console.log(`Default model selected: ${validModels[0]}`);
      } else {
        console.warn("No valid models available");
        // Maybe try again after a delay if no models found
        setTimeout(() => {
          if (selectedModels.length === 0) {
            console.log("Retrying model fetch...");
            fetchAvailableModels();
          }
        }, 3000);
      }
    } catch (error) {
      console.error("Error fetching available models:", error);
    } finally {
      setLoading(false);
    }
  };

  // Add this useEffect to track when models change
  useEffect(() => {
    console.log("Selected models updated:", selectedModels);
  }, [selectedModels]);

  // Helper function to determine model advantages
  const getModelAdvantages = (model, response, analysisTime) => {
    const advantages = [];
    
    // Speed comparison (DBSherlock baseline ~5000ms)
    if (analysisTime < 3000) {
      advantages.push({
        type: 'speed',
        icon: '🚀',
        text: `${Math.round(analysisTime/100)/10}s response (${Math.round(5000/analysisTime)}x faster than DBSherlock)`
      });
    }
    
    // Accuracy comparison (XGBoost baseline ~82%)
    if (response.rca_precision > 0.85) {
      advantages.push({
        type: 'accuracy',
        icon: '🎯',
        text: `${Math.round(response.rca_precision * 100)}% precision (${Math.round((response.rca_precision * 100) - 82)}% higher than XGBoost)`
      });
    }
    
    return advantages;
  };

  // Get a description of compound anomalies - summarizing related metrics
  const getCompoundAnomalyDescription = (anomalies) => {
    if (!anomalies || anomalies.length <= 1) return null;
    
    // Extract all related metrics from all anomalies
    const allRelatedMetrics = [];
    anomalies.forEach(anomaly => {
      if (anomaly.related_metrics && anomaly.related_metrics.length > 0) {
        anomaly.related_metrics.forEach(metric => {
          allRelatedMetrics.push({
            ...metric,
            anomaly_type: anomaly.type,
            score: anomaly.score
          });
        });
      }
    });
    
    // Group metrics by category
    const metricsByCategory = {};
    allRelatedMetrics.forEach(metric => {
      if (!metricsByCategory[metric.category]) {
        metricsByCategory[metric.category] = [];
      }
      metricsByCategory[metric.category].push(metric);
    });
    
    // Create summary of top metrics by category
    const summaries = [];
    Object.entries(metricsByCategory).forEach(([category, metrics]) => {
      // Sort by value (descending)
      metrics.sort((a, b) => b.value - a.value);
      
      // Take top 2 metrics per category
      const topMetrics = metrics.slice(0, 2);
      
      if (topMetrics.length > 0) {
        summaries.push({
          category,
          metrics: topMetrics
        });
      }
    });
    
    return summaries;
  };
  
  // Update references to use the memoized columns
  const updateRcaComparison = (model, response, analysisTime) => {
    // Update RCA data incrementally
    setRcaComparisonData(prevData => {
      const nodes = new Set(prevData?.map(d => d.node) || []);
      
      // Add new nodes from current model response
      if (response?.ranks) {
        response.ranks.forEach(rank => {
          if (rank?.node) nodes.add(rank.node);
        });
      }

      return Array.from(nodes).map(node => {
        const existing = prevData?.find(d => d.node === node) || { node };
        
        // Store response for easy access in render function
        if (!comparisonData) {
          setComparisonData({});
        }
        
        if (!comparisonData[model]) {
          setComparisonData(prev => ({
            ...prev,
            [model]: { ...response }
          }));
        }
        
        // Get all anomalies for this node from this model
        const nodeAnomalies = response?.ranks ? 
          response.ranks.filter(rank => rank?.node === node) : [];
        
        // For RCA data, include anomaly IDs to identify compound anomalies
        const anomalyTypes = nodeAnomalies.map(a => a.type).filter(Boolean);
        
        // Get root cause and confidence directly from the response data
        let rootCause = "Normal";
        let confidence = 0.5;
        
        if (nodeAnomalies && nodeAnomalies.length > 0) {
          // Sort by score descending
          nodeAnomalies.sort((a, b) => b.score - a.score);
          
          // For confidence, use highest score
          confidence = Math.max(...nodeAnomalies.map(a => a.score || 0));
          
          if (nodeAnomalies.length === 1) {
            // Single anomaly case
            rootCause = nodeAnomalies[0].type;
          } else {
            // Multiple anomalies case - the display will be handled in rendering
            rootCause = nodeAnomalies.slice(0, 3).map(a => a.type).join(", ");
          }
        }
        
        return {
          ...existing,
          [`${model}_root_cause`]: rootCause,
          [`${model}_confidence`]: confidence,
          [`${model}_anomalies`]: nodeAnomalies,
          [`${model}_anomaly_types`]: anomalyTypes,
          [`${model}_is_compound`]: anomalyTypes.length > 1,
          [`${model}_compound_description`]: getCompoundAnomalyDescription(nodeAnomalies),
          ...(response?.advantages && { 
            advantages: response.advantages 
          })
        };
      });
    });
  };

  // Modified handleCompareModels to support streaming results and use threshold
  const handleCompareModels = useCallback(async () => {
    if (selectedModels.length === 0) {
      setComparisonError("Please select at least one model to analyze.");
      return;
    }

    try {
      setComparisonError(null);
      setComparisonLoading(true);
      setProcessingModels(new Set(selectedModels));
      
      // Initialize comparison data if empty
      setComparisonData(prev => prev || {});
      
      // Process each model independently
      const modelPromises = selectedModels.map(async (model) => {
        try {
          const startTime = Date.now();
          
          // Update UI to show this model is processing
          setComparisonData(prev => ({
            ...prev,
            [model]: { processing: true }
          }));
          
          // Use threshold value from UI
          const response = await anomalyService.getModelDiagnosis(model, thresholdValue);
          const analysisTime = Date.now() - startTime;
          
          // Calculate advantages
          const advantages = getModelAdvantages(model, response, analysisTime);
          
          // Update times and advantages
          setModelAnalysisTimes(prev => ({
            ...prev,
            [model]: analysisTime
          }));
          
          setModelAdvantages(prev => ({
            ...prev,
            [model]: advantages
          }));
          
          // Update comparison data with results
          setComparisonData(prev => ({
            ...prev,
            [model]: {
              ...response,
              analysisTime,
              advantages,
              processing: false
            }
          }));
          
          // Update RCA comparison data
          updateRcaComparison(model, response, analysisTime);
          
        } catch (error) {
          console.error(`Error analyzing model ${model}:`, error);
          setComparisonData(prev => ({
            ...prev,
            [model]: {
              error: error.toString(),
              processing: false
            }
          }));
          setProcessingModels(prev => {
            const newSet = new Set(prev);
            newSet.delete(model);
            return newSet;
          });
        }
      });
      
      // Wait for all models to complete
      await Promise.all(modelPromises);
      setComparisonLoading(false);
      setProcessingModels(new Set());
      
    } catch (error) {
      console.error("Error in model comparison:", error);
      setComparisonError(error.toString());
      setComparisonLoading(false);
      setProcessingModels(new Set());
    }
  }, [selectedModels, thresholdValue]);

  // Use the memoized rcaComparisonColumns
  const rcaComparisonColumns = useMemo(() => {
    const baseColumns = [...initialRcaComparisonColumns];
    
    selectedModels.forEach((model) => {
      const modelData = comparisonData?.[model];
      const isProcessing = processingModels.has(model);
      const hasError = modelData?.error;
      const advantages = modelAdvantages[model] || [];
      
      baseColumns.push({
        title: (
          <div style={{ minWidth: 180 }}>
            <div style={{
              color: getModelColor(model),
              fontWeight: 'bold',
              borderBottom: `2px solid ${getModelColor(model)}`,
              paddingBottom: 4,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between'
            }}>
              {model}
              {isProcessing && <SyncOutlined spin style={{ marginLeft: 8 }} />}
              {hasError && <Tag color="red" style={{ marginLeft: 8 }}>Error</Tag>}
            </div>
            {advantages.map((adv, i) => (
              <div key={i} style={{
                fontSize: 11,
                color: '#666',
                lineHeight: '1.2',
                marginTop: 4,
                display: 'flex',
                alignItems: 'center',
                gap: 4
              }}>
                <span>{adv.icon}</span>
                <span>{adv.text}</span>
              </div>
            ))}
          </div>
        ),
        dataIndex: `${model}_root_cause`,
        key: `${model}_root_cause`,
        width: 200,
        render: (text, record) => {
          if (isProcessing) {
            return (
              <Text style={{ 
                color: getModelColor(model),
                opacity: 0.5 
              }}>
                Analyzing...
              </Text>
            );
          }
          return renderRootCause(model, record.node);
        }
      });

      baseColumns.push({
        title: 'Confidence',
        dataIndex: `${model}_confidence`,
        key: `${model}_confidence`,
        width: 130,
        render: (score) => {
          if (isProcessing) return <Spin size="small" />;
          if (!score && score !== 0) return <Text type="secondary">N/A</Text>;

          const percent = Math.round(score * 100);
          let color = "#52c41a";
          if (percent < 50) color = "#ff4d4f";
          else if (percent < 75) color = "#faad14";

          return (
            <div style={{ position: 'relative' }}>
              <Progress
                percent={percent}
                size="small"
                strokeColor={color}
                format={(percent) => `${percent}%`}
              />
              {modelAnalysisTimes[model] && (
                <div style={{
                  position: 'absolute',
                  right: 0,
                  bottom: -18,
                  fontSize: 10,
                  color: '#999'
                }}>
                  {Math.round(modelAnalysisTimes[model]/100)/10}s
                </div>
              )}
            </div>
          );
        },
      });
    });

    return baseColumns;
  }, [selectedModels, comparisonData, processingModels, modelAnalysisTimes, modelAdvantages, initialRcaComparisonColumns]);

  const handleModelSelectionChange = async (values) => {
    // Filter out null and undefined values
    const filteredValues = values.filter(
      (value) => value !== null && value !== undefined
    );
    
    // Only update if at least one model is selected
    if (filteredValues.length > 0) {
      // Validate models before selection
      const validatedModels = [];
      
      for (const model of filteredValues) {
        try {
          // Skip validation for models that are already selected
          if (selectedModels.includes(model)) {
            validatedModels.push(model);
            continue;
          }
          
          console.log(`Validating model: ${model}`);
          const validationResult = await anomalyService.validateModel(model);
          
          if (validationResult.valid) {
            console.log(`Model ${model} validated successfully`);
            validatedModels.push(model);
          } else {
            console.error(`Model ${model} validation failed:`, validationResult.error);
            message.error(`Failed to validate model "${model}": ${validationResult.error}`);
          }
        } catch (error) {
          console.error(`Error validating model ${model}:`, error);
          message.error(`Error validating model "${model}"`);
        }
      }
      
      // Only update if we have at least one valid model
      if (validatedModels.length > 0) {
        setSelectedModels(validatedModels);
      } else if (selectedModels.length === 0) {
        // If no models are currently selected and all validations failed, 
        // show a more prominent error
        message.error('No valid models available for selection');
      }
    }
  };

  // New helper functions for enhanced visualizations
  const getRadarChartData = () => {
    if (!comparisonData || !selectedModels.length) return [];
    
    const radarData = [];
    const metrics = ["accuracy", "recall", "precision", "f1_score", "latency"];
    
    selectedModels.forEach(model => {
      if (!comparisonData[model] || comparisonData[model].error) return;
      
      const modelData = {
        model,
        "accuracy": comparisonData[model].rca_accuracy || 0.7,
        "recall": comparisonData[model].rca_recall || 0.65,
        "precision": comparisonData[model].rca_precision || 0.75,
        "f1_score": comparisonData[model].f1_score || 0.72,
        "latency": comparisonData[model].rca_time ? 
          Math.min(1 - (comparisonData[model].rca_time / 1000), 0.99) : 0.8,
      };
      
      radarData.push(modelData);
    });
    
    return radarData;
  };
  
  const getAnomalyDistribution = () => {
    if (!comparisonData || !selectedModels.length) return [];
    
    const anomalyTypes = ["cpu", "io", "network", "cache", "indexes"];
    const distributionData = anomalyTypes.map(type => {
      const dataPoint = { type };
      
      selectedModels.forEach(model => {
        if (!comparisonData[model] || !comparisonData[model].ranks) {
          dataPoint[model] = 0;
          return;
        }
        
        const typeCount = comparisonData[model].ranks.filter(
          rank => rank && rank.type && rank.type.toLowerCase().includes(type.toLowerCase())
        ).length;
        
        dataPoint[model] = typeCount;
      });
      
      return dataPoint;
    });
    
    return distributionData;
  };
  
  const getTimelineData = () => {
    if (!comparisonData) return [];
    
    const now = new Date();
    const timelineData = [];
    
    // Generate synthetic timeline data based on current ranks
    for (let i = 0; i < 10; i++) {
      const timestamp = new Date(now.getTime() - (i * 3600000));
      const entry = { 
        timestamp: timestamp.toISOString(),
        timestampMs: timestamp.getTime() // Add millisecond timestamp for better chart processing
      };
      
      selectedModels.forEach(model => {
        if (!comparisonData[model] || !comparisonData[model].ranks) {
          entry[model] = 0;
          entry[`${model}_trend`] = 0; // Add trend data
          return;
        }
        
        // Synthetic data - in real app would use historical data
        const anomalyCount = Math.floor(
          (comparisonData[model].ranks.length / 2) * 
          (1 - (i * 0.1))
        );
        
        // Add trend direction (positive or negative change)
        const prevIndex = Math.min(i + 1, 9);
        const prevValue = prevIndex < 9 ? 
          Math.floor((comparisonData[model].ranks.length / 2) * (1 - (prevIndex * 0.1))) : 0;
        
        const trend = anomalyCount - prevValue;
        
        entry[model] = Math.max(0, anomalyCount);
        entry[`${model}_trend`] = trend;
        
        // Add feature importance metrics if available
        if (comparisonData[model].ranks && comparisonData[model].ranks[0] && comparisonData[model].ranks[0].features) {
          const features = comparisonData[model].ranks[0].features;
          Object.keys(features).forEach(feature => {
            entry[`${model}_${feature}`] = features[feature] * (1 - (i * 0.05));
          });
        }
      });
      
      timelineData.push(entry);
    }
    
    return timelineData.reverse();
  };

  const getAvailableNodes = () => {
    if (!comparisonData) return [];
    
    const nodes = new Set();
    selectedModels.forEach(model => {
      if (comparisonData[model] && comparisonData[model].ranks) {
        comparisonData[model].ranks.forEach(rank => {
          if (rank && rank.node) nodes.add(rank.node);
        });
      }
    });
    
    return Array.from(nodes);
  };

  const getFilteredComparisonData = () => {
    if (!rcaComparisonData || rcaComparisonData.length === 0) return [];
    
    return rcaComparisonData.filter(row => {
      // Filter by selected nodes if any are selected
      if (filteredNodes.length > 0 && !filteredNodes.includes(row.node)) {
        return false;
      }
      
      // Filter by confidence threshold for any model
      const passesThreshold = selectedModels.some(model => {
        const confidence = row[`${model}_confidence`];
        return confidence !== null && confidence * 100 >= thresholdValue;
      });
      
      return passesThreshold;
    });
  };

  // New handlers for enhanced UI controls
  const handleTimeRangeChange = (value) => {
    setTimeRange(value);
  };
  
  const handleNodeFilterChange = (values) => {
    setFilteredNodes(values);
  };
  
  const handleMetricSelection = (checkedValues) => {
    setSelectedMetrics(checkedValues);
  };
  
  const handleChartTypeChange = (e) => {
    setChartType(e.target.value);
  };
  
  const handleThresholdChange = (value) => {
    setThresholdValue(value);
  };

  // Add reset function to clear previous analysis results
  const resetComparisonData = useCallback(() => {
    setComparisonData(null);
    setComparisonError(null);
  }, [setComparisonData, setComparisonError]);

  // Helper function to compare arrays
  const arrayEquals = (a, b) => {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
      if (a[i] !== b[i]) return false;
    }
    return true;
  };

  // Function to fetch metrics and check for fluctuations
  const fetchAndCheckMetricsFluctuations = useCallback(async () => {
    try {
      // Check if we should skip this request
      const now = Date.now();
      if (comparisonLoading || fluctuationCooldown || (now - lastRequestTime) < REQUEST_THROTTLE_MS) {
        return;
      }
      
      // Update last request time
      setLastRequestTime(now);
      
      const metricsData = await anomalyService.getMetricsFluctuations();
      setMetricsFluctuations(metricsData.metrics);
      
      const fluctuationDetected = metricsData.summary?.has_fluctuations || false;
      
      if (hasFluctuations !== fluctuationDetected) {
        setHasFluctuations(fluctuationDetected);
      }
      
      const timeSinceLastAnalysis = now - lastAnalysisTime;
      const shouldTriggerAnalysis = 
        fluctuationDetected && 
        autoAnalysisEnabled && 
        selectedModels.length > 0 && 
        !comparisonLoading &&
        timeSinceLastAnalysis > cooldownPeriodMs;
      
      if (shouldTriggerAnalysis) {
        console.log(`Metric fluctuations detected: ${metricsData.summary.fluctuating_metrics} out of ${metricsData.summary.total_metrics} metrics`);
        console.log(`Triggering analysis (last analysis was ${timeSinceLastAnalysis/1000}s ago)`);
        
        setFluctuationCooldown(true);
        setLastAnalysisTime(now);
        
        await handleCompareModels();
        
        // Set cooldown using setTimeout
        setTimeout(() => {
          setFluctuationCooldown(false);
        }, cooldownPeriodMs);
      }
      
    } catch (error) {
      console.error('Error checking metrics fluctuations:', error);
      // Add error cooldown to prevent rapid retries on error
      setTimeout(() => setLastRequestTime(Date.now()), REQUEST_THROTTLE_MS);
    }
  }, [
    autoAnalysisEnabled,
    selectedModels,
    comparisonLoading,
    hasFluctuations,
    lastAnalysisTime,
    fluctuationCooldown,
    handleCompareModels,
    lastRequestTime
  ]);

  // Create debounced version of the fetch function
  const debouncedFetchAndCheck = useCallback(
    debounce(fetchAndCheckMetricsFluctuations, 1000),
    [fetchAndCheckMetricsFluctuations]
  );

  // Setup timer to check for metric fluctuations when auto-analysis is enabled
  useEffect(() => {
    let intervalId = null;
    
    if (autoAnalysisEnabled) {
      // Initial check with delay to prevent immediate execution
      const initialCheckTimeout = setTimeout(() => {
        debouncedFetchAndCheck();
      }, 1000);
      
      // Set up interval for regular checks
      intervalId = setInterval(() => {
        debouncedFetchAndCheck();
      }, 10000); // Check every 10 seconds
      
      // Cleanup function
      return () => {
        clearTimeout(initialCheckTimeout);
        clearInterval(intervalId);
      };
    }
    
    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [autoAnalysisEnabled, debouncedFetchAndCheck]);

  // Add function to toggle auto-analysis
  const toggleAutoAnalysis = () => {
    setAutoAnalysisEnabled(prev => !prev);
  };

  // New helper function to get feature importance data
  const getFeatureImportanceData = (model) => {
    if (!comparisonData || !comparisonData[model] || !comparisonData[model].ranks || !comparisonData[model].ranks[0] || !comparisonData[model].ranks[0].features) {
      return [];
    }
    
    const features = comparisonData[model].ranks[0].features;
    return Object.entries(features).map(([feature, value]) => ({
      name: feature,
      value: value,
    }));
  };

  useEffect(() => {
    fetchAvailableModels();
  }, []);

  return (
    <div className="tensorboard-dashboard">
      <Card className="control-panel" bordered={false}>
        <Row gutter={16} align="middle">
          <Col span={24}>
            <Title level={4}>
              Root Cause Analysis Dashboard
              {autoAnalysisEnabled && (
                <Tag color="green" style={{ marginLeft: 8 }}>
                  <SyncOutlined spin /> Auto-Analysis Active
                </Tag>
              )}
              {hasFluctuations && (
                <Tag color="orange" style={{ marginLeft: 8 }}>
                  <AlertOutlined /> Metric Fluctuations Detected
                </Tag>
              )}
            </Title>
          </Col>
        </Row>
        
        <Row gutter={16} align="middle" style={{ marginBottom: 16 }}>
          <Col xs={24} md={8}>
            <div style={{ marginBottom: 8 }}>
              <Text strong>AI Models for Analysis:</Text>
            </div>
            <Select
              mode="multiple"
              style={{ width: '100%' }}
              placeholder="Select models to analyze"
              value={selectedModels}
              onChange={handleModelSelectionChange}
              optionLabelProp="label"
              loading={loading}
              maxTagCount={2}
              showArrow
              bordered
              className="model-select"
            >
              {availableModels.map((model) => (
                <Option 
                  key={model || `model-${Math.random()}`} 
                  value={model || ""}
                  label={model || "Unknown model"}
                >
                  <Space>
                    {loading ? <SyncOutlined spin /> : null}
                    <Text>{model || "Unknown model"}</Text>
                  </Space>
                </Option>
              ))}
            </Select>
          </Col>
          
          <Col xs={24} md={6}>
            <div style={{ marginBottom: 8 }}>
              <Text strong>Time Range:</Text>
            </div>
            <Select
              style={{ width: '100%' }}
              placeholder="Time range"
              value={timeRange}
              onChange={handleTimeRangeChange}
              showArrow
              bordered
            >
              <Option value="1h">Last hour</Option>
              <Option value="6h">Last 6 hours</Option>
              <Option value="24h">Last 24 hours</Option>
              <Option value="7d">Last 7 days</Option>
            </Select>
          </Col>
          
          <Col xs={24} md={4}>
            <div style={{ marginBottom: 8 }}>
              <Text strong>&nbsp;</Text>
            </div>
            <Button
              type="primary"
              onClick={handleCompareModels}
              loading={comparisonLoading}
              disabled={selectedModels.length === 0}
              block
              icon={<SearchOutlined />}
            >
              Run Analysis
            </Button>
          </Col>
          
          <Col xs={24} md={3}>
            <div style={{ marginBottom: 8 }}>
              <Text strong>&nbsp;</Text>
            </div>
            <Button 
              icon={<SyncOutlined />} 
              onClick={fetchAvailableModels}
              disabled={loading}
            >
              Refresh Models
            </Button>
          </Col>

          <Col xs={24} md={3}>
            <div style={{ marginBottom: 8 }}>
              <Text strong>Auto Analysis:</Text>
            </div>
            <Switch 
              checked={autoAnalysisEnabled}
              onChange={toggleAutoAnalysis}
              checkedChildren="On"
              unCheckedChildren="Off"
            />
          </Col>
        </Row>
        
        {availableModels.length === 0 && loading && (
          <Alert
            message="Loading available models..."
            type="info"
            showIcon
            icon={<SyncOutlined spin />}
            style={{ marginBottom: 16 }}
          />
        )}

        {selectedModels.length === 0 && availableModels.length > 0 && (
          <Alert
            message="Please select a model to analyze"
            type="warning"
            showIcon
            style={{ marginBottom: 16 }}
          />
        )}
        
        {comparisonError && (
          <Alert
            message={comparisonError}
            type="error"
            showIcon
            style={{ marginBottom: 16 }}
          />
        )}
        
        {comparisonData && (
          <Row gutter={16} style={{ marginTop: 16 }}>
            <Col span={24}>
              <Card size="small" title="Filters & Options">
                <Row gutter={16}>
                  <Col xs={24} md={6}>
                    <div style={{ marginBottom: 8 }}>
                      <Text strong>Node Filter:</Text>
                    </div>
                    <Select
                      mode="multiple"
                      style={{ width: '100%' }}
                      placeholder="Filter by node"
                      value={filteredNodes}
                      onChange={handleNodeFilterChange}
                      maxTagCount={2}
                    >
                      {getAvailableNodes().map(node => (
                        <Option key={node} value={node}>{node}</Option>
                      ))}
                    </Select>
                  </Col>
                  
                  <Col xs={24} md={6}>
                    <div style={{ marginBottom: 8 }}>
                      <Text strong>Metrics:</Text>
                    </div>
                    <Checkbox.Group
                      options={[
                        { label: 'CPU', value: 'cpu' },
                        { label: 'Memory', value: 'memory' },
                        { label: 'I/O', value: 'io' },
                        { label: 'Network', value: 'network' }
                      ]}
                      value={selectedMetrics}
                      onChange={handleMetricSelection}
                    />
                  </Col>
                  
                  <Col xs={24} md={6}>
                    <div style={{ marginBottom: 8 }}>
                      <Text strong>Confidence Threshold:</Text>
                    </div>
                    <Slider
                      min={0}
                      max={100}
                      value={thresholdValue}
                      onChange={handleThresholdChange}
                      tooltip={{ formatter: value => `${value}%` }}
                    />
                  </Col>
                  
                  <Col xs={24} md={6}>
                    <div style={{ marginBottom: 8 }}>
                      <Text strong>Chart Type:</Text>
                    </div>
                    <Radio.Group value={chartType} onChange={handleChartTypeChange}>
                      <Radio.Button value="radar">Radar</Radio.Button>
                      <Radio.Button value="bar">Bar</Radio.Button>
                      <Radio.Button value="line">Line</Radio.Button>
                    </Radio.Group>
                  </Col>
                </Row>
              </Card>
            </Col>
          </Row>
        )}
      </Card>
      
      {/* Main Dashboard Content */}
      {comparisonData ? (
        <div className="dashboard-content">
          {/* Add overlay loading indicator */}
          {comparisonLoading && (
            <div className="loading-overlay">
              <Spin size="large" />
            </div>
          )}
          
          <Tabs activeKey={activeTab} onChange={setActiveTab} type="card">
            <TabPane tab="Overview" key="1">
              <Row gutter={16}>
                <Col xs={24} lg={12}>
                  <Card title="Model Performance Comparison" className="chart-card">
                    <ResponsiveContainer width="100%" height={350}>
                      <RadarChart outerRadius={130} data={getRadarChartData()}>
                        <PolarGrid />
                        <PolarAngleAxis dataKey="name" />
                        <PolarRadiusAxis angle={30} domain={[0, 1]} />
                        
                        {selectedModels.map((model, index) => (
                          comparisonData[model] && !comparisonData[model].error && (
                            <Radar
                              key={model}
                              name={model}
                              dataKey={model}
                              stroke={getModelColor(model, index)}
                              fill={getModelColor(model, index)}
                              fillOpacity={0.3}
                            />
                          )
                        ))}
                        
                        <Legend />
                        <Tooltip />
                      </RadarChart>
                    </ResponsiveContainer>
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
            </TabPane>
            
            <TabPane tab="Node Analysis" key="2">
              <Card title="Node Performance Heatmap" className="chart-card">
                <div style={{ overflowX: 'auto' }}>
                  <Table
                    columns={rcaComparisonColumns}
                    dataSource={getFilteredComparisonData()}
                    rowKey="node"
                    pagination={false}
                    scroll={{ x: "max-content" }}
                    loading={comparisonLoading}
                  />
                </div>
              </Card>
            </TabPane>
            
            <TabPane tab="Model Metrics" key="3">
              <Row gutter={16}>
                {selectedModels.map((model, index) => (
                  comparisonData && comparisonData[model] && (
                    <Col xs={24} lg={12} key={model}>
                      <Card 
                        title={`${model} Metrics`} 
                        className="chart-card"
                        extra={comparisonData[model].error && (
                          <Tag color="red">Error</Tag>
                        )}
                      >
                        {comparisonData[model].error ? (
                          <Alert
                            message={comparisonData[model].error}
                            type="error"
                            showIcon
                          />
                        ) : (
                          <div>
                            <Row gutter={16}>
                              <Col span={8}>
                                <Statistic
                                  title="Precision"
                                  value={(comparisonData[model].rca_precision || 0) * 100}
                                  precision={1}
                                  suffix="%"
                                  valueStyle={{ color: '#3f8600' }}
                                />
                              </Col>
                              <Col span={8}>
                                <Statistic
                                  title="Detected Anomalies"
                                  value={comparisonData[model].ranks?.length || 0}
                                  valueStyle={{ color: '#cf1322' }}
                                />
                              </Col>
                              <Col span={8}>
                                <Statistic
                                  title="Analysis Time"
                                  value={comparisonData[model].rca_time || 0}
                                  precision={2}
                                  suffix="s"
                                />
                              </Col>
                            </Row>
                            
                            <Divider />
                            
                            <div style={{ marginTop: 16 }}>
                              <Text strong>Top Anomalies:</Text>
                              {comparisonData[model].ranks && comparisonData[model].ranks.length > 0 ? (
                                <ul className="anomaly-list">
                                  {comparisonData[model].ranks.slice(0, 5).map((rank, idx) => (
                                    <li key={idx}>
                                      <Text>{rank.node}: </Text>
                                      <Tag color={getRankColor(rank.type)}>{rank.type}</Tag>
                                      <Text type="secondary"> ({Math.round(rank.score * 100) / 100}% confidence)</Text>
                                    </li>
                                  ))}
                                </ul>
                              ) : (
                                <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="No anomalies detected" />
                              )}
                            </div>
                          </div>
                        )}
                      </Card>
                    </Col>
                  )
                ))}
              </Row>
            </TabPane>
            
            <TabPane tab="Raw Data" key="4">
              <Card>
                <Tabs defaultActiveKey="1">
                  <TabPane tab="Comparison Table" key="1">
                    <Table
                      columns={rcaComparisonColumns}
                      dataSource={rcaComparisonData}
                      rowKey="node"
                      pagination={false}
                      scroll={{ x: "max-content" }}
                      size="small"
                    />
                  </TabPane>
                  <TabPane tab="JSON Data" key="2">
                    <div className="json-viewer">
                      <pre>{comparisonData ? JSON.stringify(comparisonData, null, 2) : "No data"}</pre>
                    </div>
                  </TabPane>
                </Tabs>
              </Card>
            </TabPane>
          </Tabs>
        </div>
      ) : comparisonLoading ? (
        <div style={{ textAlign: 'center', padding: '40px 0' }}>
          <Spin size="large" />
          <div style={{ marginTop: 16 }}>
            <Text>Analyzing data with selected models...</Text>
          </div>
        </div>
      ) : (
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description="Select one or more models and click Run Analysis to view results"
        />
      )}
      
      <style jsx>{`
        .tensorboard-dashboard {
          background: #f0f2f5;
          border-radius: 4px;
          padding: 8px;
        }
        
        .chart-card {
          margin-bottom: 16px;
          height: 100%;
        }
        
        .control-panel {
          margin-bottom: 16px;
        }
        
        .json-viewer {
          background: #f5f5f5;
          padding: 16px;
          border-radius: 4px;
          overflow: auto;
          max-height: 500px;
        }
        
        .anomaly-list {
          padding-left: 20px;
        }
        
        .ant-statistic-title {
          font-size: 14px;
        }
        
        .ant-statistic-content {
          font-size: 20px;
        }
        
        .model-select .ant-select-selector {
          border-radius: 4px !important;
          box-shadow: none !important;
        }
        
        .ant-select-focused .ant-select-selector,
        .ant-select-selector:focus,
        .ant-select-selector:active,
        .ant-select-open .ant-select-selector {
          border-color: #40a9ff !important;
          box-shadow: 0 0 0 2px rgba(24, 144, 255, 0.2) !important;
        }
        
        .dashboard-content {
          position: relative;
        }
        
        .loading-overlay {
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(255, 255, 255, 0.7);
          display: flex;
          justify-content: center;
          align-items: center;
          z-index: 100;
          border-radius: 4px;
        }
      `}</style>
    </div>
  );
};

export default RanksPanel;
