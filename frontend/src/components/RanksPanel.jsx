import React, { useState, useEffect, useCallback } from "react";
import { Typography } from "antd";
import { anomalyService } from "../services/anomalyService";
import { debounce } from "../utils/rankUtils.jsx";
import RanksPanelControls from "./RanksPanelControls";
import DistDiagnosisPanel from "./DistDiagnosisPanel";
import MetricRooterPanel from "./MetricRooterPanel";
import * as VisualizationComponents from "./VisualizationComponents";

const { Text } = Typography;

const RanksPanel = () => {
  const [loading, setLoading] = useState(false);
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModels, setSelectedModels] = useState([]);
  const [comparisonLoading, setComparisonLoading] = useState(false);
  const [comparisonData, setComparisonData] = useState({});
  const [comparisonError, setComparisonError] = useState(null);
  const [nodeData, setNodeData] = useState([]);
  const [autoAnalysisEnabled, setAutoAnalysisEnabled] = useState(true);
  const [lastAnalyzedAnomalies, setLastAnalyzedAnomalies] = useState([]);
  const [timeRange, setTimeRange] = useState("1h");
  const [filteredNodes, setFilteredNodes] = useState([]);
  const [selectedMetrics, setSelectedMetrics] = useState(["cpu", "memory", "io", "network"]);
  const [chartType, setChartType] = useState("radar");
  const [showConfidenceIntervals, setShowConfidenceIntervals] = useState(true);
  const [sortMethod, setSortMethod] = useState("confidence");
  const [thresholdValue, setThresholdValue] = useState(0.1);
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
  const [analysisStep, setAnalysisStep] = useState(1); // Track current analysis step: 1=DistDiagnosis, 2=DBRooter
  const [selectedAnomaly, setSelectedAnomaly] = useState(null); // Store the selected anomaly for DBRooter analysis
  const [metricRankings, setMetricRankings] = useState({}); // Store DBRooter metric rankings
  const [metricRankingLoading, setMetricRankingLoading] = useState(false); // Loading state for metric ranking

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

  // Add reset function to clear previous analysis results
  const resetComparisonData = useCallback(() => {
    setComparisonData({});
    setComparisonError(null);
    setRcaComparisonData([]);
  }, []);

  // Modified handleCompareModels to support streaming results and use threshold
  const handleCompareModels = useCallback(async () => {
    if (selectedModels.length === 0) {
      setComparisonError("Please select at least one model to analyze.");
      return;
    }

    try {
      console.log("Starting model comparison...");
      setComparisonError(null);
      setComparisonLoading(true);
      setProcessingModels(new Set(selectedModels));
      
      // Initialize comparison data as an empty object
      setComparisonData({});
      
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
          const thresholdDecimal = thresholdValue;
          console.log(`Fetching model diagnosis for ${model} with threshold ${thresholdDecimal}`);
          const response = await anomalyService.getModelDiagnosis(model, thresholdDecimal);
          console.log(`Response for model ${model}:`, response);
          const analysisTime = Date.now() - startTime;
          
          // Update times
          setModelAnalysisTimes(prev => ({
            ...prev,
            [model]: analysisTime
          }));
          
          // Update comparison data with results
          setComparisonData(prev => ({
            ...prev,
            [model]: {
              ...(response[model] || response),
              analysisTime,
              processing: false
            }
          }));
          
          // Update RCA comparison data
          updateRcaComparison(model, response);
          
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

  // Helper function to update RCA comparison data
  const updateRcaComparison = (model, response) => {
    // Update RCA data incrementally
    setRcaComparisonData(prevData => {
      // Handle case where response contains model-keyed object
      const modelResponse = response[model] || response;
      
      const nodes = new Set(prevData?.map(d => d.node) || []);
      
      // Add new nodes from current model response
      if (modelResponse?.ranks) {
        modelResponse.ranks.forEach(rank => {
          if (rank?.node) nodes.add(rank.node);
        });
      } else if (modelResponse?.node_names) {
        // Add nodes from node_names array if available
        modelResponse.node_names.forEach(node => nodes.add(node));
      }

      return Array.from(nodes).map(node => {
        const existing = prevData?.find(d => d.node === node) || { node };
        
        // Get all anomalies for this node from this model
        const nodeAnomalies = modelResponse?.ranks ? 
          modelResponse.ranks.filter(rank => rank?.node === node) : [];
        
        // For RCA data, include anomaly IDs to identify compound anomalies
        const anomalyTypes = nodeAnomalies.map(a => a.type).filter(Boolean);
        
        // Get root cause and confidence directly from the response data
        let rootCause = "Normal";
        let confidence = 0;
        
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
          [`${model}_is_compound`]: anomalyTypes.length > 1
        };
      });
    });
  };

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
          }
        } catch (error) {
          console.error(`Error validating model ${model}:`, error);
        }
      }
      
      // Only update if we have at least one valid model
      if (validatedModels.length > 0) {
        setSelectedModels(validatedModels);
      }
    }
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
    lastRequestTime,
    cooldownPeriodMs,
    REQUEST_THROTTLE_MS
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

  // Function to proceed to metric ranking (step 2)
  const proceedToMetricRanking = async (anomaly) => {
    setSelectedAnomaly(anomaly);
    setMetricRankingLoading(true);
    setAnalysisStep(2);
    
    try {
      // Call MetricRooter service to get metric rankings - independent from the anomaly type
      const rankings = await anomalyService.getMetricRankings(
        anomaly.node, 
        selectedModels.length > 0 ? selectedModels[0] : null
      );
      setMetricRankings(rankings);
    } catch (error) {
      console.error("Error fetching metric rankings:", error);
    } finally {
      setMetricRankingLoading(false);
    }
  };

  // Function to go back to anomaly selection (step 1)
  const goBackToAnomalySelection = () => {
    setAnalysisStep(1);
    setSelectedAnomaly(null);
  };

  // Toggle auto-analysis
  const toggleAutoAnalysis = () => {
    setAutoAnalysisEnabled(prev => !prev);
  };

  // Helper to get available nodes from comparison data
  const getAvailableNodes = () => {
    if (!comparisonData || Object.keys(comparisonData).length === 0) return [];
    
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

  // Connect visualization functions for passing to components
  const renderForceGraph = () => {
    return VisualizationComponents.renderForceGraph(comparisonData, selectedModels);
  };

  const renderNodeHeatmap = () => {
    return VisualizationComponents.renderNodeHeatmap(comparisonData, selectedModels);
  };

  const getAnomalyDistribution = () => {
    return VisualizationComponents.getAnomalyDistribution(comparisonData, selectedModels);
  };

  const getTimelineData = () => {
    return VisualizationComponents.getTimelineData(comparisonData, selectedModels);
  };

  useEffect(() => {
    fetchAvailableModels();
  }, []);

  // Render the appropriate content based on the current step
  const renderAnalysisContent = () => {
    if (analysisStep === 1) {
      return (
        <DistDiagnosisPanel
          selectedModels={selectedModels}
          comparisonData={comparisonData}
          comparisonLoading={comparisonLoading}
          timelineView={timelineView}
          setTimelineView={setTimelineView}
          showRawData={showRawData}
          setShowRawData={setShowRawData}
          proceedToMetricRanking={proceedToMetricRanking}
          getAnomalyDistribution={getAnomalyDistribution}
          getTimelineData={getTimelineData}
          renderForceGraph={renderForceGraph}
          renderNodeHeatmap={renderNodeHeatmap}
        />
      );
    } else if (analysisStep === 2) {
      return (
        <MetricRooterPanel
          selectedAnomaly={selectedAnomaly}
          metricRankings={metricRankings}
          metricRankingLoading={metricRankingLoading}
          proceedToMetricRanking={proceedToMetricRanking}
          goBackToAnomalySelection={goBackToAnomalySelection}
        />
      );
    }
  };

  return (
    <div className="ranks-panel">
      <style jsx>{`
        .ranks-panel {
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
      `}</style>
      
      <RanksPanelControls
        availableModels={availableModels}
        selectedModels={selectedModels}
        comparisonLoading={comparisonLoading}
        comparisonError={comparisonError}
        timeRange={timeRange}
        autoAnalysisEnabled={autoAnalysisEnabled}
        hasFluctuations={hasFluctuations}
        filteredNodes={filteredNodes}
        selectedMetrics={selectedMetrics}
        thresholdValue={thresholdValue}
        chartType={chartType}
        loading={loading}
        handleModelSelectionChange={handleModelSelectionChange}
        handleCompareModels={handleCompareModels}
        fetchAvailableModels={fetchAvailableModels}
        toggleAutoAnalysis={toggleAutoAnalysis}
        handleTimeRangeChange={handleTimeRangeChange}
        handleNodeFilterChange={handleNodeFilterChange}
        handleMetricSelection={handleMetricSelection}
        handleThresholdChange={handleThresholdChange}
        handleChartTypeChange={handleChartTypeChange}
        getAvailableNodes={getAvailableNodes}
      />
      
      {/* Render the appropriate content based on the current step */}
      {renderAnalysisContent()}
    </div>
  );
};

export default RanksPanel;

