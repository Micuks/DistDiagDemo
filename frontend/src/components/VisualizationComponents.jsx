import React from "react";
import { Empty } from "antd";
import { getModelColor } from "../utils/rankUtils.jsx";

// Force directed graph visualization of the propagation graph
export const renderForceGraph = (comparisonData, selectedModels) => {
  if (!comparisonData || !selectedModels.length) return <Empty description="No data available" />;
  
  const model = selectedModels[0]; // Use first selected model
  if (!comparisonData[model]) return <Empty description="No model data available" />;
  if (!comparisonData[model].propagation_graph) return <Empty description="No propagation data available" />;
  
  const propagationGraph = comparisonData[model].propagation_graph;
  const nodes = Object.keys(propagationGraph);
  
  if (nodes.length === 0) {
    return <Empty description="No nodes in propagation graph" />;
  }
  
  // Create array of nodes
  const nodeObjects = nodes.map((node, index) => {
    // Calculate position in a circle
    const radius = 180;
    const angle = (2 * Math.PI * index) / nodes.length;
    const x = radius * Math.cos(angle) + 220;
    const y = radius * Math.sin(angle) + 220;
    
    // Check if node has anomaly
    const hasAnomaly = comparisonData[model].ranks?.some(r => r.node === node);
    const anomalyType = hasAnomaly ? 
      comparisonData[model].ranks.find(r => r.node === node).type : null;
    
    return { id: node, x, y, hasAnomaly, anomalyType };
  });
  
  // Create array of links
  const links = [];
  nodes.forEach(source => {
    if (propagationGraph[source]) {
      propagationGraph[source].forEach(target => {
        links.push({
          source,
          target: target.target,
          correlation: target.correlation
        });
      });
    }
  });
  
  // Generate SVG
  return (
    <div style={{ padding: '20px', textAlign: 'center' }}>
      <h3>Node Propagation Graph</h3>
      <div style={{ marginBottom: '10px' }}>
        <div style={{ display: 'inline-block', marginRight: '10px', padding: '2px 8px', background: '#ff4d4f', color: 'white', borderRadius: '2px' }}>Anomaly Node</div>
        <div style={{ display: 'inline-block', marginRight: '10px', padding: '2px 8px', background: '#1890ff', color: 'white', borderRadius: '2px' }}>Strong Correlation (≥ 0.9)</div>
        <div style={{ display: 'inline-block', padding: '2px 8px', background: '#52c41a', color: 'white', borderRadius: '2px' }}>Medium Correlation (≥ 0.7)</div>
      </div>
      <svg width="500" height="440" style={{ border: '1px solid #f0f0f0', borderRadius: '5px' }}>
        {/* Draw links first so they appear behind nodes */}
        {links.map((link, index) => {
          const sourceNode = nodeObjects.find(n => n.id === link.source);
          const targetNode = nodeObjects.find(n => n.id === link.target);
          
          if (!sourceNode || !targetNode) return null;
          
          // Determine line color based on correlation strength
          let strokeColor = '#d9d9d9';
          let strokeWidth = 1;
          
          if (link.correlation >= 0.9) {
            strokeColor = '#1890ff'; // Blue for strong correlation
            strokeWidth = 3;
          } else if (link.correlation >= 0.7) {
            strokeColor = '#52c41a'; // Green for medium correlation
            strokeWidth = 2;
          }
          
          return (
            <g key={`link-${index}`}>
              <line 
                x1={sourceNode.x} 
                y1={sourceNode.y} 
                x2={targetNode.x} 
                y2={targetNode.y} 
                stroke={strokeColor}
                strokeWidth={strokeWidth}
                strokeOpacity={0.6}
              />
              {/* Add correlation label */}
              <text 
                x={(sourceNode.x + targetNode.x) / 2} 
                y={(sourceNode.y + targetNode.y) / 2}
                textAnchor="middle"
                fontSize="12"
                fill="#666"
                dy="-5"
              >
                {(link.correlation * 100).toFixed(0)}%
              </text>
            </g>
          );
        })}
        
        {/* Draw nodes */}
        {nodeObjects.map((node, index) => (
          <g key={`node-${index}`}>
            {/* Node circle */}
            <circle 
              cx={node.x} 
              cy={node.y} 
              r={node.hasAnomaly ? 25 : 20}
              fill={node.hasAnomaly ? '#ff4d4f' : '#fff'}
              stroke={node.hasAnomaly ? '#ff4d4f' : '#d9d9d9'}
              strokeWidth={node.hasAnomaly ? 2 : 1}
            />
            {/* Node label */}
            <text 
              x={node.x} 
              y={node.y}
              textAnchor="middle"
              dominantBaseline="central"
              fontSize="12"
              fill={node.hasAnomaly ? '#fff' : '#000'}
            >
              {node.id.split('.').pop()}
            </text>
            {/* Anomaly type label */}
            {node.hasAnomaly && (
              <text 
                x={node.x} 
                y={node.y + 40}
                textAnchor="middle"
                fontSize="11"
                fill="#ff4d4f"
              >
                {node.anomalyType}
              </text>
            )}
          </g>
        ))}
      </svg>
    </div>
  );
};

// Create a proper heatmap visualization
export const renderNodeHeatmap = (comparisonData, selectedModels) => {
  if (!comparisonData || !selectedModels.length) {
    return <Empty description="No data available" />;
  }
  
  const model = selectedModels[0]; // Use first selected model
  if (!comparisonData[model]) return <Empty description="No model data available" />;
  if (!comparisonData[model].ranks || !Array.isArray(comparisonData[model].ranks)) {
    return <Empty description="No anomaly data available" />;
  }
  
  const anomalies = comparisonData[model].ranks || [];
  if (anomalies.length === 0) {
    return <Empty description="No anomalies detected" />;
  }
  
  // Group anomalies by node and type
  const nodeAnomalyTypes = {};
  const allTypes = new Set();
  
  anomalies.forEach(anomaly => {
    if (!nodeAnomalyTypes[anomaly.node]) {
      nodeAnomalyTypes[anomaly.node] = {};
    }
    
    nodeAnomalyTypes[anomaly.node][anomaly.type] = anomaly.score;
    allTypes.add(anomaly.type);
  });
  
  const nodes = Object.keys(nodeAnomalyTypes);
  const anomalyTypes = Array.from(allTypes);
  
  // Generate color based on score
  const getColorForScore = (score) => {
    if (!score) return '#f0f0f0';
    
    // Color gradient from yellow to red
    const intensity = Math.min(1, score / 10); // Normalize score to 0-1 range
    const r = 255;
    const g = Math.round(255 * (1 - intensity));
    const b = 0;
    
    return `rgb(${r}, ${g}, ${b})`;
  };
  
  return (
    <div className="node-heatmap">
      <div style={{ marginBottom: '10px' }}>
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '5px' }}>
          <span style={{ marginRight: '10px' }}>Confidence Score: </span>
          <div style={{ display: 'flex', alignItems: 'center', background: 'linear-gradient(to right, #ffffcc, #ff8000, #ff0000)', width: '200px', height: '20px' }}></div>
          <span style={{ marginLeft: '5px' }}>Higher</span>
        </div>
      </div>
      
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            <th style={{ padding: '10px', textAlign: 'left', borderBottom: '1px solid #e8e8e8' }}>Node</th>
            {anomalyTypes.map(type => (
              <th 
                key={type}
                style={{ 
                  padding: '10px', 
                  textAlign: 'center', 
                  borderBottom: '1px solid #e8e8e8',
                  minWidth: '100px'
                }}
              >
                {type.replace(/_/g, ' ')}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {nodes.map(node => (
            <tr key={node}>
              <td style={{ padding: '10px', borderBottom: '1px solid #f0f0f0' }}>{node}</td>
              {anomalyTypes.map(type => {
                const score = nodeAnomalyTypes[node][type];
                return (
                  <td 
                    key={`${node}-${type}`}
                    style={{ 
                      padding: '0', 
                      borderBottom: '1px solid #f0f0f0',
                      textAlign: 'center'
                    }}
                  >
                    <div 
                      style={{ 
                        background: getColorForScore(score),
                        width: '100%',
                        height: '40px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        color: score > 5 ? 'white' : 'black',
                        fontWeight: 'bold',
                        borderRadius: '4px'
                      }}
                    >
                      {score ? score.toFixed(2) : '-'}
                    </div>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

// Get anomaly distribution data for charts
export const getAnomalyDistribution = (comparisonData, selectedModels) => {
  if (!comparisonData || typeof comparisonData !== 'object' || Object.keys(comparisonData).length === 0 || !selectedModels.length) return [];
  
  // Create data array for all anomaly types across models
  const allTypes = new Set();
  const typeCountByModel = {};
  
  selectedModels.forEach(model => {
    if (!comparisonData[model] || !comparisonData[model].ranks) {
      return;
    }
    
    typeCountByModel[model] = {};
    const typeCount = comparisonData[model].ranks.filter(
      rank => rank.type).reduce((counts, rank) => {
        if (!counts[rank.type]) counts[rank.type] = 0;
        counts[rank.type]++;
        allTypes.add(rank.type);
        return counts;
      }, {});
    
    typeCountByModel[model] = typeCount;
  });
  
  // Format for chart
  return Array.from(allTypes).map(type => {
    const result = { type };
    selectedModels.forEach(model => {
      result[model] = typeCountByModel[model]?.[type] || 0;
    });
    return result;
  });
};

// Get timeline data for charts
export const getTimelineData = (comparisonData, selectedModels) => {
  if (!comparisonData || typeof comparisonData !== 'object' || Object.keys(comparisonData).length === 0) return [];
  
  // Create timeline data points
  const timePoints = [];
  const now = Date.now();
  
  // Generate time points for the last hour (or selected time period)
  for (let i = 12; i >= 0; i--) {
    timePoints.push(now - i * 5 * 60 * 1000); // 5-minute intervals
  }
  
  return timePoints.map((timestamp, index) => {
    const result = { timestamp };
    
    selectedModels.forEach((model, modelIndex) => {
      if (!comparisonData[model] || !comparisonData[model].ranks) {
        return;
      }
      
      // Placeholder values for the timeline - in a real app these would come from historical data
      result[model] = index < 6 ? 0 : 
        Math.max(0, Math.min(
          Math.floor((comparisonData[model].ranks.length / 2) * 
            (1 - Math.abs(index - 6) / 10)), 
          comparisonData[model].ranks.length
        ));
      
      // Trend line - slight variations
      result[`${model}_trend`] = index < 6 ? 0 :
        Math.floor((comparisonData[model].ranks.length / 2) * (1 - (index * 0.1))) - 
        Math.floor((comparisonData[model].ranks.length / 2) * (1 - ((index-1) * 0.1)));
      
      // Feature importance data for each model
      if (comparisonData[model].ranks && comparisonData[model].ranks[0] && comparisonData[model].ranks[0].features) {
        const features = comparisonData[model].ranks[0].features;
        Object.entries(features).forEach(([feature, value], featureIndex) => {
          if (featureIndex < 3) { // Show only top 3 features
            result[`${model}_${feature}`] = value * (1 - index * 0.02);
          }
        });
      }
    });
    
    return result;
  });
};

// Get feature importance data
export const getFeatureImportanceData = (comparisonData, model) => {
  if (!comparisonData || !comparisonData[model] || !comparisonData[model].ranks || !comparisonData[model].ranks[0] || !comparisonData[model].ranks[0].features) {
    return [];
  }
  
  const features = comparisonData[model].ranks[0].features;
  return Object.entries(features).map(([feature, value]) => ({
    name: feature,
    value: value,
  }));
}; 