import React from "react";
import { Empty } from "antd";
import { getModelColor, getFullNodeName } from "../utils/rankUtils.jsx";

// Force directed graph visualization of the propagation graph
export const renderForceGraph = (comparisonData, selectedModels, thresholdValue = 0.1) => {
  if (!comparisonData || !selectedModels.length) return <Empty description="No data available" />;
  
  const model = selectedModels[0]; // Use first selected model
  if (!comparisonData[model]) return <Empty description="No model data available" />;
  if (!comparisonData[model].propagation_graph) return <Empty description="No propagation data available" />;
  
  const propagationGraph = comparisonData[model].propagation_graph;
  const nodes = Object.keys(propagationGraph);
  
  if (nodes.length === 0) {
    return <Empty description="No nodes in propagation graph" />;
  }
  
  // Filter anomalies using threshold
  const filteredRanks = (comparisonData[model].ranks || []).filter(r => r.positive_prob_score >= thresholdValue);
  
  // Create array of node objects
  const nodeObjects = nodes.map((node, index) => {
    const radius = 200;
    const angle = (2 * Math.PI * index) / nodes.length;
    const x = radius * Math.cos(angle) + 250;
    const y = radius * Math.sin(angle) + 220;
    
    const hasAnomaly = filteredRanks.some(r => r.node === node);
    const anomalyType = hasAnomaly ? filteredRanks.find(r => r.node === node).type : null;
    
    const fullNodeName = getFullNodeName(node);
    
    return { id: node, fullName: fullNodeName, x, y, hasAnomaly, anomalyType };
  });
  
  // Create array of unique undirected links with mean correlation if both directions exist
  const edgeMap = {};
  nodes.forEach(source => {
    const targets = propagationGraph[source];
    if (targets && typeof targets === 'object') {
      Object.entries(targets).forEach(([targetNode, correlation]) => {
        // Create a sorted key to ensure undirected edge uniqueness
        const key = source < targetNode ? `${source}|${targetNode}` : `${targetNode}|${source}`;
        if (!edgeMap[key]) {
          edgeMap[key] = { source: source, target: targetNode, correlations: [correlation] };
        } else {
          edgeMap[key].correlations.push(correlation);
        }
      });
    }
  });
  const links = Object.values(edgeMap).map(edge => {
    const meanCorrelation = edge.correlations.reduce((sum, weight) => sum + weight, 0) / edge.correlations.length;
    return { source: edge.source, target: edge.target, correlation: meanCorrelation };
  });
  
  return (
    <div style={{ padding: '20px', textAlign: 'center' }}>
      <h3>Node Propagation Graph</h3>
      <div style={{ marginBottom: '15px' }}>
        <div style={{ display: 'inline-block', marginRight: '15px', padding: '4px 10px', background: '#ff4d4f', color: 'white', borderRadius: '4px', fontWeight: 'bold' }}>Anomaly Node</div>
        <div style={{ display: 'inline-block', marginRight: '15px', padding: '4px 10px', background: '#1890ff', color: 'white', borderRadius: '4px', fontWeight: 'bold' }}>Strong Correlation (≥ 0.9)</div>
        <div style={{ display: 'inline-block', padding: '4px 10px', background: '#52c41a', color: 'white', borderRadius: '4px', fontWeight: 'bold' }}>Medium Correlation (≥ 0.7)</div>
      </div>
      <svg width="550" height="500" style={{ border: '1px solid #f0f0f0', borderRadius: '8px', background: '#fafafa', boxShadow: '0 2px 8px rgba(0,0,0,0.06)' }}>
        <defs>
          {links.map((link, index) => {
            let strokeColor = '#d9d9d9';
            if (link.correlation >= 0.9) {
              strokeColor = '#1890ff';
            } else if (link.correlation >= 0.7) {
              strokeColor = '#52c41a';
            }
            return (
              <marker
                key={`arrow-${index}`}
                id={`arrow-${index}`}
                viewBox="0 0 10 10"
                refX="5"
                refY="5"
                markerWidth="6"
                markerHeight="6"
                orient="auto"
              >
                <path d="M 0 0 L 10 5 L 0 10 z" fill={strokeColor} />
              </marker>
            );
          })}
        </defs>
        {links.map((link, index) => {
          const sourceNode = nodeObjects.find(n => n.id === link.source);
          const targetNode = nodeObjects.find(n => n.id === link.target);
          
          if (!sourceNode || !targetNode) return null;
          
          let strokeColor = '#d9d9d9';
          let strokeWidth = 1;
          if (link.correlation >= 0.9) {
            strokeColor = '#1890ff';
            strokeWidth = 3;
          } else if (link.correlation >= 0.7) {
            strokeColor = '#52c41a';
            strokeWidth = 2;
          }
          
          const dx = targetNode.x - sourceNode.x;
          const dy = targetNode.y - sourceNode.y;
          const angle = Math.atan2(dy, dx);
          const offset = links.some(l => l.source === link.target && l.target === link.source) ? 15 : 0;
          const perpX = Math.sin(angle) * offset;
          const perpY = -Math.cos(angle) * offset;
          const midX = (sourceNode.x + targetNode.x) / 2;
          const midY = (sourceNode.y + targetNode.y) / 2;
          const labelX = midX + perpX;
          const labelY = midY + perpY;
          
          return (
            <g key={`link-${index}`}>
              <line 
                x1={sourceNode.x} 
                y1={sourceNode.y} 
                x2={targetNode.x} 
                y2={targetNode.y} 
                stroke={strokeColor}
                strokeWidth={strokeWidth}
                strokeOpacity={0.8}
                markerEnd={`url(#arrow-${index})`}
              />
              <rect
                x={labelX - 16}
                y={labelY - 10}
                width={32}
                height={20}
                rx={4}
                fill="white"
                stroke={strokeColor}
                strokeWidth={1}
                opacity={0.9}
              />
              <text 
                x={labelX} 
                y={labelY}
                textAnchor="middle"
                dominantBaseline="central"
                fontSize="11"
                fontWeight="bold"
                fill="#666"
              >
                {link.correlation.toFixed(2)}
              </text>
            </g>
          );
        })}
        {nodeObjects.map((node, index) => (
          <g key={`node-${index}`}>
            <defs>
              <radialGradient id={`gradient-${index}`} cx="50%" cy="50%" r="50%" fx="50%" fy="50%">
                <stop offset="0%" stopColor={node.hasAnomaly ? '#ff7875' : '#fff'} />
                <stop offset="100%" stopColor={node.hasAnomaly ? '#ff4d4f' : '#f0f0f0'} />
              </radialGradient>
            </defs>
            
            <circle 
              cx={node.x} 
              cy={node.y} 
              r={node.hasAnomaly ? 28 : 22}
              fill={`url(#gradient-${index})`}
              stroke={node.hasAnomaly ? '#ff4d4f' : '#d9d9d9'}
              strokeWidth={node.hasAnomaly ? 2 : 1}
              filter="drop-shadow(0px 2px 3px rgba(0,0,0,0.1))"
            />
            
            <text 
              x={node.x} 
              y={node.y}
              textAnchor="middle"
              dominantBaseline="central"
              fontSize="11"
              fontWeight="bold"
              fill={node.hasAnomaly ? '#fff' : '#333'}
            >
              {node.id.match(/\d+/)[0]}
            </text>
            
            <text 
              x={node.x} 
              y={node.y + (node.hasAnomaly ? 40 : 35)}
              textAnchor="middle"
              fontSize="10"
              fill="#666"
            >
              {node.fullName}
            </text>
            
            {node.hasAnomaly && (
              <text 
                x={node.x} 
                y={node.y + 55}
                textAnchor="middle"
                fontSize="10"
                fontWeight="bold"
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
export const getAnomalyDistribution = (comparisonData, selectedModels, thresholdValue = 0.1) => {
  if (!comparisonData || typeof comparisonData !== 'object' || Object.keys(comparisonData).length === 0 || !selectedModels.length) return [];
  
  const allTypes = new Set();
  const typeCountByModel = {};
  
  selectedModels.forEach(model => {
    if (!comparisonData[model] || !comparisonData[model].ranks) {
      return;
    }
    typeCountByModel[model] = {};
    const filteredRanks = (comparisonData[model].ranks || []).filter(rank => rank.positive_prob_score >= thresholdValue);
    const typeCount = filteredRanks.filter(rank => rank.type).reduce((counts, rank) => {
      if (!counts[rank.type]) counts[rank.type] = 0;
      counts[rank.type]++;
      allTypes.add(rank.type);
      return counts;
    }, {});
    typeCountByModel[model] = typeCount;
  });
  
  return Array.from(allTypes).map(type => {
    const result = { type };
    selectedModels.forEach(model => {
      result[model] = typeCountByModel[model]?.[type] || 0;
    });
    return result;
  });
};

// Get timeline data for charts
export const getTimelineData = (comparisonData, selectedModels, thresholdValue = 0.1) => {
  if (!comparisonData || typeof comparisonData !== 'object' || Object.keys(comparisonData).length === 0) return [];
  
  // Create timeline points at 5-minute intervals
  const timePoints = [];
  const now = Date.now();
  
  for (let i = 12; i >= 0; i--) {
    timePoints.push(now - i * 5 * 60 * 1000);
  }
  
  return timePoints.map((timestamp, index) => {
    const result = { timestamp };
    
    selectedModels.forEach(model => {
      if (!comparisonData[model] || !comparisonData[model].ranks) {
        result[model] = 0;
        return;
      }
      
      // Filter anomalies by threshold
      const filteredRanks = (comparisonData[model].ranks || []).filter(rank => rank.positive_prob_score >= thresholdValue);
      
      // Count actual anomalies - in a real system, you would use timestamps of when anomalies occurred
      // Since we don't have real timestamps in this demo, we'll populate recent time points with actual anomaly counts
      // and leave earlier points at zero to show a trend
      if (index > 6) { // Show anomalies in recent time points
        result[model] = filteredRanks.length;
      } else {
        result[model] = 0; // No anomalies in earlier time points
      }
      
      // Set trend data based on actual anomaly counts instead of artificial formulas
      if (index === 0) {
        result[`${model}_trend`] = 0;
      } else {
        // Trend shows the change in anomaly count from previous time point
        const prevCount = index > 7 ? filteredRanks.length : 0;
        const currentCount = index > 6 ? filteredRanks.length : 0;
        result[`${model}_trend`] = currentCount - prevCount;
      }
      
      // Add feature data if available
      if (filteredRanks.length > 0 && filteredRanks[0].features) {
        const features = filteredRanks[0].features;
        Object.entries(features).forEach(([feature, value], featureIndex) => {
          if (featureIndex < 3) {
            // Only show feature values for recent time points where anomalies are present
            result[`${model}_${feature}`] = index > 6 ? value : 0;
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