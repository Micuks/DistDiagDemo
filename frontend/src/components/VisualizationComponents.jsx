import React from "react";
import { Empty } from "antd";
import { getModelColor, getFullNodeName } from "../utils/rankUtils.jsx";

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
    const radius = 200; // Slightly larger radius
    const angle = (2 * Math.PI * index) / nodes.length;
    const x = radius * Math.cos(angle) + 250; // Centered better
    const y = radius * Math.sin(angle) + 220;
    
    // Check if node has anomaly
    const hasAnomaly = comparisonData[model].ranks?.some(r => r.node === node);
    const anomalyType = hasAnomaly ? 
      comparisonData[model].ranks.find(r => r.node === node).type : null;
    
    // Get full node name
    const fullNodeName = getFullNodeName(node);
    
    return { id: node, fullName: fullNodeName, x, y, hasAnomaly, anomalyType };
  });
  
  // Create array of links
  const links = [];
  nodes.forEach(source => {
    const targets = propagationGraph[source];
    if (Array.isArray(targets)) {
      targets.forEach(target => {
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
      <div style={{ marginBottom: '15px' }}>
        <div style={{ display: 'inline-block', marginRight: '15px', padding: '4px 10px', background: '#ff4d4f', color: 'white', borderRadius: '4px', fontWeight: 'bold' }}>Anomaly Node</div>
        <div style={{ display: 'inline-block', marginRight: '15px', padding: '4px 10px', background: '#1890ff', color: 'white', borderRadius: '4px', fontWeight: 'bold' }}>Strong Correlation (≥ 0.9)</div>
        <div style={{ display: 'inline-block', padding: '4px 10px', background: '#52c41a', color: 'white', borderRadius: '4px', fontWeight: 'bold' }}>Medium Correlation (≥ 0.7)</div>
      </div>
      <svg width="550" height="500" style={{ border: '1px solid #f0f0f0', borderRadius: '8px', background: '#fafafa', boxShadow: '0 2px 8px rgba(0,0,0,0.06)' }}>
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
          
          // Check if this is a bidirectional link (the reverse link exists)
          const isReverseLinkPresent = links.some(l => 
            l.source === link.target && l.target === link.source
          );
          
          // Calculate label offset for bidirectional links
          const dx = targetNode.x - sourceNode.x;
          const dy = targetNode.y - sourceNode.y;
          const angle = Math.atan2(dy, dx);
          
          // Offset label position for bidirectional edges
          const offset = isReverseLinkPresent ? 15 : 0; // Offset in pixels
          const perpX = Math.sin(angle) * offset;
          const perpY = -Math.cos(angle) * offset;
          
          // Calculate midpoint of the line
          const midX = (sourceNode.x + targetNode.x) / 2;
          const midY = (sourceNode.y + targetNode.y) / 2;
          
          // Position label with offset
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
              
              {/* Arrow marker definition */}
              <defs>
                <marker
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
              </defs>
              
              {/* Add correlation label with background */}
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
        
        {/* Draw nodes */}
        {nodeObjects.map((node, index) => (
          <g key={`node-${index}`}>
            {/* Node circle with gradient */}
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
            
            {/* Node ID (shorter version, fits better) */}
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
            
            {/* Node full name label below circle */}
            <text 
              x={node.x} 
              y={node.y + (node.hasAnomaly ? 40 : 35)}
              textAnchor="middle"
              fontSize="10"
              fill="#666"
            >
              {node.fullName}
            </text>
            
            {/* Anomaly type label */}
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