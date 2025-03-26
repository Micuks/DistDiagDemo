// Helper function to determine color for rank types
export const getRankColor = (type) => {
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

// Add debounce utility 
export const debounce = (func, wait) => {
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

// Helper function to determine model advantages
export const getModelAdvantages = (model, response, analysisTime) => {
  const advantages = [];
  
  // Speed comparison (DBSherlock baseline ~5000ms)
  if (analysisTime < 3000) {
    advantages.push({
      type: 'speed',
      icon: '🚀',
      text: `${Math.round(analysisTime/100)/10}s response time`
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
export const getCompoundAnomalyDescription = (anomalies) => {
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

// Helper function for model colors
export const getModelColor = (model, index) => {
  const colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
  ];
  
  // If index is provided, use it directly
  if (index !== undefined) {
    return colors[index % colors.length];
  }
  
  // Otherwise hash the model name to get a consistent color
  const hashCode = model.split('').reduce((acc, char) => {
    return char.charCodeAt(0) + ((acc << 5) - acc);
  }, 0);
  
  return colors[Math.abs(hashCode) % colors.length];
};

// Helper function to get color for metric category
export const getCategoryColor = (category) => {
  if (!category) return "default";
  
  switch (category.toLowerCase()) {
    case 'cpu': return "red";
    case 'memory': return "blue";
    case 'io': return "orange";
    case 'network': return "green";
    case 'disk': return "purple";
    case 'transaction': return "cyan";
    case 'transactions': return "cyan"; 
    default: return "default";
  }
};

// Helper function to get hex color for pie chart
export const getCategoryColorHex = (category) => {
  if (!category) return "#999";
  
  switch (category.toLowerCase()) {
    case 'cpu': return "#ff4d4f";
    case 'memory': return "#1890ff";
    case 'io': return "#fa8c16";
    case 'network': return "#52c41a";
    case 'disk': return "#722ed1";
    case 'transaction': return "#13c2c2";
    case 'transactions': return "#13c2c2";
    default: return "#999";
  }
};

// Helper function to get status for progress indicator
export const getScoreStatus = (score) => {
  if (!score && score !== 0) return "normal";
  if (score > 0.7) return "exception";
  if (score > 0.5) return "warning";
  return "normal";
};

// Helper function to calculate category distribution
export const getCategoryDistribution = (metrics) => {
  if (!metrics || !Array.isArray(metrics)) return [];
  
  const categoryCount = {};
  metrics.forEach(metric => {
    const category = metric.category || 'unknown';
    categoryCount[category] = (categoryCount[category] || 0) + 1;
  });
  
  const total = metrics.length;
  return Object.entries(categoryCount).map(([name, count]) => ({ 
    name, 
    value: Math.round((count / total) * 100) // Convert to percentage
  }));
};

// Custom label renderer for pie chart
export const renderCustomizedLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, percent, index, name }) => {
  const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
  const x = cx + radius * Math.cos(-midAngle * Math.PI / 180);
  const y = cy + radius * Math.sin(-midAngle * Math.PI / 180);
  
  return percent > 0.05 ? (
    <text x={x} y={y} fill="white" textAnchor="middle" dominantBaseline="central">
      {`${name} ${(percent * 100).toFixed(0)}%`}
    </text>
  ) : null;
};

// Helper function to generate recommended actions
export const getRecommendedActions = (metrics) => {
  if (!metrics || !Array.isArray(metrics) || metrics.length === 0) {
    return ["No specific actions recommended at this time."];
  }
  
  const actions = [];
  
  // Group by category for consolidated recommendations
  const categoryMetrics = {};
  metrics.forEach(metric => {
    const category = metric.category || 'unknown';
    if (!categoryMetrics[category]) {
      categoryMetrics[category] = [];
    }
    categoryMetrics[category].push(metric);
  });
  
  // Generate recommendations by category
  Object.entries(categoryMetrics).forEach(([category, catMetrics]) => {
    const catLower = category.toLowerCase();
    
    // Create metric names string
    const metricNames = catMetrics.map(m => m.name).join(', ');
    
    switch (catLower) {
      case 'cpu':
        actions.push(`Investigate CPU utilization patterns for potential bottlenecks related to ${metricNames}.`);
        if (catMetrics.some(m => (m.details?.trend || 0) > 0.5)) {
          actions.push("Consider scaling CPU resources or optimizing CPU-intensive operations.");
        }
        break;
      case 'memory':
        actions.push(`Monitor memory allocation patterns for ${metricNames}.`);
        if (catMetrics.some(m => (m.details?.volatility || 0) > 0.5)) {
          actions.push("Check for memory leaks or inefficient memory usage patterns.");
        }
        break;
      case 'io':
        actions.push(`Analyze I/O patterns for potential bottlenecks in ${metricNames}.`);
        if (catMetrics.some(m => (m.details?.z_score || 0) > 0.7)) {
          actions.push("Consider optimizing database queries or storage access patterns.");
        }
        break;
      case 'network':
        actions.push(`Examine network traffic patterns associated with ${metricNames}.`);
        if (catMetrics.some(m => (m.details?.change_ratio || 0) > 0.5)) {
          actions.push("Review network configurations and consider bandwidth optimization.");
        }
        break;
      case 'transactions':
        actions.push(`Monitor transactions metrics: ${metricNames} for anomalous behavior.`);
        actions.push("Consider analyzing request queue and processing time patterns.");
        break;
      default:
        actions.push(`Monitor ${category} metrics: ${metricNames} for anomalous behavior.`);
    }
  });
  
  return actions.length > 0 ? actions : ["No specific actions recommended at this time."];
}; 