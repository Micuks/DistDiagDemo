import numpy as np
import pandas as pd
from scipy import stats
import networkx as nx
import logging
from typing import Dict, List, Tuple, Any
import requests
import json
import re

logger = logging.getLogger(__name__)

def generate_MetricRooter_summary(ranked_metrics, node, languages=["Chinese", "English"]):
    """Generate a summary of the MetricRooter analysis results using qwq:32b model"""
    if not ranked_metrics:
        return {"summary": "No significant metrics found for analysis.", "actions": ["No specific actions recommended at this time."]}
    
    try:
        # Prepare data for the model
        # Get top metrics for the summary (limited to 5 for model input)
        top_metrics = ranked_metrics[:5]
        
        # Format metrics data for model input
        metrics_data = []
        for i, metric in enumerate(top_metrics):
            importance = "critical" if i == 0 else "significant" if i == 1 else "notable"
            metrics_data.append({
                "name": metric.get('name', 'unknown'),
                "category": metric.get('category', 'unknown'),
                "z_score": metric.get('z_score', 0),
                "score": metric.get('score', 0),
                "importance": importance,
                "details": metric.get('details', {})
            })
        
        # Prepare prompt for qwq:32b model
        prompt = f"""You are performing a root cause analysis for system metrics anomalies.

For node {node}, the following metrics have been identified as anomalous (ranked by PageRank score):

{json.dumps(metrics_data, indent=2)}

Please provide two sections:

SUMMARY:
A concise summary (maximum 200 words) that identifies the most likely root cause metrics, explains their significance, and their relationships.

RECOMMENDED ACTIONS:
A list of 3-5 specific, actionable recommendations to address the issues identified. Format as a JSON array of strings.

Focus on the top-ranked metrics, especially those showing high PageRank scores and z-scores.

Provide your response in the following languages: {languages}.
"""

        # Call Ollama API
        ollama_url = "http://localhost:11434/api/generate"
        response = requests.post(
            ollama_url,
            json={
                "model": "qwq:32b",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.95,
                }
            },
            timeout=20
        )
        
        # Check if request was successful
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get("response", "")
            
            # Trim any unnecessary text markers
            generated_text = generated_text.replace("```", "").strip()
            
            # Try to extract summary and actions from the generated text
            result = parse_llm_response(generated_text)
            
            logger.info(f"Generated summary and actions using qwq:32b model for node {node}")
            return result
            
        else:
            logger.error(f"Error calling Ollama API: {response.status_code} - {response.text}")
            # Fallback to original summary generation
            fallback_result = generate_fallback_summary_and_actions(ranked_metrics, node)
            return fallback_result
            
    except Exception as e:
        logger.error(f"Error generating summary with model: {str(e)}")
        # Fallback to original summary generation
        fallback_result = generate_fallback_summary_and_actions(ranked_metrics, node)
        return fallback_result

def parse_llm_response(text):
    """Parse the LLM response to extract summary and actions"""
    try:
        # Initialize with default values
        summary = text
        actions = []
        
        # Try to find SUMMARY and RECOMMENDED ACTIONS sections
        summary_match = re.search(r"SUMMARY:?\s*(.*?)(?:RECOMMENDED ACTIONS|$)", text, re.DOTALL)
        actions_match = re.search(r"RECOMMENDED ACTIONS:?\s*(.*?)(?:$)", text, re.DOTALL)
        
        if summary_match:
            summary = summary_match.group(1).strip()
        
        if actions_match:
            # Try to parse JSON array if present
            actions_text = actions_match.group(1).strip()
            try:
                # Check if it's already in JSON format
                if actions_text.startswith('[') and actions_text.endswith(']'):
                    actions = json.loads(actions_text)
                else:
                    # Otherwise, split by newlines and filter out empty lines
                    action_items = [line.strip() for line in actions_text.split('\n') 
                                  if line.strip() and not line.strip().startswith('SUMMARY')]
                    
                    # Clean up bullet points and numbering
                    actions = [re.sub(r'^[\d\.\-\*]+\s*', '', item).strip() for item in action_items]
                    actions = [item for item in actions if item]  # Filter empty items
            except:
                # If parsing fails, try to extract bullet points or numbered items
                actions = re.findall(r'[\d\.\-\*]+\s*(.*?)(?:\n|$)', actions_text)
                if not actions:
                    # Just split by newlines if no structure is detected
                    actions = [line.strip() for line in actions_text.split('\n') if line.strip()]
        
        # If no actions were found, extract from full text for fallback
        if not actions:
            # Look for bullet points or numbered items in the entire text
            actions = re.findall(r'[\d\.\-\*]+\s*(.*?)(?:\n|$)', text)
            
            # If still empty, create a generic recommendation
            if not actions:
                actions = ["Investigate the anomalous metrics identified in the summary."]
        
        return {
            "summary": summary,
            "actions": actions
        }
        
    except Exception as e:
        logger.error(f"Error parsing LLM response: {str(e)}")
        return {
            "summary": text,
            "actions": ["Investigate the anomalous metrics identified in the summary."]
        }

def generate_fallback_summary_and_actions(ranked_metrics, node):
    """Fallback summary and actions generation when model is unavailable"""
    # Get top 3 metrics for the summary
    top_metrics = ranked_metrics[:3]
    
    # Generate analysis summary
    summary = f"MetricRooter analysis for node {node} has identified the following key metrics with anomalous behavior: "
    
    metric_descriptions = []
    for i, metric in enumerate(top_metrics):
        importance = "critical" if i == 0 else "significant" if i == 1 else "notable"
        # Add more detailed description using the component scores
        details = metric.get('details', {})
        primary_factor = max(details.items(), key=lambda x: x[1])[0] if details else None
        
        factor_descriptions = {
            'trend': "showing significant trend",
            'volatility': "exhibiting high volatility",
            'z_score': "deviating from normal range",
            'change_ratio': "showing rapid change",
            'magnitude': "with high absolute value",
            'entropy': "displaying unpredictable pattern",
            'pagerank': "having high influence in the metrics network",
            'connections': "connected to many other anomalous metrics"
        }
        
        factor_desc = factor_descriptions.get(primary_factor, "")
        metric_descriptions.append(f"{metric['name']} ({metric['category']}, {importance} {factor_desc})")
    
    if len(metric_descriptions) > 1:
        summary += ", ".join(metric_descriptions[:-1]) + " and " + metric_descriptions[-1]
    elif metric_descriptions:
        summary += metric_descriptions[0]
    else:
        summary += "none found with significant deviation"
    
    # Generate actions based on metric categories
    actions = []
    # Group by category
    categories = set(metric.get('category', 'unknown') for metric in ranked_metrics[:5])
    
    for category in categories:
        if category == "cpu":
            actions.append(f"Investigate CPU usage patterns on node {node} and consider resource allocation.")
        elif category == "memory":
            actions.append(f"Examine memory allocation patterns on node {node} and look for potential memory leaks.")
        elif category == "io":
            actions.append(f"Analyze I/O patterns on node {node} and check for storage performance issues.")
        elif category == "network":
            actions.append(f"Check network configurations on node {node} and monitor traffic patterns.")
        elif category == "transaction":
            actions.append(f"Review transaction processing on node {node} and analyze request handling performance.")
    
    # Add a general recommendation
    if len(categories) > 1:
        actions.append(f"Since multiple categories ({', '.join(categories)}) are affected, perform a holistic system review of node {node}.")
    
    # Ensure we have at least one action
    if not actions:
        actions.append(f"Monitor the anomalous metrics on node {node} and investigate their behavior.")
    
    return {
        "summary": summary,
        "actions": actions
    }

def create_wudg_from_metrics(metrics: Dict[str, Dict], anomaly_threshold: float = 1.5) -> Tuple[nx.Graph, List[Dict]]:
    """
    Create a Weighted Undirected Dependency Graph (WUDG) from metrics data.
    
    Args:
        metrics: Dictionary of metrics data by category and metric name
        anomaly_threshold: Z-score threshold to identify anomalous metrics
        
    Returns:
        Tuple containing the WUDG graph and list of anomalous metrics metadata
    """
    try:
        # Extract metrics time series data
        anomalous_metrics = []
        metric_series = {}
        
        # Process metrics by category and find anomalous ones
        for category, metrics_dict in metrics.items():
            for metric_name, metric_data in metrics_dict.items():
                # Skip metrics without time series data
                if not isinstance(metric_data, list) or not metric_data:
                    continue
                
                # Extract time series values
                values = [point.get('value', 0) for point in metric_data if isinstance(point, dict)]
                if not values:
                    continue
                
                # Calculate statistics for anomaly detection
                mean = np.mean(values)
                std = np.std(values) if len(values) > 1 else 0
                
                # Compute a smoothed value using a rolling window average over recent data points
                window_size = 3
                if len(values) >= window_size:
                    smoothed_value = np.mean(values[-window_size:])
                else:
                    smoothed_value = np.mean(values)
                
                # Calculate z-score using the smoothed value
                z_score = 0
                if std > 0:
                    z_score = abs(smoothed_value - mean) / std
                
                # Store the time series for correlation analysis
                metric_key = f"{category}:{metric_name}"
                metric_series[metric_key] = values
                
                # Check if this metric is anomalous based on z-score
                if z_score > anomaly_threshold:
                    anomalous_metrics.append({
                        'key': metric_key,
                        'category': category,
                        'name': metric_name,
                        'z_score': z_score,
                        'latest_value': smoothed_value,
                        'mean': mean,
                        'std': std
                    })
        
        logger.info(f"Found {len(anomalous_metrics)} anomalous metrics out of {len(metric_series)} total metrics")
        
        if not anomalous_metrics or len(anomalous_metrics) < 2:
            logger.warning("Insufficient anomalous metrics to create WUDG")
            return nx.Graph(), anomalous_metrics
        
        # Create a complete graph with anomalous metrics
        G = nx.Graph()
        anomaly_keys = [m['key'] for m in anomalous_metrics]
        
        # Add nodes
        for metric in anomalous_metrics:
            G.add_node(metric['key'], **metric)
        
        # Add edges with Fisher-Z independence test
        for i in range(len(anomaly_keys)):
            for j in range(i+1, len(anomaly_keys)):
                key_i = anomaly_keys[i]
                key_j = anomaly_keys[j]
                
                # Get time series data
                series_i = metric_series[key_i]
                series_j = metric_series[key_j]
                
                # Ensure same length for correlation
                min_len = min(len(series_i), len(series_j))
                if min_len < 3:  # Need at least 3 points for Fisher-Z
                    continue
                
                series_i = series_i[-min_len:]
                series_j = series_j[-min_len:]
                
                # Calculate Pearson correlation
                r, p_value = stats.pearsonr(series_i, series_j)
                
                # Fisher-Z test
                # Zs = (√(m-3)/2) * log((1+r)/(1-r))
                # Where m is the sample size and r is Pearson correlation
                m = min_len
                if abs(r) < 1.0:  # Avoid numerical issues when r is close to 1 or -1
                    fisher_z = 0.5 * np.log((1 + r) / (1 - r)) * np.sqrt(m - 3)
                    
                    # Calculate p-value from Fisher-Z test
                    p_value = 2 * (1 - stats.norm.cdf(abs(fisher_z)))
                    
                    # Set edge weight as 1/p_value if significant
                    # Clip p_value to avoid division by zero
                    min_p_value = 1e-10
                    if p_value < 0.05:  # Significant dependency
                        edge_weight = 1.0 / max(p_value, min_p_value)
                        G.add_edge(key_i, key_j, weight=edge_weight, r=r, p_value=p_value)
        
        logger.info(f"Created WUDG with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G, anomalous_metrics
                
    except Exception as e:
        logger.error(f"Error creating WUDG: {str(e)}")
        return nx.Graph(), []

def rank_metrics_with_weighted_pagerank(G: nx.Graph, anomalous_metrics: List[Dict]) -> List[Dict]:
    """
    Rank metrics using weighted PageRank algorithm on the WUDG
    
    Args:
        G: Weighted Undirected Dependency Graph
        anomalous_metrics: List of anomalous metrics metadata
        
    Returns:
        List of ranked metrics with scores
    """
    try:
        if not G.nodes():
            logger.warning("Empty graph, cannot perform PageRank ranking")
            return anomalous_metrics
        
        # Create weight^2 dictionary for personalized PageRank
        weight_squared = {(u, v): data['weight']**2 for u, v, data in G.edges(data=True)}
        
        # Define personalized PageRank function with custom weights
        def weight_function(G, node):
            weights = {}
            for nbr in G.neighbors(node):
                weights[nbr] = weight_squared.get((node, nbr), weight_squared.get((nbr, node), 1.0))
            return weights
        
        # Calculate weighted PageRank
        d = 0.85  # Damping factor
        pr = nx.pagerank(G, alpha=d, personalization=None, weight=weight_function)
        
        # Organize results
        ranked_metrics = []
        for metric in anomalous_metrics:
            key = metric['key']
            if key in pr:
                # Add PageRank score to metric data
                ranked_metric = metric.copy()
                ranked_metric['score'] = pr[key]
                
                # Add component scores as details
                node_data = G.nodes[key] if key in G.nodes else {}
                ranked_metric['details'] = {
                    'pagerank': pr[key],
                    'z_score': metric['z_score'],
                    'connections': G.degree(key) if key in G.nodes else 0,
                }
                
                ranked_metrics.append(ranked_metric)
        
        # Sort by PageRank score (descending)
        ranked_metrics.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        logger.info(f"Metrics ranked by weighted PageRank: {len(ranked_metrics)} metrics")
        return ranked_metrics
        
    except Exception as e:
        logger.error(f"Error ranking metrics with PageRank: {str(e)}")
        return anomalous_metrics

def analyze_node_metrics(metrics: Dict[str, Dict], node: str) -> Dict:
    """
    Perform metric root cause analysis using WUDG and weighted PageRank
    
    Args:
        metrics: Node metrics from get_detailed_metrics
        node: Node name being analyzed
        
    Returns:
        Dictionary with ranked metrics and analysis results
    """
    try:
        # Create Weighted Undirected Dependency Graph
        wudg, anomalous_metrics = create_wudg_from_metrics(metrics)
        
        # Rank metrics using weighted PageRank
        ranked_metrics = rank_metrics_with_weighted_pagerank(wudg, anomalous_metrics)
        
        # Build graph representation for visualization
        graph_data = {
            "nodes": [],
            "links": []
        }
        
        for node_key in wudg.nodes():
            node_data = wudg.nodes[node_key]
            category, name = node_key.split(':', 1) if ':' in node_key else ('unknown', node_key)
            
            # Find the score from ranked metrics
            score = 0
            for m in ranked_metrics:
                if m['key'] == node_key:
                    score = m.get('score', 0)
                    break
                    
            graph_data["nodes"].append({
                "id": node_key,
                "name": name,
                "category": category,
                "score": score,
                "z_score": node_data.get('z_score', 0)
            })
        
        for u, v, data in wudg.edges(data=True):
            graph_data["links"].append({
                "source": u,
                "target": v,
                "weight": data.get('weight', 1.0),
                "correlation": data.get('r', 0)
            })
        
        return {
            "metrics": ranked_metrics,
            "summary": "",  # Empty summary initially
            "summary_pending": True,  # Indicate summary needs to be generated
            "node": node,
            "graph": graph_data,
            "anomaly_count": len(anomalous_metrics),
            "metric_count": len(metrics) if isinstance(metrics, dict) else 0
        }
        
    except Exception as e:
        logger.error(f"Error in metric root cause analysis: {str(e)}")
        return {
            "metrics": [],
            "summary": f"Error in analysis: {str(e)}",
            "summary_pending": False,
            "node": node,
            "graph": {"nodes": [], "links": []},
            "error": str(e)
        }

def get_metric_summary(ranked_metrics: List[Dict], node: str, languages=["Chinese", "English"]) -> Dict:
    """
    Get summary and actions for metrics analysis separately from the main analysis
    
    Args:
        ranked_metrics: List of ranked metrics from analyze_node_metrics
        node: Node name being analyzed
        languages: Languages to generate summary in
        
    Returns:
        Dictionary with generated summary and actions
    """
    try:
        return generate_MetricRooter_summary(ranked_metrics, node, languages)
    except Exception as e:
        logger.error(f"Error generating metrics summary: {str(e)}")
        return {
            "summary": f"Error generating summary: {str(e)}",
            "actions": ["Investigate the node for potential issues based on anomalous metrics."]
        }
