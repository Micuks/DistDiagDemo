#!/usr/bin/env python3

import asyncio
import logging
import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from dotenv import load_dotenv
from tests.setup_logging import setup_logging


# Add project root to Python path to allow imports from backend.app
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import your actual services
from backend.app.services.k8s_service import K8sService, k8s_service
from backend.app.services.metrics_service import MetricsService, metrics_service

logger = setup_logging()

# Load environment variables from .env file
env_path = Path(__file__).resolve().parent.parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    logger.info(f"Loaded environment variables from {env_path}")
else:
    logger.warning(f".env file not found at {env_path}")

def calculate_average_metric(history: list, start_time: datetime, end_time: datetime) -> float:
    """Calculates the average value of a metric history list within a time window."""
    relevant_values = []
    if not history or not isinstance(history, list):
        logger.warning(f"Invalid history format or empty history: {history}")
        return 0.0

    for point in history:
        try:
            # Ensure point is a dict with 'timestamp' and 'value'
            if not isinstance(point, dict) or 'timestamp' not in point or 'value' not in point:
                # logger.debug(f"Skipping invalid point format: {point}")
                continue

            ts_str = point['timestamp']
            # Attempt to parse ISO format timestamp, assuming UTC if no offset
            ts = datetime.fromisoformat(ts_str)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc) # Assume UTC if naive

            # Ensure start/end times are also timezone-aware (UTC)
            aware_start_time = start_time if start_time.tzinfo else start_time.replace(tzinfo=timezone.utc)
            aware_end_time = end_time if end_time.tzinfo else end_time.replace(tzinfo=timezone.utc)

            if aware_start_time <= ts <= aware_end_time:
                value = point['value']
                # Ensure value is numeric
                if isinstance(value, (int, float)):
                    relevant_values.append(float(value))
                else:
                    # logger.debug(f"Skipping non-numeric value in point: {point}")
                    pass # Skip non-numeric values silently for now

        except (ValueError, KeyError, TypeError) as e:
            # Log specific errors if needed, but continue processing other points
            # logger.warning(f"Could not parse timestamp/value or unexpected error: {point}, Error: {e}")
            continue # Skip this point

    if not relevant_values:
        # logger.debug(f"No relevant metric values found between {start_time} and {end_time}")
        return 0.0

    return sum(relevant_values) / len(relevant_values)


async def test_anomaly_effect(
    anomaly_type: str,
    severity: str = "medium",
    target_node: str = None,
    test_duration: int = 30
) -> Dict[str, Any]:
    """
    Test the effect of an anomaly by applying it, monitoring metrics, and comparing results.
    
    Args:
        anomaly_type: Type of anomaly to test
        severity: Severity level (low/medium/high)
        target_node: Target node name (if applicable)
        test_duration: Duration of the test in seconds
        
    Returns:
        Dictionary containing test results and metrics comparison
    """
    logger.info(f"Starting anomaly effect test for {anomaly_type} ({severity})")
    
    # --- Wait for initial metrics collection ---
    # Give MetricsService time to populate history before measuring baseline
    # Collection interval is 5s, wait slightly longer
    logger.info(f"Waiting {metrics_service.collection_interval + 2} seconds for initial metrics collection...")
    await asyncio.sleep(metrics_service.collection_interval + 2)
    
    # Define time windows
    baseline_start_time = datetime.now(timezone.utc)
    # Get baseline metrics history
    logger.info("Collecting baseline metrics history...")
    baseline_metrics_history = await metrics_service.get_detailed_metrics()
    baseline_end_time = datetime.now(timezone.utc)
    logger.info(f"Baseline period: {baseline_start_time} to {baseline_end_time}")
    
    # Check if baseline history is empty
    if not baseline_metrics_history:
        logger.error("Baseline metrics history is empty! Cannot proceed with comparison.")
        # Optionally raise an error or return a specific result indicating failure
        # raise ValueError("Baseline metrics history is empty")
        return { 
             "anomaly_type": anomaly_type, "severity": severity, "target_node": target_node,
             "error": "Baseline metrics history was empty after initial wait."
        }
    
    # Apply the anomaly
    anomaly_apply_time = datetime.now(timezone.utc)
    logger.info(f"Applying {anomaly_type} anomaly at {anomaly_apply_time}...")
    await k8s_service.apply_chaos_experiment(
        experiment_type=anomaly_type,
        severity=severity,
        target_node=target_node
    )
    
    # Wait for anomaly to take effect - this defines the anomaly measurement period
    anomaly_period_start_time = datetime.now(timezone.utc)
    logger.info(f"Waiting {test_duration} seconds for anomaly effect measurement...")
    await asyncio.sleep(test_duration)
    anomaly_period_end_time = datetime.now(timezone.utc)
    logger.info(f"Anomaly measurement period: {anomaly_period_start_time} to {anomaly_period_end_time}")
    
    # Get metrics history again (includes baseline + anomaly period)
    logger.info("Collecting full metrics history post-anomaly...")
    full_metrics_history = await metrics_service.get_detailed_metrics()
    
    # Compare metrics using averages from the defined periods
    logger.info("Comparing metrics using averages...")
    result = {
        "anomaly_type": anomaly_type,
        "severity": severity,
        "target_node": target_node,
        "test_duration": test_duration,
        # Keep raw history for debugging if needed, but comparison uses averages
        # "baseline_metrics_history": baseline_metrics_history, # Can be large, maybe omit
        # "full_metrics_history": full_metrics_history, # Can be large, maybe omit
        "effect_detected": False,
        "metric_changes": {},
        "comparison_errors": [] # Add a field for potential errors during comparison
    }
    
    # Define metric thresholds for different anomaly types
    # Thresholds represent the minimum *relative* increase (e.g., 0.1 = 10%)
    # OR *absolute* value for specific metrics (e.g., clock_drift seconds)
    thresholds = {
        "cpu_stress": {
            "cpu usage": 0.1,  # 10% relative increase
            "worker time": 0.1, # 10% relative increase
            "cpu time": 0.1, # 10% relative increase
            # "query_latency": 0.2  # Placeholder, needs actual metric
        },
        "io_bottleneck": {
            "io read count": 0.2, # 20% relative increase
            "io write count": 0.2, # 20% relative increase
            "io read delay": 0.3, # 30% relative increase
            "io write delay": 0.3, # 30% relative increase
            # "query_latency": 0.3  # Placeholder
        },
        "network_bottleneck": {
            "rpc net delay": 0.2, # 20% relative increase
            "rpc net frame delay": 0.2, # 20% relative increase
            # "query_latency": 0.2  # Placeholder
        },
        "time_skew": {
            # "clock_drift": 1.0,  # 1 second absolute difference - NEEDS ACTUAL METRIC IMPLEMENTATION
            # "query_latency": 0.1  # Placeholder
        },
        "replication_lag": {
            # "replication_delay": 1.0,  # 1 second absolute difference - NEEDS ACTUAL METRIC IMPLEMENTATION
            # "query_latency": 0.2  # Placeholder
        },
        "consensus_delay": {
            # "consensus_latency": 0.5,  # 500ms absolute difference - NEEDS ACTUAL METRIC IMPLEMENTATION
            # "query_latency": 0.2  # Placeholder
        },
        "too_many_indexes": {
            # "query_latency": 0.3,  # Placeholder
            # "disk usage": 0.1 # Placeholder - NEEDS ACTUAL METRIC (e.g. observer memory hold size?)
        }
        # Add other essential metrics to thresholds if needed
    }
    
    # Map known absolute threshold metrics - Add actual metric names here when implemented
    absolute_threshold_metrics = { # e.g., "clock_drift_metric_name"
        }
    
    # Compare metrics based on anomaly type
    if anomaly_type in thresholds:
        anomaly_thresholds = thresholds[anomaly_type]
        
        # Iterate through all nodes present in the baseline metrics history
        nodes_to_compare = list(baseline_metrics_history.keys())
        if not nodes_to_compare:
             result["comparison_errors"].append("No nodes found in baseline metrics history.")
             logger.warning("No nodes found in baseline metrics history to compare.")
        
        for node_ip in nodes_to_compare:
            node_metric_changes = {} # Store changes for this node
            
            # Ensure node exists in the full history (it should, but check)
            if node_ip not in full_metrics_history:
                err_msg = f"Node {node_ip} missing in full metrics history post-anomaly."
                result["comparison_errors"].append(err_msg)
                logger.warning(err_msg)
                continue
            
            baseline_node_hist = baseline_metrics_history.get(node_ip, {})
            full_node_hist = full_metrics_history.get(node_ip, {})
            
            for metric_name_threshold, threshold_value in anomaly_thresholds.items():
                 # Use lowercase, stripped metric name for matching
                metric_name_clean = metric_name_threshold.lower().strip()
                
                # Find the category for the metric using the service's mapping
                metric_category = metrics_service.metric_to_category.get(metric_name_clean)
                
                if not metric_category:
                    err_msg = f"Could not find category for metric '{metric_name_clean}' (threshold key: '{metric_name_threshold}') on node {node_ip} using metrics_service mapping."
                    result["comparison_errors"].append(err_msg)
                    logger.warning(err_msg)
                    continue
                
                # Get specific metric history lists - handle missing category or metric gracefully
                # Use the original (non-cleaned) metric name key from thresholds to access history dict
                baseline_metric_hist = baseline_node_hist.get(metric_category, {}).get(metric_name_threshold, [])
                full_metric_hist = full_node_hist.get(metric_category, {}).get(metric_name_threshold, [])
                
                # Calculate average values for the specific time windows
                avg_baseline_value = calculate_average_metric(baseline_metric_hist, baseline_start_time, baseline_end_time)
                avg_anomaly_value = calculate_average_metric(full_metric_hist, anomaly_period_start_time, anomaly_period_end_time)
                
                # Perform comparison using averages
                change_pct = 0.0
                absolute_change = avg_anomaly_value - avg_baseline_value
                is_significant = False
                
                try:
                    if metric_name_clean in absolute_threshold_metrics:
                        # Compare absolute change against threshold
                        is_significant = abs(absolute_change) > threshold_value
                        # Calculate relative change for info, avoid division by zero
                        if abs(avg_baseline_value) > 1e-9:
                           change_pct = (absolute_change / avg_baseline_value) * 100
                        elif abs(avg_anomaly_value) > 1e-9:
                           change_pct = float('inf') # Indicate large relative change from zero baseline
                    else:
                        # Compare relative change (increase) against threshold
                        if abs(avg_baseline_value) > 1e-9:
                            relative_change = absolute_change / avg_baseline_value
                            change_pct = relative_change * 100
                            # Check if relative increase exceeds threshold
                            is_significant = relative_change > threshold_value
                        elif avg_anomaly_value > 1e-9: # Baseline is zero/small, anomaly is positive
                             change_pct = float('inf')
                             # Consider this significant if threshold is positive
                             is_significant = threshold_value >= 0 # Significant if any positive value appears from zero
                        # else: baseline and anomaly are both near zero, relative change is ~0

                    change_details = {
                        "baseline_avg": round(avg_baseline_value, 4),
                        "anomaly_avg": round(avg_anomaly_value, 4),
                        "absolute_change": round(absolute_change, 4),
                        "change_pct": round(change_pct, 2) if change_pct != float('inf') else 'inf',
                        "threshold_value": threshold_value,
                        "threshold_type": "absolute" if metric_name_clean in absolute_threshold_metrics else "relative_increase",
                        "significant_change": is_significant
                    }
                    node_metric_changes[metric_name_threshold] = change_details
                    
                    if is_significant:
                        result["effect_detected"] = True
                        logger.info(f"Significant change detected for '{metric_name_threshold}' on {node_ip}: {change_details}")

                except Exception as e:
                    err_msg = f"Error comparing metric '{metric_name_threshold}' on {node_ip}: {e}"
                    result["comparison_errors"].append(err_msg)
                    logger.error(err_msg)
                    node_metric_changes[metric_name_threshold] = {"error": str(e)}
            
            # Add this node's comparison results
            if node_metric_changes:
                 result["metric_changes"][node_ip] = node_metric_changes

    else:
        err_msg = f"No defined thresholds for anomaly type '{anomaly_type}'. Cannot perform comparison."
        result["comparison_errors"].append(err_msg)
        logger.warning(err_msg)
    
    # Clean up the anomaly
    logger.info("Cleaning up anomaly...")
    await k8s_service.delete_chaos_experiment(
        experiment_type=anomaly_type,
        target_node=target_node # Pass target_node for cleanup too
    )
    
    # Wait for cleanup to potentially reflect in metrics if checked immediately after
    await asyncio.sleep(5)
    
    logger.info(f"Test completed for {anomaly_type}. Effect detected: {result['effect_detected']}")
    return result

async def run_test(
    anomaly_type: str,
    severity: str = "medium",
    target_node: str = None,
    duration: int = 30
) -> Dict[str, Any]:
    """Helper function to run a single test and print results."""
    logger.info(f"=== Running test for {anomaly_type} ({severity}) ===")
    test_result = await test_anomaly_effect(
        anomaly_type=anomaly_type,
        severity=severity,
        target_node=target_node,
        test_duration=duration
    )
    print("\n--- Test Result ---")
    print(json.dumps(test_result, indent=2, default=str))
    print(f"=== Finished test for {anomaly_type} ({severity}) ===\n")
    return test_result

async def main():
    nodes = await k8s_service.get_available_nodes();
    # Ensure metrics service is running its collection
    await asyncio.sleep(5)
    
    # Define the tests to run
    tests = [
        # CPU stress tests
        ("cpu_stress", "low",nodes[0], 45),
        ("cpu_stress", "medium",nodes[0], 45),
        ("cpu_stress", "high",nodes[0], 45),
        
        # IO bottleneck tests
        ("io_bottleneck", "low",nodes[0], 60),
        ("io_bottleneck", "medium",nodes[0], 60),
        ("io_bottleneck", "high",nodes[0], 60),
        
        # Network bottleneck tests
        ("network_bottleneck", "low", None, 60),
        ("network_bottleneck", "medium", None, 60),
        ("network_bottleneck", "high", None, 60),
        
        # Time skew tests
        ("time_skew", "low",nodes[0], 60),
        ("time_skew", "medium",nodes[0], 60),
        ("time_skew", "high",nodes[0], 60),
        
        # Replication lag tests
        ("replication_lag", "low", None, 60),
        ("replication_lag", "medium", None, 60),
        ("replication_lag", "high", None, 60),
        
        # Consensus delay tests
        ("consensus_delay", "low", None, 60),
        ("consensus_delay", "medium", None, 60),
        ("consensus_delay", "high", None, 60),
        
        # Too many indexes test
        ("too_many_indexes", "high", None, 60)
    ]
    
    # Run all tests
    for anomaly_type, severity, target_node, duration in tests:
        try:
            await run_test(anomaly_type, severity, target_node, duration)
            # Add delay between tests
            await asyncio.sleep(10)
        except Exception as e:
            logger.error(f"Error running test for {anomaly_type} ({severity}): {str(e)}")
            continue
    
    logger.info("All tests finished.")

if __name__ == '__main__':
    asyncio.run(main()) 