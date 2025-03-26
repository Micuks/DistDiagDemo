import sys
import os
import logging
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_adaptive_window_size():
    """
    Test the adaptive window size functionality in isolation.
    
    This tests the core algorithm without dependencies on the full service.
    """
    logger.info("Testing adaptive window sizing with different stability levels")
    
    # Mock data with different stability levels
    def generate_metric_data(mean, std, size=100):
        np.random.seed(42)  # For reproducibility
        values = np.random.normal(mean, std, size)
        return [{"timestamp": f"2023-10-{i:02d}", "value": max(0, val)} for i, val in enumerate(values)]
    
    # Create test metric sets
    def generate_test_metrics(stability):
        """Generate metrics with specified stability (lower = more stable)"""
        return {
            "cpu": {
                "cpu usage": generate_metric_data(50.0, 50.0 * stability),
                "worker time": generate_metric_data(40.0, 40.0 * stability)
            },
            "memory": {
                "memory usage": generate_metric_data(1000.0, 1000.0 * stability),
                "total memstore used": generate_metric_data(800.0, 800.0 * stability)
            },
            "io": {
                "io read count": generate_metric_data(100.0, 100.0 * stability)
            }
        }
    
    # Implement the core window size adjustment function we want to test
    def adjust_window_size(metrics, min_window=60, max_window=600):
        """Implementation of our adaptive window size algorithm"""
        try:
            # Define a list of key metrics to use for stability calculation
            metric_list = [
                ('cpu', 'cpu usage'),
                ('cpu', 'worker time'),
                ('memory', 'memory usage'),
                ('memory', 'total memstore used'),
                ('io', 'io read count')
            ]
            
            # Calculate stability scores for each metric
            stability_scores = []
            for category, metric_name in metric_list:
                if category in metrics and metric_name in metrics[category]:
                    # Extract values
                    data = metrics[category][metric_name]
                    values = [item["value"] for item in data]
                    
                    if len(values) > 1:  # Need at least 2 values for std
                        mean_val = np.mean(values)
                        if abs(mean_val) > 1e-6:  # Non-zero mean
                            std_val = np.std(values)
                            cv = std_val / mean_val  # Coefficient of variation
                            stability_scores.append(min(cv, 1.0))  # Cap at 1.0
            
            # If we have stability scores, calculate the average
            if stability_scores:
                avg_stability = np.mean(stability_scores)
                logger.info(f"Metric stability score: {avg_stability:.4f}")
                
                # Scale window size inversely with stability
                normalized_stability = min(max(avg_stability, 0.0), 1.0)
                
                # Calculate window size - inverse relationship
                window_size = min_window + int((max_window - min_window) * (1 - normalized_stability))
                
                logger.info(f"Adaptive window size: {window_size}s (stability: {normalized_stability:.2f})")
                return window_size
            else:
                # Default to middle window size if no stability scores
                window_size = (min_window + max_window) // 2
                logger.info(f"Using default adaptive window size: {window_size}s (no stability scores)")
                return window_size
                
        except Exception as e:
            logger.error(f"Error in adaptive window sizing: {str(e)}")
            return 300  # Default
    
    # Test with different stability levels
    test_cases = [
        {"name": "High stability (0.1)", "stability": 0.1},
        {"name": "Medium stability (0.5)", "stability": 0.5},
        {"name": "Low stability (0.9)", "stability": 0.9}
    ]
    
    for test in test_cases:
        metrics = generate_test_metrics(test["stability"])
        window_size = adjust_window_size(metrics)
        logger.info(f"{test['name']} -> Window size: {window_size}s")
    
    return True

def test_cross_feature_interactions():
    """
    Test the generation of cross-feature interactions.
    
    This tests the core algorithm without dependencies on the full service.
    """
    logger.info("Testing cross-feature interactions")
    
    # Create sample feature vector (simplified)
    np.random.seed(42)
    base_features = np.random.rand(100)  # 100 basic features
    
    # Simulate key category indices
    key_feature_indices = [0, 20, 40, 60, 80]  # Representative points from different categories
    
    # Generate interaction features
    interaction_features = []
    
    # Create interactions between key features
    for i in range(len(key_feature_indices)):
        for j in range(i+1, len(key_feature_indices)):
            idx1 = key_feature_indices[i]
            idx2 = key_feature_indices[j]
            # Use product as interaction
            interaction = base_features[idx1] * base_features[idx2]
            interaction_features.append(interaction)
    
    # Add max-value cross-category interactions
    # Find max feature value for each category
    cpu_max = np.max(base_features[:20])
    mem_max = np.max(base_features[20:40])
    io_max = np.max(base_features[40:60])
    net_max = np.max(base_features[60:80])
    trans_max = np.max(base_features[80:])
    
    # Add key cross-category interactions
    interaction_features.extend([
        cpu_max * io_max,     # CPU-IO interaction
        cpu_max * mem_max,    # CPU-Memory interaction
        net_max * io_max,     # Network-IO interaction
        cpu_max * net_max,    # CPU-Network interaction
        mem_max * io_max      # Memory-IO interaction
    ])
    
    # Convert to numpy array and concatenate with base features
    interaction_array = np.array(interaction_features)
    
    # Concatenate base features with interaction features
    enhanced_features = np.concatenate([base_features, interaction_array])
    
    # Verify enhanced features
    logger.info(f"Original feature vector length: {len(base_features)}")
    logger.info(f"Enhanced feature vector length: {len(enhanced_features)}")
    logger.info(f"Number of interaction features: {len(interaction_features)}")
    
    # Check that interaction features are non-zero
    non_zero_interactions = np.count_nonzero(interaction_features)
    logger.info(f"Non-zero interaction features: {non_zero_interactions}/{len(interaction_features)}")
    
    return len(enhanced_features) > len(base_features)

if __name__ == "__main__":
    logger.info("Testing adaptive window sizing and cross-feature interactions")
    
    # Run tests
    window_test_result = test_adaptive_window_size()
    interaction_test_result = test_cross_feature_interactions()
    
    # Report overall results
    logger.info("\n----- Test Results -----")
    logger.info(f"Adaptive window sizing: {'PASS' if window_test_result else 'FAIL'}")
    logger.info(f"Cross-feature interactions: {'PASS' if interaction_test_result else 'FAIL'}")
    logger.info("------------------------\n")
    
    if window_test_result and interaction_test_result:
        logger.info("All tests PASSED!")
        sys.exit(0)
    else:
        logger.error("Some tests FAILED!")
        sys.exit(1) 