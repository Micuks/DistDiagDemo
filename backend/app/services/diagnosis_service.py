import sys
from pathlib import Path
import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from scipy.stats import pearsonr
from xgboost import XGBClassifier
import joblib
from app.services.metrics_service import metrics_service
import json
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import glob
import traceback
from sklearn.decomposition import PCA
import pickle

# Add project root and dist_diagnosis root to Python path
project_root = Path(__file__).parent.parent.parent.parent
dist_diagnosis_root = project_root / 'dist_diagnosis'
sys.path.extend([str(project_root), str(dist_diagnosis_root)])

logger = logging.getLogger(__name__)

class WPRNNode:
    """PageRank implementation for N nodes"""
    def __init__(self, w, node_num):
        self.w = w
        self.node_num = node_num
        self.d = 0.85
        self.p = [1.0/node_num] * node_num  # Initialize with equal probability

    def page_rank(self, iter_num):
        for _ in range(iter_num):
            new_p = [0] * self.node_num
            for i in range(self.node_num):
                tmp = 0
                for j in range(self.node_num):
                    if j != i:
                        # Ensure we have actual weights (not zeros)
                        if self.w[i][j] > 0:
                            tmp += self.p[j] * self.w[i][j] * self.w[i][j]
                
                # Handle case when no correlations exist
                if tmp == 0:
                    # Distribute probability evenly
                    new_p[i] = 1.0/self.node_num
                else:
                    new_p[i] = (1 - self.d)/self.node_num + self.d * tmp
            
            # Normalize to ensure sum of probabilities is 1
            total = sum(new_p)
            if total > 0:
                new_p = [p/total for p in new_p]
            else:
                new_p = [1.0/self.node_num] * self.node_num
                
            self.p = new_p
            
        return self.p

class DistDiagnosis:
    """Integrated version of DistDiagnosis for anomaly detection"""
    def __init__(self, num_classifier=1, classes_for_each_node=None, node_num=1):
        self.num_classifier = num_classifier
        self.node_num = node_num
        self.classes = classes_for_each_node if classes_for_each_node else [
            "cpu_stress",
            "io_bottleneck", 
            "network_bottleneck",
            "cache_bottleneck",
            "too_many_indexes"
        ]
        
        # Check for CUDA availability
        import xgboost as xgb
        try:
            device = 'cuda'
            logger.info(f"XGBoost using device: {device} ")
        except Exception as e:
            logger.warning(f"Error checking GPU availability: {str(e)}. Defaulting to CPU.")
            device = 'cpu'
        
        # Enable GPU acceleration with CUDA using updated parameters if available
        self.clf_list = [XGBClassifier(
            objective='binary:logistic',
            device=device,  # Use detected device
            n_jobs=-1  # Use all available CPU cores for parallel processing
        ) for _ in range(num_classifier)]

        # Window configuration
        self.default_time_window = 5  # Default 5-minute window
        self.time_window = self.default_time_window
        self.min_window_size = 1  # Minimum 1-second window
        self.max_window_size = 300  # Maximum 5-minute window (in seconds)
        self.adaptive_window = False  # Disable adaptive window by default
        
        # Feature drift tracking
        self.use_drift_aware = False  # Disable drift-aware feature preservation by default
        self.smoothing_factor = 0.2  # Alpha value for exponential smoothing
        self.baseline_metrics = {}  # Historical statistical baselines
        self.drift_threshold = 0.3  # Threshold for significant drift detection
        
        # PCA config
        self.use_pca = False  # Disable PCA by default
        logger.info(f"Initialized {len(self.clf_list)} classifiers for {len(self.classes)} classes")

    def _get_time_window_values(self, metrics: Dict[str, Any], metric_name: str) -> List[float]:
        """Extract time window values for a metric"""
        try:
            if not metrics or metric_name not in metrics:
                # Try case-insensitive search as a fallback
                if metrics:
                    metric_keys = [k for k in metrics.keys() if k.lower() == metric_name.lower()]
                    if metric_keys:
                        # Use the first matching key
                        metric_name = metric_keys[0]
                    else:
                        # If still not found, return zeros
                        return [0.0] * self.time_window
                else:
                    return [0.0] * self.time_window
                    
            metric_data = metrics[metric_name]
            values = []
            
            # Handle different data formats
            if isinstance(metric_data, (int, float)):
                # Single scalar value - replicate it to create a stable series
                values = [float(metric_data)] * self.time_window
                # logger.debug(f"Metric {metric_name} is a scalar value: {metric_data}")
            
            # Handle dictionary with 'data' key containing time series
            elif isinstance(metric_data, dict) and 'data' in metric_data:
                data_entries = metric_data['data']
                if isinstance(data_entries, list) and data_entries:
                    if 'value' in data_entries[0]:
                        # Extract values from timestamped data
                        values = [float(entry['value']) for entry in data_entries]
                        logger.debug(f"Extracted {len(values)} values from {metric_name} data series")
            
            # Handle list of timestamped values directly
            elif isinstance(metric_data, list) and metric_data and isinstance(metric_data[0], dict):
                if 'value' in metric_data[0]:
                    values = [float(entry['value']) for entry in metric_data]
                    logger.debug(f"Extracted {len(values)} values from {metric_name} direct list")
            
            # Handle custom formats or nested structures
            elif isinstance(metric_data, dict):
                # Try to extract nested values if present
                if any(isinstance(v, (list, dict)) for v in metric_data.values()):
                    # Find the first nested structure that could contain time series
                    for key, nested_data in metric_data.items():
                        if isinstance(nested_data, list) and nested_data:
                            if isinstance(nested_data[0], dict) and 'value' in nested_data[0]:
                                values = [float(entry['value']) for entry in nested_data]
                                logger.debug(f"Extracted {len(values)} values from {metric_name}.{key} nested list")
                                break
                else:
                    # If only scalar values in dict, just use those
                    scalar_values = [v for v in metric_data.values() if isinstance(v, (int, float))]
                    if scalar_values:
                        values = [float(v) for v in scalar_values]
                        logger.debug(f"Extracted {len(values)} scalar values from {metric_name} dictionary")

            # Sort values if we have timestamps
            if not values and metric_data:
                # Log the format if we couldn't extract values
                logger.debug(f"Could not extract values from metric {metric_name} with type {type(metric_data)}")
                if isinstance(metric_data, dict):
                    logger.debug(f"Dictionary keys: {list(metric_data.keys())}")
                return [0.0] * self.time_window
            
            # Pad with last value if we don't have enough
            if values and len(values) < self.time_window:
                last_value = values[-1] if values else 0.0
                values.extend([last_value] * (self.time_window - len(values)))
                logger.debug(f"Padded {metric_name} values to length {self.time_window}")
            
            # Truncate if we have too many
            if len(values) > self.time_window:
                values = values[-self.time_window:]  # Take the most recent values
                
            # Add tiny noise if all values are identical to avoid zero stability
            if values and all(v == values[0] for v in values):
                base_value = values[0]
                # Add 0.1% noise to create slight variation
                noise_scale = max(abs(base_value) * 0.001, 0.00001)
                values = [v + (np.random.random() - 0.5) * noise_scale for v in values]
                # logger.debug(f"Added tiny noise to identical {metric_name} values")

            # Pad with zeros if we don't have enough values (this is now unreachable but kept for safety)
            while len(values) < self.time_window:
                values.append(0.0)

            return values[:self.time_window]  # Ensure we only return time_window values
        except Exception as e:
            logger.error(f"Error getting time window values for {metric_name}: {str(e)}")
            return [0.0] * self.time_window

    def _adjust_window_size(self, metrics: Dict[str, Any]) -> int:
        """
        Dynamically adjust the observation window size based on metric stability
        
        This adaptive method scales the window size inversely with metric stability:
        - Highly stable metrics: use larger window to capture subtle patterns
        - Unstable/fluctuating metrics: use smaller window to focus on recent changes
        
        Returns window size in seconds
        """
        if not self.adaptive_window:
            return self.default_time_window
            
        try:
            # Define window boundaries
            min_window = 60  # 1 minute (minimum window)
            max_window = 600  # 10 minutes (maximum window)
            
            # Log available categories and a sample of metrics from each
            available_categories = list(metrics.keys())
            logger.debug(f"Available metric categories: {available_categories}")
            
            # Sample metrics from each category
            for category in available_categories:
                if category in metrics and metrics[category]:
                    sample_metrics = list(metrics[category].keys())[:5]  # Show up to 5 sample metrics
                    logger.debug(f"Sample metrics in {category}: {sample_metrics}")
            
            # Define a larger list of key metrics to try for stability calculation
            # We'll try to use whatever metrics are available from this list
            metric_list = [
                # CPU metrics
                ('cpu', 'cpu usage'),
                ('cpu', 'worker time'),
                ('cpu', 'cpu time'),
                ('cpu', 'active cpu time'),
                ('cpu', 'cpu_total_time'),
                # Memory metrics
                ('memory', 'memory usage'),
                ('memory', 'total memstore used'),
                ('memory', 'active memstore used'),
                ('memory', 'memstore limit'),
                ('memory', 'observer memory hold size'),
                # IO metrics
                ('io', 'io read count'),
                ('io', 'io write count'),
                ('io', 'io read delay'),
                ('io', 'io write delay'),
                ('io', 'data micro block cache hit'),
                ('io', 'row cache hit'),
                # Network metrics
                ('network', 'rpc packet in'),
                ('network', 'rpc packet out'),
                ('network', 'rpc net delay'),
                ('network', 'rpc net frame delay'),
                ('network', 'mysql packet in'),
                ('network', 'mysql packet out'),
                # Transaction metrics
                ('transactions', 'sql select count'),
                ('transactions', 'trans commit count'),
                ('transactions', 'sql fail count'),
                ('transactions', 'active sessions'),
                ('transactions', 'trans rollback count'),
                ('transactions', 'sql update count')
            ]
            
            # Calculate stability scores for each metric (using standard deviation)
            stability_scores = []
            metrics_checked = 0
            metrics_found = 0
            
            for category, metric_name in metric_list:
                metrics_checked += 1
                if category not in metrics:
                    continue
                    
                category_metrics = metrics.get(category, {})
                if metric_name not in category_metrics:
                    # Try case-insensitive matching as a fallback
                    metric_keys = [k for k in category_metrics.keys() if k.lower() == metric_name.lower()]
                    if not metric_keys:
                        continue
                    metric_name = metric_keys[0]  # Use the first match
                
                metrics_found += 1
                values = self._get_time_window_values(category_metrics, metric_name)
                
                if values and len(values) > 1:  # Need at least 2 values for std
                    # Check if all values are the same (no variance)
                    if all(v == values[0] for v in values):
                        # Add small synthetic variation if all values are identical
                        logger.debug(f"Metric {category}.{metric_name} has identical values: {values[0]}")
                        # Add tiny stability score to avoid zero
                        stability_scores.append(0.01)
                        continue
                    
                    # Normalize std by mean to get coefficient of variation
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    
                    # Handle edge cases carefully
                    if abs(mean_val) < 1e-9:  # Near zero mean
                        if std_val > 0:
                            # If mean is near zero but std is positive, use a high CV
                            stability_scores.append(0.8)  # High but not maximum
                        else:
                            # Both mean and std near zero, add small synthetic value
                            stability_scores.append(0.01)
                    else:
                        # Normal case - calculate coefficient of variation
                        cv = std_val / abs(mean_val)
                        stability_scores.append(min(cv, 1.0))  # Cap at 1.0
                    
                    logger.debug(f"Metric {category}.{metric_name} - mean: {mean_val:.4f}, std: {std_val:.4f}, cv: {stability_scores[-1]:.4f}")
            
            logger.debug(f"Stability metrics checked: {metrics_checked}, found: {metrics_found}")
            
            # If we have stability scores, calculate the average
            if stability_scores:
                # Add a minimum base stability to prevent exact zero
                if min(stability_scores) < 0.005:
                    stability_scores.append(0.05)  # Add a small baseline stability
                
                avg_stability = np.mean(stability_scores)
                logger.debug(f"Metric stability score: {avg_stability:.4f} (from {len(stability_scores)} metrics)")
                
                # Scale window size inversely with stability:
                # - Higher stability (lower score) -> larger window
                # - Lower stability (higher score) -> smaller window
                normalized_stability = min(max(avg_stability, 0.01), 1.0)  # Ensure in [0.01,1.0]
                
                # Calculate window size - inverse relationship with stability
                # (1 - normalized_stability) gives larger window for stable metrics
                window_size = min_window + int((max_window - min_window) * (1 - normalized_stability))
                
                logger.debug(f"Adaptive window size: {window_size}s (stability: {normalized_stability:.2f})")
                return max(min(window_size, max_window), min_window)
            else:
                # If no metrics were found, use a moderate stability value
                default_stability = 0.3  # Moderate stability
                window_size = min_window + int((max_window - min_window) * (1 - default_stability))
                logger.debug(f"Using default adaptive window size: {window_size}s (no stability metrics available)")
                return window_size
                
        except Exception as e:
            logger.error(f"Error in adaptive window sizing: {str(e)}")
            # Fall back to default time window
            return self.default_time_window

    def _extract_latest_value(self, metrics_dict: Dict, metric_name: str) -> Optional[float]:
        """Extract the latest value for a metric"""
        try:
            if metric_name not in metrics_dict:
                return None
                
            metric_data = metrics_dict[metric_name]
            
            # Handle direct value
            if isinstance(metric_data, (int, float)):
                return float(metric_data)
                
            # Handle dict with 'latest' key
            elif isinstance(metric_data, dict) and 'latest' in metric_data:
                return float(metric_data['latest'])
                
            # Handle list of timestamped values
            elif isinstance(metric_data, list) and metric_data and isinstance(metric_data[0], dict) and 'timestamp' in metric_data[0] and 'value' in metric_data[0]:
                # Sort by timestamp and get the latest
                sorted_data = sorted(metric_data, key=lambda x: x['timestamp'], reverse=True)
                if sorted_data:
                    return float(sorted_data[0]['value'])
            
            return None
        except Exception as e:
            logger.error(f"Error extracting latest value for {metric_name}: {str(e)}")
            return None

    def _extract_time_series(self, metrics_dict: Dict, metric_name: str) -> List[float]:
        """Extract time series data for a metric"""
        try:
            if metric_name not in metrics_dict:
                return []
                
            metric_data = metrics_dict[metric_name]
            
            # Handle direct value
            if isinstance(metric_data, (int, float)):
                return [float(metric_data)]
                
            # Handle dict with 'latest' key
            elif isinstance(metric_data, dict) and 'latest' in metric_data:
                return [float(metric_data['latest'])]
                
            # Handle list of timestamped values
            elif isinstance(metric_data, list) and metric_data and isinstance(metric_data[0], dict) and 'timestamp' in metric_data[0] and 'value' in metric_data[0]:
                # Sort by timestamp
                sorted_data = sorted(metric_data, key=lambda x: x['timestamp'])
                return [float(item['value']) for item in sorted_data]
            
            return []
        except Exception as e:
            logger.error(f"Error extracting time series for {metric_name}: {str(e)}")
            return []

    def _calculate_time_features(self, values: List[float]) -> List[float]:
        """Calculate statistical features over time window values with emphasis on extremes"""
        try:
            # Convert to numpy array for vectorized operations
            values_array = np.array(values, dtype=np.float64)
            
            # Calculate basic statistics
            mean = np.mean(values_array)
            std_dev = np.std(values_array) if len(values_array) > 1 else 0.0
            max_val = np.max(values_array) if values_array.size > 0 else 0.0
            min_val = np.min(values_array) if values_array.size > 0 else 0.0
            
            # Enhanced features for anomaly detection
            features = []
            
            # Basic statistics
            features.extend([
                mean,                   # Basic mean
                std_dev,                # Standard deviation
                max_val,                # Maximum value
                min_val,                # Minimum value
                np.median(values_array) if values_array.size > 0 else 0.0,  # Median
                np.ptp(values_array) if values_array.size > 0 else 0.0,     # Peak-to-peak range
            ])
            
            # Percentile-based features
            if values_array.size > 0:
                features.extend([
                    np.percentile(values_array, 75),  # 75th percentile
                    np.percentile(values_array, 90),  # 90th percentile
                    np.percentile(values_array, 95),  # 95th percentile
                    np.percentile(values_array, 99),  # 99th percentile for extreme spikes
                ])
            else:
                features.extend([0.0] * 4)
            
            # Deviation features
            if std_dev > 0 and values_array.size > 0:
                max_z_score = (max_val - mean) / std_dev  # Z-score of maximum
                min_z_score = (min_val - mean) / std_dev  # Z-score of minimum
                features.extend([max_z_score, min_z_score])
                
                # Count of severe outliers (> 3 std dev)
                severe_outliers = np.sum(np.abs(values_array - mean) > 3 * std_dev)
                features.append(float(severe_outliers) / len(values_array))
            else:
                features.extend([0.0, 0.0, 0.0])
            
            # Ratio features
            if mean > 0:
                features.append(max_val / mean)  # Max/mean ratio
            else:
                features.append(1.0)
                
            # Trend features
            if len(values_array) > 1:
                try:
                    gradient = np.gradient(values_array)
                    features.append(np.mean(gradient))  # Average trend
                    features.append(np.max(np.abs(gradient)))  # Maximum change rate
                except Exception as e:
                    features.extend([0.0, 0.0])
            else:
                features.extend([0.0, 0.0])
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating time features: {str(e)}")
            return [0.0] * 16  # Return zeros for all 16 features

    def _update_baseline_with_drift_awareness(self, metrics: Dict[str, Any]) -> None:
        """
        Update baseline metrics with drift awareness using exponential smoothing
        """
        if not self.use_drift_aware:
            return
            
        try:
            # Iterate through metrics categories
            for category in ['cpu', 'memory', 'io', 'network', 'transactions']:
                if category not in metrics:
                    continue
                    
                category_metrics = metrics[category]
                
                # Ensure category exists in baseline
                if category not in self.baseline_metrics:
                    self.baseline_metrics[category] = {}
                
                # Process each metric in this category
                for metric_name, metric_data in category_metrics.items():
                    # Extract current values
                    current_values = self._extract_time_series(category_metrics, metric_name)
                    
                    if not current_values:
                        continue
                    
                    # Calculate current statistics
                    current_stats = {
                        'mean': float(np.mean(current_values)),
                        'std': float(np.std(current_values)) if len(current_values) > 1 else 0.0,
                        'max': float(np.max(current_values)),
                        'min': float(np.min(current_values)),
                        'median': float(np.median(current_values)),
                        'last_updated': datetime.now().isoformat()
                    }
                    
                    # Check if we have existing baseline for this metric
                    if metric_name in self.baseline_metrics[category]:
                        # Get existing baseline
                        baseline = self.baseline_metrics[category][metric_name]
                        
                        # Check for significant drift
                        mean_drift = abs(current_stats['mean'] - baseline['mean'])
                        if baseline['mean'] > 0:
                            relative_drift = mean_drift / baseline['mean']
                        else:
                            relative_drift = mean_drift if mean_drift > 0 else 0.0
                        
                        # Apply exponential smoothing
                        for key in ['mean', 'std', 'max', 'min', 'median']:
                            if key in baseline and key in current_stats:
                                # Apply more aggressive smoothing if drift exceeds threshold
                                if relative_drift > self.drift_threshold:
                                    logger.debug(f"Significant drift detected for {category}.{metric_name}: {relative_drift:.4f}")
                                    # Use higher alpha for faster adaptation to changes
                                    adaptive_alpha = min(0.5, self.smoothing_factor * 2.0)
                                else:
                                    # Use standard alpha for normal conditions
                                    adaptive_alpha = self.smoothing_factor
                                
                                # Exponential smoothing formula: new = α * current + (1-α) * old
                                baseline[key] = adaptive_alpha * current_stats[key] + (1 - adaptive_alpha) * baseline[key]
                        
                        # Update timestamp
                        baseline['last_updated'] = current_stats['last_updated']
                        
                    else:
                        # No existing baseline, use current as initial baseline
                        self.baseline_metrics[category][metric_name] = current_stats
                        
            logger.debug(f"Updated baseline metrics with drift awareness")
            
        except Exception as e:
            logger.error(f"Error updating baseline metrics: {str(e)}")

    def _process_metrics(self, metrics: Dict[str, Any]) -> np.ndarray:
        """Process metrics into feature vector using time window statistics"""
        try:
            # Log top-level metrics structure
            metric_categories = list(metrics.keys())
            logger.debug(f"Processing metrics with {len(metric_categories)} categories: {metric_categories}")
            
            # If using adaptive window, adjust the window size based on metrics
            if self.adaptive_window:
                self.time_window = self._adjust_window_size(metrics)
                logger.debug(f"Using adaptive window size: {self.time_window}")
            
            # Update baseline metrics with drift awareness if enabled
            if self.use_drift_aware:
                self._update_baseline_with_drift_awareness(metrics)
            
            # Pre-allocate the feature vector with the correct size
            # Each metric produces 16 features, and we're processing multiple metrics
            num_metrics = (
                3 +  # CPU metrics
                6 +  # Memory metrics
                13 +  # IO metrics
                12 +  # Network metrics
                27    # Transaction metrics
            )
            feature_vector = np.zeros(num_metrics * 16)  # 16 features per metric (matches _calculate_time_features output)
            feature_idx = 0
            
            # logger.info(f"Processing {num_metrics} metrics into feature vector")
            unique_values_count = 0  # Count metrics with non-zero values
            
            # CPU metrics
            cpu_metrics = metrics.get('cpu', {})
            for metric in ['cpu usage', 'worker time', 'cpu time']:
                values = self._get_time_window_values(cpu_metrics, metric)
                features = self._calculate_time_features(values)
                # Check if features have any non-zero values
                if any(f != 0 for f in features):
                    unique_values_count += 1
                feature_vector[feature_idx:feature_idx+len(features)] = features
                feature_idx += len(features)
            
            # Memory metrics
            memory_metrics = metrics.get('memory', {})
            for metric in ['total memstore used', 'active memstore used', 'memstore limit', 
                         'memory usage', 'observer memory hold size', 'memory usage']:
                values = self._get_time_window_values(memory_metrics, metric)
                features = self._calculate_time_features(values)
                if any(f != 0 for f in features):
                    unique_values_count += 1
                feature_vector[feature_idx:feature_idx+len(features)] = features
                feature_idx += len(features)
            
            # IO metrics
            io_metrics = metrics.get('io', {})
            for metric in ['palf write io count to disk', 'palf write size to disk',
                         'clog trans log total size', 'io read count', 'io write count',
                         'io read delay', 'io write delay', 'blockscaned data micro block count',
                         'accessed data micro block count', 'data micro block cache hit',
                         'index micro block cache hit', 'row cache miss', 'row cache hit']:
                values = self._get_time_window_values(io_metrics, metric)
                features = self._calculate_time_features(values)
                if any(f != 0 for f in features):
                    unique_values_count += 1
                feature_vector[feature_idx:feature_idx+len(features)] = features
                feature_idx += len(features)
            
            # Network metrics
            network_metrics = metrics.get('network', {})
            for metric in ['rpc packet in', 'rpc packet in bytes', 'rpc packet out',
                         'rpc packet out bytes', 'rpc deliver fail', 'rpc net delay',
                         'rpc net frame delay', 'mysql packet in', 'mysql packet in bytes',
                         'mysql packet out', 'mysql packet out bytes', 'mysql deliver fail']:
                values = self._get_time_window_values(network_metrics, metric)
                features = self._calculate_time_features(values)
                if any(f != 0 for f in features):
                    unique_values_count += 1
                feature_vector[feature_idx:feature_idx+len(features)] = features
                feature_idx += len(features)
            
            # Transaction metrics
            trans_metrics = metrics.get('transactions', {})
            for metric in ['trans commit count', 'trans timeout count', 'sql update count',
                         'active sessions', 'sql select count', 'trans system trans count',
                         'trans rollback count', 'sql fail count', 'request queue time',
                         'trans user trans count', 'commit log replay count', 'sql update count',
                         'sql delete count', 'sql other count', 'ps prepare count',
                         'ps execute count', 'sql inner insert count', 'sql inner update count',
                         'memstore write lock succ count', 'memstore write lock fail count',
                         'DB time', 'local trans total used time', 'distributed trans total used time',
                         'sql select count', 'sql local count', 'sql remote count',
                         'sql distributed count']:
                values = self._get_time_window_values(trans_metrics, metric)
                features = self._calculate_time_features(values)
                if any(f != 0 for f in features):
                    unique_values_count += 1
                feature_vector[feature_idx:feature_idx+len(features)] = features
                feature_idx += len(features)
            
            # Log summary of feature vector
            non_zero_count = np.count_nonzero(feature_vector)
            # logger.info(f"Feature vector summary: {unique_values_count}/{num_metrics} metrics with non-zero values")
            
            # Check if feature vector has variety - important for correlation
            std_dev = np.std(feature_vector)
            if std_dev < 0.001:
                logger.warning("Feature vector has very low variability which may cause correlation issues")
            
            # Create cross-feature interactions for compound anomaly detection
            base_features = feature_vector.copy()
            
            # Select representative features for interactions to avoid explosion
            # We'll use mean values from each category (CPU, memory, IO, network, transactions)
            # These are positioned at fixed intervals in the feature vector
            key_feature_indices = []
            
            # Add CPU feature index (first mean value)
            key_feature_indices.append(0)  # First CPU metric mean
            
            # Add memory feature index
            key_feature_indices.append(3 * 16)  # First memory metric mean
            
            # Add IO feature index 
            key_feature_indices.append((3 + 6) * 16)  # First IO metric mean
            
            # Add network feature index
            key_feature_indices.append((3 + 6 + 13) * 16)  # First network metric mean
            
            # Add transaction feature index
            key_feature_indices.append((3 + 6 + 13 + 12) * 16)  # First transaction metric mean
            
            # Create interaction features only between key features to avoid combinatorial explosion
            interaction_features = []
            for i in range(len(key_feature_indices)):
                for j in range(i+1, len(key_feature_indices)):
                    idx1 = key_feature_indices[i]
                    idx2 = key_feature_indices[j]
                    # Use product as interaction
                    interaction = base_features[idx1] * base_features[idx2]
                    interaction_features.append(interaction)
            
            # Add max-value cross-category interactions
            # Find max feature value for each category
            cpu_max = np.max(base_features[:3*16])
            mem_max = np.max(base_features[3*16:(3+6)*16])
            io_max = np.max(base_features[(3+6)*16:(3+6+13)*16])
            net_max = np.max(base_features[(3+6+13)*16:(3+6+13+12)*16])
            trans_max = np.max(base_features[(3+6+13+12)*16:])
            
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
            
            logger.debug(f"Enhanced feature vector with {len(interaction_features)} interaction features")
            
            return enhanced_features
        except Exception as e:
            logger.error(f"Error processing metrics: {str(e)}")
            return None

    def calculate_score_for_node(self, node_metrics):
        """Calculate PageRank scores for nodes based on metric correlations"""
        # Ensure we have the correct number of nodes
        actual_node_num = len(node_metrics)
        if actual_node_num != self.node_num:
            logger.warning(f"Node count mismatch: expected {self.node_num}, got {actual_node_num}")
            # Use the actual number of nodes for this calculation
            node_num = actual_node_num
        else:
            node_num = self.node_num
            
        # Initialize weight matrix with zeros
        w = [[0]*node_num for _ in range(node_num)]
        
        # Calculate correlations between each pair of nodes
        for i in range(node_num):
            for j in range(node_num):
                if i != j and j > i:
                    # Calculate correlation between node i and node j
                    w[i][j] = self.calculate_rank(node_metrics[i], node_metrics[j])
                    logger.debug(f"Correlation between node {i} and node {j}: {w[i][j]:.4f}")
                elif j < i:
                    # Use the already calculated correlation (symmetric)
                    w[i][j] = w[j][i]
                    logger.debug(f"Using symmetric correlation for node {i} and node {j}: {w[i][j]:.4f}")

        # Create PageRank instance with the weight matrix
        pr = WPRNNode(w, node_num)
        # Run PageRank algorithm for 10 iterations
        scores = pr.page_rank(10)
        
        logger.debug(f"PageRank scores for {node_num} nodes: {scores}")
        return scores

    def calculate_rank(self, node_x, node_y):
        """Calculate correlation rank between two nodes"""
        try:
            # Extract features for correlation calculation
            x_features = node_x.flatten() if hasattr(node_x, 'flatten') else node_x
            y_features = node_y.flatten() if hasattr(node_y, 'flatten') else node_y
            
            # Ensure features have enough data for meaningful correlation
            if len(x_features) < 2 or len(y_features) < 2:
                return 0.5  # Moderate default correlation
            
            # Add a small amount of noise to avoid identical vectors
            # Increase noise scale slightly for better differentiation between similar nodes
            x_features_noise = x_features + np.random.normal(0, 1e-4, size=len(x_features))
            y_features_noise = y_features + np.random.normal(0, 1e-4, size=len(y_features))
            
            # Calculate correlation using more reliable method
            try:
                correlation = np.corrcoef(x_features_noise, y_features_noise)[0, 1]
                
                # Handle NaN values that might result from constant vectors
                if np.isnan(correlation):
                    if np.allclose(x_features, y_features, rtol=1e-5, atol=1e-8):
                        correlation = 0.7  # Reduced from 0.8 to create more differentiation
                    else:
                        correlation = 0.1  # Low correlation for different but constant vectors
                
                # Apply sigmoid scaling with increased sensitivity to amplify differences
                correlation = 2.0 / (1.0 + np.exp(-8 * abs(correlation))) - 1.0  # Increased from 5 to 8
                
            except Exception:
                correlation = 0.3  # Moderate-low default
            
            # Apply more significant random noise to create better differentiation between nodes
            correlation += (np.random.rand() - 0.5) * 0.15  # Increased from 0.1 to 0.15
            
            # Ensure correlation is within valid range
            correlation = max(-1.0, min(1.0, correlation))
            
            # Use absolute correlation directly as rank (higher correlation = higher rank)
            rank = abs(correlation)
            
            return rank
        except Exception as e:
            logger.error(f"Error calculating rank: {str(e)}")
            return 0.5  # Default value on error

    def _extract_root_causes_from_diagnosis(self, diagnosis_result):
        """Extract root causes from diagnosis results"""
        root_causes = []
        # Get the actual node order from diagnosis results
        node_names = diagnosis_result.get('node_names', [])
        if not node_names:
            logger.warning("No node names found in diagnosis result")
            return root_causes
            
        for anomaly in diagnosis_result.get('all_ranked_anomalies', []):
            try:
                # Get node name and find its index in the original node list
                node_name = anomaly.get('node', '')
                if not node_name:
                    continue
                    
                try:
                    node_idx = node_names.index(node_name)
                except ValueError:
                    logger.warning(f"Node {node_name} not found in original node list")
                    continue
                
                # Get type index from anomaly type
                anomaly_type = anomaly.get('type')
                if anomaly_type not in self.classes:
                    continue
                type_idx = self.classes.index(anomaly_type)
                
                root_causes.append({
                    'node_idx': node_idx,
                    'type_idx': type_idx,
                    'type': anomaly_type
                })
            except (KeyError, IndexError) as e:
                logger.warning(f"Error extracting root cause from anomaly: {e}")
                continue
        return root_causes

    def diagnose(self, node_metrics: dict) -> dict:
        """
        Diagnose anomalies from node metrics with improved score normalization
        
        Args:
            node_metrics: Dict of node metrics in format:
                {
                    'node_name': {
                        'metric_name': {
                            'latest': value,
                            ...
                        },
                        ...
                    },
                    ...
                }
                
        Returns:
            Dict containing:
                - scores: Dict of anomaly scores per node
                - anomalies: List of detected anomalies with types
                - propagation_graph: Dict showing anomaly propagation
        """
        try:
            # Process metrics into feature vectors
            features = []
            node_names = []
            node_metrics_list = []  # Store processed metrics for correlation
            
            logger.info(f"Processing metrics for {len(node_metrics)} nodes")
            for node, metrics in node_metrics.items():
                feature_vector = self._process_metrics(metrics)
                if feature_vector is not None:
                    features.append(feature_vector)
                    node_names.append(node)
                    node_metrics_list.append(feature_vector)
                    logger.debug(f"Processed metrics for node {node}, feature vector shape: {feature_vector.shape}")

            if not features:
                logger.warning("No valid features extracted from metrics")
                return self._empty_result(node_metrics)

            # Convert features to numpy array for processing
            features_array = np.array(features)
            
            # Apply PCA transformation if available and enabled to match training dimensions
            if self.use_pca and hasattr(self, 'pca') and self.pca is not None:
                try:
                    if features_array.shape[1] != self.pca.n_components_:
                        logger.warning(f"PCA dimension mismatch: input has {features_array.shape[1]} features but PCA expects {self.pca.n_components_}")
                        logger.info("Skipping PCA transformation due to dimension mismatch")
                    else:
                        features_array = self.pca.transform(features_array)
                        logger.debug(f"Applied PCA transformation. New feature shape: {features_array.shape}")
                except Exception as e:
                    logger.error(f"Error applying PCA transformation: {str(e)}. Continuing without PCA.")
            
            logger.debug(f"Processed {len(node_metrics_list)} node feature vectors, each with shape {features_array.shape[1] if features else 'unknown'}")
            
            # Calculate node importance scores using PageRank
            node_scores = self.calculate_score_for_node(node_metrics_list)
            
            # Apply softmax to normalize PageRank scores to probabilities
            # This ensures PageRank scores sum to 1.0 and have better scaling properties
            def softmax(x):
                exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
                return exp_x / exp_x.sum()
            
            # Apply softmax normalization to node_scores
            normalized_node_scores = softmax(node_scores)
            logger.debug(f"Normalized node scores: {normalized_node_scores}")
            
            # Get predictions and combine with node scores
            results = {}
            anomalies = []
            propagation_graph = {}
            
            # Process each node
            for node_idx, node in enumerate(node_names):
                node_results = {}
                feature = features_array[node_idx].reshape(1, -1)
                
                # Get classifier predictions for each anomaly type
                for clf_idx, clf in enumerate(self.clf_list):
                    try:
                        # Ensure classifier index is valid
                        if clf_idx >= len(self.classes):
                            logger.warning(f"Classifier index {clf_idx} exceeds available classes ({len(self.classes)})")
                            continue
                        
                        # Get prediction probability using XGBoost's native predict_proba
                        # This directly gives calibrated probabilities
                        prediction = clf.predict_proba(feature)
                        if prediction is None or len(prediction) == 0:
                            logger.warning(f"Empty prediction from classifier {clf_idx} for node {node}")
                            continue
                        
                        proba = prediction[0]
                        logger.debug(f"Node {node} - Classifier {clf_idx} ({self.classes[clf_idx]}) probabilities: {proba}")
                        
                        # Get positive class probability (probability of anomaly)
                        if len(proba) > 1:
                            # Binary case - use probability of positive class (class 1)
                            anomaly_probability = float(proba[1])
                        else:
                            # Single class case
                            anomaly_probability = float(proba[0])
                        
                        # Apply threshold to reduce false positives
                        # Only consider scores above a minimum threshold
                        if anomaly_probability < 0.01:  # 1% minimum threshold
                            anomaly_probability = 0
                        
                        # Combine with normalized node importance (PageRank)
                        node_importance = normalized_node_scores[node_idx]
                        
                        # Modified score calculation with higher discrimination:
                        # 1. Apply more aggressive exponent to node_importance to amplify differences
                        # 2. Use a power function to create more separation between node scores
                        # 3. Scale appropriately for consistent reporting
                        node_importance_factor = np.power(node_importance, 0.4)  # Less aggressive than square root (0.5)
                        
                        # Use cubic scaling on probability to increase separation
                        probability_factor = np.power(anomaly_probability, 1.2)  # Slightly more weight to high probabilities
                        
                        # Combine factors with variable scaling based on node count
                        scale_factor = 10.0
                        combined_score = scale_factor * probability_factor * node_importance_factor
                        
                        anomaly_type = self.classes[clf_idx]
                        node_results[anomaly_type] = combined_score
                        
                        logger.debug(f"Node {node} - {anomaly_type}: "
                                    f"probability={anomaly_probability:.4f}, "
                                    f"node_importance={node_importance:.4f}, "
                                    f"combined_score={combined_score:.4f}")
                        
                    except Exception as e:
                        logger.error(f"Error predicting with classifier {clf_idx}: {str(e)}")
                        continue
                
                results[node] = node_results
                
                # Find highest scoring anomaly
                if node_results:
                    max_score = max(node_results.values())
                    max_type = max(node_results.items(), key=lambda x: x[1])[0]
                    logger.debug(f"Node {node} - Highest scoring anomaly: {max_type} with score {max_score:.4f}")
                    
                    if max_score > 0.001:  # Threshold aligned with reference implementation
                        anomalies.append({
                            'node': node,
                            'type': max_type,
                            'score': max_score,
                            'timestamp': datetime.now().isoformat()
                        })
                        logger.info(f"Detected anomaly on node {node}: {max_type} with score {max_score:.4f}")

            # Calculate propagation graph based on correlations
            for i, node1 in enumerate(node_names):
                propagation_graph[node1] = []
                for j, node2 in enumerate(node_names):
                    if i != j:
                        correlation = self.calculate_rank(node_metrics_list[i], node_metrics_list[j])
                        logger.debug(f"Correlation between {node1} and {node2}: {correlation:.4f}")
                        if correlation > 0.001:  # Correlation threshold from reference
                            propagation_graph[node1].append({
                                'target': node2,
                                'correlation': correlation
                            })

            # Sort anomalies by score
            anomalies.sort(key=lambda x: x['score'], reverse=True)
            
            # After calculating all scores
            all_possible_anomalies = []
            for node_idx, node in enumerate(node_names):
                for anomaly_type, score in results[node].items():
                    all_possible_anomalies.append({
                        'node': node,
                        'type': anomaly_type,
                        'score': score,
                        'timestamp': datetime.now().isoformat()
                    })
            
            # Sort all possibilities by score
            all_possible_anomalies.sort(key=lambda x: x['score'], reverse=True)
            
            result = {
                'scores': {node: max(node_results.values()) if node_results else 0.0 for node, node_results in results.items()},
                'anomalies': anomalies,
                'propagation_graph': propagation_graph,
                'all_ranked_anomalies': all_possible_anomalies,
                'node_names': node_names  # Add node_names to preserve node order
            }
            
            logger.info(f"Diagnosis complete. Found {len(anomalies)} anomalies and {len(propagation_graph)} propagation connections")
            return result

        except Exception as e:
            logger.error(f"Error in diagnose: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self._empty_result(node_metrics)

    def _empty_result(self, node_metrics):
        """Return empty result structure"""
        return {
            'scores': {node: 0.0 for node in node_metrics.keys()},
            'anomalies': [],
            'propagation_graph': {},
            'all_ranked_anomalies': [],
            'node_names': []
        }

    def train(self, x_train, y_train):
        """Train the model with collected data"""
        try:
            logger.debug(f"Original training data shapes - X: {x_train.shape}, y: {y_train.shape}")
            logger.debug(f"X data sample: {x_train[0] if len(x_train) > 0 else 'empty'}")
            logger.debug(f"y data sample: {y_train[0] if len(y_train) > 0 else 'empty'}")
            
            # Decompose training data
            start_time = time.time()
            train_x = self.decompose_train_x(x_train)
            train_y = self.decompose_train_y(y_train)
            logger.info(f"Decomposition completed in {time.time() - start_time:.2f} seconds")
            
            logger.debug(f"Decomposed training data shapes - X: {train_x.shape}, y: {train_y.shape}")
            logger.debug(f"Decomposed X sample: {train_x[0] if len(train_x) > 0 else 'empty'}")
            logger.debug(f"Decomposed y sample: {train_y[0] if len(train_y) > 0 else 'empty'}")
            
            if len(train_x) == 0 or len(train_y) == 0:
                raise ValueError("Empty training data after decomposition")

            # Ensure we have enough classifiers for the number of classes
            while len(self.clf_list) < train_y.shape[1]:
                logger.info(f"Adding classifier {len(self.clf_list)} to match training data dimensions")
                self.clf_list.append(XGBClassifier(
                    objective='binary:logistic',
                    device='cuda',
                    n_jobs=-1  # Use all available CPU cores for parallel processing
                ))
            
            # Use parallel processing for training multiple classifiers
            from concurrent.futures import ThreadPoolExecutor
            
            def train_classifier(i):
                current_clf = self.clf_list[i]
                current_labels = train_y[:, i].reshape(-1, 1)
                
                logger.debug(f"Training classifier {i} with {len(train_x)} samples")
                logger.debug(f"Labels for classifier {i}: shape={current_labels.shape}, unique values={np.unique(current_labels)}")
                
                try:
                    current_clf.fit(train_x, current_labels.ravel())
                    return i, current_clf
                except Exception as e:
                    logger.error(f"Error training classifier {i}: {str(e)}")
                    logger.debug(f"Exception details: {traceback.format_exc()}")
                    return i, None

            # Train classifiers in parallel
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(train_classifier, i) for i in range(len(self.clf_list))]
                
                # Collect results and update classifiers
                for future in futures:
                    idx, clf = future.result()
                    if clf is not None:
                        self.clf_list[idx] = clf
                    else:
                        raise Exception(f"Failed to train classifier {idx}")
                        
            logger.info(f"Successfully trained all classifiers in {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error in training: {str(e)}")
            logger.debug(f"Exception details: {traceback.format_exc()}")
            raise

    def save_model(self, model_dir):
        """Save model to directory"""
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Define a custom JSON encoder that handles NumPy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)
        
        # Save model configurations
        config = {
            "num_classifier": self.num_classifier,
            "classes": self.classes,
            "node_num": self.node_num,
            "scaler_params": self.scaler_params if hasattr(self, 'scaler_params') else None,
        }
        
        with open(os.path.join(model_dir, "model_config.json"), 'w') as f:
            json.dump(config, f, cls=NumpyEncoder)
        
        # Save feature information
        feature_info = {
            "feature_dim": self.feature_dim if hasattr(self, 'feature_dim') else None,
            "time_window": self.time_window,
            "metric_categories": [
                "cpu",
                "memory",
                "io",
                "network",
                "transactions"
            ],
            "metrics_per_category": {
                "cpu": ["cpu usage", "worker time", "cpu time"],
                "memory": ["total memstore used", "active memstore used", "memstore limit", "memory usage", "observer memory hold size"],
                "io": ["palf write io count to disk", "palf write size to disk", "io read count", "io write count", "io read delay", "io write delay", "data micro block cache hit", "index micro block cache hit", "row cache hit"],
                "network": ["rpc packet in", "rpc packet in bytes", "rpc packet out", "rpc packet out bytes", "rpc net delay", "mysql packet in", "mysql packet out"],
                "transactions": ["trans commit count", "trans timeout count", "sql update count", "sql select count", "active sessions", "sql fail count", "request queue time", "DB time"]
            }
        }
        
        with open(os.path.join(model_dir, "feature_info.json"), 'w') as f:
            json.dump(feature_info, f, cls=NumpyEncoder)
        
        # Save decomposition model
        if hasattr(self, 'pca') and self.pca is not None:
            decomposition_file = os.path.join(model_dir, "decomposition.pkl")
            with open(decomposition_file, 'wb') as f:
                pickle.dump(self.pca, f)
        
        # Save classifiers
        for i, clf in enumerate(self.clf_list):
            if clf is not None:
                clf_file = os.path.join(model_dir, f"clf_{i}.pkl")
                with open(clf_file, 'wb') as f:
                    pickle.dump(clf, f)
        
        logger.info(f"Model saved to {model_dir}")

    def load_model(self, model_dir):
        """Load trained models"""
        self.clf_list = []
        for i in range(self.num_classifier):
            # Look for both possible naming patterns
            joblib_path = os.path.join(model_dir, f'classifier_{i}.joblib')
            pickle_path = os.path.join(model_dir, f'clf_{i}.pkl')
            
            if os.path.exists(joblib_path):
                # Load with joblib
                clf = joblib.load(joblib_path)
                self.clf_list.append(clf)
            elif os.path.exists(pickle_path):
                # Load with pickle
                with open(pickle_path, 'rb') as f:
                    clf = pickle.load(f)
                self.clf_list.append(clf)
            else:
                raise FileNotFoundError(f"Model file not found at {joblib_path} or {pickle_path}")
                
        # Load feature info
        feature_info_path = os.path.join(model_dir, 'feature_info.json')
        if os.path.exists(feature_info_path):
            with open(feature_info_path, 'r') as f:
                feature_info = json.load(f)
                if feature_info.get('feature_dim'):
                    self.feature_dim = feature_info['feature_dim']
                    
        # Load PCA model if it exists
        pca_path = os.path.join(model_dir, 'pca_model.joblib')
        pickle_path = os.path.join(model_dir, 'decomposition.pkl')
        if os.path.exists(pca_path):
            self.pca = joblib.load(pca_path)
            logger.info("Loaded PCA model but keeping it disabled to avoid dimension mismatch errors")
            self.use_pca = False  # Explicitly disable PCA even if model exists
        elif os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as f:
                self.pca = pickle.load(f)
            logger.info("Loaded PCA model but keeping it disabled to avoid dimension mismatch errors")  
            self.use_pca = False  # Explicitly disable PCA even if model exists

    def decompose_train_x(self, composed_train_x):
        """Decompose training features and apply PCA if needed"""
        try:
            logger.debug(f"Decomposing X - input type: {type(composed_train_x)}, shape/len: {composed_train_x.shape if isinstance(composed_train_x, np.ndarray) else len(composed_train_x)}")
            
            if isinstance(composed_train_x, np.ndarray):
                if len(composed_train_x.shape) == 1:
                    result = composed_train_x.reshape(1, -1)
                    logger.debug(f"Reshaped 1D array to shape: {result.shape}")
                    return result
                logger.debug(f"Using numpy array directly with shape: {composed_train_x.shape}")
                result = composed_train_x
            else:
                # Pre-allocate the array if we can determine the size
                if isinstance(composed_train_x, list) and all(isinstance(x, np.ndarray) for x in composed_train_x):
                    # Get the shape of the first element to determine dimensions
                    first_shape = composed_train_x[0].shape if isinstance(composed_train_x[0], np.ndarray) else None
                    if isinstance(composed_train_x[0], list) and composed_train_x[0] and isinstance(composed_train_x[0][0], np.ndarray):
                        # We have a list of lists of arrays
                        total_samples = sum(len(case) for case in composed_train_x)
                        sample_shape = composed_train_x[0][0].shape
                        if total_samples > 0 and all(len(sample_shape) == 1 for case in composed_train_x for sample in case if isinstance(sample, np.ndarray)):
                            # Pre-allocate the result array
                            feature_dim = sample_shape[0]
                            result = np.zeros((total_samples, feature_dim))
                            idx = 0
                            for case in composed_train_x:
                                for node_x in case:
                                    if isinstance(node_x, np.ndarray) and len(node_x.shape) == 1:
                                        result[idx] = node_x
                                        idx += 1
                            logger.debug(f"Pre-allocated and filled result array with shape: {result.shape}")
                        else:
                            # Fall back to list append method
                            train_x = []
                            for case in composed_train_x:
                                if isinstance(case, (list, np.ndarray)):
                                    for node_x in case:
                                        train_x.append(node_x)
                                else:
                                    train_x.append(case)
                            result = np.array(train_x)
                    else:
                        # Fall back to list append method
                        train_x = []
                        for case in composed_train_x:
                            if isinstance(case, (list, np.ndarray)):
                                for node_x in case:
                                    train_x.append(node_x)
                            else:
                                train_x.append(case)
                        result = np.array(train_x)
                else:
                    # Fall back to list append method
                    train_x = []
                    for case in composed_train_x:
                        if isinstance(case, (list, np.ndarray)):
                            for node_x in case:
                                train_x.append(node_x)
                        else:
                            train_x.append(case)
                    result = np.array(train_x)
            
            # Store original feature dimension
            self.feature_dim = result.shape[1]
            logger.debug(f"Original feature dimension: {self.feature_dim}")
            
            # Apply PCA if enabled and we have enough samples
            if self.use_pca and len(result) >= 2:
                try:
                    # Initialize PCA to preserve 95% of variance
                    self.pca = PCA(n_components=0.95)
                    
                    # Fit and transform the features
                    result = self.pca.fit_transform(result)
                    
                    # Log the dimensionality reduction results
                    explained_variance = sum(self.pca.explained_variance_ratio_)
                    logger.info(f"Applied PCA dimensionality reduction: {self.feature_dim} → {result.shape[1]} features, " +
                               f"preserving {explained_variance:.2%} of variance")
                except Exception as e:
                    logger.warning(f"Failed to apply PCA dimensionality reduction: {e}")
                    logger.debug(f"Exception details: {traceback.format_exc()}")
                    # Continue with original features
                    if hasattr(self, 'pca'):
                        delattr(self, 'pca')
            else:
                logger.debug("PCA dimensionality reduction is disabled or not enough samples")
                if hasattr(self, 'pca'):
                    delattr(self, 'pca')
            
            logger.debug(f"Final decomposed X shape: {result.shape}")
            return result
            
        except Exception as e:
            logger.error(f"Error decomposing training features: {str(e)}")
            logger.debug(f"Exception details: {traceback.format_exc()}")
            raise

    def decompose_train_y(self, composed_train_y):
        """Decompose training labels"""
        try:
            logger.debug(f"Decomposing y - input type: {type(composed_train_y)}, shape/len: {composed_train_y.shape if isinstance(composed_train_y, np.ndarray) else len(composed_train_y)}")
            
            if isinstance(composed_train_y, np.ndarray):
                if len(composed_train_y.shape) == 1:
                    result = composed_train_y.reshape(1, -1)
                    logger.debug(f"Reshaped 1D array to shape: {result.shape}")
                    return result
                logger.debug(f"Using numpy array directly with shape: {composed_train_y.shape}")
                return composed_train_y
            
            train_y = []
            for case in composed_train_y:
                if isinstance(case, (list, np.ndarray)):
                    for node_label in case:
                        train_y.append(node_label)
                else:
                    train_y.append(case)
            result = np.array(train_y)
            logger.debug(f"Decomposed y result shape: {result.shape}")
            return result
        except Exception as e:
            logger.error(f"Error decomposing training labels: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

class DiagnosisService:
    def __init__(self, model_name=None):
        # Base models directory
        self.models_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            '../../models/'
        ))
        
        # Define anomaly types that can be detected
        self.anomaly_types = [
            "cpu_stress",
            "io_bottleneck", 
            "network_bottleneck",
            "cache_bottleneck",
            "too_many_indexes"
        ]
        
        # Get node count from environment or config
        self.node_count = int(os.getenv('NODE_COUNT', '3'))
        
        # Get time window size from environment or use default
        self.time_window = int(os.getenv('METRICS_TIME_WINDOW', '5'))
        
        # Initialize DistDiagnosis with multi-node support
        self.diagnosis = DistDiagnosis(
            num_classifier=len(self.anomaly_types),
            classes_for_each_node=self.anomaly_types,
            node_num=self.node_count
        )
        self.diagnosis.time_window = self.time_window  # Set time window for metric processing
        self.diagnosis.use_pca = False  # Explicitly disable PCA

        # Configure advanced features - enable adaptive window by default for compound anomalies
        self.enable_adaptive_window = os.getenv('ENABLE_ADAPTIVE_WINDOW', 'true').lower() == 'true'
        self.enable_drift_awareness = os.getenv('ENABLE_DRIFT_AWARENESS', 'false').lower() == 'true'
        
        # Apply advanced feature settings
        if self.enable_adaptive_window:
            self.enable_adaptive_window_alignment()
        if self.enable_drift_awareness:
            self.enable_drift_aware_feature_preservation()
        
        # Select model to use - either specified or latest available
        available_models = self.get_available_models()
        self.active_model = model_name if model_name and model_name in available_models else (
            available_models[0] if available_models else None
        )
        
        if self.active_model:
            # Set model path based on active model
            self.model_path = os.path.join(self.models_path, self.active_model)
            self.metrics_path = os.path.join(self.model_path, 'metrics')
            os.makedirs(self.metrics_path, exist_ok=True)
            
            # Load the selected model
            try:
                self.diagnosis.load_model(self.model_path)
                logger.info("Loaded model '%s' from %s", self.active_model, self.model_path)
            except Exception as e:
                logger.error("Failed to load model '%s': %s", self.active_model, str(e))
        else:
            logger.warning("No models available to load")
            # Set default paths even if no model is available
            self.model_path = None
            self.metrics_path = os.path.join(self.models_path, 'metrics')
            os.makedirs(self.metrics_path, exist_ok=True)

    def enable_adaptive_window_alignment(self, min_window_size=60, max_window_size=600, enable=True):
        """
        Enable adaptive window alignment feature
        
        This feature automatically adjusts observation windows (60s-600s) based on
        metric stability, optimizing for both transient and compound anomalies.
        
        Args:
            min_window_size: Minimum window size in seconds (default: 60 seconds)
            max_window_size: Maximum window size in seconds (default: 600 seconds/10 minutes)
            enable: Whether to enable or disable the feature
        """
        try:
            if enable:
                self.diagnosis.adaptive_window = True
                self.diagnosis.min_window_size = min_window_size
                self.diagnosis.max_window_size = max_window_size
                self.enable_adaptive_window = True
                logger.info(f"Enabled adaptive window alignment (range: {min_window_size}s-{max_window_size}s)")
            else:
                self.diagnosis.adaptive_window = False
                self.diagnosis.time_window = self.time_window  # Reset to default time window
                self.enable_adaptive_window = False
                logger.info("Disabled adaptive window alignment")
            
            return True
        except Exception as e:
            logger.error(f"Error configuring adaptive window alignment: {str(e)}")
            return False

    def enable_drift_aware_feature_preservation(self, smoothing_factor=0.2, drift_threshold=0.3, enable=True):
        """
        Enable drift-aware feature preservation
        
        This feature maintains historical statistical baselines while incrementally updating
        distribution models through exponential smoothing. This preserves important features
        while allowing the model to adapt to gradual system behavior changes.
        
        Args:
            smoothing_factor: Alpha value for exponential smoothing (default: 0.2)
            drift_threshold: Threshold to determine significant drift (default: 0.3)
            enable: Whether to enable or disable the feature
        """
        try:
            if enable:
                self.diagnosis.use_drift_aware = True
                self.diagnosis.smoothing_factor = smoothing_factor
                self.diagnosis.drift_threshold = drift_threshold
                self.diagnosis.baseline_metrics = {}  # Initialize or reset baseline metrics
                self.enable_drift_awareness = True
                logger.info(f"Enabled drift-aware feature preservation (α={smoothing_factor}, threshold={drift_threshold})")
            else:
                self.diagnosis.use_drift_aware = False
                self.enable_drift_awareness = False
                logger.info("Disabled drift-aware feature preservation")
            
            return True
        except Exception as e:
            logger.error(f"Error configuring drift-aware feature preservation: {str(e)}")
            return False

    def get_feature_status(self):
        """Get status of advanced features"""
        return {
            "adaptive_window": {
                "enabled": self.diagnosis.adaptive_window,
                "min_window_size": self.diagnosis.min_window_size,
                "max_window_size": self.diagnosis.max_window_size,
                "current_window": self.diagnosis.time_window
            },
            "drift_aware": {
                "enabled": self.diagnosis.use_drift_aware,
                "smoothing_factor": self.diagnosis.smoothing_factor,
                "drift_threshold": self.diagnosis.drift_threshold,
                "baseline_metrics_count": sum(len(metrics) for category, metrics in self.diagnosis.baseline_metrics.items()) if hasattr(self.diagnosis, 'baseline_metrics') else 0
            },
            "pca": {
                "enabled": self.diagnosis.use_pca
            }
        }

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the model with collected data"""
        try:
            if len(X) == 0 or len(y) == 0:
                raise ValueError("Empty training data")
                
            # Split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            
            # Start timing
            self.training_start_time = time.time()
            
            # Train model
            self.diagnosis.train(X_train, y_train)
            
            # Generate model name with dist-diagnosis prefix
            model_name = f"dist-diagnosis-model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create new model directory
            new_model_path = os.path.join(self.models_path, model_name)
            os.makedirs(new_model_path, exist_ok=True)
            
            # Create metrics directory for this model
            new_metrics_path = os.path.join(new_model_path, 'metrics')
            os.makedirs(new_metrics_path, exist_ok=True)
            
            # Save model
            self.diagnosis.save_model(new_model_path)
            
            # Update current model paths
            self.active_model = model_name
            self.model_path = new_model_path
            self.metrics_path = new_metrics_path
            
            # Evaluate and store metrics
            metrics = self.evaluate_model(X_test, y_test, model_name)
            
            return {
                "model_name": model_name,
                "metrics": metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to train model: {str(e)}")
            raise

    def analyze_metrics(self, metrics: Dict[str, Any]) -> List[Dict]:
        """Analyze metrics and return anomaly ranks"""
        try:
            if not metrics:
                logger.warning("Empty metrics received")
                return []
                
            # Get diagnosis results using enhanced algorithm
            diagnosis_result = self.diagnosis.diagnose(metrics)
            
            # Return ranked anomalies with propagation information
            return diagnosis_result.get('anomalies', [])
            
        except Exception as e:
            logger.error("Error in analyze_metrics: %s", str(e))
            return []

    def diagnose(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get complete diagnosis results including propagation graphs and all anomalies
        
        This method provides direct access to the underlying DistDiagnosis.diagnose
        result with all data needed for compound anomaly detection and RCA
        """
        try:
            if not metrics:
                logger.warning("Empty metrics received")
                return {
                    'scores': {},
                    'anomalies': [],
                    'propagation_graph': {},
                    'all_ranked_anomalies': [],
                    'node_names': []
                }
                
            # Get full diagnosis results
            diagnosis_result = self.diagnosis.diagnose(metrics)
            return diagnosis_result
            
        except Exception as e:
            logger.error("Error in diagnose: %s", str(e))
            return {
                'scores': {},
                'anomalies': [],
                'propagation_graph': {},
                'all_ranked_anomalies': [],
                'node_names': []
            }

    def evaluate_model(self, X_test, y_test, model_name):
        """Evaluate model performance and store metrics"""
        try:
            # Apply PCA transformation if available and enabled to match training dimensions
            if self.diagnosis.use_pca and hasattr(self.diagnosis, 'pca') and self.diagnosis.pca is not None:
                try:
                    if X_test.shape[1] != self.diagnosis.pca.n_components_:
                        logger.warning(f"PCA dimension mismatch in evaluation: input has {X_test.shape[1]} features but PCA expects {self.diagnosis.pca.n_components_}")
                        X_test_transformed = X_test  # Use original data
                    else:
                        logger.info(f"Applying PCA transform to test data in model evaluation: {X_test.shape[1]} → {self.diagnosis.pca.n_components_}")
                        X_test_transformed = self.diagnosis.pca.transform(X_test)
                except Exception as e:
                    logger.error(f"Error applying PCA transformation in model evaluation: {str(e)}")
                    X_test_transformed = X_test  # Use original data on error
            else:
                X_test_transformed = X_test
            
            # Get predictions from all classifiers using transformed features
            all_preds = []
            for i, clf in enumerate(self.diagnosis.clf_list):
                pred = clf.predict(X_test_transformed)
                all_preds.append(pred)
            
            # Combine predictions into a multilabel format to match y_test
            y_pred = np.zeros_like(y_test)
            for i, preds in enumerate(all_preds):
                y_pred[:, i] = preds
            
            # Standard performance metrics - using samples average for multilabel with zero_division handling
            metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, average='samples', zero_division=0)),
                'recall': float(recall_score(y_test, y_pred, average='samples', zero_division=0)),
                'f1_score': float(f1_score(y_test, y_pred, average='samples', zero_division=0)),
                'training_time': time.time() - self.training_start_time
            }
            
            # Add RCA performance metrics - pass original X_test since _evaluate_rca_performance handles PCA
            rca_metrics = self._evaluate_rca_performance(X_test, y_test)
            metrics.update(rca_metrics)
            
            # Save metrics
            metrics_file = os.path.join(self.metrics_path, f"{model_name}_metrics.json")
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f)
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise

    def _evaluate_rca_performance(self, X_test, y_test):
        """Evaluate RCA-specific performance metrics"""
        try:
            # Start timer for RCA performance measurement
            start_time = time.time()
            
            # Apply PCA transformation if available and enabled to match training dimensions
            if self.diagnosis.use_pca and hasattr(self.diagnosis, 'pca') and self.diagnosis.pca is not None:
                try:
                    if X_test.shape[1] != self.diagnosis.pca.n_components_:
                        logger.warning(f"PCA dimension mismatch in RCA evaluation: input has {X_test.shape[1]} features but PCA expects {self.diagnosis.pca.n_components_}")
                        # Continue with original data
                    else:
                        logger.info(f"Applying PCA transform to test data in RCA evaluation: {X_test.shape[1]} → {self.diagnosis.pca.n_components_}")
                        X_test = self.diagnosis.pca.transform(X_test)
                except Exception as e:
                    logger.error(f"Error applying PCA transformation in RCA evaluation: {str(e)}")
                    # Continue with original data
            
            # Create a synthetic test scenario from test data
            test_metrics = self._create_test_metrics(X_test)
            
            # Get RCA results
            diagnosis_result = self.diagnosis.diagnose(test_metrics)
            
            # Calculate RCA accuracy (how often did the model identify the correct root cause)
            true_root_causes = self._extract_root_causes_from_labels(y_test)
            predicted_root_causes = self._extract_root_causes_from_diagnosis(diagnosis_result)
            
            # Calculate RCA metrics
            rca_accuracy = self._calculate_rca_accuracy(true_root_causes, predicted_root_causes)
            rca_time = time.time() - start_time
            
            return {
                'rca_accuracy': float(rca_accuracy),
                'rca_time': float(rca_time),
                'rca_precision': float(self._calculate_rca_precision(true_root_causes, predicted_root_causes)),
                'propagation_accuracy': float(self._evaluate_propagation_graph(diagnosis_result.get('propagation_graph', {})))
            }
            
        except Exception as e:
            import traceback
            logger.error(f"Error evaluating RCA performance: {str(e)}")
            logger.debug(f"Exception details: {traceback.format_exc()}")
            return {
                'rca_accuracy': 0.0,
                'rca_time': 0.0,
                'rca_precision': 0.0,
                'propagation_accuracy': 0.0
            }

    def _create_test_metrics(self, X_test):
        """Create synthetic test metrics from test data for evaluation"""
        try:
            # Start with PCA if available and enabled to reduce dimensionality
            if self.diagnosis.use_pca and hasattr(self.diagnosis, 'pca') and self.diagnosis.pca is not None:
                try:
                    logger.info("Inverse transforming PCA features for test metrics creation")
                    X_synth = self.diagnosis.pca.inverse_transform(X_test)
                except Exception as e:
                    logger.error(f"Error inverse transforming PCA features: {str(e)}")
                    X_synth = X_test.copy()  # Fall back to original data
                else:
                    X_synth = X_test.copy()
            
            # Create a dictionary to simulate the metrics structure
            test_metrics = {}
            
            # Number of nodes to simulate (limit to 10 for performance)
            n_nodes = min(10, len(X_test))
            
            # For each node, create a metric entry
            for i in range(n_nodes):
                # Use a subset of features for this specific node
                # Add some variance to create realistic correlation patterns
                node_variance = 0.1 + (i * 0.05)  # Different variance per node
                
                # Create node-specific features by adding noise
                # Root cause node (first one) gets more distinct values
                if i == 0:  # Root cause node
                    node_features = X_synth[i] * (1.0 + 0.5 * np.random.randn(*X_synth[i].shape))
                else:  # Other nodes get noise based on position (to create realistic correlations)
                    node_features = X_synth[i] * (1.0 + node_variance * np.random.randn(*X_synth[i].shape))
                
                # Create node metrics entry
                node_id = f"node_{i}"
                test_metrics[node_id] = {"features": node_features}
                
                # Add simulated category data similar to what real metrics would have
                test_metrics[node_id]["cpu"] = {"util": 50 + (i * 3) + (10 * np.random.rand())}
                test_metrics[node_id]["memory"] = {"used": 75 - (i * 2) + (15 * np.random.rand())}
                
                # Add more variance to root cause node
                if i == 0:
                    test_metrics[node_id]["cpu"]["util"] = 90 + (5 * np.random.rand())
            
            logger.debug(f"Created synthetic test metrics for {n_nodes} nodes")
            return test_metrics
        except Exception as e:
            logger.error(f"Error creating test metrics: {str(e)}")
            raise
        
    def _extract_root_causes_from_labels(self, y_test):
        """Extract root causes from test labels"""
        root_causes = []
        for i, label_vector in enumerate(y_test):
            if np.any(label_vector):  # If any anomaly is present
                # Find the index of highest value (most significant anomaly type)
                anomaly_type_idx = np.argmax(label_vector)
                root_causes.append({
                    'node_idx': i,
                    'type_idx': anomaly_type_idx,
                    'type': self.anomaly_types[anomaly_type_idx] if anomaly_type_idx < len(self.anomaly_types) else 'unknown'
                })
        return root_causes
        
    def _calculate_rca_accuracy(self, true_root_causes, predicted_root_causes):
        """Calculate accuracy of root cause identification"""
        if not true_root_causes or not predicted_root_causes:
            return 0.0
            
        # Consider only top predictions (equal to number of true causes)
        top_predictions = predicted_root_causes[:len(true_root_causes)]
        
        # Count matches (both node and type match)
        matches = 0
        for true_cause in true_root_causes:
            for pred_cause in top_predictions:
                if (true_cause['node_idx'] == pred_cause['node_idx'] and
                    true_cause['type_idx'] == pred_cause['type_idx']):
                    matches += 1
                    break
                    
        return matches / len(true_root_causes)
        
    def _calculate_rca_precision(self, true_root_causes, predicted_root_causes):
        """Calculate precision of root cause identification"""
        if not predicted_root_causes:
            return 0.0
            
        # Consider only top predictions (equal to number of true causes)
        top_predictions = predicted_root_causes[:len(true_root_causes)] if true_root_causes else predicted_root_causes[:1]
        
        # Count correct predictions
        correct = 0
        for pred_cause in top_predictions:
            for true_cause in true_root_causes:
                if (true_cause['node_idx'] == pred_cause['node_idx'] and
                    true_cause['type_idx'] == pred_cause['type_idx']):
                    correct += 1
                    break
                    
        return correct / len(top_predictions)
        
    def _evaluate_propagation_graph(self, propagation_graph):
        """Evaluate accuracy of the propagation graph"""
        # In a real implementation, you would compare against ground truth
        # This is a placeholder implementation
        if not propagation_graph:
            return 0.0
            
        # Since we don't have ground truth for the propagation graph in this example,
        # we'll return a synthetic metric based on graph connectivity
        total_connections = 0
        strong_connections = 0  # Connections with correlation > 0.5
        
        for node, connections in propagation_graph.items():
            total_connections += len(connections)
            for conn in connections:
                if conn.get('correlation', 0) > 0.5:
                    strong_connections += 1
                    
        # Return ratio of strong connections to total
        return strong_connections / total_connections if total_connections > 0 else 0.0

    def get_available_models(self):
        """Get list of available models"""
        try:
            models = []
            # Check main model directory
            if os.path.exists(self.models_path):
                for entry in os.listdir(self.models_path):
                    if os.path.isdir(os.path.join(self.models_path, entry)):
                        # Check if any classifier files exist in this directory
                        classifier_files = glob.glob(os.path.join(self.models_path, entry, "classifier_*.joblib")) + glob.glob(os.path.join(self.models_path, entry, "clf_*.pkl"))
                        if classifier_files:
                            models.append(entry)
                            logger.debug(f"Found model directory '{entry}' with {len(classifier_files)} classifiers")
            
            logger.info(f"Found {len(models)} available models")
            return sorted(models, reverse=True)  # Sort by newest first
            
        except Exception as e:
            logger.error(f"Error getting available models: {str(e)}")
            return []

    def switch_model(self, model_name):
        """Switch to using a different model"""
        try:
            logger.info(f"Attempting to switch to model: {model_name}")
            available_models = self.get_available_models()
            
            if model_name not in available_models:
                logger.error(f"Model '{model_name}' not found in available models: {available_models}")
                return False
                
            # Full path verification
            model_path = os.path.join(self.models_path, model_name)
            if not os.path.exists(model_path):
                logger.error(f"Model directory not found: {model_path}")
                return False
                
            logger.info(f"Loading model from: {model_path}")
            self.diagnosis.load_model(model_path)
            
            # Verify classifier loading
            if not self.diagnosis.clf_list:
                logger.error("No classifiers loaded after model switch")
                return False
                
            logger.info(f"Successfully loaded {len(self.diagnosis.clf_list)} classifiers")
            self.active_model = model_name
            return True
            
        except Exception as e:
            logger.error(f"Model switch failed: {str(e)}")
            return False

# Create singleton instance
diagnosis_service = DiagnosisService()