#!/usr/bin/env python
import os
os.environ["TRAINING_TEST_MODE"] = "true"

import sys
import logging
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from logging.handlers import RotatingFileHandler

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import necessary services
from app.services.training_service import training_service
from app.services.diagnosis_service import diagnosis_service

# Configure logging
class VerboseFilter(logging.Filter):
    def filter(self, record):
        # Filter out debug messages containing response body
        if record.levelno == logging.DEBUG and "response body:" in record.getMessage():
            return False
        return True

def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    main_log_file = log_dir / 'test_model_split.log'
    file_handler = RotatingFileHandler(
        main_log_file,
        maxBytes=10*1024*1024,
        backupCount=2
    )
    file_handler.setLevel(logging.DEBUG)
    
    # Get the root logger
    logger = logging.getLogger()
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Set the root logger level
    logger.setLevel(logging.DEBUG)
    
    # Create console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    # Add filter to console handler
    console_handler.addFilter(VerboseFilter())

    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Set specific loggers to higher level to reduce noise
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('kubernetes').setLevel(logging.WARNING)
    
    return logger

logger = setup_logging()

class ModelSplitTester:
    def __init__(self):
        self.training_service = training_service
        self.diagnosis_service = diagnosis_service
        
    def get_and_split_data(self, split_ratio=0.5):
        """
        Retrieve training data and split it into training and test sets with a 1:1 ratio
        """
        logger.info("Retrieving training data...")
        X_full, y_full = self.training_service.get_training_data()
        
        if len(X_full) == 0 or len(y_full) == 0:
            logger.error("No training data available")
            return None, None, None, None
        
        logger.info(f"Retrieved data with shapes: X={X_full.shape}, y={y_full.shape}")
        
        # Create indices for random shuffling
        indices = list(range(len(X_full)))
        random.shuffle(indices)
        
        # Calculate split point
        split_idx = int(len(indices) * split_ratio)
        
        # Split indices
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        # Split data using indices
        X_train = X_full[train_indices]
        y_train = y_full[train_indices]
        X_test = X_full[test_indices]
        y_test = y_full[test_indices]
        
        logger.info(f"Split data into train set ({len(X_train)} samples) and test set ({len(X_test)} samples)")
        
        return X_train, y_train, X_test, y_test
    
    def train_model(self, X_train, y_train):
        """
        Train the model using the training set
        """
        logger.info("Training model...")
        start_time = time.time()
        
        # Use direct_train method for training
        result = self.diagnosis_service.direct_train(X_train, y_train)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Training completed in {elapsed_time:.2f} seconds")
        
        if not result.get('success', False):
            logger.error(f"Training failed: {result.get('error', 'Unknown error')}")
            return None
        
        logger.info("Model trained successfully")
        return result.get('result', {}).get('model_name')
    
    def evaluate_model(self, X_test, y_test, model_name=None):
        """
        Evaluate the model using the test set
        """
        if model_name:
            logger.info(f"Switching to model {model_name} for evaluation")
            self.diagnosis_service.switch_model(model_name)
        
        logger.info("Evaluating model on test data...")
        
        # Initialize performance metrics
        all_predictions = []
        all_labels = []
        
        # Evaluate each test case
        for idx, (test_X, test_y) in enumerate(zip(X_test, y_test)):
            # Convert the test case into the format expected by diagnose
            anomaly_types = self.diagnosis_service.anomaly_types
            # Treat the entire feature vector as one node
            node_metrics = {"node_0": np.atleast_1d(test_X)}
            
            logger.debug(f"Test case {idx} node_metrics: {node_metrics}")
            
            # Run diagnosis
            diagnosis_result = self.diagnosis_service.diagnose(node_metrics)
            logger.debug(f"Test case {idx} diagnosis result: {diagnosis_result}")
            
            if "error" in diagnosis_result:
                logger.warning(f"Diagnosis error for test case {idx}: {diagnosis_result['error']}")
                continue
            
            # Updated predictions extraction:
            # Since we have a single node, use 'node_0' directly
            node_name = "node_0"
            true_labels = test_y
            
            # Get predictions for this node
            node_predictions = {}
            for pred_idx, pred in enumerate(diagnosis_result.get("all_ranked_anomalies", [])):
                if pred.get("node") == node_name:
                    anomaly_type = pred.get("type")
                    score = pred.get("score", 0)
                    node_predictions[anomaly_type] = score
            
            logger.debug(f"Test case {idx}, node {node_name} predictions: {node_predictions}")
            
            # Compare predictions with ground truth for each anomaly type
            for type_idx, anomaly_type in enumerate(anomaly_types):
                true_label = true_labels[type_idx] > 0.5
                predicted_raw = node_predictions.get(anomaly_type, 0)
                predicted_label = predicted_raw > 0.5
                logger.debug(f"Test case {idx}, node {node_name}, anomaly {anomaly_type}: true_label={true_label}, predicted_label={predicted_label}, raw_pred={predicted_raw}")
                
                all_labels.append(true_label)
                all_predictions.append(predicted_label)
        
        # Calculate overall metrics
        if all_labels and all_predictions:
            accuracy = accuracy_score(all_labels, all_predictions)
            precision = precision_score(all_labels, all_predictions, zero_division=0)
            recall = recall_score(all_labels, all_predictions, zero_division=0)
            f1 = f1_score(all_labels, all_predictions, zero_division=0)
            
            metrics = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "total_samples": len(all_labels),
                "timestamp": datetime.now().isoformat(),
                "model_name": self.diagnosis_service.active_model
            }
            
            logger.info(f"Evaluation metrics: {metrics}")
            
            # Save metrics to file
            if self.diagnosis_service.metrics_path:
                metrics_file = os.path.join(self.diagnosis_service.metrics_path, 'custom_evaluation.json')
                with open(metrics_file, 'w') as f:
                    json.dump(metrics, f, indent=2)
                logger.info(f"Saved evaluation metrics to {metrics_file}")
            
            return metrics
        else:
            logger.warning("No predictions were made during evaluation")
            return {"error": "No predictions made"}
    
    def run_test(self):
        """
        Run the full test process:
        1. Get and split data
        2. Train the model
        3. Evaluate the model
        """
        logger.info("Starting model split test...")
        
        # Get and split data
        X_train, y_train, X_test, y_test = self.get_and_split_data(split_ratio=0.5)
        if X_train is None or y_train is None:
            logger.error("Failed to retrieve and split training data")
            return False
        
        # Train model
        model_name = self.train_model(X_train, y_train)
        if not model_name:
            logger.error("Model training failed")
            return False
        
        # Evaluate model
        metrics = self.evaluate_model(X_test, y_test, model_name)
        
        success = "error" not in metrics
        logger.info(f"Model test {'succeeded' if success else 'failed'}")
        return success

if __name__ == "__main__":
    tester = ModelSplitTester()
    success = tester.run_test()
    sys.exit(0 if success else 1) 