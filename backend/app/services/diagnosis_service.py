import numpy as np
from typing import List, Dict
from datetime import datetime
import tensorflow as tf
import os

class DiagnosisService:
    def __init__(self):
        self.model_path = os.getenv('MODEL_PATH', 'models/anomaly_detector')
        self.model = self._load_model()
        self.scaler = self._load_scaler()
        self.feature_columns = ['cpu', 'memory', 'network']

    def _load_model(self):
        """Load the trained TensorFlow model"""
        try:
            return tf.keras.models.load_model(self.model_path)
        except Exception as e:
            print(f"Warning: Failed to load model: {str(e)}")
            return self._create_default_model()

    def _load_scaler(self):
        """Load the trained scaler for feature normalization"""
        try:
            return np.load(os.path.join(self.model_path, 'scaler.npy'), allow_pickle=True).item()
        except Exception as e:
            print(f"Warning: Failed to load scaler: {str(e)}")
            return None

    def _create_default_model(self):
        """Create a default model if loading fails"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(len(self.feature_columns),)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    async def analyze_metrics(self, metrics: List[Dict]) -> List[Dict]:
        """Analyze metrics and return anomaly ranks"""
        if not metrics:
            return []

        # Extract features
        features = np.array([[m[col] for col in self.feature_columns] for m in metrics])

        # Normalize features if scaler is available
        if self.scaler:
            features = self.scaler.transform(features)

        # Get predictions
        try:
            predictions = self.model.predict(features)
            ranks = predictions.flatten()
        except Exception as e:
            print(f"Warning: Prediction failed: {str(e)}")
            # Fallback to simple threshold-based detection
            ranks = self._fallback_detection(features)

        # Create response
        return [
            {
                'timestamp': metrics[i]['timestamp'],
                'rank': float(ranks[i])
            }
            for i in range(len(metrics))
        ]

    def _fallback_detection(self, features: np.ndarray) -> np.ndarray:
        """Simple threshold-based detection as fallback"""
        # Calculate Euclidean distance from the origin as an anomaly score
        distances = np.linalg.norm(features, axis=1)
        # Normalize to [0, 1] range
        ranks = (distances - np.min(distances)) / (np.max(distances) - np.min(distances) + 1e-10)
        return ranks 