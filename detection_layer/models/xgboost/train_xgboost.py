import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import classification_report
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XGBoostAnomalyDetector:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def prepare_data(self, data_path):
        """Load and prepare data for training."""
        try:
            df = pd.read_csv(data_path)
            
            # Extract features and labels
            features = ['cpu_usage', 'memory_usage', 'io_usage', 'network_in', 'network_out', 
                       'disk_read', 'disk_write', 'active_sessions']
            X = df[features]
            y = df['anomaly_label']
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise
    
    def train(self, X_train, y_train):
        """Train the XGBoost model."""
        try:
            self.model = xgb.XGBClassifier(
                objective='multi:softprob',
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            self.model.fit(X_train, y_train)
            logger.info("Model training completed successfully")
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def evaluate(self, X_test, y_test):
        """Evaluate the trained model."""
        try:
            y_pred = self.model.predict(X_test)
            report = classification_report(y_test, y_pred)
            logger.info(f"\nModel Performance:\n{report}")
            return report
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise
    
    def save_model(self, model_path, scaler_path):
        """Save the trained model and scaler."""
        try:
            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Model saved to {model_path}")
            logger.info(f"Scaler saved to {scaler_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

def main():
    try:
        # Initialize detector
        detector = XGBoostAnomalyDetector()
        
        # Prepare data
        data_path = '../../data/processed/training_data.csv'
        X_train, X_test, y_train, y_test = detector.prepare_data(data_path)
        
        # Train model
        detector.train(X_train, y_train)
        
        # Evaluate model
        detector.evaluate(X_test, y_test)
        
        # Save model and scaler
        detector.save_model(
            '../../../models/xgboost_model.joblib',
            '../../../models/xgboost_scaler.joblib'
        )
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 