"""
Main script for running the distributed diagnosis detector
"""

import time
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.dist_diagnosis_config import NODE_CONFIG, MONITOR_CONFIG, MODEL_CONFIG, INTEGRATION_CONFIG
from detection_layer.models.dist_diagnosis_adapter import DistDiagnosisAdapter

class DistributedDiagnosisDetector:
    def __init__(self):
        self.config = {
            **NODE_CONFIG,
            'model_path': MODEL_CONFIG['model_path'],
            'monitor_interval': MONITOR_CONFIG['collection_interval']
        }
        self.adapter = DistDiagnosisAdapter(self.config)
        
    def start(self):
        """Start the distributed diagnosis detection"""
        print("Starting distributed diagnosis detection...")
        self.adapter.start_monitoring()
        
        try:
            while True:
                # Perform detection
                results = self.adapter.detect_anomalies()
                
                # Process results
                self._process_results(results)
                
                # Wait for next interval
                time.sleep(MONITOR_CONFIG['collection_interval'])
                
        except KeyboardInterrupt:
            print("\nStopping distributed diagnosis detection...")
            self.adapter.stop_monitoring()
    
    def _process_results(self, results):
        """Process detection results"""
        anomalies = results['anomalies']
        scores = results['scores']
        propagation = results['propagation']
        
        # Process anomalies
        if anomalies:
            print("\nDetected anomalies:")
            for anomaly in anomalies:
                print(f"- {anomaly}")
        
        # Process node scores
        print("\nNode health scores:")
        for node, score in scores.items():
            status = "HEALTHY" if score < INTEGRATION_CONFIG['alert_threshold'] else "SUSPICIOUS"
            print(f"- {node}: {score:.3f} ({status})")
        
        # Process propagation graph
        if propagation:
            print("\nAnomaly propagation:")
            for source, targets in propagation.items():
                print(f"- {source} affects: {', '.join(targets)}")
        
        print("\n" + "="*50)

if __name__ == "__main__":
    detector = DistributedDiagnosisDetector()
    detector.start() 