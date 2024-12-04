import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from dist_diagnosis.dist_diagnosis.dist_diagnosis import DistDiagnosis
from dist_diagnosis.monitor.dstat_monitor import DstatMonitor
from dist_diagnosis.monitor.node_monitor import NodeMonitor

class DistDiagnosisAdapter:
    def __init__(self, config):
        """
        Initialize the DistDiagnosis adapter
        
        Args:
            config: Configuration dictionary containing:
                - node_list: List of database node addresses
                - monitor_interval: Monitoring interval in seconds
                - model_path: Path to saved model (if any)
        """
        self.config = config
        self.node_monitor = NodeMonitor(
            node_list=config['node_list'],
            monitor_interval=config.get('monitor_interval', 5)
        )
        self.dstat_monitor = DstatMonitor(
            node_list=config['node_list'],
            monitor_interval=config.get('monitor_interval', 5)
        )
        self.diagnosis = DistDiagnosis()
        
        if 'model_path' in config:
            self.diagnosis.load_model(config['model_path'])
    
    def start_monitoring(self):
        """Start monitoring the database nodes"""
        self.node_monitor.start()
        self.dstat_monitor.start()
    
    def stop_monitoring(self):
        """Stop monitoring the database nodes"""
        self.node_monitor.stop()
        self.dstat_monitor.stop()
    
    def detect_anomalies(self):
        """
        Perform anomaly detection using the dist_diagnosis system
        
        Returns:
            dict: Detection results containing:
                - anomalies: List of detected anomalies
                - scores: Anomaly scores per node
                - propagation: Anomaly propagation graph
        """
        # Get monitoring data
        node_metrics = self.node_monitor.get_metrics()
        system_metrics = self.dstat_monitor.get_metrics()
        
        # Combine metrics
        combined_metrics = {
            node: {
                **node_metrics.get(node, {}),
                **system_metrics.get(node, {})
            }
            for node in self.config['node_list']
        }
        
        # Perform diagnosis
        results = self.diagnosis.diagnose(combined_metrics)
        
        return {
            'anomalies': results.get('anomalies', []),
            'scores': results.get('scores', {}),
            'propagation': results.get('propagation_graph', {})
        }
    
    def train(self, training_data):
        """
        Train the diagnosis model on historical data
        
        Args:
            training_data: Historical monitoring data with labels
        """
        self.diagnosis.train(training_data)
    
    def save_model(self, path):
        """Save the trained model"""
        self.diagnosis.save_model(path) 