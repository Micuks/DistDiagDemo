import aiohttp
import asyncio
import os
from datetime import datetime, timedelta
from typing import List, Dict

class MetricsService:
    def __init__(self):
        self.prometheus_url = os.getenv('PROMETHEUS_URL', 'http://prometheus:9090')
        self.metrics_window = int(os.getenv('METRICS_WINDOW_MINUTES', '30'))

    async def get_system_metrics(self) -> List[Dict]:
        """Fetch system metrics from Prometheus"""
        # For testing, return mock data until Prometheus is set up
        now = datetime.now()
        metrics = []
        
        # Generate 30 minutes of mock data
        for i in range(self.metrics_window):
            timestamp = now - timedelta(minutes=i)
            
            # Add some variation to make it interesting
            base_cpu = 0.5 + 0.1 * ((i % 10) - 5) / 5  # Oscillate between 0.4 and 0.6
            base_memory = 1024 + 100 * ((i % 15) - 7) / 7  # Oscillate around 1024
            
            metrics.append({
                'timestamp': timestamp,
                'cpu': max(0, min(1, base_cpu)),  # Keep between 0 and 1
                'memory': max(0, base_memory),
                'network': 1000 + 100 * (i % 5),
                'disk_io_bytes': 500 + 50 * (i % 3),
                'disk_iops': 100 + 10 * (i % 4),
                'active_sessions': 10 + (i % 5),
                'sql_response_time': 0.1 + 0.02 * (i % 3),
                'cache_hit_ratio': min(1, 0.95 + 0.01 * (i % 3))
            })
        
        return metrics

    async def _fetch_metric(
        self,
        session: aiohttp.ClientSession,
        query: str,
        start_time: float,
        end_time: float,
        metric_name: str
    ) -> List:
        """Fetch a single metric from Prometheus"""
        params = {
            'query': query,
            'start': start_time,
            'end': end_time,
            'step': '15s'  # 15-second resolution
        }

        async with session.get(f"{self.prometheus_url}/api/v1/query_range", params=params) as response:
            if response.status != 200:
                raise Exception(f"Failed to fetch {metric_name} metrics: {await response.text()}")
            
            data = await response.json()
            if data['status'] != 'success':
                raise Exception(f"Prometheus query failed for {metric_name}: {data.get('error', 'Unknown error')}")

            # Extract the values from the response
            try:
                return data['data']['result'][0]['values']
            except (KeyError, IndexError):
                return [] 