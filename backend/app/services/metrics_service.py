import aiohttp
import os
from datetime import datetime, timedelta
from typing import List, Dict

class MetricsService:
    def __init__(self):
        self.prometheus_url = os.getenv('PROMETHEUS_URL', 'http://prometheus:9090')
        self.metrics_window = int(os.getenv('METRICS_WINDOW_MINUTES', '30'))

    async def get_system_metrics(self) -> List[Dict]:
        """Fetch system metrics from Prometheus"""
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=self.metrics_window)

        # Prometheus queries for different metrics
        queries = {
            'cpu': 'avg(rate(container_cpu_usage_seconds_total{container="oceanbase"}[5m]))',
            'memory': 'avg(container_memory_usage_bytes{container="oceanbase"}) / 1024 / 1024',
            'network': 'sum(rate(container_network_transmit_bytes_total{container="oceanbase"}[5m]))'
        }

        metrics_data = []
        async with aiohttp.ClientSession() as session:
            # Fetch all metrics in parallel
            tasks = []
            for metric_name, query in queries.items():
                task = self._fetch_metric(
                    session,
                    query,
                    start_time.timestamp(),
                    end_time.timestamp(),
                    metric_name
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process and combine the results
        metrics_by_timestamp = {}
        for metric_name, result in zip(queries.keys(), results):
            if isinstance(result, Exception):
                print(f"Error fetching {metric_name}: {str(result)}")
                continue

            for value in result:
                timestamp = datetime.fromtimestamp(value[0])
                if timestamp not in metrics_by_timestamp:
                    metrics_by_timestamp[timestamp] = {
                        'timestamp': timestamp,
                        'cpu': 0,
                        'memory': 0,
                        'network': 0
                    }
                metrics_by_timestamp[timestamp][metric_name] = float(value[1])

        # Convert to list and sort by timestamp
        metrics_data = list(metrics_by_timestamp.values())
        metrics_data.sort(key=lambda x: x['timestamp'])

        return metrics_data

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