import aiohttp
import asyncio
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class OBMetricsService:
    def __init__(self):
        self.prometheus_url = os.getenv('PROMETHEUS_URL', 'http://prometheus:9090')
        self.metrics_window = int(os.getenv('METRICS_WINDOW_MINUTES', '30'))

    async def get_database_metrics(self) -> List[Dict]:
        """Fetch OceanBase database metrics from Prometheus"""
        # For testing, return mock data until Prometheus is set up
        now = datetime.now()
        metrics = []
        
        # Generate 30 minutes of mock data
        for i in range(self.metrics_window):
            timestamp = now - timedelta(minutes=i)
            
            # Add some variation to make it interesting
            base_qps = 1000 + 100 * ((i % 20) - 10) / 10  # Oscillate around 1000
            base_tps = base_qps / 2  # TPS is roughly half of QPS
            
            metrics.append({
                'timestamp': timestamp,
                'qps': max(0, base_qps),
                'tps': max(0, base_tps),
                'active_sessions': 100 + 10 * (i % 5),
                'sql_response_time': 0.1 + 0.02 * (i % 3),
                'disk_io_bytes': 5000000 + 500000 * (i % 4),
                'disk_iops': 1000 + 100 * (i % 3),
                'memory_usage': 8192 + 512 * (i % 3),
                'cache_hit_ratio': min(100, 95.5 + 1.0 * (i % 3)),
                'slow_queries': 5 + (i % 3),
                'deadlocks': i % 2,  # Occasional deadlocks
                'replication_lag': max(0, 0.5 + 0.1 * (i % 4)),
                'connection_count': 200 + 20 * (i % 3)
            })
        
        return metrics

    async def get_tenant_metrics(self, tenant_name: Optional[str] = None) -> List[Dict]:
        """Fetch OceanBase tenant-specific metrics from Prometheus"""
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=self.metrics_window)

        tenant_label = f'tenant="{tenant_name}"' if tenant_name else ''
        
        # Prometheus queries for tenant metrics
        queries = {
            'cpu_percent': f'sum(rate(ob_tenant_cpu_used_seconds[5m])) by (tenant) * 100 {tenant_label}',
            'memory_used': f'sum(ob_tenant_memory_used_bytes) by (tenant) / 1024 / 1024 {tenant_label}',
            'disk_used': f'sum(ob_tenant_disk_used_bytes) by (tenant) / 1024 / 1024 / 1024 {tenant_label}',
            'iops': f'sum(rate(ob_tenant_io_count[5m])) by (tenant) {tenant_label}',
            'session_count': f'sum(ob_tenant_session_count) by (tenant) {tenant_label}',
            'active_session_count': f'sum(ob_tenant_active_session_count) by (tenant) {tenant_label}'
        }

        metrics_data = []
        async with aiohttp.ClientSession() as session:
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
                logger.error(f"Error fetching {metric_name}: {str(result)}")
                continue

            for value in result:
                timestamp = datetime.fromtimestamp(value[0])
                tenant = value[1].get('tenant', 'default')
                
                if (timestamp, tenant) not in metrics_by_timestamp:
                    metrics_by_timestamp[(timestamp, tenant)] = {
                        'timestamp': timestamp,
                        'tenant': tenant,
                        'cpu_percent': 0,
                        'memory_used': 0,
                        'disk_used': 0,
                        'iops': 0,
                        'session_count': 0,
                        'active_session_count': 0
                    }
                metrics_by_timestamp[(timestamp, tenant)][metric_name] = float(value[1])

        # Convert to list and sort by timestamp
        metrics_data = list(metrics_by_timestamp.values())
        metrics_data.sort(key=lambda x: (x['timestamp'], x['tenant']))

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