import os
import json
import asyncio
import subprocess
from datetime import datetime
from typing import Dict, List, Any
import logging
from ..schemas.metrics import MetricsData, TimeSeriesMetric

logger = logging.getLogger(__name__)


class MetricsService:
    def __init__(self):
        self.collection_interval = 300  # 5 minutes
        self.metrics_cache: Dict[str, Any] = {}
        self.is_collecting = False
        self.collection_task = None

    async def start_collection(self):
        """Start periodic metrics collection"""
        if not self.is_collecting:
            self.is_collecting = True
            self.collection_task = asyncio.create_task(self._collect_metrics_loop())
            logger.info("Started periodic metrics collection")

    async def stop_collection(self):
        """Stop periodic metrics collection"""
        if self.is_collecting:
            self.is_collecting = False
            if self.collection_task:
                self.collection_task.cancel()
                try:
                    await self.collection_task
                except asyncio.CancelledError:
                    pass
            logger.info("Stopped periodic metrics collection")

    async def _collect_metrics_loop(self):
        """Main collection loop"""
        while self.is_collecting:
            try:
                await self._collect_metrics()
            except Exception as e:
                logger.error(f"Error collecting metrics: {str(e)}")
            await asyncio.sleep(self.collection_interval)

    async def _collect_metrics(self):
        """Collect metrics using obdiag"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        pack_dir = f"obdiag_gather_pack_{timestamp}"

        # Run obdiag gather sysstat
        cmd = f"obdiag gather sysstat --store-dir={pack_dir}"
        process = await asyncio.create_subprocess_shell(
            cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            logger.error(f"Error running obdiag: {stderr.decode()}")
            return

        # Parse collected stats
        metrics = {}
        for node_dir in os.listdir(pack_dir):
            if node_dir.startswith("sysstat_"):
                node_ip = node_dir.split("_")[1]
                metrics[node_ip] = await self._parse_node_metrics(
                    os.path.join(pack_dir, node_dir)
                )

        # Update cache with timestamp
        self.metrics_cache = {"timestamp": timestamp, "metrics": metrics}

        # Cleanup
        subprocess.run(f"rm -rf {pack_dir}", shell=True)

    def _parse_time_series_data(
        self, lines: List[str], metric_indices: Dict[str, int], skip_first_n: int = 0
    ) -> Dict[str, TimeSeriesMetric]:
        """Parse time series data from lines into metrics
        Args:
            lines: List of data lines
            metric_indices: Dictionary mapping metric names to their column indices
            skip_first_n: Number of columns to skip
        """
        metrics = {}
        for metric_name, idx in metric_indices.items():
            data_points = []
            latest = 0.0

            for line in lines:
                if line.startswith(("#", "MAX", "MEAN", "MIN")) or not line.strip():
                    continue

                parts = line.strip().split()
                if len(parts) > idx:
                    try:
                        timestamp = parts[0]
                        value = float(
                            "K" in parts[idx]
                            and float(parts[idx]) * 1000
                            or float(parts[idx])
                        )
                        data_points.append({"timestamp": timestamp, "value": value})
                        latest = value
                    except (ValueError, IndexError) as e:
                        logger.error(f"Error parsing metric {metric_name}: {str(e)}")
                        continue

            metrics[metric_name] = {"data": data_points, "latest": latest}

        return metrics

    async def _parse_node_metrics(self, node_dir: str) -> Dict[str, Any]:
        """Parse metrics from node directory"""
        metrics = {}

        # Parse CPU metrics
        cpu_file = os.path.join(node_dir, "one_day_cpu_data.txt")
        if os.path.exists(cpu_file):
            with open(cpu_file) as f:
                lines = f.readlines()
                cpu_indices = {
                    "user": 1,
                    "system": 2,
                    "wait": 3,
                    "hirq": 4,
                    "sirq": 5,
                    "util": 6,
                }
                metrics["cpu"] = self._parse_time_series_data(lines, cpu_indices)

        # Parse Memory metrics
        mem_file = os.path.join(node_dir, "one_day_mem_data.txt")
        if os.path.exists(mem_file):
            with open(mem_file) as f:
                lines = f.readlines()
                mem_indices = {
                    "free": 1,
                    "used": 2,
                    "buff": 3,
                    "cach": 4,
                    "total": 5,
                    "util": 6,
                }
                metrics["memory"] = self._parse_time_series_data(lines, mem_indices)

        # Parse Swap metrics
        swap_file = os.path.join(node_dir, "tsar_swap_data.txt")
        if os.path.exists(swap_file):
            with open(swap_file) as f:
                lines = f.readlines()
                swap_indices = {
                    "total": 1,
                    "used": 2,
                    "free": 3,
                    "util": 4,
                    "in_rate": 5,
                    "out_rate": 6,
                }
                metrics["swap"] = self._parse_time_series_data(lines, swap_indices)

        # Parse TCP/UDP metrics
        tcp_udp_file = os.path.join(node_dir, "tsar_tcp_udp_data.txt")
        if os.path.exists(tcp_udp_file):
            with open(tcp_udp_file) as f:
                lines = f.readlines()
                tcp_udp_indices = {
                    "active": 1,
                    "pasv": 2,
                    "retrans": 3,
                    "in_segs": 4,
                    "out_segs": 5,
                    "udp_in_datagrams": 6,
                    "udp_out_datagrams": 7,
                    "udp_in_errors": 8,
                }
                metrics["tcp_udp"] = self._parse_time_series_data(
                    lines, tcp_udp_indices
                )

        # Parse IO metrics
        io_file = os.path.join(node_dir, "tsar_io_data.txt")
        if os.path.exists(io_file):
            with open(io_file) as f:
                lines = f.readlines()
                if len(lines) >= 2:
                    device_line = lines[0]
                    metric_line = lines[1]
                    data_lines = [
                        line
                        for line in lines[2:]
                        if not any(x in line for x in ["#", "MAX", "MEAN", "MIN"])
                        and line.strip()
                    ]

                    # Parse device names
                    devices = []
                    current_device = ""
                    in_device = False
                    for char in device_line:
                        if char == "-":
                            if current_device.strip():
                                devices.append(current_device.strip())
                                current_device = ""
                            in_device = True
                        elif in_device and char != "-":
                            in_device = False
                        elif not in_device and char != "-":
                            current_device += char

                    if devices and data_lines:
                        metrics_per_device = 17
                        device_metrics = {}
                        aggregated_data = {
                            metric: []
                            for metric in [
                                "rrqms",
                                "wrqms",
                                "rs",
                                "ws",
                                "rsecs",
                                "wsecs",
                                "rqsize",
                                "rarqsz",
                                "warqsz",
                                "qusize",
                                "io_await",
                                "rawait",
                                "wawait",
                                "svctm",
                                "util",
                            ]
                        }

                        # Process each timestamp's data
                        for line in data_lines:
                            values = line.strip().split()
                            timestamp = values[0]

                            # Process each device's metrics at this timestamp
                            current_totals = {k: 0.0 for k in aggregated_data}
                            device_count = 0

                            for i, device in enumerate(devices):
                                if device.startswith("loop"):
                                    continue

                                start_idx = i * metrics_per_device + 1
                                if start_idx + metrics_per_device <= len(values):
                                    try:
                                        device_data = {
                                            "rrqms": float(values[start_idx]),
                                            "wrqms": float(values[start_idx + 1]),
                                            "rs": float(values[start_idx + 4]),
                                            "ws": float(values[start_idx + 5]),
                                            "rsecs": float(
                                                "K" in values[start_idx + 6]
                                                and float(values[start_idx + 6]) * 1000
                                                or float(values[start_idx + 6])
                                            ),
                                            "wsecs": float(
                                                "K" in values[start_idx + 7]
                                                and float(values[start_idx + 7]) * 1000
                                                or float(values[start_idx + 7])
                                            ),
                                            "rqsize": float(values[start_idx + 8]),
                                            "rarqsz": float(values[start_idx + 9]),
                                            "warqsz": float(values[start_idx + 10]),
                                            "qusize": float(values[start_idx + 11]),
                                            "io_await": float(values[start_idx + 12]),
                                            "rawait": float(values[start_idx + 13]),
                                            "wawait": float(values[start_idx + 14]),
                                            "svctm": float(values[start_idx + 15]),
                                            "util": float(values[start_idx + 16]),
                                        }

                                        # Initialize device's time series if needed
                                        if device not in device_metrics:
                                            device_metrics[device] = {
                                                k: [] for k in device_data.keys()
                                            }

                                        # Add data point for each metric
                                        for metric, value in device_data.items():
                                            device_metrics[device][metric].append(
                                                {"timestamp": timestamp, "value": value}
                                            )

                                            # Aggregate metrics
                                            if metric not in ["util"]:
                                                current_totals[metric] += value
                                            else:
                                                current_totals[metric] = max(
                                                    current_totals[metric], value
                                                )

                                        device_count += 1
                                    except (ValueError, IndexError) as e:
                                        logger.error(
                                            f"Error parsing device metrics for {device}: {str(e)}"
                                        )
                                        continue

                            # Add aggregated metrics for this timestamp
                            if device_count > 0:
                                for metric, total in current_totals.items():
                                    if metric in ["io_await", "rawait", "wawait", "svctm"]:
                                        total /= device_count
                                    aggregated_data[metric].append(
                                        {"timestamp": timestamp, "value": total}
                                    )

                        # Convert device metrics to final format
                        io_metrics = {
                            "devices": {
                                device: {
                                    metric: {
                                        "data": data,
                                        "latest": data[-1]["value"] if data else 0.0,
                                    }
                                    for metric, data in metrics.items()
                                }
                                for device, metrics in device_metrics.items()
                            },
                            "aggregated": {
                                metric: {
                                    "data": data,
                                    "latest": data[-1]["value"] if data else 0.0,
                                }
                                for metric, data in aggregated_data.items()
                            },
                        }
                        metrics["io"] = io_metrics

        # Parse Network metrics
        net_file = os.path.join(node_dir, "tsar_traffic_data.txt")
        if os.path.exists(net_file):
            with open(net_file) as f:
                lines = f.readlines()
                net_indices = {
                    "bytin": 1,
                    "bytout": 2,
                    "pktin": 3,
                    "pktout": 4,
                    "pkterr": 5,
                    "pktdrp": 6,
                }
                metrics["network"] = self._parse_time_series_data(lines, net_indices)

        return metrics

    def get_latest_metrics(self) -> Dict[str, Any]:
        """Get the latest collected metrics"""
        return self.metrics_cache


metrics_service = MetricsService()
