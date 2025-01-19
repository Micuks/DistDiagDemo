import os
import json
import psutil
import time
import asyncio
import subprocess
from typing import Dict, Any, List
from datetime import datetime
import threading
import logging

logger = logging.getLogger(__name__)

class MetricsService:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.metrics = {}
        self.timestamp = None
        self.collection_active = False
        self.collection_thread = None
        self.collection_interval = 5  # seconds
        self.use_obdiag = True  # Flag to control collection method

    def start_collection(self):
        """Start collecting metrics in a background thread."""
        if not self.collection_active:
            self.collection_active = True
            self.collection_thread = threading.Thread(target=self._collect_metrics_loop)
            self.collection_thread.start()
            logger.info("Started metrics collection")

    def stop_collection(self):
        """Stop collecting metrics."""
        if self.collection_active:
            self.collection_active = False
            if self.collection_thread and self.collection_thread.is_alive():
                self.collection_thread.join(timeout=5)
                if self.collection_thread.is_alive():
                    logger.warning("Metrics collection thread did not stop gracefully")
            logger.info("Stopped metrics collection")

    def _collect_metrics_loop(self):
        """Continuously collect metrics while collection is active."""
        logger.info("Starting metrics collection loop")
        while self.collection_active:
            try:
                if self.use_obdiag:
                    try:
                        asyncio.run(self._collect_obdiag_metrics())
                    except Exception as e:
                        logger.error(f"Error collecting metrics with obdiag: {e}")
                        logger.info("Falling back to psutil collection")
                        self._collect_psutil_metrics()
                else:
                    self._collect_psutil_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                if not self.collection_active:  # Break the loop if collection is stopped
                    break
                time.sleep(self.collection_interval)  # Wait before retrying
        logger.info("Metrics collection loop ended")

    async def _collect_obdiag_metrics(self):
        """Collect metrics using obdiag"""
        timestamp = datetime.now().isoformat()
        dir_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        base_pack_dir = f"obdiag_gather_pack_{dir_timestamp}"
        obdiag_cmd = os.getenv('OBDIAG_CMD', 'obdiag')
        
        cmd = f"{obdiag_cmd} gather sysstat --store_dir={base_pack_dir}"
        logger.debug(f"Running command: {cmd}")
        process = await asyncio.create_subprocess_shell(
            cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        logger.debug(f"Command output: {stdout.decode()}")

        if process.returncode != 0:
            logger.error(f"Error running obdiag: {stderr.decode()}")
            raise Exception("Failed to collect metrics with obdiag")

        if not os.path.exists(base_pack_dir):
            logger.error(f"Base pack directory {base_pack_dir} does not exist")
            raise Exception("Obdiag output directory not found")

        nested_dirs = [d for d in os.listdir(base_pack_dir) if d.startswith("obdiag_gather_pack_")]
        if not nested_dirs:
            logger.error(f"No nested pack directory found in {base_pack_dir}")
            raise Exception("Nested pack directory not found")

        pack_dir = os.path.join(base_pack_dir, nested_dirs[0])
        logger.debug(f"Using pack directory: {pack_dir}")

        metrics = {}
        for zip_file in os.listdir(pack_dir):
            if zip_file.startswith("sysstat_"):
                node_ip = zip_file.split("_")[1]
                logger.debug(f"Processing metrics for node: {node_ip}")
                zip_path = os.path.join(pack_dir, zip_file)
                
                extract_dir = pack_dir
                try:
                    # hide unzip output
                    cmd = f"unzip -o {zip_path} -d {extract_dir} > /dev/null 2>&1"
                    logger.debug(f"Extracting metrics data: {cmd}")
                    subprocess.run(cmd, shell=True, check=True)
                    metrics[node_ip] = await self._parse_node_metrics(extract_dir)
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to extract metrics data for {node_ip}: {e}")
                    continue

        if not metrics:
            logger.warning("No metrics were collected from any nodes")
        else:
            logger.info(f"Successfully collected metrics from nodes: {list(metrics.keys())}")

        self.metrics = metrics
        self.timestamp = timestamp

        logger.debug(f"Cleaning up directory: {base_pack_dir}")
        subprocess.run(f"rm -rf {base_pack_dir}", shell=True)

    def _collect_psutil_metrics(self):
        """Collect metrics using psutil as fallback."""
        metrics = {
            "localhost": {  # Using localhost as we're collecting local metrics
                "cpu": {
                    "util": self._get_cpu_metrics(),
                },
                "memory": self._get_memory_metrics(),
                "io": {
                    "aggregated": self._get_io_metrics(),
                },
                "network": self._get_network_metrics(),
            }
        }
        
        self.metrics = metrics
        self.timestamp = datetime.now().isoformat()

    async def _parse_node_metrics(self, node_dir: str) -> Dict[str, Any]:
        """Parse metrics from node directory"""
        logger.debug(f"Parsing metrics from directory: {node_dir}")
        metrics = {}

        node_dirs = [d for d in os.listdir(node_dir) if d.startswith("sysstat_") and not d.endswith(".zip")]
        if not node_dirs:
            logger.error("No node-specific directory found")
            return metrics

        node_specific_dir = os.path.join(node_dir, node_dirs[0])
        logger.debug(f"Using node-specific directory: {node_specific_dir}")

        # Parse CPU metrics
        cpu_file = os.path.join(node_specific_dir, "one_day_cpu_data.txt")
        if os.path.exists(cpu_file):
            with open(cpu_file) as f:
                lines = f.readlines()
                cpu_indices = {
                    "user": 1, "system": 2, "wait": 3,
                    "hirq": 4, "sirq": 5, "util": 6,
                }
                metrics["cpu"] = self._parse_time_series_data(lines, cpu_indices)
        else:
            logger.debug(f"CPU metrics file not found at {cpu_file}")

        # Parse Memory metrics
        mem_file = os.path.join(node_specific_dir, "one_day_mem_data.txt")
        if os.path.exists(mem_file):
            with open(mem_file) as f:
                lines = f.readlines()
                mem_indices = {
                    "free": 1, "used": 2, "buff": 3,
                    "cach": 4, "total": 5, "util": 6,
                }
                metrics["memory"] = self._parse_time_series_data(lines, mem_indices)
        else:
            logger.debug(f"Memory metrics file not found at {mem_file}")

        # Parse Swap metrics
        swap_file = os.path.join(node_specific_dir, "tsar_swap_data.txt")
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
        tcp_udp_file = os.path.join(node_specific_dir, "tsar_tcp_udp_data.txt")
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
        io_file = os.path.join(node_specific_dir, "tsar_io_data.txt")
        if os.path.exists(io_file):
            with open(io_file) as f:
                lines = f.readlines()
                truncated_lines = [line[:30] + '...' if len(line) > 30 else line for line in lines[:3]]
                if len(lines) >= 2:
                    # Clean and parse device line
                    device_line = ''.join(c if c.isalnum() or c in ['-', ' '] else ' ' for c in lines[0])
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
                    
                    # Add last device if exists
                    if current_device.strip():
                        devices.append(current_device.strip())

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
                            # Skip header and empty lines
                            if line.startswith(("Time", "#", "MAX", "MEAN", "MIN", "-")) or not line.strip():
                                continue

                            values = line.strip().split()
                            if not values:
                                continue

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
                                        # Convert values with units (K, M, G) to base values
                                        def convert_io_value(val_str):
                                            try:
                                                return self._convert_to_bytes(val_str)
                                            except ValueError:
                                                logger.error(f"Error converting IO value: {val_str}")
                                                return 0.0

                                        device_data = {
                                            "rrqms": convert_io_value(values[start_idx]),
                                            "wrqms": convert_io_value(values[start_idx + 1]),
                                            "rs": convert_io_value(values[start_idx + 4]),
                                            "ws": convert_io_value(values[start_idx + 5]),
                                            "rsecs": convert_io_value(values[start_idx + 6]),
                                            "wsecs": convert_io_value(values[start_idx + 7]),
                                            "rqsize": convert_io_value(values[start_idx + 8]),
                                            "rarqsz": convert_io_value(values[start_idx + 9]),
                                            "warqsz": convert_io_value(values[start_idx + 10]),
                                            "qusize": convert_io_value(values[start_idx + 11]),
                                            "io_await": convert_io_value(values[start_idx + 12]),
                                            "rawait": convert_io_value(values[start_idx + 13]),
                                            "wawait": convert_io_value(values[start_idx + 14]),
                                            "svctm": convert_io_value(values[start_idx + 15]),
                                            "util": convert_io_value(values[start_idx + 16]),
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
        net_file = os.path.join(node_specific_dir, "tsar_traffic_data.txt")
        if os.path.exists(net_file):
            with open(net_file) as f:
                lines = f.readlines()
                if len(lines) > 3:
                    # Update indices based on actual tsar output format
                    net_indices = {
                        "bytin": 1,    # KB/s
                        "bytout": 2,   # KB/s
                        "pktin": 3,    # packets/s
                        "pktout": 4,   # packets/s
                        "pkterr": 5,   # packets/s
                        "pktdrp": 6,   # packets/s
                    }
                    try:
                        metrics["network"] = self._parse_time_series_data(lines, net_indices)
                        # Convert bytin/bytout from KB/s to B/s for frontend
                        if "bytin" in metrics["network"]:
                            for point in metrics["network"]["bytin"]["data"]:
                                point["value"] *= 1024  # Convert KB/s to B/s
                            metrics["network"]["bytin"]["latest"] *= 1024
                        if "bytout" in metrics["network"]:
                            for point in metrics["network"]["bytout"]["data"]:
                                point["value"] *= 1024  # Convert KB/s to B/s
                            metrics["network"]["bytout"]["latest"] *= 1024
                    except Exception as e:
                        logger.error(f"Error parsing network metrics: {e}")
                        logger.error(f"Network file content sample: {lines[:5] if lines else 'No lines'}")
        else:
            logger.warning(f"Network metrics file not found at {net_file}")

        logger.debug(f"Finished parsing metrics. Available metrics: {list(metrics.keys())}")
        return metrics

    def _convert_to_bytes(self, value_str: str) -> float:
        """Convert a string value with units (K, M, G) to float in bytes."""
        try:
            # Skip header lines or empty values
            if not value_str or value_str.startswith('-') or value_str in ['free', 'used', 'buff', 'cach', 'total', 'util']:
                return 0.0
            
            # Remove % if present and convert directly
            if value_str.endswith('%'):
                return float(value_str[:-1])

            # Convert number with units
            value_str = value_str.strip().upper()
            if value_str.endswith('K'):
                return float(value_str[:-1]) * 1024
            elif value_str.endswith('M'):
                return float(value_str[:-1]) * 1024 * 1024
            elif value_str.endswith('G'):
                return float(value_str[:-1]) * 1024 * 1024 * 1024
            else:
                return float(value_str)
        except ValueError as e:
            logger.error(f"Error converting value '{value_str}': {e}")
            return 0.0

    def _parse_time_series_data(
        self, lines: List[str], metric_indices: Dict[str, int], skip_first_n: int = 0
    ) -> Dict[str, Any]:
        """Parse time series data from lines into metrics"""
        logger.debug(f"Parsing time series data with {len(lines)} lines")
        metrics = {}
        max_points = 60  # Keep last 60 data points (5 minutes with 5s interval)
        
        for metric_name, idx in metric_indices.items():
            data_points = []
            latest = 0.0
            valid_points = 0

            parsing_data = False
            for line_num, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue

                if any(header in line for header in ["Time", "-----", "MAX", "MEAN", "MIN"]):
                    parsing_data = "Time" in line and not "-----" in line
                    continue

                if not parsing_data:
                    continue

                parts = line.split()
                if len(parts) > idx:
                    try:
                        timestamp = parts[0]
                        raw_value = parts[idx]
                        value = self._convert_to_bytes(raw_value)
                        
                        if metric_name in ['free', 'used', 'buff', 'cach', 'total'] and value > 1024:
                            value = value / (1024 * 1024)
                            
                        data_points.append({"timestamp": timestamp, "value": value})
                        latest = value
                        valid_points += 1
                    except (ValueError, IndexError) as e:
                        logger.error(f"Error parsing metric {metric_name} at line {line_num + 1}: {e}")
                        continue

            # Only keep the most recent data points
            if len(data_points) > max_points:
                data_points = data_points[-max_points:]

            metrics[metric_name] = {"data": data_points, "latest": latest}
            logger.debug(f"Parsed {valid_points} points for {metric_name}, kept last {len(data_points)} points")

        return metrics

    def _get_cpu_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get CPU utilization metrics using psutil."""
        value = psutil.cpu_percent(interval=1)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return {
            "latest": value,
            "min": 0,
            "max": 100,
            "data": [{"timestamp": timestamp, "value": value}]
        }

    def _get_memory_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get memory utilization metrics using psutil."""
        mem = psutil.virtual_memory()
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return {
            "total": {
                "latest": mem.total / (1024 * 1024),  # Convert to MB
                "min": 0,
                "max": mem.total / (1024 * 1024),
                "data": [{"timestamp": timestamp, "value": mem.total / (1024 * 1024)}]
            },
            "used": {
                "latest": mem.used / (1024 * 1024),
                "min": 0,
                "max": mem.total / (1024 * 1024),
                "data": [{"timestamp": timestamp, "value": mem.used / (1024 * 1024)}]
            },
            "util": {
                "latest": mem.percent,
                "min": 0,
                "max": 100,
                "data": [{"timestamp": timestamp, "value": mem.percent}]
            }
        }

    def _get_io_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get disk I/O metrics using psutil."""
        io = psutil.disk_io_counters()
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        if not io:
            return {
                "read_bytes": {
                    "latest": 0, "min": 0, "max": 100,
                    "data": [{"timestamp": timestamp, "value": 0}]
                },
                "write_bytes": {
                    "latest": 0, "min": 0, "max": 100,
                    "data": [{"timestamp": timestamp, "value": 0}]
                },
                "util": {
                    "latest": 0, "min": 0, "max": 100,
                    "data": [{"timestamp": timestamp, "value": 0}]
                }
            }
        
        disk_usage = psutil.disk_usage('/')
        read_bytes = io.read_bytes / (1024 * 1024)  # Convert to MB
        write_bytes = io.write_bytes / (1024 * 1024)
        
        return {
            "read_bytes": {
                "latest": read_bytes,
                "min": 0,
                "max": read_bytes * 1.2,  # 20% headroom
                "data": [{"timestamp": timestamp, "value": read_bytes}]
            },
            "write_bytes": {
                "latest": write_bytes,
                "min": 0,
                "max": write_bytes * 1.2,
                "data": [{"timestamp": timestamp, "value": write_bytes}]
            },
            "util": {
                "latest": disk_usage.percent,
                "min": 0,
                "max": 100,
                "data": [{"timestamp": timestamp, "value": disk_usage.percent}]
            }
        }

    def _get_network_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get network I/O metrics using psutil."""
        net = psutil.net_io_counters()
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        bytes_sent = net.bytes_sent / 1024  # Convert to KB
        bytes_recv = net.bytes_recv / 1024
        
        return {
            "bytes_sent": {
                "latest": bytes_sent,
                "min": 0,
                "max": bytes_sent * 1.2,
                "data": [{"timestamp": timestamp, "value": bytes_sent}]
            },
            "bytes_recv": {
                "latest": bytes_recv,
                "min": 0,
                "max": bytes_recv * 1.2,
                "data": [{"timestamp": timestamp, "value": bytes_recv}]
            },
            "packets_sent": {
                "latest": net.packets_sent,
                "min": 0,
                "max": net.packets_sent * 1.2,
                "data": [{"timestamp": timestamp, "value": net.packets_sent}]
            },
            "packets_recv": {
                "latest": net.packets_recv,
                "min": 0,
                "max": net.packets_recv * 1.2,
                "data": [{"timestamp": timestamp, "value": net.packets_recv}]
            }
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get the latest collected metrics."""
        current_timestamp = self.timestamp
        if current_timestamp and len(current_timestamp) == 14:  # Format: YYYYMMDDHHMMSS
            try:
                dt = datetime.strptime(current_timestamp, "%Y%m%d%H%M%S")
                current_timestamp = dt.isoformat()
            except ValueError:
                current_timestamp = datetime.now().isoformat()

        # Create a simplified version of metrics for initial load
        simplified_metrics = {}
        for node_ip, node_data in self.metrics.items():
            simplified_metrics[node_ip] = {}
            for category, metrics in node_data.items():
                if category == 'io':
                    # For IO, only include aggregated metrics
                    simplified_metrics[node_ip][category] = {
                        'aggregated': {
                            metric: {'latest': data['latest']}
                            for metric, data in metrics['aggregated'].items()
                        }
                    }
                else:
                    # For other categories, only include latest values
                    simplified_metrics[node_ip][category] = {
                        metric: {'latest': data['latest']}
                        for metric, data in metrics.items()
                    }

        return {
            "timestamp": current_timestamp,
            "metrics": simplified_metrics
        }

    def get_detailed_metrics(self, node_ip: str = None, category: str = None) -> Dict[str, Any]:
        """Get detailed metrics with historical data for specific node and category."""
        if not node_ip or not category or node_ip not in self.metrics:
            return {"error": "Invalid node_ip or category"}

        node_data = self.metrics[node_ip]
        if category not in node_data:
            return {"error": "Category not found"}

        return {
            "timestamp": self.timestamp,
            "metrics": {
                node_ip: {
                    category: node_data[category]
                }
            }
        }

# Create a singleton instance
metrics_service = MetricsService()
