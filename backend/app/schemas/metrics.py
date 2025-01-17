from pydantic import BaseModel
from typing import Dict, Any, Optional, List

class TimeSeriesPoint(BaseModel):
    timestamp: str
    value: float

class TimeSeriesMetric(BaseModel):
    data: List[TimeSeriesPoint]
    latest: float

class CPUMetrics(BaseModel):
    user: TimeSeriesMetric
    system: TimeSeriesMetric
    wait: TimeSeriesMetric
    hirq: TimeSeriesMetric
    sirq: TimeSeriesMetric
    util: TimeSeriesMetric

class MemoryMetrics(BaseModel):
    free: TimeSeriesMetric
    used: TimeSeriesMetric
    buff: TimeSeriesMetric
    cach: TimeSeriesMetric
    total: TimeSeriesMetric
    util: TimeSeriesMetric

class SwapMetrics(BaseModel):
    total: TimeSeriesMetric
    used: TimeSeriesMetric
    free: TimeSeriesMetric
    util: TimeSeriesMetric
    in_rate: TimeSeriesMetric
    out_rate: TimeSeriesMetric

class TCPUDPMetrics(BaseModel):
    active: TimeSeriesMetric
    pasv: TimeSeriesMetric
    retrans: TimeSeriesMetric
    in_segs: TimeSeriesMetric
    out_segs: TimeSeriesMetric
    udp_in_datagrams: TimeSeriesMetric
    udp_out_datagrams: TimeSeriesMetric
    udp_in_errors: TimeSeriesMetric

class DeviceIOMetrics(BaseModel):
    rrqms: TimeSeriesMetric
    wrqms: TimeSeriesMetric
    rs: TimeSeriesMetric
    ws: TimeSeriesMetric
    rsecs: TimeSeriesMetric
    wsecs: TimeSeriesMetric
    rqsize: TimeSeriesMetric
    rarqsz: TimeSeriesMetric
    warqsz: TimeSeriesMetric
    qusize: TimeSeriesMetric
    io_await: TimeSeriesMetric
    rawait: TimeSeriesMetric
    wawait: TimeSeriesMetric
    svctm: TimeSeriesMetric
    util: TimeSeriesMetric

class IOMetrics(BaseModel):
    devices: Dict[str, DeviceIOMetrics]  # Device name -> metrics
    aggregated: DeviceIOMetrics  # Aggregated metrics across all devices

class NetworkMetrics(BaseModel):
    rx_bytes: TimeSeriesMetric
    tx_bytes: TimeSeriesMetric
    rx_packets: TimeSeriesMetric
    tx_packets: TimeSeriesMetric

class NodeMetrics(BaseModel):
    cpu: Optional[CPUMetrics]
    memory: Optional[MemoryMetrics]
    swap: Optional[SwapMetrics]
    tcp_udp: Optional[TCPUDPMetrics]
    io: Optional[IOMetrics]
    network: Optional[NetworkMetrics]

class MetricsData(BaseModel):
    timestamp: str
    metrics: Dict[str, NodeMetrics]  # IP -> metrics mapping