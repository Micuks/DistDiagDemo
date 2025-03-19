import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Table, Button, Space, Typography, Statistic, Tag, Spin, Divider, message, Tabs } from 'antd';
import { ReloadOutlined, StopOutlined, WarningOutlined, CheckCircleOutlined, LoadingOutlined, PlusOutlined } from '@ant-design/icons';
import { workloadService } from '../services/workloadService';
import { anomalyService } from '../services/anomalyService';
import { useAnomalyData } from '../hooks/useAnomalyData';

const { Title, Text } = Typography;
const { TabPane } = Tabs;

const ExecutionDashboard = ({ workloadConfig, anomalyConfig, onReset, onNewExecution, tasksHistory, setTasksHistory }) => {
  const [activeWorkloads, setActiveWorkloads] = useState([]);
  const [workloadLoading, setWorkloadLoading] = useState(false);
  const [workloadStopLoading, setWorkloadStopLoading] = useState(false);
  const { data: activeAnomalies = [], isLoading: anomalyLoading, refetch: refetchAnomalies } = useAnomalyData();
  const [anomalyStopLoading, setAnomalyStopLoading] = useState(false);
  const [refreshInterval, setRefreshInterval] = useState(null);

  // Define columns for tasks table
  const taskColumns = [
    {
      title: 'Task Name',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: 'Workload Type',
      dataIndex: ['workload', 'type'],
      key: 'workloadType',
      render: (type) => {
        const types = {
          sysbench: 'Sysbench OLTP',
          tpcc: 'TPC-C',
          tpch: 'TPC-H'
        };
        return <Tag color="blue">{types[type] || type}</Tag>;
      }
    },
    {
      title: 'Anomalies',
      dataIndex: 'anomalies',
      key: 'anomalies',
      render: (anomalies) => {
        if (!anomalies || anomalies.length === 0) {
          return <Tag color="green">Normal Scenario</Tag>;
        }
        return (
          <Space size={[0, 4]} wrap>
            {anomalies.map((anomaly) => (
              <Tag color="red" key={anomaly.id}>
                {anomaly.type}
              </Tag>
            ))}
          </Space>
        );
      }
    },
    {
      title: 'Start Time',
      dataIndex: 'startTime',
      key: 'startTime',
      render: (time) => new Date(time).toLocaleString()
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status) => (
        <Tag color={status === 'running' ? 'green' : status === 'error' ? 'red' : 'orange'}>
          {status.toUpperCase()}
        </Tag>
      )
    },
    {
      title: 'Action',
      key: 'action',
      render: (_, record) => (
        <Button
          type="primary"
          danger
          icon={<StopOutlined />}
          onClick={() => handleStopTask(record)}
          loading={workloadStopLoading}
        >
          Stop Task
        </Button>
      ),
    },
  ];

  // Define columns for workload table
  const workloadColumns = [
    {
      title: 'ID',
      dataIndex: 'id',
      key: 'id',
    },
    {
      title: 'Type',
      dataIndex: 'type',
      key: 'type',
      render: (type) => {
        const types = {
          sysbench: 'Sysbench OLTP',
          tpcc: 'TPC-C',
          tpch: 'TPC-H'
        };
        return <Tag color="blue">{types[type] || type}</Tag>;
      }
    },
    {
      title: 'Start Time',
      dataIndex: 'startTime',
      key: 'startTime',
      render: (time) => new Date(time).toLocaleString()
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status) => (
        <Tag color={status === 'running' ? 'green' : status === 'error' ? 'red' : 'orange'}>
          {status.toUpperCase()}
        </Tag>
      )
    },
    {
      title: 'Action',
      key: 'action',
      render: (_, record) => (
        <Button
          type="primary"
          danger
          icon={<StopOutlined />}
          onClick={() => handleStopWorkload(record.id)}
          loading={workloadStopLoading}
        >
          Stop
        </Button>
      ),
    },
  ];

  // Define columns for anomaly table
  const anomalyColumns = [
    {
      title: 'Type',
      dataIndex: 'type',
      key: 'type',
      render: (type) => {
        const types = {
          cpu_stress: 'CPU Stress',
          io_bottleneck: 'I/O Bottleneck',
          network_bottleneck: 'Network Bottleneck',
          cache_bottleneck: 'Cache Bottleneck',
          too_many_indexes: 'Too Many Indexes',
        };
        return <Tag color="red">{types[type] || type}</Tag>;
      }
    },
    {
      title: 'Target Node',
      dataIndex: 'node',
      key: 'node',
      render: (node) => <Tag color="orange">{node || 'Default'}</Tag>
    },
    {
      title: 'Start Time',
      dataIndex: 'start_time',
      key: 'start_time',
      render: (time) => new Date(time).toLocaleString()
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status) => (
        <Tag color={status === 'active' ? 'green' : 'red'}>
          {String(status).toUpperCase()}
        </Tag>
      )
    },
    {
      title: 'Action',
      key: 'action',
      render: (_, record) => (
        <Button 
          type="primary" 
          danger 
          icon={<StopOutlined />}
          onClick={() => handleStopAnomaly(record.type)}
          loading={anomalyStopLoading}
        >
          Stop
        </Button>
      ),
    },
  ];

  // Functions to fetch active workloads and anomalies
  const fetchActiveWorkloads = async () => {
    try {
      setWorkloadLoading(true);
      const data = await workloadService.getActiveWorkloads();
      
      // Update the activeWorkloads state
      setActiveWorkloads(data.workloads || []);
      
      // Also update task status in tasksHistory
      if (data.workloads && data.workloads.length > 0) {
        const updatedTasks = tasksHistory.map(task => {
          const matchingWorkload = data.workloads.find(w => w.id === task.workloadId);
          if (matchingWorkload) {
            return { ...task, status: matchingWorkload.status };
          }
          return task;
        });
        setTasksHistory(updatedTasks);
      }
    } catch (error) {
      console.error('Failed to fetch workloads:', error);
      message.error('Failed to load active workloads');
    } finally {
      setWorkloadLoading(false);
    }
  };

  // Functions to handle stopping workloads and anomalies
  const handleStopTask = async (task) => {
    try {
      setWorkloadStopLoading(true);
      
      // Stop the workload
      if (task.workloadId) {
        await workloadService.stopWorkload(task.workloadId);
      }
      
      // Stop any associated anomalies
      if (task.anomalies && task.anomalies.length > 0) {
        for (const anomaly of task.anomalies) {
          await anomalyService.stopAnomaly(anomaly.type);
        }
      }
      
      // Update task status
      const updatedTasks = tasksHistory.map(t => {
        if (t.id === task.id) {
          return { ...t, status: 'stopped' };
        }
        return t;
      });
      
      setTasksHistory(updatedTasks);
      message.success('Task stopped successfully');
      
      // Refresh data
      fetchActiveWorkloads();
      refetchAnomalies();
    } catch (error) {
      console.error('Failed to stop task:', error);
      message.error(`Failed to stop task: ${error.message}`);
    } finally {
      setWorkloadStopLoading(false);
    }
  };

  const handleStopWorkload = async (workloadId) => {
    try {
      setWorkloadStopLoading(true);
      await workloadService.stopWorkload(workloadId);
      
      // Update task status if this workload is associated with a task
      const updatedTasks = tasksHistory.map(task => {
        if (task.workloadId === workloadId) {
          return { ...task, status: 'stopped' };
        }
        return task;
      });
      
      setTasksHistory(updatedTasks);
      message.success('Workload stopped successfully');
      fetchActiveWorkloads();
    } catch (error) {
      console.error('Failed to stop workload:', error);
      message.error(`Failed to stop workload: ${error.message}`);
    } finally {
      setWorkloadStopLoading(false);
    }
  };

  const handleStopAnomaly = async (anomalyType) => {
    try {
      setAnomalyStopLoading(true);
      await anomalyService.stopAnomaly(anomalyType);
      message.success(`Anomaly ${anomalyType} stopped`);
      refetchAnomalies();
    } catch (error) {
      console.error('Failed to stop anomaly:', error);
      message.error(`Failed to stop anomaly: ${error.message}`);
    } finally {
      setAnomalyStopLoading(false);
    }
  };

  const handleStopAllWorkloads = async () => {
    try {
      setWorkloadStopLoading(true);
      await workloadService.stopAllWorkloads();
      
      // Update all tasks with running status to stopped
      const updatedTasks = tasksHistory.map(task => {
        if (task.status === 'running') {
          return { ...task, status: 'stopped' };
        }
        return task;
      });
      
      setTasksHistory(updatedTasks);
      message.success('All workloads stopped');
      fetchActiveWorkloads();
    } catch (error) {
      console.error('Failed to stop all workloads:', error);
      message.error(`Failed to stop all workloads: ${error.message}`);
    } finally {
      setWorkloadStopLoading(false);
    }
  };

  const handleStopAllAnomalies = async () => {
    try {
      setAnomalyStopLoading(true);
      await anomalyService.stopAllAnomalies();
      message.success('All anomalies stopped');
      refetchAnomalies();
    } catch (error) {
      console.error('Failed to stop all anomalies:', error);
      message.error(`Failed to stop all anomalies: ${error.message}`);
    } finally {
      setAnomalyStopLoading(false);
    }
  };

  // Set up periodic refresh
  useEffect(() => {
    // Initial fetch
    fetchActiveWorkloads();
    refetchAnomalies();

    // Set up interval for periodic refresh
    const interval = setInterval(() => {
      fetchActiveWorkloads();
      refetchAnomalies();
    }, 5000);
    setRefreshInterval(interval);

    // Clean up on unmount
    return () => {
      if (refreshInterval) {
        clearInterval(refreshInterval);
      }
    };
  }, []);

  const handleRefreshData = () => {
    fetchActiveWorkloads();
    refetchAnomalies();
    message.info('Refreshing data...');
  };

  const calculateSystemStatus = () => {
    const hasActiveAnomalies = activeAnomalies.length > 0;
    const hasActiveWorkloads = activeWorkloads.length > 0;
    
    if (!hasActiveWorkloads && !hasActiveAnomalies) {
      return { status: 'idle', color: 'gray', text: 'System Idle' };
    } else if (hasActiveAnomalies) {
      return { status: 'anomaly', color: 'red', text: 'Anomalies Active' };
    } else if (hasActiveWorkloads) {
      return { status: 'running', color: 'green', text: 'Workload Running' };
    }
  };

  const systemStatus = calculateSystemStatus();
  
  // Get active tasks by filtering tasks with running status
  const activeTasks = tasksHistory.filter(task => task.status === 'running');

  return (
    <Space direction="vertical" style={{ width: '100%' }} size="large">
      <Card>
        <Row gutter={[16, 16]}>
          <Col span={24}>
            <Space style={{ width: '100%', justifyContent: 'space-between' }}>
              <Title level={4}>Execution Dashboard</Title>
              <Space>
                <Button 
                  icon={<ReloadOutlined />} 
                  onClick={handleRefreshData}
                >
                  Refresh
                </Button>
                <Button 
                  type="primary"
                  icon={<PlusOutlined />}
                  onClick={onNewExecution}
                >
                  New Execution
                </Button>
                <Button 
                  type="default"
                  onClick={onReset}
                >
                  Reset Dashboard
                </Button>
              </Space>
            </Space>
          </Col>
        </Row>

        {/* Status Overview */}
        <Row gutter={[16, 24]} style={{ marginTop: 16 }}>
          <Col xs={24} sm={8}>
            <Card>
              <Statistic 
                title="System Status"
                value={systemStatus.text}
                valueStyle={{ color: systemStatus.status === 'anomaly' ? '#cf1322' : 
                              systemStatus.status === 'running' ? '#3f8600' : '#8c8c8c' }}
                prefix={systemStatus.status === 'anomaly' ? <WarningOutlined /> : 
                        systemStatus.status === 'running' ? <CheckCircleOutlined /> : <LoadingOutlined />}
              />
            </Card>
          </Col>
          <Col xs={24} sm={8}>
            <Card>
              <Statistic 
                title="Active Tasks" 
                value={activeTasks.length} 
                valueStyle={{ color: activeTasks.length > 0 ? '#3f8600' : '#8c8c8c' }}
              />
            </Card>
          </Col>
          <Col xs={24} sm={8}>
            <Card>
              <Statistic 
                title="Active Anomalies" 
                value={activeAnomalies.length} 
                valueStyle={{ color: activeAnomalies.length > 0 ? '#cf1322' : '#8c8c8c' }}
              />
            </Card>
          </Col>
        </Row>

        <Tabs defaultActiveKey="tasks" style={{ marginTop: 16 }}>
          <TabPane tab="Tasks" key="tasks">
            <Card 
              title={
                <Space>
                  <Title level={5}>Task Status</Title>
                  {workloadLoading && <Spin size="small" />}
                </Space>
              }
              extra={
                <Button 
                  type="primary" 
                  danger 
                  icon={<StopOutlined />}
                  onClick={handleStopAllWorkloads}
                  disabled={activeTasks.length === 0}
                  loading={workloadStopLoading}
                >
                  Stop All Tasks
                </Button>
              }
            >
              <Table 
                columns={taskColumns} 
                dataSource={tasksHistory} 
                rowKey="id" 
                pagination={false}
                locale={{ emptyText: 'No tasks created yet' }}
                loading={workloadLoading}
              />
            </Card>
          </TabPane>

          <TabPane tab="Workloads" key="workloads">
            <Card 
              title={
                <Space>
                  <Title level={5}>Workload Status</Title>
                  {workloadLoading && <Spin size="small" />}
                </Space>
              }
              extra={
                <Button 
                  type="primary" 
                  danger 
                  icon={<StopOutlined />}
                  onClick={handleStopAllWorkloads}
                  disabled={activeWorkloads.length === 0}
                  loading={workloadStopLoading}
                >
                  Stop All Workloads
                </Button>
              }
            >
              <Table 
                columns={workloadColumns} 
                dataSource={activeWorkloads} 
                rowKey="id" 
                pagination={false}
                locale={{ emptyText: 'No active workloads' }}
                loading={workloadLoading}
              />
            </Card>
          </TabPane>

          <TabPane tab="Anomalies" key="anomalies">
            <Card 
              title={
                <Space>
                  <Title level={5}>Anomaly Status</Title>
                  {anomalyLoading && <Spin size="small" />}
                </Space>
              }
              extra={
                <Button 
                  type="primary" 
                  danger 
                  icon={<StopOutlined />}
                  onClick={handleStopAllAnomalies}
                  disabled={activeAnomalies.length === 0}
                  loading={anomalyStopLoading}
                >
                  Stop All Anomalies
                </Button>
              }
            >
              <Table 
                columns={anomalyColumns} 
                dataSource={activeAnomalies} 
                rowKey={(record) => record.type + record.node} 
                pagination={false}
                locale={{ emptyText: 'No active anomalies' }}
                loading={anomalyLoading}
              />
            </Card>
          </TabPane>
        </Tabs>
      </Card>
    </Space>
  );
};

export default ExecutionDashboard;