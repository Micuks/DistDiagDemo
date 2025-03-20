import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Table, Button, Space, Typography, Statistic, Tag, Spin, Divider, message, Tabs } from 'antd';
import { ReloadOutlined, StopOutlined, WarningOutlined, CheckCircleOutlined, LoadingOutlined, PlusOutlined } from '@ant-design/icons';
import { workloadService } from '../services/workloadService';
import { anomalyService } from '../services/anomalyService';
import { useAnomalyData } from '../hooks/useAnomalyData';

const { Title, Text } = Typography;
const { TabPane } = Tabs;

const ExecutionDashboard = ({ workloadConfig, anomalyConfig, onReset, onNewExecution }) => {
  const [activeWorkloads, setActiveWorkloads] = useState([]);
  const [workloadLoading, setWorkloadLoading] = useState(true);
  const [workloadStopLoading, setWorkloadStopLoading] = useState(false);
  const { data: activeAnomalies = [], isLoading: anomalyLoading, refetch: refetchAnomalies } = useAnomalyData();
  const [anomalyStopLoading, setAnomalyStopLoading] = useState(false);
  const [refreshInterval, setRefreshInterval] = useState(null);
  const [isInitialLoad, setIsInitialLoad] = useState(true);
  const [tasks, setTasks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);

  // Set flag for execution dashboard in localStorage
  useEffect(() => {
    // Set the flag when component mounts
    localStorage.setItem('onExecutionDashboard', 'true');
    
    // Dispatch a custom event to ensure other components are notified
    window.dispatchEvent(new Event('storage'));
    
    // Clear the flag when component unmounts
    return () => {
      localStorage.removeItem('onExecutionDashboard');
      // Dispatch event again when removed
      window.dispatchEvent(new Event('storage'));
    };
  }, []);

  // Define columns for tasks table
  const taskColumns = [
    {
      title: 'Task Name',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: 'Workload Type',
      dataIndex: 'workload_type',
      key: 'workload_type',
      render: (type) => (
        <Tag color={type === 'tpcc' ? 'blue' : 'green'}>
          {type.toUpperCase()}
        </Tag>
      )
    },
    {
      title: 'Anomalies',
      dataIndex: 'anomalies',
      key: 'anomalies',
      render: (anomalies) => (
        <Space>
          {anomalies?.map((anomaly, index) => (
            <Tag key={index} color="red">
              {typeof anomaly === 'object' ? anomaly.type.replace(/_/g, ' ') : anomaly}
            </Tag>
          ))}
        </Space>
      )
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
        <Tag color={status === 'running' ? 'green' : 'red'}>
          {status.toUpperCase()}
        </Tag>
      )
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_, record) => (
        <Button
          type="primary"
          danger
          onClick={() => handleStopWorkload(record.workload_id)}
          disabled={record.status !== 'running'}
        >
          Stop
        </Button>
      ),
    },
  ];

  // Define columns for workloads table
  const workloadColumns = [
    {
      title: 'Workload ID',
      dataIndex: 'id',
      key: 'id',
    },
    {
      title: 'Type',
      dataIndex: 'type',
      key: 'type',
      render: (type) => (
        <Tag color={type === 'tpcc' ? 'blue' : type === 'sysbench' ? 'green' : 'purple'}>
          {type.toUpperCase()}
        </Tag>
      )
    },
    {
      title: 'Threads',
      dataIndex: 'threads',
      key: 'threads',
    },
    {
      title: "Start Time",
      dataIndex: "start_time",
      key: "start_time",
      render: (time) => new Date(time).toLocaleString()
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status) => (
        <Tag color={status === 'running' ? 'green' : 'red'}>
          {status.toUpperCase()}
        </Tag>
      )
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_, record) => (
        <Button
          type="primary"
          danger
          onClick={() => handleStopWorkload(record.id)}
          disabled={record.status !== 'running'}
        >
          Stop
        </Button>
      ),
    },
  ];

  // Define columns for anomalies table
  const anomalyColumns = [
    {
      title: 'Type',
      dataIndex: 'type',
      key: 'type',
      render: (type) => (
        <Tag color="red">
          {type.toUpperCase().replace(/_/g, ' ')}
        </Tag>
      )
    },
    {
      title: 'Node',
      dataIndex: 'node',
      key: 'node',
      render: (node) => Array.isArray(node) ? node.join(', ') : node
    },
    {
      title: 'Start Time',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (time) => time ? new Date(time).toLocaleString() : 'N/A'
    },
    {
      title: 'Duration',
      key: 'duration',
      render: (_, record) => {
        if (!record.created_at) return 'N/A';
        const start = new Date(record.created_at);
        const now = new Date();
        const diffMs = now - start;
        const diffMins = Math.floor(diffMs / 60000);
        const diffSecs = Math.floor((diffMs % 60000) / 1000);
        return `${diffMins}m ${diffSecs}s`;
      }
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_, record) => (
        <Button
          type="primary"
          danger
          onClick={() => handleStopAnomaly(record.type)}
          loading={anomalyStopLoading === record.type}
        >
          Stop
        </Button>
      ),
    },
  ];

  const fetchTasks = async () => {
    try {
      if (!isRefreshing) setLoading(true);
      const activeTasks = await workloadService.getActiveTasks();
      console.log("Active tasks data:", activeTasks);
      setTasks(activeTasks);
    } catch (error) {
      console.error('Error fetching tasks:', error);
      if (isInitialLoad) {
        message.error(`Failed to fetch tasks: ${error.message}`);
      }
    } finally {
      setLoading(false);
    }
  };

  // Fetch active workloads
  const fetchActiveWorkloads = async () => {
    if (workloadLoading && isRefreshing) return; // Prevent multiple concurrent fetches
    
    try {
      if (!isRefreshing) setWorkloadLoading(true);
      const data = await workloadService.getActiveWorkloads();
      console.log("Active workloads data:", data);
      setActiveWorkloads(data);
    } catch (error) {
      console.error('Failed to fetch workloads:', error);
      if (isInitialLoad) {
        message.error(`Failed to load active workloads: ${error.message}`);
      }
    } finally {
      setWorkloadLoading(false);
    }
  };

  // Set up periodic refresh
  useEffect(() => {
    // Initial fetch
    fetchTasks();
    fetchActiveWorkloads();
    refetchAnomalies();

    // Set up interval for periodic refresh
    const interval = setInterval(() => {
      setIsRefreshing(true);
      Promise.all([
        fetchTasks(),
        fetchActiveWorkloads(),
        refetchAnomalies()
      ]).finally(() => {
        setIsRefreshing(false);
        setIsInitialLoad(false);
      });
    }, 5000);
    setRefreshInterval(interval);

    // Clean up on unmount
    return () => {
      if (refreshInterval) {
        clearInterval(refreshInterval);
      }
    };
  }, []);

  const handleStopWorkload = async (workloadId) => {
    try {
      setWorkloadStopLoading(workloadId);
      await workloadService.stopWorkload(workloadId);
      
      message.success('Workload stopped successfully');
      
      // Refresh data
      setIsRefreshing(true);
      await Promise.all([fetchTasks(), fetchActiveWorkloads()]);
      setIsRefreshing(false);
    } catch (error) {
      console.error('Failed to stop workload:', error);
      message.error(`Failed to stop workload: ${error.message}`);
    } finally {
      setWorkloadStopLoading(false);
    }
  };

  const handleStopAnomaly = async (anomalyType) => {
    try {
      setAnomalyStopLoading(anomalyType);
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
      setWorkloadStopLoading('all');
      await workloadService.stopAllWorkloads();
      
      message.success('All workloads stopped');
      setIsRefreshing(true);
      await Promise.all([fetchTasks(), fetchActiveWorkloads()]);
      setIsRefreshing(false);
    } catch (error) {
      console.error('Failed to stop all workloads:', error);
      message.error(`Failed to stop all workloads: ${error.message}`);
    } finally {
      setWorkloadStopLoading(false);
    }
  };

  const handleStopAllAnomalies = async () => {
    try {
      setAnomalyStopLoading('all');
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

  const handleRefreshData = () => {
    setIsRefreshing(true);
    Promise.all([
      fetchTasks(),
      fetchActiveWorkloads(),
      refetchAnomalies()
    ]).finally(() => {
      setIsRefreshing(false);
      message.info('Data refreshed');
    });
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
  const activeTasks = tasks.filter(task => task.status === 'running');

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
                  loading={isRefreshing}
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
                  {(loading && !isRefreshing) && <Spin size="small" />}
                </Space>
              }
              extra={
                <Button 
                  type="primary" 
                  danger 
                  icon={<StopOutlined />}
                  onClick={handleStopAllWorkloads}
                  disabled={activeTasks.length === 0}
                  loading={workloadStopLoading === 'all'}
                >
                  Stop All Tasks
                </Button>
              }
            >
              <Table 
                columns={taskColumns} 
                dataSource={tasks} 
                rowKey="id" 
                pagination={false}
                locale={{ emptyText: 'No tasks created yet' }}
                loading={loading && isInitialLoad}
              />
            </Card>
          </TabPane>

          <TabPane tab="Workloads" key="workloads">
            <Card 
              title={
                <Space>
                  <Title level={5}>Workload Status</Title>
                  {(workloadLoading && !isRefreshing) && <Spin size="small" />}
                </Space>
              }
              extra={
                <Button 
                  type="primary" 
                  danger 
                  icon={<StopOutlined />}
                  onClick={handleStopAllWorkloads}
                  disabled={activeWorkloads.length === 0}
                  loading={workloadStopLoading === 'all'}
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
                loading={workloadLoading && isInitialLoad}
              />
            </Card>
          </TabPane>

          <TabPane tab="Anomalies" key="anomalies">
            <Card 
              title={
                <Space>
                  <Title level={5}>Anomaly Status</Title>
                  {(anomalyLoading && !isRefreshing) && <Spin size="small" />}
                </Space>
              }
              extra={
                <Button 
                  type="primary" 
                  danger 
                  icon={<StopOutlined />}
                  onClick={handleStopAllAnomalies}
                  disabled={activeAnomalies.length === 0}
                  loading={anomalyStopLoading === 'all'}
                >
                  Stop All Anomalies
                </Button>
              }
            >
              <Table 
                columns={anomalyColumns} 
                dataSource={activeAnomalies} 
                rowKey="name" 
                pagination={false}
                locale={{ emptyText: 'No active anomalies' }}
                loading={anomalyLoading && isInitialLoad}
              />
            </Card>
          </TabPane>
        </Tabs>
      </Card>
    </Space>
  );
};

export default ExecutionDashboard;