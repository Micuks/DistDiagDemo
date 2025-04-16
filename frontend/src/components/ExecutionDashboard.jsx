import React, { useState, useEffect, useRef } from 'react';
import { Card, Row, Col, Table, Button, Space, Typography, Statistic, Tag, Spin, Divider, message, Tabs, Tooltip } from 'antd';
import { ReloadOutlined, StopOutlined, WarningOutlined, CheckCircleOutlined, LoadingOutlined, PlusOutlined, SyncOutlined, InfoCircleOutlined, PlayCircleOutlined } from '@ant-design/icons';
import { useTask } from '../hooks/useTask';
import axios from 'axios';

const { Title, Text } = Typography;

const ExecutionDashboard = ({ onReset, onNewExecution }) => {
  const { tasks, loading: taskLoading, error: taskError, fetchTasks, stopTask } = useTask();
  const [stopTaskLoading, setStopTaskLoading] = useState(null);
  const [stopAllLoading, setStopAllLoading] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [resetCollectionLoading, setResetCollectionLoading] = useState(false);

  useEffect(() => {
    localStorage.setItem('onExecutionDashboard', 'true');
    window.dispatchEvent(new Event('storage'));
    return () => {
      localStorage.removeItem('onExecutionDashboard');
      window.dispatchEvent(new Event('storage'));
    };
  }, []);

  const formatAnomalyType = (type) => {
    if (!type) return '';
    return type
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
  };

  const taskColumns = [
    {
      title: 'Task Name',
      dataIndex: 'name',
      key: 'name',
      ellipsis: true,
    },
    {
      title: 'Workload',
      dataIndex: 'workload_type',
      key: 'workload',
      render: (type, record) => (
        <Tag color={type === 'tpcc' ? 'blue' : type === 'sysbench' ? 'green' : 'purple'}>
          {type.toUpperCase()} ({record.workload_config?.num_threads || 'N/A'} threads)
        </Tag>
      )
    },
    {
      title: 'Anomalies',
      dataIndex: 'anomalies',
      key: 'anomalies',
      render: (anomalies) => {
        if (!anomalies || anomalies.length === 0) return '—';
        const displayAnomalies = anomalies.slice(0, 2);
        const remainingCount = anomalies.length - displayAnomalies.length;
        return (
            <Tooltip title={anomalies.map(a => `${formatAnomalyType(a.type)}${a.target ? ` on ${a.target}` : ''}`).join(', ')}>
                 <Space direction="vertical" size="small">
                    {displayAnomalies.map((anomaly, index) => (
                        <Tag key={index} color="red">
                            {formatAnomalyType(anomaly.type)}
                            {anomaly.target ? ` on ${anomaly.target}` : ''}
                            {anomaly.severity ? ` (${anomaly.severity})` : ''}
                        </Tag>
                    ))}
                    {remainingCount > 0 && <Text type="secondary">+{remainingCount} more...</Text>}
                 </Space>
            </Tooltip>
        );
      },
      ellipsis: true,
    },
    {
      title: 'Start Time',
      dataIndex: 'start_time',
      key: 'start_time',
      render: (time) => time ? new Date(time).toLocaleString() : '-',
      sorter: (a, b) => new Date(a.start_time) - new Date(b.start_time),
      defaultSortOrder: 'descend',
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      filters: [
        { text: 'Running', value: 'running' },
        { text: 'Pending', value: 'pending' },
        { text: 'Stopping', value: 'stopping' },
      ],
      onFilter: (value, record) => record.status === value,
      render: (status) => {
            let color = 'default';
            let icon = null;
            if (status === 'running') { color = 'processing'; icon = <SyncOutlined spin />; }
            else if (status === 'pending') { color = 'warning'; icon = <LoadingOutlined />; }
            else if (status === 'stopping') { color = 'warning'; icon = <StopOutlined />; }
            else if (status === 'stopped') { color = 'success'; icon = <CheckCircleOutlined />; }
            else if (status === 'error') { color = 'error'; icon = <WarningOutlined />; }

            return <Tag icon={icon} color={color}>{status ? status.toUpperCase() : 'UNKNOWN'}</Tag>;
        },
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_, record) => {
          const canStop = ['pending', 'running'].includes(record.status);
          return (
            <Button
                type="primary"
                danger
                icon={<StopOutlined />}
                onClick={() => handleStopTask(record.id)}
                disabled={!canStop || stopTaskLoading === record.id}
                loading={stopTaskLoading === record.id}
            >
                Stop
            </Button>
          );
      },
    },
  ];

  const completedTaskColumns = [
    {
      title: 'Task Name',
      dataIndex: 'name',
      key: 'name',
      ellipsis: true,
    },
    {
      title: 'Workload',
      dataIndex: 'workload_type',
      key: 'workload',
      render: (type, record) => (
        <Tag color={type === 'tpcc' ? 'blue' : type === 'sysbench' ? 'green' : 'purple'}>
          {type.toUpperCase()} ({record.workload_config?.num_threads || 'N/A'} threads)
        </Tag>
      )
    },
    {
      title: 'Anomalies',
      dataIndex: 'anomalies',
      key: 'anomalies',
      render: (anomalies) => {
        if (!anomalies || anomalies.length === 0) return '—';
         const displayAnomalies = anomalies.slice(0, 2);
         const remainingCount = anomalies.length - displayAnomalies.length;
         return (
            <Tooltip title={anomalies.map(a => `${formatAnomalyType(a.type)}${a.target ? ` on ${a.target}` : ''}`).join(', ')}>
                 <Space direction="vertical" size="small">
                    {displayAnomalies.map((anomaly, index) => (
                        <Tag key={index} color="red">
                            {formatAnomalyType(anomaly.type)}
                            {anomaly.target ? ` on ${anomaly.target}` : ''}
                            {anomaly.severity ? ` (${anomaly.severity})` : ''}
                        </Tag>
                    ))}
                    {remainingCount > 0 && <Text type="secondary">+{remainingCount} more...</Text>}
                 </Space>
            </Tooltip>
        );
      },
      ellipsis: true,
    },
    {
      title: 'Start Time',
      dataIndex: 'start_time',
      key: 'start_time',
      render: (time) => time ? new Date(time).toLocaleString() : '-',
      sorter: (a, b) => new Date(a.start_time) - new Date(b.start_time),
      defaultSortOrder: 'descend',
    },
    {
      title: 'End Time',
      dataIndex: 'end_time',
      key: 'end_time',
      render: (time) => time ? new Date(time).toLocaleString() : '-',
      sorter: (a, b) => new Date(a.end_time || 0) - new Date(b.end_time || 0),
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      filters: [
        { text: 'Stopped', value: 'stopped' },
        { text: 'Error', value: 'error' },
      ],
      onFilter: (value, record) => record.status === value,
      render: (status) => (
        <Tag color={status === 'stopped' ? 'success' : 'error'}>
          {status ? status.toUpperCase() : 'UNKNOWN'}
        </Tag>
      )
    },
    {
      title: 'Error Message',
      dataIndex: 'error_message',
      key: 'error_message',
      ellipsis: true,
      render: (msg) => msg || '—'
    }
  ];

  const handleStopTask = async (taskId) => {
      setStopTaskLoading(taskId);
      try {
          await stopTask(taskId);
      } catch (error) {
          console.error("Error stopping task caught in dashboard:", error);
      } finally {
          setStopTaskLoading(null);
      }
  };

  const handleStopAllTasks = async () => {
      setStopAllLoading(true);
      const stoppableTasks = tasks.filter(task => ['pending', 'running'].includes(task.status));
      if (stoppableTasks.length === 0) {
          message.info("No running or pending tasks to stop.");
          setStopAllLoading(false);
          return;
      }
      message.loading(`Attempting to stop ${stoppableTasks.length} tasks...`, 0);
      let successCount = 0;
      let failCount = 0;
      const promises = stoppableTasks.map(task =>
          stopTask(task.id)
              .then(() => successCount++)
              .catch(e => {
                  console.error(`Failed to stop task ${task.id}:`, e);
                  failCount++;
              })
      );
      try {
          await Promise.all(promises);
          message.destroy();
          if (failCount === 0) {
              message.success(`Successfully requested stop for ${successCount} tasks.`);
          } else {
              message.warning(`Requested stop for ${successCount} tasks, but ${failCount} failed.`);
          }
      } catch (error) {
          message.destroy();
          message.error("An unexpected error occurred while stopping tasks.");
      } finally {
          setStopAllLoading(false);
      }
  };

  const handleRefreshData = async () => {
    setIsRefreshing(true);
    try {
        await fetchTasks();
        message.info('Data refreshed');
    } catch (error) {
         console.error("Manual refresh failed");
    } finally {
        setIsRefreshing(false);
    }
  };

  const calculateSystemStatus = () => {
    const runningTasks = tasks.filter(t => t.status === 'running');
    const pendingTasks = tasks.filter(t => t.status === 'pending');
    const stoppingTasks = tasks.filter(t => t.status === 'stopping');
    const errorTasks = tasks.filter(t => t.status === 'error');

    if (runningTasks.length > 0) {
      const hasActiveAnomalies = runningTasks.some(t => t.anomalies && t.anomalies.length > 0);
      if (hasActiveAnomalies) {
          return { status: 'anomaly', color: 'red', text: 'Task Running with Anomalies' };
      } else {
          return { status: 'running', color: 'green', text: 'Task Running' };
      }
    } else if (pendingTasks.length > 0) {
         return { status: 'pending', color: 'orange', text: 'Task Pending' };
    } else if (stoppingTasks.length > 0) {
         return { status: 'stopping', color: 'orange', text: 'Task Stopping' };
    } else if (errorTasks.length > 0) {
        return { status: 'error', color: 'red', text: 'Task Error State' };
    } else {
      return { status: 'idle', color: 'gray', text: 'System Idle' };
    }
  };

  const systemStatus = calculateSystemStatus();

  const currentTasks = tasks.filter(task => ['running', 'pending', 'stopping'].includes(task.status));
  const completedTasks = tasks.filter(task => ['stopped', 'error'].includes(task.status));

  const handleResetCollectionState = async () => {
    try {
      setResetCollectionLoading(true);
      const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8001';
      await axios.post(`${API_BASE_URL}/api/training/force-reset`);
      message.success('Collection state reset successfully');
      handleRefreshData();
    } catch (error) {
      console.error('Error resetting collection state:', error);
      message.error(`Failed to reset collection state: ${error.response?.data?.detail || error.message}`);
    } finally {
      setResetCollectionLoading(false);
    }
  };

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
                  loading={isRefreshing || taskLoading}
                >
                  Refresh
                </Button>
                <Button
                  type="primary"
                  icon={<PlusOutlined />}
                  onClick={onNewExecution}
                >
                  New Task
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

        <Row gutter={[16, 24]} style={{ marginTop: 16 }}>
          <Col xs={24} sm={8}>
            <Card>
              <Statistic
                title="System Status"
                value={systemStatus.text}
                valueStyle={{ color: systemStatus.status === 'anomaly' || systemStatus.status === 'error' ? '#cf1322' :
                              systemStatus.status === 'running' ? '#3f8600' :
                              systemStatus.status === 'pending' || systemStatus.status === 'stopping' ? '#faad14' : '#8c8c8c' }}
                prefix={systemStatus.status === 'anomaly' || systemStatus.status === 'error' ? <WarningOutlined /> :
                        systemStatus.status === 'running' ? <SyncOutlined spin /> :
                        systemStatus.status === 'pending' || systemStatus.status === 'stopping' ? <LoadingOutlined /> : <CheckCircleOutlined />}
              />
            </Card>
          </Col>
          <Col xs={24} sm={8}>
            <Card>
              <Statistic
                title="Active/Pending Tasks"
                value={currentTasks.length}
                valueStyle={{ color: currentTasks.length > 0 ? '#1890ff' : '#8c8c8c' }}
              />
            </Card>
          </Col>
          <Col xs={24} sm={8}>
            <Card>
               <Statistic
                title="Completed Tasks"
                value={completedTasks.length}
                valueStyle={{ color: '#8c8c8c' }}
              />
            </Card>
          </Col>
        </Row>

        <Tabs 
          defaultActiveKey="active_tasks" 
          style={{ marginTop: 16 }} 
          destroyInactiveTabPane
          items={[
            {
              key: 'active_tasks',
              label: 'Active Tasks',
              children: (
                <Card
                  title={
                    <Space>
                      <Title level={5}>Current Tasks (Pending/Running/Stopping)</Title>
                      {(taskLoading && !isRefreshing) && <Spin size="small" />}
                    </Space>
                  }
                  extra={
                    <Button
                      type="primary"
                      danger
                      icon={<StopOutlined />}
                      onClick={handleStopAllTasks}
                      disabled={currentTasks.length === 0}
                      loading={stopAllLoading}
                    >
                      Stop All Tasks
                    </Button>
                  }
                >
                  <Table
                    columns={taskColumns}
                    dataSource={currentTasks}
                    rowKey="id"
                    pagination={false}
                    locale={{
                      emptyText: taskError ? `Error loading tasks: ${taskError}` : 'No active tasks'
                    }}
                    loading={taskLoading && !isRefreshing}
                  />
                </Card>
              )
            },
            {
              key: 'history',
              label: 'Task History',
              children: (
                <Card
                  title={
                    <Space>
                      <Title level={5}>Completed Tasks (Stopped/Error)</Title>
                      {(taskLoading && !isRefreshing) && <Spin size="small" />}
                    </Space>
                  }
                  extra={
                     <Button
                        icon={<SyncOutlined />}
                        onClick={handleResetCollectionState}
                        loading={resetCollectionLoading}
                     >
                        Reset Collection State
                     </Button>
                  }
                >
                  <Table
                    columns={completedTaskColumns}
                    dataSource={completedTasks}
                    rowKey="id"
                    pagination={{ pageSize: 10, showSizeChanger: true }}
                    locale={{
                      emptyText: taskError ? `Error loading history: ${taskError}` : 'No completed tasks found'
                    }}
                    loading={taskLoading && !isRefreshing}
                  />
                </Card>
              )
            }
          ]}
        />
      </Card>
    </Space>
  );
};

export default ExecutionDashboard;