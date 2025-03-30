import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Table, Button, Space, Typography, Statistic, Tag, Spin, Divider, message, Tabs, Tooltip } from 'antd';
import { ReloadOutlined, StopOutlined, WarningOutlined, CheckCircleOutlined, LoadingOutlined, PlusOutlined, SyncOutlined } from '@ant-design/icons';
import { workloadService } from '../services/workloadService';
import { anomalyService } from '../services/anomalyService';
import { useAnomalyData } from '../hooks/useAnomalyData';
import axios from 'axios';

const { Title, Text } = Typography;
const { TabPane } = Tabs;

const ExecutionDashboard = ({ workloadConfig, anomalyConfig, onReset, onNewExecution }) => {
  const [activeWorkloads, setActiveWorkloads] = useState([]);
  const [workloadLoading, setWorkloadLoading] = useState(true);
  const [workloadStopLoading, setWorkloadStopLoading] = useState(false);
  const [groupedWorkloads, setGroupedWorkloads] = useState([]);
  const { data: activeAnomalies = [], isLoading: anomalyLoading, refetch: refetchAnomalies } = useAnomalyData();
  const [anomalyStopLoading, setAnomalyStopLoading] = useState(false);
  const [refreshInterval, setRefreshInterval] = useState(null);
  const [isInitialLoad, setIsInitialLoad] = useState(true);
  const [tasks, setTasks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [error, setError] = useState(null);
  const [groupedAnomalies, setGroupedAnomalies] = useState([]);
  const [resetCollectionLoading, setResetCollectionLoading] = useState(false);
  const [cleanupTasksLoading, setCleanupTasksLoading] = useState(false);
  const [optimisticAnomalyCleared, setOptimisticAnomalyCleared] = useState(false);

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

  // Group anomalies by type and similar creation time
  useEffect(() => {
    // Group anomalies that are similar (same type, created within 30 seconds of each other)
    const groupAnomalies = (anomalies) => {
      if (!anomalies || anomalies.length === 0) return [];
      
      // Sort by time to ensure consistent grouping
      const sorted = [...anomalies].sort((a, b) => {
        const timeA = new Date(a.start_time).getTime();
        const timeB = new Date(b.start_time).getTime();
        return timeA - timeB;
      });
      
      const groups = {};
      
      sorted.forEach(anomaly => {
        const anomalyType = anomaly.type;
        const timeStamp = new Date(anomaly.start_time).getTime();
        
        // Create a key based on anomaly type
        const key = anomalyType;
        
        if (!groups[key]) {
          // Create a new group
          groups[key] = {
            ...anomaly,
            nodes: [anomaly.node], // Array of nodes
            names: [anomaly.name], // Array of names (useful for stopping individual ones)
            original_entries: [anomaly] // Keep original entries for reference
          };
        } else {
          // Add to existing group
          if (!groups[key].nodes.includes(anomaly.node)) {
            groups[key].nodes.push(anomaly.node);
          }
          if (!groups[key].names.includes(anomaly.name)) {
            groups[key].names.push(anomaly.name);
          }
          groups[key].original_entries.push(anomaly);
        }
      });
      
      return Object.values(groups);
    };
    
    const grouped = groupAnomalies(activeAnomalies);
    setGroupedAnomalies(grouped);
  }, [activeAnomalies]);

  // Group workloads by type and similar start time
  useEffect(() => {
    const groupWorkloads = (workloads) => {
      if (!workloads || workloads.length === 0) return [];
      
      const groups = {};
      
      workloads.forEach(workload => {
        // Create a key based on workload type and threads
        const key = `${workload.type}_${workload.threads}`;
        
        if (!groups[key]) {
          // Create a new group
          groups[key] = {
            ...workload,
            ids: [workload.id],
            pids: [workload.pid],
            nodes: [workload.node || workload.id.split('_')[2]], // Extract node from ID if not explicitly provided
            original_entries: [workload]
          };
        } else {
          // Add to existing group
          const nodeValue = workload.node || workload.id.split('_')[2]; // Extract node from ID if not explicitly provided
          if (!groups[key].nodes.includes(nodeValue)) {
            groups[key].nodes.push(nodeValue);
          }
          if (!groups[key].ids.includes(workload.id)) {
            groups[key].ids.push(workload.id);
          }
          if (!groups[key].pids.includes(workload.pid)) {
            groups[key].pids.push(workload.pid);
          }
          groups[key].original_entries.push(workload);
        }
      });
      
      return Object.values(groups);
    };
    
    const grouped = groupWorkloads(activeWorkloads);
    setGroupedWorkloads(grouped);
  }, [activeWorkloads]);

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
      title: 'Threads',
      dataIndex: 'workload_config',
      key: 'threads',
      render: (config) => config?.num_threads || '—'
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
          {(!anomalies || anomalies.length === 0) && '—'}
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
          onClick={() => handleStopTask(record.id)}
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
      title: 'PIDs',
      dataIndex: 'pids',
      key: 'pids',
      render: (pids) => {
        if (Array.isArray(pids)) {
          if (pids.length === 1) {
            return pids[0];
          }
          return (
            <span>
              {pids[0]} (+{pids.length - 1} more)
              <div style={{ fontSize: '11px', marginTop: '2px' }}>
                {pids.slice(1, 4).join(', ')}
                {pids.length > 4 ? '...' : ''}
              </div>
            </span>
          );
        }
        return pids;
      }
    },
    {
      title: 'Nodes',
      dataIndex: 'nodes',
      key: 'nodes',
      render: (nodes) => {
        if (Array.isArray(nodes)) {
          if (nodes.length === 1) {
            return nodes[0];
          }
          return (
            <span>
              {nodes[0]} (+{nodes.length - 1} more)
              <div style={{ fontSize: '11px', marginTop: '2px' }}>
                {nodes.slice(1, 4).join(', ')}
                {nodes.length > 4 ? '...' : ''}
              </div>
            </span>
          );
        }
        return nodes;
      }
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
          onClick={() => handleStopWorkload(record.ids)}
          disabled={record.status !== 'running'}
        >
          Stop
        </Button>
      ),
    },
  ];

  // Better empty or null handling for anomaly target and nodes
  const renderNode = (node) => {
    if (!node) return '—';
    if (Array.isArray(node)) {
      return node.length > 0 ? node.join(', ') : '—';
    }
    return node;
  };

  // Helper to format durations
  const formatDuration = (startTime) => {
    if (!startTime) return 'N/A';
    
    try {
      const start = new Date(startTime);
      const now = new Date();
      const diffMs = now - start;
      
      if (isNaN(diffMs) || diffMs < 0) return 'N/A';
      
      const diffMins = Math.floor(diffMs / 60000);
      const diffSecs = Math.floor((diffMs % 60000) / 1000);
      return `${diffMins}m ${diffSecs}s`;
    } catch (error) {
      console.error('Error formatting duration:', error);
      return 'N/A';
    }
  };

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
      title: 'Experiment Name',
      dataIndex: 'name',
      key: 'name',
      ellipsis: true,
      render: (_, record) => {
        // If there are multiple names, show the first one with a count
        if (record.names && record.names.length > 1) {
          return <span>{record.name} (+{record.names.length - 1} more)</span>;
        }
        return record.name;
      }
    },
    {
      title: 'Target',
      dataIndex: 'target',
      key: 'target',
      render: renderNode
    },
    {
      title: 'Node',
      dataIndex: 'nodes',
      key: 'nodes',
      render: (nodes) => {
        if (Array.isArray(nodes)) {
          if (nodes.length === 1) {
            return renderNode(nodes[0]);
          }
          return (
            <span>
              {renderNode(nodes[0])} (+{nodes.length - 1} more)
              <div style={{ fontSize: '11px', marginTop: '2px' }}>
                {nodes.slice(1, 4).map(node => node).join(', ')}
                {nodes.length > 4 ? '...' : ''}
              </div>
            </span>
          );
        }
        return renderNode(nodes);
      }
    },
    {
      title: 'Start Time',
      dataIndex: 'start_time',
      key: 'start_time',
      render: (time) => time ? new Date(time).toLocaleString() : 'N/A'
    },
    {
      title: 'Duration',
      key: 'duration',
      render: (_, record) => formatDuration(record.start_time)
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status) => (
        <Tag color={status === 'active' ? 'red' : 'green'}>
          {status ? status.toUpperCase() : 'ACTIVE'}
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
          onClick={() => handleStopAnomaly(record.type, record.name)}
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
      setError(error.message);
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
      setError(error.message);
    } finally {
      setWorkloadLoading(false);
    }
  };

  // Set up periodic refresh
  useEffect(() => {
    // Initial fetch
    fetchTasks();
    fetchActiveWorkloads();
    
    // Initial anomaly fetch - happens inside useAnomalyData hook
    // No need to force refresh here since the hook already does this

    // Set up interval for periodic refresh
    const interval = setInterval(() => {
      setIsRefreshing(true);
      Promise.all([
        fetchTasks(),
        fetchActiveWorkloads(),
        // Don't force refetch if SSE is working, just read from cache
        // The hook will handle refreshing anomaly data via SSE or its own polling
      ]).finally(() => {
        setIsRefreshing(false);
        setIsInitialLoad(false);
      });
    }, 10000); // Increased from 5000 to 10000 ms
    setRefreshInterval(interval);

    // Clean up on unmount
    return () => {
      if (refreshInterval) {
        clearInterval(refreshInterval);
      }
    };
  }, []);

  const handleStopTask = async (taskId) => {
    try {
      setWorkloadStopLoading(taskId);
      
      // First, get the task details to find its workload and anomalies
      const taskDetails = await workloadService.getTask(taskId);
      
      // Stop the task itself
      await workloadService.stopTask(taskId);
      
      // If the task has a workload ID, stop that workload too
      if (taskDetails.workload_id) {
        try {
          await workloadService.stopWorkload(taskDetails.workload_id);
        } catch (workloadError) {
          console.error(`Failed to stop workload ${taskDetails.workload_id}:`, workloadError);
        }
      }
      
      // Stop any anomalies associated with this task
      if (taskDetails.anomalies && taskDetails.anomalies.length > 0) {
        for (const anomaly of taskDetails.anomalies) {
          try {
            // Handle both string and object anomaly formats
            const anomalyType = typeof anomaly === 'object' ? anomaly.type : anomaly;
            const anomalyName = typeof anomaly === 'object' ? anomaly.name : null;
            
            await anomalyService.stopAnomaly(anomalyType, anomalyName);
          } catch (anomalyError) {
            console.error(`Failed to stop anomaly:`, anomalyError);
          }
        }
      }
      
      message.success('Task and associated workloads/anomalies stopped successfully');
      
      // Refresh data
      setIsRefreshing(true);
      await Promise.all([fetchTasks(), fetchActiveWorkloads(), refetchAnomalies()]);
      setIsRefreshing(false);
    } catch (error) {
      console.error('Failed to stop task:', error);
      message.error(`Failed to stop task: ${error.message}`);
    } finally {
      setWorkloadStopLoading(false);
    }
  };

  const handleStopWorkload = async (workloadIds) => {
    try {
      setWorkloadStopLoading(Array.isArray(workloadIds) ? workloadIds[0] : workloadIds);
      
      if (Array.isArray(workloadIds)) {
        message.loading(`Stopping workloads...`, 1);
        
        // Find the matching grouped workload to get original entries
        const matchingGroup = groupedWorkloads.find(
          group => group.ids && group.ids.some(id => workloadIds.includes(id))
        );
        
        if (matchingGroup && matchingGroup.original_entries) {
          // Stop each workload using its original ID
          for (const workload of matchingGroup.original_entries) {
            try {
              // Use the original ID for stopping
              await workloadService.stopWorkload(workload.id);
            } catch (innerError) {
              console.error(`Failed to stop workload ${workload.id}:`, innerError);
            }
          }
        } else {
          // Fallback to using the IDs directly
          for (const id of workloadIds) {
            try {
              await workloadService.stopWorkload(id);
            } catch (innerError) {
              console.error(`Failed to stop workload ${id}:`, innerError);
            }
          }
        }
        
        message.success('Workloads stopped successfully');
      } else {
        try {
          await workloadService.stopWorkload(workloadIds);
          message.success('Workload stopped successfully');
        } catch (error) {
          // If workload not found (likely already stopped), try to refresh tasks anyway
          console.warn(`Workload ${workloadIds} not found, refreshing tasks to update status`);
        }
      }
      
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

  const handleStopAnomaly = async (anomalyType, anomalyName) => {
    try {
      setAnomalyStopLoading(anomalyType);
      
      // Check if this is a grouped anomaly record with multiple names
      const matchingGroupEntry = groupedAnomalies.find(
        g => g.type === anomalyType && (g.name === anomalyName || g.names?.includes(anomalyName))
      );
      
      if (matchingGroupEntry && matchingGroupEntry.names && matchingGroupEntry.names.length > 1) {
        // If it's a grouped entry, stop all experiments in the group
        message.loading(`Stopping all ${anomalyType} anomalies...`, 1);
        
        // Stop each experiment individually
        for (const name of matchingGroupEntry.names) {
          try {
            await anomalyService.stopAnomaly(anomalyType, name);
          } catch (innerError) {
            console.error(`Failed to stop anomaly ${name}:`, innerError);
            // Continue with others even if one fails
          }
        }
        
        message.success(`All ${anomalyType} anomalies stopped`);
      } else {
        // If it's a single entry, just stop that one
        await anomalyService.stopAnomaly(anomalyType, anomalyName);
        
        if (anomalyName) {
          message.success(`Anomaly ${anomalyName} stopped`);
        } else {
          message.success(`Anomaly ${anomalyType} stopped`);
        }
      }
      
      // Wait a short timeout before refreshing to allow backend to process
      setTimeout(() => {
        refetchAnomalies();
      }, 1000);
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
      setActiveWorkloads([]);
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

  const handleStopAllTasks = async () => {
    try {
      setWorkloadStopLoading('all');
      await Promise.all([
        workloadService.stopAllWorkloads(),
        anomalyService.stopAllAnomalies()
      ]);
      message.success('All tasks, workloads and anomalies stopped');
      setTasks([]);
      setActiveWorkloads([]);
      setGroupedAnomalies([]);
      setOptimisticAnomalyCleared(true);
      setIsRefreshing(true);
      await Promise.all([fetchTasks(), fetchActiveWorkloads(), refetchAnomalies()]);
      setIsRefreshing(false);
      setOptimisticAnomalyCleared(false);
    } catch (error) {
      console.error('Failed to stop all tasks:', error);
      message.error(`Failed to stop all tasks: ${error.message}`);
    } finally {
      setWorkloadStopLoading(false);
    }
  };

  const handleStopAllAnomalies = async () => {
    try {
      setAnomalyStopLoading('all');
      await anomalyService.stopAllAnomalies();
      setOptimisticAnomalyCleared(true);
      setGroupedAnomalies([]);
      message.success('All anomalies stopped');
      
      // Wait a short timeout before refreshing to allow backend to process
      setTimeout(() => {
        refetchAnomalies();
        setOptimisticAnomalyCleared(false);
      }, 1000);
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

  // Add function to handle training collection reset
  const handleResetCollectionState = async () => {
    try {
      setResetCollectionLoading(true);
      
      const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      await axios.post(`${API_BASE_URL}/api/training/force-reset`);
      
      message.success('Collection state reset successfully');
      
      // Refresh data after reset
      handleRefreshData();
    } catch (error) {
      console.error('Error resetting collection state:', error);
      message.error(`Failed to reset collection state: ${error.response?.data?.detail || error.message}`);
    } finally {
      setResetCollectionLoading(false);
    }
  };

  // Add function to clean up old tasks
  const handleCleanupOldTasks = async () => {
    try {
      setCleanupTasksLoading(true);
      const result = await workloadService.cleanupOldTasks();
      
      if (result.removed_count > 0) {
        message.success(`Cleaned up ${result.removed_count} old tasks`);
      } else {
        message.info('No old tasks to clean up');
      }
      
      // Refresh data after cleanup
      handleRefreshData();
    } catch (error) {
      console.error('Error cleaning up old tasks:', error);
      message.error(`Failed to clean up old tasks: ${error.message}`);
    } finally {
      setCleanupTasksLoading(false);
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
                  onClick={handleStopAllTasks}
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
                rowKey={(record) => record.id || Math.random().toString(36).substring(2, 9)}
                pagination={false}
                locale={{ 
                  emptyText: error ? 'Error loading tasks: ' + error : 'No tasks created yet' 
                }}
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
                dataSource={groupedWorkloads} 
                rowKey={(record) => record.ids ? record.ids.join('-') : Math.random().toString(36).substring(2, 9)}
                pagination={false}
                locale={{ 
                  emptyText: error ? 'Error loading workloads: ' + error : 'No active workloads' 
                }}
                loading={workloadLoading && isInitialLoad}
              />
            </Card>
          </TabPane>

          <TabPane tab="Anomalies" key="anomalies">
            <Card 
              title={
                <Space>
                  <Title level={5}>Active Anomalies</Title>
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
                dataSource={optimisticAnomalyCleared ? [] : groupedAnomalies} 
                rowKey={(record) => record.name || record.type || Math.random().toString(36).substring(2, 9)}
                pagination={false}
                locale={{ 
                  emptyText: error ? 'Error loading anomalies: ' + error : 'No active anomalies' 
                }}
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