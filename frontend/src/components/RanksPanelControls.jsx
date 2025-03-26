import React from "react";
import { SyncOutlined, SearchOutlined, AlertOutlined } from '@ant-design/icons';
import {
  Row,
  Col,
  Typography,
  Select,
  Button,
  Alert,
  Switch,
  Tag,
  Slider,
  Checkbox,
  Card,
  Radio
} from "antd";

const { Text, Title } = Typography;
const { Option } = Select;

const RanksPanelControls = ({
  availableModels = [],
  selectedModels = [],
  comparisonLoading = false,
  comparisonError = null,
  timeRange = "1h",
  autoAnalysisEnabled = true,
  hasFluctuations = false,
  filteredNodes = [],
  selectedMetrics = [],
  thresholdValue = 0.5,
  chartType = "radar",
  loading = false,
  handleModelSelectionChange,
  handleCompareModels,
  fetchAvailableModels,
  toggleAutoAnalysis,
  handleTimeRangeChange,
  handleNodeFilterChange,
  handleMetricSelection,
  handleThresholdChange,
  handleChartTypeChange,
  getAvailableNodes
}) => {
  return (
    <div className="control-panel">
      <Row gutter={16} align="middle">
        <Col span={24}>
          <Title level={4}>
            Root Cause Analysis Dashboard
            {autoAnalysisEnabled && (
              <Tag color="green" style={{ marginLeft: 8 }}>
                <SyncOutlined spin /> Auto-Analysis Active
              </Tag>
            )}
            {hasFluctuations && (
              <Tag color="orange" style={{ marginLeft: 8 }}>
                <AlertOutlined /> Metric Fluctuations Detected
              </Tag>
            )}
          </Title>
        </Col>
      </Row>
      
      <Row gutter={16} align="middle" style={{ marginBottom: 16 }}>
        <Col xs={24} md={8}>
          <div style={{ marginBottom: 8 }}>
            <Text strong>AI Models for Analysis:</Text>
          </div>
          <Select
            mode="multiple"
            style={{ width: '100%' }}
            placeholder="Select models to analyze"
            value={selectedModels}
            onChange={handleModelSelectionChange}
            optionLabelProp="label"
            loading={loading}
            maxTagCount={2}
            showArrow
            bordered
            className="model-select"
          >
            {availableModels.map((model) => (
              <Option 
                key={model || `model-${Math.random()}`} 
                value={model || ""}
                label={model || "Unknown model"}
              >
                <Text>{model || "Unknown model"}</Text>
              </Option>
            ))}
          </Select>
        </Col>
        
        <Col xs={24} md={6}>
          <div style={{ marginBottom: 8 }}>
            <Text strong>Time Range:</Text>
          </div>
          <Select
            style={{ width: '100%' }}
            placeholder="Time range"
            value={timeRange}
            onChange={handleTimeRangeChange}
            showArrow
            bordered
          >
            <Option value="1h">Last hour</Option>
            <Option value="6h">Last 6 hours</Option>
            <Option value="24h">Last 24 hours</Option>
            <Option value="7d">Last 7 days</Option>
          </Select>
        </Col>
        
        <Col xs={24} md={4}>
          <div style={{ marginBottom: 8 }}>
            <Text strong>&nbsp;</Text>
          </div>
          <Button
            type="primary"
            onClick={handleCompareModels}
            loading={comparisonLoading}
            disabled={selectedModels.length === 0}
            block
            icon={<SearchOutlined />}
          >
            Run Analysis
          </Button>
        </Col>
        
        <Col xs={24} md={3}>
          <div style={{ marginBottom: 8 }}>
            <Text strong>&nbsp;</Text>
          </div>
          <Button 
            icon={<SyncOutlined />} 
            onClick={fetchAvailableModels}
            disabled={loading}
          >
            Refresh Models
          </Button>
        </Col>

        <Col xs={24} md={3}>
          <div style={{ marginBottom: 8 }}>
            <Text strong>Auto Analysis:</Text>
          </div>
          <Switch 
            checked={autoAnalysisEnabled}
            onChange={toggleAutoAnalysis}
            checkedChildren="On"
            unCheckedChildren="Off"
          />
        </Col>
      </Row>
      
      {availableModels.length === 0 && loading && (
        <Alert
          message="Loading available models..."
          type="info"
          showIcon
          icon={<SyncOutlined spin />}
          style={{ marginBottom: 16 }}
        />
      )}

      {selectedModels.length === 0 && availableModels.length > 0 && (
        <Alert
          message="Please select a model to analyze"
          type="warning"
          showIcon
          style={{ marginBottom: 16 }}
        />
      )}
      
      {comparisonError && (
        <Alert
          message={comparisonError}
          type="error"
          showIcon
          style={{ marginBottom: 16 }}
        />
      )}
      
      {selectedModels.length > 0 && (
        <Row gutter={16} style={{ marginTop: 16 }}>
          <Col span={24}>
            <Card size="small" title="Filters & Options">
              <Row gutter={16}>
                <Col xs={24} md={6}>
                  <div style={{ marginBottom: 8 }}>
                    <Text strong>Node Filter:</Text>
                  </div>
                  <Select
                    mode="multiple"
                    style={{ width: '100%' }}
                    placeholder="Filter by node"
                    value={filteredNodes}
                    onChange={handleNodeFilterChange}
                    maxTagCount={2}
                  >
                    {getAvailableNodes().map(node => (
                      <Option key={node} value={node}>{node}</Option>
                    ))}
                  </Select>
                </Col>
                
                <Col xs={24} md={6}>
                  <div style={{ marginBottom: 8 }}>
                    <Text strong>Metrics:</Text>
                  </div>
                  <Checkbox.Group
                    options={[
                      { label: 'CPU', value: 'cpu' },
                      { label: 'Memory', value: 'memory' },
                      { label: 'I/O', value: 'io' },
                      { label: 'Network', value: 'network' }
                    ]}
                    value={selectedMetrics}
                    onChange={handleMetricSelection}
                  />
                </Col>
                
                <Col xs={24} md={6}>
                  <div style={{ marginBottom: 8 }}>
                    <Text strong>Confidence Threshold:</Text>
                  </div>
                  <Slider
                    min={0}
                    max={100}
                    value={thresholdValue}
                    onChange={handleThresholdChange}
                    tooltip={{ formatter: value => `${value}%` }}
                  />
                </Col>
                
                <Col xs={24} md={6}>
                  <div style={{ marginBottom: 8 }}>
                    <Text strong>Chart Type:</Text>
                  </div>
                  <Radio.Group value={chartType} onChange={handleChartTypeChange}>
                    <Radio.Button value="radar">Radar</Radio.Button>
                    <Radio.Button value="bar">Bar</Radio.Button>
                    <Radio.Button value="line">Line</Radio.Button>
                  </Radio.Group>
                </Col>
              </Row>
            </Card>
          </Col>
        </Row>
      )}
    </div>
  );
};

export default RanksPanelControls; 