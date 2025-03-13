import React from 'react';
import { Steps, Progress, Card, Typography, Alert, Spin } from 'antd';
import { ClockCircleOutlined, CheckCircleOutlined, CloseCircleOutlined, LoadingOutlined } from '@ant-design/icons';

const { Title, Text } = Typography;

const ModelTrainingProgress = ({ status }) => {
  if (!status) return null;

  const { stage, progress, message, error } = status;

  // Define training stages
  const stages = [
    { key: 'preprocessing', title: 'Data Preprocessing', description: 'Preparing and processing training data' },
    { key: 'training', title: 'Model Training', description: 'Training the machine learning model' },
    { key: 'evaluating', title: 'Model Evaluation', description: 'Evaluating model performance' },
    { key: 'completed', title: 'Completed', description: 'Training process complete' },
  ];

  // Convert stage to step number
  const getStepIndex = () => {
    const stageIndex = stages.findIndex(s => s.key === stage);
    if (stageIndex === -1) return 0;
    return stageIndex;
  };

  // Calculate current step
  const currentStep = stage === 'failed' ? -1 : getStepIndex();

  // Get status for each step
  const getStepStatus = (index) => {
    if (stage === 'failed') {
      return index <= getStepIndex() ? 'error' : 'wait';
    }
    
    if (index < currentStep) return 'finish';
    if (index === currentStep) return 'process';
    return 'wait';
  };

  // Define icons for the steps
  const icons = {
    wait: <ClockCircleOutlined />,
    process: <LoadingOutlined />,
    finish: <CheckCircleOutlined />,
    error: <CloseCircleOutlined />,
  };

  return (
    <Card>
      <Title level={4}>Model Training Progress</Title>
      
      {stage === 'idle' ? (
        <div style={{ textAlign: 'center', padding: '20px' }}>
          <Text>No active training process</Text>
        </div>
      ) : (
        <>
          <Steps
            current={currentStep}
            status={stage === 'failed' ? 'error' : 'process'}
            style={{ marginBottom: '20px' }}
          >
            {stages.map((s, index) => (
              <Steps.Step 
                key={s.key} 
                title={s.title} 
                description={s.description}
                status={getStepStatus(index)}
                icon={icons[getStepStatus(index)]}
              />
            ))}
          </Steps>

          {stage === 'failed' ? (
            <Alert 
              message="Training Failed" 
              description={error || message || "An unknown error occurred during training."} 
              type="error" 
              showIcon 
            />
          ) : (
            <>
              <Progress 
                percent={progress} 
                status={stage === 'completed' ? 'success' : 'active'}
                style={{ marginBottom: '15px' }}
              />
              
              <div style={{ textAlign: 'center' }}>
                {stage !== 'completed' ? (
                  <Spin spinning={true} indicator={<LoadingOutlined style={{ fontSize: 24 }} spin />} />
                ) : (
                  <CheckCircleOutlined style={{ fontSize: 24, color: '#52c41a' }} />
                )}
                <Text style={{ marginLeft: 10, fontSize: '16px' }}>
                  {message || `Current stage: ${stages.find(s => s.key === stage)?.title || stage}`}
                </Text>
              </div>
            </>
          )}
        </>
      )}
    </Card>
  );
};

export default ModelTrainingProgress; 