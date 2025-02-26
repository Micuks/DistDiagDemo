import React, { useRef, useState, useEffect } from 'react';
import { Space, Typography, Card, Modal, Button } from 'antd';
import WorkloadControlPanel from './WorkloadControlPanel';
import AnomalyControlPanel from './AnomalyControlPanel';

const { Title, Text, Paragraph } = Typography;

const ControlPanel = () => {
    const workloadRef = useRef(null);
    const anomalyRef = useRef(null);
    const [isFirstVisit, setIsFirstVisit] = useState(true);

    const scrollToComponent = (ref) => {
        if (ref.current) {
            ref.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
            ref.current.style.backgroundColor = '#e6f7ff';
            setTimeout(() => {
                ref.current.style.backgroundColor = 'transparent';
            }, 2000);
        }
        setIsFirstVisit(false);
    };

    return (
        <Space direction="vertical" style={{ width: '100%' }}>
            <Modal
                title="🚀 Getting Started Guide"
                open={isFirstVisit}
                onOk={() => setIsFirstVisit(false)}
                onCancel={() => setIsFirstVisit(false)}
                width={800}
                footer={[
                    <Button key="ok" type="primary" onClick={() => setIsFirstVisit(false)}>
                        Got it!
                    </Button>
                ]}
            >
                <Space direction="vertical" style={{ width: '100%' }}>
                    <Paragraph
                        onClick={() => scrollToComponent(workloadRef)}
                        style={{ 
                            cursor: 'pointer',
                            padding: '8px 12px',
                            borderRadius: 4,
                            transition: 'all 0.3s ease',
                            '&:hover': {
                                backgroundColor: '#f5f5f5'
                            }
                        }}
                    >
                        <Text strong>Step 1: Configure Workload</Text>
                        <br />
                        <Text>Select and start a database workload pattern to simulate real traffic.</Text>
                        <br />
                        <Text type="secondary" style={{ fontSize: '12px' }}>👆 Click to jump to workload panel</Text>
                    </Paragraph>
                    <Paragraph
                        onClick={() => scrollToComponent(anomalyRef)}
                        style={{ 
                            cursor: 'pointer',
                            padding: '8px 12px',
                            borderRadius: 4,
                            transition: 'all 0.3s ease',
                            '&:hover': {
                                backgroundColor: '#f5f5f5'
                            }
                        }}
                    >
                        <Text strong>Step 2: Inject Anomalies</Text>
                        <br />
                        <Text>Choose and activate anomaly scenarios to test system behavior.</Text>
                        <br />
                        <Text type="secondary" style={{ fontSize: '12px' }}>👆 Click to jump to anomaly panel</Text>
                    </Paragraph>
                    <Paragraph>
                        <Text strong>Step 3: Monitor Metrics</Text>
                        <br />
                        <Text>Observe system metrics changes in the dashboard below.</Text>
                    </Paragraph>
                    <Paragraph>
                        <Text strong>Step 4: Review Diagnosis</Text>
                        <br />
                        <Text>Check the anomaly detection results and rankings.</Text>
                    </Paragraph>
                    <Paragraph type="secondary" style={{ marginTop: 8 }}>
                        💡 Tip: Always start the workload before injecting anomalies for accurate results.
                    </Paragraph>
                </Space>
            </Modal>

            <div ref={workloadRef} style={{ transition: 'background-color 0.3s ease' }}>
                <WorkloadControlPanel />
            </div>
            <div ref={anomalyRef} style={{ transition: 'background-color 0.3s ease' }}>
                <AnomalyControlPanel />
            </div>
        </Space>
    );
};

export default ControlPanel; 