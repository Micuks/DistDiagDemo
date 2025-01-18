import React from 'react';
import { Space } from 'antd';
import AnomalyControlPanel from './AnomalyControlPanel';
import WorkloadControlPanel from './WorkloadControlPanel';

const ControlPanel = () => {
    return (
        <Space direction="vertical" style={{ width: '100%' }}>
            <AnomalyControlPanel />
            <WorkloadControlPanel />
        </Space>
    );
};

export default ControlPanel; 