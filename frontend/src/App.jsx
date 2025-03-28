import React, { useState, useEffect } from "react";
import {
  Layout,
  Menu,
  Steps,
  Modal,
  Button,
  Typography,
  Tooltip,
  Drawer,
  Badge,
} from "antd";
import {
  DashboardOutlined,
  ExperimentOutlined,
  LineChartOutlined,
  ApiOutlined,
  ControlOutlined,
  QuestionCircleOutlined,
  RocketOutlined,
  ArrowRightOutlined,
  SettingOutlined,
} from "@ant-design/icons";
import {
  BrowserRouter as Router,
  useNavigate,
  useLocation,
} from "react-router-dom";
import AppRoutes from "./routes";

const { Header, Content, Sider } = Layout;
const { Text, Title } = Typography;

const Navigation = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [workflowModalVisible, setWorkflowModalVisible] = useState(false);
  const [siderCollapsed, setSiderCollapsed] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [isExecutionDashboard, setIsExecutionDashboard] = useState(false);

  // Check if it's the user's first visit
  useEffect(() => {
    const firstVisit = localStorage.getItem("firstVisit") !== "false";
    if (firstVisit) {
      setWorkflowModalVisible(true);
      localStorage.setItem("firstVisit", "false");
    }

    // Determine current step based on path
    const pathToStepMap = {
      "/control": 0,
      "/metrics": 1,
      "/ranks": 2,
    };

    if (pathToStepMap[location.pathname] >= 0) {
      setCurrentStep(pathToStepMap[location.pathname]);
    }
    
    // Check if we're on execution dashboard by checking URL and localStorage flag
    const checkExecutionDashboard = () => {
      const onExecutionDashboard = location.pathname.includes("/control") && localStorage.getItem("onExecutionDashboard") === "true";
      setIsExecutionDashboard(onExecutionDashboard);
    };
    
    // Initial check
    checkExecutionDashboard();
    
    // Set up a short timeout to recheck (in case localStorage update happens shortly after navigation)
    const timeoutId = setTimeout(checkExecutionDashboard, 100);
    
    return () => {
      clearTimeout(timeoutId);
    };
  }, [location.pathname]);

  // Add a separate useEffect to listen for localStorage changes
  useEffect(() => {
    const handleStorageChange = () => {
      if (location.pathname.includes("/control")) {
        const isOnExecutionDashboard = localStorage.getItem("onExecutionDashboard") === "true";
        setIsExecutionDashboard(isOnExecutionDashboard);
      }
    };

    window.addEventListener('storage', handleStorageChange);
    
    // Check immediately in case localStorage was already set
    handleStorageChange();
    
    return () => {
      window.removeEventListener('storage', handleStorageChange);
    };
  }, [location.pathname]);

  const workflowSteps = [
    {
      title: "Setup",
      description: "Configure workload & anomalies",
      path: "/control",
      icon: <ControlOutlined />,
    },
    {
      title: "Monitor",
      description: "View system metrics",
      path: "/metrics",
      icon: <LineChartOutlined />,
    },
    {
      title: "Analyze",
      description: "Review RCA results",
      path: "/ranks",
      icon: <ExperimentOutlined />,
    },
  ];

  const menuItems = [
    {
      key: "/control",
      icon: <ControlOutlined style={{ color: "#fff" }} />,
      label: (
        <Badge dot={currentStep === 0} offset={[5, 0]}>
          <Tooltip title="Step 1: Configure workloads and anomalies">
            <span style={{ color: "#fff" }}>Control Panel</span>
          </Tooltip>
        </Badge>
      ),
    },
    {
      key: "/metrics",
      icon: <LineChartOutlined style={{ color: "#fff" }} />,
      label: (
        <Badge dot={currentStep === 1} offset={[5, 0]}>
          <Tooltip title="Step 2: View database system metrics">
            <span style={{ color: "#fff" }}>System Metrics</span>
          </Tooltip>
        </Badge>
      ),
    },
    {
      key: "/ranks",
      icon: <ExperimentOutlined style={{ color: "#fff" }} />,
      label: (
        <Badge dot={currentStep === 2} offset={[5, 0]}>
          <Tooltip title="Step 3: View root cause analysis">
            <span style={{ color: "#fff" }}>Anomaly Ranks</span>
          </Tooltip>
        </Badge>
      ),
    },
    {
      key: "/training",
      icon: <ApiOutlined style={{ color: "#fff" }} />,
      label: (
        <Tooltip title="Train models (optional)">
          <span style={{ color: "#fff" }}>Model Training</span>
        </Tooltip>
      ),
    },
    {
      key: "workflow",
      icon: <QuestionCircleOutlined style={{ color: "#fff" }} />,
      label: (
        <Tooltip title="View workflow guide">
          <span style={{ color: "#fff" }}>Workflow Guide</span>
        </Tooltip>
      ),
    },
    {
      key: "/admin",
      icon: <SettingOutlined style={{ color: "#fff" }} />,
      label: (
        <Tooltip title="Admin settings">
          <span style={{ color: "#fff" }}>Admin</span>
        </Tooltip>
      ),
    },
  ];

  const handleMenuClick = ({ key }) => {
    if (key === "workflow") {
      setWorkflowModalVisible(true);
    } else {
      navigate(key);
    }
  };

  const getNextStep = () => {
    // If we're at the last step (Analyze), don't calculate a next step
    if (currentStep === workflowSteps.length - 1) {
      return null;
    }

    const nextStepIndex = (currentStep + 1) % workflowSteps.length;
    return workflowSteps[nextStepIndex];
  };

  const getNextStepTips = () => {
    switch (currentStep) {
      case 0:
        return "Step 2: View database system metrics";
      case 1:
        return "Step 3: View root cause analysis";
      default:
        return "Next step in workflow";
    }
  };

  const nextStep = getNextStep();
  const nextStepTips = getNextStepTips(currentStep);

  return (
    <>
      <Menu
        theme="dark"
        mode="horizontal"
        selectedKeys={[location.pathname]}
        items={menuItems}
        onClick={handleMenuClick}
        style={{
          backgroundColor: "#001529",
          color: "#0f0f0f",
          overflow: "auto",
          whiteSpace: "nowrap",
          flex: 1,
          scrollbarWidth: "none",  // For Firefox
          msOverflowStyle: "none", // For IE
          "&::-webkit-scrollbar": { // For Chrome/Safari
            display: "none"
          }
        }}
      />

      <Modal
        title={
          <div style={{ display: "flex", alignItems: "center" }}>
            <RocketOutlined style={{ fontSize: "24px", marginRight: "10px" }} />
            <span>DBPecker Workflow Guide</span>
          </div>
        }
        open={workflowModalVisible}
        onCancel={() => setWorkflowModalVisible(false)}
        footer={[
          <Button key="close" onClick={() => setWorkflowModalVisible(false)}>
            Close
          </Button>,
          <Button
            key="start"
            type="primary"
            onClick={() => {
              setWorkflowModalVisible(false);
              navigate("/control");
            }}
          >
            Start Workflow
          </Button>,
        ]}
        width={800}
      >
        <Steps
          current={currentStep}
          items={workflowSteps.map((step) => ({
            title: step.title,
            description: step.description,
            icon: step.icon,
          }))}
          style={{ margin: "24px 0", overflowX: "auto" }}
        />

        <div style={{ margin: "20px 0" }}>
          <Title level={4}>How to use DBPecker:</Title>
          <ol style={{ fontSize: "16px", lineHeight: "2", margin: "16px 0" }}>
            <li>
              <strong>Control Panel:</strong> Configure your workload and
              anomaly scenarios
            </li>
            <li>
              <strong>System Metrics:</strong> View real-time system metrics to
              monitor performance
            </li>
            <li>
              <strong>Anomaly Ranks:</strong> Review root cause analysis from
              different models
            </li>
          </ol>

          <Text
            type="secondary"
            style={{ display: "block", marginTop: "16px", fontSize: "14px" }}
          >
            Follow this workflow for the best experience or click on any section
            to jump directly to it.
          </Text>
        </div>
      </Modal>

      {location.pathname !== "/dashboard" && nextStep && (
        <div
          style={{
            position: "fixed",
            bottom: "20px",
            right: "20px",
            zIndex: 1000,
            background: location.pathname.includes("/control") && !isExecutionDashboard ? "#8c8c8c" : "#1890ff",
            padding: "10px 15px",
            borderRadius: "4px",
            boxShadow: "0 2px 8px rgba(0,0,0,0.15)",
            cursor: location.pathname.includes("/control") && !isExecutionDashboard ? "not-allowed" : "pointer",
            display: "flex",
            alignItems: "center",
            color: "white",
            opacity: location.pathname.includes("/control") && !isExecutionDashboard ? 0.7 : 1,
          }}
          onClick={() => {
            if (!location.pathname.includes("/control") || isExecutionDashboard) {
              navigate(nextStep.path);
            }
          }}
        >
          <Tooltip title={location.pathname.includes("/control") && !isExecutionDashboard ? "Complete the current configuration first" : nextStepTips} style={{paddingRight: "0px"}}>
            <Text style={{ color: "white", marginRight: "10px" }}>
              Next: {nextStep.title}
            </Text>
          </Tooltip>
          <ArrowRightOutlined />
        </div>
      )}
    </>
  );
};

const App = () => {
  const navigate = useNavigate();

  return (
    <Layout style={{ minHeight: "100vh" }}>
      <Header
        style={{
          display: "flex",
          alignItems: "center",
          backgroundColor: "#001529",
          padding: "0 24px",
          height: "64px",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            color: "white",
            marginRight: "40px",
            fontSize: "18px",
            fontWeight: "bold",
            display: "flex",
            alignItems: "center",
            cursor: "pointer",
            flexShrink: 0,
          }}
          onClick={() => navigate("/dashboard")}
        >
          <DashboardOutlined style={{ marginRight: "8px", fontSize: "20px" }} />
          DBPecker
        </div>
        <div
          style={{
            flex: 1,
            overflowX: "auto",
            overflowY: "hidden",
            display: "flex",
            alignItems: "center",
          }}
        >
          <Navigation />
        </div>
      </Header>
      <Content style={{ padding: "24px", minHeight: "calc(100vh - 64px)" }}>
        <AppRoutes />
      </Content>
    </Layout>
  );
};

const AppWrapper = () => {
  return (
    <Router>
      <App />
    </Router>
  );
};

export default AppWrapper;
