import React, { useState, useEffect, useRef } from "react";
import {
    Space,
    Card,
    Row,
    Col,
    Statistic,
    Spin,
    Select,
    Modal,
    Button,
    Tooltip,
} from "antd";
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Legend,
    ResponsiveContainer,
} from "recharts";
import {
    fetchMetrics as fetchMetricPoint,
    fetchDetailedMetrics,
    fetchAllDetailedMetrics,
} from "../services/metricsService";
import { anomalyService } from "../services/anomalyService";
import { message } from "antd";

const { Option } = Select;

const MetricsChart = ({ data, title, dataKey, suffix = "" }) => (
    <div style={{ width: "100%", height: 200 }}>
        <ResponsiveContainer>
            <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="timestamp" />
                <YAxis />
                <Tooltip formatter={(value) => `${value}${suffix}`} />
                <Legend />
                <Line
                    type="monotone"
                    dataKey="value"
                    name={title}
                    stroke="#8884d8"
                    dot={false}
                />
            </LineChart>
        </ResponsiveContainer>
    </div>
);

const formatValue = (category, metric, value = 0) => {
    if (typeof value !== "number") return "0";

    // Handle time values (convert from ns)
    if (
        typeof metric === "string"
            ? metric.includes("time")
            : metric.name.includes("time")
    ) {
        return (value / 1e9).toFixed(2);
    }

    // Handle delay values (convert from µs to ms)
    if (
        typeof metric === "string"
            ? metric.includes("delay")
            : metric.name.includes("delay")
    ) {
        if (value < 1000) return value.toFixed(0); // µs
        if (value < 1000 * 1000) return (value / 1000).toFixed(2); // ms
        return (value / (1000 * 1000)).toFixed(2); // s
    }

    // Handle memory values (convert to MB)
    if (
        category === "memory" &&
        (typeof metric === "string"
            ? metric.includes("memstore") || metric.includes("memory")
            : metric.name.includes("memstore") || metric.name.includes("memory"))
    ) {
        return (value / (1024 * 1024)).toFixed(2);
    }

    // Handle network bytes (convert to B/KB/MB/GB)
    if (
        category === "network" &&
        (typeof metric === "string"
            ? metric.includes("bytes")
            : metric.name.includes("bytes"))
    ) {
        if (value < 1024) return value.toFixed(0); // Bytes
        if (value < 1024 * 1024) return (value / 1024).toFixed(2); // KB
        if (value < 1024 * 1024 * 1024) return (value / (1024 * 1024)).toFixed(2); // MB
        return (value / (1024 * 1024 * 1024)).toFixed(2); // GB
    }

    // Handle disk write size (convert to B/KB/MB/GB)
    if (
        typeof metric === "string"
            ? metric.includes("write size") || metric.includes("log total size")
            : metric.name.includes("write size") || metric.name.includes("log total size")
    ) {
        if (value < 1024) return value.toFixed(0); // Bytes
        if (value < 1024 * 1024) return (value / 1024).toFixed(2); // KB
        if (value < 1024 * 1024 * 1024) return (value / (1024 * 1024)).toFixed(2); // MB
        return (value / (1024 * 1024 * 1024)).toFixed(2); // GB
    }

    // Handle percentages
    if (
        typeof metric === "string"
            ? metric === "cpu usage" || metric.includes("util")
            : metric.name === "cpu usage" || metric.name.includes("util")
    ) {
        return value.toFixed(1);
    }

    // Handle count metrics with thousands separators
    if (
        typeof metric === "string"
            ? metric.includes("count")
            : metric.name.includes("count")
    ) {
        return Math.round(value).toString();
    }

    // Default formatting
    return value.toFixed(0);
};

const MetricsPanel = () => {
    const [metricPoint, setMetricPoint] = useState(null);
    const [loading, setLoading] = useState(true);
    const [metricSeries, setMetricSeries] = useState({});
    const [chartModal, setChartModal] = useState({
        visible: false,
        title: "",
        data: [],
        suffix: "",
        loading: false,
    });
    const [selectedMetrics, setSelectedMetrics] = useState({
        cpu: "cpu usage",
        memory: "total memstore used",
        io: "io read count",
        network: "rpc net delay",
        transactions: "trans commit count",
    });
    const [loadedSeries, setLoadedSeries] = useState({});
    const activeFetches = useRef(new Set()); // Track active fetch operations to prevent duplicates
    const fetchInterval = 5000;

    const fetchChartData = async (nodeIp, category, metric) => {
        const now = Date.now();
        const seriesKey = `${nodeIp}-${category}-${metric}`;
        const lastUpdated = loadedSeries[seriesKey];

        if (lastUpdated && now - lastUpdated < fetchInterval) {
            return;
        }

        // Check if this request is already in progress
        if (activeFetches.current.has(seriesKey)) {
            console.debug(
                `Already fetching data for ${nodeIp}/${category}/${metric}, skipping duplicate request`
            );
            return;
        }

        // Mark this request as in progress
        activeFetches.current.add(seriesKey);

        try {
            // Use the optimized endpoint that fetches only the selected metrics
            const selectedMetricsForCategory = { [category]: metric };
            const detailedData = await fetchAllDetailedMetrics(
                nodeIp,
                selectedMetricsForCategory
            );
            console.debug(
                `Fetched detailed data for ${nodeIp}/${category}/${metric}:`,
                detailedData
            );

            // Now that we have the data, update the lastUpdated timestamp
            setLoadedSeries((prev) => ({
                ...prev,
                [seriesKey]: now,
            }));

            // Use functional updates to avoid race conditions
            if (detailedData?.[category]?.[metric]) {
                setMetricSeries((prev) => {
                    // Skip update if nothing changed to prevent unnecessary re-renders
                    const currentData = prev[nodeIp]?.[category]?.[metric];
                    const newData = detailedData[category][metric];

                    // If data is the same, don't trigger a re-render
                    if (JSON.stringify(currentData) === JSON.stringify(newData)) {
                        console.debug(
                            `metricSeries data is the same with newData, metricSeries not updated`
                        );
                        return prev;
                    }

                    return {
                        ...prev,
                        [nodeIp]: {
                            ...prev?.[nodeIp],
                            [category]: {
                                ...prev?.[nodeIp]?.[category],
                                [metric]: detailedData[category][metric],
                            },
                        },
                    };
                });
            }
        } catch (error) {
            console.error(`Error fetching ${category} chart data for ${nodeIp}:`, error);
            // Only update if there was an error
            setLoadedSeries((prev) => ({
                ...prev,
                [seriesKey]: null,
            }));
        } finally {
            // Remove this request from the in-progress set
            activeFetches.current.delete(seriesKey);
        }
    };

    // New optimized function to fetch all metrics for a node in a single API call
    const fetchAllNodeMetrics = async (nodeIp) => {
        const now = Date.now();
        // Get the list of selected metrics for each category
        const categories = Object.keys(selectedMetrics);
        const metrics = { ...selectedMetrics };

        // Create a unique key for this batch fetch
        const batchKey = `batch-${nodeIp}`;

        // Check if we need to update any of the series for this node
        let needsUpdate = false;
        for (const category of categories) {
            const metric = metrics[category];
            const seriesKey = `${nodeIp}-${category}-${metric}`;
            const lastUpdated = loadedSeries[seriesKey];

            if (!lastUpdated || now - lastUpdated >= fetchInterval) {
                needsUpdate = true;
                break;
            }
        }

        if (!needsUpdate) {
            return;
        }

        // Check if this batch request is already in progress
        if (activeFetches.current.has(batchKey)) {
            console.debug(
                `Already fetching batch data for ${nodeIp}, skipping duplicate request`
            );
            return;
        }

        // Mark this batch request as in progress
        activeFetches.current.add(batchKey);

        // Also mark individual series as being fetched to prevent redundant individual requests
        for (const category of categories) {
            const metric = metrics[category];
            const seriesKey = `${nodeIp}-${category}-${metric}`;
            activeFetches.current.add(seriesKey);
        }

        try {
            // Get only the selected metrics for each category in one API call
            const allDetailedData = await fetchAllDetailedMetrics(
                nodeIp,
                selectedMetrics
            );
            console.debug(
                `Fetched selected detailed data for ${nodeIp}`,
                allDetailedData
            );

            // Update timestamps for all series
            const updatedLoadedSeries = { ...loadedSeries };

            // Update metricSeries using a single state update to minimize re-renders
            setMetricSeries((prev) => {
                let newMetricSeries = { ...prev };
                let hasChanges = false;

                // Initialize node data if not exists
                if (!newMetricSeries[nodeIp]) {
                    newMetricSeries[nodeIp] = {};
                    hasChanges = true;
                }

                // Process each category
                for (const category of categories) {
                    const metric = metrics[category];
                    const seriesKey = `${nodeIp}-${category}-${metric}`;

                    // Only process if we have data for this category and metric
                    if (allDetailedData?.[category]?.[metric]) {
                        // Initialize category if not exists
                        if (!newMetricSeries[nodeIp][category]) {
                            newMetricSeries[nodeIp][category] = {};
                        }

                        const currentData = newMetricSeries[nodeIp][category][metric];
                        const newData = allDetailedData[category][metric];

                        // Skip if data is the same
                        if (JSON.stringify(currentData) !== JSON.stringify(newData)) {
                            newMetricSeries[nodeIp][category][metric] = newData;
                            hasChanges = true;
                        }

                        // Update the timestamp
                        updatedLoadedSeries[seriesKey] = now;
                    }
                }

                // Only return new object if there are changes
                return hasChanges ? newMetricSeries : prev;
            });

            // Update all series timestamps in one go
            setLoadedSeries(updatedLoadedSeries);
        } catch (error) {
            console.error(`Error fetching all detailed metrics for ${nodeIp}:`, error);
        } finally {
            // Remove all tracking for this batch
            activeFetches.current.delete(batchKey);

            // Also remove tracking for individual series
            for (const category of categories) {
                const metric = metrics[category];
                const seriesKey = `${nodeIp}-${category}-${metric}`;
                activeFetches.current.delete(seriesKey);
            }
        }
    };

    useEffect(() => {
        const fetchData = async () => {
            try {
                const data = await fetchMetricPoint();
                console.log("Fetched metricPoint data:", data);

                // Validate data structure
                if (!data || !data.metrics || typeof data.metrics !== "object") {
                    console.error("Invalid data format received:", data);
                    setMetricPoint({ metrics: {}, timestamp: Date.now() });
                    setLoading(false);
                    return;
                }

                setMetricPoint(data);

                // Validate node metrics data
                const nodeIps = Object.keys(data.metrics);
                console.debug(`Found ${nodeIps.length} nodes:`, nodeIps);

                if (nodeIps.length === 0) {
                    console.warn("No nodes found in metrics data");
                    setLoading(false);
                    return;
                }

                // Initialize categories for each node if missing
                const categories = ["cpu", "memory", "io", "network", "transactions"];

                // Use the new optimized approach to fetch all detailed metrics
                nodeIps.forEach((nodeIp) => {
                    // Validate node data
                    if (
                        !data.metrics[nodeIp] ||
                        typeof data.metrics[nodeIp] !== "object"
                    ) {
                        console.warn(
                            `Invalid node data for ${nodeIp}:`,
                            data.metrics[nodeIp]
                        );
                        data.metrics[nodeIp] = {
                            cpu: {},
                            memory: {},
                            io: {},
                            network: {},
                            transactions: {},
                        };
                    }

                    // Ensure all categories exist
                    categories.forEach((category) => {
                        if (!data.metrics[nodeIp][category]) {
                            console.warn(`Missing ${category} category for ${nodeIp}`);
                            data.metrics[nodeIp][category] = {};
                        }
                    });

                    // Fetch all detailed metrics for this node in a single API call
                    fetchAllNodeMetrics(nodeIp);
                });
            } catch (error) {
                console.error("Error fetching metrics:", error);
                setMetricPoint({ metrics: {}, timestamp: Date.now() });
            } finally {
                setLoading(false);
            }
        };

        fetchData();
        const interval = setInterval(fetchData, fetchInterval);
        return () => clearInterval(interval);
    }, [selectedMetrics]);

    const handleMetricClick = (nodeIp, category, metric) => {
        // Update the selected metrics
        setSelectedMetrics((prev) => {
            const newSelectedMetrics = {
                ...prev,
                [category]: metric,
            };

            // Use the batch fetch approach to get all selected metrics at once
            // But first update the state, so the nodeIp will have the latest selected metrics
            setTimeout(() => {
                fetchAllNodeMetrics(nodeIp);
            }, 0);

            return newSelectedMetrics;
        });
    };

    const formatChartValue = (category, metric, value = 0) => {
        if (typeof value !== "number") return "0";

        // Handle time values (convert from ns)
        if (metric.includes("time")) {
            return (value / 1e9).toFixed(2);
        }

        // Handle delay values (convert from µs to ms)
        if (metric.includes("delay")) {
            if (value < 1000) return value.toFixed(0); // µs
            if (value < 1000 * 1000) return (value / 1000).toFixed(2); // ms
            return (value / (1000 * 1000)).toFixed(2); // s
        }

        // Handle memory values (convert to MB)
        if (
            category === "memory" &&
            (metric.includes("memstore") || metric.includes("memory"))
        ) {
            return (value / (1024 * 1024)).toLocaleString(undefined, {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2,
            });
        }

        // Handle network bytes (convert to B/KB/MB/GB)
        if (category === "network" && metric.includes("bytes")) {
            if (value < 1024) return value.toFixed(0); // Bytes
            if (value < 1024 * 1024) return (value / 1024).toFixed(2); // KB
            if (value < 1024 * 1024 * 1024) return (value / (1024 * 1024)).toFixed(2); // MB
            return (value / (1024 * 1024 * 1024)).toFixed(2); // GB
        }

        // Handle disk write size (convert to B/KB/MB/GB)
        if (metric.includes("write size") || metric.includes("log total size")) {
            if (value < 1024) return value.toFixed(0); // Bytes
            if (value < 1024 * 1024) return (value / 1024).toFixed(2); // KB
            if (value < 1024 * 1024 * 1024) return (value / (1024 * 1024)).toFixed(2); // MB
            return (value / (1024 * 1024 * 1024)).toFixed(2); // GB
        }

        // Handle percentages
        if (metric === "cpu usage" || metric.includes("util")) {
            return parseFloat(value).toFixed(1);
        }

        // Handle count metrics with thousands separators
        if (metric.includes("count")) {
            return Math.round(value).toLocaleString();
        }

        // Default formatting with thousands separators
        return value.toLocaleString(undefined, {
            maximumFractionDigits: 0,
            useGrouping: true,
        });
    };

    const getMetricSuffix = (category, metric, value = 0) => {
        const metricName = typeof metric === "string" ? metric : metric.name;
        let metricValue = 0;
        if (value) {
            metricValue = value;
        } else {
            metricValue = typeof metric === "object" ? metric.value : 0;
        }

        // Handle network bytes suffix
        if (category === "network" && metricName.includes("bytes")) {
            if (metricValue < 1024) return "B";
            if (metricValue < 1024 * 1024) return "KB";
            if (metricValue < 1024 * 1024 * 1024) return "MB";
            return "GB";
        }

        // Handle disk write size suffix
        if (metricName.includes("write size") || metricName.includes("log total size")) {
            if (metricValue < 1024) return "B";
            if (metricValue < 1024 * 1024) return "KB";
            if (metricValue < 1024 * 1024 * 1024) return "MB";
            return "GB";
        }

        if (category === "memory") return "MB";
        if (metricName.includes("time") && category === "cpu") return "s";
        if (metricName.includes("delay")) {
            // Fixed delay suffix logic
            if (metricValue < 1000) return "µs";
            if (metricValue < 1000 * 1000) return "ms";
            return "s";
        }
        if (metricName === "cpu usage" || metricName.includes("util")) return "%";
        if (metricName.includes("count")) return "";
        return "";
    };

    const renderMetricItem = (nodeIp, category, metricName) => {
        // Get both data sources
        const pointData = metricPoint?.metrics?.[nodeIp]?.[category]?.[metricName];
        const seriesData = metricSeries[nodeIp]?.[category]?.[metricName];

        // Extract values and fluctuation data
        let value = 0;
        let hasFluctuation = false;
        let pct_change = 0;
        let z_score = 0;
        let displayTimestamp = "";

        // Get the latest value from either source
        const seriesValue = seriesData?.[seriesData?.length - 1]?.value;
        const pointValue =
            pointData?.value ||
            (Array.isArray(pointData) ? pointData?.[pointData?.length - 1]?.value : 0);

        // Use series value if it exists, otherwise fall back to point data
        value = seriesValue !== undefined ? seriesValue : pointValue;

        // Always use fluctuation data from metricPoint
        if (pointData) {
            const pointEntry = Array.isArray(pointData)
                ? pointData[pointData.length - 1]
                : pointData;
            hasFluctuation = pointEntry?.has_fluctuation || false;
            pct_change = pointEntry?.pct_change || 0;
            z_score = pointEntry?.z_score || 0;
            displayTimestamp = pointEntry?.timestamp || "";
        }

        // Check if the value is valid
        if (value === undefined || value === null || isNaN(value)) {
            console.warn(
                `Invalid value for metric ${nodeIp}/${category}/${metricName}:`,
                value
            );
            value = 0;
        }

        if (!pointData && !seriesData) {
            // Create a placeholder for missing metrics with zero values
            console.warn(`Missing metric data for ${nodeIp}/${category}/${metricName}`);
            return (
                <Col span={8} key={metricName}>
                    <div
                        onClick={() => handleMetricClick(nodeIp, category, metricName)}
                        style={{
                            cursor: "pointer",
                            padding: "8px",
                            borderRadius: "4px",
                            backgroundColor: "transparent",
                            borderLeft: "1px solid transparent",
                            height: "100%",
                            display: "flex",
                            flexDirection: "column",
                            justifyContent: "space-between",
                            opacity: 0.7,
                        }}
                    >
                        <div
                            style={{
                                display: "flex",
                                alignItems: "flex-start",
                                justifyContent: "space-between",
                                marginBottom: "4px",
                            }}
                        >
                            <span
                                style={{
                                    fontSize: "16px",
                                    fontWeight: "500",
                                    color: "rgba(0, 0, 0, 0.5)",
                                }}
                            >
                                0 {getMetricSuffix(category, metricName)}
                            </span>
                        </div>

                        <div
                            style={{
                                fontSize: "13px",
                                color: "rgba(0, 0, 0, 0.5)",
                                lineHeight: "1.2",
                                height: "31px",
                            }}
                        >
                            {metricName
                                .replace(/_/g, " ")
                                .replace(/\b\w/g, (l) => l.toUpperCase())}
                        </div>
                    </div>
                </Col>
            );
        }

        const formattedValue = formatValue(category, metricName, value);
        const suffix = getMetricSuffix(category, metricName, value);
        const isSelected = selectedMetrics[category] === metricName;

        // Clean up metric name for display
        const displayName = metricName
            .replace(/_/g, " ")
            .replace(/\b\w/g, (l) => l.toUpperCase())
            .replace(/\bCpu\b/g, "CPU")
            .replace(/\bIo\b/g, "IO")
            .replace(/\bRpc\b/g, "RPC")
            .replace(/\bSql\b/g, "SQL")
            .replace(/\bMysql\b/g, "MySQL")
            .replace(/\bIops\b/g, "IOPS")
            .replace(/\bDb\b/g, "DB")
            .replace(/\bApi\b/g, "API")
            .replace(/\bUuid\b/g, "UUID")
            .replace(/\bTps\b/g, "TPS")
            .replace(/\bQps\b/g, "QPS")
            .replace(/\bIp\b/g, "IP")
            .replace(/\bTcp\b/g, "TCP")
            .replace(/\bUdp\b/g, "UDP")
            .replace(/\bHttp\b/g, "HTTP")
            .replace(/\bHttps\b/g, "HTTPS")
            .replace(/\bDns\b/g, "DNS")
            .replace(/\bOs\b/g, "OS");

        return (
            <Col span={8} key={metricName}>
                <div
                    onClick={() => handleMetricClick(nodeIp, category, metricName)}
                    style={{
                        cursor: "pointer",
                        padding: "8px",
                        borderRadius: "4px",
                        backgroundColor: hasFluctuation
                            ? "#fffbe6"
                            : isSelected
                            ? "#e6f7ff"
                            : "transparent",
                        borderLeft: hasFluctuation
                            ? "3px solid #ffd666"
                            : isSelected
                            ? "1px solid #91d5ff"
                            : "1px solid transparent",
                        boxShadow: hasFluctuation
                            ? "0 2px 8px rgba(255, 214, 102, 0.1)"
                            : "none",
                        transition: "all 0.3s ease",
                        height: "100%",
                        display: "flex",
                        flexDirection: "column",
                        justifyContent: "space-between",
                    }}
                    onMouseEnter={(e) => {
                        if (!hasFluctuation) {
                            e.currentTarget.style.backgroundColor = isSelected
                                ? "#e6f7ff"
                                : "#f5f5f5";
                        }
                    }}
                    onMouseLeave={(e) => {
                        if (!hasFluctuation) {
                            e.currentTarget.style.backgroundColor = isSelected
                                ? "#e6f7ff"
                                : "transparent";
                        }
                    }}
                >
                    <div
                        style={{
                            display: "flex",
                            alignItems: "flex-start",
                            justifyContent: "space-between",
                            marginBottom: "4px",
                        }}
                    >
                        <span
                            style={{
                                fontSize: "16px",
                                fontWeight: "500",
                                color: isSelected ? "#1890ff" : "rgba(0, 0, 0, 0.8)",
                            }}
                        >
                            {formattedValue} {suffix}
                        </span>
                        {hasFluctuation && (
                            <div style={getBadgeStyle({ pct_change })}>
                                {getFluctuationText({ pct_change, z_score })}
                            </div>
                        )}
                    </div>

                    <Tooltip title={displayName} placement="bottomLeft">
                        <div
                            style={{
                                fontSize: "13px",
                                color: "rgba(0, 0, 0, 0.6)",
                                lineHeight: "1.2",
                                textOverflow: "ellipsis",
                                overflow: "hidden",
                                display: "-webkit-box",
                                WebkitBoxOrient: "vertical",
                                WebkitLineClamp: 2,
                                height: "31px",
                                wordBreak: "break-word",
                            }}
                        >
                            {displayName}
                        </div>
                    </Tooltip>
                </div>
            </Col>
        );
    };

    const renderTimeSeriesChart = (nodeIp, category, metric) => {
        // Get the time series data from metricSeries
        let data = metricSeries[nodeIp]?.[category]?.[metric];
        const seriesKey = `${nodeIp}-${category}-${metric}`;
        const lastUpdated = loadedSeries[seriesKey];
        const isInitialLoading = !lastUpdated && !data; // Only true on first load when no data exists
        const now = Date.now();
        const isExpired = lastUpdated && now - lastUpdated >= fetchInterval;

        // For normal auto-refresh, don't trigger individual fetches
        // The main useEffect with fetchAllNodeMetrics will handle periodic updates
        // Only fetch individually for:
        // 1. Initial loading when no data exists yet
        // 2. When data has been expired for much longer than the fetchInterval (indicating the automatic refresh might have failed)
        const manualFetchNeeded =
            !data || (isExpired && now - lastUpdated >= fetchInterval * 2);

        if (manualFetchNeeded && lastUpdated !== now) {
            // Prevent duplicate fetches during the same render cycle
            const timeSinceLastUpdate = now - lastUpdated || fetchInterval * 3; // Use a fallback if lastUpdated is null
            if (timeSinceLastUpdate > 100) {
                // Add a small buffer to prevent edge case rapid fetches
                console.debug(
                    `Triggering manual fetch for ${nodeIp}/${category}/${metric} (lastUpdated: ${
                        lastUpdated ? new Date(lastUpdated).toISOString() : "never"
                    })`
                );
                fetchChartData(nodeIp, category, metric);
            }
        }

        // Only show loading spinner on initial load, not on refreshes
        if (isInitialLoading) {
            console.debug(
                `Chart ${nodeIp}/${category}/${metric} is loading initially...`
            );
            return (
                <div
                    style={{
                        textAlign: "center",
                        padding: "20px",
                        height: "200px",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                    }}
                >
                    <Spin size="large" />
                </div>
            );
        }

        // Check if data exists and has the expected format
        console.debug(
            `Rendering time series chart ${nodeIp}/${category}/${metric}:`,
            data
        );
        if (!data) {
            console.warn(`No data for chart ${nodeIp}/${category}/${metric}`);
            return (
                <div style={{ textAlign: "center", padding: "20px", height: "200px" }}>
                    No data available
                </div>
            );
        }

        // Ensure data is an array
        let chartData = Array.isArray(data) ? [...data] : [];

        // Merge metricPoint data if it's newer
        const pointData = metricPoint?.metrics?.[nodeIp]?.[category]?.[metric];
        if (pointData) {
            const pointEntry = Array.isArray(pointData)
                ? pointData[pointData.length - 1]
                : pointData;
            const existsInChart = chartData.some(
                (d) => d.timestamp === pointEntry.timestamp
            );

            if (!existsInChart) {
                chartData = [
                    ...chartData,
                    {
                        timestamp: pointEntry.timestamp,
                        value: pointEntry.value,
                    },
                ].sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
            }
        }

        if (chartData.length === 0) {
            console.warn(`Empty data array for chart ${nodeIp}/${category}/${metric}`);
            return (
                <div style={{ textAlign: "center", padding: "20px", height: "200px" }}>
                    No data available
                </div>
            );
        }

        const suffix = getMetricSuffix(category, metric);

        // Don't include lastUpdated in the key to prevent complete re-renders when data refreshes
        // Instead, use a stable key based on the chart's identity
        const chartKey = `${nodeIp}-${category}-${metric}`;

        return (
            <div
                style={{
                    width: "100%",
                    height: 200,
                    marginBottom: 16,
                    position: "relative",
                }}
                key={chartKey}
            >
                {isExpired && (
                    <div style={{ position: "absolute", right: 10, top: 10, zIndex: 10 }}>
                        <Spin size="small" />
                    </div>
                )}
                <ResponsiveContainer>
                    <LineChart
                        data={chartData}
                        margin={{ left: 5, right: 30, top: 10, bottom: 25 }}
                        animationDuration={300}
                        animationBegin={0}
                    >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis
                            dataKey="timestamp"
                            tickFormatter={(ts) => new Date(ts).toLocaleTimeString()}
                        />
                        <YAxis
                            width={85}
                            tickFormatter={(value) =>
                                formatValue(category, metric, value)
                            }
                            domain={[0, "auto"]}
                            label={{
                                value: `(${suffix})`,
                                position: "bottom",
                                offset: 15,
                                style: {
                                    textAnchor: "middle",
                                    fontSize: 12,
                                    fill: "#666",
                                },
                            }}
                            tick={{
                                fontSize: 12,
                                fill: "#666",
                            }}
                        />
                        <Tooltip
                            formatter={(value) => [
                                `${formatValue(category, metric, value)} ${suffix}`,
                                metric,
                            ]}
                            labelFormatter={(ts) => new Date(ts).toLocaleString()}
                        />
                        <Line
                            type="monotone"
                            dataKey="value"
                            name={metric}
                            stroke="#8884d8"
                            dot={false}
                        />
                    </LineChart>
                </ResponsiveContainer>
            </div>
        );
    };

    const renderMetricsCard = (title, metrics = {}, nodeIp, category) => {
        console.debug(`Rendering metrics card ${title} for ${nodeIp}/${category}`);
        const selectedMetric = selectedMetrics[category];

        // Safety check for metrics format
        let metricsObject = metrics;
        if (!metrics || typeof metrics !== "object") {
            console.warn(`Invalid metrics object for ${title}:`, metrics);
            metricsObject = {};
        }

        // Get all metrics keys or use an empty array if none
        const metricKeys = Object.keys(metricsObject);

        // If no metrics, show a message
        if (metricKeys.length === 0) {
            return (
                <Card title={title} size="small">
                    <div style={{ textAlign: "center", padding: "20px" }}>
                        No metrics available
                    </div>
                </Card>
            );
        }

        // Create a memoization key to avoid re-rendering when nothing has changed
        const cardKey = `${nodeIp}-${category}-${selectedMetric}`;

        return (
            <Card title={title} size="small" key={cardKey}>
                {renderTimeSeriesChart(nodeIp, category, selectedMetric)}
                <Row gutter={[16, 16]}>
                    {metricKeys.map((metric) =>
                        renderMetricItem(nodeIp, category, metric)
                    )}
                </Row>
            </Card>
        );
    };

    const renderNodeMetrics = (nodeIp, nodeData = {}) => {
        console.debug(`Rendering node metrics for ${nodeIp}`);

        // Handle potential data structure issues - ensure we have proper category objects
        const cpu = nodeData.cpu && typeof nodeData.cpu === "object" ? nodeData.cpu : {};
        const memory =
            nodeData.memory && typeof nodeData.memory === "object" ? nodeData.memory : {};
        const io = nodeData.io && typeof nodeData.io === "object" ? nodeData.io : {};
        const network =
            nodeData.network && typeof nodeData.network === "object"
                ? nodeData.network
                : {};
        const transactions =
            nodeData.transactions && typeof nodeData.transactions === "object"
                ? nodeData.transactions
                : {};

        // Use the timestamp to help prevent unnecessary re-renders
        const nodeKey = `node-${nodeIp}-${metricPoint?.timestamp || Date.now()}`;

        return (
            <Card title={`Node: ${nodeIp}`} key={nodeKey} style={{ marginBottom: 16 }}>
                <Row gutter={[16, 16]}>
                    {Object.keys(cpu).length > 0 && (
                        <Col span={12}>
                            {renderMetricsCard("CPU Usage", cpu, nodeIp, "cpu")}
                        </Col>
                    )}

                    {Object.keys(memory).length > 0 && (
                        <Col span={12}>
                            {renderMetricsCard("Memory Usage", memory, nodeIp, "memory")}
                        </Col>
                    )}

                    {Object.keys(io).length > 0 && (
                        <Col span={12}>
                            {renderMetricsCard("Disk I/O", io, nodeIp, "io")}
                        </Col>
                    )}

                    {Object.keys(network).length > 0 && (
                        <Col span={12}>
                            {renderMetricsCard("Network", network, nodeIp, "network")}
                        </Col>
                    )}

                    {Object.keys(transactions).length > 0 && (
                        <Col span={12}>
                            {renderMetricsCard(
                                "Transactions",
                                transactions,
                                nodeIp,
                                "transactions"
                            )}
                        </Col>
                    )}
                </Row>
            </Card>
        );
    };

    const getFluctuationStyle = (metric) => ({
        backgroundColor: metric.has_fluctuation ? "#fffbe6" : "inherit",
        borderLeft: metric.has_fluctuation ? "3px solid #ffd666" : "none",
        boxShadow: metric.has_fluctuation ? "0 2px 8px rgba(255, 214, 102, 0.1)" : "none",
        padding: "12px",
        marginBottom: "8px",
        borderRadius: "4px",
        transition: "all 0.3s ease",
    });

    const getBadgeStyle = (metric) => ({
        color: metric.pct_change > 0 ? "#389e0d" : "#cf1322",
        marginLeft: 8,
        fontWeight: 600,
        fontSize: "0.9em",
        padding: "2px 6px",
        borderRadius: "3px",
        backgroundColor:
            metric.pct_change > 0 ? "rgba(56, 158, 13, 0.1)" : "rgba(207, 19, 34, 0.1)",
    });

    const getFluctuationText = (metric) => {
        // For constant rate changes in large values, show the absolute change
        if (
            metric.pct_change === 0 &&
            metric.value > 1e6 &&
            Math.abs(metric.z_score) < 2.0
        ) {
            // Don't show any fluctuation text for monotonically increasing counters with normal z-scores
            return "";
        }

        // If pct_change is very small but z-score is significant, display as <1% instead of 0%
        let pct = Math.abs(Math.round(metric.pct_change * 100));
        if (pct === 0 && Math.abs(metric.z_score) >= 2.0) {
            pct = "<1";
        }
        const direction = metric.pct_change > 0 ? "↑" : "↓";
        const z = metric.z_score.toFixed(1);
        return `${direction}${pct}% (z=${z})`;
    };

    if (loading) {
        return (
            <div style={{ textAlign: "center", padding: "50px" }}>
                <Spin size="large" />
            </div>
        );
    }

    return (
        <Space direction="vertical" style={{ width: "100%" }}>
            <Card title="System Metrics" style={{ marginBottom: 16 }}>
                {loading ? (
                    <div style={{ textAlign: "center", padding: "20px" }}>
                        <Spin size="large" />
                    </div>
                ) : (
                    <>
                        <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
                            <Col>
                                Last Updated:{" "}
                                {metricPoint?.timestamp
                                    ? new Date(metricPoint.timestamp).toLocaleString()
                                    : "N/A"}
                            </Col>
                        </Row>
                        {metricPoint?.metrics &&
                            Object.entries(metricPoint.metrics).map(
                                ([nodeIp, nodeData]) =>
                                    renderNodeMetrics(nodeIp, nodeData)
                            )}
                    </>
                )}
            </Card>
        </Space>
    );
};

export default MetricsPanel;
