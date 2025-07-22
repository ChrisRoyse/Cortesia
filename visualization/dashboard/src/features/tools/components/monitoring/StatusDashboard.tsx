import React, { useState, useEffect, useMemo } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  IconButton,
  Tooltip,
  Chip,
  LinearProgress,
  Divider,
  Button,
  Menu,
  MenuItem,
  Alert,
  Snackbar
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Settings as SettingsIcon,
  Timeline as TimelineIcon,
  Speed as SpeedIcon,
  Error as ErrorIcon,
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon,
  Block as BlockIcon,
  MoreVert as MoreVertIcon,
  Download as DownloadIcon,
  Notifications as NotificationsIcon
} from '@mui/icons-material';
import { MCPTool, ToolStatus } from '../../types';
import ToolStatusMonitor, { StatusHistory } from '../../services/ToolStatusMonitor';
import StatusIndicator from './StatusIndicator';
import HealthMatrix from './HealthMatrix';
import { useAppSelector } from '../../../../app/hooks';
import { selectAllTools } from '../../stores/toolsSlice';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip as ChartTooltip,
  Legend,
  Filler
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  ChartTooltip,
  Legend,
  Filler
);

interface StatusDashboardProps {
  tools?: MCPTool[];
  refreshInterval?: number;
  showHistory?: boolean;
  onToolSelect?: (tool: MCPTool) => void;
}

interface StatusMetrics {
  healthy: number;
  degraded: number;
  unavailable: number;
  unknown: number;
  avgResponseTime: number;
  totalErrors: number;
  uptime: number;
}

const StatusDashboard: React.FC<StatusDashboardProps> = ({
  tools: propTools,
  refreshInterval = 30000,
  showHistory = true,
  onToolSelect
}) => {
  const allTools = useAppSelector(selectAllTools);
  const tools = propTools || allTools;
  
  const [selectedTool, setSelectedTool] = useState<MCPTool | null>(null);
  const [statusHistory, setStatusHistory] = useState<Map<string, StatusHistory[]>>(new Map());
  const [metrics, setMetrics] = useState<StatusMetrics>({
    healthy: 0,
    degraded: 0,
    unavailable: 0,
    unknown: 0,
    avgResponseTime: 0,
    totalErrors: 0,
    uptime: 0
  });
  const [alerts, setAlerts] = useState<string[]>([]);
  const [showAlert, setShowAlert] = useState(false);
  const [alertMessage, setAlertMessage] = useState('');
  const [settingsAnchor, setSettingsAnchor] = useState<null | HTMLElement>(null);
  const [isRefreshing, setIsRefreshing] = useState(false);

  // Calculate metrics
  useEffect(() => {
    const calculateMetrics = () => {
      const newMetrics: StatusMetrics = {
        healthy: 0,
        degraded: 0,
        unavailable: 0,
        unknown: 0,
        avgResponseTime: 0,
        totalErrors: 0,
        uptime: 0
      };

      let totalResponseTime = 0;
      let responseCount = 0;
      let availableCount = 0;

      tools.forEach(tool => {
        // Count by status
        newMetrics[tool.status.health]++;

        // Calculate average response time
        if (tool.status.responseTime) {
          totalResponseTime += tool.status.responseTime;
          responseCount++;
        }

        // Count errors
        if (tool.metrics.errorCount) {
          newMetrics.totalErrors += tool.metrics.errorCount;
        }

        // Count available tools
        if (tool.status.available) {
          availableCount++;
        }
      });

      // Calculate averages
      if (responseCount > 0) {
        newMetrics.avgResponseTime = totalResponseTime / responseCount;
      }

      if (tools.length > 0) {
        newMetrics.uptime = (availableCount / tools.length) * 100;
      }

      setMetrics(newMetrics);
    };

    calculateMetrics();
  }, [tools]);

  // Set up monitoring
  useEffect(() => {
    const toolIds = tools.map(t => t.id);
    
    // Start monitoring
    ToolStatusMonitor.startMonitoring(toolIds, refreshInterval);

    // Set up status change listener
    const unsubscribeStatusChange = ToolStatusMonitor.onStatusChange((toolId, oldStatus, newStatus) => {
      if (newStatus === 'degraded' || newStatus === 'unavailable') {
        const tool = tools.find(t => t.id === toolId);
        if (tool) {
          const message = `${tool.name} changed from ${oldStatus} to ${newStatus}`;
          setAlerts(prev => [...prev, message]);
          setAlertMessage(message);
          setShowAlert(true);
        }
      }
    });

    // Set up alert listener
    const unsubscribeAlert = ToolStatusMonitor.onAlert((toolId, status, message) => {
      setAlerts(prev => [...prev, message]);
      setAlertMessage(message);
      setShowAlert(true);
    });

    // Load initial history
    const loadHistory = () => {
      const newHistory = new Map<string, StatusHistory[]>();
      toolIds.forEach(toolId => {
        const history = ToolStatusMonitor.getStatusHistory(toolId, 24);
        newHistory.set(toolId, history);
      });
      setStatusHistory(newHistory);
    };

    loadHistory();
    const historyInterval = setInterval(loadHistory, 60000); // Update history every minute

    return () => {
      ToolStatusMonitor.stopMonitoring(toolIds);
      unsubscribeStatusChange();
      unsubscribeAlert();
      clearInterval(historyInterval);
    };
  }, [tools, refreshInterval]);

  // Refresh all tools
  const handleRefresh = async () => {
    setIsRefreshing(true);
    const toolIds = tools.map(t => t.id);
    
    // Force immediate check for all tools
    for (const toolId of toolIds) {
      const tool = tools.find(t => t.id === toolId);
      if (tool) {
        await ToolStatusMonitor.checkToolHealth(tool);
      }
    }
    
    setIsRefreshing(false);
  };

  // Export status report
  const handleExportReport = () => {
    const report = {
      timestamp: new Date().toISOString(),
      summary: metrics,
      tools: tools.map(tool => ({
        id: tool.id,
        name: tool.name,
        category: tool.category,
        status: tool.status,
        metrics: tool.metrics
      })),
      alerts: alerts
    };

    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `status-report-${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Status icon component
  const StatusIcon: React.FC<{ status: ToolStatus }> = ({ status }) => {
    switch (status) {
      case 'healthy':
        return <CheckCircleIcon sx={{ color: 'success.main' }} />;
      case 'degraded':
        return <WarningIcon sx={{ color: 'warning.main' }} />;
      case 'unavailable':
        return <ErrorIcon sx={{ color: 'error.main' }} />;
      default:
        return <BlockIcon sx={{ color: 'text.disabled' }} />;
    }
  };

  // Chart data for response time history
  const responseTimeChartData = useMemo(() => {
    if (!selectedTool) return null;

    const history = statusHistory.get(selectedTool.id) || [];
    const labels = history.slice(-30).map(h => 
      new Date(h.timestamp).toLocaleTimeString()
    );
    const data = history.slice(-30).map(h => h.responseTime);

    return {
      labels,
      datasets: [{
        label: 'Response Time (ms)',
        data,
        fill: true,
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        borderColor: 'rgba(75, 192, 192, 1)',
        borderWidth: 2,
        tension: 0.4
      }]
    };
  }, [selectedTool, statusHistory]);

  // Chart options
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      },
      tooltip: {
        mode: 'index' as const,
        intersect: false
      }
    },
    scales: {
      x: {
        display: true,
        grid: {
          display: false
        }
      },
      y: {
        display: true,
        title: {
          display: true,
          text: 'Response Time (ms)'
        }
      }
    }
  };

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h5" fontWeight="bold">
          Live Status Monitoring
        </Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Tooltip title="Refresh All">
            <IconButton onClick={handleRefresh} disabled={isRefreshing}>
              <RefreshIcon className={isRefreshing ? 'rotating' : ''} />
            </IconButton>
          </Tooltip>
          <Tooltip title="Export Report">
            <IconButton onClick={handleExportReport}>
              <DownloadIcon />
            </IconButton>
          </Tooltip>
          <Tooltip title="Settings">
            <IconButton onClick={(e) => setSettingsAnchor(e.currentTarget)}>
              <SettingsIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* Summary Cards */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card elevation={2}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography color="text.secondary" variant="body2">
                    Healthy Tools
                  </Typography>
                  <Typography variant="h4" fontWeight="bold" color="success.main">
                    {metrics.healthy}
                  </Typography>
                </Box>
                <CheckCircleIcon sx={{ fontSize: 40, color: 'success.light', opacity: 0.3 }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card elevation={2}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography color="text.secondary" variant="body2">
                    Degraded Tools
                  </Typography>
                  <Typography variant="h4" fontWeight="bold" color="warning.main">
                    {metrics.degraded}
                  </Typography>
                </Box>
                <WarningIcon sx={{ fontSize: 40, color: 'warning.light', opacity: 0.3 }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card elevation={2}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography color="text.secondary" variant="body2">
                    Avg Response Time
                  </Typography>
                  <Typography variant="h4" fontWeight="bold">
                    {metrics.avgResponseTime.toFixed(0)}ms
                  </Typography>
                </Box>
                <SpeedIcon sx={{ fontSize: 40, color: 'info.light', opacity: 0.3 }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card elevation={2}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography color="text.secondary" variant="body2">
                    System Uptime
                  </Typography>
                  <Typography variant="h4" fontWeight="bold">
                    {metrics.uptime.toFixed(1)}%
                  </Typography>
                </Box>
                <Box sx={{ width: 40, height: 40 }}>
                  <LinearProgress
                    variant="determinate"
                    value={metrics.uptime}
                    sx={{
                      height: 40,
                      borderRadius: 1,
                      transform: 'rotate(-90deg)',
                      '& .MuiLinearProgress-bar': {
                        backgroundColor: metrics.uptime > 95 ? 'success.main' : 
                                       metrics.uptime > 90 ? 'warning.main' : 'error.main'
                      }
                    }}
                  />
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Health Matrix */}
      <Paper elevation={2} sx={{ p: 2, mb: 3 }}>
        <Typography variant="h6" sx={{ mb: 2 }}>
          Tool Health Matrix
        </Typography>
        <HealthMatrix 
          tools={tools} 
          onToolClick={(tool) => {
            setSelectedTool(tool);
            onToolSelect?.(tool);
          }}
        />
      </Paper>

      {/* Selected Tool Details */}
      {selectedTool && showHistory && (
        <Paper elevation={2} sx={{ p: 2, mb: 3 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <StatusIndicator status={selectedTool.status} size="large" showLabel />
              <Box>
                <Typography variant="h6">{selectedTool.name}</Typography>
                <Typography variant="body2" color="text.secondary">
                  {selectedTool.category} - Last checked: {new Date(selectedTool.status.lastChecked).toLocaleTimeString()}
                </Typography>
              </Box>
            </Box>
            <IconButton onClick={() => setSelectedTool(null)} size="small">
              <BlockIcon />
            </IconButton>
          </Box>

          <Divider sx={{ my: 2 }} />

          {/* Response Time Chart */}
          {responseTimeChartData && (
            <Box sx={{ height: 200 }}>
              <Line data={responseTimeChartData} options={chartOptions} />
            </Box>
          )}

          {/* Tool Metrics */}
          <Grid container spacing={2} sx={{ mt: 2 }}>
            <Grid item xs={6} sm={3}>
              <Typography variant="body2" color="text.secondary">Error Rate</Typography>
              <Typography variant="h6">
                {(selectedTool.status.errorRate * 100).toFixed(1)}%
              </Typography>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Typography variant="body2" color="text.secondary">Success Rate</Typography>
              <Typography variant="h6">
                {selectedTool.metrics.successRate.toFixed(1)}%
              </Typography>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Typography variant="body2" color="text.secondary">Total Executions</Typography>
              <Typography variant="h6">
                {selectedTool.metrics.totalExecutions}
              </Typography>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Typography variant="body2" color="text.secondary">P95 Response Time</Typography>
              <Typography variant="h6">
                {selectedTool.metrics.p95ResponseTime}ms
              </Typography>
            </Grid>
          </Grid>

          {/* Status Details */}
          {selectedTool.status.details && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                Additional Details
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                {Object.entries(selectedTool.status.details).map(([key, value]) => (
                  <Chip
                    key={key}
                    label={`${key}: ${typeof value === 'number' ? value.toFixed(2) : value}`}
                    size="small"
                    variant="outlined"
                  />
                ))}
              </Box>
            </Box>
          )}
        </Paper>
      )}

      {/* Recent Alerts */}
      {alerts.length > 0 && (
        <Paper elevation={2} sx={{ p: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
            <Typography variant="h6">Recent Alerts</Typography>
            <Chip 
              icon={<NotificationsIcon />} 
              label={alerts.length} 
              color="warning" 
              size="small" 
            />
          </Box>
          <Box sx={{ maxHeight: 200, overflowY: 'auto' }}>
            {alerts.slice(-10).reverse().map((alert, index) => (
              <Alert 
                key={index} 
                severity="warning" 
                sx={{ mb: 1 }}
                onClose={() => setAlerts(prev => prev.filter((_, i) => i !== prev.length - 1 - index))}
              >
                {alert}
              </Alert>
            ))}
          </Box>
        </Paper>
      )}

      {/* Settings Menu */}
      <Menu
        anchorEl={settingsAnchor}
        open={Boolean(settingsAnchor)}
        onClose={() => setSettingsAnchor(null)}
      >
        <MenuItem onClick={() => {
          ToolStatusMonitor.setConfig({ interval: 15000 });
          setSettingsAnchor(null);
        }}>
          Refresh every 15s
        </MenuItem>
        <MenuItem onClick={() => {
          ToolStatusMonitor.setConfig({ interval: 30000 });
          setSettingsAnchor(null);
        }}>
          Refresh every 30s
        </MenuItem>
        <MenuItem onClick={() => {
          ToolStatusMonitor.setConfig({ interval: 60000 });
          setSettingsAnchor(null);
        }}>
          Refresh every 60s
        </MenuItem>
        <Divider />
        <MenuItem onClick={() => {
          setAlerts([]);
          setSettingsAnchor(null);
        }}>
          Clear Alerts
        </MenuItem>
      </Menu>

      {/* Alert Snackbar */}
      <Snackbar
        open={showAlert}
        autoHideDuration={6000}
        onClose={() => setShowAlert(false)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert severity="warning" onClose={() => setShowAlert(false)}>
          {alertMessage}
        </Alert>
      </Snackbar>

      {/* CSS for rotating icon */}
      <style>
        {`
          @keyframes rotate {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
          }
          .rotating {
            animation: rotate 1s linear infinite;
          }
        `}
      </style>
    </Box>
  );
};

export default StatusDashboard;