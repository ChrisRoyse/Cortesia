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
  Button,
  ButtonGroup,
  Stack,
  Chip,
  Alert,
  AlertTitle,
  CircularProgress,
  useTheme,
  alpha,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  TrendingUp,
  TrendingDown,
  Warning,
  CheckCircle,
  Error,
  Download,
  Refresh,
  Compare,
  Timeline,
  Speed,
  BugReport,
  Psychology,
  NetworkCheck,
} from '@mui/icons-material';
import { useSelector } from 'react-redux';
import { selectAllTools, selectExecutionHistory } from '../../stores/toolsSlice';
import useToolAnalytics from '../../hooks/useToolAnalytics';
import { MCPTool, ToolExecution } from '../../types';
import { TimeRange } from '../../services/ToolAnalytics';
import UsageChart from './UsageChart';
import PerformanceMetrics from './PerformanceMetrics';
import TrendAnalysis from './TrendAnalysis';

interface PerformanceDashboardProps {
  tools?: MCPTool[];
  timeRange?: TimeRange;
  compareMode?: boolean;
  selectedToolIds?: string[];
  onToolSelect?: (toolId: string) => void;
  onExportReport?: (format: 'csv' | 'pdf' | 'json') => void;
}

const defaultTimeRange: TimeRange = {
  start: new Date(Date.now() - 24 * 60 * 60 * 1000), // 24 hours ago
  end: new Date(),
  period: 'day',
};

const timeRangeOptions = [
  { label: '1 Hour', value: 'hour', duration: 60 * 60 * 1000 },
  { label: '24 Hours', value: 'day', duration: 24 * 60 * 60 * 1000 },
  { label: '7 Days', value: 'week', duration: 7 * 24 * 60 * 60 * 1000 },
  { label: '30 Days', value: 'month', duration: 30 * 24 * 60 * 60 * 1000 },
];

interface MetricCardProps {
  title: string;
  value: number | string;
  unit?: string;
  trend?: number;
  icon: React.ReactNode;
  color?: string;
  onClick?: () => void;
}

const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  unit,
  trend,
  icon,
  color = 'primary',
  onClick,
}) => {
  const theme = useTheme();
  const isPositiveTrend = trend && trend > 0;
  const isNegativeTrend = trend && trend < 0;

  return (
    <Card 
      sx={{ 
        height: '100%',
        cursor: onClick ? 'pointer' : 'default',
        transition: 'all 0.3s ease',
        '&:hover': onClick ? {
          transform: 'translateY(-4px)',
          boxShadow: theme.shadows[8],
        } : {},
      }}
      onClick={onClick}
    >
      <CardContent>
        <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
          <Box
            sx={{
              p: 1,
              borderRadius: 2,
              backgroundColor: alpha(theme.palette[color].main, 0.1),
              color: theme.palette[color].main,
            }}
          >
            {icon}
          </Box>
          {trend !== undefined && (
            <Box display="flex" alignItems="center">
              {isPositiveTrend ? (
                <TrendingUp color="success" fontSize="small" />
              ) : isNegativeTrend ? (
                <TrendingDown color="error" fontSize="small" />
              ) : null}
              <Typography
                variant="caption"
                color={isPositiveTrend ? 'success.main' : isNegativeTrend ? 'error.main' : 'text.secondary'}
                ml={0.5}
              >
                {isPositiveTrend ? '+' : ''}{trend.toFixed(1)}%
              </Typography>
            </Box>
          )}
        </Box>
        <Typography color="text.secondary" variant="body2" gutterBottom>
          {title}
        </Typography>
        <Typography variant="h4" component="div">
          {typeof value === 'number' ? value.toLocaleString() : value}
          {unit && (
            <Typography variant="body2" component="span" color="text.secondary" ml={0.5}>
              {unit}
            </Typography>
          )}
        </Typography>
      </CardContent>
    </Card>
  );
};

const PerformanceDashboard: React.FC<PerformanceDashboardProps> = ({
  tools: propTools,
  timeRange = defaultTimeRange,
  compareMode = false,
  selectedToolIds = [],
  onToolSelect,
  onExportReport,
}) => {
  const theme = useTheme();
  const storeTools = useSelector(selectAllTools);
  const executionHistory = useSelector(selectExecutionHistory);
  const tools = propTools || storeTools;

  const [selectedTimeRange, setSelectedTimeRange] = useState(timeRange);
  const [refreshing, setRefreshing] = useState(false);
  const [activeView, setActiveView] = useState<'overview' | 'trends' | 'insights'>('overview');

  const {
    loading,
    error,
    report,
    trends,
    anomalies,
    insights,
    cognitiveMetrics,
    refresh,
  } = useToolAnalytics(
    selectedToolIds.length > 0 ? selectedToolIds : tools.map(t => t.id),
    selectedTimeRange
  );

  const handleTimeRangeChange = (period: string) => {
    const option = timeRangeOptions.find(opt => opt.value === period);
    if (option) {
      setSelectedTimeRange({
        start: new Date(Date.now() - option.duration),
        end: new Date(),
        period: period as TimeRange['period'],
      });
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    await refresh();
    setTimeout(() => setRefreshing(false), 500);
  };

  const handleExport = (format: 'csv' | 'pdf' | 'json') => {
    if (onExportReport) {
      onExportReport(format);
    }
  };

  // Calculate key metrics
  const overallMetrics = useMemo(() => {
    if (!report) {
      return {
        totalExecutions: 0,
        successRate: 0,
        avgResponseTime: 0,
        errorCount: 0,
        activeTools: 0,
        peakConcurrency: 0,
      };
    }

    return {
      totalExecutions: report.summary.totalExecutions,
      successRate: report.summary.overallSuccessRate * 100,
      avgResponseTime: report.summary.averageResponseTime,
      errorCount: executionHistory.filter(e => e.status === 'error').length,
      activeTools: report.summary.totalTools,
      peakConcurrency: report.summary.peakConcurrency,
    };
  }, [report, executionHistory]);

  // Group anomalies by severity
  const anomalyGroups = useMemo(() => {
    const groups = {
      critical: [] as typeof anomalies,
      high: [] as typeof anomalies,
      medium: [] as typeof anomalies,
      low: [] as typeof anomalies,
    };

    anomalies.forEach(anomaly => {
      groups[anomaly.severity].push(anomaly);
    });

    return groups;
  }, [anomalies]);

  if (loading && !refreshing) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight={400}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error">
        <AlertTitle>Error Loading Analytics</AlertTitle>
        {error}
      </Alert>
    );
  }

  return (
    <Box>
      {/* Header */}
      <Box mb={3}>
        <Grid container alignItems="center" justifyContent="space-between">
          <Grid item>
            <Typography variant="h4" gutterBottom>
              Performance Analytics
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Real-time insights into tool performance and usage patterns
            </Typography>
          </Grid>
          <Grid item>
            <Stack direction="row" spacing={2}>
              <ButtonGroup size="small">
                {timeRangeOptions.map(option => (
                  <Button
                    key={option.value}
                    variant={selectedTimeRange.period === option.value ? 'contained' : 'outlined'}
                    onClick={() => handleTimeRangeChange(option.value)}
                  >
                    {option.label}
                  </Button>
                ))}
              </ButtonGroup>
              <Tooltip title="Refresh data">
                <IconButton onClick={handleRefresh} disabled={refreshing}>
                  <Refresh className={refreshing ? 'rotating' : ''} />
                </IconButton>
              </Tooltip>
              <ButtonGroup size="small">
                <Button
                  variant="outlined"
                  startIcon={<Download />}
                  onClick={() => handleExport('csv')}
                >
                  CSV
                </Button>
                <Button
                  variant="outlined"
                  onClick={() => handleExport('pdf')}
                >
                  PDF
                </Button>
                <Button
                  variant="outlined"
                  onClick={() => handleExport('json')}
                >
                  JSON
                </Button>
              </ButtonGroup>
            </Stack>
          </Grid>
        </Grid>
      </Box>

      {/* Anomaly Alerts */}
      {anomalyGroups.critical.length > 0 && (
        <Alert severity="error" sx={{ mb: 2 }}>
          <AlertTitle>Critical Issues Detected</AlertTitle>
          {anomalyGroups.critical.map((anomaly, index) => (
            <Box key={index}>
              <Typography variant="body2">
                {anomaly.description} - {anomaly.suggestedAction}
              </Typography>
            </Box>
          ))}
        </Alert>
      )}

      {/* View Tabs */}
      <Box mb={3}>
        <ButtonGroup>
          <Button
            variant={activeView === 'overview' ? 'contained' : 'outlined'}
            onClick={() => setActiveView('overview')}
            startIcon={<DashboardIcon />}
          >
            Overview
          </Button>
          <Button
            variant={activeView === 'trends' ? 'contained' : 'outlined'}
            onClick={() => setActiveView('trends')}
            startIcon={<Timeline />}
          >
            Trends
          </Button>
          <Button
            variant={activeView === 'insights' ? 'contained' : 'outlined'}
            onClick={() => setActiveView('insights')}
            startIcon={<Psychology />}
          >
            Insights
          </Button>
        </ButtonGroup>
      </Box>

      {/* Overview Section */}
      {activeView === 'overview' && (
        <Grid container spacing={3}>
          {/* Key Metrics */}
          <Grid item xs={12}>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6} md={4} lg={2}>
                <MetricCard
                  title="Total Executions"
                  value={overallMetrics.totalExecutions}
                  icon={<Dashboard />}
                  color="primary"
                />
              </Grid>
              <Grid item xs={12} sm={6} md={4} lg={2}>
                <MetricCard
                  title="Success Rate"
                  value={overallMetrics.successRate.toFixed(1)}
                  unit="%"
                  trend={5.2}
                  icon={<CheckCircle />}
                  color="success"
                />
              </Grid>
              <Grid item xs={12} sm={6} md={4} lg={2}>
                <MetricCard
                  title="Avg Response Time"
                  value={overallMetrics.avgResponseTime.toFixed(0)}
                  unit="ms"
                  trend={-12.5}
                  icon={<Speed />}
                  color="info"
                />
              </Grid>
              <Grid item xs={12} sm={6} md={4} lg={2}>
                <MetricCard
                  title="Error Count"
                  value={overallMetrics.errorCount}
                  trend={anomalyGroups.high.length > 0 ? 15.3 : -8.1}
                  icon={<BugReport />}
                  color="error"
                />
              </Grid>
              <Grid item xs={12} sm={6} md={4} lg={2}>
                <MetricCard
                  title="Active Tools"
                  value={overallMetrics.activeTools}
                  icon={<NetworkCheck />}
                  color="secondary"
                />
              </Grid>
              <Grid item xs={12} sm={6} md={4} lg={2}>
                <MetricCard
                  title="Peak Concurrency"
                  value={overallMetrics.peakConcurrency}
                  icon={<Timeline />}
                  color="warning"
                />
              </Grid>
            </Grid>
          </Grid>

          {/* Usage Chart */}
          <Grid item xs={12} lg={8}>
            <Paper sx={{ p: 3, height: '100%' }}>
              <Typography variant="h6" gutterBottom>
                Usage Patterns
              </Typography>
              <UsageChart
                executions={executionHistory}
                timeRange={selectedTimeRange}
                groupBy="hour"
                showComparison={compareMode}
              />
            </Paper>
          </Grid>

          {/* Performance Metrics */}
          <Grid item xs={12} lg={4}>
            <Paper sx={{ p: 3, height: '100%' }}>
              <Typography variant="h6" gutterBottom>
                Performance Distribution
              </Typography>
              <PerformanceMetrics
                tools={tools}
                executions={executionHistory}
                showDetails
              />
            </Paper>
          </Grid>

          {/* Top/Bottom Performers */}
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom color="success.main">
                Top Performers
              </Typography>
              <Stack spacing={1}>
                {report?.topPerformers.slice(0, 5).map((item, index) => (
                  <Box
                    key={index}
                    display="flex"
                    alignItems="center"
                    justifyContent="space-between"
                    p={1}
                    borderRadius={1}
                    bgcolor={alpha(theme.palette.success.main, 0.08)}
                    sx={{ cursor: 'pointer' }}
                    onClick={() => onToolSelect?.(item.tool.id)}
                  >
                    <Typography variant="body2">{item.tool.name || item.tool.id}</Typography>
                    <Chip
                      label={`${item.metrics.averageResponseTime.toFixed(0)}ms`}
                      size="small"
                      color="success"
                    />
                  </Box>
                ))}
              </Stack>
            </Paper>
          </Grid>

          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom color="error.main">
                Performance Issues
              </Typography>
              <Stack spacing={1}>
                {report?.bottomPerformers.slice(0, 5).map((item, index) => (
                  <Box
                    key={index}
                    display="flex"
                    alignItems="center"
                    justifyContent="space-between"
                    p={1}
                    borderRadius={1}
                    bgcolor={alpha(theme.palette.error.main, 0.08)}
                    sx={{ cursor: 'pointer' }}
                    onClick={() => onToolSelect?.(item.tool.id)}
                  >
                    <Typography variant="body2">{item.tool.name || item.tool.id}</Typography>
                    <Chip
                      label={`${item.metrics.averageResponseTime.toFixed(0)}ms`}
                      size="small"
                      color="error"
                    />
                  </Box>
                ))}
              </Stack>
            </Paper>
          </Grid>

          {/* Cognitive Metrics for LLMKG */}
          {cognitiveMetrics && cognitiveMetrics.length > 0 && (
            <Grid item xs={12}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Cognitive Performance Metrics
                </Typography>
                <Grid container spacing={2}>
                  {cognitiveMetrics.map((metric, index) => (
                    <Grid item xs={12} sm={6} md={4} key={index}>
                      <Box p={2} bgcolor={alpha(theme.palette.primary.main, 0.05)} borderRadius={2}>
                        <Typography variant="caption" color="text.secondary">
                          {metric.toolId}
                        </Typography>
                        <Stack spacing={1} mt={1}>
                          <Box display="flex" justifyContent="space-between">
                            <Typography variant="body2">Neural Processing</Typography>
                            <Typography variant="body2" fontWeight="bold">
                              {metric.neuralProcessingSpeed.toFixed(1)}ms
                            </Typography>
                          </Box>
                          <Box display="flex" justifyContent="space-between">
                            <Typography variant="body2">Memory Consolidation</Typography>
                            <Typography variant="body2" fontWeight="bold">
                              {(metric.memoryConsolidationRate * 100).toFixed(1)}%
                            </Typography>
                          </Box>
                          <Box display="flex" justifyContent="space-between">
                            <Typography variant="body2">Graph Query Time</Typography>
                            <Typography variant="body2" fontWeight="bold">
                              {metric.knowledgeGraphQueryTime.toFixed(1)}ms
                            </Typography>
                          </Box>
                          <Box display="flex" justifyContent="space-between">
                            <Typography variant="body2">Federation Latency</Typography>
                            <Typography variant="body2" fontWeight="bold">
                              {metric.federationLatency.toFixed(1)}ms
                            </Typography>
                          </Box>
                          <Box display="flex" justifyContent="space-between">
                            <Typography variant="body2">Pattern Recognition</Typography>
                            <Typography variant="body2" fontWeight="bold">
                              {(metric.patternRecognitionAccuracy * 100).toFixed(1)}%
                            </Typography>
                          </Box>
                        </Stack>
                      </Box>
                    </Grid>
                  ))}
                </Grid>
              </Paper>
            </Grid>
          )}
        </Grid>
      )}

      {/* Trends Section */}
      {activeView === 'trends' && (
        <Grid container spacing={3}>
          {selectedToolIds.map(toolId => (
            <Grid item xs={12} key={toolId}>
              <TrendAnalysis
                toolId={toolId}
                timeRange={selectedTimeRange}
                showPredictions
                onAnomalyClick={(anomaly) => console.log('Anomaly clicked:', anomaly)}
              />
            </Grid>
          ))}
          {selectedToolIds.length === 0 && (
            <Grid item xs={12}>
              <Alert severity="info">
                Select one or more tools to view their performance trends
              </Alert>
            </Grid>
          )}
        </Grid>
      )}

      {/* Insights Section */}
      {activeView === 'insights' && (
        <Grid container spacing={3}>
          {insights.map((insight, index) => (
            <Grid item xs={12} md={6} key={insight.id}>
              <Paper sx={{ p: 3, height: '100%' }}>
                <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
                  <Chip
                    label={insight.type}
                    size="small"
                    color={
                      insight.type === 'optimization' ? 'primary' :
                      insight.type === 'anomaly' ? 'error' :
                      insight.type === 'pattern' ? 'info' : 'default'
                    }
                  />
                  <Chip
                    label={insight.impact}
                    size="small"
                    variant="outlined"
                    color={
                      insight.impact === 'high' ? 'error' :
                      insight.impact === 'medium' ? 'warning' : 'default'
                    }
                  />
                </Box>
                <Typography variant="h6" gutterBottom>
                  {insight.title}
                </Typography>
                <Typography variant="body2" color="text.secondary" paragraph>
                  {insight.description}
                </Typography>
                <Typography variant="subtitle2" gutterBottom>
                  Recommended Actions:
                </Typography>
                <Stack spacing={1}>
                  {insight.actions.map((action, actionIndex) => (
                    <Box key={actionIndex} display="flex" alignItems="flex-start">
                      <Typography variant="body2" mr={1}>•</Typography>
                      <Typography variant="body2">{action}</Typography>
                    </Box>
                  ))}
                </Stack>
              </Paper>
            </Grid>
          ))}
        </Grid>
      )}

      <style jsx>{`
        @keyframes rotate {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        
        .rotating {
          animation: rotate 1s linear infinite;
        }
      `}</style>
    </Box>
  );
};

export { PerformanceDashboard };
export default PerformanceDashboard;