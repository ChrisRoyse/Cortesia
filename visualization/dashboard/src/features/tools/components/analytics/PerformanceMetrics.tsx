import React, { useMemo } from 'react';
import {
  Box,
  Typography,
  Stack,
  LinearProgress,
  Tooltip,
  Chip,
  Paper,
  Grid,
  useTheme,
  alpha,
} from '@mui/material';
import {
  Speed,
  Timer,
  TrendingUp,
  Warning,
  CheckCircle,
  Error as ErrorIcon,
} from '@mui/icons-material';
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  Cell,
  PieChart,
  Pie,
  Legend,
} from 'recharts';
import { MCPTool, ToolExecution, ToolMetrics } from '../../types';
import ToolAnalytics from '../../services/ToolAnalytics';

interface PerformanceMetricsProps {
  tools: MCPTool[];
  executions: ToolExecution[];
  selectedToolId?: string;
  showDetails?: boolean;
  compareMode?: boolean;
  height?: number;
}

interface MetricBar {
  label: string;
  value: number;
  max: number;
  unit: string;
  color: 'primary' | 'success' | 'warning' | 'error';
  icon: React.ReactNode;
}

interface ResponseTimeDistribution {
  range: string;
  count: number;
  percentage: number;
}

const PerformanceMetrics: React.FC<PerformanceMetricsProps> = ({
  tools,
  executions,
  selectedToolId,
  showDetails = false,
  compareMode = false,
  height = 300,
}) => {
  const theme = useTheme();

  // Calculate metrics for selected tool or overall
  const metrics = useMemo(() => {
    const filteredExecutions = selectedToolId
      ? executions.filter(e => e.toolId === selectedToolId)
      : executions;

    return ToolAnalytics.calculateMetrics(filteredExecutions);
  }, [executions, selectedToolId]);

  // Calculate response time distribution
  const responseTimeDistribution = useMemo(() => {
    const ranges = [
      { min: 0, max: 100, label: '0-100ms' },
      { min: 100, max: 500, label: '100-500ms' },
      { min: 500, max: 1000, label: '500ms-1s' },
      { min: 1000, max: 5000, label: '1-5s' },
      { min: 5000, max: Infinity, label: '>5s' },
    ];

    const distribution: ResponseTimeDistribution[] = ranges.map(range => ({
      range: range.label,
      count: 0,
      percentage: 0,
    }));

    const successfulExecutions = executions.filter(e => 
      e.status === 'success' && e.endTime && (!selectedToolId || e.toolId === selectedToolId)
    );

    successfulExecutions.forEach(exec => {
      const responseTime = exec.endTime! - exec.startTime;
      const rangeIndex = ranges.findIndex(r => responseTime >= r.min && responseTime < r.max);
      if (rangeIndex !== -1) {
        distribution[rangeIndex].count++;
      }
    });

    const total = successfulExecutions.length;
    distribution.forEach(d => {
      d.percentage = total > 0 ? (d.count / total) * 100 : 0;
    });

    return distribution;
  }, [executions, selectedToolId]);

  // Calculate performance score (0-100)
  const performanceScore = useMemo(() => {
    const weights = {
      responseTime: 0.3,
      successRate: 0.3,
      consistency: 0.2,
      throughput: 0.2,
    };

    // Response time score (inverse - lower is better)
    const responseTimeScore = Math.max(0, 100 - (metrics.averageResponseTime / 50));

    // Success rate score
    const successRateScore = metrics.successRate * 100;

    // Consistency score (based on variance between p50 and p95)
    const variance = metrics.p95ResponseTime > 0 
      ? (metrics.p95ResponseTime - metrics.p50ResponseTime) / metrics.p95ResponseTime
      : 0;
    const consistencyScore = Math.max(0, 100 - (variance * 100));

    // Throughput score (normalized)
    const throughputScore = Math.min(100, (metrics.totalExecutions / 100) * 100);

    return Math.round(
      responseTimeScore * weights.responseTime +
      successRateScore * weights.successRate +
      consistencyScore * weights.consistency +
      throughputScore * weights.throughput
    );
  }, [metrics]);

  // Performance radar chart data
  const radarData = useMemo(() => {
    if (!compareMode) {
      return [
        { metric: 'Speed', value: Math.max(0, 100 - (metrics.averageResponseTime / 50)) },
        { metric: 'Reliability', value: metrics.successRate * 100 },
        { metric: 'Consistency', value: Math.max(0, 100 - ((metrics.p95ResponseTime - metrics.p50ResponseTime) / metrics.p95ResponseTime * 100)) },
        { metric: 'Throughput', value: Math.min(100, (metrics.totalExecutions / 100) * 100) },
        { metric: 'Efficiency', value: performanceScore },
      ];
    }

    // Compare mode: show multiple tools
    const toolsToCompare = selectedToolId 
      ? tools.filter(t => t.id === selectedToolId)
      : tools.slice(0, 3);

    const radarMetrics = ['Speed', 'Reliability', 'Consistency', 'Throughput', 'Efficiency'];
    const data: any[] = radarMetrics.map(metric => ({ metric }));

    toolsToCompare.forEach(tool => {
      const toolExecutions = executions.filter(e => e.toolId === tool.id);
      const toolMetrics = ToolAnalytics.calculateMetrics(toolExecutions);

      data.forEach(item => {
        switch (item.metric) {
          case 'Speed':
            item[tool.name] = Math.max(0, 100 - (toolMetrics.averageResponseTime / 50));
            break;
          case 'Reliability':
            item[tool.name] = toolMetrics.successRate * 100;
            break;
          case 'Consistency':
            const variance = toolMetrics.p95ResponseTime > 0
              ? (toolMetrics.p95ResponseTime - toolMetrics.p50ResponseTime) / toolMetrics.p95ResponseTime
              : 0;
            item[tool.name] = Math.max(0, 100 - (variance * 100));
            break;
          case 'Throughput':
            item[tool.name] = Math.min(100, (toolMetrics.totalExecutions / 100) * 100);
            break;
          case 'Efficiency':
            // Calculate efficiency score for this tool
            const score = calculateToolScore(toolMetrics);
            item[tool.name] = score;
            break;
        }
      });
    });

    return data;
  }, [tools, executions, metrics, compareMode, selectedToolId, performanceScore]);

  // Helper function to calculate tool score
  function calculateToolScore(toolMetrics: ToolMetrics): number {
    const weights = { responseTime: 0.3, successRate: 0.3, consistency: 0.2, throughput: 0.2 };
    const responseTimeScore = Math.max(0, 100 - (toolMetrics.averageResponseTime / 50));
    const successRateScore = toolMetrics.successRate * 100;
    const variance = toolMetrics.p95ResponseTime > 0
      ? (toolMetrics.p95ResponseTime - toolMetrics.p50ResponseTime) / toolMetrics.p95ResponseTime
      : 0;
    const consistencyScore = Math.max(0, 100 - (variance * 100));
    const throughputScore = Math.min(100, (toolMetrics.totalExecutions / 100) * 100);

    return Math.round(
      responseTimeScore * weights.responseTime +
      successRateScore * weights.successRate +
      consistencyScore * weights.consistency +
      throughputScore * weights.throughput
    );
  }

  // Metric bars configuration
  const metricBars: MetricBar[] = [
    {
      label: 'Avg Response Time',
      value: metrics.averageResponseTime,
      max: 5000,
      unit: 'ms',
      color: metrics.averageResponseTime < 1000 ? 'success' : metrics.averageResponseTime < 3000 ? 'warning' : 'error',
      icon: <Timer />,
    },
    {
      label: 'P95 Response Time',
      value: metrics.p95ResponseTime,
      max: 10000,
      unit: 'ms',
      color: metrics.p95ResponseTime < 2000 ? 'success' : metrics.p95ResponseTime < 5000 ? 'warning' : 'error',
      icon: <Speed />,
    },
    {
      label: 'Success Rate',
      value: metrics.successRate * 100,
      max: 100,
      unit: '%',
      color: metrics.successRate > 0.95 ? 'success' : metrics.successRate > 0.9 ? 'warning' : 'error',
      icon: <CheckCircle />,
    },
    {
      label: 'Error Rate',
      value: (1 - metrics.successRate) * 100,
      max: 100,
      unit: '%',
      color: metrics.successRate > 0.95 ? 'success' : metrics.successRate > 0.9 ? 'warning' : 'error',
      icon: <ErrorIcon />,
    },
  ];

  // Custom tooltip for charts
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload || !payload.length) return null;

    return (
      <Paper
        sx={{
          p: 1.5,
          backgroundColor: alpha(theme.palette.background.paper, 0.95),
          border: `1px solid ${theme.palette.divider}`,
        }}
      >
        <Typography variant="caption" fontWeight="bold">
          {label}
        </Typography>
        {payload.map((entry: any, index: number) => (
          <Box key={index} display="flex" alignItems="center" gap={0.5} mt={0.5}>
            <Box
              sx={{
                width: 8,
                height: 8,
                borderRadius: '50%',
                backgroundColor: entry.color,
              }}
            />
            <Typography variant="caption">
              {entry.name}: {entry.value.toFixed(1)}
            </Typography>
          </Box>
        ))}
      </Paper>
    );
  };

  // Get performance level
  function getPerformanceLevel(score: number): { label: string; color: string; icon: React.ReactNode } {
    if (score >= 90) {
      return { label: 'Excellent', color: theme.palette.success.main, icon: <CheckCircle /> };
    } else if (score >= 70) {
      return { label: 'Good', color: theme.palette.info.main, icon: <TrendingUp /> };
    } else if (score >= 50) {
      return { label: 'Fair', color: theme.palette.warning.main, icon: <Warning /> };
    } else {
      return { label: 'Poor', color: theme.palette.error.main, icon: <ErrorIcon /> };
    }
  }

  const performanceLevel = getPerformanceLevel(performanceScore);

  return (
    <Box>
      {/* Overall Performance Score */}
      <Box mb={3}>
        <Stack direction="row" alignItems="center" justifyContent="space-between" mb={1}>
          <Typography variant="subtitle2" color="text.secondary">
            Overall Performance Score
          </Typography>
          <Chip
            label={performanceLevel.label}
            color={
              performanceLevel.label === 'Excellent' ? 'success' :
              performanceLevel.label === 'Good' ? 'info' :
              performanceLevel.label === 'Fair' ? 'warning' : 'error'
            }
            size="small"
            icon={performanceLevel.icon}
          />
        </Stack>
        <Box position="relative">
          <LinearProgress
            variant="determinate"
            value={performanceScore}
            sx={{
              height: 24,
              borderRadius: 12,
              backgroundColor: alpha(performanceLevel.color, 0.1),
              '& .MuiLinearProgress-bar': {
                backgroundColor: performanceLevel.color,
                borderRadius: 12,
              },
            }}
          />
          <Box
            position="absolute"
            top="50%"
            left="50%"
            sx={{ transform: 'translate(-50%, -50%)' }}
          >
            <Typography variant="subtitle2" fontWeight="bold">
              {performanceScore}/100
            </Typography>
          </Box>
        </Box>
      </Box>

      {/* Performance Radar Chart */}
      {showDetails && (
        <Box mb={3}>
          <Typography variant="subtitle2" color="text.secondary" gutterBottom>
            Performance Profile
          </Typography>
          <ResponsiveContainer width="100%" height={height}>
            <RadarChart data={radarData}>
              <PolarGrid stroke={theme.palette.divider} />
              <PolarAngleAxis 
                dataKey="metric" 
                tick={{ fontSize: 12 }}
                stroke={theme.palette.text.secondary}
              />
              <PolarRadiusAxis 
                angle={90} 
                domain={[0, 100]}
                tick={{ fontSize: 10 }}
                stroke={theme.palette.text.secondary}
              />
              {!compareMode ? (
                <Radar
                  name="Performance"
                  dataKey="value"
                  stroke={theme.palette.primary.main}
                  fill={theme.palette.primary.main}
                  fillOpacity={0.6}
                />
              ) : (
                tools.slice(0, 3).map((tool, index) => (
                  <Radar
                    key={tool.id}
                    name={tool.name}
                    dataKey={tool.name}
                    stroke={getToolColor(index)}
                    fill={getToolColor(index)}
                    fillOpacity={0.3}
                  />
                ))
              )}
              <RechartsTooltip content={<CustomTooltip />} />
              {compareMode && <Legend />}
            </RadarChart>
          </ResponsiveContainer>
        </Box>
      )}

      {/* Key Metrics */}
      <Grid container spacing={2} mb={3}>
        {metricBars.map((metric, index) => (
          <Grid item xs={12} sm={6} key={index}>
            <Stack spacing={1}>
              <Stack direction="row" alignItems="center" justifyContent="space-between">
                <Stack direction="row" alignItems="center" spacing={1}>
                  <Box color={`${metric.color}.main`}>{metric.icon}</Box>
                  <Typography variant="caption" color="text.secondary">
                    {metric.label}
                  </Typography>
                </Stack>
                <Typography variant="subtitle2" fontWeight="bold">
                  {metric.value.toFixed(metric.unit === '%' ? 1 : 0)}{metric.unit}
                </Typography>
              </Stack>
              <LinearProgress
                variant="determinate"
                value={(metric.value / metric.max) * 100}
                sx={{
                  height: 6,
                  borderRadius: 3,
                  backgroundColor: alpha(theme.palette[metric.color].main, 0.1),
                  '& .MuiLinearProgress-bar': {
                    backgroundColor: theme.palette[metric.color].main,
                    borderRadius: 3,
                  },
                }}
              />
            </Stack>
          </Grid>
        ))}
      </Grid>

      {/* Response Time Distribution */}
      {showDetails && (
        <Box>
          <Typography variant="subtitle2" color="text.secondary" gutterBottom>
            Response Time Distribution
          </Typography>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={responseTimeDistribution}>
              <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
              <XAxis 
                dataKey="range" 
                tick={{ fontSize: 11 }}
                stroke={theme.palette.text.secondary}
              />
              <YAxis 
                tick={{ fontSize: 11 }}
                stroke={theme.palette.text.secondary}
                label={{ value: 'Percentage (%)', angle: -90, position: 'insideLeft' }}
              />
              <RechartsTooltip content={<CustomTooltip />} />
              <Bar dataKey="percentage" name="Percentage">
                {responseTimeDistribution.map((entry, index) => (
                  <Cell 
                    key={`cell-${index}`} 
                    fill={
                      index === 0 ? theme.palette.success.main :
                      index === 1 ? theme.palette.info.main :
                      index === 2 ? theme.palette.warning.main :
                      theme.palette.error.main
                    }
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Box>
      )}

      {/* Summary Stats */}
      <Box mt={2} p={2} bgcolor={alpha(theme.palette.action.hover, 0.05)} borderRadius={1}>
        <Grid container spacing={2}>
          <Grid item xs={6}>
            <Typography variant="caption" color="text.secondary">
              Total Executions
            </Typography>
            <Typography variant="h6">{metrics.totalExecutions.toLocaleString()}</Typography>
          </Grid>
          <Grid item xs={6}>
            <Typography variant="caption" color="text.secondary">
              Error Count
            </Typography>
            <Typography variant="h6" color={metrics.errorCount > 0 ? 'error.main' : 'text.primary'}>
              {metrics.errorCount}
            </Typography>
          </Grid>
          <Grid item xs={6}>
            <Typography variant="caption" color="text.secondary">
              P50 Response Time
            </Typography>
            <Typography variant="h6">{metrics.p50ResponseTime.toFixed(0)}ms</Typography>
          </Grid>
          <Grid item xs={6}>
            <Typography variant="caption" color="text.secondary">
              P99 Response Time
            </Typography>
            <Typography variant="h6">{metrics.p99ResponseTime.toFixed(0)}ms</Typography>
          </Grid>
        </Grid>
      </Box>
    </Box>
  );

  function getToolColor(index: number): string {
    const colors = [
      theme.palette.primary.main,
      theme.palette.secondary.main,
      theme.palette.success.main,
    ];
    return colors[index % colors.length];
  }
};

export default PerformanceMetrics;