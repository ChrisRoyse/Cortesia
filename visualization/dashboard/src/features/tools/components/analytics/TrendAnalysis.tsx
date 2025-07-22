import React, { useMemo, useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  Stack,
  Chip,
  IconButton,
  Tooltip,
  Alert,
  AlertTitle,
  Grid,
  FormControlLabel,
  Switch,
  Menu,
  MenuItem,
  useTheme,
  alpha,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  TrendingFlat,
  ShowChart,
  Timeline,
  Warning,
  Info,
  MoreVert,
  ZoomIn,
  Download,
  Insights,
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  ReferenceArea,
  Dot,
} from 'recharts';
import { format } from 'date-fns';
import useToolAnalytics from '../../hooks/useToolAnalytics';
import { TimeRange, TrendAnalysis as TrendData, Anomaly } from '../../services/ToolAnalytics';

interface TrendAnalysisProps {
  toolId: string;
  timeRange: TimeRange;
  showPredictions?: boolean;
  showAnomalies?: boolean;
  showConfidenceInterval?: boolean;
  onAnomalyClick?: (anomaly: Anomaly) => void;
  height?: number;
}

interface MetricConfig {
  key: keyof TrendData['metrics'];
  label: string;
  color: string;
  unit: string;
  format: (value: number) => string;
  visible: boolean;
}

const TrendAnalysis: React.FC<TrendAnalysisProps> = ({
  toolId,
  timeRange,
  showPredictions = true,
  showAnomalies = true,
  showConfidenceInterval = false,
  onAnomalyClick,
  height = 400,
}) => {
  const theme = useTheme();
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [visibleMetrics, setVisibleMetrics] = useState<Record<string, boolean>>({
    responseTime: true,
    successRate: true,
    throughput: false,
    errorRate: false,
  });

  const { trends, anomalies, loading, error } = useToolAnalytics([toolId], timeRange);

  // Get trend data for this tool
  const trendData = useMemo(() => {
    return trends.find(t => t.toolId === toolId);
  }, [trends, toolId]);

  // Get anomalies for this tool
  const toolAnomalies = useMemo(() => {
    return anomalies.filter(a => a.toolId === toolId);
  }, [anomalies, toolId]);

  // Metric configurations
  const metricConfigs: MetricConfig[] = [
    {
      key: 'responseTime',
      label: 'Response Time',
      color: theme.palette.primary.main,
      unit: 'ms',
      format: (v) => `${v.toFixed(0)}ms`,
      visible: visibleMetrics.responseTime,
    },
    {
      key: 'successRate',
      label: 'Success Rate',
      color: theme.palette.success.main,
      unit: '%',
      format: (v) => `${(v * 100).toFixed(1)}%`,
      visible: visibleMetrics.successRate,
    },
    {
      key: 'throughput',
      label: 'Throughput',
      color: theme.palette.info.main,
      unit: 'req/min',
      format: (v) => `${v.toFixed(0)}`,
      visible: visibleMetrics.throughput,
    },
    {
      key: 'errorRate',
      label: 'Error Rate',
      color: theme.palette.error.main,
      unit: '%',
      format: (v) => `${(v * 100).toFixed(1)}%`,
      visible: visibleMetrics.errorRate,
    },
  ];

  // Process chart data
  const chartData = useMemo(() => {
    if (!trendData) return [];

    const data: any[] = [];
    const metrics = trendData.metrics;

    // Combine all data points from different metrics
    const allDataPoints = new Map<string, any>();

    Object.entries(metrics).forEach(([metricKey, metricData]) => {
      metricData.dataPoints.forEach(point => {
        const timeKey = point.time.toISOString();
        if (!allDataPoints.has(timeKey)) {
          allDataPoints.set(timeKey, {
            time: point.time,
            timestamp: point.time.getTime(),
          });
        }
        const dataPoint = allDataPoints.get(timeKey);
        dataPoint[metricKey] = point.value;
      });
    });

    // Convert to array and sort by time
    return Array.from(allDataPoints.values())
      .sort((a, b) => a.timestamp - b.timestamp)
      .map(point => ({
        ...point,
        time: format(point.time, 'MMM dd HH:mm'),
      }));
  }, [trendData]);

  // Calculate prediction data points
  const predictionData = useMemo(() => {
    if (!showPredictions || !trendData || chartData.length === 0) return [];

    const lastDataPoint = chartData[chartData.length - 1];
    const predictions = trendData.predictions;

    // Simple linear projection for next hour
    const projectedPoints = [];
    const intervals = 4; // 15-minute intervals

    for (let i = 1; i <= intervals; i++) {
      const futureTime = new Date(lastDataPoint.timestamp + (i * 15 * 60 * 1000));
      const trendFactor = predictions.performanceTrend === 'improving' ? 0.95 :
                         predictions.performanceTrend === 'degrading' ? 1.05 : 1;

      projectedPoints.push({
        time: format(futureTime, 'MMM dd HH:mm'),
        timestamp: futureTime.getTime(),
        responseTime: lastDataPoint.responseTime * Math.pow(trendFactor, i),
        successRate: Math.max(0, Math.min(1, lastDataPoint.successRate + (Math.random() * 0.02 - 0.01))),
        throughput: predictions.nextHourLoad / 4,
        errorRate: lastDataPoint.errorRate,
        isPrediction: true,
      });
    }

    return projectedPoints;
  }, [showPredictions, trendData, chartData]);

  // Combine actual and prediction data
  const combinedData = useMemo(() => {
    return [...chartData, ...predictionData];
  }, [chartData, predictionData]);

  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload || !payload.length) return null;

    const isPrediction = payload[0]?.payload?.isPrediction;

    return (
      <Paper
        sx={{
          p: 2,
          backgroundColor: alpha(theme.palette.background.paper, 0.95),
          border: `1px solid ${theme.palette.divider}`,
        }}
      >
        <Typography variant="subtitle2" gutterBottom>
          {label} {isPrediction && <Chip label="Prediction" size="small" color="info" />}
        </Typography>
        <Stack spacing={0.5}>
          {payload.map((entry: any) => {
            const config = metricConfigs.find(m => m.key === entry.dataKey);
            if (!config || !config.visible) return null;

            return (
              <Box key={entry.dataKey} display="flex" alignItems="center" gap={1}>
                <Box
                  sx={{
                    width: 12,
                    height: 12,
                    borderRadius: '50%',
                    backgroundColor: entry.color,
                  }}
                />
                <Typography variant="body2">
                  {entry.name}: {config.format(entry.value)}
                </Typography>
              </Box>
            );
          })}
        </Stack>
      </Paper>
    );
  };

  // Render trend indicator
  const renderTrendIndicator = (metric: keyof TrendData['metrics']) => {
    if (!trendData) return null;

    const metricData = trendData.metrics[metric];
    const trend = metricData.trend;
    const changePercent = metricData.changePercent;

    const icon = trend === 'up' ? <TrendingUp /> :
                 trend === 'down' ? <TrendingDown /> :
                 <TrendingFlat />;

    const color = metric === 'responseTime' || metric === 'errorRate'
      ? (trend === 'up' ? 'error' : trend === 'down' ? 'success' : 'default')
      : (trend === 'up' ? 'success' : trend === 'down' ? 'error' : 'default');

    return (
      <Chip
        icon={icon}
        label={`${changePercent > 0 ? '+' : ''}${changePercent.toFixed(1)}%`}
        size="small"
        color={color}
        variant="outlined"
      />
    );
  };

  // Handle metric visibility toggle
  const handleMetricToggle = (metricKey: string) => {
    setVisibleMetrics(prev => ({
      ...prev,
      [metricKey]: !prev[metricKey],
    }));
  };

  if (loading) {
    return (
      <Paper sx={{ p: 3, height: height }}>
        <Box display="flex" alignItems="center" justifyContent="center" height="100%">
          <Typography variant="body2" color="text.secondary">
            Loading trend analysis...
          </Typography>
        </Box>
      </Paper>
    );
  }

  if (error || !trendData) {
    return (
      <Alert severity="error">
        <AlertTitle>Error</AlertTitle>
        {error || 'Failed to load trend data'}
      </Alert>
    );
  }

  return (
    <Paper sx={{ p: 3 }}>
      {/* Header */}
      <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
        <Typography variant="h6">
          Performance Trends - {toolId}
        </Typography>
        <Stack direction="row" spacing={1} alignItems="center">
          {showPredictions && trendData.predictions && (
            <Chip
              icon={<Insights />}
              label={`Trend: ${trendData.predictions.performanceTrend}`}
              size="small"
              color={
                trendData.predictions.performanceTrend === 'improving' ? 'success' :
                trendData.predictions.performanceTrend === 'degrading' ? 'error' : 'default'
              }
            />
          )}
          <IconButton size="small" onClick={(e) => setAnchorEl(e.currentTarget)}>
            <MoreVert />
          </IconButton>
        </Stack>
      </Box>

      {/* Anomaly alerts */}
      {showAnomalies && toolAnomalies.length > 0 && (
        <Alert 
          severity={toolAnomalies.some(a => a.severity === 'critical') ? 'error' : 'warning'}
          sx={{ mb: 2 }}
          action={
            <IconButton
              color="inherit"
              size="small"
              onClick={() => onAnomalyClick?.(toolAnomalies[0])}
            >
              <ZoomIn />
            </IconButton>
          }
        >
          <AlertTitle>
            {toolAnomalies.length} Anomal{toolAnomalies.length === 1 ? 'y' : 'ies'} Detected
          </AlertTitle>
          {toolAnomalies[0].description}
        </Alert>
      )}

      {/* Metric summary cards */}
      <Grid container spacing={2} mb={3}>
        {metricConfigs.map(config => {
          const metricData = trendData.metrics[config.key];
          return (
            <Grid item xs={12} sm={6} md={3} key={config.key}>
              <Box
                p={2}
                borderRadius={1}
                bgcolor={alpha(config.color, 0.08)}
                border={`1px solid ${alpha(config.color, 0.2)}`}
              >
                <Stack direction="row" alignItems="center" justifyContent="space-between" mb={1}>
                  <Typography variant="caption" color="text.secondary">
                    {config.label}
                  </Typography>
                  {renderTrendIndicator(config.key)}
                </Stack>
                <Typography variant="h6" color={config.color}>
                  {config.format(metricData.current)}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  vs {config.format(metricData.previous)} previous
                </Typography>
              </Box>
            </Grid>
          );
        })}
      </Grid>

      {/* Chart */}
      <Box mb={2}>
        <ResponsiveContainer width="100%" height={height}>
          <LineChart data={combinedData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
            <XAxis 
              dataKey="time" 
              stroke={theme.palette.text.secondary}
              style={{ fontSize: 12 }}
            />
            <YAxis 
              yAxisId="left"
              stroke={theme.palette.text.secondary}
              style={{ fontSize: 12 }}
            />
            <YAxis 
              yAxisId="right"
              orientation="right"
              stroke={theme.palette.text.secondary}
              style={{ fontSize: 12 }}
            />
            <RechartsTooltip content={<CustomTooltip />} />
            <Legend />

            {/* Render lines for visible metrics */}
            {metricConfigs.map(config => {
              if (!config.visible) return null;

              const yAxisId = config.key === 'successRate' || config.key === 'errorRate' ? 'right' : 'left';

              return (
                <Line
                  key={config.key}
                  type="monotone"
                  dataKey={config.key}
                  name={config.label}
                  stroke={config.color}
                  strokeWidth={2}
                  yAxisId={yAxisId}
                  dot={false}
                  strokeDasharray={predictionData.length > 0 ? "0 5" : "0"}
                />
              );
            })}

            {/* Anomaly markers */}
            {showAnomalies && toolAnomalies.map((anomaly, index) => {
              const dataPoint = chartData.find(d => 
                Math.abs(new Date(d.timestamp).getTime() - anomaly.timestamp.getTime()) < 60000
              );
              if (!dataPoint) return null;

              return (
                <ReferenceLine
                  key={index}
                  x={dataPoint.time}
                  stroke={theme.palette.error.main}
                  strokeDasharray="5 5"
                  label={
                    <Tooltip title={anomaly.description}>
                      <Warning />
                    </Tooltip>
                  }
                />
              );
            })}

            {/* Prediction area */}
            {showPredictions && predictionData.length > 0 && (
              <ReferenceArea
                x1={predictionData[0].time}
                x2={predictionData[predictionData.length - 1].time}
                fill={theme.palette.info.main}
                fillOpacity={0.1}
                label="Predictions"
              />
            )}

            {/* Confidence intervals */}
            {showConfidenceInterval && visibleMetrics.responseTime && (
              <>
                <Area
                  type="monotone"
                  dataKey="responseTimeUpper"
                  stroke="none"
                  fill={theme.palette.primary.main}
                  fillOpacity={0.1}
                  yAxisId="left"
                />
                <Area
                  type="monotone"
                  dataKey="responseTimeLower"
                  stroke="none"
                  fill={theme.palette.primary.main}
                  fillOpacity={0.1}
                  yAxisId="left"
                />
              </>
            )}
          </LineChart>
        </ResponsiveContainer>
      </Box>

      {/* Controls */}
      <Stack direction="row" spacing={2} flexWrap="wrap">
        {metricConfigs.map(config => (
          <FormControlLabel
            key={config.key}
            control={
              <Switch
                checked={config.visible}
                onChange={() => handleMetricToggle(config.key)}
                size="small"
                sx={{
                  '& .MuiSwitch-switchBase.Mui-checked': {
                    color: config.color,
                  },
                  '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
                    backgroundColor: config.color,
                  },
                }}
              />
            }
            label={
              <Typography variant="caption" color={config.visible ? 'text.primary' : 'text.secondary'}>
                {config.label}
              </Typography>
            }
          />
        ))}
      </Stack>

      {/* Options menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={() => setAnchorEl(null)}
      >
        <MenuItem onClick={() => setAnchorEl(null)}>
          <Stack direction="row" spacing={1} alignItems="center">
            <Download fontSize="small" />
            <Typography variant="body2">Export Data</Typography>
          </Stack>
        </MenuItem>
        <MenuItem onClick={() => setAnchorEl(null)}>
          <Stack direction="row" spacing={1} alignItems="center">
            <ShowChart fontSize="small" />
            <Typography variant="body2">View Details</Typography>
          </Stack>
        </MenuItem>
      </Menu>

      {/* Risk assessment */}
      {trendData.predictions && (
        <Box mt={3} p={2} bgcolor={alpha(theme.palette.action.hover, 0.05)} borderRadius={1}>
          <Typography variant="subtitle2" gutterBottom>
            Risk Assessment
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={4}>
              <Typography variant="caption" color="text.secondary">
                Risk Score
              </Typography>
              <Typography variant="h6" color={
                trendData.predictions.riskScore < 0.3 ? 'success.main' :
                trendData.predictions.riskScore < 0.7 ? 'warning.main' : 'error.main'
              }>
                {(trendData.predictions.riskScore * 100).toFixed(0)}%
              </Typography>
            </Grid>
            <Grid item xs={4}>
              <Typography variant="caption" color="text.secondary">
                Predicted Load
              </Typography>
              <Typography variant="h6">
                {trendData.predictions.nextHourLoad} req/hr
              </Typography>
            </Grid>
            <Grid item xs={4}>
              <Typography variant="caption" color="text.secondary">
                Performance Trend
              </Typography>
              <Typography variant="h6" color={
                trendData.predictions.performanceTrend === 'improving' ? 'success.main' :
                trendData.predictions.performanceTrend === 'degrading' ? 'error.main' : 'text.primary'
              }>
                {trendData.predictions.performanceTrend}
              </Typography>
            </Grid>
          </Grid>
        </Box>
      )}
    </Paper>
  );
};

export default TrendAnalysis;