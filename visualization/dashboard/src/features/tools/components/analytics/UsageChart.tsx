import React, { useMemo } from 'react';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Brush,
  ReferenceArea,
  ReferenceLine,
} from 'recharts';
import {
  Box,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Stack,
  Chip,
  Typography,
  useTheme,
  alpha,
} from '@mui/material';
import { format, startOfHour, startOfDay, startOfWeek, startOfMonth } from 'date-fns';
import { ToolExecution } from '../../types';
import { TimeRange } from '../../services/ToolAnalytics';

interface UsageChartProps {
  executions: ToolExecution[];
  timeRange: TimeRange;
  groupBy?: 'minute' | 'hour' | 'day' | 'week';
  chartType?: 'line' | 'area' | 'bar';
  showComparison?: boolean;
  selectedToolIds?: string[];
  showBrush?: boolean;
  showAnnotations?: boolean;
  height?: number;
}

interface ChartData {
  time: string;
  timestamp: number;
  total: number;
  successful: number;
  failed: number;
  avgResponseTime: number;
  [toolId: string]: number | string;
}

const UsageChart: React.FC<UsageChartProps> = ({
  executions,
  timeRange,
  groupBy = 'hour',
  chartType = 'area',
  showComparison = false,
  selectedToolIds = [],
  showBrush = true,
  showAnnotations = true,
  height = 400,
}) => {
  const theme = useTheme();

  // Process and group execution data
  const chartData = useMemo(() => {
    const dataMap = new Map<string, ChartData>();

    // Filter executions within time range
    const filteredExecutions = executions.filter(exec => {
      const execTime = new Date(exec.startTime);
      return execTime >= timeRange.start && execTime <= timeRange.end;
    });

    // Group executions by time bucket
    filteredExecutions.forEach(exec => {
      const execTime = new Date(exec.startTime);
      let bucketTime: Date;

      switch (groupBy) {
        case 'minute':
          bucketTime = new Date(execTime.setSeconds(0, 0));
          break;
        case 'hour':
          bucketTime = startOfHour(execTime);
          break;
        case 'day':
          bucketTime = startOfDay(execTime);
          break;
        case 'week':
          bucketTime = startOfWeek(execTime);
          break;
        default:
          bucketTime = startOfHour(execTime);
      }

      const key = bucketTime.toISOString();
      const existing = dataMap.get(key) || {
        time: key,
        timestamp: bucketTime.getTime(),
        total: 0,
        successful: 0,
        failed: 0,
        avgResponseTime: 0,
        responseTimes: [] as number[],
      };

      existing.total += 1;
      if (exec.status === 'success') {
        existing.successful += 1;
        if (exec.endTime) {
          existing.responseTimes.push(exec.endTime - exec.startTime);
        }
      } else if (exec.status === 'error') {
        existing.failed += 1;
      }

      // Track per-tool counts if comparison mode is enabled
      if (showComparison && exec.toolId) {
        existing[exec.toolId] = (existing[exec.toolId] as number || 0) + 1;
      }

      dataMap.set(key, existing);
    });

    // Calculate averages and format data
    const data = Array.from(dataMap.values())
      .map(item => {
        const { responseTimes, ...rest } = item as any;
        return {
          ...rest,
          avgResponseTime: responseTimes.length > 0 
            ? responseTimes.reduce((a: number, b: number) => a + b, 0) / responseTimes.length 
            : 0,
          time: format(new Date(item.time), getDateFormat(groupBy)),
        };
      })
      .sort((a, b) => a.timestamp - b.timestamp);

    // Fill in missing time buckets with zero values
    if (data.length > 0) {
      const filledData: ChartData[] = [];
      const startTime = data[0].timestamp;
      const endTime = data[data.length - 1].timestamp;
      let currentTime = startTime;

      const increment = getTimeIncrement(groupBy);

      while (currentTime <= endTime) {
        const existing = data.find(d => d.timestamp === currentTime);
        if (existing) {
          filledData.push(existing);
        } else {
          filledData.push({
            time: format(new Date(currentTime), getDateFormat(groupBy)),
            timestamp: currentTime,
            total: 0,
            successful: 0,
            failed: 0,
            avgResponseTime: 0,
          });
        }
        currentTime += increment;
      }

      return filledData;
    }

    return data;
  }, [executions, timeRange, groupBy, showComparison]);

  // Calculate statistics for annotations
  const statistics = useMemo(() => {
    if (chartData.length === 0) return null;

    const totals = chartData.map(d => d.total);
    const avgTotal = totals.reduce((a, b) => a + b, 0) / totals.length;
    const maxTotal = Math.max(...totals);
    const peakTime = chartData.find(d => d.total === maxTotal);

    return {
      average: avgTotal,
      peak: maxTotal,
      peakTime: peakTime?.time,
    };
  }, [chartData]);

  // Custom tooltip component
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload || !payload.length) return null;

    return (
      <Box
        sx={{
          backgroundColor: alpha(theme.palette.background.paper, 0.95),
          p: 2,
          borderRadius: 1,
          boxShadow: theme.shadows[3],
          border: `1px solid ${theme.palette.divider}`,
        }}
      >
        <Typography variant="subtitle2" gutterBottom>
          {label}
        </Typography>
        <Stack spacing={0.5}>
          {payload.map((entry: any, index: number) => (
            <Box key={index} display="flex" alignItems="center" gap={1}>
              <Box
                sx={{
                  width: 12,
                  height: 12,
                  borderRadius: '50%',
                  backgroundColor: entry.color,
                }}
              />
              <Typography variant="body2">
                {entry.name}: {entry.value}
              </Typography>
            </Box>
          ))}
        </Stack>
      </Box>
    );
  };

  // Render chart based on type
  const renderChart = () => {
    const ChartComponent = 
      chartType === 'line' ? LineChart :
      chartType === 'bar' ? BarChart :
      AreaChart;

    const DataComponent = 
      chartType === 'line' ? Line :
      chartType === 'bar' ? Bar :
      Area;

    return (
      <ResponsiveContainer width="100%" height={height}>
        <ChartComponent data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
          <XAxis 
            dataKey="time" 
            stroke={theme.palette.text.secondary}
            style={{ fontSize: 12 }}
          />
          <YAxis 
            stroke={theme.palette.text.secondary}
            style={{ fontSize: 12 }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend 
            wrapperStyle={{ paddingTop: '20px' }}
            iconType="circle"
          />

          {/* Main data series */}
          {!showComparison ? (
            <>
              <DataComponent
                type="monotone"
                dataKey="total"
                name="Total Executions"
                stroke={theme.palette.primary.main}
                fill={theme.palette.primary.main}
                fillOpacity={0.8}
                strokeWidth={2}
              />
              <DataComponent
                type="monotone"
                dataKey="successful"
                name="Successful"
                stroke={theme.palette.success.main}
                fill={theme.palette.success.main}
                fillOpacity={0.6}
                strokeWidth={2}
              />
              <DataComponent
                type="monotone"
                dataKey="failed"
                name="Failed"
                stroke={theme.palette.error.main}
                fill={theme.palette.error.main}
                fillOpacity={0.6}
                strokeWidth={2}
              />
            </>
          ) : (
            // Tool comparison mode
            selectedToolIds.map((toolId, index) => (
              <DataComponent
                key={toolId}
                type="monotone"
                dataKey={toolId}
                name={toolId}
                stroke={getToolColor(index)}
                fill={getToolColor(index)}
                fillOpacity={0.6}
                strokeWidth={2}
              />
            ))
          )}

          {/* Annotations */}
          {showAnnotations && statistics && (
            <>
              <ReferenceLine
                y={statistics.average}
                stroke={theme.palette.info.main}
                strokeDasharray="5 5"
                label={{ value: `Avg: ${statistics.average.toFixed(0)}`, position: 'right' }}
              />
              {statistics.peak > statistics.average * 2 && (
                <ReferenceArea
                  y1={statistics.average * 2}
                  y2={statistics.peak}
                  fill={theme.palette.warning.main}
                  fillOpacity={0.1}
                  label="Peak Usage"
                />
              )}
            </>
          )}

          {/* Brush for zooming */}
          {showBrush && chartData.length > 20 && (
            <Brush
              dataKey="time"
              height={30}
              stroke={theme.palette.primary.main}
              fill={alpha(theme.palette.primary.main, 0.1)}
            />
          )}
        </ChartComponent>
      </ResponsiveContainer>
    );
  };

  // Helper function to get date format based on grouping
  function getDateFormat(grouping: string): string {
    switch (grouping) {
      case 'minute':
        return 'HH:mm';
      case 'hour':
        return 'MMM dd HH:00';
      case 'day':
        return 'MMM dd';
      case 'week':
        return 'MMM dd';
      default:
        return 'MMM dd HH:00';
    }
  }

  // Helper function to get time increment
  function getTimeIncrement(grouping: string): number {
    switch (grouping) {
      case 'minute':
        return 60 * 1000;
      case 'hour':
        return 60 * 60 * 1000;
      case 'day':
        return 24 * 60 * 60 * 1000;
      case 'week':
        return 7 * 24 * 60 * 60 * 1000;
      default:
        return 60 * 60 * 1000;
    }
  }

  // Helper function to get tool colors
  function getToolColor(index: number): string {
    const colors = [
      theme.palette.primary.main,
      theme.palette.secondary.main,
      theme.palette.success.main,
      theme.palette.warning.main,
      theme.palette.info.main,
      theme.palette.error.main,
    ];
    return colors[index % colors.length];
  }

  return (
    <Box>
      {/* Chart controls */}
      <Stack direction="row" spacing={2} mb={2} alignItems="center">
        <FormControl size="small" sx={{ minWidth: 120 }}>
          <InputLabel>Group By</InputLabel>
          <Select value={groupBy} label="Group By">
            <MenuItem value="minute">Minute</MenuItem>
            <MenuItem value="hour">Hour</MenuItem>
            <MenuItem value="day">Day</MenuItem>
            <MenuItem value="week">Week</MenuItem>
          </Select>
        </FormControl>

        <FormControl size="small" sx={{ minWidth: 120 }}>
          <InputLabel>Chart Type</InputLabel>
          <Select value={chartType} label="Chart Type">
            <MenuItem value="line">Line</MenuItem>
            <MenuItem value="area">Area</MenuItem>
            <MenuItem value="bar">Bar</MenuItem>
          </Select>
        </FormControl>

        {statistics && (
          <Stack direction="row" spacing={1} ml="auto">
            <Chip
              label={`Avg: ${statistics.average.toFixed(0)}/period`}
              size="small"
              color="info"
              variant="outlined"
            />
            <Chip
              label={`Peak: ${statistics.peak} at ${statistics.peakTime}`}
              size="small"
              color="warning"
              variant="outlined"
            />
          </Stack>
        )}
      </Stack>

      {/* Chart */}
      {chartData.length > 0 ? (
        renderChart()
      ) : (
        <Box
          display="flex"
          alignItems="center"
          justifyContent="center"
          height={height}
          bgcolor={alpha(theme.palette.action.hover, 0.05)}
          borderRadius={1}
        >
          <Typography variant="body2" color="text.secondary">
            No data available for the selected time range
          </Typography>
        </Box>
      )}
    </Box>
  );
};

export default UsageChart;