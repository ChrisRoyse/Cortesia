import React from 'react';
import { Box, Grid, Typography, Paper, useTheme } from '@mui/material';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

interface PerformanceData {
  cpu: number;
  memory: number;
  latency: number;
  throughput: number;
}

interface Props {
  performance?: PerformanceData;
}

export const PerformanceMetricsCard: React.FC<Props> = ({ performance }) => {
  const theme = useTheme();

  // Generate mock time series data
  const generateTimeSeriesData = (baseValue: number) => {
    return Array.from({ length: 20 }, (_, i) => 
      baseValue + (Math.random() - 0.5) * 20
    );
  };

  const chartData = {
    labels: Array.from({ length: 20 }, (_, i) => `${i * 3}s`),
    datasets: [
      {
        label: 'CPU Usage',
        data: performance ? generateTimeSeriesData(performance.cpu) : [],
        borderColor: theme.palette.primary.main,
        backgroundColor: theme.palette.primary.main + '20',
        fill: true,
        tension: 0.4
      },
      {
        label: 'Memory Usage',
        data: performance ? generateTimeSeriesData(performance.memory) : [],
        borderColor: theme.palette.secondary.main,
        backgroundColor: theme.palette.secondary.main + '20',
        fill: true,
        tension: 0.4
      }
    ]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          color: theme.palette.text.primary
        }
      },
      tooltip: {
        mode: 'index' as const,
        intersect: false,
      }
    },
    scales: {
      x: {
        grid: {
          color: theme.palette.divider
        },
        ticks: {
          color: theme.palette.text.secondary
        }
      },
      y: {
        grid: {
          color: theme.palette.divider
        },
        ticks: {
          color: theme.palette.text.secondary,
          callback: function(value: any) {
            return value + '%';
          }
        },
        max: 100,
        min: 0
      }
    }
  };

  if (!performance) {
    return (
      <Box sx={{ p: 3, textAlign: 'center' }} data-testid="performance-no-data">
        <Typography color="text.secondary">
          No performance data available
        </Typography>
      </Box>
    );
  }

  return (
    <Grid container spacing={3} data-testid="performance-metrics-card">
      <Grid item xs={12} md={8}>
        <Box sx={{ height: 300 }} data-testid="performance-chart-container">
          <Line data={chartData} options={chartOptions} data-testid="performance-chart" />
        </Box>
      </Grid>
      
      <Grid item xs={12} md={4}>
        <Grid container spacing={2} data-testid="performance-metrics-grid">
          <Grid item xs={6} md={12}>
            <Paper sx={{ p: 2, bgcolor: 'background.default' }} data-testid="cpu-usage-card">
              <Typography variant="caption" color="text.secondary">
                CPU Usage
              </Typography>
              <Typography variant="h4" fontWeight="bold" color="primary.main" data-testid="cpu-metric">
                {performance.cpu.toFixed(1)}%
              </Typography>
            </Paper>
          </Grid>
          
          <Grid item xs={6} md={12}>
            <Paper sx={{ p: 2, bgcolor: 'background.default' }} data-testid="memory-usage-card">
              <Typography variant="caption" color="text.secondary">
                Memory Usage
              </Typography>
              <Typography variant="h4" fontWeight="bold" color="secondary.main" data-testid="memory-metric">
                {performance.memory.toFixed(1)}%
              </Typography>
            </Paper>
          </Grid>
          
          <Grid item xs={6} md={12}>
            <Paper sx={{ p: 2, bgcolor: 'background.default' }} data-testid="latency-card">
              <Typography variant="caption" color="text.secondary">
                Average Latency
              </Typography>
              <Typography variant="h4" fontWeight="bold" color="warning.main" data-testid="latency-metric">
                {performance.latency.toFixed(0)}ms
              </Typography>
            </Paper>
          </Grid>
          
          <Grid item xs={6} md={12}>
            <Paper sx={{ p: 2, bgcolor: 'background.default' }} data-testid="throughput-card">
              <Typography variant="caption" color="text.secondary">
                Throughput
              </Typography>
              <Typography variant="h4" fontWeight="bold" color="success.main" data-testid="throughput-metric">
                {performance.throughput.toFixed(0)}/s
              </Typography>
            </Paper>
          </Grid>
        </Grid>
      </Grid>
    </Grid>
  );
};