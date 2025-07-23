import React from 'react';
import { Line, Bar, Doughnut, Radar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  RadialLinearScale,
  Title,
  Tooltip,
  Legend,
  Filler,
  ChartOptions,
  ChartData
} from 'chart.js';
import { Box, useTheme } from '@mui/material';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  RadialLinearScale,
  Title,
  Tooltip,
  Legend,
  Filler
);

export interface ChartProps {
  type: 'line' | 'bar' | 'doughnut' | 'radar';
  data: ChartData<any>;
  options?: ChartOptions<any>;
  height?: number | string;
  width?: number | string;
}

export const Chart: React.FC<ChartProps> = ({
  type,
  data,
  options = {},
  height = 300,
  width = '100%'
}) => {
  const theme = useTheme();

  const defaultOptions: ChartOptions<any> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: true,
        position: 'top' as const,
        labels: {
          color: theme.palette.text.primary,
          font: {
            family: theme.typography.fontFamily,
            size: 12
          }
        }
      },
      tooltip: {
        backgroundColor: theme.palette.background.paper,
        titleColor: theme.palette.text.primary,
        bodyColor: theme.palette.text.secondary,
        borderColor: theme.palette.divider,
        borderWidth: 1,
        cornerRadius: 4,
        padding: 8
      }
    },
    scales: type === 'line' || type === 'bar' ? {
      x: {
        grid: {
          color: theme.palette.divider,
          display: true
        },
        ticks: {
          color: theme.palette.text.secondary,
          font: {
            size: 11
          }
        }
      },
      y: {
        grid: {
          color: theme.palette.divider,
          display: true
        },
        ticks: {
          color: theme.palette.text.secondary,
          font: {
            size: 11
          }
        }
      }
    } : undefined
  };

  const mergedOptions = {
    ...defaultOptions,
    ...options
  };

  const ChartComponent = {
    line: Line,
    bar: Bar,
    doughnut: Doughnut,
    radar: Radar
  }[type];

  return (
    <Box sx={{ height, width, position: 'relative' }}>
      <ChartComponent data={data} options={mergedOptions} />
    </Box>
  );
};