import React from 'react';
import { Box, LinearProgress, Typography, useTheme } from '@mui/material';

export interface ProgressBarProps {
  value: number;
  max?: number;
  label?: string;
  showPercentage?: boolean;
  color?: 'primary' | 'secondary' | 'error' | 'warning' | 'info' | 'success';
  variant?: 'determinate' | 'indeterminate' | 'buffer' | 'query';
  height?: number;
  animated?: boolean;
}

export const ProgressBar: React.FC<ProgressBarProps> = ({
  value,
  max = 100,
  label,
  showPercentage = true,
  color = 'primary',
  variant = 'determinate',
  height = 8,
  animated = true
}) => {
  const theme = useTheme();
  const percentage = Math.round((value / max) * 100);

  return (
    <Box sx={{ width: '100%' }}>
      {(label || showPercentage) && (
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
          {label && (
            <Typography variant="body2" color="text.secondary">
              {label}
            </Typography>
          )}
          {showPercentage && variant === 'determinate' && (
            <Typography variant="body2" color="text.secondary">
              {percentage}%
            </Typography>
          )}
        </Box>
      )}
      <LinearProgress
        variant={variant}
        value={percentage}
        color={color}
        sx={{
          height,
          borderRadius: height / 2,
          backgroundColor: theme.palette.action.hover,
          '& .MuiLinearProgress-bar': {
            borderRadius: height / 2,
            transition: animated ? 'transform 0.4s ease' : 'none'
          }
        }}
      />
    </Box>
  );
};