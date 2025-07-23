import React from 'react';
import { Card, CardContent, Box, Typography, CircularProgress } from '@mui/material';
import { CheckCircle, Warning, Error as ErrorIcon } from '@mui/icons-material';
import { LLMKGData } from '../types';

interface Props {
  data: LLMKGData | null;
}

export const SystemHealthIndicator: React.FC<Props> = ({ data }) => {
  const calculateHealth = () => {
    if (!data) return 0;
    
    let score = 100;
    
    // Check performance metrics
    if (data.performance) {
      if (data.performance.cpu > 80) score -= 20;
      if (data.performance.memory > 80) score -= 20;
      if (data.performance.latency > 100) score -= 10;
    }
    
    // Check neural activity
    if (data.neural) {
      const avgActivity = data.neural.activity.reduce((a, b) => a + b, 0) / data.neural.activity.length;
      if (avgActivity < 20) score -= 15;
    }
    
    return Math.max(0, score);
  };

  const health = calculateHealth();
  
  const getHealthStatus = () => {
    if (health >= 80) return { label: 'Healthy', color: 'success.main', icon: CheckCircle };
    if (health >= 60) return { label: 'Warning', color: 'warning.main', icon: Warning };
    return { label: 'Critical', color: 'error.main', icon: ErrorIcon };
  };

  const status = getHealthStatus();
  const StatusIcon = status.icon;

  return (
    <Card elevation={2} sx={{ height: '100%' }} data-testid="system-health-indicator">
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Box>
            <Typography color="text.secondary" variant="body2" data-testid="health-status-text">
              System Health
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 1 }}>
              <StatusIcon sx={{ color: status.color }} data-testid="health-status-icon" />
              <Typography variant="h6" sx={{ color: status.color, fontWeight: 600 }} data-testid="health-status-label">
                {status.label}
              </Typography>
            </Box>
          </Box>
          <Box sx={{ position: 'relative', display: 'inline-flex' }}>
            <CircularProgress
              variant="determinate"
              value={health}
              size={60}
              thickness={4}
              data-testid="health-progress-circle"
              sx={{
                color: status.color,
                [`& .MuiCircularProgress-circle`]: {
                  strokeLinecap: 'round',
                },
              }}
            />
            <Box
              sx={{
                top: 0,
                left: 0,
                bottom: 0,
                right: 0,
                position: 'absolute',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <Typography variant="h6" component="div" color="text.primary" data-testid="health-percentage">
                {health}%
              </Typography>
            </Box>
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};