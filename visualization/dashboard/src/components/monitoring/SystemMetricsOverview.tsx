import React, { useMemo } from 'react';
import { Box, Grid, Paper, Typography, Card, CardContent, LinearProgress, Chip, Alert } from '@mui/material';
import { Speed, Memory, Storage, AccessTime, TrendingUp, Warning, CheckCircle } from '@mui/icons-material';
import { Line, Doughnut } from 'react-chartjs-2';
import { useTheme } from '@mui/material/styles';
import { MetricCard } from '../common/MetricCard';
import { StatusIndicator } from '../common/StatusIndicator';

interface SystemMetricsOverviewProps {
  data: any;
}

export const SystemMetricsOverview: React.FC<SystemMetricsOverviewProps> = ({ data }) => {
  const theme = useTheme();

  if (!data) {
    return (
      <Box sx={{ p: 3, textAlign: 'center' }}>
        <Typography variant="h6" color="text.secondary">
          No metrics data available
        </Typography>
      </Box>
    );
  }

  const performance = data.performance || {};
  const metrics = (data as any).metrics || {};

  // System health score calculation
  const healthScore = useMemo(() => {
    const cpu = performance.cpu || 0;
    const memory = performance.memory || 0;
    const errorRate = 0; // Would come from real data
    
    const cpuScore = cpu < 70 ? 100 : cpu < 90 ? 50 : 0;
    const memScore = memory < 70 ? 100 : memory < 90 ? 50 : 0;
    const errorScore = errorRate < 0.01 ? 100 : errorRate < 0.05 ? 50 : 0;
    
    return Math.round((cpuScore + memScore + errorScore) / 3);
  }, [performance]);

  const healthStatus = healthScore > 80 ? 'healthy' : healthScore > 50 ? 'warning' : 'critical';

  return (
    <Box>
      {/* System Health Alert */}
      <Alert 
        severity={healthStatus === 'healthy' ? 'success' : healthStatus === 'warning' ? 'warning' : 'error'}
        icon={healthStatus === 'healthy' ? <CheckCircle /> : <Warning />}
        sx={{ mb: 3 }}
      >
        <strong>System Health: {healthScore}%</strong> - 
        {healthStatus === 'healthy' ? ' All systems operating normally' : 
         healthStatus === 'warning' ? ' Some metrics need attention' : 
         ' Critical issues detected'}
      </Alert>

      <Grid container spacing={3}>
        {/* Key Metrics */}
        <Grid item xs={12} md={3}>
          <MetricCard
            title="CPU Usage"
            value={`${Math.round(performance.cpu || 0)}%`}
            trend={performance.cpu > 50 ? 'up' : 'stable'}
            icon={<Speed />}
            status={performance.cpu > 90 ? 'critical' : performance.cpu > 70 ? 'warning' : 'success'}
          />
        </Grid>
        <Grid item xs={12} md={3}>
          <MetricCard
            title="Memory Usage"
            value={`${Math.round(performance.memory || 0)}%`}
            trend={performance.memory > 50 ? 'up' : 'stable'}
            icon={<Memory />}
            status={performance.memory > 90 ? 'critical' : performance.memory > 70 ? 'warning' : 'success'}
          />
        </Grid>
        <Grid item xs={12} md={3}>
          <MetricCard
            title="Query Latency"
            value={`${performance.latency || 0}ms`}
            trend="stable"
            icon={<AccessTime />}
            status={performance.latency > 100 ? 'warning' : 'success'}
          />
        </Grid>
        <Grid item xs={12} md={3}>
          <MetricCard
            title="Throughput"
            value={`${performance.throughput || 0} ops/s`}
            trend="up"
            icon={<TrendingUp />}
            status="success"
          />
        </Grid>

        {/* Brain Metrics Overview */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, height: 300 }}>
            <Typography variant="h6" gutterBottom>Brain Metrics</Typography>
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <Typography variant="body2" color="text.secondary">Entities</Typography>
                <Typography variant="h4">{metrics.brain_entity_count || 0}</Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2" color="text.secondary">Relationships</Typography>
                <Typography variant="h4">{metrics.brain_relationship_count || 0}</Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2" color="text.secondary">Active Entities</Typography>
                <Typography variant="h5">{metrics.brain_active_entities || 0}</Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2" color="text.secondary">Avg Activation</Typography>
                <Typography variant="h5">{(metrics.brain_avg_activation || 0).toFixed(3)}</Typography>
              </Grid>
              <Grid item xs={12}>
                <Box sx={{ mt: 2 }}>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Graph Density
                  </Typography>
                  <LinearProgress 
                    variant="determinate" 
                    value={(metrics.brain_graph_density || 0) * 100}
                    sx={{ height: 10, borderRadius: 5 }}
                  />
                </Box>
              </Grid>
            </Grid>
          </Paper>
        </Grid>

        {/* System Resources */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, height: 300 }}>
            <Typography variant="h6" gutterBottom>Resource Allocation</Typography>
            <Doughnut
              data={{
                labels: ['Brain Graph', 'Embeddings', 'Indexes', 'Free'],
                datasets: [{
                  data: [
                    metrics.brain_memory_bytes || 0,
                    metrics.brain_embedding_memory_bytes || 0,
                    metrics.brain_index_memory_bytes || 0,
                    1000000 // placeholder for free memory
                  ],
                  backgroundColor: [
                    theme.palette.primary.main,
                    theme.palette.secondary.main,
                    theme.palette.warning.main,
                    theme.palette.grey[300]
                  ]
                }]
              }}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                  legend: {
                    position: 'right' as const,
                  }
                }
              }}
            />
          </Paper>
        </Grid>

        {/* Recent Activity */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>System Activity</Typography>
            <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
              <StatusIndicator 
                status={healthStatus === 'healthy' ? 'online' : healthStatus === 'warning' ? 'warning' : 'error'}
                label="Overall Health"
                size="large"
              />
              <StatusIndicator 
                status={performance.cpu < 90 ? 'online' : 'error'}
                label="CPU Status"
              />
              <StatusIndicator 
                status={performance.memory < 90 ? 'online' : 'error'}
                label="Memory Status"
              />
              <StatusIndicator 
                status="online"
                label="API Status"
              />
              <StatusIndicator 
                status="online"
                label="WebSocket"
              />
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};