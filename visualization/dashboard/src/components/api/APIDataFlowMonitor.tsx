import React, { useState, useEffect, useMemo } from 'react';
import { Box, Paper, Typography, Grid, Card, CardContent, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Chip, LinearProgress, Alert, IconButton, Tooltip } from '@mui/material';
import { Refresh, CheckCircle, Error, Warning, TrendingUp, TrendingDown } from '@mui/icons-material';
import { Line, Bar } from 'react-chartjs-2';
import { useTheme } from '@mui/material/styles';

interface APIEndpoint {
  path: string;
  method: string;
  status: 'healthy' | 'degraded' | 'down';
  requestRate: number; // requests per second
  errorRate: number; // percentage
  avgResponseTime: number; // ms
  p95ResponseTime: number; // ms
  p99ResponseTime: number; // ms
  lastError?: string;
  lastChecked: number;
}

interface DataFlow {
  source: string;
  destination: string;
  volume: number; // bytes per second
  latency: number; // ms
  status: 'active' | 'idle' | 'error';
}

export const APIDataFlowMonitor: React.FC = () => {
  const theme = useTheme();
  const [selectedEndpoint, setSelectedEndpoint] = useState<string | null>(null);

  // Mock data - in real implementation, this would come from WebSocket
  const endpoints: APIEndpoint[] = [
    {
      path: '/api/v1/triple',
      method: 'POST',
      status: 'healthy',
      requestRate: 45.2,
      errorRate: 0.02,
      avgResponseTime: 23,
      p95ResponseTime: 45,
      p99ResponseTime: 120,
      lastChecked: Date.now()
    },
    {
      path: '/api/v1/query',
      method: 'POST',
      status: 'healthy',
      requestRate: 120.5,
      errorRate: 0.01,
      avgResponseTime: 15,
      p95ResponseTime: 30,
      p99ResponseTime: 85,
      lastChecked: Date.now()
    },
    {
      path: '/api/v1/search',
      method: 'POST',
      status: 'degraded',
      requestRate: 30.1,
      errorRate: 2.5,
      avgResponseTime: 150,
      p95ResponseTime: 400,
      p99ResponseTime: 800,
      lastError: 'High latency detected',
      lastChecked: Date.now()
    },
    {
      path: '/api/v1/entity',
      method: 'POST',
      status: 'healthy',
      requestRate: 25.7,
      errorRate: 0.0,
      avgResponseTime: 18,
      p95ResponseTime: 35,
      p99ResponseTime: 90,
      lastChecked: Date.now()
    },
    {
      path: '/api/v1/metrics',
      method: 'GET',
      status: 'healthy',
      requestRate: 5.2,
      errorRate: 0.0,
      avgResponseTime: 8,
      p95ResponseTime: 12,
      p99ResponseTime: 25,
      lastChecked: Date.now()
    }
  ];

  const dataFlows: DataFlow[] = [
    { source: 'API Gateway', destination: 'Entity Processor', volume: 1024 * 50, latency: 5, status: 'active' },
    { source: 'Entity Processor', destination: 'Brain Graph', volume: 1024 * 40, latency: 10, status: 'active' },
    { source: 'Brain Graph', destination: 'Index Storage', volume: 1024 * 30, latency: 15, status: 'active' },
    { source: 'Query Engine', destination: 'Brain Graph', volume: 1024 * 100, latency: 8, status: 'active' },
    { source: 'Brain Graph', destination: 'Response Builder', volume: 1024 * 80, latency: 3, status: 'active' }
  ];

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy': return <CheckCircle color="success" fontSize="small" />;
      case 'degraded': return <Warning color="warning" fontSize="small" />;
      case 'down': return <Error color="error" fontSize="small" />;
      default: return null;
    }
  };

  const totalRequestRate = endpoints.reduce((sum, ep) => sum + ep.requestRate, 0);
  const avgErrorRate = endpoints.reduce((sum, ep) => sum + ep.errorRate, 0) / endpoints.length;

  return (
    <Box>
      <Grid container spacing={3}>
        {/* API Health Overview */}
        <Grid item xs={12}>
          <Alert 
            severity={endpoints.some(ep => ep.status === 'down') ? 'error' : 
                    endpoints.some(ep => ep.status === 'degraded') ? 'warning' : 'success'}
            sx={{ mb: 2 }}
          >
            API Health: {endpoints.filter(ep => ep.status === 'healthy').length}/{endpoints.length} endpoints healthy
          </Alert>
        </Grid>

        {/* Key Metrics */}
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>Total Request Rate</Typography>
              <Typography variant="h4">{totalRequestRate.toFixed(1)}</Typography>
              <Typography variant="body2" color="text.secondary">requests/sec</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>Average Error Rate</Typography>
              <Typography variant="h4" color={avgErrorRate > 1 ? 'error' : 'inherit'}>
                {avgErrorRate.toFixed(2)}%
              </Typography>
              <Typography variant="body2" color="text.secondary">across all endpoints</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>Data Throughput</Typography>
              <Typography variant="h4">
                {(dataFlows.reduce((sum, df) => sum + df.volume, 0) / 1024).toFixed(1)}
              </Typography>
              <Typography variant="body2" color="text.secondary">KB/sec</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>Active Flows</Typography>
              <Typography variant="h4">{dataFlows.filter(df => df.status === 'active').length}</Typography>
              <Typography variant="body2" color="text.secondary">of {dataFlows.length} total</Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Endpoint Status Table */}
        <Grid item xs={12} lg={7}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>API Endpoint Status</Typography>
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Endpoint</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell align="right">Req/s</TableCell>
                    <TableCell align="right">Error %</TableCell>
                    <TableCell align="right">Avg (ms)</TableCell>
                    <TableCell align="right">P95 (ms)</TableCell>
                    <TableCell align="right">P99 (ms)</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {endpoints.map((endpoint) => (
                    <TableRow 
                      key={`${endpoint.method}-${endpoint.path}`}
                      hover
                      selected={selectedEndpoint === endpoint.path}
                      onClick={() => setSelectedEndpoint(endpoint.path)}
                      sx={{ cursor: 'pointer' }}
                    >
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Chip label={endpoint.method} size="small" />
                          {endpoint.path}
                        </Box>
                      </TableCell>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          {getStatusIcon(endpoint.status)}
                          {endpoint.status}
                        </Box>
                      </TableCell>
                      <TableCell align="right">{endpoint.requestRate.toFixed(1)}</TableCell>
                      <TableCell align="right">
                        <Typography color={endpoint.errorRate > 1 ? 'error' : 'inherit'}>
                          {endpoint.errorRate.toFixed(2)}
                        </Typography>
                      </TableCell>
                      <TableCell align="right">{endpoint.avgResponseTime}</TableCell>
                      <TableCell align="right">{endpoint.p95ResponseTime}</TableCell>
                      <TableCell align="right">{endpoint.p99ResponseTime}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </Grid>

        {/* Data Flow Visualization */}
        <Grid item xs={12} lg={5}>
          <Paper sx={{ p: 2, height: 400 }}>
            <Typography variant="h6" gutterBottom>Data Flow Pipeline</Typography>
            <Box sx={{ height: 'calc(100% - 40px)', position: 'relative' }}>
              {/* Simplified flow visualization */}
              {dataFlows.map((flow, index) => (
                <Box key={index} sx={{ mb: 2 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                    <Typography variant="body2">{flow.source} → {flow.destination}</Typography>
                    <Chip 
                      label={`${(flow.volume / 1024).toFixed(1)} KB/s`}
                      size="small"
                      color={flow.status === 'active' ? 'success' : 'default'}
                    />
                  </Box>
                  <LinearProgress 
                    variant="determinate" 
                    value={(flow.volume / (1024 * 100)) * 100}
                    sx={{ height: 8, borderRadius: 4 }}
                  />
                  <Typography variant="caption" color="text.secondary">
                    Latency: {flow.latency}ms
                  </Typography>
                </Box>
              ))}
            </Box>
          </Paper>
        </Grid>

        {/* Response Time Distribution */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2, height: 300 }}>
            <Typography variant="h6" gutterBottom>Response Time Distribution</Typography>
            <Bar
              data={{
                labels: endpoints.map(ep => ep.path.replace('/api/v1/', '')),
                datasets: [
                  {
                    label: 'Average',
                    data: endpoints.map(ep => ep.avgResponseTime),
                    backgroundColor: theme.palette.primary.light,
                  },
                  {
                    label: 'P95',
                    data: endpoints.map(ep => ep.p95ResponseTime),
                    backgroundColor: theme.palette.warning.light,
                  },
                  {
                    label: 'P99',
                    data: endpoints.map(ep => ep.p99ResponseTime),
                    backgroundColor: theme.palette.error.light,
                  }
                ]
              }}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                  y: {
                    beginAtZero: true,
                    title: {
                      display: true,
                      text: 'Response Time (ms)'
                    }
                  }
                },
                plugins: {
                  legend: {
                    position: 'top' as const,
                  }
                }
              }}
            />
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};