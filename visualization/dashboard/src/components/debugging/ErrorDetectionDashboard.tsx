import React, { useState, useMemo } from 'react';
import { Box, Paper, Typography, Grid, Card, CardContent, Alert, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Chip, IconButton, Collapse, Button, TextField, MenuItem, Select, FormControl, InputLabel } from '@mui/material';
import { Error, Warning, Info, ExpandMore, ExpandLess, BugReport, Timeline, FilterList } from '@mui/icons-material';
import { Line } from 'react-chartjs-2';
import { useTheme } from '@mui/material/styles';

interface ErrorEvent {
  id: string;
  timestamp: number;
  level: 'error' | 'warning' | 'info';
  category: 'api' | 'brain' | 'memory' | 'network' | 'validation';
  message: string;
  details?: string;
  stackTrace?: string;
  affectedEntities?: string[];
  resolved: boolean;
}

interface PerformanceAnomaly {
  id: string;
  type: 'latency_spike' | 'memory_leak' | 'cpu_spike' | 'throughput_drop';
  severity: 'low' | 'medium' | 'high';
  detected: number;
  description: string;
  metrics: Record<string, number>;
}

interface GraphHealthIssue {
  type: 'disconnected_component' | 'circular_dependency' | 'orphaned_entity' | 'invalid_relationship';
  count: number;
  entities: string[];
  severity: 'low' | 'medium' | 'high';
}

export const ErrorDetectionDashboard: React.FC = () => {
  const theme = useTheme();
  const [expandedError, setExpandedError] = useState<string | null>(null);
  const [errorFilter, setErrorFilter] = useState<string>('all');
  const [timeRange, setTimeRange] = useState(3600); // 1 hour

  // Mock data
  const errors: ErrorEvent[] = [
    {
      id: '1',
      timestamp: Date.now() - 300000,
      level: 'error',
      category: 'api',
      message: 'Failed to process entity: Invalid embedding dimension',
      details: 'Expected 384 dimensions, received 256',
      stackTrace: 'at EntityProcessor.validateEmbedding()\n  at EntityProcessor.process()\n  at APIHandler.handleEntity()',
      affectedEntities: ['entity_123', 'entity_124'],
      resolved: false
    },
    {
      id: '2',
      timestamp: Date.now() - 600000,
      level: 'warning',
      category: 'brain',
      message: 'Activation anomaly detected: Stuck at maximum',
      details: 'Entity has been at activation 1.0 for 5 minutes',
      affectedEntities: ['entity_789'],
      resolved: true
    },
    {
      id: '3',
      timestamp: Date.now() - 900000,
      level: 'error',
      category: 'memory',
      message: 'Memory consolidation failed: Buffer overflow',
      details: 'Working memory buffer exceeded capacity during consolidation',
      resolved: false
    }
  ];

  const anomalies: PerformanceAnomaly[] = [
    {
      id: 'a1',
      type: 'latency_spike',
      severity: 'high',
      detected: Date.now() - 1800000,
      description: 'Query latency increased by 300%',
      metrics: { normal: 50, current: 200, threshold: 100 }
    },
    {
      id: 'a2',
      type: 'memory_leak',
      severity: 'medium',
      detected: Date.now() - 3600000,
      description: 'Memory usage growing without corresponding entity growth',
      metrics: { growthRate: 5.2, expectedRate: 1.0 }
    }
  ];

  const graphHealthIssues: GraphHealthIssue[] = [
    {
      type: 'disconnected_component',
      count: 3,
      entities: ['cluster_1', 'cluster_2', 'cluster_3'],
      severity: 'medium'
    },
    {
      type: 'orphaned_entity',
      count: 12,
      entities: ['entity_901', 'entity_902', '...'],
      severity: 'low'
    }
  ];

  const filteredErrors = useMemo(() => {
    if (errorFilter === 'all') return errors;
    if (errorFilter === 'unresolved') return errors.filter(e => !e.resolved);
    return errors.filter(e => e.category === errorFilter);
  }, [errors, errorFilter]);

  const errorTimeline = useMemo(() => {
    const buckets = new Map<number, number>();
    const now = Date.now();
    const bucketSize = 300000; // 5 minutes

    errors.forEach(error => {
      const bucket = Math.floor((now - error.timestamp) / bucketSize) * bucketSize;
      buckets.set(bucket, (buckets.get(bucket) || 0) + 1);
    });

    const sortedBuckets = Array.from(buckets.entries()).sort((a, b) => b[0] - a[0]).slice(0, 12);
    
    return {
      labels: sortedBuckets.map(([time]) => new Date(now - time).toLocaleTimeString()),
      datasets: [{
        label: 'Errors',
        data: sortedBuckets.map(([, count]) => count),
        borderColor: theme.palette.error.main,
        backgroundColor: theme.palette.error.light,
        tension: 0.4
      }]
    };
  }, [errors, theme]);

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'high': return 'error';
      case 'medium': return 'warning';
      case 'low': return 'info';
      default: return 'default';
    }
  };

  return (
    <Box>
      <Grid container spacing={3}>
        {/* Error Summary */}
        <Grid item xs={12}>
          <Grid container spacing={2}>
            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Error color="error" />
                    <Box>
                      <Typography color="text.secondary" variant="body2">Total Errors</Typography>
                      <Typography variant="h4">{errors.filter(e => e.level === 'error').length}</Typography>
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Warning color="warning" />
                    <Box>
                      <Typography color="text.secondary" variant="body2">Warnings</Typography>
                      <Typography variant="h4">{errors.filter(e => e.level === 'warning').length}</Typography>
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <BugReport color="error" />
                    <Box>
                      <Typography color="text.secondary" variant="body2">Unresolved</Typography>
                      <Typography variant="h4">{errors.filter(e => !e.resolved).length}</Typography>
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Timeline color="primary" />
                    <Box>
                      <Typography color="text.secondary" variant="body2">Anomalies</Typography>
                      <Typography variant="h4">{anomalies.length}</Typography>
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Grid>

        {/* Error Timeline */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, height: 300 }}>
            <Typography variant="h6" gutterBottom>Error Timeline</Typography>
            <Line
              data={errorTimeline}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                  y: {
                    beginAtZero: true,
                    ticks: { stepSize: 1 }
                  }
                }
              }}
            />
          </Paper>
        </Grid>

        {/* Performance Anomalies */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, height: 300 }}>
            <Typography variant="h6" gutterBottom>Performance Anomalies</Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              {anomalies.map(anomaly => (
                <Alert 
                  key={anomaly.id} 
                  severity={getSeverityColor(anomaly.severity) as any}
                  onClose={() => {}}
                >
                  <Typography variant="subtitle2">{anomaly.description}</Typography>
                  <Typography variant="body2">
                    Detected: {new Date(anomaly.detected).toLocaleString()}
                  </Typography>
                  <Box sx={{ mt: 1 }}>
                    {Object.entries(anomaly.metrics).map(([key, value]) => (
                      <Chip 
                        key={key} 
                        label={`${key}: ${value}`} 
                        size="small" 
                        sx={{ mr: 0.5 }}
                      />
                    ))}
                  </Box>
                </Alert>
              ))}
            </Box>
          </Paper>
        </Grid>

        {/* Error Log */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">Error Log</Typography>
              <FormControl size="small" sx={{ minWidth: 150 }}>
                <InputLabel>Filter</InputLabel>
                <Select
                  value={errorFilter}
                  onChange={(e) => setErrorFilter(e.target.value)}
                  label="Filter"
                >
                  <MenuItem value="all">All</MenuItem>
                  <MenuItem value="unresolved">Unresolved</MenuItem>
                  <MenuItem value="api">API</MenuItem>
                  <MenuItem value="brain">Brain</MenuItem>
                  <MenuItem value="memory">Memory</MenuItem>
                  <MenuItem value="network">Network</MenuItem>
                </Select>
              </FormControl>
            </Box>
            
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell width={40}></TableCell>
                    <TableCell>Time</TableCell>
                    <TableCell>Level</TableCell>
                    <TableCell>Category</TableCell>
                    <TableCell>Message</TableCell>
                    <TableCell>Status</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {filteredErrors.map(error => (
                    <React.Fragment key={error.id}>
                      <TableRow hover>
                        <TableCell>
                          <IconButton
                            size="small"
                            onClick={() => setExpandedError(expandedError === error.id ? null : error.id)}
                          >
                            {expandedError === error.id ? <ExpandLess /> : <ExpandMore />}
                          </IconButton>
                        </TableCell>
                        <TableCell>{new Date(error.timestamp).toLocaleString()}</TableCell>
                        <TableCell>
                          <Chip 
                            label={error.level} 
                            size="small" 
                            color={error.level === 'error' ? 'error' : error.level === 'warning' ? 'warning' : 'info'}
                          />
                        </TableCell>
                        <TableCell>{error.category}</TableCell>
                        <TableCell>{error.message}</TableCell>
                        <TableCell>
                          <Chip 
                            label={error.resolved ? 'Resolved' : 'Open'} 
                            size="small"
                            color={error.resolved ? 'success' : 'default'}
                          />
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell colSpan={6} sx={{ py: 0 }}>
                          <Collapse in={expandedError === error.id}>
                            <Box sx={{ p: 2 }}>
                              {error.details && (
                                <Typography variant="body2" paragraph>
                                  <strong>Details:</strong> {error.details}
                                </Typography>
                              )}
                              {error.stackTrace && (
                                <Box sx={{ bgcolor: 'grey.100', p: 1, borderRadius: 1, mb: 1 }}>
                                  <Typography variant="body2" component="pre" sx={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>
                                    {error.stackTrace}
                                  </Typography>
                                </Box>
                              )}
                              {error.affectedEntities && (
                                <Box>
                                  <Typography variant="body2"><strong>Affected Entities:</strong></Typography>
                                  <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap', mt: 0.5 }}>
                                    {error.affectedEntities.map(entity => (
                                      <Chip key={entity} label={entity} size="small" />
                                    ))}
                                  </Box>
                                </Box>
                              )}
                            </Box>
                          </Collapse>
                        </TableCell>
                      </TableRow>
                    </React.Fragment>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </Grid>

        {/* Graph Health Monitor */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>Graph Health Issues</Typography>
            <Grid container spacing={2}>
              {graphHealthIssues.map((issue, index) => (
                <Grid item xs={12} md={4} key={index}>
                  <Alert severity={getSeverityColor(issue.severity) as any}>
                    <Typography variant="subtitle2">
                      {issue.type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                    </Typography>
                    <Typography variant="body2">
                      {issue.count} {issue.count === 1 ? 'instance' : 'instances'} found
                    </Typography>
                    <Box sx={{ mt: 1 }}>
                      <Button size="small">View Details</Button>
                    </Box>
                  </Alert>
                </Grid>
              ))}
            </Grid>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};