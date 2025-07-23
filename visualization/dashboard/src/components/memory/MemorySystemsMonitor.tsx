import React, { useState, useMemo, useEffect } from 'react';
import { Box, Paper, Typography, Grid, Card, CardContent, LinearProgress, Chip, IconButton, Tooltip, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Alert, CircularProgress, FormControl, Select, MenuItem, Button } from '@mui/material';
import { Line, Bar, Doughnut } from 'react-chartjs-2';
import { Memory, Speed, Storage, TrendingDown, Warning, CheckCircle, Refresh, GetApp, Timer, Layers } from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';
import * as d3 from 'd3';

interface MemoryBuffer {
  id: string;
  name: string;
  capacity: number;
  used: number;
  items: MemoryItem[];
  decayRate: number; // items per second
  accessPattern: 'sequential' | 'random' | 'spatial';
}

interface MemoryItem {
  id: string;
  content: any;
  size: number;
  timestamp: number;
  accessCount: number;
  lastAccess: number;
  strength: number; // 0-1
  decaying: boolean;
}

interface ConsolidationProcess {
  id: string;
  sourceBuffer: string;
  targetStore: string;
  itemsProcessed: number;
  totalItems: number;
  startTime: number;
  estimatedCompletion: number;
  status: 'running' | 'paused' | 'completed' | 'failed';
}

interface SDRPattern {
  id: string;
  bits: boolean[];
  sparsity: number;
  overlap: number;
  associations: string[];
}

interface MemorySystemsData {
  workingMemory: {
    buffers: MemoryBuffer[];
    totalCapacity: number;
    totalUsed: number;
  };
  longTermMemory: {
    consolidationRate: number;
    retrievalSpeed: number; // ms
    totalItems: number;
    indexSize: number; // bytes
    compressionRatio: number;
  };
  consolidation: {
    processes: ConsolidationProcess[];
    queue: string[]; // buffer IDs waiting for consolidation
  };
  sdr: {
    patterns: SDRPattern[];
    totalBits: number;
    averageSparsity: number;
  };
  zeroCopy: {
    enabled: boolean;
    mappedRegions: number;
    totalMappedSize: number; // bytes
    accessLatency: number; // microseconds
  };
  forgettingCurve: Array<{
    timestamp: number;
    retentionRate: number;
    itemsRetained: number;
  }>;
}

interface MemorySystemsMonitorProps {
  data: MemorySystemsData;
  onConsolidationControl?: (processId: string, action: 'pause' | 'resume' | 'cancel') => void;
  onBufferClear?: (bufferId: string) => void;
  height?: number | string;
}

export const MemorySystemsMonitor: React.FC<MemorySystemsMonitorProps> = ({
  data,
  onConsolidationControl,
  onBufferClear,
  height = 800
}) => {
  const theme = useTheme();
  const [selectedBuffer, setSelectedBuffer] = useState<string | null>(null);
  const [showSDRDetails, setShowSDRDetails] = useState(false);
  const [timeRange, setTimeRange] = useState(3600); // 1 hour

  // Working memory usage by buffer
  const bufferUsageData = useMemo(() => {
    const labels = data.workingMemory.buffers.map(b => b.name);
    const usage = data.workingMemory.buffers.map(b => (b.used / b.capacity) * 100);
    const colors = usage.map(u => {
      if (u > 90) return theme.palette.error.main;
      if (u > 70) return theme.palette.warning.main;
      return theme.palette.success.main;
    });

    return {
      labels,
      datasets: [{
        label: 'Buffer Usage %',
        data: usage,
        backgroundColor: colors,
        borderColor: colors,
        borderWidth: 1,
      }]
    };
  }, [data.workingMemory.buffers, theme]);

  // Consolidation progress
  const activeConsolidations = useMemo(() => 
    data.consolidation.processes.filter(p => p.status === 'running'),
    [data.consolidation.processes]
  );

  // Forgetting curve chart
  const forgettingCurveData = useMemo(() => {
    const cutoff = Date.now() - (timeRange * 1000);
    const recentData = data.forgettingCurve.filter(d => d.timestamp > cutoff);
    
    return {
      labels: recentData.map(d => new Date(d.timestamp).toLocaleTimeString()),
      datasets: [{
        label: 'Retention Rate',
        data: recentData.map(d => d.retentionRate * 100),
        borderColor: theme.palette.primary.main,
        backgroundColor: theme.palette.primary.light,
        tension: 0.4,
        yAxisID: 'y',
      }, {
        label: 'Items Retained',
        data: recentData.map(d => d.itemsRetained),
        borderColor: theme.palette.secondary.main,
        backgroundColor: theme.palette.secondary.light,
        tension: 0.4,
        yAxisID: 'y1',
      }]
    };
  }, [data.forgettingCurve, timeRange, theme]);

  // SDR visualization
  const renderSDRPattern = (pattern: SDRPattern, size: number = 100) => {
    const gridSize = Math.sqrt(pattern.bits.length);
    const cellSize = size / gridSize;

    return (
      <svg width={size} height={size} style={{ border: '1px solid #ccc' }}>
        {pattern.bits.map((bit, idx) => {
          const row = Math.floor(idx / gridSize);
          const col = idx % gridSize;
          return (
            <rect
              key={idx}
              x={col * cellSize}
              y={row * cellSize}
              width={cellSize}
              height={cellSize}
              fill={bit ? theme.palette.primary.main : theme.palette.grey[200]}
              stroke={theme.palette.grey[400]}
              strokeWidth={0.5}
            />
          );
        })}
      </svg>
    );
  };

  // Memory access heatmap
  const renderAccessHeatmap = (buffer: MemoryBuffer) => {
    const items = buffer.items.slice(0, 50); // Show first 50 items
    const maxAccess = Math.max(...items.map(i => i.accessCount));
    const colorScale = d3.scaleSequential(d3.interpolateOrRd)
      .domain([0, maxAccess]);

    return (
      <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
        {items.map(item => (
          <Tooltip key={item.id} title={`${item.id}: ${item.accessCount} accesses`}>
            <Box
              sx={{
                width: 20,
                height: 20,
                backgroundColor: colorScale(item.accessCount),
                border: '1px solid #ccc',
                cursor: 'pointer'
              }}
            />
          </Tooltip>
        ))}
      </Box>
    );
  };

  const selectedBufferData = useMemo(() => 
    data.workingMemory.buffers.find(b => b.id === selectedBuffer),
    [data.workingMemory.buffers, selectedBuffer]
  );

  return (
    <Box sx={{ height, display: 'flex', flexDirection: 'column', gap: 2 }}>
      <Grid container spacing={2}>
        {/* Working Memory Buffers */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, height: 400 }}>
            <Typography variant="h6" gutterBottom>Working Memory Buffers</Typography>
            
            <Box sx={{ mb: 2 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="body2">Total Usage</Typography>
                <Typography variant="body2">
                  {data.workingMemory.totalUsed} / {data.workingMemory.totalCapacity}
                </Typography>
              </Box>
              <LinearProgress
                variant="determinate"
                value={(data.workingMemory.totalUsed / data.workingMemory.totalCapacity) * 100}
                color={(data.workingMemory.totalUsed / data.workingMemory.totalCapacity) > 0.8 ? 'warning' : 'primary'}
              />
            </Box>

            <Bar
              data={bufferUsageData}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y' as const,
                scales: {
                  x: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                      display: true,
                      text: 'Usage %'
                    }
                  }
                },
                plugins: {
                  legend: {
                    display: false
                  }
                },
                onClick: (event, elements) => {
                  if (elements.length > 0) {
                    const index = elements[0].index;
                    setSelectedBuffer(data.workingMemory.buffers[index].id);
                  }
                }
              }}
              height={250}
            />

            {selectedBufferData && (
              <Box sx={{ mt: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  {selectedBufferData.name} Details
                </Typography>
                <Grid container spacing={1}>
                  <Grid item xs={4}>
                    <Typography variant="caption" color="text.secondary">Items</Typography>
                    <Typography variant="body2">{selectedBufferData.items.length}</Typography>
                  </Grid>
                  <Grid item xs={4}>
                    <Typography variant="caption" color="text.secondary">Decay Rate</Typography>
                    <Typography variant="body2">{selectedBufferData.decayRate}/s</Typography>
                  </Grid>
                  <Grid item xs={4}>
                    <Typography variant="caption" color="text.secondary">Pattern</Typography>
                    <Typography variant="body2">{selectedBufferData.accessPattern}</Typography>
                  </Grid>
                </Grid>
              </Box>
            )}
          </Paper>
        </Grid>

        {/* Long-term Memory Consolidation */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, height: 400 }}>
            <Typography variant="h6" gutterBottom>Memory Consolidation</Typography>
            
            <Box sx={{ mb: 2 }}>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Card variant="outlined">
                    <CardContent sx={{ p: 1.5 }}>
                      <Typography variant="body2" color="text.secondary">Consolidation Rate</Typography>
                      <Typography variant="h6">{data.longTermMemory.consolidationRate.toFixed(2)}/s</Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={6}>
                  <Card variant="outlined">
                    <CardContent sx={{ p: 1.5 }}>
                      <Typography variant="body2" color="text.secondary">Retrieval Speed</Typography>
                      <Typography variant="h6">{data.longTermMemory.retrievalSpeed}ms</Typography>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            </Box>

            <Typography variant="subtitle2" gutterBottom>Active Processes</Typography>
            <TableContainer sx={{ maxHeight: 200 }}>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Source</TableCell>
                    <TableCell>Progress</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {activeConsolidations.map(process => (
                    <TableRow key={process.id}>
                      <TableCell>{process.sourceBuffer}</TableCell>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <LinearProgress
                            variant="determinate"
                            value={(process.itemsProcessed / process.totalItems) * 100}
                            sx={{ flexGrow: 1 }}
                          />
                          <Typography variant="caption">
                            {Math.round((process.itemsProcessed / process.totalItems) * 100)}%
                          </Typography>
                        </Box>
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={process.status}
                          size="small"
                          color={process.status === 'running' ? 'success' : 'default'}
                        />
                      </TableCell>
                      <TableCell>
                        <IconButton
                          size="small"
                          onClick={() => onConsolidationControl?.(process.id, 'pause')}
                        >
                          <Timer />
                        </IconButton>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>

            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle2" gutterBottom>Consolidation Queue</Typography>
              <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                {data.consolidation.queue.map(bufferId => (
                  <Chip key={bufferId} label={bufferId} size="small" variant="outlined" />
                ))}
                {data.consolidation.queue.length === 0 && (
                  <Typography variant="body2" color="text.secondary">No items in queue</Typography>
                )}
              </Box>
            </Box>
          </Paper>
        </Grid>

        {/* SDR Visualization */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2, height: 350 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">SDR Patterns</Typography>
              <Button
                size="small"
                onClick={() => setShowSDRDetails(!showSDRDetails)}
              >
                {showSDRDetails ? 'Hide' : 'Show'} Details
              </Button>
            </Box>

            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" color="text.secondary">
                Average Sparsity: {(data.sdr.averageSparsity * 100).toFixed(1)}%
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Total Bits: {data.sdr.totalBits}
              </Typography>
            </Box>

            {showSDRDetails && (
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                {data.sdr.patterns.slice(0, 3).map(pattern => (
                  <Box key={pattern.id}>
                    <Typography variant="caption">{pattern.id}</Typography>
                    <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                      {renderSDRPattern(pattern, 80)}
                      <Box>
                        <Typography variant="caption" display="block">
                          Sparsity: {(pattern.sparsity * 100).toFixed(1)}%
                        </Typography>
                        <Typography variant="caption" display="block">
                          Overlap: {(pattern.overlap * 100).toFixed(1)}%
                        </Typography>
                      </Box>
                    </Box>
                  </Box>
                ))}
              </Box>
            )}

            {!showSDRDetails && (
              <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 200 }}>
                <Layers sx={{ fontSize: 100, color: theme.palette.action.disabled }} />
              </Box>
            )}
          </Paper>
        </Grid>

        {/* Zero-Copy Monitor */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2, height: 350 }}>
            <Typography variant="h6" gutterBottom>Zero-Copy Engine</Typography>
            
            <Alert 
              severity={data.zeroCopy.enabled ? 'success' : 'warning'}
              sx={{ mb: 2 }}
            >
              {data.zeroCopy.enabled ? 'Zero-Copy Enabled' : 'Zero-Copy Disabled'}
            </Alert>

            {data.zeroCopy.enabled && (
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="body2" color="text.secondary">Mapped Regions</Typography>
                      <Typography variant="h4">{data.zeroCopy.mappedRegions}</Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={6}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="body2" color="text.secondary">Total Size</Typography>
                      <Typography variant="body1">
                        {(data.zeroCopy.totalMappedSize / 1024 / 1024).toFixed(2)} MB
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={6}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="body2" color="text.secondary">Latency</Typography>
                      <Typography variant="body1">{data.zeroCopy.accessLatency} μs</Typography>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            )}
          </Paper>
        </Grid>

        {/* Forgetting Curve */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2, height: 350 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">Forgetting Curve</Typography>
              <FormControl size="small">
                <Select
                  value={timeRange}
                  onChange={(e) => setTimeRange(Number(e.target.value))}
                >
                  <MenuItem value={300}>5m</MenuItem>
                  <MenuItem value={900}>15m</MenuItem>
                  <MenuItem value={3600}>1h</MenuItem>
                  <MenuItem value={86400}>24h</MenuItem>
                </Select>
              </FormControl>
            </Box>

            <Line
              data={forgettingCurveData}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                  mode: 'index',
                  intersect: false,
                },
                plugins: {
                  legend: {
                    position: 'bottom' as const,
                  }
                },
                scales: {
                  y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                      display: true,
                      text: 'Retention %'
                    }
                  },
                  y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    grid: {
                      drawOnChartArea: false,
                    },
                    title: {
                      display: true,
                      text: 'Items'
                    }
                  }
                }
              }}
              height={200}
            />
          </Paper>
        </Grid>

        {/* Memory Access Patterns */}
        {selectedBufferData && (
          <Grid item xs={12}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                {selectedBufferData.name} - Access Heatmap
              </Typography>
              {renderAccessHeatmap(selectedBufferData)}
              <Box sx={{ mt: 2, display: 'flex', justifyContent: 'space-between' }}>
                <Typography variant="body2" color="text.secondary">
                  Showing first 50 items. Darker = more accesses
                </Typography>
                <Button
                  size="small"
                  color="error"
                  onClick={() => onBufferClear?.(selectedBufferData.id)}
                >
                  Clear Buffer
                </Button>
              </Box>
            </Paper>
          </Grid>
        )}
      </Grid>
    </Box>
  );
};