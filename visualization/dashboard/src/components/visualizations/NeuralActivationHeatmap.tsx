import React, { useMemo, useState, useEffect, useRef } from 'react';
import { Box, Paper, Typography, Slider, FormControlLabel, Switch, Select, MenuItem, FormControl, InputLabel, Chip, Grid, Card, CardContent, IconButton, Tooltip } from '@mui/material';
import { Refresh, ZoomIn, ZoomOut, FilterList, Timeline, BubbleChart, Layers } from '@mui/icons-material';
import { Line, Bar } from 'react-chartjs-2';
import { useTheme } from '@mui/material/styles';
import * as d3 from 'd3';
import { BrainEntity, ActivationDistribution } from '../../types/brain';

interface NeuralLayer {
  id: string;
  name: string;
  entities: BrainEntity[];
  depth: number;
}

interface NeuralActivationHeatmapProps {
  entities: BrainEntity[];
  activationDistribution: ActivationDistribution;
  onEntityClick?: (entity: BrainEntity) => void;
  height?: number | string;
}

export const NeuralActivationHeatmap: React.FC<NeuralActivationHeatmapProps> = ({
  entities,
  activationDistribution,
  onEntityClick,
  height = 600
}) => {
  const theme = useTheme();
  const svgRef = useRef<SVGSVGElement>(null);
  const [selectedLayer, setSelectedLayer] = useState<string>('all');
  const [showSpikes, setShowSpikes] = useState(true);
  const [timeWindow, setTimeWindow] = useState(60); // seconds
  const [zoomLevel, setZoomLevel] = useState(1);
  const [activationHistory, setActivationHistory] = useState<Array<{ timestamp: number; values: number[] }>>([]);

  // Organize entities into layers
  const layers = useMemo(() => {
    const layerMap = new Map<string, NeuralLayer>();
    
    entities.forEach(entity => {
      const layerKey = entity.direction;
      if (!layerMap.has(layerKey)) {
        layerMap.set(layerKey, {
          id: layerKey,
          name: layerKey,
          entities: [],
          depth: ['Input', 'Hidden', 'Gate', 'Output'].indexOf(layerKey)
        });
      }
      layerMap.get(layerKey)!.entities.push(entity);
    });

    return Array.from(layerMap.values()).sort((a, b) => a.depth - b.depth);
  }, [entities]);

  // Calculate grid dimensions for heatmap
  const gridDimensions = useMemo(() => {
    const maxEntitiesPerLayer = Math.max(...layers.map(l => l.entities.length));
    const cols = Math.ceil(Math.sqrt(maxEntitiesPerLayer));
    const rows = Math.ceil(maxEntitiesPerLayer / cols);
    return { cols, rows };
  }, [layers]);

  // Filter entities based on selected layer
  const filteredEntities = useMemo(() => {
    if (selectedLayer === 'all') return entities;
    return entities.filter(e => e.direction === selectedLayer);
  }, [entities, selectedLayer]);

  // Detect spikes (rapid activation changes)
  const spikeDetection = useMemo(() => {
    const spikes: BrainEntity[] = [];
    const threshold = 0.3; // 30% change threshold

    entities.forEach(entity => {
      // In real implementation, compare with previous activation
      // For now, flag high activation entities as potential spikes
      if (entity.activation > 0.8) {
        spikes.push(entity);
      }
    });

    return spikes;
  }, [entities]);

  // Update activation history
  useEffect(() => {
    const newHistory = {
      timestamp: Date.now(),
      values: filteredEntities.map(e => e.activation)
    };

    setActivationHistory(prev => {
      const cutoff = Date.now() - (timeWindow * 1000);
      const filtered = prev.filter(h => h.timestamp > cutoff);
      return [...filtered, newHistory].slice(-100); // Keep last 100 samples
    });
  }, [filteredEntities, timeWindow]);

  // D3 Heatmap rendering
  useEffect(() => {
    if (!svgRef.current || filteredEntities.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 50, right: 50, bottom: 50, left: 50 };
    const width = svgRef.current.clientWidth - margin.left - margin.right;
    const height = svgRef.current.clientHeight - margin.top - margin.bottom;

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Color scale
    const colorScale = d3.scaleSequential(d3.interpolateRdYlBu)
      .domain([1, 0]); // Reversed for heat colors

    // Create heatmap cells
    const cellSize = Math.min(width / gridDimensions.cols, height / layers.length) * zoomLevel;

    layers.forEach((layer, layerIndex) => {
      const layerGroup = g.append('g')
        .attr('transform', `translate(0, ${layerIndex * (cellSize + 10)})`);

      // Layer label
      layerGroup.append('text')
        .attr('x', -10)
        .attr('y', cellSize / 2)
        .attr('text-anchor', 'end')
        .attr('alignment-baseline', 'middle')
        .style('fill', theme.palette.text.primary)
        .style('font-size', '12px')
        .text(layer.name);

      // Entity cells
      layer.entities.forEach((entity, entityIndex) => {
        const col = entityIndex % gridDimensions.cols;
        const row = Math.floor(entityIndex / gridDimensions.cols);
        const x = col * cellSize;
        const y = row * cellSize;

        const cell = layerGroup.append('g')
          .attr('transform', `translate(${x}, ${y})`);

        // Background rect
        cell.append('rect')
          .attr('width', cellSize - 2)
          .attr('height', cellSize - 2)
          .attr('fill', colorScale(entity.activation))
          .attr('stroke', showSpikes && spikeDetection.includes(entity) ? '#ff0000' : 'none')
          .attr('stroke-width', 2)
          .style('cursor', 'pointer')
          .on('click', () => onEntityClick?.(entity))
          .append('title')
          .text(`${entity.id}: ${entity.activation.toFixed(3)}`);

        // Show value for high activations
        if (entity.activation > 0.7 && cellSize > 30) {
          cell.append('text')
            .attr('x', cellSize / 2)
            .attr('y', cellSize / 2)
            .attr('text-anchor', 'middle')
            .attr('alignment-baseline', 'middle')
            .style('fill', '#000')
            .style('font-size', '10px')
            .style('font-weight', 'bold')
            .text(entity.activation.toFixed(2));
        }
      });
    });

    // Add color scale legend
    const legendWidth = 200;
    const legendHeight = 20;
    const legend = svg.append('g')
      .attr('transform', `translate(${width - legendWidth}, 20)`);

    const legendScale = d3.scaleLinear()
      .domain([0, 1])
      .range([0, legendWidth]);

    const legendAxis = d3.axisBottom(legendScale)
      .ticks(5)
      .tickFormat(d3.format('.1f'));

    // Create gradient
    const defs = svg.append('defs');
    const gradient = defs.append('linearGradient')
      .attr('id', 'activation-gradient');

    gradient.selectAll('stop')
      .data(d3.range(0, 1.1, 0.1))
      .enter().append('stop')
      .attr('offset', d => `${d * 100}%`)
      .attr('stop-color', d => colorScale(1 - d));

    legend.append('rect')
      .attr('width', legendWidth)
      .attr('height', legendHeight)
      .style('fill', 'url(#activation-gradient)');

    legend.append('g')
      .attr('transform', `translate(0, ${legendHeight})`)
      .call(legendAxis);

  }, [filteredEntities, layers, gridDimensions, theme, zoomLevel, showSpikes, spikeDetection, onEntityClick]);

  // Prepare time series data
  const timeSeriesData = useMemo(() => {
    const labels = activationHistory.map(h => new Date(h.timestamp).toLocaleTimeString());
    const datasets = [{
      label: 'Average Activation',
      data: activationHistory.map(h => {
        const sum = h.values.reduce((a, b) => a + b, 0);
        return sum / h.values.length || 0;
      }),
      borderColor: theme.palette.primary.main,
      backgroundColor: theme.palette.primary.light,
      tension: 0.4
    }];

    if (showSpikes) {
      datasets.push({
        label: 'Spike Count',
        data: activationHistory.map(h => h.values.filter(v => v > 0.8).length),
        borderColor: theme.palette.error.main,
        backgroundColor: theme.palette.error.light,
        yAxisID: 'y1',
        tension: 0.4
      });
    }

    return { labels, datasets };
  }, [activationHistory, showSpikes, theme]);

  // Activation distribution chart
  const distributionData = useMemo(() => {
    return {
      labels: ['Very Low', 'Low', 'Medium', 'High', 'Very High'],
      datasets: [{
        label: 'Entity Count',
        data: [
          activationDistribution.veryLow,
          activationDistribution.low,
          activationDistribution.medium,
          activationDistribution.high,
          activationDistribution.veryHigh
        ],
        backgroundColor: [
          theme.palette.info.light,
          theme.palette.info.main,
          theme.palette.warning.light,
          theme.palette.warning.main,
          theme.palette.error.main
        ]
      }]
    };
  }, [activationDistribution, theme]);

  return (
    <Box sx={{ height, display: 'flex', flexDirection: 'column', gap: 2 }}>
      <Grid container spacing={2}>
        {/* Main Heatmap */}
        <Grid item xs={12} lg={8}>
          <Paper sx={{ p: 2, height: '100%' }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">Neural Activation Heatmap</Typography>
              <Box sx={{ display: 'flex', gap: 1 }}>
                <FormControl size="small" sx={{ minWidth: 120 }}>
                  <InputLabel>Layer</InputLabel>
                  <Select
                    value={selectedLayer}
                    onChange={(e) => setSelectedLayer(e.target.value)}
                    label="Layer"
                  >
                    <MenuItem value="all">All Layers</MenuItem>
                    {layers.map(layer => (
                      <MenuItem key={layer.id} value={layer.id}>
                        {layer.name} ({layer.entities.length})
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <FormControlLabel
                  control={
                    <Switch
                      checked={showSpikes}
                      onChange={(e) => setShowSpikes(e.target.checked)}
                    />
                  }
                  label="Show Spikes"
                />
                <IconButton onClick={() => setZoomLevel(z => Math.min(z + 0.2, 2))}>
                  <ZoomIn />
                </IconButton>
                <IconButton onClick={() => setZoomLevel(z => Math.max(z - 0.2, 0.5))}>
                  <ZoomOut />
                </IconButton>
              </Box>
            </Box>
            
            <svg
              ref={svgRef}
              style={{ width: '100%', height: 'calc(100% - 60px)' }}
            />

            {showSpikes && spikeDetection.length > 0 && (
              <Box sx={{ mt: 1 }}>
                <Typography variant="body2" color="error">
                  <strong>Spike Alert:</strong> {spikeDetection.length} neurons showing high activation
                </Typography>
              </Box>
            )}
          </Paper>
        </Grid>

        {/* Side Panels */}
        <Grid item xs={12} lg={4}>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            {/* Activation Distribution */}
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Activation Distribution
                </Typography>
                <Bar
                  data={distributionData}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: { display: false }
                    },
                    scales: {
                      y: {
                        beginAtZero: true,
                        title: {
                          display: true,
                          text: 'Entity Count'
                        }
                      }
                    }
                  }}
                  height={200}
                />
              </CardContent>
            </Card>

            {/* Time Series */}
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                  <Typography variant="h6">Activation Timeline</Typography>
                  <FormControl size="small">
                    <Select
                      value={timeWindow}
                      onChange={(e) => setTimeWindow(Number(e.target.value))}
                    >
                      <MenuItem value={30}>30s</MenuItem>
                      <MenuItem value={60}>1m</MenuItem>
                      <MenuItem value={300}>5m</MenuItem>
                    </Select>
                  </FormControl>
                </Box>
                <Line
                  data={timeSeriesData}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                      mode: 'index',
                      intersect: false,
                    },
                    plugins: {
                      legend: { position: 'top' as const }
                    },
                    scales: {
                      y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        min: 0,
                        max: 1,
                        title: {
                          display: true,
                          text: 'Activation'
                        }
                      },
                      y1: {
                        type: 'linear',
                        display: showSpikes,
                        position: 'right',
                        grid: { drawOnChartArea: false },
                        title: {
                          display: true,
                          text: 'Spike Count'
                        }
                      }
                    }
                  }}
                  height={200}
                />
              </CardContent>
            </Card>

            {/* Pattern Detection */}
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Active Patterns
                </Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                  {spikeDetection.length > 0 && (
                    <Chip
                      label={`Spike Pattern: ${spikeDetection.length} neurons`}
                      color="error"
                      size="small"
                    />
                  )}
                  {layers.map(layer => {
                    const avgActivation = layer.entities.reduce((sum, e) => sum + e.activation, 0) / layer.entities.length;
                    return (
                      <Box key={layer.id} sx={{ display: 'flex', justifyContent: 'space-between' }}>
                        <Typography variant="body2">{layer.name}:</Typography>
                        <Typography variant="body2" color={avgActivation > 0.6 ? 'error' : 'text.secondary'}>
                          {avgActivation.toFixed(3)}
                        </Typography>
                      </Box>
                    );
                  })}
                </Box>
              </CardContent>
            </Card>
          </Box>
        </Grid>
      </Grid>
    </Box>
  );
};