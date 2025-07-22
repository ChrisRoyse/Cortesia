import React, { useState, useMemo, useRef, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Tabs,
  Tab,
  Grid,
  Chip,
  IconButton,
  Tooltip,
  Slider,
  FormControlLabel,
  Switch,
  useTheme,
  alpha,
} from '@mui/material';
import {
  Timeline,
  BubbleChart,
  GridOn,
  PhotoCamera,
  PlayArrow,
  Pause,
  SkipNext,
  SkipPrevious,
} from '@mui/icons-material';
import * as d3 from 'd3';
import { saveAs } from 'file-saver';

interface NeuralActivity {
  timestamp: number;
  neurons: {
    id: string;
    activity: number;
    type?: string;
  }[];
  patterns?: {
    id: string;
    strength: number;
    neurons: string[];
  }[];
}

interface SDRData {
  dimensions: number[];
  active_bits: number[];
  density: number;
  overlap?: number;
  metadata?: Record<string, any>;
}

interface MemoryData {
  consolidation_progress: number;
  memory_usage: {
    total: number;
    used: number;
    consolidations: number;
  };
  recent_consolidations: {
    timestamp: number;
    from_tier: string;
    to_tier: string;
    size: number;
  }[];
}

interface NeuralDataViewerProps {
  data: {
    neural_activity?: NeuralActivity[];
    sdr_data?: SDRData[];
    memory_data?: MemoryData;
    cognitive_patterns?: {
      pattern_id: string;
      strength: number;
      timestamp: number;
      type: string;
    }[];
  };
  fullscreen?: boolean;
}

const NeuralDataViewer: React.FC<NeuralDataViewerProps> = ({ data, fullscreen = false }) => {
  const theme = useTheme();
  const [activeTab, setActiveTab] = useState(0);
  const [timeIndex, setTimeIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const heatmapRef = useRef<SVGSVGElement>(null);
  const spikeTrainRef = useRef<SVGSVGElement>(null);
  const sdrRef = useRef<SVGSVGElement>(null);

  // Determine available tabs based on data
  const availableTabs = useMemo(() => {
    const tabs = [];
    if (data.neural_activity) tabs.push({ label: 'Neural Activity', icon: <Timeline /> });
    if (data.sdr_data) tabs.push({ label: 'SDR Visualization', icon: <GridOn /> });
    if (data.memory_data) tabs.push({ label: 'Memory Consolidation', icon: <BubbleChart /> });
    if (data.cognitive_patterns) tabs.push({ label: 'Cognitive Patterns', icon: <Timeline /> });
    return tabs;
  }, [data]);

  // Playback control
  useEffect(() => {
    if (!isPlaying) return;

    const interval = setInterval(() => {
      setTimeIndex((prev) => {
        const maxIndex = data.neural_activity?.length || 0;
        return prev >= maxIndex - 1 ? 0 : prev + 1;
      });
    }, 1000 / playbackSpeed);

    return () => clearInterval(interval);
  }, [isPlaying, playbackSpeed, data.neural_activity]);

  // Neural Activity Heatmap
  useEffect(() => {
    if (!heatmapRef.current || !data.neural_activity || activeTab !== 0) return;

    const activity = data.neural_activity[timeIndex];
    if (!activity) return;

    const width = 800;
    const height = 400;
    const margin = { top: 20, right: 20, bottom: 40, left: 60 };

    d3.select(heatmapRef.current).selectAll('*').remove();

    const svg = d3.select(heatmapRef.current)
      .attr('width', width)
      .attr('height', height);

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Create scales
    const neuronIds = activity.neurons.map(n => n.id);
    const xScale = d3.scaleBand()
      .domain(neuronIds)
      .range([0, width - margin.left - margin.right])
      .padding(0.1);

    const yScale = d3.scaleLinear()
      .domain([0, 1])
      .range([height - margin.top - margin.bottom, 0]);

    const colorScale = d3.scaleSequential(d3.interpolateViridis)
      .domain([0, 1]);

    // Draw bars
    g.selectAll('.neuron-bar')
      .data(activity.neurons)
      .enter().append('rect')
      .attr('class', 'neuron-bar')
      .attr('x', d => xScale(d.id)!)
      .attr('y', d => yScale(d.activity))
      .attr('width', xScale.bandwidth())
      .attr('height', d => yScale(0) - yScale(d.activity))
      .attr('fill', d => colorScale(d.activity))
      .on('mouseover', function(event, d) {
        const tooltip = d3.select('body').append('div')
          .attr('class', 'tooltip')
          .style('opacity', 0)
          .style('position', 'absolute')
          .style('background', theme.palette.background.paper)
          .style('padding', '8px')
          .style('border-radius', '4px')
          .style('box-shadow', '0 2px 4px rgba(0,0,0,0.1)');

        tooltip.transition()
          .duration(200)
          .style('opacity', .9);

        tooltip.html(`Neuron: ${d.id}<br/>Activity: ${d.activity.toFixed(3)}<br/>Type: ${d.type || 'Unknown'}`)
          .style('left', (event.pageX + 10) + 'px')
          .style('top', (event.pageY - 28) + 'px');
      })
      .on('mouseout', function() {
        d3.select('.tooltip').remove();
      });

    // Add axes
    g.append('g')
      .attr('transform', `translate(0,${height - margin.top - margin.bottom})`)
      .call(d3.axisBottom(xScale));

    g.append('g')
      .call(d3.axisLeft(yScale));

    // Add labels
    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', 0 - margin.left)
      .attr('x', 0 - (height / 2))
      .attr('dy', '1em')
      .style('text-anchor', 'middle')
      .text('Neural Activity');

  }, [data.neural_activity, timeIndex, activeTab, theme, showHeatmap]);

  // SDR Visualization
  useEffect(() => {
    if (!sdrRef.current || !data.sdr_data || activeTab !== 1) return;

    const sdrData = data.sdr_data[0]; // Use first SDR for now
    if (!sdrData) return;

    const width = 800;
    const height = 600;
    const cellSize = 10;

    d3.select(sdrRef.current).selectAll('*').remove();

    const svg = d3.select(sdrRef.current)
      .attr('width', width)
      .attr('height', height);

    const totalBits = sdrData.dimensions.reduce((a, b) => a * b, 1);
    const cols = Math.ceil(Math.sqrt(totalBits));
    const rows = Math.ceil(totalBits / cols);

    const activeBitsSet = new Set(sdrData.active_bits);

    // Create bit visualization
    const bits = [];
    for (let i = 0; i < totalBits; i++) {
      bits.push({
        index: i,
        active: activeBitsSet.has(i),
        x: (i % cols) * cellSize,
        y: Math.floor(i / cols) * cellSize,
      });
    }

    svg.selectAll('.sdr-bit')
      .data(bits)
      .enter().append('rect')
      .attr('class', 'sdr-bit')
      .attr('x', d => d.x)
      .attr('y', d => d.y)
      .attr('width', cellSize - 1)
      .attr('height', cellSize - 1)
      .attr('fill', d => d.active ? theme.palette.primary.main : theme.palette.action.disabled)
      .attr('opacity', d => d.active ? 1 : 0.2);

    // Add density info
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', height - 20)
      .attr('text-anchor', 'middle')
      .text(`Density: ${(sdrData.density * 100).toFixed(2)}% | Active bits: ${sdrData.active_bits.length}`);

  }, [data.sdr_data, activeTab, theme]);

  // Cognitive Patterns Timeline
  const renderCognitivePatterns = () => {
    if (!data.cognitive_patterns) return null;

    const patterns = data.cognitive_patterns;
    const patternTypes = [...new Set(patterns.map(p => p.type))];
    const colorScale = d3.scaleOrdinal(d3.schemeCategory10).domain(patternTypes);

    return (
      <Box sx={{ p: 2 }}>
        <Grid container spacing={2}>
          {patternTypes.map((type) => {
            const typePatterns = patterns.filter(p => p.type === type);
            const avgStrength = typePatterns.reduce((sum, p) => sum + p.strength, 0) / typePatterns.length;

            return (
              <Grid item xs={12} md={6} key={type}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    {type}
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                    <Box sx={{ flex: 1 }}>
                      <Box
                        sx={{
                          height: 20,
                          bgcolor: colorScale(type),
                          width: `${avgStrength * 100}%`,
                          borderRadius: 1,
                          transition: 'width 0.3s',
                        }}
                      />
                    </Box>
                    <Typography variant="body2">
                      {(avgStrength * 100).toFixed(1)}%
                    </Typography>
                  </Box>
                  <Typography variant="caption" color="text.secondary">
                    {typePatterns.length} patterns
                  </Typography>
                </Paper>
              </Grid>
            );
          })}
        </Grid>
      </Box>
    );
  };

  // Memory Consolidation View
  const renderMemoryConsolidation = () => {
    if (!data.memory_data) return null;

    const { consolidation_progress, memory_usage, recent_consolidations } = data.memory_data;

    return (
      <Box sx={{ p: 2 }}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Consolidation Progress
              </Typography>
              <Box sx={{ position: 'relative', height: 200 }}>
                <Box
                  sx={{
                    position: 'absolute',
                    top: '50%',
                    left: '50%',
                    transform: 'translate(-50%, -50%)',
                  }}
                >
                  <Typography variant="h3">
                    {(consolidation_progress * 100).toFixed(1)}%
                  </Typography>
                </Box>
                <svg width="200" height="200" style={{ position: 'absolute', top: 0, left: '50%', transform: 'translateX(-50%)' }}>
                  <circle
                    cx="100"
                    cy="100"
                    r="80"
                    fill="none"
                    stroke={theme.palette.action.disabled}
                    strokeWidth="10"
                  />
                  <circle
                    cx="100"
                    cy="100"
                    r="80"
                    fill="none"
                    stroke={theme.palette.primary.main}
                    strokeWidth="10"
                    strokeDasharray={`${consolidation_progress * 502.65} ${502.65}`}
                    transform="rotate(-90 100 100)"
                  />
                </svg>
              </Box>
            </Paper>
          </Grid>

          <Grid item xs={12} md={4}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Memory Usage
              </Typography>
              <Box sx={{ mt: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Used: {(memory_usage.used / 1024 / 1024).toFixed(2)} MB
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Total: {(memory_usage.total / 1024 / 1024).toFixed(2)} MB
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Consolidations: {memory_usage.consolidations}
                </Typography>
              </Box>
              <Box sx={{ mt: 2 }}>
                <Box
                  sx={{
                    height: 8,
                    bgcolor: theme.palette.action.disabled,
                    borderRadius: 1,
                    overflow: 'hidden',
                  }}
                >
                  <Box
                    sx={{
                      height: '100%',
                      width: `${(memory_usage.used / memory_usage.total) * 100}%`,
                      bgcolor: theme.palette.primary.main,
                    }}
                  />
                </Box>
              </Box>
            </Paper>
          </Grid>

          <Grid item xs={12} md={4}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Recent Consolidations
              </Typography>
              <Box sx={{ mt: 2 }}>
                {recent_consolidations.slice(0, 5).map((cons, idx) => (
                  <Box key={idx} sx={{ mb: 1 }}>
                    <Typography variant="caption" color="text.secondary">
                      {new Date(cons.timestamp).toLocaleTimeString()}
                    </Typography>
                    <Typography variant="body2">
                      {cons.from_tier} → {cons.to_tier}
                    </Typography>
                    <Typography variant="caption">
                      {(cons.size / 1024).toFixed(2)} KB
                    </Typography>
                  </Box>
                ))}
              </Box>
            </Paper>
          </Grid>
        </Grid>
      </Box>
    );
  };

  const exportVisualization = () => {
    let svgElement: SVGSVGElement | null = null;
    let filename = 'neural-visualization';

    switch (activeTab) {
      case 0:
        svgElement = heatmapRef.current;
        filename = 'neural-activity';
        break;
      case 1:
        svgElement = sdrRef.current;
        filename = 'sdr-visualization';
        break;
    }

    if (svgElement) {
      const svgData = new XMLSerializer().serializeToString(svgElement);
      const blob = new Blob([svgData], { type: 'image/svg+xml' });
      saveAs(blob, `${filename}.svg`);
    }
  };

  return (
    <Paper
      elevation={0}
      sx={{
        height: fullscreen ? '100vh' : 'auto',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      {/* Header with tabs */}
      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs
          value={activeTab}
          onChange={(e, value) => setActiveTab(value)}
          variant="scrollable"
          scrollButtons="auto"
        >
          {availableTabs.map((tab, index) => (
            <Tab
              key={index}
              label={tab.label}
              icon={tab.icon}
              iconPosition="start"
            />
          ))}
        </Tabs>
      </Box>

      {/* Playback controls for temporal data */}
      {data.neural_activity && activeTab === 0 && (
        <Box
          sx={{
            p: 2,
            borderBottom: 1,
            borderColor: 'divider',
            display: 'flex',
            alignItems: 'center',
            gap: 2,
          }}
        >
          <IconButton onClick={() => setTimeIndex(Math.max(0, timeIndex - 1))}>
            <SkipPrevious />
          </IconButton>
          <IconButton onClick={() => setIsPlaying(!isPlaying)}>
            {isPlaying ? <Pause /> : <PlayArrow />}
          </IconButton>
          <IconButton
            onClick={() =>
              setTimeIndex(Math.min(data.neural_activity!.length - 1, timeIndex + 1))
            }
          >
            <SkipNext />
          </IconButton>

          <Slider
            value={timeIndex}
            onChange={(e, value) => setTimeIndex(value as number)}
            min={0}
            max={(data.neural_activity?.length || 1) - 1}
            sx={{ flex: 1 }}
          />

          <Typography variant="body2">
            {timeIndex + 1} / {data.neural_activity?.length || 0}
          </Typography>

          <FormControlLabel
            control={
              <Switch
                checked={showHeatmap}
                onChange={(e) => setShowHeatmap(e.target.checked)}
              />
            }
            label="Heatmap"
          />

          <Tooltip title="Export">
            <IconButton onClick={exportVisualization}>
              <PhotoCamera />
            </IconButton>
          </Tooltip>
        </Box>
      )}

      {/* Content */}
      <Box sx={{ flex: 1, overflow: 'auto', p: 2 }}>
        {activeTab === 0 && data.neural_activity && (
          <Box>
            <svg ref={heatmapRef} />
            {data.neural_activity[timeIndex]?.patterns && (
              <Box sx={{ mt: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Active Patterns
                </Typography>
                <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                  {data.neural_activity[timeIndex].patterns!.map((pattern) => (
                    <Chip
                      key={pattern.id}
                      label={`${pattern.id}: ${(pattern.strength * 100).toFixed(1)}%`}
                      size="small"
                      color="primary"
                      variant={pattern.strength > 0.7 ? 'filled' : 'outlined'}
                    />
                  ))}
                </Box>
              </Box>
            )}
          </Box>
        )}

        {activeTab === 1 && data.sdr_data && (
          <Box>
            <svg ref={sdrRef} />
            {data.sdr_data[0]?.metadata && (
              <Box sx={{ mt: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  SDR Metadata
                </Typography>
                <pre style={{ fontSize: '0.875rem' }}>
                  {JSON.stringify(data.sdr_data[0].metadata, null, 2)}
                </pre>
              </Box>
            )}
          </Box>
        )}

        {activeTab === 2 && renderMemoryConsolidation()}
        {activeTab === 3 && renderCognitivePatterns()}
      </Box>
    </Paper>
  );
};

export default NeuralDataViewer;