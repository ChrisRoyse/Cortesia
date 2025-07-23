import React, { useState, useMemo, useEffect } from 'react';
import { Box, Paper, Typography, Grid, Card, CardContent, LinearProgress, Chip, IconButton, Tooltip, FormControl, Select, MenuItem, InputLabel, Switch, FormControlLabel, Alert } from '@mui/material';
import { Radar, Doughnut, Line } from 'react-chartjs-2';
import { Psychology, TrendingUp, Lightbulb, Hub, Analytics, Security, Tune, SwapHoriz, Timeline, Warning } from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';
import {
  Chart as ChartJS,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  ArcElement,
  CategoryScale,
  LinearScale,
  Title,
  Tooltip as ChartTooltip,
  Legend
} from 'chart.js';

ChartJS.register(
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  ArcElement,
  CategoryScale,
  LinearScale,
  Title,
  ChartTooltip,
  Legend
);

interface CognitivePattern {
  id: string;
  name: string;
  type: 'convergent' | 'divergent' | 'lateral' | 'systems' | 'critical' | 'abstract' | 'adaptive';
  strength: number; // 0-1
  confidence: number; // 0-1
  active: boolean;
  resources: number; // percentage of resources allocated
  effectiveness: number; // 0-1
  lastSwitch: number; // timestamp
}

interface AttentionTarget {
  id: string;
  name: string;
  priority: number; // 0-1
  resources: number; // percentage
  type: 'entity' | 'concept' | 'pattern' | 'task';
}

interface InhibitoryConnection {
  from: string;
  to: string;
  strength: number;
  active: boolean;
}

interface CognitiveSystemsData {
  patterns: CognitivePattern[];
  attention: {
    targets: AttentionTarget[];
    totalCapacity: number;
    usedCapacity: number;
  };
  inhibitory: {
    connections: InhibitoryConnection[];
    balance: number; // -1 to 1 (negative = too much inhibition)
    patterns: string[];
  };
  patternHistory: Array<{
    timestamp: number;
    activePatterns: string[];
    switchReason?: string;
  }>;
  executiveCommands: Array<{
    id: string;
    command: string;
    status: 'pending' | 'executing' | 'completed' | 'failed';
    timestamp: number;
  }>;
}

interface CognitiveSystemsDashboardProps {
  data: CognitiveSystemsData;
  onPatternSwitch?: (pattern: CognitivePattern) => void;
  onAttentionShift?: (target: AttentionTarget) => void;
  height?: number | string;
}

const patternIcons: Record<string, React.ElementType> = {
  convergent: TrendingUp,
  divergent: Lightbulb,
  lateral: SwapHoriz,
  systems: Hub,
  critical: Security,
  abstract: Psychology,
  adaptive: Tune
};

const patternDescriptions: Record<string, string> = {
  convergent: 'Focused problem-solving toward a single solution',
  divergent: 'Creative exploration of multiple possibilities',
  lateral: 'Cross-domain connections and associations',
  systems: 'Holistic analysis of interconnected elements',
  critical: 'Validation and verification of conclusions',
  abstract: 'Pattern extraction and generalization',
  adaptive: 'Dynamic strategy selection based on context'
};

export const CognitiveSystemsDashboard: React.FC<CognitiveSystemsDashboardProps> = ({
  data,
  onPatternSwitch,
  onAttentionShift,
  height = 800
}) => {
  const theme = useTheme();
  const [selectedPattern, setSelectedPattern] = useState<string | null>(null);
  const [showInactive, setShowInactive] = useState(false);
  const [timeRange, setTimeRange] = useState(300); // last 5 minutes

  // Active patterns
  const activePatterns = useMemo(() => 
    data.patterns.filter(p => p.active || showInactive),
    [data.patterns, showInactive]
  );

  // Pattern strength radar data
  const radarData = useMemo(() => {
    const labels = data.patterns.map(p => p.name);
    const strengths = data.patterns.map(p => p.strength * 100);
    const effectiveness = data.patterns.map(p => p.effectiveness * 100);

    return {
      labels,
      datasets: [
        {
          label: 'Strength',
          data: strengths,
          backgroundColor: theme.palette.primary.main + '40',
          borderColor: theme.palette.primary.main,
          pointBackgroundColor: theme.palette.primary.main,
          pointBorderColor: '#fff',
          pointHoverBackgroundColor: '#fff',
          pointHoverBorderColor: theme.palette.primary.main,
        },
        {
          label: 'Effectiveness',
          data: effectiveness,
          backgroundColor: theme.palette.success.main + '40',
          borderColor: theme.palette.success.main,
          pointBackgroundColor: theme.palette.success.main,
          pointBorderColor: '#fff',
          pointHoverBackgroundColor: '#fff',
          pointHoverBorderColor: theme.palette.success.main,
        }
      ]
    };
  }, [data.patterns, theme]);

  // Attention allocation doughnut data
  const attentionData = useMemo(() => {
    const colors = [
      theme.palette.primary.main,
      theme.palette.secondary.main,
      theme.palette.warning.main,
      theme.palette.info.main,
      theme.palette.success.main,
    ];

    return {
      labels: data.attention.targets.map(t => t.name),
      datasets: [{
        data: data.attention.targets.map(t => t.resources),
        backgroundColor: colors,
        borderColor: colors.map(c => c),
        borderWidth: 2,
      }]
    };
  }, [data.attention.targets, theme]);

  // Pattern switching timeline
  const timelineData = useMemo(() => {
    const cutoff = Date.now() - (timeRange * 1000);
    const recentHistory = data.patternHistory.filter(h => h.timestamp > cutoff);
    
    const labels = recentHistory.map(h => new Date(h.timestamp).toLocaleTimeString());
    const patternCounts = recentHistory.map(h => h.activePatterns.length);

    return {
      labels,
      datasets: [{
        label: 'Active Patterns',
        data: patternCounts,
        borderColor: theme.palette.primary.main,
        backgroundColor: theme.palette.primary.light,
        tension: 0.4
      }]
    };
  }, [data.patternHistory, timeRange, theme]);

  // Inhibitory balance indicator
  const inhibitoryStatus = useMemo(() => {
    const balance = data.inhibitory.balance;
    if (balance < -0.5) return { level: 'error', text: 'Over-inhibited' };
    if (balance < -0.2) return { level: 'warning', text: 'High inhibition' };
    if (balance > 0.5) return { level: 'error', text: 'Under-inhibited' };
    if (balance > 0.2) return { level: 'warning', text: 'Low inhibition' };
    return { level: 'success', text: 'Balanced' };
  }, [data.inhibitory.balance]);

  const handlePatternClick = (pattern: CognitivePattern) => {
    setSelectedPattern(pattern.id);
    if (pattern.active) {
      onPatternSwitch?.(pattern);
    }
  };

  return (
    <Box sx={{ height, display: 'flex', flexDirection: 'column', gap: 2 }}>
      <Grid container spacing={2}>
        {/* Active Reasoning Patterns */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, height: 400 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">Active Reasoning Patterns</Typography>
              <FormControlLabel
                control={
                  <Switch
                    checked={showInactive}
                    onChange={(e) => setShowInactive(e.target.checked)}
                  />
                }
                label="Show All"
                labelPlacement="start"
              />
            </Box>
            
            <Box sx={{ position: 'relative', height: 'calc(100% - 60px)' }}>
              <Radar 
                data={radarData}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  scales: {
                    r: {
                      beginAtZero: true,
                      max: 100,
                      ticks: {
                        stepSize: 20
                      }
                    }
                  },
                  plugins: {
                    legend: {
                      position: 'top' as const,
                    },
                    tooltip: {
                      callbacks: {
                        label: (context) => {
                          return `${context.dataset.label}: ${context.parsed.r}%`;
                        }
                      }
                    }
                  },
                  onClick: (event, elements) => {
                    if (elements.length > 0) {
                      const index = elements[0].index;
                      handlePatternClick(data.patterns[index]);
                    }
                  }
                }}
              />
            </Box>
          </Paper>
        </Grid>

        {/* Pattern Details */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, height: 400 }}>
            <Typography variant="h6" gutterBottom>Pattern Details</Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, height: 'calc(100% - 40px)', overflowY: 'auto' }}>
              {activePatterns.map(pattern => {
                const Icon = patternIcons[pattern.type];
                return (
                  <Card 
                    key={pattern.id}
                    sx={{ 
                      cursor: 'pointer',
                      bgcolor: selectedPattern === pattern.id ? 'action.selected' : 'background.paper',
                      opacity: pattern.active ? 1 : 0.6
                    }}
                    onClick={() => handlePatternClick(pattern)}
                  >
                    <CardContent sx={{ p: 2 }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                        <Icon color={pattern.active ? 'primary' : 'disabled'} />
                        <Typography variant="subtitle1" sx={{ flexGrow: 1 }}>
                          {pattern.name}
                        </Typography>
                        <Chip 
                          label={pattern.active ? 'Active' : 'Inactive'}
                          color={pattern.active ? 'success' : 'default'}
                          size="small"
                        />
                      </Box>
                      <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                        {patternDescriptions[pattern.type]}
                      </Typography>
                      <Grid container spacing={1}>
                        <Grid item xs={4}>
                          <Typography variant="caption" color="text.secondary">Strength</Typography>
                          <LinearProgress 
                            variant="determinate" 
                            value={pattern.strength * 100} 
                            sx={{ mt: 0.5 }}
                          />
                        </Grid>
                        <Grid item xs={4}>
                          <Typography variant="caption" color="text.secondary">Confidence</Typography>
                          <LinearProgress 
                            variant="determinate" 
                            value={pattern.confidence * 100} 
                            color="secondary"
                            sx={{ mt: 0.5 }}
                          />
                        </Grid>
                        <Grid item xs={4}>
                          <Typography variant="caption" color="text.secondary">Resources</Typography>
                          <Typography variant="body2">{pattern.resources}%</Typography>
                        </Grid>
                      </Grid>
                    </CardContent>
                  </Card>
                );
              })}
            </Box>
          </Paper>
        </Grid>

        {/* Attention Focus */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2, height: 350 }}>
            <Typography variant="h6" gutterBottom>Attention Focus</Typography>
            <Box sx={{ position: 'relative', height: 200 }}>
              <Doughnut
                data={attentionData}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: {
                    legend: {
                      position: 'bottom' as const,
                    },
                    tooltip: {
                      callbacks: {
                        label: (context) => {
                          return `${context.label}: ${context.parsed}%`;
                        }
                      }
                    }
                  },
                  onClick: (event, elements) => {
                    if (elements.length > 0) {
                      const index = elements[0].index;
                      onAttentionShift?.(data.attention.targets[index]);
                    }
                  }
                }}
              />
            </Box>
            <Box sx={{ mt: 2 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="body2">Capacity Usage</Typography>
                <Typography variant="body2">
                  {data.attention.usedCapacity}/{data.attention.totalCapacity}
                </Typography>
              </Box>
              <LinearProgress 
                variant="determinate" 
                value={(data.attention.usedCapacity / data.attention.totalCapacity) * 100}
                color={data.attention.usedCapacity > data.attention.totalCapacity * 0.8 ? 'warning' : 'primary'}
              />
            </Box>
          </Paper>
        </Grid>

        {/* Inhibitory System */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2, height: 350 }}>
            <Typography variant="h6" gutterBottom>Inhibitory System</Typography>
            
            <Alert severity={inhibitoryStatus.level as any} sx={{ mb: 2 }}>
              {inhibitoryStatus.text}
            </Alert>

            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" gutterBottom>System Balance</Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Typography variant="caption">Inhibitory</Typography>
                <Box sx={{ flexGrow: 1, mx: 1 }}>
                  <LinearProgress 
                    variant="determinate" 
                    value={50 + (data.inhibitory.balance * 50)}
                    sx={{
                      height: 10,
                      borderRadius: 5,
                      '& .MuiLinearProgress-bar': {
                        backgroundColor: data.inhibitory.balance < 0 ? theme.palette.error.main : theme.palette.success.main
                      }
                    }}
                  />
                </Box>
                <Typography variant="caption">Excitatory</Typography>
              </Box>
            </Box>

            <Box>
              <Typography variant="body2" gutterBottom>Active Connections</Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                {data.inhibitory.connections
                  .filter(c => c.active)
                  .slice(0, 5)
                  .map((conn, idx) => (
                    <Box key={idx} sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="caption">
                        {conn.from} → {conn.to}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {(conn.strength * 100).toFixed(0)}%
                      </Typography>
                    </Box>
                  ))}
              </Box>
            </Box>

            <Box sx={{ mt: 2 }}>
              <Typography variant="body2" gutterBottom>Suppression Patterns</Typography>
              <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                {data.inhibitory.patterns.map(pattern => (
                  <Chip key={pattern} label={pattern} size="small" variant="outlined" />
                ))}
              </Box>
            </Box>
          </Paper>
        </Grid>

        {/* Pattern Switching Timeline */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2, height: 350 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">Pattern History</Typography>
              <FormControl size="small">
                <Select
                  value={timeRange}
                  onChange={(e) => setTimeRange(Number(e.target.value))}
                >
                  <MenuItem value={60}>1m</MenuItem>
                  <MenuItem value={300}>5m</MenuItem>
                  <MenuItem value={900}>15m</MenuItem>
                </Select>
              </FormControl>
            </Box>
            
            <Box sx={{ height: 'calc(100% - 60px)' }}>
              <Line
                data={timelineData}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  scales: {
                    y: {
                      beginAtZero: true,
                      ticks: {
                        stepSize: 1
                      }
                    }
                  },
                  plugins: {
                    legend: {
                      display: false
                    }
                  }
                }}
              />
            </Box>

            {/* Recent switches */}
            <Box sx={{ mt: 2 }}>
              <Typography variant="body2" gutterBottom>Recent Switches</Typography>
              {data.patternHistory.slice(-3).reverse().map((h, idx) => (
                <Box key={idx} sx={{ display: 'flex', gap: 1, alignItems: 'center', mb: 0.5 }}>
                  <Typography variant="caption" color="text.secondary">
                    {new Date(h.timestamp).toLocaleTimeString()}
                  </Typography>
                  <Chip 
                    label={h.activePatterns.join(', ')} 
                    size="small" 
                    variant="outlined"
                  />
                  {h.switchReason && (
                    <Tooltip title={h.switchReason}>
                      <Warning fontSize="small" color="warning" />
                    </Tooltip>
                  )}
                </Box>
              ))}
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};