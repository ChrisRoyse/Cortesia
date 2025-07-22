/**
 * LLMKG Theme Showcase
 * Demonstrates the comprehensive theme system
 */

import React from 'react';
import {
  Box,
  Typography,
  Paper,
  Button,
  Card,
  CardContent,
  Chip,
  TextField,
  Alert,
  IconButton,
  Tooltip,
  Grid,
  Stack,
  Divider,
} from '@mui/material';
import {
  CpuChipIcon,
  BrainIcon,
  ChartBarIcon,
  CubeIcon,
  SparklesIcon,
  BeakerIcon,
} from '@heroicons/react/24/outline';
import { useTheme, useThemeStyles } from '../hooks/useTheme';
import { ThemeToggle } from '../components/ThemeProvider/ThemeProvider';

export const ThemeShowcase: React.FC = () => {
  const {
    mode,
    colors,
    spacing,
    shadows,
    borderRadius,
    transitions,
    createGradient,
    withAlpha,
    getColor,
    getShadow,
    getRadius,
  } = useTheme();
  
  const themeStyles = useThemeStyles();

  // Sample data for visualizations
  const cognitiveMetrics = [
    { label: 'Pattern Recognition', value: 85, color: colors.cognitive[500] },
    { label: 'Reasoning', value: 72, color: colors.cognitive[600] },
    { label: 'Inference', value: 90, color: colors.cognitive[400] },
  ];

  const neuralActivity = [
    { label: 'High Activity', value: 65, color: colors.neural[500] },
    { label: 'Medium Activity', value: 45, color: colors.neural[400] },
    { label: 'Low Activity', value: 25, color: colors.neural[300] },
  ];

  const memoryStatus = [
    { label: 'Working Memory', value: 80, color: colors.memory[400] },
    { label: 'Long-term Memory', value: 95, color: colors.memory[600] },
    { label: 'Consolidation', value: 60, color: colors.memory[500] },
  ];

  return (
    <Box sx={{ p: 4, maxWidth: 1400, mx: 'auto' }}>
      {/* Header */}
      <Box sx={{ mb: 6 }}>
        <Typography variant="h3" sx={{ mb: 2, fontWeight: 700 }}>
          LLMKG Theme Showcase
        </Typography>
        <Typography variant="subtitle1" color="text.secondary" sx={{ mb: 3 }}>
          Explore the brain-inspired color palettes and design system
        </Typography>
        
        {/* Theme Toggle */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Typography variant="body2" color="text.secondary">
            Current theme: {mode}
          </Typography>
          <ThemeToggle size="medium" showLabel />
        </Box>
      </Box>

      {/* Color Palettes */}
      <Section title="Color Palettes">
        <Grid container spacing={3}>
          {/* Cognitive Palette */}
          <Grid item xs={12} md={6} lg={3}>
            <ColorPaletteCard
              title="Cognitive"
              icon={<BrainIcon className="w-5 h-5" />}
              colors={colors.cognitive}
              description="For reasoning and pattern recognition"
            />
          </Grid>

          {/* Neural Palette */}
          <Grid item xs={12} md={6} lg={3}>
            <ColorPaletteCard
              title="Neural"
              icon={<CpuChipIcon className="w-5 h-5" />}
              colors={colors.neural}
              description="For activity and processing states"
            />
          </Grid>

          {/* Memory Palette */}
          <Grid item xs={12} md={6} lg={3}>
            <ColorPaletteCard
              title="Memory"
              icon={<ChartBarIcon className="w-5 h-5" />}
              colors={colors.memory}
              description="For storage and consolidation"
            />
          </Grid>

          {/* Attention Palette */}
          <Grid item xs={12} md={6} lg={3}>
            <ColorPaletteCard
              title="Attention"
              icon={<SparklesIcon className="w-5 h-5" />}
              colors={colors.attention}
              description="For focus and selection mechanisms"
            />
          </Grid>
        </Grid>
      </Section>

      {/* Typography */}
      <Section title="Typography">
        <Paper sx={{ p: 4 }}>
          <Stack spacing={3}>
            <Typography variant="h1">Heading 1 - Display Large</Typography>
            <Typography variant="h2">Heading 2 - Display Medium</Typography>
            <Typography variant="h3">Heading 3 - Display Small</Typography>
            <Typography variant="h4">Heading 4</Typography>
            <Typography variant="h5">Heading 5</Typography>
            <Typography variant="h6">Heading 6</Typography>
            <Typography variant="body1">
              Body 1 - This is the main body text style used throughout the application.
              It provides optimal readability for content-heavy sections.
            </Typography>
            <Typography variant="body2">
              Body 2 - A smaller body text variant for secondary content and descriptions.
            </Typography>
            <Typography variant="caption">
              Caption - Used for small print and auxiliary information
            </Typography>
            <Typography variant="overline">
              Overline - Label text with uppercase styling
            </Typography>
          </Stack>
        </Paper>
      </Section>

      {/* Components */}
      <Section title="Components">
        <Grid container spacing={3}>
          {/* Buttons */}
          <Grid item xs={12} md={6}>
            <ComponentCard title="Buttons">
              <Stack direction="row" spacing={2} flexWrap="wrap" useFlexGap>
                <Button variant="contained" color="primary">
                  Primary
                </Button>
                <Button variant="contained" color="secondary">
                  Secondary
                </Button>
                <Button variant="outlined" color="primary">
                  Outlined
                </Button>
                <Button variant="text" color="primary">
                  Text
                </Button>
              </Stack>
            </ComponentCard>
          </Grid>

          {/* Chips */}
          <Grid item xs={12} md={6}>
            <ComponentCard title="Chips">
              <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                <Chip label="Cognitive" sx={{ bgcolor: colors.cognitive[500], color: 'white' }} />
                <Chip label="Neural" sx={{ bgcolor: colors.neural[500], color: 'white' }} />
                <Chip label="Memory" sx={{ bgcolor: colors.memory[500], color: 'white' }} />
                <Chip label="Attention" sx={{ bgcolor: colors.attention[500], color: 'white' }} />
                <Chip label="Default" variant="outlined" />
              </Stack>
            </ComponentCard>
          </Grid>

          {/* Text Fields */}
          <Grid item xs={12} md={6}>
            <ComponentCard title="Text Fields">
              <Stack spacing={2}>
                <TextField
                  label="Standard Input"
                  variant="outlined"
                  fullWidth
                  placeholder="Enter text..."
                />
                <TextField
                  label="With Helper Text"
                  variant="outlined"
                  fullWidth
                  helperText="This is helper text"
                />
              </Stack>
            </ComponentCard>
          </Grid>

          {/* Alerts */}
          <Grid item xs={12} md={6}>
            <ComponentCard title="Alerts">
              <Stack spacing={2}>
                <Alert severity="success">Success message</Alert>
                <Alert severity="error">Error message</Alert>
                <Alert severity="warning">Warning message</Alert>
                <Alert severity="info">Info message</Alert>
              </Stack>
            </ComponentCard>
          </Grid>
        </Grid>
      </Section>

      {/* Effects */}
      <Section title="Visual Effects">
        <Grid container spacing={3}>
          {/* Glass Morphism */}
          <Grid item xs={12} md={4}>
            <Box
              sx={{
                ...themeStyles.glassMorphism,
                p: 3,
                borderRadius: getRadius('lg'),
                height: 200,
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                alignItems: 'center',
              }}
            >
              <BeakerIcon className="w-8 h-8 mb-2" style={{ color: colors.cognitive[500] }} />
              <Typography variant="h6">Glass Morphism</Typography>
              <Typography variant="body2" color="text.secondary" align="center">
                Frosted glass effect with backdrop blur
              </Typography>
            </Box>
          </Grid>

          {/* Neumorphism */}
          <Grid item xs={12} md={4}>
            <Box
              sx={{
                ...themeStyles.neumorphism,
                p: 3,
                borderRadius: getRadius('lg'),
                height: 200,
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                alignItems: 'center',
              }}
            >
              <CubeIcon className="w-8 h-8 mb-2" style={{ color: colors.neural[500] }} />
              <Typography variant="h6">Neumorphism</Typography>
              <Typography variant="body2" color="text.secondary" align="center">
                Soft, extruded appearance
              </Typography>
            </Box>
          </Grid>

          {/* Cognitive Glow */}
          <Grid item xs={12} md={4}>
            <Box
              sx={{
                ...themeStyles.cognitiveGlow,
                p: 3,
                borderRadius: getRadius('lg'),
                height: 200,
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                alignItems: 'center',
                backgroundColor: colors.surface.primary,
              }}
            >
              <SparklesIcon className="w-8 h-8 mb-2" style={{ color: colors.cognitive[500] }} />
              <Typography variant="h6">Cognitive Glow</Typography>
              <Typography variant="body2" color="text.secondary" align="center">
                Glowing border effect
              </Typography>
            </Box>
          </Grid>
        </Grid>
      </Section>

      {/* Gradients */}
      <Section title="Gradients">
        <Grid container spacing={2}>
          {colors.chart.gradient.map((gradient, index) => (
            <Grid item xs={6} md={3} key={index}>
              <Box
                sx={{
                  height: 120,
                  background: gradient,
                  borderRadius: getRadius('md'),
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                <Typography variant="body2" sx={{ color: 'white', fontWeight: 600 }}>
                  Gradient {index + 1}
                </Typography>
              </Box>
            </Grid>
          ))}
        </Grid>
      </Section>

      {/* Shadows */}
      <Section title="Shadows">
        <Grid container spacing={3}>
          {Object.entries(shadows).slice(0, 7).map(([key, shadow]) => (
            <Grid item xs={6} md={3} key={key}>
              <Paper
                sx={{
                  p: 3,
                  height: 100,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  boxShadow: shadow,
                }}
              >
                <Typography variant="body2">{key}</Typography>
              </Paper>
            </Grid>
          ))}
        </Grid>
      </Section>

      {/* Metrics Visualization */}
      <Section title="Brain-Inspired Metrics">
        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <MetricCard
              title="Cognitive Processing"
              metrics={cognitiveMetrics}
              icon={<BrainIcon className="w-6 h-6" />}
              gradient={createGradient([colors.cognitive[400], colors.cognitive[600]])}
            />
          </Grid>
          <Grid item xs={12} md={4}>
            <MetricCard
              title="Neural Activity"
              metrics={neuralActivity}
              icon={<CpuChipIcon className="w-6 h-6" />}
              gradient={createGradient([colors.neural[400], colors.neural[600]])}
            />
          </Grid>
          <Grid item xs={12} md={4}>
            <MetricCard
              title="Memory Systems"
              metrics={memoryStatus}
              icon={<ChartBarIcon className="w-6 h-6" />}
              gradient={createGradient([colors.memory[400], colors.memory[600]])}
            />
          </Grid>
        </Grid>
      </Section>
    </Box>
  );
};

// Helper Components
const Section: React.FC<{ title: string; children: React.ReactNode }> = ({ title, children }) => (
  <Box sx={{ mb: 8 }}>
    <Typography variant="h4" sx={{ mb: 4, fontWeight: 600 }}>
      {title}
    </Typography>
    {children}
  </Box>
);

const ColorPaletteCard: React.FC<{
  title: string;
  icon: React.ReactNode;
  colors: any;
  description: string;
}> = ({ title, icon, colors, description }) => {
  const { mode } = useTheme();
  
  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2, gap: 1 }}>
          {icon}
          <Typography variant="h6">{title}</Typography>
        </Box>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          {description}
        </Typography>
        <Stack spacing={1}>
          {Object.entries(colors).slice(3, 8).map(([shade, color]) => (
            <Box
              key={shade}
              sx={{
                display: 'flex',
                alignItems: 'center',
                gap: 2,
              }}
            >
              <Box
                sx={{
                  width: 40,
                  height: 40,
                  bgcolor: color as string,
                  borderRadius: 1,
                  boxShadow: 1,
                }}
              />
              <Box>
                <Typography variant="caption" sx={{ fontFamily: 'monospace' }}>
                  {shade}
                </Typography>
                <Typography variant="caption" display="block" color="text.secondary">
                  {color}
                </Typography>
              </Box>
            </Box>
          ))}
        </Stack>
      </CardContent>
    </Card>
  );
};

const ComponentCard: React.FC<{ title: string; children: React.ReactNode }> = ({ title, children }) => (
  <Card sx={{ height: '100%' }}>
    <CardContent>
      <Typography variant="h6" sx={{ mb: 3 }}>
        {title}
      </Typography>
      {children}
    </CardContent>
  </Card>
);

const MetricCard: React.FC<{
  title: string;
  metrics: Array<{ label: string; value: number; color: string }>;
  icon: React.ReactNode;
  gradient: string;
}> = ({ title, metrics, icon, gradient }) => (
  <Card
    sx={{
      height: '100%',
      background: gradient,
      color: 'white',
    }}
  >
    <CardContent>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 3, gap: 1 }}>
        {icon}
        <Typography variant="h6">{title}</Typography>
      </Box>
      <Stack spacing={2}>
        {metrics.map((metric) => (
          <Box key={metric.label}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
              <Typography variant="body2">{metric.label}</Typography>
              <Typography variant="body2">{metric.value}%</Typography>
            </Box>
            <Box
              sx={{
                height: 8,
                bgcolor: 'rgba(255, 255, 255, 0.2)',
                borderRadius: 1,
                overflow: 'hidden',
              }}
            >
              <Box
                sx={{
                  height: '100%',
                  width: `${metric.value}%`,
                  bgcolor: 'rgba(255, 255, 255, 0.8)',
                  transition: 'width 0.5s ease',
                }}
              />
            </Box>
          </Box>
        ))}
      </Stack>
    </CardContent>
  </Card>
);

export default ThemeShowcase;