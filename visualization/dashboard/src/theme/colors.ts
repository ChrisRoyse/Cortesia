/**
 * LLMKG Color System
 * Brain-inspired color palettes for cognitive visualization
 */

export interface ColorShade {
  50: string;
  100: string;
  200: string;
  300: string;
  400: string;
  500: string;
  600: string;
  700: string;
  800: string;
  900: string;
  950: string;
}

export interface StatusColors {
  success: string;
  warning: string;
  error: string;
  info: string;
}

export interface ColorPalette {
  cognitive: ColorShade;
  neural: ColorShade;
  memory: ColorShade;
  attention: ColorShade;
  background: {
    primary: string;
    secondary: string;
    tertiary: string;
    elevated: string;
  };
  surface: {
    primary: string;
    secondary: string;
    hover: string;
    active: string;
  };
  text: {
    primary: string;
    secondary: string;
    tertiary: string;
    disabled: string;
    inverse: string;
  };
  border: {
    primary: string;
    secondary: string;
    focus: string;
  };
  status: StatusColors;
  chart: {
    primary: string[];
    secondary: string[];
    gradient: string[];
  };
}

// Brain-inspired color scales
const cognitiveScale: ColorShade = {
  50: '#eef2ff',
  100: '#e0e7ff',
  200: '#c7d2fe',
  300: '#a5b4fc',
  400: '#818cf8',
  500: '#6366f1',  // Primary cognitive indigo
  600: '#4f46e5',
  700: '#4338ca',
  800: '#3730a3',
  900: '#312e81',
  950: '#1e1b4b',
};

const neuralScale: ColorShade = {
  50: '#fef2f2',
  100: '#fee2e2',
  200: '#fecaca',
  300: '#fca5a5',
  400: '#f87171',
  500: '#ef4444',  // Primary neural red
  600: '#dc2626',
  700: '#b91c1c',
  800: '#991b1b',
  900: '#7f1d1d',
  950: '#450a0a',
};

const memoryScale: ColorShade = {
  50: '#f0fdf4',
  100: '#dcfce7',
  200: '#bbf7d0',
  300: '#86efac',
  400: '#4ade80',
  500: '#22c55e',  // Primary memory green
  600: '#16a34a',
  700: '#15803d',
  800: '#166534',
  900: '#14532d',
  950: '#052e16',
};

const attentionScale: ColorShade = {
  50: '#ecfeff',
  100: '#cffafe',
  200: '#a5f3fc',
  300: '#67e8f9',
  400: '#22d3ee',
  500: '#06b6d4',  // Primary attention cyan
  600: '#0891b2',
  700: '#0e7490',
  800: '#155e75',
  900: '#164e63',
  950: '#083344',
};

// Light theme palette
export const lightPalette: ColorPalette = {
  cognitive: cognitiveScale,
  neural: neuralScale,
  memory: memoryScale,
  attention: attentionScale,
  background: {
    primary: '#ffffff',
    secondary: '#f9fafb',
    tertiary: '#f3f4f6',
    elevated: '#ffffff',
  },
  surface: {
    primary: '#ffffff',
    secondary: '#f9fafb',
    hover: 'rgba(0, 0, 0, 0.05)',
    active: 'rgba(0, 0, 0, 0.1)',
  },
  text: {
    primary: '#111827',
    secondary: '#4b5563',
    tertiary: '#9ca3af',
    disabled: '#d1d5db',
    inverse: '#ffffff',
  },
  border: {
    primary: '#e5e7eb',
    secondary: '#f3f4f6',
    focus: cognitiveScale[500],
  },
  status: {
    success: memoryScale[500],
    warning: '#f59e0b',
    error: neuralScale[500],
    info: attentionScale[500],
  },
  chart: {
    primary: [
      cognitiveScale[500],
      neuralScale[500],
      memoryScale[500],
      attentionScale[500],
      '#8b5cf6',
      '#f97316',
    ],
    secondary: [
      cognitiveScale[300],
      neuralScale[300],
      memoryScale[300],
      attentionScale[300],
      '#c084fc',
      '#fdba74',
    ],
    gradient: [
      'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
      'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
      'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)',
    ],
  },
};

// Dark theme palette
export const darkPalette: ColorPalette = {
  cognitive: cognitiveScale,
  neural: neuralScale,
  memory: memoryScale,
  attention: attentionScale,
  background: {
    primary: '#0f0f0f',
    secondary: '#1a1a1a',
    tertiary: '#262626',
    elevated: '#1f1f1f',
  },
  surface: {
    primary: '#1a1a1a',
    secondary: '#262626',
    hover: 'rgba(255, 255, 255, 0.08)',
    active: 'rgba(255, 255, 255, 0.12)',
  },
  text: {
    primary: '#f9fafb',
    secondary: '#d1d5db',
    tertiary: '#9ca3af',
    disabled: '#4b5563',
    inverse: '#111827',
  },
  border: {
    primary: '#374151',
    secondary: '#1f2937',
    focus: cognitiveScale[400],
  },
  status: {
    success: memoryScale[400],
    warning: '#fbbf24',
    error: neuralScale[400],
    info: attentionScale[400],
  },
  chart: {
    primary: [
      cognitiveScale[400],
      neuralScale[400],
      memoryScale[400],
      attentionScale[400],
      '#a78bfa',
      '#fb923c',
    ],
    secondary: [
      cognitiveScale[600],
      neuralScale[600],
      memoryScale[600],
      attentionScale[600],
      '#7c3aed',
      '#ea580c',
    ],
    gradient: [
      'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
      'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
      'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)',
    ],
  },
};

// Semantic color mappings for LLMKG concepts
export const semanticColors = {
  // Cognitive processes
  patternRecognition: cognitiveScale[500],
  reasoning: cognitiveScale[600],
  inference: cognitiveScale[400],
  
  // Neural activity
  highActivity: neuralScale[500],
  mediumActivity: neuralScale[400],
  lowActivity: neuralScale[300],
  
  // Memory systems
  workingMemory: memoryScale[400],
  longTermMemory: memoryScale[600],
  consolidation: memoryScale[500],
  
  // Attention mechanisms
  focused: attentionScale[500],
  distributed: attentionScale[400],
  selective: attentionScale[600],
};

// Accessibility-focused color utilities
export const getContrastColor = (background: string): string => {
  // Simple contrast calculation
  const rgb = background.match(/\d+/g);
  if (!rgb) return '#000000';
  
  const [r, g, b] = rgb.map(Number);
  const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
  
  return luminance > 0.5 ? '#000000' : '#ffffff';
};

// Generate alpha variants
export const withAlpha = (color: string, alpha: number): string => {
  const rgb = color.match(/\d+/g);
  if (!rgb) return color;
  
  const [r, g, b] = rgb.map(Number);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
};

// Color mixing utility
export const mixColors = (color1: string, color2: string, ratio = 0.5): string => {
  const rgb1 = color1.match(/\d+/g)?.map(Number) || [0, 0, 0];
  const rgb2 = color2.match(/\d+/g)?.map(Number) || [0, 0, 0];
  
  const mixed = rgb1.map((val, i) => Math.round(val * (1 - ratio) + rgb2[i] * ratio));
  return `rgb(${mixed.join(', ')})`;
};

export const llmkgColors = {
  light: lightPalette,
  dark: darkPalette,
  semantic: semanticColors,
  utils: {
    getContrastColor,
    withAlpha,
    mixColors,
  },
};