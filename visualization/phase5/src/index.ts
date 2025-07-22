// Phase 5: Interactive System Architecture Diagram
// Main entry point for LLMKG Phase 5 components and utilities

// Core Components
export * from './components';

// Custom Hooks
export * from './hooks';

// Core Engine
export * from './core';

// Type Definitions
export * from './types';

// Default theme configuration
import { ThemeConfiguration } from './types';

export const defaultTheme: ThemeConfiguration = {
  name: 'llmkg-default',
  colors: {
    primary: '#2563eb',
    secondary: '#64748b',
    background: '#f8fafc',
    surface: '#ffffff',
    text: '#1e293b',
    highlight: '#fbbf24',
    activity: '#10b981',
    cognitive: {
      subcortical: '#dc2626',
      cortical: '#2563eb',
      thalamic: '#7c3aed',
    },
    mcp: {
      primary: '#059669',
      secondary: '#10b981',
    },
    storage: {
      primary: '#ea580c',
      secondary: '#fb923c',
    },
    network: {
      primary: '#7c2d12',
      secondary: '#a3a3a3',
    },
    default: '#64748b',
  },
  connections: {
    'excitation': { stroke: '#10b981', strokeWidth: 2, opacity: 0.7 },
    'inhibition': { stroke: '#dc2626', strokeWidth: 2, opacity: 0.7 },
    'bidirectional': { stroke: '#2563eb', strokeWidth: 3, opacity: 0.8 },
    'data-flow': { stroke: '#7c3aed', strokeWidth: 1, opacity: 0.6 },
    'dependency': { stroke: '#64748b', strokeWidth: 1, opacity: 0.5 },
  },
  fonts: {
    primary: 'Inter, system-ui, sans-serif',
    monospace: 'JetBrains Mono, Consolas, monospace',
  },
};

// Utility functions for theme manipulation
export const createCustomTheme = (overrides: Partial<ThemeConfiguration>): ThemeConfiguration => {
  return {
    ...defaultTheme,
    ...overrides,
    colors: {
      ...defaultTheme.colors,
      ...overrides.colors,
    },
    connections: {
      ...defaultTheme.connections,
      ...overrides.connections,
    },
    fonts: {
      ...defaultTheme.fonts,
      ...overrides.fonts,
    },
  };
};

// Version information
export const VERSION = '1.0.0';
export const PHASE = 5;