/**
 * LLMKG Hooks Index
 * Central export for all custom hooks
 */

// Data hooks
export { useCognitivePatterns } from './useCognitivePatterns';
export { useKnowledgeGraph } from './useKnowledgeGraph';
export { useMemoryMetrics } from './useMemoryMetrics';
export { useNeuralActivity } from './useNeuralActivity';
export { useRealTimeData } from './useRealTimeData';

// UI hooks
export { useBreakpoint } from './useBreakpoint';
export { useOnClickOutside } from './useOnClickOutside';
export { useResponsiveLayout } from './useResponsiveLayout';

// Theme hooks
export { useTheme, useThemeStyles, useResponsiveTheme } from './useTheme';
export { useThemeContext } from '../components/ThemeProvider/ThemeProvider';