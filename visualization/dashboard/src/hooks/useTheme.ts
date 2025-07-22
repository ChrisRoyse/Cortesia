/**
 * LLMKG Theme Hook
 * Advanced theme management utilities
 */

import { useTheme as useMuiTheme } from '@mui/material/styles';
import { useMediaQuery } from '@mui/material';
import { useCallback, useMemo } from 'react';
import { LLMKGTheme, llmkgColors } from '../theme';
import { useThemeContext } from '../components/ThemeProvider/ThemeProvider';

export interface UseThemeReturn {
  theme: LLMKGTheme;
  mode: 'light' | 'dark';
  toggleTheme: () => void;
  setTheme: (mode: 'light' | 'dark') => void;
  systemPreference: 'light' | 'dark';
  colors: LLMKGTheme['llmkg']['colors'];
  spacing: LLMKGTheme['llmkg']['spacing'];
  typography: LLMKGTheme['llmkg']['typography'];
  shadows: LLMKGTheme['llmkg']['shadows'];
  transitions: LLMKGTheme['llmkg']['transitions'];
  borderRadius: LLMKGTheme['llmkg']['borderRadius'];
  breakpoints: LLMKGTheme['llmkg']['breakpoints'];
  zIndex: LLMKGTheme['llmkg']['zIndex'];
  prefersDarkMode: boolean;
  prefersReducedMotion: boolean;
  prefersHighContrast: boolean;
  getColor: (path: string) => string;
  getSpacing: (...args: (keyof LLMKGTheme['llmkg']['spacing'] | number)[]) => string;
  getShadow: (key: keyof LLMKGTheme['llmkg']['shadows']) => string;
  getRadius: (key: keyof LLMKGTheme['llmkg']['borderRadius']) => string;
  createGradient: (colors: string[], angle?: number) => string;
  withAlpha: (color: string, alpha: number) => string;
  mixColors: (color1: string, color2: string, ratio?: number) => string;
  getContrastColor: (background: string) => string;
}

export const useTheme = (): UseThemeReturn => {
  const muiTheme = useMuiTheme() as LLMKGTheme;
  const themeContext = useThemeContext();
  
  // Media queries for accessibility
  const prefersDarkMode = useMediaQuery('(prefers-color-scheme: dark)');
  const prefersReducedMotion = useMediaQuery('(prefers-reduced-motion: reduce)');
  const prefersHighContrast = useMediaQuery('(prefers-contrast: high)');
  
  // Color getter with path support (e.g., "cognitive.500")
  const getColor = useCallback((path: string): string => {
    const parts = path.split('.');
    let current: any = muiTheme.llmkg.colors;
    
    for (const part of parts) {
      if (current && typeof current === 'object' && part in current) {
        current = current[part];
      } else {
        console.warn(`Color path "${path}" not found`);
        return '#000000';
      }
    }
    
    return typeof current === 'string' ? current : '#000000';
  }, [muiTheme]);
  
  // Spacing getter with multiple arguments support
  const getSpacing = useCallback((...args: (keyof LLMKGTheme['llmkg']['spacing'] | number)[]): string => {
    return args
      .map(arg => {
        if (typeof arg === 'number') {
          return `${arg * 8}px`; // 8px base spacing
        }
        return muiTheme.llmkg.spacing[arg] || '0';
      })
      .join(' ');
  }, [muiTheme]);
  
  // Shadow getter
  const getShadow = useCallback((key: keyof LLMKGTheme['llmkg']['shadows']): string => {
    return muiTheme.llmkg.shadows[key] || 'none';
  }, [muiTheme]);
  
  // Border radius getter
  const getRadius = useCallback((key: keyof LLMKGTheme['llmkg']['borderRadius']): string => {
    return muiTheme.llmkg.borderRadius[key] || '0';
  }, [muiTheme]);
  
  // Create gradient
  const createGradient = useCallback((colors: string[], angle = 135): string => {
    if (colors.length < 2) {
      console.warn('Gradient requires at least 2 colors');
      return colors[0] || 'transparent';
    }
    
    const stops = colors
      .map((color, index) => {
        const position = (index / (colors.length - 1)) * 100;
        return `${color} ${position}%`;
      })
      .join(', ');
    
    return `linear-gradient(${angle}deg, ${stops})`;
  }, []);
  
  // Color utilities from the colors module
  const { withAlpha, mixColors, getContrastColor } = useMemo(() => llmkgColors.utils, []);
  
  return {
    theme: muiTheme,
    mode: themeContext.mode,
    toggleTheme: themeContext.toggleTheme,
    setTheme: themeContext.setTheme,
    systemPreference: themeContext.systemPreference,
    colors: muiTheme.llmkg.colors,
    spacing: muiTheme.llmkg.spacing,
    typography: muiTheme.llmkg.typography,
    shadows: muiTheme.llmkg.shadows,
    transitions: muiTheme.llmkg.transitions,
    borderRadius: muiTheme.llmkg.borderRadius,
    breakpoints: muiTheme.llmkg.breakpoints,
    zIndex: muiTheme.llmkg.zIndex,
    prefersDarkMode,
    prefersReducedMotion,
    prefersHighContrast,
    getColor,
    getSpacing,
    getShadow,
    getRadius,
    createGradient,
    withAlpha,
    mixColors,
    getContrastColor,
  };
};

// Theme-aware style utilities
export const useThemeStyles = () => {
  const { theme, mode, prefersReducedMotion } = useTheme();
  
  const styles = useMemo(() => ({
    // Glass morphism effect
    glassMorphism: {
      background: mode === 'light' 
        ? 'rgba(255, 255, 255, 0.7)' 
        : 'rgba(26, 26, 26, 0.7)',
      backdropFilter: 'blur(10px)',
      WebkitBackdropFilter: 'blur(10px)',
      border: `1px solid ${theme.llmkg.colors.border.primary}`,
    },
    
    // Neumorphism effect
    neumorphism: {
      background: theme.llmkg.colors.background.primary,
      boxShadow: mode === 'light'
        ? `20px 20px 60px #bebebe, -20px -20px 60px #ffffff`
        : `20px 20px 60px #0a0a0a, -20px -20px 60px #141414`,
    },
    
    // Cognitive glow effect
    cognitiveGlow: {
      boxShadow: `0 0 20px ${llmkgColors.utils.withAlpha(theme.llmkg.colors.cognitive[500], 0.3)}`,
      border: `1px solid ${theme.llmkg.colors.cognitive[500]}`,
    },
    
    // Neural pulse animation
    neuralPulse: prefersReducedMotion ? {} : {
      animation: 'neural-pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      '@keyframes neural-pulse': {
        '0%, 100%': {
          opacity: 1,
          transform: 'scale(1)',
        },
        '50%': {
          opacity: 0.8,
          transform: 'scale(1.05)',
        },
      },
    },
    
    // Memory fade transition
    memoryFade: {
      transition: prefersReducedMotion 
        ? 'none' 
        : 'opacity 500ms cubic-bezier(0.4, 0, 0.2, 1)',
    },
    
    // Attention focus ring
    attentionFocus: {
      '&:focus-visible': {
        outline: 'none',
        boxShadow: `0 0 0 3px ${llmkgColors.utils.withAlpha(theme.llmkg.colors.attention[500], 0.5)}`,
      },
    },
  }), [theme, mode, prefersReducedMotion]);
  
  return styles;
};

// Responsive theme utilities
export const useResponsiveTheme = () => {
  const theme = useTheme();
  const { breakpoints } = theme;
  
  // Check current breakpoint
  const isXs = useMediaQuery(`(max-width: ${breakpoints.sm - 1}px)`);
  const isSm = useMediaQuery(`(min-width: ${breakpoints.sm}px) and (max-width: ${breakpoints.md - 1}px)`);
  const isMd = useMediaQuery(`(min-width: ${breakpoints.md}px) and (max-width: ${breakpoints.lg - 1}px)`);
  const isLg = useMediaQuery(`(min-width: ${breakpoints.lg}px) and (max-width: ${breakpoints.xl - 1}px)`);
  const isXl = useMediaQuery(`(min-width: ${breakpoints.xl}px) and (max-width: ${breakpoints['2xl'] - 1}px)`);
  const is2xl = useMediaQuery(`(min-width: ${breakpoints['2xl']}px)`);
  
  // Breakpoint utilities
  const up = useCallback((breakpoint: keyof typeof breakpoints) => {
    return useMediaQuery(`(min-width: ${breakpoints[breakpoint]}px)`);
  }, [breakpoints]);
  
  const down = useCallback((breakpoint: keyof typeof breakpoints) => {
    return useMediaQuery(`(max-width: ${breakpoints[breakpoint] - 1}px)`);
  }, [breakpoints]);
  
  const between = useCallback((start: keyof typeof breakpoints, end: keyof typeof breakpoints) => {
    return useMediaQuery(`(min-width: ${breakpoints[start]}px) and (max-width: ${breakpoints[end] - 1}px)`);
  }, [breakpoints]);
  
  const currentBreakpoint = useMemo(() => {
    if (is2xl) return '2xl';
    if (isXl) return 'xl';
    if (isLg) return 'lg';
    if (isMd) return 'md';
    if (isSm) return 'sm';
    return 'xs';
  }, [is2xl, isXl, isLg, isMd, isSm]);
  
  return {
    isXs,
    isSm,
    isMd,
    isLg,
    isXl,
    is2xl,
    up,
    down,
    between,
    currentBreakpoint,
  };
};

export default useTheme;