/**
 * LLMKG Theme System
 * Comprehensive theming for brain-inspired visualization
 */

import { createTheme, ThemeOptions, Theme as MuiTheme } from '@mui/material/styles';
import { llmkgColors, ColorPalette } from './colors';
import { llmkgTypography, textStyles } from './typography';

export type ThemeMode = 'light' | 'dark';

export interface LLMKGTheme extends MuiTheme {
  llmkg: {
    colors: ColorPalette;
    typography: typeof llmkgTypography;
    spacing: typeof spacing;
    borderRadius: typeof borderRadius;
    shadows: typeof shadows;
    transitions: typeof transitions;
    breakpoints: typeof breakpoints;
    zIndex: typeof zIndex;
  };
}

// Spacing system (8px base)
export const spacing = {
  xs: '0.25rem',   // 4px
  sm: '0.5rem',    // 8px
  md: '1rem',      // 16px
  lg: '1.5rem',    // 24px
  xl: '2rem',      // 32px
  '2xl': '3rem',   // 48px
  '3xl': '4rem',   // 64px
  '4xl': '6rem',   // 96px
  '5xl': '8rem',   // 128px
} as const;

// Border radius system
export const borderRadius = {
  none: '0',
  sm: '0.125rem',    // 2px
  base: '0.25rem',   // 4px
  md: '0.375rem',    // 6px
  lg: '0.5rem',      // 8px
  xl: '0.75rem',     // 12px
  '2xl': '1rem',     // 16px
  '3xl': '1.5rem',   // 24px
  full: '9999px',
} as const;

// Shadow system
export const shadows = {
  none: 'none',
  sm: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
  base: '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)',
  md: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
  lg: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
  xl: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
  '2xl': '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
  inner: 'inset 0 2px 4px 0 rgba(0, 0, 0, 0.06)',
  // Colored shadows for hover effects
  cognitive: '0 10px 30px -10px rgba(99, 102, 241, 0.3)',
  neural: '0 10px 30px -10px rgba(239, 68, 68, 0.3)',
  memory: '0 10px 30px -10px rgba(34, 197, 94, 0.3)',
  attention: '0 10px 30px -10px rgba(6, 182, 212, 0.3)',
} as const;

// Transition system
export const transitions = {
  duration: {
    instant: '0ms',
    fast: '150ms',
    base: '200ms',
    slow: '300ms',
    slower: '500ms',
  },
  timing: {
    ease: 'cubic-bezier(0.4, 0, 0.2, 1)',
    easeIn: 'cubic-bezier(0.4, 0, 1, 1)',
    easeOut: 'cubic-bezier(0, 0, 0.2, 1)',
    easeInOut: 'cubic-bezier(0.4, 0, 0.2, 1)',
    bounce: 'cubic-bezier(0.68, -0.55, 0.265, 1.55)',
  },
  all: 'all 200ms cubic-bezier(0.4, 0, 0.2, 1)',
  colors: 'background-color 200ms cubic-bezier(0.4, 0, 0.2, 1), border-color 200ms cubic-bezier(0.4, 0, 0.2, 1), color 200ms cubic-bezier(0.4, 0, 0.2, 1)',
  transform: 'transform 200ms cubic-bezier(0.4, 0, 0.2, 1)',
  opacity: 'opacity 200ms cubic-bezier(0.4, 0, 0.2, 1)',
} as const;

// Breakpoints
export const breakpoints = {
  xs: 0,
  sm: 640,
  md: 768,
  lg: 1024,
  xl: 1280,
  '2xl': 1536,
} as const;

// Z-index system
export const zIndex = {
  hide: -1,
  base: 0,
  dropdown: 1000,
  sticky: 1100,
  fixed: 1200,
  overlay: 1300,
  modal: 1400,
  popover: 1500,
  tooltip: 1600,
  toast: 1700,
} as const;

// Create base theme options
const createBaseThemeOptions = (mode: ThemeMode): ThemeOptions => {
  const colors = mode === 'light' ? llmkgColors.light : llmkgColors.dark;
  
  return {
    palette: {
      mode,
      primary: {
        main: colors.cognitive[500],
        light: colors.cognitive[400],
        dark: colors.cognitive[600],
        contrastText: '#ffffff',
      },
      secondary: {
        main: colors.neural[500],
        light: colors.neural[400],
        dark: colors.neural[600],
        contrastText: '#ffffff',
      },
      success: {
        main: colors.status.success,
        light: colors.memory[400],
        dark: colors.memory[600],
        contrastText: '#ffffff',
      },
      error: {
        main: colors.status.error,
        light: colors.neural[400],
        dark: colors.neural[600],
        contrastText: '#ffffff',
      },
      warning: {
        main: colors.status.warning,
        contrastText: '#000000',
      },
      info: {
        main: colors.status.info,
        light: colors.attention[400],
        dark: colors.attention[600],
        contrastText: '#ffffff',
      },
      background: {
        default: colors.background.primary,
        paper: colors.surface.primary,
      },
      text: {
        primary: colors.text.primary,
        secondary: colors.text.secondary,
        disabled: colors.text.disabled,
      },
      divider: colors.border.primary,
    },
    typography: {
      fontFamily: llmkgTypography.fontFamily.sans,
      h1: textStyles.displayLarge,
      h2: textStyles.displayMedium,
      h3: textStyles.displaySmall,
      h4: {
        ...textStyles.displaySmall,
        fontSize: llmkgTypography.fontSize['2xl'],
      },
      h5: {
        ...textStyles.displaySmall,
        fontSize: llmkgTypography.fontSize.xl,
      },
      h6: {
        ...textStyles.displaySmall,
        fontSize: llmkgTypography.fontSize.lg,
      },
      body1: textStyles.bodyMedium,
      body2: textStyles.bodySmall,
      subtitle1: textStyles.bodyLarge,
      subtitle2: textStyles.bodyMedium,
      button: textStyles.labelLarge,
      caption: textStyles.labelMedium,
      overline: textStyles.labelSmall,
    },
    shape: {
      borderRadius: parseInt(borderRadius.md),
    },
    breakpoints: {
      values: breakpoints,
    },
    zIndex: {
      appBar: zIndex.sticky,
      drawer: zIndex.overlay,
      modal: zIndex.modal,
      snackbar: zIndex.toast,
      tooltip: zIndex.tooltip,
    },
    components: {
      MuiButton: {
        styleOverrides: {
          root: {
            textTransform: 'none',
            borderRadius: borderRadius.md,
            fontWeight: llmkgTypography.fontWeight.medium,
            transition: transitions.all,
            '&:hover': {
              transform: 'translateY(-1px)',
              boxShadow: shadows.md,
            },
          },
          containedPrimary: {
            background: `linear-gradient(135deg, ${colors.cognitive[500]} 0%, ${colors.cognitive[600]} 100%)`,
            '&:hover': {
              background: `linear-gradient(135deg, ${colors.cognitive[600]} 0%, ${colors.cognitive[700]} 100%)`,
            },
          },
        },
      },
      MuiPaper: {
        styleOverrides: {
          root: {
            backgroundImage: 'none',
            backgroundColor: colors.surface.primary,
            borderColor: colors.border.primary,
          },
          elevation1: {
            boxShadow: shadows.sm,
          },
          elevation2: {
            boxShadow: shadows.base,
          },
          elevation3: {
            boxShadow: shadows.md,
          },
          elevation4: {
            boxShadow: shadows.lg,
          },
        },
      },
      MuiCard: {
        styleOverrides: {
          root: {
            borderRadius: borderRadius.lg,
            border: `1px solid ${colors.border.primary}`,
            transition: transitions.all,
            '&:hover': {
              transform: 'translateY(-2px)',
              boxShadow: shadows.lg,
              borderColor: colors.border.focus,
            },
          },
        },
      },
      MuiChip: {
        styleOverrides: {
          root: {
            borderRadius: borderRadius.full,
            fontWeight: llmkgTypography.fontWeight.medium,
          },
        },
      },
      MuiTooltip: {
        styleOverrides: {
          tooltip: {
            backgroundColor: mode === 'light' ? colors.text.primary : colors.surface.secondary,
            color: mode === 'light' ? colors.text.inverse : colors.text.primary,
            ...textStyles.bodySmall,
            borderRadius: borderRadius.md,
            padding: `${spacing.xs} ${spacing.sm}`,
          },
        },
      },
      MuiTextField: {
        styleOverrides: {
          root: {
            '& .MuiOutlinedInput-root': {
              borderRadius: borderRadius.md,
              transition: transitions.all,
              '&:hover': {
                '& .MuiOutlinedInput-notchedOutline': {
                  borderColor: colors.border.focus,
                },
              },
              '&.Mui-focused': {
                '& .MuiOutlinedInput-notchedOutline': {
                  borderColor: colors.cognitive[500],
                  borderWidth: 2,
                },
              },
            },
          },
        },
      },
      MuiAlert: {
        styleOverrides: {
          root: {
            borderRadius: borderRadius.md,
          },
          standardSuccess: {
            backgroundColor: llmkgColors.utils.withAlpha(colors.status.success, 0.1),
            color: colors.status.success,
            '& .MuiAlert-icon': {
              color: colors.status.success,
            },
          },
          standardError: {
            backgroundColor: llmkgColors.utils.withAlpha(colors.status.error, 0.1),
            color: colors.status.error,
            '& .MuiAlert-icon': {
              color: colors.status.error,
            },
          },
          standardWarning: {
            backgroundColor: llmkgColors.utils.withAlpha(colors.status.warning, 0.1),
            color: colors.status.warning,
            '& .MuiAlert-icon': {
              color: colors.status.warning,
            },
          },
          standardInfo: {
            backgroundColor: llmkgColors.utils.withAlpha(colors.status.info, 0.1),
            color: colors.status.info,
            '& .MuiAlert-icon': {
              color: colors.status.info,
            },
          },
        },
      },
    },
  };
};

// Create LLMKG theme
export const createLLMKGTheme = (mode: ThemeMode = 'light'): LLMKGTheme => {
  const baseTheme = createTheme(createBaseThemeOptions(mode));
  const colors = mode === 'light' ? llmkgColors.light : llmkgColors.dark;
  
  return {
    ...baseTheme,
    llmkg: {
      colors,
      typography: llmkgTypography,
      spacing,
      borderRadius,
      shadows,
      transitions,
      breakpoints,
      zIndex,
    },
  } as LLMKGTheme;
};

// Export pre-configured themes
export const lightTheme = createLLMKGTheme('light');
export const darkTheme = createLLMKGTheme('dark');

// Theme utilities
export const getThemeValue = <T extends keyof LLMKGTheme['llmkg']>(
  theme: LLMKGTheme,
  category: T,
  key: keyof LLMKGTheme['llmkg'][T]
): any => {
  return theme.llmkg[category][key];
};

export const createResponsiveValue = (
  values: { xs?: any; sm?: any; md?: any; lg?: any; xl?: any; '2xl'?: any }
) => {
  return Object.entries(values).reduce((acc, [breakpoint, value]) => {
    if (value !== undefined) {
      acc[`@media (min-width: ${breakpoints[breakpoint as keyof typeof breakpoints]}px)`] = value;
    }
    return acc;
  }, {} as Record<string, any>);
};

export const themeUtils = {
  getThemeValue,
  createResponsiveValue,
};

// Re-export everything from colors and typography
export * from './colors';
export * from './typography';

export type { LLMKGTheme, ThemeMode };