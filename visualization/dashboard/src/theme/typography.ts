/**
 * LLMKG Typography System
 * Responsive and accessible typography configuration
 */

export interface FontWeight {
  thin: number;
  light: number;
  regular: number;
  medium: number;
  semibold: number;
  bold: number;
  extrabold: number;
  black: number;
}

export interface FontSize {
  xs: string;
  sm: string;
  base: string;
  lg: string;
  xl: string;
  '2xl': string;
  '3xl': string;
  '4xl': string;
  '5xl': string;
  '6xl': string;
  '7xl': string;
  '8xl': string;
  '9xl': string;
}

export interface LineHeight {
  none: number;
  tight: number;
  snug: number;
  normal: number;
  relaxed: number;
  loose: number;
}

export interface LetterSpacing {
  tighter: string;
  tight: string;
  normal: string;
  wide: string;
  wider: string;
  widest: string;
}

export interface Typography {
  fontFamily: {
    sans: string;
    serif: string;
    mono: string;
    display: string;
  };
  fontSize: FontSize;
  fontWeight: FontWeight;
  lineHeight: LineHeight;
  letterSpacing: LetterSpacing;
}

// Font stacks with fallbacks
const systemFontStack = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji"';
const monoFontStack = '"SF Mono", "Monaco", "Inconsolata", "Fira Code", "Droid Sans Mono", "Courier New", monospace';

export const typography: Typography = {
  fontFamily: {
    sans: `"Inter", ${systemFontStack}`,
    serif: '"Georgia", "Cambria", "Times New Roman", Times, serif',
    mono: monoFontStack,
    display: '"Inter Display", "SF Pro Display", "Helvetica Neue", Arial, sans-serif',
  },
  fontSize: {
    xs: '0.75rem',     // 12px
    sm: '0.875rem',    // 14px
    base: '1rem',      // 16px
    lg: '1.125rem',    // 18px
    xl: '1.25rem',     // 20px
    '2xl': '1.5rem',   // 24px
    '3xl': '1.875rem', // 30px
    '4xl': '2.25rem',  // 36px
    '5xl': '3rem',     // 48px
    '6xl': '3.75rem',  // 60px
    '7xl': '4.5rem',   // 72px
    '8xl': '6rem',     // 96px
    '9xl': '8rem',     // 128px
  },
  fontWeight: {
    thin: 100,
    light: 300,
    regular: 400,
    medium: 500,
    semibold: 600,
    bold: 700,
    extrabold: 800,
    black: 900,
  },
  lineHeight: {
    none: 1,
    tight: 1.25,
    snug: 1.375,
    normal: 1.5,
    relaxed: 1.625,
    loose: 2,
  },
  letterSpacing: {
    tighter: '-0.05em',
    tight: '-0.025em',
    normal: '0em',
    wide: '0.025em',
    wider: '0.05em',
    widest: '0.1em',
  },
};

// Responsive typography scales
export const responsiveFontSizes = {
  // Mobile first approach
  base: {
    h1: { fontSize: typography.fontSize['4xl'], lineHeight: typography.lineHeight.tight },
    h2: { fontSize: typography.fontSize['3xl'], lineHeight: typography.lineHeight.tight },
    h3: { fontSize: typography.fontSize['2xl'], lineHeight: typography.lineHeight.snug },
    h4: { fontSize: typography.fontSize.xl, lineHeight: typography.lineHeight.snug },
    h5: { fontSize: typography.fontSize.lg, lineHeight: typography.lineHeight.normal },
    h6: { fontSize: typography.fontSize.base, lineHeight: typography.lineHeight.normal },
    body1: { fontSize: typography.fontSize.base, lineHeight: typography.lineHeight.relaxed },
    body2: { fontSize: typography.fontSize.sm, lineHeight: typography.lineHeight.relaxed },
    caption: { fontSize: typography.fontSize.xs, lineHeight: typography.lineHeight.normal },
    overline: { fontSize: typography.fontSize.xs, lineHeight: typography.lineHeight.normal, letterSpacing: typography.letterSpacing.wider },
  },
  // Tablet and above
  md: {
    h1: { fontSize: typography.fontSize['5xl'], lineHeight: typography.lineHeight.tight },
    h2: { fontSize: typography.fontSize['4xl'], lineHeight: typography.lineHeight.tight },
    h3: { fontSize: typography.fontSize['3xl'], lineHeight: typography.lineHeight.snug },
    h4: { fontSize: typography.fontSize['2xl'], lineHeight: typography.lineHeight.snug },
    h5: { fontSize: typography.fontSize.xl, lineHeight: typography.lineHeight.normal },
    h6: { fontSize: typography.fontSize.lg, lineHeight: typography.lineHeight.normal },
  },
  // Desktop and above
  lg: {
    h1: { fontSize: typography.fontSize['6xl'], lineHeight: typography.lineHeight.none },
    h2: { fontSize: typography.fontSize['5xl'], lineHeight: typography.lineHeight.tight },
    h3: { fontSize: typography.fontSize['4xl'], lineHeight: typography.lineHeight.tight },
    h4: { fontSize: typography.fontSize['3xl'], lineHeight: typography.lineHeight.snug },
    h5: { fontSize: typography.fontSize['2xl'], lineHeight: typography.lineHeight.normal },
    h6: { fontSize: typography.fontSize.xl, lineHeight: typography.lineHeight.normal },
  },
};

// Text styles presets
export const textStyles = {
  // Headings
  displayLarge: {
    fontFamily: typography.fontFamily.display,
    fontSize: typography.fontSize['7xl'],
    fontWeight: typography.fontWeight.bold,
    lineHeight: typography.lineHeight.none,
    letterSpacing: typography.letterSpacing.tight,
  },
  displayMedium: {
    fontFamily: typography.fontFamily.display,
    fontSize: typography.fontSize['5xl'],
    fontWeight: typography.fontWeight.semibold,
    lineHeight: typography.lineHeight.tight,
    letterSpacing: typography.letterSpacing.tight,
  },
  displaySmall: {
    fontFamily: typography.fontFamily.display,
    fontSize: typography.fontSize['3xl'],
    fontWeight: typography.fontWeight.medium,
    lineHeight: typography.lineHeight.tight,
    letterSpacing: typography.letterSpacing.normal,
  },
  
  // Body text
  bodyLarge: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.regular,
    lineHeight: typography.lineHeight.relaxed,
    letterSpacing: typography.letterSpacing.normal,
  },
  bodyMedium: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.base,
    fontWeight: typography.fontWeight.regular,
    lineHeight: typography.lineHeight.relaxed,
    letterSpacing: typography.letterSpacing.normal,
  },
  bodySmall: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.regular,
    lineHeight: typography.lineHeight.relaxed,
    letterSpacing: typography.letterSpacing.normal,
  },
  
  // Labels and UI text
  labelLarge: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.medium,
    lineHeight: typography.lineHeight.normal,
    letterSpacing: typography.letterSpacing.wide,
  },
  labelMedium: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.medium,
    lineHeight: typography.lineHeight.normal,
    letterSpacing: typography.letterSpacing.wide,
  },
  labelSmall: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.medium,
    lineHeight: typography.lineHeight.normal,
    letterSpacing: typography.letterSpacing.wider,
    textTransform: 'uppercase' as const,
  },
  
  // Code and monospace
  codeLarge: {
    fontFamily: typography.fontFamily.mono,
    fontSize: typography.fontSize.base,
    fontWeight: typography.fontWeight.regular,
    lineHeight: typography.lineHeight.normal,
    letterSpacing: typography.letterSpacing.normal,
  },
  codeMedium: {
    fontFamily: typography.fontFamily.mono,
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.regular,
    lineHeight: typography.lineHeight.normal,
    letterSpacing: typography.letterSpacing.normal,
  },
  codeSmall: {
    fontFamily: typography.fontFamily.mono,
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.regular,
    lineHeight: typography.lineHeight.normal,
    letterSpacing: typography.letterSpacing.normal,
  },
};

// Fluid typography utilities
export const fluidFontSize = (minSize: number, maxSize: number, minViewport = 320, maxViewport = 1920): string => {
  const slope = (maxSize - minSize) / (maxViewport - minViewport);
  const yAxisIntersection = -minViewport * slope + minSize;
  
  return `clamp(${minSize}px, ${yAxisIntersection}px + ${slope * 100}vw, ${maxSize}px)`;
};

// Typography utilities
export const truncate = (lines = 1) => ({
  overflow: 'hidden',
  textOverflow: 'ellipsis',
  display: '-webkit-box',
  WebkitLineClamp: lines,
  WebkitBoxOrient: 'vertical' as const,
});

export const typographyUtils = {
  fluidFontSize,
  truncate,
};

export const llmkgTypography = {
  ...typography,
  textStyles,
  responsiveFontSizes,
  utils: typographyUtils,
};