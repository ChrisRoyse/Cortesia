import { useState, useEffect, useCallback, useMemo } from 'react';

export interface ResponsiveBreakpoints {
  [key: string]: number;
}

export interface UseResponsiveLayoutOptions {
  breakpoints?: ResponsiveBreakpoints;
  debounceMs?: number;
  enableOrientation?: boolean;
  enableTouchDetection?: boolean;
}

export interface ResponsiveLayoutState {
  currentBreakpoint: string;
  windowSize: {
    width: number;
    height: number;
  };
  isMobile: boolean;
  isTablet: boolean;
  isDesktop: boolean;
  isLandscape: boolean;
  isPortrait: boolean;
  isTouchDevice: boolean;
  pixelRatio: number;
  availableSpace: {
    width: number;
    height: number;
  };
}

const defaultBreakpoints: ResponsiveBreakpoints = {
  xs: 0,
  sm: 576,
  md: 768,
  lg: 992,
  xl: 1200,
  xxl: 1400
};

export const useResponsiveLayout = (options: UseResponsiveLayoutOptions = {}) => {
  const {
    breakpoints = defaultBreakpoints,
    debounceMs = 150,
    enableOrientation = true,
    enableTouchDetection = true
  } = options;

  const [windowSize, setWindowSize] = useState({
    width: typeof window !== 'undefined' ? window.innerWidth : 1200,
    height: typeof window !== 'undefined' ? window.innerHeight : 800
  });

  const [orientation, setOrientation] = useState({
    isLandscape: typeof window !== 'undefined' ? window.innerWidth > window.innerHeight : true,
    isPortrait: typeof window !== 'undefined' ? window.innerHeight > window.innerWidth : false
  });

  const [touchDevice, setTouchDevice] = useState(false);
  const [pixelRatio, setPixelRatio] = useState(
    typeof window !== 'undefined' ? window.devicePixelRatio : 1
  );

  // Debounce function
  const debounce = useCallback(<T extends (...args: any[]) => void>(
    func: T,
    wait: number
  ): T => {
    let timeout: NodeJS.Timeout;
    return ((...args: Parameters<T>) => {
      clearTimeout(timeout);
      timeout = setTimeout(() => func(...args), wait);
    }) as T;
  }, []);

  // Detect touch device
  const detectTouchDevice = useCallback(() => {
    if (!enableTouchDetection || typeof window === 'undefined') return false;
    
    return !!(
      'ontouchstart' in window ||
      navigator.maxTouchPoints > 0 ||
      (navigator as any).msMaxTouchPoints > 0
    );
  }, [enableTouchDetection]);

  // Update window size
  const updateWindowSize = useCallback(() => {
    if (typeof window === 'undefined') return;

    const newWidth = window.innerWidth;
    const newHeight = window.innerHeight;

    setWindowSize({ width: newWidth, height: newHeight });

    if (enableOrientation) {
      setOrientation({
        isLandscape: newWidth > newHeight,
        isPortrait: newHeight > newWidth
      });
    }

    setPixelRatio(window.devicePixelRatio);
  }, [enableOrientation]);

  // Debounced resize handler
  const debouncedUpdateWindowSize = useMemo(
    () => debounce(updateWindowSize, debounceMs),
    [updateWindowSize, debounceMs, debounce]
  );

  // Get current breakpoint
  const getCurrentBreakpoint = useCallback((width: number): string => {
    const sortedBreakpoints = Object.entries(breakpoints)
      .sort(([, a], [, b]) => b - a);

    for (const [name, minWidth] of sortedBreakpoints) {
      if (width >= minWidth) {
        return name;
      }
    }

    return sortedBreakpoints[sortedBreakpoints.length - 1][0];
  }, [breakpoints]);

  // Calculate device categories
  const deviceCategories = useMemo(() => {
    const { width } = windowSize;
    
    return {
      isMobile: width < 768,
      isTablet: width >= 768 && width < 1024,
      isDesktop: width >= 1024
    };
  }, [windowSize]);

  // Calculate available space (excluding typical browser UI)
  const availableSpace = useMemo(() => {
    const { width, height } = windowSize;
    
    // Account for browser UI, scrollbars, etc.
    const availableWidth = width - (deviceCategories.isDesktop ? 20 : 0); // scrollbar
    const availableHeight = height - (deviceCategories.isMobile ? 60 : 40); // address bar, etc.

    return {
      width: Math.max(0, availableWidth),
      height: Math.max(0, availableHeight)
    };
  }, [windowSize, deviceCategories]);

  // Setup event listeners
  useEffect(() => {
    if (typeof window === 'undefined') return;

    // Initial setup
    updateWindowSize();
    setTouchDevice(detectTouchDevice());

    // Event listeners
    window.addEventListener('resize', debouncedUpdateWindowSize);
    
    if (enableOrientation) {
      window.addEventListener('orientationchange', debouncedUpdateWindowSize);
    }

    // Cleanup
    return () => {
      window.removeEventListener('resize', debouncedUpdateWindowSize);
      if (enableOrientation) {
        window.removeEventListener('orientationchange', debouncedUpdateWindowSize);
      }
    };
  }, [
    updateWindowSize,
    debouncedUpdateWindowSize,
    detectTouchDevice,
    enableOrientation
  ]);

  // Responsive layout state
  const responsiveState: ResponsiveLayoutState = useMemo(() => ({
    currentBreakpoint: getCurrentBreakpoint(windowSize.width),
    windowSize,
    ...deviceCategories,
    ...orientation,
    isTouchDevice: touchDevice,
    pixelRatio,
    availableSpace
  }), [
    getCurrentBreakpoint,
    windowSize,
    deviceCategories,
    orientation,
    touchDevice,
    pixelRatio,
    availableSpace
  ]);

  // Utility functions
  const isBreakpoint = useCallback((breakpoint: string): boolean => {
    return responsiveState.currentBreakpoint === breakpoint;
  }, [responsiveState.currentBreakpoint]);

  const isBreakpointUp = useCallback((breakpoint: string): boolean => {
    const currentWidth = windowSize.width;
    const targetWidth = breakpoints[breakpoint];
    return currentWidth >= targetWidth;
  }, [windowSize.width, breakpoints]);

  const isBreakpointDown = useCallback((breakpoint: string): boolean => {
    const currentWidth = windowSize.width;
    const targetWidth = breakpoints[breakpoint];
    return currentWidth < targetWidth;
  }, [windowSize.width, breakpoints]);

  const isBreakpointBetween = useCallback((
    minBreakpoint: string, 
    maxBreakpoint: string
  ): boolean => {
    const currentWidth = windowSize.width;
    const minWidth = breakpoints[minBreakpoint];
    const maxWidth = breakpoints[maxBreakpoint];
    return currentWidth >= minWidth && currentWidth < maxWidth;
  }, [windowSize.width, breakpoints]);

  // Grid columns calculation
  const getGridColumns = useCallback((breakpoint?: string): number => {
    const bp = breakpoint || responsiveState.currentBreakpoint;
    
    const columnMap: { [key: string]: number } = {
      xs: 1,
      sm: 2,
      md: 6,
      lg: 12,
      xl: 12,
      xxl: 12
    };

    return columnMap[bp] || 12;
  }, [responsiveState.currentBreakpoint]);

  // Container width calculation
  const getContainerWidth = useCallback((maxWidth?: number): number => {
    const { width } = windowSize;
    
    const containerMaxWidths: { [key: string]: number } = {
      sm: 540,
      md: 720,
      lg: 960,
      xl: 1140,
      xxl: 1320
    };

    const breakpointMaxWidth = containerMaxWidths[responsiveState.currentBreakpoint];
    const calculatedWidth = maxWidth ? 
      Math.min(width, maxWidth) : 
      (breakpointMaxWidth ? Math.min(width, breakpointMaxWidth) : width);

    return calculatedWidth;
  }, [windowSize, responsiveState.currentBreakpoint]);

  // Media query hook
  const useMediaQuery = useCallback((query: string): boolean => {
    const [matches, setMatches] = useState(
      typeof window !== 'undefined' ? window.matchMedia(query).matches : false
    );

    useEffect(() => {
      if (typeof window === 'undefined') return;

      const mediaQuery = window.matchMedia(query);
      const handler = () => setMatches(mediaQuery.matches);
      
      mediaQuery.addEventListener('change', handler);
      return () => mediaQuery.removeEventListener('change', handler);
    }, [query]);

    return matches;
  }, []);

  // Performance optimization - viewport calculations
  const getViewportInfo = useCallback(() => {
    if (typeof window === 'undefined') {
      return {
        scrollY: 0,
        innerHeight: 800,
        outerHeight: 800,
        visualViewport: { width: 1200, height: 800 }
      };
    }

    return {
      scrollY: window.scrollY,
      innerHeight: window.innerHeight,
      outerHeight: window.outerHeight,
      visualViewport: window.visualViewport ? {
        width: window.visualViewport.width,
        height: window.visualViewport.height
      } : { width: window.innerWidth, height: window.innerHeight }
    };
  }, []);

  return {
    ...responsiveState,
    isBreakpoint,
    isBreakpointUp,
    isBreakpointDown,
    isBreakpointBetween,
    getGridColumns,
    getContainerWidth,
    useMediaQuery,
    getViewportInfo,
    breakpoints
  };
};

export default useResponsiveLayout;