import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useResponsiveLayout } from '../../hooks/useResponsiveLayout';

export interface ResponsiveBreakpoint {
  name: string;
  minWidth: number;
  maxWidth?: number;
  cols?: number;
  margin?: [number, number];
  padding?: [number, number];
}

export interface ResponsiveContainerProps {
  children: React.ReactNode;
  minWidth?: number;
  maxWidth?: number;
  aspectRatio?: number;
  maintainAspectRatio?: boolean;
  breakpoint?: string;
  breakpoints?: ResponsiveBreakpoint[];
  onBreakpointChange?: (breakpoint: string) => void;
  className?: string;
  style?: React.CSSProperties;
  observeResize?: boolean;
  debounceMs?: number;
  fillContainer?: boolean;
  centerContent?: boolean;
}

const defaultBreakpoints: ResponsiveBreakpoint[] = [
  { name: 'xxs', minWidth: 0, maxWidth: 479, cols: 1, margin: [5, 5], padding: [10, 10] },
  { name: 'xs', minWidth: 480, maxWidth: 767, cols: 2, margin: [8, 8], padding: [15, 15] },
  { name: 'sm', minWidth: 768, maxWidth: 991, cols: 6, margin: [10, 10], padding: [20, 20] },
  { name: 'md', minWidth: 992, maxWidth: 1199, cols: 8, margin: [15, 15], padding: [25, 25] },
  { name: 'lg', minWidth: 1200, maxWidth: 1599, cols: 12, margin: [20, 20], padding: [30, 30] },
  { name: 'xl', minWidth: 1600, cols: 12, margin: [25, 25], padding: [35, 35] }
];

export const ResponsiveContainer: React.FC<ResponsiveContainerProps> = ({
  children,
  minWidth = 0,
  maxWidth = Infinity,
  aspectRatio,
  maintainAspectRatio = false,
  breakpoint: forcedBreakpoint,
  breakpoints = defaultBreakpoints,
  onBreakpointChange,
  className = '',
  style = {},
  observeResize = true,
  debounceMs = 100,
  fillContainer = false,
  centerContent = false
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });
  const [mounted, setMounted] = useState(false);
  
  const { 
    currentBreakpoint, 
    windowSize, 
    isMobile, 
    isTablet, 
    isDesktop 
  } = useResponsiveLayout({
    breakpoints: breakpoints.reduce((acc, bp) => ({
      ...acc,
      [bp.name]: bp.minWidth
    }), {}),
    debounceMs
  });

  useEffect(() => {
    setMounted(true);
  }, []);

  const activeBreakpoint = forcedBreakpoint || currentBreakpoint;
  const breakpointConfig = breakpoints.find(bp => bp.name === activeBreakpoint) || breakpoints[0];

  const updateContainerSize = useCallback(() => {
    if (!containerRef.current || !observeResize) {
      return;
    }

    const rect = containerRef.current.getBoundingClientRect();
    let newWidth = rect.width;
    let newHeight = rect.height;

    // Apply size constraints
    if (minWidth) newWidth = Math.max(newWidth, minWidth);
    if (maxWidth && maxWidth !== Infinity) newWidth = Math.min(newWidth, maxWidth);

    // Maintain aspect ratio if required
    if (maintainAspectRatio && aspectRatio) {
      newHeight = newWidth / aspectRatio;
    }

    setContainerSize({ width: newWidth, height: newHeight });
  }, [minWidth, maxWidth, aspectRatio, maintainAspectRatio, observeResize]);

  useEffect(() => {
    if (!mounted) return;

    updateContainerSize();
    
    if (observeResize) {
      const resizeObserver = new ResizeObserver(() => {
        updateContainerSize();
      });

      if (containerRef.current) {
        resizeObserver.observe(containerRef.current);
      }

      return () => {
        resizeObserver.disconnect();
      };
    }

    return undefined;
  }, [mounted, updateContainerSize, observeResize]);

  useEffect(() => {
    onBreakpointChange?.(activeBreakpoint);
  }, [activeBreakpoint, onBreakpointChange]);

  const getResponsiveStyles = (): React.CSSProperties => {
    const baseStyles: React.CSSProperties = {
      width: fillContainer ? '100%' : containerSize.width || '100%',
      height: maintainAspectRatio && aspectRatio ? containerSize.height : 'auto',
      minWidth,
      maxWidth: maxWidth === Infinity ? undefined : maxWidth,
      padding: breakpointConfig.padding ? 
        `${breakpointConfig.padding[1]}px ${breakpointConfig.padding[0]}px` : 
        undefined,
      margin: centerContent ? '0 auto' : undefined,
      boxSizing: 'border-box',
      position: 'relative',
      ...style
    };

    if (maintainAspectRatio && aspectRatio) {
      baseStyles.aspectRatio = aspectRatio.toString();
    }

    return baseStyles;
  };

  const containerClasses = [
    'responsive-container',
    `breakpoint-${activeBreakpoint}`,
    isMobile && 'mobile',
    isTablet && 'tablet',
    isDesktop && 'desktop',
    className
  ].filter(Boolean).join(' ');

  if (!mounted) {
    return null; // Prevent hydration issues
  }

  return (
    <div
      ref={containerRef}
      className={containerClasses}
      style={getResponsiveStyles()}
      data-breakpoint={activeBreakpoint}
      data-cols={breakpointConfig.cols}
      data-width={containerSize.width}
      data-height={containerSize.height}
    >
      {centerContent ? (
        <div 
          className="responsive-container-content centered"
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: '100%',
            height: '100%'
          }}
        >
          {children}
        </div>
      ) : (
        <div 
          className="responsive-container-content"
          style={{
            width: '100%',
            height: '100%'
          }}
        >
          {children}
        </div>
      )}
      
      <style>{`
        .responsive-container {
          transition: padding 0.3s ease;
        }
        
        .responsive-container.mobile {
          font-size: 14px;
        }
        
        .responsive-container.tablet {
          font-size: 15px;
        }
        
        .responsive-container.desktop {
          font-size: 16px;
        }
        
        .breakpoint-xxs {
          --container-spacing: 5px;
        }
        
        .breakpoint-xs {
          --container-spacing: 8px;
        }
        
        .breakpoint-sm {
          --container-spacing: 10px;
        }
        
        .breakpoint-md {
          --container-spacing: 15px;
        }
        
        .breakpoint-lg {
          --container-spacing: 20px;
        }
        
        .breakpoint-xl {
          --container-spacing: 25px;
        }
        
        @media (max-width: 479px) {
          .responsive-container {
            padding: 10px 5px !important;
          }
        }
        
        @media (min-width: 480px) and (max-width: 767px) {
          .responsive-container {
            padding: 15px 8px !important;
          }
        }
        
        @media (min-width: 768px) and (max-width: 991px) {
          .responsive-container {
            padding: 20px 10px !important;
          }
        }
        
        @media (min-width: 992px) and (max-width: 1199px) {
          .responsive-container {
            padding: 25px 15px !important;
          }
        }
        
        @media (min-width: 1200px) {
          .responsive-container {
            padding: 30px 20px !important;
          }
        }
        
        @media (orientation: portrait) and (max-width: 768px) {
          .responsive-container {
            flex-direction: column;
          }
        }
        
        @media (orientation: landscape) and (max-height: 500px) {
          .responsive-container {
            padding: 10px !important;
          }
        }
        
        @media (prefers-reduced-motion: reduce) {
          .responsive-container {
            transition: none;
          }
        }
      `}</style>
    </div>
  );
};

export default ResponsiveContainer;