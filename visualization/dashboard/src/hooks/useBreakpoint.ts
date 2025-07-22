import { useState, useEffect } from 'react';

interface BreakpointState {
  isMobile: boolean;
  isTablet: boolean;
  isDesktop: boolean;
  isLarge: boolean;
  currentBreakpoint: 'mobile' | 'tablet' | 'desktop' | 'large';
  width: number;
}

// Tailwind CSS breakpoints
const breakpoints = {
  sm: 640,
  md: 768,
  lg: 1024,
  xl: 1280,
  '2xl': 1536,
};

export const useBreakpoint = (): BreakpointState => {
  const [breakpointState, setBreakpointState] = useState<BreakpointState>(() => {
    const width = typeof window !== 'undefined' ? window.innerWidth : 1024;
    
    return {
      isMobile: width < breakpoints.md,
      isTablet: width >= breakpoints.md && width < breakpoints.lg,
      isDesktop: width >= breakpoints.lg && width < breakpoints.xl,
      isLarge: width >= breakpoints.xl,
      currentBreakpoint: width < breakpoints.md 
        ? 'mobile' 
        : width < breakpoints.lg 
        ? 'tablet' 
        : width < breakpoints.xl 
        ? 'desktop' 
        : 'large',
      width,
    };
  });

  useEffect(() => {
    const handleResize = () => {
      const width = window.innerWidth;
      
      setBreakpointState({
        isMobile: width < breakpoints.md,
        isTablet: width >= breakpoints.md && width < breakpoints.lg,
        isDesktop: width >= breakpoints.lg && width < breakpoints.xl,
        isLarge: width >= breakpoints.xl,
        currentBreakpoint: width < breakpoints.md 
          ? 'mobile' 
          : width < breakpoints.lg 
          ? 'tablet' 
          : width < breakpoints.xl 
          ? 'desktop' 
          : 'large',
        width,
      });
    };

    // Add event listener
    window.addEventListener('resize', handleResize);
    
    // Call handler immediately to set initial state
    handleResize();

    // Cleanup
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return breakpointState;
};

export default useBreakpoint;