/**
 * LLMKG Theme Provider
 * Manages theme state and provides theme context to the application
 */

import React, { createContext, useContext, useState, useEffect, useMemo, useCallback } from 'react';
import { ThemeProvider as MuiThemeProvider, CssBaseline } from '@mui/material';
import { createLLMKGTheme, ThemeMode, LLMKGTheme } from '../../theme';

export interface ThemeContextValue {
  mode: ThemeMode;
  theme: LLMKGTheme;
  toggleTheme: () => void;
  setTheme: (mode: ThemeMode) => void;
  systemPreference: ThemeMode;
}

const ThemeContext = createContext<ThemeContextValue | undefined>(undefined);

export interface ThemeProviderProps {
  children: React.ReactNode;
  defaultMode?: ThemeMode;
  storageKey?: string;
  enableSystemPreference?: boolean;
}

export const ThemeProvider: React.FC<ThemeProviderProps> = ({
  children,
  defaultMode = 'light',
  storageKey = 'llmkg-theme-mode',
  enableSystemPreference = true,
}) => {
  // Get system theme preference
  const getSystemPreference = (): ThemeMode => {
    if (typeof window === 'undefined') return 'light';
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  };

  const [systemPreference, setSystemPreference] = useState<ThemeMode>(getSystemPreference());

  // Get initial theme mode
  const getInitialMode = (): ThemeMode => {
    if (typeof window === 'undefined') return defaultMode;
    
    const stored = localStorage.getItem(storageKey);
    if (stored === 'light' || stored === 'dark') {
      return stored;
    }
    
    return enableSystemPreference ? systemPreference : defaultMode;
  };

  const [mode, setMode] = useState<ThemeMode>(getInitialMode());

  // Create theme
  const theme = useMemo(() => createLLMKGTheme(mode), [mode]);

  // Handle theme toggle
  const toggleTheme = useCallback(() => {
    setMode((prevMode) => {
      const newMode = prevMode === 'light' ? 'dark' : 'light';
      localStorage.setItem(storageKey, newMode);
      return newMode;
    });
  }, [storageKey]);

  // Handle theme set
  const setTheme = useCallback((newMode: ThemeMode) => {
    setMode(newMode);
    localStorage.setItem(storageKey, newMode);
  }, [storageKey]);

  // Listen for system preference changes
  useEffect(() => {
    if (!enableSystemPreference) return;

    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    
    const handleChange = (e: MediaQueryListEvent) => {
      const newSystemPreference = e.matches ? 'dark' : 'light';
      setSystemPreference(newSystemPreference);
      
      // Only update theme if no user preference is stored
      const stored = localStorage.getItem(storageKey);
      if (!stored) {
        setMode(newSystemPreference);
      }
    };

    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, [enableSystemPreference, storageKey]);

  // Apply theme class to document root
  useEffect(() => {
    const root = document.documentElement;
    root.classList.remove('light', 'dark');
    root.classList.add(mode);
    
    // Also set data-theme attribute for testing
    root.setAttribute('data-theme', mode);
    
    // Also set color-scheme for native elements
    root.style.colorScheme = mode;
    
    // Update meta theme-color
    const metaThemeColor = document.querySelector('meta[name="theme-color"]');
    if (metaThemeColor) {
      metaThemeColor.setAttribute('content', theme.llmkg.colors.background.primary);
    }
  }, [mode, theme]);

  const contextValue = useMemo<ThemeContextValue>(
    () => ({
      mode,
      theme,
      toggleTheme,
      setTheme,
      systemPreference,
    }),
    [mode, theme, toggleTheme, setTheme, systemPreference]
  );

  return (
    <ThemeContext.Provider value={contextValue}>
      <MuiThemeProvider theme={theme}>
        <CssBaseline />
        {children}
      </MuiThemeProvider>
    </ThemeContext.Provider>
  );
};

// Hook to use theme context
export const useThemeContext = (): ThemeContextValue => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useThemeContext must be used within a ThemeProvider');
  }
  return context;
};

// Theme toggle component
export interface ThemeToggleProps {
  className?: string;
  showLabel?: boolean;
  size?: 'small' | 'medium' | 'large';
}

export const ThemeToggle: React.FC<ThemeToggleProps> = ({ 
  className = '', 
  showLabel = false,
  size = 'medium' 
}) => {
  const { mode, toggleTheme } = useThemeContext();
  
  const sizes = {
    small: { button: 'w-8 h-8', icon: 'w-4 h-4' },
    medium: { button: 'w-10 h-10', icon: 'w-5 h-5' },
    large: { button: 'w-12 h-12', icon: 'w-6 h-6' },
  };
  
  const { button: buttonSize, icon: iconSize } = sizes[size];
  
  return (
    <button
      onClick={toggleTheme}
      className={`
        ${buttonSize}
        relative inline-flex items-center justify-center
        rounded-lg transition-all duration-200
        hover:bg-gray-100 dark:hover:bg-gray-800
        focus:outline-none focus:ring-2 focus:ring-offset-2
        focus:ring-cognitive-500 dark:focus:ring-cognitive-400
        ${className}
      `}
      aria-label={`Switch to ${mode === 'light' ? 'dark' : 'light'} theme`}
    >
      <span className="sr-only">Toggle theme</span>
      
      {/* Sun icon */}
      <svg
        className={`
          ${iconSize}
          absolute transition-all duration-300
          ${mode === 'light' ? 'rotate-0 scale-100' : 'rotate-90 scale-0'}
        `}
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
        strokeWidth={2}
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z"
        />
      </svg>
      
      {/* Moon icon */}
      <svg
        className={`
          ${iconSize}
          absolute transition-all duration-300
          ${mode === 'dark' ? 'rotate-0 scale-100' : '-rotate-90 scale-0'}
        `}
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
        strokeWidth={2}
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"
        />
      </svg>
      
      {showLabel && (
        <span className="ml-2 text-sm font-medium">
          {mode === 'light' ? 'Light' : 'Dark'}
        </span>
      )}
    </button>
  );
};

export default ThemeProvider;