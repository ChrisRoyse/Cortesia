import { version } from '../../package.json';

// Application initialization configuration
export interface AppInitConfig {
  version: string;
  buildDate: string;
  environment: 'development' | 'production' | 'staging';
  features: {
    enablePerformanceMonitoring: boolean;
    enableServiceWorker: boolean;
    enableErrorReporting: boolean;
    enableAnalytics: boolean;
  };
  api: {
    baseURL: string;
    wsURL: string;
    timeout: number;
  };
  ui: {
    theme: 'light' | 'dark' | 'auto';
    animations: boolean;
    lazyLoading: boolean;
  };
}

// Default configuration
const defaultConfig: AppInitConfig = {
  version: version || '2.0.0',
  buildDate: new Date().toISOString(),
  environment: (import.meta.env.MODE as any) || 'development',
  features: {
    enablePerformanceMonitoring: import.meta.env.PROD,
    enableServiceWorker: import.meta.env.PROD,
    enableErrorReporting: import.meta.env.PROD,
    enableAnalytics: import.meta.env.PROD,
  },
  api: {
    baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8080',
    wsURL: import.meta.env.VITE_WS_URL || 'ws://localhost:8080',
    timeout: 30000,
  },
  ui: {
    theme: 'dark',
    animations: true,
    lazyLoading: true,
  },
};

let appConfig: AppInitConfig = defaultConfig;

/**
 * Initialize the application with configuration
 */
export function initializeApp(customConfig?: Partial<AppInitConfig>): void {
  // Merge custom config with defaults
  if (customConfig) {
    appConfig = {
      ...defaultConfig,
      ...customConfig,
      features: {
        ...defaultConfig.features,
        ...customConfig.features,
      },
      api: {
        ...defaultConfig.api,
        ...customConfig.api,
      },
      ui: {
        ...defaultConfig.ui,
        ...customConfig.ui,
      },
    };
  }

  // Log initialization in development
  if (appConfig.environment === 'development') {
    console.log('🚀 LLMKG Visualization System Initializing...');
    console.log('📊 Configuration:', appConfig);
    console.log('🔧 Environment:', appConfig.environment);
    console.log('📦 Version:', appConfig.version);
  }

  // Set up global error handling
  if (appConfig.features.enableErrorReporting) {
    setupGlobalErrorHandling();
  }

  // Initialize performance monitoring
  if (appConfig.features.enablePerformanceMonitoring) {
    setupPerformanceMonitoring();
  }

  // Set document title
  document.title = 'LLMKG Visualization Dashboard';

  // Set meta tags
  const metaDescription = document.querySelector('meta[name="description"]');
  if (metaDescription) {
    metaDescription.setAttribute('content', 'Brain-Inspired Cognitive Architecture Visualization System');
  }

  // Set theme color meta tag
  const themeColorMeta = document.querySelector('meta[name="theme-color"]');
  if (themeColorMeta) {
    themeColorMeta.setAttribute('content', '#001529');
  }

  // Initialize CSS custom properties for theming
  setupThemeCustomProperties();

  console.log('✅ LLMKG Visualization System Initialized');
}

/**
 * Get current application configuration
 */
export function getAppConfig(): AppInitConfig {
  return appConfig;
}

/**
 * Update application configuration
 */
export function updateAppConfig(updates: Partial<AppInitConfig>): void {
  appConfig = {
    ...appConfig,
    ...updates,
  };
}

/**
 * Setup global error handling
 */
function setupGlobalErrorHandling(): void {
  // Handle uncaught promise rejections
  window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled Promise Rejection:', event.reason);
    
    // Prevent the default browser console error
    event.preventDefault();
    
    // In production, send to error reporting service
    if (appConfig.environment === 'production') {
      reportError(event.reason, 'unhandledrejection');
    }
  });

  // Handle uncaught errors
  window.addEventListener('error', (event) => {
    console.error('Global Error:', event.error);
    
    // In production, send to error reporting service
    if (appConfig.environment === 'production') {
      reportError(event.error, 'error');
    }
  });
}

/**
 * Setup performance monitoring
 */
function setupPerformanceMonitoring(): void {
  // Monitor page load performance
  window.addEventListener('load', () => {
    // Use setTimeout to ensure all resources are loaded
    setTimeout(() => {
      const perfData = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
      
      if (perfData) {
        const metrics = {
          loadTime: perfData.loadEventEnd - perfData.loadEventStart,
          domContentLoaded: perfData.domContentLoadedEventEnd - perfData.domContentLoadedEventStart,
          firstPaint: performance.getEntriesByName('first-paint')[0]?.startTime || 0,
          firstContentfulPaint: performance.getEntriesByName('first-contentful-paint')[0]?.startTime || 0,
        };

        console.log('📈 Performance Metrics:', metrics);
        
        // In production, send to analytics
        if (appConfig.environment === 'production') {
          reportPerformanceMetrics(metrics);
        }
      }
    }, 0);
  });
}

/**
 * Setup CSS custom properties for theming
 */
function setupThemeCustomProperties(): void {
  const root = document.documentElement;
  
  // Dark theme colors (default)
  const darkTheme = {
    '--primary-color': '#1890ff',
    '--background-color': '#001529',
    '--surface-color': '#141414',
    '--text-color': '#ffffff',
    '--text-secondary': 'rgba(255, 255, 255, 0.65)',
    '--border-color': '#303030',
    '--success-color': '#52c41a',
    '--warning-color': '#faad14',
    '--error-color': '#ff4d4f',
    '--info-color': '#1890ff',
  };

  // Apply theme variables
  Object.entries(darkTheme).forEach(([property, value]) => {
    root.style.setProperty(property, value);
  });
}

/**
 * Report error to monitoring service (placeholder)
 */
function reportError(error: any, type: string): void {
  // In a real application, this would send errors to a service like Sentry
  const errorData = {
    error: error?.message || String(error),
    stack: error?.stack,
    type,
    timestamp: new Date().toISOString(),
    url: window.location.href,
    userAgent: navigator.userAgent,
    version: appConfig.version,
  };

  console.log('📊 Error reported:', errorData);
  
  // Example: Send to error reporting service
  // fetch('/api/errors', {
  //   method: 'POST',
  //   headers: { 'Content-Type': 'application/json' },
  //   body: JSON.stringify(errorData)
  // });
}

/**
 * Report performance metrics (placeholder)
 */
function reportPerformanceMetrics(metrics: any): void {
  // In a real application, this would send metrics to an analytics service
  const performanceData = {
    ...metrics,
    timestamp: new Date().toISOString(),
    url: window.location.href,
    version: appConfig.version,
  };

  console.log('📊 Performance metrics reported:', performanceData);
  
  // Example: Send to analytics service
  // fetch('/api/analytics/performance', {
  //   method: 'POST',
  //   headers: { 'Content-Type': 'application/json' },
  //   body: JSON.stringify(performanceData)
  // });
}

// Feature flags for progressive enhancement
export const featureFlags = {
  enableVirtualization: true,
  enableWebGL: (() => {
    try {
      const canvas = document.createElement('canvas');
      return !!(canvas.getContext('webgl') || canvas.getContext('experimental-webgl'));
    } catch {
      return false;
    }
  })(),
  enableWebWorkers: typeof Worker !== 'undefined',
  enableIntersectionObserver: 'IntersectionObserver' in window,
  enableResizeObserver: 'ResizeObserver' in window,
};