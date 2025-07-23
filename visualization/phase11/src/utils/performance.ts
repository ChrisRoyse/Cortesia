// Performance monitoring and optimization utilities

export interface PerformanceEntry {
  name: string;
  startTime: number;
  duration: number;
  timestamp: number;
}

export interface WebVital {
  name: string;
  value: number;
  rating: 'good' | 'needs-improvement' | 'poor';
  timestamp: number;
}

class PerformanceMonitor {
  private entries: PerformanceEntry[] = [];
  private vitals: WebVital[] = [];
  private observers: Map<string, PerformanceObserver> = new Map();
  private memoryUsageInterval?: number;

  init() {
    if (typeof window === 'undefined') return;

    // Initialize performance observers
    this.initPerformanceObservers();
    
    // Start memory usage monitoring
    this.startMemoryMonitoring();
    
    // Monitor long tasks
    this.monitorLongTasks();
    
    // Monitor resource timing
    this.monitorResourceTiming();

    console.log('Performance monitoring initialized');
  }

  private initPerformanceObservers() {
    // Paint timing observer
    if ('PerformanceObserver' in window) {
      try {
        const paintObserver = new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) {
            this.recordEntry({
              name: entry.name,
              startTime: entry.startTime,
              duration: entry.duration || 0,
              timestamp: Date.now(),
            });
          }
        });
        paintObserver.observe({ entryTypes: ['paint'] });
        this.observers.set('paint', paintObserver);
      } catch (e) {
        console.warn('Paint observer not supported:', e);
      }

      // Layout shift observer
      try {
        const layoutShiftObserver = new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) {
            if ('value' in entry) {
              this.recordEntry({
                name: 'layout-shift',
                startTime: entry.startTime,
                duration: (entry as any).value,
                timestamp: Date.now(),
              });
            }
          }
        });
        layoutShiftObserver.observe({ entryTypes: ['layout-shift'] });
        this.observers.set('layout-shift', layoutShiftObserver);
      } catch (e) {
        console.warn('Layout shift observer not supported:', e);
      }

      // Largest contentful paint observer
      try {
        const lcpObserver = new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) {
            this.recordEntry({
              name: 'largest-contentful-paint',
              startTime: entry.startTime,
              duration: 0,
              timestamp: Date.now(),
            });
          }
        });
        lcpObserver.observe({ entryTypes: ['largest-contentful-paint'] });
        this.observers.set('lcp', lcpObserver);
      } catch (e) {
        console.warn('LCP observer not supported:', e);
      }
    }
  }

  private startMemoryMonitoring() {
    if ('memory' in performance) {
      this.memoryUsageInterval = window.setInterval(() => {
        const memory = (performance as any).memory;
        this.recordEntry({
          name: 'memory-usage',
          startTime: performance.now(),
          duration: memory.usedJSHeapSize,
          timestamp: Date.now(),
        });
      }, 5000);
    }
  }

  private monitorLongTasks() {
    if ('PerformanceObserver' in window) {
      try {
        const longTaskObserver = new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) {
            this.recordEntry({
              name: 'long-task',
              startTime: entry.startTime,
              duration: entry.duration,
              timestamp: Date.now(),
            });

            // Log warning for long tasks
            if (entry.duration > 50) {
              console.warn(`Long task detected: ${entry.duration}ms`);
            }
          }
        });
        longTaskObserver.observe({ entryTypes: ['longtask'] });
        this.observers.set('longtask', longTaskObserver);
      } catch (e) {
        console.warn('Long task observer not supported:', e);
      }
    }
  }

  private monitorResourceTiming() {
    if ('PerformanceObserver' in window) {
      try {
        const resourceObserver = new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) {
            const resourceEntry = entry as PerformanceResourceTiming;
            this.recordEntry({
              name: `resource-${this.getResourceType(resourceEntry.name)}`,
              startTime: resourceEntry.startTime,
              duration: resourceEntry.duration,
              timestamp: Date.now(),
            });
          }
        });
        resourceObserver.observe({ entryTypes: ['resource'] });
        this.observers.set('resource', resourceObserver);
      } catch (e) {
        console.warn('Resource observer not supported:', e);
      }
    }
  }

  private getResourceType(url: string): string {
    if (url.includes('.js')) return 'script';
    if (url.includes('.css')) return 'stylesheet';
    if (url.includes('.png') || url.includes('.jpg') || url.includes('.svg')) return 'image';
    if (url.includes('.woff') || url.includes('.ttf')) return 'font';
    return 'other';
  }

  private recordEntry(entry: PerformanceEntry) {
    this.entries.push(entry);
    
    // Keep only last 1000 entries
    if (this.entries.length > 1000) {
      this.entries = this.entries.slice(-1000);
    }
  }

  // Web Vitals handlers
  onCLS = (metric: any) => {
    this.recordVital({
      name: 'CLS',
      value: metric.value,
      rating: metric.value <= 0.1 ? 'good' : metric.value <= 0.25 ? 'needs-improvement' : 'poor',
      timestamp: Date.now(),
    });
  };

  onFID = (metric: any) => {
    this.recordVital({
      name: 'FID',
      value: metric.value,
      rating: metric.value <= 100 ? 'good' : metric.value <= 300 ? 'needs-improvement' : 'poor',
      timestamp: Date.now(),
    });
  };

  onFCP = (metric: any) => {
    this.recordVital({
      name: 'FCP',
      value: metric.value,
      rating: metric.value <= 1800 ? 'good' : metric.value <= 3000 ? 'needs-improvement' : 'poor',
      timestamp: Date.now(),
    });
  };

  onLCP = (metric: any) => {
    this.recordVital({
      name: 'LCP',
      value: metric.value,
      rating: metric.value <= 2500 ? 'good' : metric.value <= 4000 ? 'needs-improvement' : 'poor',
      timestamp: Date.now(),
    });
  };

  onTTFB = (metric: any) => {
    this.recordVital({
      name: 'TTFB',
      value: metric.value,
      rating: metric.value <= 800 ? 'good' : metric.value <= 1800 ? 'needs-improvement' : 'poor',
      timestamp: Date.now(),
    });
  };

  private recordVital(vital: WebVital) {
    this.vitals.push(vital);
    console.log(`Web Vital - ${vital.name}: ${vital.value} (${vital.rating})`);
  }

  // Public methods
  getEntries(): PerformanceEntry[] {
    return [...this.entries];
  }

  getVitals(): WebVital[] {
    return [...this.vitals];
  }

  measure(name: string, fn: () => void): number {
    const start = performance.now();
    fn();
    const duration = performance.now() - start;
    
    this.recordEntry({
      name,
      startTime: start,
      duration,
      timestamp: Date.now(),
    });

    return duration;
  }

  async measureAsync(name: string, fn: () => Promise<void>): Promise<number> {
    const start = performance.now();
    await fn();
    const duration = performance.now() - start;
    
    this.recordEntry({
      name,
      startTime: start,
      duration,
      timestamp: Date.now(),
    });

    return duration;
  }

  reportError(error: Error, errorInfo?: any) {
    // In production, this would send to error reporting service
    console.error('Performance Monitor - Error reported:', {
      error: error.message,
      stack: error.stack,
      errorInfo,
      timestamp: Date.now(),
      performance: this.getPerformanceSummary(),
    });
  }

  getPerformanceSummary() {
    const recentEntries = this.entries.slice(-100);
    const recentVitals = this.vitals.slice(-10);

    return {
      entryCount: this.entries.length,
      vitalCount: this.vitals.length,
      recentEntries: recentEntries.map(e => ({
        name: e.name,
        duration: e.duration,
        timestamp: e.timestamp,
      })),
      recentVitals: recentVitals.map(v => ({
        name: v.name,
        value: v.value,
        rating: v.rating,
        timestamp: v.timestamp,
      })),
      memoryUsage: 'memory' in performance ? (performance as any).memory : null,
    };
  }

  cleanup() {
    // Clear intervals
    if (this.memoryUsageInterval) {
      clearInterval(this.memoryUsageInterval);
    }

    // Disconnect observers
    this.observers.forEach(observer => {
      try {
        observer.disconnect();
      } catch (e) {
        console.warn('Error disconnecting observer:', e);
      }
    });
    this.observers.clear();

    // Clear data
    this.entries = [];
    this.vitals = [];
  }
}

export const performanceMonitor = new PerformanceMonitor();

// React component performance utilities
export function withPerformanceTracking<P extends object>(
  Component: React.ComponentType<P>,
  componentName?: string
) {
  return React.memo((props: P) => {
    const name = componentName || Component.displayName || Component.name;
    
    React.useEffect(() => {
      const start = performance.now();
      
      return () => {
        const duration = performance.now() - start;
        performanceMonitor.measure(`component-${name}`, () => {});
      };
    }, [name]);

    return React.createElement(Component, props);
  });
}

// Hook for component performance tracking
export function usePerformanceTracking(componentName: string) {
  React.useEffect(() => {
    const start = performance.now();
    
    return () => {
      const duration = performance.now() - start;
      console.log(`Component ${componentName} render time: ${duration.toFixed(2)}ms`);
    };
  }, [componentName]);
}

// Lazy loading utilities
export function createLazyComponent<P extends object>(
  importFn: () => Promise<{ default: React.ComponentType<P> }>,
  fallback?: React.ReactElement
) {
  const LazyComponent = React.lazy(importFn);
  
  return (props: P) => (
    <React.Suspense fallback={fallback || <div>Loading...</div>}>
      <LazyComponent {...props} />
    </React.Suspense>
  );
}

// Bundle analyzer utilities
export function analyzeBundleSize() {
  if (typeof window !== 'undefined' && 'performance' in window) {
    const resources = performance.getEntriesByType('resource') as PerformanceResourceTiming[];
    
    const bundleInfo = resources
      .filter(resource => resource.name.includes('.js') || resource.name.includes('.css'))
      .map(resource => ({
        name: resource.name.split('/').pop(),
        size: resource.transferSize || resource.encodedBodySize || 0,
        loadTime: resource.duration,
        type: resource.name.includes('.js') ? 'javascript' : 'stylesheet',
      }))
      .sort((a, b) => b.size - a.size);

    console.table(bundleInfo);
    return bundleInfo;
  }
  
  return [];
}

// Memory leak detection
export function detectMemoryLeaks() {
  if ('memory' in performance) {
    const memory = (performance as any).memory;
    const threshold = 50 * 1024 * 1024; // 50MB
    
    if (memory.usedJSHeapSize > threshold) {
      console.warn('Potential memory leak detected:', {
        used: `${(memory.usedJSHeapSize / 1024 / 1024).toFixed(2)}MB`,
        total: `${(memory.totalJSHeapSize / 1024 / 1024).toFixed(2)}MB`,
        limit: `${(memory.jsHeapSizeLimit / 1024 / 1024).toFixed(2)}MB`,
      });
      
      return true;
    }
  }
  
  return false;
}

// Performance optimization suggestions
export function getOptimizationSuggestions(): string[] {
  const suggestions: string[] = [];
  const summary = performanceMonitor.getPerformanceSummary();
  
  // Check for slow renders
  const slowRenders = summary.recentEntries.filter(
    entry => entry.name.startsWith('component-') && entry.duration > 16
  );
  
  if (slowRenders.length > 0) {
    suggestions.push('Consider optimizing slow-rendering components with React.memo or useMemo');
  }

  // Check for long tasks
  const longTasks = summary.recentEntries.filter(
    entry => entry.name === 'long-task' && entry.duration > 50
  );
  
  if (longTasks.length > 0) {
    suggestions.push('Break down long tasks using time slicing or web workers');
  }

  // Check memory usage
  if (detectMemoryLeaks()) {
    suggestions.push('Investigate potential memory leaks and clean up unused references');
  }

  // Check bundle size
  const bundles = analyzeBundleSize();
  const largeBundles = bundles.filter(bundle => bundle.size > 1024 * 1024); // > 1MB
  
  if (largeBundles.length > 0) {
    suggestions.push('Consider code splitting for large bundles');
  }

  return suggestions;
}