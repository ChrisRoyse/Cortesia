import React, { useState, useRef, useCallback, useEffect, useMemo } from 'react';

export interface ViewportItem {
  id: string;
  bounds: DOMRect;
  priority: number;
  loadTime?: number;
  renderCost?: number;
}

export interface ViewportOptimizerProps {
  children: React.ReactNode;
  threshold?: number;
  rootMargin?: string;
  enableLazyLoading?: boolean;
  enableVirtualization?: boolean;
  maxVisibleItems?: number;
  itemHeight?: number;
  bufferSize?: number;
  onVisibilityChange?: (visibleIds: string[]) => void;
  onPerformanceMetrics?: (metrics: PerformanceMetrics) => void;
  className?: string;
  style?: React.CSSProperties;
}

export interface PerformanceMetrics {
  visibleItems: number;
  totalItems: number;
  renderTime: number;
  memoryUsage: number;
  fps: number;
  loadedComponents: number;
}

interface VirtualizedItem {
  index: number;
  id: string;
  top: number;
  height: number;
  isVisible: boolean;
  component: React.ReactNode;
}

export const ViewportOptimizer: React.FC<ViewportOptimizerProps> = ({
  children,
  threshold = 0.1,
  rootMargin = '100px',
  enableLazyLoading = true,
  enableVirtualization = false,
  maxVisibleItems = 50,
  itemHeight = 200,
  bufferSize = 5,
  onVisibilityChange,
  onPerformanceMetrics,
  className = '',
  style = {}
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const observerRef = useRef<IntersectionObserver | null>(null);
  const performanceRef = useRef({
    frameCount: 0,
    lastTime: performance.now(),
    renderTimes: [] as number[]
  });

  const [visibleItems, setVisibleItems] = useState<Set<string>>(new Set());
  const [loadedItems, setLoadedItems] = useState<Set<string>>(new Set());
  const [virtualizedItems, setVirtualizedItems] = useState<VirtualizedItem[]>([]);
  const [scrollTop, setScrollTop] = useState(0);
  const [containerHeight, setContainerHeight] = useState(0);

  // Performance monitoring
  const [performanceMetrics, setPerformanceMetrics] = useState<PerformanceMetrics>({
    visibleItems: 0,
    totalItems: 0,
    renderTime: 0,
    memoryUsage: 0,
    fps: 60,
    loadedComponents: 0
  });

  // Initialize intersection observer
  useEffect(() => {
    if (!enableLazyLoading) return;

    observerRef.current = new IntersectionObserver(
      (entries) => {
        const newVisibleItems = new Set(visibleItems);
        const newLoadedItems = new Set(loadedItems);

        entries.forEach((entry) => {
          const itemId = entry.target.getAttribute('data-viewport-id');
          if (!itemId) return;

          if (entry.isIntersecting) {
            newVisibleItems.add(itemId);
            newLoadedItems.add(itemId);
          } else {
            newVisibleItems.delete(itemId);
          }
        });

        setVisibleItems(newVisibleItems);
        setLoadedItems(newLoadedItems);
        onVisibilityChange?.(Array.from(newVisibleItems));
      },
      {
        threshold,
        rootMargin
      }
    );

    return () => {
      observerRef.current?.disconnect();
    };
  }, [threshold, rootMargin, enableLazyLoading, visibleItems, loadedItems, onVisibilityChange]);

  // Handle scroll for virtualization
  const handleScroll = useCallback((e: React.UIEvent<HTMLDivElement>) => {
    if (!enableVirtualization) return;

    const target = e.currentTarget;
    setScrollTop(target.scrollTop);
    setContainerHeight(target.clientHeight);
  }, [enableVirtualization]);

  // Calculate virtualized items
  const calculateVirtualizedItems = useCallback((): VirtualizedItem[] => {
    if (!enableVirtualization) return [];

    const items: VirtualizedItem[] = [];
    const childArray = React.Children.toArray(children);
    const totalHeight = childArray.length * itemHeight;
    
    const startIndex = Math.max(0, Math.floor(scrollTop / itemHeight) - bufferSize);
    const endIndex = Math.min(
      childArray.length - 1,
      Math.ceil((scrollTop + containerHeight) / itemHeight) + bufferSize
    );

    for (let i = startIndex; i <= endIndex; i++) {
      if (i < childArray.length) {
        items.push({
          index: i,
          id: `item-${i}`,
          top: i * itemHeight,
          height: itemHeight,
          isVisible: i >= startIndex + bufferSize && i <= endIndex - bufferSize,
          component: childArray[i]
        });
      }
    }

    return items;
  }, [children, scrollTop, containerHeight, itemHeight, bufferSize, enableVirtualization]);

  // Update virtualized items
  useEffect(() => {
    if (enableVirtualization) {
      setVirtualizedItems(calculateVirtualizedItems());
    }
  }, [enableVirtualization, calculateVirtualizedItems]);

  // Performance monitoring
  const updatePerformanceMetrics = useCallback(() => {
    const now = performance.now();
    const deltaTime = now - performanceRef.current.lastTime;
    
    if (deltaTime >= 1000) {
      const fps = (performanceRef.current.frameCount * 1000) / deltaTime;
      const avgRenderTime = performanceRef.current.renderTimes.length > 0 ?
        performanceRef.current.renderTimes.reduce((a, b) => a + b, 0) / performanceRef.current.renderTimes.length : 0;

      const metrics: PerformanceMetrics = {
        visibleItems: visibleItems.size,
        totalItems: React.Children.count(children),
        renderTime: avgRenderTime,
        memoryUsage: (performance as any).memory?.usedJSHeapSize || 0,
        fps: Math.round(fps),
        loadedComponents: loadedItems.size
      };

      setPerformanceMetrics(metrics);
      onPerformanceMetrics?.(metrics);

      // Reset counters
      performanceRef.current.frameCount = 0;
      performanceRef.current.lastTime = now;
      performanceRef.current.renderTimes = [];
    }

    performanceRef.current.frameCount++;
    requestAnimationFrame(updatePerformanceMetrics);
  }, [visibleItems.size, children, loadedItems.size, onPerformanceMetrics]);

  // Start performance monitoring
  useEffect(() => {
    const rafId = requestAnimationFrame(updatePerformanceMetrics);
    return () => cancelAnimationFrame(rafId);
  }, [updatePerformanceMetrics]);

  // Observe elements for lazy loading
  const observeElement = useCallback((element: HTMLDivElement | null) => {
    if (element && observerRef.current && enableLazyLoading) {
      observerRef.current.observe(element);
    }
  }, [enableLazyLoading]);

  // Unobserve elements
  const unobserveElement = useCallback((element: HTMLDivElement | null) => {
    if (element && observerRef.current && enableLazyLoading) {
      observerRef.current.unobserve(element);
    }
  }, [enableLazyLoading]);

  // Process children with lazy loading wrapper
  const processChildren = useCallback(() => {
    if (!enableLazyLoading && !enableVirtualization) {
      return children;
    }

    if (enableVirtualization) {
      const totalHeight = React.Children.count(children) * itemHeight;
      
      return (
        <div style={{ height: totalHeight, position: 'relative' }}>
          {virtualizedItems.map((item) => (
            <div
              key={item.id}
              style={{
                position: 'absolute',
                top: item.top,
                left: 0,
                right: 0,
                height: item.height
              }}
              data-viewport-id={item.id}
              ref={observeElement}
            >
              {loadedItems.has(item.id) || item.isVisible ? (
                item.component
              ) : (
                <div 
                  style={{
                    width: '100%',
                    height: '100%',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    backgroundColor: '#f8f9fa',
                    color: '#6c757d',
                    fontSize: '14px'
                  }}
                >
                  Loading...
                </div>
              )}
            </div>
          ))}
        </div>
      );
    }

    // Lazy loading without virtualization
    return React.Children.map(children, (child, index) => {
      const itemId = `lazy-item-${index}`;
      const isVisible = visibleItems.has(itemId);
      const isLoaded = loadedItems.has(itemId);
      
      return (
        <div
          data-viewport-id={itemId}
          ref={observeElement}
          style={{ minHeight: '50px' }}
        >
          {isLoaded ? (
            child
          ) : (
            <div 
              style={{
                width: '100%',
                height: '150px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                backgroundColor: '#f8f9fa',
                color: '#6c757d',
                fontSize: '14px',
                border: '1px solid #e9ecef',
                borderRadius: '8px'
              }}
            >
              {isVisible ? 'Loading...' : 'Scroll to load'}
            </div>
          )}
        </div>
      );
    });
  }, [children, enableLazyLoading, enableVirtualization, visibleItems, loadedItems, virtualizedItems, itemHeight, observeElement]);

  // Memory cleanup
  useEffect(() => {
    return () => {
      observerRef.current?.disconnect();
    };
  }, []);

  const memoizedChildren = useMemo(processChildren, [processChildren]);

  return (
    <div
      ref={containerRef}
      className={`viewport-optimizer ${className}`}
      style={{
        width: '100%',
        height: '100%',
        overflow: enableVirtualization ? 'auto' : 'visible',
        position: 'relative',
        ...style
      }}
      onScroll={handleScroll}
    >
      {/* Performance indicator (development only) */}
      {import.meta.env.DEV && (
        <div 
          style={{
            position: 'absolute',
            top: '10px',
            right: '10px',
            background: 'rgba(0, 0, 0, 0.8)',
            color: 'white',
            padding: '8px 12px',
            borderRadius: '6px',
            fontSize: '12px',
            zIndex: 9999,
            fontFamily: 'monospace'
          }}
        >
          <div>FPS: {performanceMetrics.fps}</div>
          <div>Visible: {performanceMetrics.visibleItems}/{performanceMetrics.totalItems}</div>
          <div>Loaded: {performanceMetrics.loadedComponents}</div>
          <div>Render: {performanceMetrics.renderTime.toFixed(2)}ms</div>
          {performanceMetrics.memoryUsage > 0 && (
            <div>Memory: {(performanceMetrics.memoryUsage / 1024 / 1024).toFixed(1)}MB</div>
          )}
        </div>
      )}

      {/* Optimized content */}
      <div className="viewport-optimizer-content">
        {memoizedChildren}
      </div>

      {/* Loading indicator for lazy loading */}
      {enableLazyLoading && loadedItems.size < React.Children.count(children) && (
        <div 
          style={{
            padding: '20px',
            textAlign: 'center',
            color: '#6c757d',
            fontSize: '14px'
          }}
        >
          <div>Scroll to load more content...</div>
          <div style={{ marginTop: '5px', fontSize: '12px' }}>
            Loaded: {loadedItems.size} / {React.Children.count(children)}
          </div>
        </div>
      )}

      <style>{`
        .viewport-optimizer {
          contain: layout style paint;
        }
        
        .viewport-optimizer-content {
          will-change: transform;
        }
        
        @media (prefers-reduced-motion: reduce) {
          .viewport-optimizer * {
            transition: none !important;
            animation: none !important;
          }
        }
        
        .viewport-optimizer::-webkit-scrollbar {
          width: 8px;
        }
        
        .viewport-optimizer::-webkit-scrollbar-track {
          background: #f1f1f1;
          border-radius: 4px;
        }
        
        .viewport-optimizer::-webkit-scrollbar-thumb {
          background: #c1c1c1;
          border-radius: 4px;
        }
        
        .viewport-optimizer::-webkit-scrollbar-thumb:hover {
          background: #a8a8a8;
        }
      `}</style>
    </div>
  );
};

export default ViewportOptimizer;