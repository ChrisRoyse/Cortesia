import React, { useState, useCallback, useEffect } from 'react';
import { Responsive, WidthProvider, Layout, Layouts } from 'react-grid-layout';
import 'react-grid-layout/css/styles.css';
import 'react-resizable/css/styles.css';

const ResponsiveGridLayout = WidthProvider(Responsive);

export interface LayoutItem {
  i: string;
  x: number;
  y: number;
  w: number;
  h: number;
  minW?: number;
  minH?: number;
  maxW?: number;
  maxH?: number;
  static?: boolean;
  isDraggable?: boolean;
  isResizable?: boolean;
  component: React.ReactNode;
}

export interface Breakpoints {
  lg: number;
  md: number;
  sm: number;
  xs: number;
  xxs: number;
}

export interface GridLayoutProps {
  items: LayoutItem[];
  onLayoutChange: (layout: Layout[], layouts: Layouts) => void;
  breakpoints?: Breakpoints;
  cols?: { [key: string]: number };
  isDraggable?: boolean;
  isResizable?: boolean;
  rowHeight?: number;
  margin?: [number, number];
  containerPadding?: [number, number];
  layouts?: Layouts;
  compactType?: 'vertical' | 'horizontal' | null;
  preventCollision?: boolean;
  autoSize?: boolean;
  onBreakpointChange?: (breakpoint: string, cols: number) => void;
  onDragStart?: (layout: Layout[], oldItem: Layout, newItem: Layout, placeholder: Layout, e: MouseEvent, element: HTMLElement) => void;
  onDragStop?: (layout: Layout[], oldItem: Layout, newItem: Layout, placeholder: Layout, e: MouseEvent, element: HTMLElement) => void;
  onResizeStart?: (layout: Layout[], oldItem: Layout, newItem: Layout, placeholder: Layout, e: MouseEvent, element: HTMLElement) => void;
  onResizeStop?: (layout: Layout[], oldItem: Layout, newItem: Layout, placeholder: Layout, e: MouseEvent, element: HTMLElement) => void;
}

const defaultBreakpoints: Breakpoints = {
  lg: 1200,
  md: 996,
  sm: 768,
  xs: 480,
  xxs: 0
};

const defaultCols = {
  lg: 12,
  md: 10,
  sm: 6,
  xs: 4,
  xxs: 2
};

export const GridLayout: React.FC<GridLayoutProps> = ({
  items,
  onLayoutChange,
  breakpoints = defaultBreakpoints,
  cols = defaultCols,
  isDraggable = true,
  isResizable = true,
  rowHeight = 150,
  margin = [10, 10],
  containerPadding = [10, 10],
  layouts,
  compactType = 'vertical',
  preventCollision = false,
  autoSize = true,
  onBreakpointChange,
  onDragStart,
  onDragStop,
  onResizeStart,
  onResizeStop
}) => {
  const [currentBreakpoint, setCurrentBreakpoint] = useState<string>('lg');
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const handleBreakpointChange = useCallback((breakpoint: string, ncols: number) => {
    setCurrentBreakpoint(breakpoint);
    onBreakpointChange?.(breakpoint, ncols);
  }, [onBreakpointChange]);

  const handleLayoutChange = useCallback((layout: Layout[], allLayouts: Layouts) => {
    onLayoutChange(layout, allLayouts);
  }, [onLayoutChange]);

  const generateLayouts = useCallback((): Layouts => {
    if (layouts) return layouts;

    const generatedLayouts: Layouts = {};
    Object.keys(cols).forEach((breakpoint) => {
      generatedLayouts[breakpoint] = items.map(item => ({
        i: item.i,
        x: item.x,
        y: item.y,
        w: item.w,
        h: item.h,
        minW: item.minW,
        minH: item.minH,
        maxW: item.maxW,
        maxH: item.maxH,
        static: item.static,
        isDraggable: item.isDraggable ?? isDraggable,
        isResizable: item.isResizable ?? isResizable
      }));
    });
    return generatedLayouts;
  }, [items, cols, isDraggable, isResizable, layouts]);

  const renderGridItems = useCallback(() => {
    return items.map((item) => (
      <div 
        key={item.i} 
        className="grid-item"
        style={{
          border: '1px solid #ddd',
          borderRadius: '8px',
          backgroundColor: '#fff',
          overflow: 'hidden',
          boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
          transition: 'box-shadow 0.2s ease'
        }}
      >
        <div 
          className="grid-item-content"
          style={{
            width: '100%',
            height: '100%',
            padding: '10px',
            overflow: 'auto'
          }}
        >
          {item.component}
        </div>
        {(item.isDraggable ?? isDraggable) && (
          <div 
            className="drag-handle"
            style={{
              position: 'absolute',
              top: '5px',
              right: '5px',
              width: '20px',
              height: '20px',
              cursor: 'move',
              opacity: 0.5,
              transition: 'opacity 0.2s ease'
            }}
            onMouseEnter={(e) => e.currentTarget.style.opacity = '1'}
            onMouseLeave={(e) => e.currentTarget.style.opacity = '0.5'}
          >
            <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
              <path d="M7 2a2 2 0 1 1-4 0 2 2 0 0 1 4 0zM7 8a2 2 0 1 1-4 0 2 2 0 0 1 4 0zM7 14a2 2 0 1 1-4 0 2 2 0 0 1 4 0zM13 2a2 2 0 1 1-4 0 2 2 0 0 1 4 0zM13 8a2 2 0 1 1-4 0 2 2 0 0 1 4 0zM13 14a2 2 0 1 1-4 0 2 2 0 0 1 4 0z"/>
            </svg>
          </div>
        )}
      </div>
    ));
  }, [items, isDraggable]);

  if (!mounted) {
    return null; // Prevent SSR issues
  }

  return (
    <div className="grid-layout-container" style={{ width: '100%' }}>
      <ResponsiveGridLayout
        className="layout"
        layouts={generateLayouts()}
        onLayoutChange={handleLayoutChange}
        onBreakpointChange={handleBreakpointChange}
        breakpoints={breakpoints}
        cols={cols}
        rowHeight={rowHeight}
        margin={margin}
        containerPadding={containerPadding}
        isDraggable={isDraggable}
        isResizable={isResizable}
        compactType={compactType}
        preventCollision={preventCollision}
        autoSize={autoSize}
        onDragStart={onDragStart}
        onDragStop={onDragStop}
        onResizeStart={onResizeStart}
        onResizeStop={onResizeStop}
        draggableHandle=".drag-handle"
        useCSSTransforms={true}
        transformScale={1}
        measureBeforeMount={true}
      >
        {renderGridItems()}
      </ResponsiveGridLayout>
      <style jsx>{`
        .grid-item:hover {
          box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15) !important;
        }
        
        .react-grid-item.react-grid-placeholder {
          background: rgba(0, 0, 0, 0.1);
          border-radius: 8px;
        }
        
        .react-grid-item > .react-resizable-handle::after {
          border-right: 2px solid rgba(0, 0, 0, 0.2);
          border-bottom: 2px solid rgba(0, 0, 0, 0.2);
        }
        
        .react-grid-item.resizing {
          opacity: 0.9;
        }
        
        .react-grid-item.static {
          background: #f0f0f0;
        }
      `}</style>
    </div>
  );
};

export default GridLayout;