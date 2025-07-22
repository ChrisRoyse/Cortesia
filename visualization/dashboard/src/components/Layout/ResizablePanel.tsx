import React, { useState, useRef, useCallback, useEffect } from 'react';

export interface PanelDimensions {
  width: number;
  height: number;
}

export interface ResizeHandle {
  direction: 'n' | 'e' | 's' | 'w' | 'ne' | 'nw' | 'se' | 'sw';
  size: number;
  style?: React.CSSProperties;
}

export interface ResizablePanelProps {
  children: React.ReactNode;
  width?: number;
  height?: number;
  minWidth?: number;
  minHeight?: number;
  maxWidth?: number;
  maxHeight?: number;
  isResizable?: boolean;
  handles?: ResizeHandle[];
  onResize?: (dimensions: PanelDimensions) => void;
  onResizeStart?: (dimensions: PanelDimensions) => void;
  onResizeEnd?: (dimensions: PanelDimensions) => void;
  className?: string;
  style?: React.CSSProperties;
  aspectRatio?: number;
  maintainAspectRatio?: boolean;
  snap?: {
    x?: number[];
    y?: number[];
  };
}

const defaultHandles: ResizeHandle[] = [
  { direction: 'e', size: 10 },
  { direction: 's', size: 10 },
  { direction: 'se', size: 10 }
];

export const ResizablePanel: React.FC<ResizablePanelProps> = ({
  children,
  width: initialWidth = 300,
  height: initialHeight = 200,
  minWidth = 50,
  minHeight = 50,
  maxWidth = Infinity,
  maxHeight = Infinity,
  isResizable = true,
  handles = defaultHandles,
  onResize,
  onResizeStart,
  onResizeEnd,
  className = '',
  style = {},
  aspectRatio,
  maintainAspectRatio = false,
  snap
}) => {
  const [dimensions, setDimensions] = useState<PanelDimensions>({
    width: initialWidth,
    height: initialHeight
  });
  const [isResizing, setIsResizing] = useState(false);
  const [resizeDirection, setResizeDirection] = useState<string>('');
  
  const panelRef = useRef<HTMLDivElement>(null);
  const startPos = useRef({ x: 0, y: 0 });
  const startDimensions = useRef<PanelDimensions>({ width: 0, height: 0 });

  useEffect(() => {
    setDimensions({
      width: initialWidth,
      height: initialHeight
    });
  }, [initialWidth, initialHeight]);

  const snapToGrid = useCallback((value: number, snapPoints?: number[]): number => {
    if (!snapPoints || snapPoints.length === 0) return value;
    
    const closest = snapPoints.reduce((prev, curr) => 
      Math.abs(curr - value) < Math.abs(prev - value) ? curr : prev
    );
    
    return Math.abs(closest - value) < 10 ? closest : value;
  }, []);

  const constrainDimensions = useCallback((newDimensions: PanelDimensions): PanelDimensions => {
    let { width, height } = newDimensions;

    // Apply snap if enabled
    if (snap?.x) width = snapToGrid(width, snap.x);
    if (snap?.y) height = snapToGrid(height, snap.y);

    // Apply min/max constraints
    width = Math.max(minWidth, Math.min(maxWidth, width));
    height = Math.max(minHeight, Math.min(maxHeight, height));

    // Maintain aspect ratio if required
    if (maintainAspectRatio && aspectRatio) {
      const currentRatio = width / height;
      if (Math.abs(currentRatio - aspectRatio) > 0.01) {
        if (currentRatio > aspectRatio) {
          width = height * aspectRatio;
        } else {
          height = width / aspectRatio;
        }
      }
    }

    return { width, height };
  }, [minWidth, minHeight, maxWidth, maxHeight, maintainAspectRatio, aspectRatio, snap, snapToGrid]);

  const handleMouseDown = useCallback((e: React.MouseEvent, direction: string) => {
    if (!isResizable) return;

    e.preventDefault();
    e.stopPropagation();

    setIsResizing(true);
    setResizeDirection(direction);
    
    startPos.current = { x: e.clientX, y: e.clientY };
    startDimensions.current = { ...dimensions };

    onResizeStart?.(dimensions);

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    document.body.style.cursor = getCursor(direction);
    document.body.style.userSelect = 'none';
  }, [isResizable, dimensions, onResizeStart]);

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!isResizing) return;

    const deltaX = e.clientX - startPos.current.x;
    const deltaY = e.clientY - startPos.current.y;
    
    let newWidth = startDimensions.current.width;
    let newHeight = startDimensions.current.height;

    // Apply resize based on direction
    if (resizeDirection.includes('e')) newWidth += deltaX;
    if (resizeDirection.includes('w')) newWidth -= deltaX;
    if (resizeDirection.includes('s')) newHeight += deltaY;
    if (resizeDirection.includes('n')) newHeight -= deltaY;

    const constrainedDimensions = constrainDimensions({ width: newWidth, height: newHeight });
    setDimensions(constrainedDimensions);
    onResize?.(constrainedDimensions);
  }, [isResizing, resizeDirection, constrainDimensions, onResize]);

  const handleMouseUp = useCallback(() => {
    if (!isResizing) return;

    setIsResizing(false);
    setResizeDirection('');

    document.removeEventListener('mousemove', handleMouseMove);
    document.removeEventListener('mouseup', handleMouseUp);
    document.body.style.cursor = '';
    document.body.style.userSelect = '';

    onResizeEnd?.(dimensions);
  }, [isResizing, dimensions, onResizeEnd, handleMouseMove]);

  const getCursor = (direction: string): string => {
    const cursors: { [key: string]: string } = {
      n: 'n-resize',
      e: 'e-resize',
      s: 's-resize',
      w: 'w-resize',
      ne: 'ne-resize',
      nw: 'nw-resize',
      se: 'se-resize',
      sw: 'sw-resize'
    };
    return cursors[direction] || 'default';
  };

  const renderHandle = (handle: ResizeHandle) => {
    const { direction, size, style: handleStyle = {} } = handle;
    
    const baseStyle: React.CSSProperties = {
      position: 'absolute',
      backgroundColor: 'transparent',
      cursor: getCursor(direction),
      opacity: 0,
      transition: 'opacity 0.2s ease',
      ...handleStyle
    };

    let positionStyle: React.CSSProperties = {};

    switch (direction) {
      case 'n':
        positionStyle = { top: 0, left: 0, right: 0, height: size };
        break;
      case 'e':
        positionStyle = { top: 0, right: 0, bottom: 0, width: size };
        break;
      case 's':
        positionStyle = { bottom: 0, left: 0, right: 0, height: size };
        break;
      case 'w':
        positionStyle = { top: 0, left: 0, bottom: 0, width: size };
        break;
      case 'ne':
        positionStyle = { top: 0, right: 0, width: size, height: size };
        break;
      case 'nw':
        positionStyle = { top: 0, left: 0, width: size, height: size };
        break;
      case 'se':
        positionStyle = { bottom: 0, right: 0, width: size, height: size };
        break;
      case 'sw':
        positionStyle = { bottom: 0, left: 0, width: size, height: size };
        break;
    }

    return (
      <div
        key={direction}
        className={`resize-handle resize-handle-${direction}`}
        data-testid="resize-handle"
        style={{ ...baseStyle, ...positionStyle }}
        onMouseDown={(e) => handleMouseDown(e, direction)}
      />
    );
  };

  return (
    <div
      ref={panelRef}
      className={`resizable-panel ${className} ${isResizing ? 'resizing' : ''}`}
      style={{
        position: 'relative',
        width: dimensions.width,
        height: dimensions.height,
        boxSizing: 'border-box',
        ...style
      }}
      onMouseEnter={() => {
        if (isResizable && panelRef.current) {
          const handleElements = panelRef.current.querySelectorAll('.resize-handle');
          handleElements.forEach(el => {
            (el as HTMLElement).style.opacity = '0.3';
          });
        }
      }}
      onMouseLeave={() => {
        if (isResizable && panelRef.current && !isResizing) {
          const handleElements = panelRef.current.querySelectorAll('.resize-handle');
          handleElements.forEach(el => {
            (el as HTMLElement).style.opacity = '0';
          });
        }
      }}
    >
      <div 
        className="resizable-panel-content"
        style={{
          width: '100%',
          height: '100%',
          overflow: 'hidden'
        }}
      >
        {children}
      </div>
      
      {isResizable && handles.map(renderHandle)}
      
      <style jsx>{`
        .resizable-panel {
          border: 1px solid #ddd;
          border-radius: 8px;
          background: #fff;
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
          transition: box-shadow 0.2s ease;
        }
        
        .resizable-panel:hover {
          box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        }
        
        .resizable-panel.resizing {
          box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
          z-index: 1000;
        }
        
        .resize-handle {
          z-index: 10;
        }
        
        .resize-handle:hover {
          opacity: 0.6 !important;
          background-color: rgba(0, 123, 255, 0.2) !important;
        }
        
        .resize-handle-se::after {
          content: '';
          position: absolute;
          right: 2px;
          bottom: 2px;
          width: 0;
          height: 0;
          border-left: 6px solid transparent;
          border-bottom: 6px solid #999;
        }
      `}</style>
    </div>
  );
};

export default ResizablePanel;