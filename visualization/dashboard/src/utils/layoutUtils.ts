import { Layout, Layouts } from 'react-grid-layout';

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
}

export interface GridDimensions {
  cols: number;
  rows: number;
  width: number;
  height: number;
}

export interface LayoutConstraints {
  minWidth: number;
  minHeight: number;
  maxWidth?: number;
  maxHeight?: number;
  aspectRatio?: number;
  snapToGrid?: boolean;
  preventOverlap?: boolean;
}

export interface LayoutMetrics {
  density: number;
  utilization: number;
  averageSize: { width: number; height: number };
  gaps: number;
  overlaps: number;
  efficiency: number;
}

/**
 * Calculate optimal grid dimensions for a given container size and item count
 */
export const calculateOptimalGridDimensions = (
  containerWidth: number,
  containerHeight: number,
  itemCount: number,
  itemAspectRatio: number = 1.5,
  margin: [number, number] = [10, 10]
): GridDimensions => {
  const [marginX, marginY] = margin;
  const availableWidth = containerWidth - marginX * 2;
  const availableHeight = containerHeight - marginY * 2;

  // Start with a square-ish grid
  const cols = Math.ceil(Math.sqrt(itemCount * (availableWidth / availableHeight) / itemAspectRatio));
  const rows = Math.ceil(itemCount / cols);

  const itemWidth = (availableWidth - marginX * (cols - 1)) / cols;
  const itemHeight = (availableHeight - marginY * (rows - 1)) / rows;

  return {
    cols,
    rows,
    width: itemWidth,
    height: itemHeight
  };
};

/**
 * Generate auto-layout for items based on container constraints
 */
export const generateAutoLayout = (
  itemIds: string[],
  containerCols: number,
  defaultItemSize: { width: number; height: number } = { width: 2, height: 2 },
  constraints: Partial<LayoutConstraints> = {}
): Layout[] => {
  const layout: Layout[] = [];
  let currentX = 0;
  let currentY = 0;

  itemIds.forEach((id, index) => {
    const item: Layout = {
      i: id,
      x: currentX,
      y: currentY,
      w: defaultItemSize.width,
      h: defaultItemSize.height
    };

    // Apply constraints
    if (constraints.minWidth) item.minW = constraints.minWidth;
    if (constraints.minHeight) item.minH = constraints.minHeight;
    if (constraints.maxWidth) item.maxW = constraints.maxWidth;
    if (constraints.maxHeight) item.maxH = constraints.maxHeight;

    layout.push(item);

    // Calculate next position
    currentX += defaultItemSize.width;
    if (currentX + defaultItemSize.width > containerCols) {
      currentX = 0;
      currentY += defaultItemSize.height;
    }
  });

  return layout;
};

/**
 * Optimize layout to minimize gaps and improve space utilization
 */
export const optimizeLayout = (
  layout: Layout[],
  containerCols: number,
  allowResize: boolean = true,
  allowReorder: boolean = true
): Layout[] => {
  if (!allowResize && !allowReorder) return layout;

  const optimized = [...layout];

  // Sort by priority (area, then position)
  if (allowReorder) {
    optimized.sort((a, b) => {
      const areaA = a.w * a.h;
      const areaB = b.w * b.h;
      if (areaA !== areaB) return areaB - areaA; // Larger items first
      return (a.y * containerCols + a.x) - (b.y * containerCols + b.x);
    });
  }

  // Compact layout by moving items up
  const compacted = compactLayout(optimized, containerCols);

  // Resize items to better fill available space
  if (allowResize) {
    return resizeToFillSpace(compacted, containerCols);
  }

  return compacted;
};

/**
 * Compact layout by moving items up to fill gaps
 */
export const compactLayout = (layout: Layout[], containerCols: number): Layout[] => {
  const compacted = [...layout];
  const occupiedCells = new Set<string>();

  // Create grid representation
  const grid: (string | null)[][] = [];
  const maxY = Math.max(...compacted.map(item => item.y + item.h));

  for (let y = 0; y <= maxY + 10; y++) {
    grid[y] = new Array(containerCols).fill(null);
  }

  // Place items and compact
  compacted.forEach(item => {
    const newY = findOptimalPosition(item, grid, containerCols);
    if (newY !== item.y) {
      item.y = newY;
    }
    
    // Mark cells as occupied
    for (let y = item.y; y < item.y + item.h; y++) {
      for (let x = item.x; x < item.x + item.w; x++) {
        if (grid[y] && x < containerCols) {
          grid[y][x] = item.i;
        }
      }
    }
  });

  return compacted;
};

/**
 * Find the optimal Y position for an item to minimize gaps
 */
const findOptimalPosition = (
  item: Layout,
  grid: (string | null)[][],
  containerCols: number
): number => {
  for (let y = 0; y < grid.length - item.h; y++) {
    let canPlace = true;
    
    for (let dy = 0; dy < item.h && canPlace; dy++) {
      for (let dx = 0; dx < item.w && canPlace; dx++) {
        const gridX = item.x + dx;
        const gridY = y + dy;
        
        if (gridX >= containerCols || !grid[gridY] || grid[gridY][gridX] !== null) {
          canPlace = false;
        }
      }
    }
    
    if (canPlace) {
      return y;
    }
  }
  
  return item.y;
};

/**
 * Resize items to better utilize available space
 */
export const resizeToFillSpace = (layout: Layout[], containerCols: number): Layout[] => {
  const resized = [...layout];
  
  resized.forEach(item => {
    // Find available space to the right
    const spaceRight = findAvailableSpace(item, 'right', layout, containerCols);
    if (spaceRight > 0 && item.w + spaceRight <= (item.maxW || containerCols)) {
      item.w += Math.min(spaceRight, 2); // Grow conservatively
    }
    
    // Find available space below
    const spaceBelow = findAvailableSpace(item, 'below', layout, containerCols);
    if (spaceBelow > 0 && item.h + spaceBelow <= (item.maxH || 10)) {
      item.h += Math.min(spaceBelow, 1);
    }
  });
  
  return resized;
};

/**
 * Find available space in a given direction
 */
const findAvailableSpace = (
  item: Layout,
  direction: 'right' | 'below',
  layout: Layout[],
  containerCols: number
): number => {
  const otherItems = layout.filter(i => i.i !== item.i);
  
  if (direction === 'right') {
    let space = 0;
    for (let x = item.x + item.w; x < containerCols; x++) {
      const hasConflict = otherItems.some(other => 
        other.x <= x && x < other.x + other.w &&
        other.y < item.y + item.h && item.y < other.y + other.h
      );
      
      if (hasConflict) break;
      space++;
    }
    return space;
  } else {
    let space = 0;
    for (let y = item.y + item.h; y < item.y + item.h + 5; y++) {
      const hasConflict = otherItems.some(other => 
        other.y <= y && y < other.y + other.h &&
        other.x < item.x + item.w && item.x < other.x + other.w
      );
      
      if (hasConflict) break;
      space++;
    }
    return space;
  }
};

/**
 * Calculate layout metrics for analysis and optimization
 */
export const calculateLayoutMetrics = (layout: Layout[], containerCols: number): LayoutMetrics => {
  if (layout.length === 0) {
    return {
      density: 0,
      utilization: 0,
      averageSize: { width: 0, height: 0 },
      gaps: 0,
      overlaps: 0,
      efficiency: 0
    };
  }

  const totalArea = layout.reduce((sum, item) => sum + (item.w * item.h), 0);
  const maxY = Math.max(...layout.map(item => item.y + item.h));
  const containerArea = containerCols * maxY;
  
  const averageWidth = layout.reduce((sum, item) => sum + item.w, 0) / layout.length;
  const averageHeight = layout.reduce((sum, item) => sum + item.h, 0) / layout.length;
  
  const gaps = calculateGaps(layout, containerCols, maxY);
  const overlaps = calculateOverlaps(layout);
  
  const density = layout.length / containerArea;
  const utilization = totalArea / containerArea;
  const efficiency = utilization - (gaps + overlaps) * 0.1;

  return {
    density,
    utilization,
    averageSize: { width: averageWidth, height: averageHeight },
    gaps,
    overlaps,
    efficiency: Math.max(0, efficiency)
  };
};

/**
 * Calculate number of gap cells in the layout
 */
const calculateGaps = (layout: Layout[], containerCols: number, maxY: number): number => {
  const grid: boolean[][] = [];
  
  for (let y = 0; y < maxY; y++) {
    grid[y] = new Array(containerCols).fill(false);
  }
  
  layout.forEach(item => {
    for (let y = item.y; y < item.y + item.h; y++) {
      for (let x = item.x; x < item.x + item.w; x++) {
        if (grid[y] && x < containerCols) {
          grid[y][x] = true;
        }
      }
    }
  });
  
  let gaps = 0;
  for (let y = 0; y < maxY; y++) {
    for (let x = 0; x < containerCols; x++) {
      if (!grid[y][x]) gaps++;
    }
  }
  
  return gaps;
};

/**
 * Calculate number of overlapping cells
 */
const calculateOverlaps = (layout: Layout[]): number => {
  let overlaps = 0;
  
  for (let i = 0; i < layout.length; i++) {
    for (let j = i + 1; j < layout.length; j++) {
      const item1 = layout[i];
      const item2 = layout[j];
      
      const overlapWidth = Math.max(0, Math.min(item1.x + item1.w, item2.x + item2.w) - Math.max(item1.x, item2.x));
      const overlapHeight = Math.max(0, Math.min(item1.y + item1.h, item2.y + item2.h) - Math.max(item1.y, item2.y));
      
      overlaps += overlapWidth * overlapHeight;
    }
  }
  
  return overlaps;
};

/**
 * Validate layout for conflicts and constraints
 */
export const validateLayout = (layout: Layout[], containerCols: number): {
  isValid: boolean;
  errors: string[];
  warnings: string[];
} => {
  const errors: string[] = [];
  const warnings: string[] = [];

  layout.forEach((item, index) => {
    // Check bounds
    if (item.x < 0 || item.y < 0) {
      errors.push(`Item ${item.i} has negative position`);
    }
    
    if (item.x + item.w > containerCols) {
      errors.push(`Item ${item.i} exceeds container width`);
    }
    
    if (item.w <= 0 || item.h <= 0) {
      errors.push(`Item ${item.i} has invalid size`);
    }
    
    // Check constraints
    if (item.minW && item.w < item.minW) {
      errors.push(`Item ${item.i} width below minimum`);
    }
    
    if (item.minH && item.h < item.minH) {
      errors.push(`Item ${item.i} height below minimum`);
    }
    
    if (item.maxW && item.w > item.maxW) {
      warnings.push(`Item ${item.i} width above maximum`);
    }
    
    if (item.maxH && item.h > item.maxH) {
      warnings.push(`Item ${item.i} height above maximum`);
    }
    
    // Check for overlaps
    for (let i = index + 1; i < layout.length; i++) {
      const other = layout[i];
      if (itemsOverlap(item, other)) {
        errors.push(`Item ${item.i} overlaps with ${other.i}`);
      }
    }
  });

  return {
    isValid: errors.length === 0,
    errors,
    warnings
  };
};

/**
 * Check if two layout items overlap
 */
const itemsOverlap = (item1: Layout, item2: Layout): boolean => {
  return !(
    item1.x + item1.w <= item2.x ||
    item2.x + item2.w <= item1.x ||
    item1.y + item1.h <= item2.y ||
    item2.y + item2.h <= item1.y
  );
};

/**
 * Convert layout to responsive layouts for different breakpoints
 */
export const generateResponsiveLayouts = (
  baseLayout: Layout[],
  breakpoints: { [key: string]: { cols: number; rowHeight: number } }
): Layouts => {
  const layouts: Layouts = {};

  Object.entries(breakpoints).forEach(([breakpoint, config]) => {
    layouts[breakpoint] = baseLayout.map(item => {
      const scaleFactor = config.cols / 12; // Assuming base is 12 columns
      
      return {
        ...item,
        w: Math.max(1, Math.round(item.w * scaleFactor)),
        x: Math.max(0, Math.round(item.x * scaleFactor))
      };
    });
  });

  return layouts;
};

/**
 * Save layout to localStorage with compression
 */
export const saveLayoutToStorage = (key: string, layout: Layouts, items?: any[]): void => {
  try {
    const data = {
      layout,
      items: items || [],
      timestamp: Date.now(),
      version: '1.0'
    };
    
    localStorage.setItem(key, JSON.stringify(data));
  } catch (error) {
    console.warn('Failed to save layout to localStorage:', error);
  }
};

/**
 * Load layout from localStorage with validation
 */
export const loadLayoutFromStorage = (key: string): { layout: Layouts; items: any[] } | null => {
  try {
    const stored = localStorage.getItem(key);
    if (!stored) return null;
    
    const data = JSON.parse(stored);
    
    // Validate data structure
    if (!data.layout || typeof data.layout !== 'object') {
      return null;
    }
    
    return {
      layout: data.layout,
      items: data.items || []
    };
  } catch (error) {
    console.warn('Failed to load layout from localStorage:', error);
    return null;
  }
};

/**
 * Clear layout from localStorage
 */
export const clearLayoutFromStorage = (key: string): void => {
  try {
    localStorage.removeItem(key);
  } catch (error) {
    console.warn('Failed to clear layout from localStorage:', error);
  }
};

export default {
  calculateOptimalGridDimensions,
  generateAutoLayout,
  optimizeLayout,
  compactLayout,
  calculateLayoutMetrics,
  validateLayout,
  generateResponsiveLayouts,
  saveLayoutToStorage,
  loadLayoutFromStorage,
  clearLayoutFromStorage
};