# Micro-Phase 9.23: Build Cortical Column Visualizer

## Objective
Create an interactive canvas-based visualization component for displaying cortical columns, their states, and activation patterns in real-time.

## Prerequisites
- JavaScript project setup complete
- WASM module loaded and initialized
- Understanding of Canvas API

## Task Description
Implement a responsive, performant visualization that shows the cortical column grid with activation levels, allocations, and spreading activation patterns.

## Specific Actions

1. **Create CorticalVisualizer class**:
   ```typescript
   // src/ui/CorticalVisualizer.ts
   import { CortexKGWasm } from 'cortexkg-wasm';
   
   export interface VisualizerConfig {
     canvas: HTMLCanvasElement;
     updateInterval?: number;
     colorScheme?: ColorScheme;
     interactive?: boolean;
     showLabels?: boolean;
   }
   
   export interface ColorScheme {
     background: string;
     unallocated: string;
     allocated: string;
     activationGradient: [string, string];
     selection: string;
     connectionColor: string;
   }
   
   export class CorticalVisualizer {
     private canvas: HTMLCanvasElement;
     private ctx: CanvasRenderingContext2D;
     private wasmModule: CortexKGWasm;
     private config: Required<VisualizerConfig>;
     private animationFrame: number | null = null;
     private columns: ColumnVisualState[] = [];
     private selectedColumn: number | null = null;
     private mousePos: { x: number; y: number } | null = null;
     
     constructor(wasmModule: CortexKGWasm, config: VisualizerConfig) {
       this.wasmModule = wasmModule;
       this.canvas = config.canvas;
       this.ctx = this.canvas.getContext('2d')!;
       
       this.config = {
         canvas: config.canvas,
         updateInterval: config.updateInterval ?? 100,
         colorScheme: config.colorScheme ?? this.getDefaultColorScheme(),
         interactive: config.interactive ?? true,
         showLabels: config.showLabels ?? false
       };
       
       this.setupCanvas();
       this.setupEventListeners();
       this.initializeColumns();
     }
     
     private getDefaultColorScheme(): ColorScheme {
       return {
         background: '#1a1a1a',
         unallocated: '#2a2a2a',
         allocated: '#4a4a4a',
         activationGradient: ['#0066cc', '#ff6600'],
         selection: '#ffff00',
         connectionColor: 'rgba(100, 200, 255, 0.3)'
       };
     }
   }
   ```

2. **Implement canvas setup and responsive sizing**:
   ```typescript
   private setupCanvas(): void {
     // Make canvas responsive
     const resizeCanvas = (): void => {
       const container = this.canvas.parentElement!;
       const dpr = window.devicePixelRatio || 1;
       
       this.canvas.width = container.clientWidth * dpr;
       this.canvas.height = container.clientHeight * dpr;
       this.canvas.style.width = container.clientWidth + 'px';
       this.canvas.style.height = container.clientHeight + 'px';
       
       this.ctx.scale(dpr, dpr);
       this.updateColumnLayout();
     };
     
     window.addEventListener('resize', resizeCanvas);
     resizeCanvas();
   }
   
   private updateColumnLayout(): void {
     const columnCount = this.wasmModule.column_count;
     const cols = Math.ceil(Math.sqrt(columnCount));
     const rows = Math.ceil(columnCount / cols);
     
     const width = this.canvas.width / window.devicePixelRatio;
     const height = this.canvas.height / window.devicePixelRatio;
     
     const cellWidth = width / cols;
     const cellHeight = height / rows;
     const cellSize = Math.min(cellWidth, cellHeight) * 0.9;
     const padding = cellSize * 0.1;
     
     // Update column positions
     this.columns = [];
     for (let i = 0; i < columnCount; i++) {
       const col = i % cols;
       const row = Math.floor(i / cols);
       
       this.columns.push({
         id: i,
         x: col * cellWidth + (cellWidth - cellSize) / 2,
         y: row * cellHeight + (cellHeight - cellSize) / 2,
         size: cellSize,
         activation: 0,
         allocated: false,
         hovered: false,
         connections: []
       });
     }
   }
   ```

3. **Implement rendering logic**:
   ```typescript
   interface ColumnVisualState {
     id: number;
     x: number;
     y: number;
     size: number;
     activation: number;
     allocated: boolean;
     hovered: boolean;
     connections: number[];
   }
   
   private render(): void {
     const { background } = this.config.colorScheme;
     
     // Clear canvas
     this.ctx.fillStyle = background;
     this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
     
     // Update column states from WASM
     this.updateColumnStates();
     
     // Render connections first (behind columns)
     if (this.selectedColumn !== null) {
       this.renderConnections();
     }
     
     // Render columns
     this.columns.forEach(column => {
       this.renderColumn(column);
     });
     
     // Render overlays
     if (this.config.showLabels && this.mousePos) {
       this.renderTooltip();
     }
   }
   
   private renderColumn(column: ColumnVisualState): void {
     const { unallocated, allocated, activationGradient, selection } = this.config.colorScheme;
     
     // Calculate color based on state
     let fillColor: string;
     if (column.activation > 0) {
       // Interpolate between activation gradient colors
       const t = column.activation;
       fillColor = this.interpolateColor(activationGradient[0], activationGradient[1], t);
     } else if (column.allocated) {
       fillColor = allocated;
     } else {
       fillColor = unallocated;
     }
     
     // Draw column
     this.ctx.fillStyle = fillColor;
     this.ctx.fillRect(column.x, column.y, column.size, column.size);
     
     // Draw selection highlight
     if (column.id === this.selectedColumn) {
       this.ctx.strokeStyle = selection;
       this.ctx.lineWidth = 3;
       this.ctx.strokeRect(column.x - 1, column.y - 1, column.size + 2, column.size + 2);
     }
     
     // Draw hover effect
     if (column.hovered) {
       this.ctx.fillStyle = 'rgba(255, 255, 255, 0.1)';
       this.ctx.fillRect(column.x, column.y, column.size, column.size);
     }
     
     // Draw allocation indicator
     if (column.allocated) {
       const centerX = column.x + column.size / 2;
       const centerY = column.y + column.size / 2;
       const radius = column.size * 0.15;
       
       this.ctx.fillStyle = '#ffffff';
       this.ctx.beginPath();
       this.ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
       this.ctx.fill();
     }
   }
   ```

4. **Implement interaction handling**:
   ```typescript
   private setupEventListeners(): void {
     if (!this.config.interactive) return;
     
     this.canvas.addEventListener('mousemove', (e) => {
       const rect = this.canvas.getBoundingClientRect();
       this.mousePos = {
         x: e.clientX - rect.left,
         y: e.clientY - rect.top
       };
       
       // Update hover states
       this.columns.forEach(column => {
         column.hovered = this.isPointInColumn(this.mousePos, column);
       });
     });
     
     this.canvas.addEventListener('click', (e) => {
       const column = this.getColumnAtPoint(this.mousePos!);
       if (column) {
         this.selectedColumn = column.id === this.selectedColumn ? null : column.id;
         this.onColumnSelected(column.id);
       }
     });
     
     this.canvas.addEventListener('mouseleave', () => {
       this.mousePos = null;
       this.columns.forEach(column => column.hovered = false);
     });
   }
   
   private isPointInColumn(point: { x: number; y: number }, column: ColumnVisualState): boolean {
     return point.x >= column.x && 
            point.x <= column.x + column.size &&
            point.y >= column.y && 
            point.y <= column.y + column.size;
   }
   
   private getColumnAtPoint(point: { x: number; y: number }): ColumnVisualState | null {
     return this.columns.find(col => this.isPointInColumn(point, col)) || null;
   }
   ```

5. **Implement animation system**:
   ```typescript
   public start(): void {
     if (this.animationFrame) return;
     
     let lastUpdate = 0;
     const animate = (timestamp: number): void => {
       if (timestamp - lastUpdate >= this.config.updateInterval) {
         this.render();
         lastUpdate = timestamp;
       }
       
       this.animationFrame = requestAnimationFrame(animate);
     };
     
     this.animationFrame = requestAnimationFrame(animate);
   }
   
   public stop(): void {
     if (this.animationFrame) {
       cancelAnimationFrame(this.animationFrame);
       this.animationFrame = null;
     }
   }
   
   public animateAllocation(columnId: number): void {
     const startTime = performance.now();
     const duration = 1000;
     
     const pulseAnimation = (currentTime: number): void => {
       const elapsed = currentTime - startTime;
       const progress = Math.min(elapsed / duration, 1);
       
       // Pulsing effect
       const intensity = Math.sin(progress * Math.PI * 4) * (1 - progress);
       
       if (this.columns[columnId]) {
         // Temporarily boost activation for visual effect
         const originalActivation = this.columns[columnId].activation;
         this.columns[columnId].activation = Math.max(originalActivation, intensity);
         
         if (progress >= 1) {
           this.columns[columnId].activation = originalActivation;
         } else {
           requestAnimationFrame(pulseAnimation);
         }
       }
     };
     
     requestAnimationFrame(pulseAnimation);
   }
   ```

6. **Add utility methods**:
   ```typescript
   private interpolateColor(color1: string, color2: string, t: number): string {
     // Convert hex to RGB
     const c1 = this.hexToRgb(color1);
     const c2 = this.hexToRgb(color2);
     
     const r = Math.round(c1.r + (c2.r - c1.r) * t);
     const g = Math.round(c1.g + (c2.g - c1.g) * t);
     const b = Math.round(c1.b + (c2.b - c1.b) * t);
     
     return `rgb(${r}, ${g}, ${b})`;
   }
   
   private hexToRgb(hex: string): { r: number; g: number; b: number } {
     const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
     return result ? {
       r: parseInt(result[1], 16),
       g: parseInt(result[2], 16),
       b: parseInt(result[3], 16)
     } : { r: 0, g: 0, b: 0 };
   }
   
   public highlightActivationPath(path: number[]): void {
     // Animate activation spreading along path
     path.forEach((columnId, index) => {
       setTimeout(() => {
         if (this.columns[columnId]) {
           this.columns[columnId].activation = 1.0;
           
           // Decay after a moment
           setTimeout(() => {
             this.columns[columnId].activation *= 0.5;
           }, 500);
         }
       }, index * 100);
     });
   }
   
   public setColorScheme(scheme: Partial<ColorScheme>): void {
     this.config.colorScheme = { ...this.config.colorScheme, ...scheme };
   }
   
   public exportAsImage(): string {
     return this.canvas.toDataURL('image/png');
   }
   ```

## Expected Outputs
- Interactive cortical column visualization
- Real-time activation display
- Column selection and hover effects
- Responsive canvas sizing
- Animation system for allocations
- Customizable color schemes
- Performance-optimized rendering

## Validation
1. Visualizer renders all columns correctly
2. Interactions (hover, click) work smoothly
3. Animations play without stuttering
4. Responsive sizing maintains aspect ratio
5. Performance stays above 30 FPS with 1024 columns

## Next Steps
- Create query interface component (micro-phase 9.24)
- Build allocation interface (micro-phase 9.25)