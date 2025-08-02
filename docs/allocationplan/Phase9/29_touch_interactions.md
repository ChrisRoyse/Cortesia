# Micro-Phase 9.29: Touch Interactions Component

## Objective
Create a comprehensive touch and mouse interaction system supporting gestures, multi-touch, and responsive controls for neuromorphic visualization.

## Prerequisites
- Completed micro-phase 9.28 (Activation Animation)
- ActivationAnimator class available
- Understanding of touch event handling and gesture recognition

## Task Description
Implement an advanced interaction system that handles mouse, touch, and pen inputs with gesture recognition, smooth panning/zooming, and context-sensitive actions.

## Specific Actions

1. **Create TouchInteractionManager class structure**:
   ```typescript
   // src/ui/TouchInteractionManager.ts
   import { CanvasManager } from './CanvasManager';
   import { ColumnRenderer, ColumnState } from './ColumnRenderer';
   import { ActivationAnimator } from './ActivationAnimator';
   
   export interface InteractionConfig {
     canvasManager: CanvasManager;
     columnRenderer: ColumnRenderer;
     activationAnimator: ActivationAnimator;
     enablePanZoom?: boolean;
     enableMultiTouch?: boolean;
     enableGestures?: boolean;
     enableContextMenu?: boolean;
     panSensitivity?: number;
     zoomSensitivity?: number;
     doubleTapDelay?: number;
     longPressDelay?: number;
   }
   
   export interface TouchPoint {
     id: number;
     x: number;
     y: number;
     startX: number;
     startY: number;
     startTime: number;
     lastX: number;
     lastY: number;
     lastTime: number;
     isActive: boolean;
   }
   
   export interface GestureState {
     type: 'none' | 'pan' | 'zoom' | 'rotate' | 'select' | 'drag';
     startTime: number;
     startDistance?: number;
     startAngle?: number;
     currentDistance?: number;
     currentAngle?: number;
     velocity: [number, number];
     acceleration: [number, number];
   }
   
   export interface InteractionEvent {
     type: 'tap' | 'double-tap' | 'long-press' | 'pan' | 'zoom' | 'rotate' | 'drag-start' | 'drag' | 'drag-end';
     position: [number, number];
     delta?: [number, number];
     scale?: number;
     rotation?: number;
     columnId?: number;
     touches: TouchPoint[];
     originalEvent: Event;
   }
   
   export interface ViewTransform {
     x: number;
     y: number;
     scale: number;
     rotation: number;
   }
   
   export class TouchInteractionManager {
     private canvasManager: CanvasManager;
     private columnRenderer: ColumnRenderer;
     private activationAnimator: ActivationAnimator;
     private config: Required<InteractionConfig>;
     private canvas: HTMLCanvasElement;
     private activeTouches: Map<number, TouchPoint> = new Map();
     private gestureState: GestureState = { type: 'none', startTime: 0, velocity: [0, 0], acceleration: [0, 0] };
     private viewTransform: ViewTransform = { x: 0, y: 0, scale: 1, rotation: 0 };
     private selectedColumns: Set<number> = new Set();
     private hoveredColumn: number | null = null;
     private isDragging = false;
     private dragStartPosition: [number, number] | null = null;
     private lastTapTime = 0;
     private lastTapPosition: [number, number] | null = null;
     private longPressTimer: number | null = null;
     private panVelocity: [number, number] = [0, 0];
     private momentumAnimation: number | null = null;
     
     // Event callbacks
     public onColumnTap?: (columnId: number, position: [number, number]) => void;
     public onColumnDoubleTap?: (columnId: number, position: [number, number]) => void;
     public onColumnLongPress?: (columnId: number, position: [number, number]) => void;
     public onSelectionChanged?: (selectedColumns: number[]) => void;
     public onViewTransformChanged?: (transform: ViewTransform) => void;
     
     constructor(config: InteractionConfig) {
       this.canvasManager = config.canvasManager;
       this.columnRenderer = config.columnRenderer;
       this.activationAnimator = config.activationAnimator;
       this.canvas = this.canvasManager.canvasElement;
       
       this.config = {
         canvasManager: config.canvasManager,
         columnRenderer: config.columnRenderer,
         activationAnimator: config.activationAnimator,
         enablePanZoom: config.enablePanZoom ?? true,
         enableMultiTouch: config.enableMultiTouch ?? true,
         enableGestures: config.enableGestures ?? true,
         enableContextMenu: config.enableContextMenu ?? true,
         panSensitivity: config.panSensitivity ?? 1.0,
         zoomSensitivity: config.zoomSensitivity ?? 0.1,
         doubleTapDelay: config.doubleTapDelay ?? 300,
         longPressDelay: config.longPressDelay ?? 500
       };
       
       this.setupEventListeners();
     }
   }
   ```

2. **Implement event listener setup and touch handling**:
   ```typescript
   private setupEventListeners(): void {
     // Touch events
     this.canvas.addEventListener('touchstart', this.handleTouchStart.bind(this), { passive: false });
     this.canvas.addEventListener('touchmove', this.handleTouchMove.bind(this), { passive: false });
     this.canvas.addEventListener('touchend', this.handleTouchEnd.bind(this), { passive: false });
     this.canvas.addEventListener('touchcancel', this.handleTouchCancel.bind(this), { passive: false });
     
     // Mouse events (for desktop compatibility)
     this.canvas.addEventListener('mousedown', this.handleMouseDown.bind(this));
     this.canvas.addEventListener('mousemove', this.handleMouseMove.bind(this));
     this.canvas.addEventListener('mouseup', this.handleMouseUp.bind(this));
     this.canvas.addEventListener('mouseleave', this.handleMouseLeave.bind(this));
     this.canvas.addEventListener('wheel', this.handleWheel.bind(this), { passive: false });
     
     // Pointer events (for pen/stylus support)
     if ('PointerEvent' in window) {
       this.canvas.addEventListener('pointerdown', this.handlePointerDown.bind(this));
       this.canvas.addEventListener('pointermove', this.handlePointerMove.bind(this));
       this.canvas.addEventListener('pointerup', this.handlePointerUp.bind(this));
       this.canvas.addEventListener('pointercancel', this.handlePointerCancel.bind(this));
     }
     
     // Context menu
     if (this.config.enableContextMenu) {
       this.canvas.addEventListener('contextmenu', this.handleContextMenu.bind(this));
     }
     
     // Prevent default behaviors
     this.canvas.addEventListener('selectstart', (e) => e.preventDefault());
     this.canvas.addEventListener('dragstart', (e) => e.preventDefault());
   }
   
   private handleTouchStart(event: TouchEvent): void {
     event.preventDefault();
     
     const currentTime = performance.now();
     
     for (let i = 0; i < event.changedTouches.length; i++) {
       const touch = event.changedTouches[i];
       const canvasPos = this.getCanvasPosition(touch.clientX, touch.clientY);
       
       const touchPoint: TouchPoint = {
         id: touch.identifier,
         x: canvasPos.x,
         y: canvasPos.y,
         startX: canvasPos.x,
         startY: canvasPos.y,
         startTime: currentTime,
         lastX: canvasPos.x,
         lastY: canvasPos.y,
         lastTime: currentTime,
         isActive: true
       };
       
       this.activeTouches.set(touch.identifier, touchPoint);
     }
     
     this.updateGestureState();
     this.handleInteractionStart(currentTime);
   }
   
   private handleTouchMove(event: TouchEvent): void {
     event.preventDefault();
     
     const currentTime = performance.now();
     let hasMovement = false;
     
     for (let i = 0; i < event.changedTouches.length; i++) {
       const touch = event.changedTouches[i];
       const touchPoint = this.activeTouches.get(touch.identifier);
       
       if (!touchPoint) continue;
       
       const canvasPos = this.getCanvasPosition(touch.clientX, touch.clientY);
       const deltaX = canvasPos.x - touchPoint.lastX;
       const deltaY = canvasPos.y - touchPoint.lastY;
       const deltaTime = currentTime - touchPoint.lastTime;
       
       if (Math.abs(deltaX) > 1 || Math.abs(deltaY) > 1) {
         hasMovement = true;
         
         // Update velocity
         if (deltaTime > 0) {
           const velocityX = deltaX / deltaTime * 1000; // pixels per second
           const velocityY = deltaY / deltaTime * 1000;
           this.panVelocity = [velocityX, velocityY];
         }
         
         touchPoint.lastX = canvasPos.x;
         touchPoint.lastY = canvasPos.y;
         touchPoint.lastTime = currentTime;
         touchPoint.x = canvasPos.x;
         touchPoint.y = canvasPos.y;
       }
     }
     
     if (hasMovement) {
       this.cancelLongPress();
       this.updateGestureState();
       this.handleGestureMove(currentTime);
     }
   }
   
   private handleTouchEnd(event: TouchEvent): void {
     event.preventDefault();
     
     const currentTime = performance.now();
     
     for (let i = 0; i < event.changedTouches.length; i++) {
       const touch = event.changedTouches[i];
       const touchPoint = this.activeTouches.get(touch.identifier);
       
       if (touchPoint) {
         touchPoint.isActive = false;
         this.handleTouchPointEnd(touchPoint, currentTime);
         this.activeTouches.delete(touch.identifier);
       }
     }
     
     this.updateGestureState();
     
     if (this.activeTouches.size === 0) {
       this.handleInteractionEnd(currentTime);
     }
   }
   
   private handleTouchCancel(event: TouchEvent): void {
     this.handleTouchEnd(event);
   }
   
   private getCanvasPosition(clientX: number, clientY: number): { x: number, y: number } {
     const rect = this.canvas.getBoundingClientRect();
     const dpr = this.canvasManager.viewportConfig.devicePixelRatio;
     
     return {
       x: (clientX - rect.left) * dpr,
       y: (clientY - rect.top) * dpr
     };
   }
   ```

3. **Implement gesture recognition and handling**:
   ```typescript
   private updateGestureState(): void {
     const touchCount = this.activeTouches.size;
     const currentTime = performance.now();
     
     if (touchCount === 0) {
       this.gestureState.type = 'none';
       return;
     }
     
     const touches = Array.from(this.activeTouches.values());
     
     if (touchCount === 1) {
       const touch = touches[0];
       const startDistance = this.calculateDistance(touch.startX, touch.startY, touch.x, touch.y);
       
       if (startDistance < 10 && this.gestureState.type === 'none') {
         // Potential tap
         this.setupLongPress(touch, currentTime);
       } else if (startDistance > 10 && this.gestureState.type !== 'pan') {
         this.gestureState.type = 'pan';
         this.gestureState.startTime = currentTime;
         this.cancelLongPress();
       }
     } else if (touchCount === 2 && this.config.enableMultiTouch) {
       const [touch1, touch2] = touches;
       const currentDistance = this.calculateDistance(touch1.x, touch1.y, touch2.x, touch2.y);
       const currentAngle = this.calculateAngle(touch1.x, touch1.y, touch2.x, touch2.y);
       
       if (this.gestureState.type === 'none' || this.gestureState.type === 'pan') {
         // Start two-finger gesture
         const startDistance = this.calculateDistance(touch1.startX, touch1.startY, touch2.startX, touch2.startY);
         
         this.gestureState.type = 'zoom';
         this.gestureState.startTime = currentTime;
         this.gestureState.startDistance = startDistance;
         this.gestureState.startAngle = this.calculateAngle(touch1.startX, touch1.startY, touch2.startX, touch2.startY);
         this.cancelLongPress();
       }
       
       this.gestureState.currentDistance = currentDistance;
       this.gestureState.currentAngle = currentAngle;
     }
   }
   
   private handleGestureMove(currentTime: number): void {
     switch (this.gestureState.type) {
       case 'pan':
         this.handlePan();
         break;
       case 'zoom':
         this.handleZoom();
         break;
       case 'select':
         this.handleSelection();
         break;
       case 'drag':
         this.handleDrag();
         break;
     }
   }
   
   private handlePan(): void {
     if (!this.config.enablePanZoom) return;
     
     const touches = Array.from(this.activeTouches.values());
     if (touches.length !== 1) return;
     
     const touch = touches[0];
     const deltaX = touch.x - touch.lastX;
     const deltaY = touch.y - touch.lastY;
     
     // Apply pan with sensitivity
     this.viewTransform.x += deltaX * this.config.panSensitivity;
     this.viewTransform.y += deltaY * this.config.panSensitivity;
     
     this.applyViewTransform();
     
     if (this.onViewTransformChanged) {
       this.onViewTransformChanged({ ...this.viewTransform });
     }
   }
   
   private handleZoom(): void {
     if (!this.config.enablePanZoom || !this.config.enableMultiTouch) return;
     
     const touches = Array.from(this.activeTouches.values());
     if (touches.length !== 2) return;
     
     const { startDistance, currentDistance } = this.gestureState;
     if (!startDistance || !currentDistance) return;
     
     // Calculate zoom factor
     const zoomFactor = currentDistance / startDistance;
     const newScale = Math.max(0.1, Math.min(5.0, this.viewTransform.scale * zoomFactor));
     
     // Calculate zoom center (midpoint between touches)
     const [touch1, touch2] = touches;
     const centerX = (touch1.x + touch2.x) / 2;
     const centerY = (touch1.y + touch2.y) / 2;
     
     // Apply zoom with proper centering
     const scaleChange = newScale / this.viewTransform.scale;
     this.viewTransform.x -= (centerX - this.viewTransform.x) * (scaleChange - 1);
     this.viewTransform.y -= (centerY - this.viewTransform.y) * (scaleChange - 1);
     this.viewTransform.scale = newScale;
     
     // Update gesture state
     this.gestureState.startDistance = currentDistance;
     
     this.applyViewTransform();
     
     if (this.onViewTransformChanged) {
       this.onViewTransformChanged({ ...this.viewTransform });
     }
   }
   
   private handleSelection(): void {
     const touches = Array.from(this.activeTouches.values());
     if (touches.length !== 1) return;
     
     const touch = touches[0];
     const worldPos = this.screenToWorldPosition(touch.x, touch.y);
     const columnId = this.getColumnAtPosition(worldPos.x, worldPos.y);
     
     if (columnId !== null && columnId !== this.hoveredColumn) {
       this.updateHoveredColumn(columnId);
     }
   }
   
   private handleDrag(): void {
     const touches = Array.from(this.activeTouches.values());
     if (touches.length !== 1) return;
     
     const touch = touches[0];
     const worldPos = this.screenToWorldPosition(touch.x, touch.y);
     
     // Update drag visualization
     this.updateDragVisualization(worldPos);
   }
   
   private handleTouchPointEnd(touchPoint: TouchPoint, currentTime: number): void {
     const duration = currentTime - touchPoint.startTime;
     const distance = this.calculateDistance(touchPoint.startX, touchPoint.startY, touchPoint.x, touchPoint.y);
     
     // Determine interaction type
     if (distance < 10 && duration < 300) {
       // Tap
       this.handleTap(touchPoint, currentTime);
     } else if (this.gestureState.type === 'pan') {
       this.handlePanEnd();
     } else if (this.gestureState.type === 'drag') {
       this.handleDragEnd(touchPoint);
     }
     
     this.cancelLongPress();
   }
   
   private handleTap(touchPoint: TouchPoint, currentTime: number): void {
     const worldPos = this.screenToWorldPosition(touchPoint.x, touchPoint.y);
     const columnId = this.getColumnAtPosition(worldPos.x, worldPos.y);
     
     // Check for double tap
     const isDoubleTap = this.lastTapTime > 0 && 
       (currentTime - this.lastTapTime) < this.config.doubleTapDelay &&
       this.lastTapPosition &&
       this.calculateDistance(touchPoint.x, touchPoint.y, this.lastTapPosition[0], this.lastTapPosition[1]) < 50;
     
     if (isDoubleTap) {
       this.handleDoubleTap(columnId, [worldPos.x, worldPos.y]);
       this.lastTapTime = 0;
       this.lastTapPosition = null;
     } else {
       this.handleSingleTap(columnId, [worldPos.x, worldPos.y]);
       this.lastTapTime = currentTime;
       this.lastTapPosition = [touchPoint.x, touchPoint.y];
     }
   }
   
   private handleSingleTap(columnId: number | null, position: [number, number]): void {
     if (columnId !== null) {
       // Toggle column selection
       if (this.selectedColumns.has(columnId)) {
         this.selectedColumns.delete(columnId);
         this.columnRenderer.updateColumn(columnId, { selected: false });
       } else {
         this.selectedColumns.add(columnId);
         this.columnRenderer.updateColumn(columnId, { selected: true });
         
         // Trigger activation animation
         this.activationAnimator.animatePulse(columnId, 0.5, 1, 500);
       }
       
       if (this.onColumnTap) {
         this.onColumnTap(columnId, position);
       }
       
       if (this.onSelectionChanged) {
         this.onSelectionChanged(Array.from(this.selectedColumns));
       }
     } else {
       // Clear selection
       this.clearSelection();
     }
   }
   
   private handleDoubleTap(columnId: number | null, position: [number, number]): void {
     if (columnId !== null) {
       // Trigger special activation animation
       this.activationAnimator.animateGlow(columnId, 1.0, 1000);
       this.activationAnimator.animateSparkle(columnId, 8, 1500);
       
       if (this.onColumnDoubleTap) {
         this.onColumnDoubleTap(columnId, position);
       }
     }
   }
   ```

4. **Implement mouse and wheel event handling**:
   ```typescript
   private handleMouseDown(event: MouseEvent): void {
     if (event.button !== 0) return; // Only handle left mouse button
     
     const canvasPos = this.getCanvasPosition(event.clientX, event.clientY);
     const currentTime = performance.now();
     
     const touchPoint: TouchPoint = {
       id: -1, // Use -1 for mouse
       x: canvasPos.x,
       y: canvasPos.y,
       startX: canvasPos.x,
       startY: canvasPos.y,
       startTime: currentTime,
       lastX: canvasPos.x,
       lastY: canvasPos.y,
       lastTime: currentTime,
       isActive: true
     };
     
     this.activeTouches.set(-1, touchPoint);
     this.updateGestureState();
     this.handleInteractionStart(currentTime);
   }
   
   private handleMouseMove(event: MouseEvent): void {
     const canvasPos = this.getCanvasPosition(event.clientX, event.clientY);
     const currentTime = performance.now();
     
     // Update hover state
     const worldPos = this.screenToWorldPosition(canvasPos.x, canvasPos.y);
     const columnId = this.getColumnAtPosition(worldPos.x, worldPos.y);
     this.updateHoveredColumn(columnId);
     
     // Handle mouse drag if mouse is down
     const mouseTouch = this.activeTouches.get(-1);
     if (mouseTouch) {
       mouseTouch.lastX = mouseTouch.x;
       mouseTouch.lastY = mouseTouch.y;
       mouseTouch.x = canvasPos.x;
       mouseTouch.y = canvasPos.y;
       mouseTouch.lastTime = currentTime;
       
       this.updateGestureState();
       this.handleGestureMove(currentTime);
     }
   }
   
   private handleMouseUp(event: MouseEvent): void {
     const mouseTouch = this.activeTouches.get(-1);
     if (mouseTouch) {
       const currentTime = performance.now();
       mouseTouch.isActive = false;
       this.handleTouchPointEnd(mouseTouch, currentTime);
       this.activeTouches.delete(-1);
       this.updateGestureState();
       this.handleInteractionEnd(currentTime);
     }
   }
   
   private handleMouseLeave(event: MouseEvent): void {
     this.updateHoveredColumn(null);
     this.handleMouseUp(event);
   }
   
   private handleWheel(event: WheelEvent): void {
     if (!this.config.enablePanZoom) return;
     
     event.preventDefault();
     
     const canvasPos = this.getCanvasPosition(event.clientX, event.clientY);
     const worldPos = this.screenToWorldPosition(canvasPos.x, canvasPos.y);
     
     // Calculate zoom
     const zoomFactor = event.deltaY > 0 ? 0.9 : 1.1;
     const newScale = Math.max(0.1, Math.min(5.0, this.viewTransform.scale * zoomFactor));
     
     // Apply zoom centered on mouse position
     const scaleChange = newScale / this.viewTransform.scale;
     this.viewTransform.x -= (worldPos.x - this.viewTransform.x) * (scaleChange - 1);
     this.viewTransform.y -= (worldPos.y - this.viewTransform.y) * (scaleChange - 1);
     this.viewTransform.scale = newScale;
     
     this.applyViewTransform();
     
     if (this.onViewTransformChanged) {
       this.onViewTransformChanged({ ...this.viewTransform });
     }
   }
   ```

5. **Implement helper methods and utilities**:
   ```typescript
   private setupLongPress(touchPoint: TouchPoint, currentTime: number): void {
     this.cancelLongPress();
     
     this.longPressTimer = window.setTimeout(() => {
       const worldPos = this.screenToWorldPosition(touchPoint.x, touchPoint.y);
       const columnId = this.getColumnAtPosition(worldPos.x, worldPos.y);
       
       if (columnId !== null) {
         // Trigger long press feedback
         this.activationAnimator.animatePulse(columnId, 0.8, 2, 1000);
         
         if (this.onColumnLongPress) {
           this.onColumnLongPress(columnId, [worldPos.x, worldPos.y]);
         }
       }
       
       this.longPressTimer = null;
     }, this.config.longPressDelay);
   }
   
   private cancelLongPress(): void {
     if (this.longPressTimer) {
       clearTimeout(this.longPressTimer);
       this.longPressTimer = null;
     }
   }
   
   private handlePanEnd(): void {
     // Apply momentum scrolling
     if (this.config.enablePanZoom && this.panVelocity[0] !== 0 || this.panVelocity[1] !== 0) {
       this.startMomentumAnimation();
     }
   }
   
   private startMomentumAnimation(): void {
     if (this.momentumAnimation) {
       cancelAnimationFrame(this.momentumAnimation);
     }
     
     const friction = 0.95;
     const threshold = 1; // pixels per second
     
     const animate = () => {
       const speed = Math.sqrt(this.panVelocity[0] ** 2 + this.panVelocity[1] ** 2);
       
       if (speed < threshold) {
         this.panVelocity = [0, 0];
         this.momentumAnimation = null;
         return;
       }
       
       // Apply momentum
       this.viewTransform.x += this.panVelocity[0] * 0.016; // 60fps
       this.viewTransform.y += this.panVelocity[1] * 0.016;
       
       // Apply friction
       this.panVelocity[0] *= friction;
       this.panVelocity[1] *= friction;
       
       this.applyViewTransform();
       
       if (this.onViewTransformChanged) {
         this.onViewTransformChanged({ ...this.viewTransform });
       }
       
       this.momentumAnimation = requestAnimationFrame(animate);
     };
     
     this.momentumAnimation = requestAnimationFrame(animate);
   }
   
   private updateHoveredColumn(columnId: number | null): void {
     if (this.hoveredColumn !== columnId) {
       // Clear previous hover
       if (this.hoveredColumn !== null) {
         this.columnRenderer.updateColumn(this.hoveredColumn, { hovered: false });
       }
       
       // Set new hover
       this.hoveredColumn = columnId;
       if (columnId !== null) {
         this.columnRenderer.updateColumn(columnId, { hovered: true });
       }
     }
   }
   
   private clearSelection(): void {
     for (const columnId of this.selectedColumns) {
       this.columnRenderer.updateColumn(columnId, { selected: false });
     }
     this.selectedColumns.clear();
     
     if (this.onSelectionChanged) {
       this.onSelectionChanged([]);
     }
   }
   
   private screenToWorldPosition(screenX: number, screenY: number): { x: number, y: number } {
     return {
       x: (screenX - this.viewTransform.x) / this.viewTransform.scale,
       y: (screenY - this.viewTransform.y) / this.viewTransform.scale
     };
   }
   
   private worldToScreenPosition(worldX: number, worldY: number): { x: number, y: number } {
     return {
       x: worldX * this.viewTransform.scale + this.viewTransform.x,
       y: worldY * this.viewTransform.scale + this.viewTransform.y
     };
   }
   
   private getColumnAtPosition(x: number, y: number): number | null {
     // This would typically query the column renderer for the column at position
     // Implementation depends on column layout and spatial indexing
     // For now, return null as placeholder
     return null;
   }
   
   private applyViewTransform(): void {
     // This would update the rendering transform
     // Implementation depends on how the view transform is applied to rendering
   }
   
   private calculateDistance(x1: number, y1: number, x2: number, y2: number): number {
     const dx = x2 - x1;
     const dy = y2 - y1;
     return Math.sqrt(dx * dx + dy * dy);
   }
   
   private calculateAngle(x1: number, y1: number, x2: number, y2: number): number {
     return Math.atan2(y2 - y1, x2 - x1);
   }
   
   private handleInteractionStart(currentTime: number): void {
     // Override in subclasses for custom interaction start handling
   }
   
   private handleInteractionEnd(currentTime: number): void {
     // Override in subclasses for custom interaction end handling
   }
   
   private handleDragEnd(touchPoint: TouchPoint): void {
     // Handle end of drag operation
     this.isDragging = false;
     this.dragStartPosition = null;
   }
   
   private updateDragVisualization(position: { x: number, y: number }): void {
     // Update drag visualization
   }
   
   private handleContextMenu(event: MouseEvent): void {
     event.preventDefault();
     
     const canvasPos = this.getCanvasPosition(event.clientX, event.clientY);
     const worldPos = this.screenToWorldPosition(canvasPos.x, canvasPos.y);
     const columnId = this.getColumnAtPosition(worldPos.x, worldPos.y);
     
     // Show context menu
     this.showContextMenu(columnId, [event.clientX, event.clientY]);
   }
   
   private showContextMenu(columnId: number | null, screenPosition: [number, number]): void {
     // Implementation for context menu display
   }
   
   // Pointer event handlers for pen/stylus support
   private handlePointerDown(event: PointerEvent): void {
     // Handle pointer down (supports pressure, tilt, etc.)
   }
   
   private handlePointerMove(event: PointerEvent): void {
     // Handle pointer move with pressure sensitivity
   }
   
   private handlePointerUp(event: PointerEvent): void {
     // Handle pointer up
   }
   
   private handlePointerCancel(event: PointerEvent): void {
     // Handle pointer cancel
   }
   
   // Public API
   public getSelectedColumns(): number[] {
     return Array.from(this.selectedColumns);
   }
   
   public selectColumn(columnId: number): void {
     this.selectedColumns.add(columnId);
     this.columnRenderer.updateColumn(columnId, { selected: true });
     
     if (this.onSelectionChanged) {
       this.onSelectionChanged(Array.from(this.selectedColumns));
     }
   }
   
   public deselectColumn(columnId: number): void {
     this.selectedColumns.delete(columnId);
     this.columnRenderer.updateColumn(columnId, { selected: false });
     
     if (this.onSelectionChanged) {
       this.onSelectionChanged(Array.from(this.selectedColumns));
     }
   }
   
   public setViewTransform(transform: Partial<ViewTransform>): void {
     Object.assign(this.viewTransform, transform);
     this.applyViewTransform();
     
     if (this.onViewTransformChanged) {
       this.onViewTransformChanged({ ...this.viewTransform });
     }
   }
   
   public resetView(): void {
     this.viewTransform = { x: 0, y: 0, scale: 1, rotation: 0 };
     this.applyViewTransform();
     
     if (this.onViewTransformChanged) {
       this.onViewTransformChanged({ ...this.viewTransform });
     }
   }
   
   public dispose(): void {
     this.cancelLongPress();
     
     if (this.momentumAnimation) {
       cancelAnimationFrame(this.momentumAnimation);
       this.momentumAnimation = null;
     }
     
     this.activeTouches.clear();
     this.selectedColumns.clear();
   }
   ```

## Expected Outputs
- Responsive touch and mouse interaction system
- Multi-touch gesture recognition (pan, zoom, rotate)
- Smooth momentum scrolling and view transforms
- Column selection and hover feedback
- Context menu support for right-click interactions
- Pressure-sensitive pen/stylus support

## Validation
1. Touch interactions work smoothly on mobile devices
2. Multi-touch gestures (pinch-to-zoom) respond correctly
3. Mouse wheel zoom centers properly on cursor position
4. Selection state updates correctly with visual feedback
5. Momentum scrolling feels natural and responsive

## Next Steps
- Create real-time updates component (micro-phase 9.30)
- Integrate touch interactions with all visualization components