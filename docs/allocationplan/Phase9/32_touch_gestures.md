# Micro-Phase 9.32: Advanced Touch Gesture System

## Objective
Create sophisticated touch gesture recognition system supporting multi-touch interactions, gestures, and mobile-specific navigation patterns for neuromorphic visualization.

## Prerequisites
- Completed micro-phase 9.31 (Mobile Detection)
- MobileDetector class available
- Understanding of touch event handling and gesture recognition

## Task Description
Implement comprehensive touch gesture system that recognizes complex multi-touch patterns, provides smooth interaction feedback, and enables intuitive navigation of cortical column visualizations on mobile devices.

## Specific Actions

1. **Create TouchGestureManager class with multi-touch support**:
   ```typescript
   // src/mobile/TouchGestureManager.ts
   import { MobileDetector } from './MobileDetector';

   export interface TouchPoint {
     id: number;
     x: number;
     y: number;
     timestamp: number;
     pressure?: number;
     radiusX?: number;
     radiusY?: number;
     force?: number;
   }

   export interface GestureState {
     type: 'none' | 'tap' | 'double-tap' | 'long-press' | 'pan' | 'pinch' | 'rotate' | 'swipe' | 'multi-select';
     startTime: number;
     duration: number;
     touches: TouchPoint[];
     center: { x: number; y: number };
     scale: number;
     rotation: number;
     velocity: { x: number; y: number };
     distance: number;
     direction?: 'up' | 'down' | 'left' | 'right';
     isActive: boolean;
   }

   export interface GestureConfig {
     tapThreshold: number;
     doubleTapDelay: number;
     longPressDelay: number;
     panThreshold: number;
     pinchThreshold: number;
     rotationThreshold: number;
     swipeThreshold: number;
     velocityThreshold: number;
     enableMultiTouch: boolean;
     enableGesturePreview: boolean;
     maxTouches: number;
   }

   export interface GestureCallbacks {
     onTap?: (point: TouchPoint, target?: number) => void;
     onDoubleTap?: (point: TouchPoint, target?: number) => void;
     onLongPress?: (point: TouchPoint, target?: number) => void;
     onPanStart?: (gesture: GestureState) => void;
     onPanMove?: (gesture: GestureState) => void;
     onPanEnd?: (gesture: GestureState) => void;
     onPinchStart?: (gesture: GestureState) => void;
     onPinchMove?: (gesture: GestureState) => void;
     onPinchEnd?: (gesture: GestureState) => void;
     onRotateStart?: (gesture: GestureState) => void;
     onRotateMove?: (gesture: GestureState) => void;
     onRotateEnd?: (gesture: GestureState) => void;
     onSwipe?: (gesture: GestureState) => void;
     onMultiSelect?: (selectedPoints: TouchPoint[]) => void;
   }

   export class TouchGestureManager {
     private element: HTMLElement;
     private mobileDetector: MobileDetector;
     private config: GestureConfig;
     private callbacks: GestureCallbacks;

     private activeTouches: Map<number, TouchPoint> = new Map();
     private currentGesture: GestureState | null = null;
     private gestureHistory: GestureState[] = [];
     private lastTapTime = 0;
     private lastTapPoint: TouchPoint | null = null;
     private longPressTimer: number | null = null;
     private isEnabled = true;

     // Gesture recognition state
     private initialDistance = 0;
     private initialRotation = 0;
     private initialCenter = { x: 0, y: 0 };
     private lastPanPoint = { x: 0, y: 0 };
     private velocityTracker: Array<{ x: number; y: number; time: number }> = [];

     constructor(
       element: HTMLElement,
       mobileDetector: MobileDetector,
       callbacks: GestureCallbacks,
       config?: Partial<GestureConfig>
     ) {
       this.element = element;
       this.mobileDetector = mobileDetector;
       this.callbacks = callbacks;
       
       this.config = {
         tapThreshold: 10,
         doubleTapDelay: 300,
         longPressDelay: 500,
         panThreshold: 10,
         pinchThreshold: 0.1,
         rotationThreshold: 10,
         swipeThreshold: 50,
         velocityThreshold: 100,
         enableMultiTouch: true,
         enableGesturePreview: true,
         maxTouches: 10,
         ...config
       };

       this.setupEventListeners();
     }

     private setupEventListeners(): void {
       // Prevent default touch behaviors
       this.element.style.touchAction = 'none';
       this.element.style.userSelect = 'none';

       // Touch events
       this.element.addEventListener('touchstart', this.handleTouchStart.bind(this), { passive: false });
       this.element.addEventListener('touchmove', this.handleTouchMove.bind(this), { passive: false });
       this.element.addEventListener('touchend', this.handleTouchEnd.bind(this), { passive: false });
       this.element.addEventListener('touchcancel', this.handleTouchCancel.bind(this), { passive: false });

       // Mouse events for desktop testing
       if (this.mobileDetector.isDesktop()) {
         this.element.addEventListener('mousedown', this.handleMouseDown.bind(this));
         this.element.addEventListener('mousemove', this.handleMouseMove.bind(this));
         this.element.addEventListener('mouseup', this.handleMouseUp.bind(this));
         this.element.addEventListener('contextmenu', (e) => e.preventDefault());
       }

       // Pointer events (modern approach)
       if ('PointerEvent' in window) {
         this.element.addEventListener('pointerdown', this.handlePointerDown.bind(this));
         this.element.addEventListener('pointermove', this.handlePointerMove.bind(this));
         this.element.addEventListener('pointerup', this.handlePointerUp.bind(this));
         this.element.addEventListener('pointercancel', this.handlePointerCancel.bind(this));
       }
     }

     private handleTouchStart(event: TouchEvent): void {
       event.preventDefault();
       
       if (!this.isEnabled) return;

       const currentTime = performance.now();
       
       // Process new touches
       for (let i = 0; i < event.changedTouches.length; i++) {
         const touch = event.changedTouches[i];
         const touchPoint: TouchPoint = {
           id: touch.identifier,
           x: touch.clientX,
           y: touch.clientY,
           timestamp: currentTime,
           pressure: touch.force,
           radiusX: touch.radiusX,
           radiusY: touch.radiusY,
           force: touch.force
         };

         this.activeTouches.set(touch.identifier, touchPoint);
       }

       // Analyze gesture
       this.analyzeGesture(currentTime);
     }

     private handleTouchMove(event: TouchEvent): void {
       event.preventDefault();
       
       if (!this.isEnabled) return;

       const currentTime = performance.now();

       // Update active touches
       for (let i = 0; i < event.changedTouches.length; i++) {
         const touch = event.changedTouches[i];
         const existingTouch = this.activeTouches.get(touch.identifier);
         
         if (existingTouch) {
           const touchPoint: TouchPoint = {
             ...existingTouch,
             x: touch.clientX,
             y: touch.clientY,
             timestamp: currentTime,
             pressure: touch.force,
             radiusX: touch.radiusX,
             radiusY: touch.radiusY,
             force: touch.force
           };

           this.activeTouches.set(touch.identifier, touchPoint);
         }
       }

       // Update gesture
       this.updateGesture(currentTime);
     }

     private handleTouchEnd(event: TouchEvent): void {
       event.preventDefault();
       
       if (!this.isEnabled) return;

       const currentTime = performance.now();

       // Remove ended touches
       for (let i = 0; i < event.changedTouches.length; i++) {
         const touch = event.changedTouches[i];
         this.activeTouches.delete(touch.identifier);
       }

       // Finalize gesture
       this.finalizeGesture(currentTime);
     }

     private handleTouchCancel(event: TouchEvent): void {
       this.clearGesture();
     }

     private analyzeGesture(timestamp: number): void {
       const touchCount = this.activeTouches.size;
       const touches = Array.from(this.activeTouches.values());

       // Clear previous long press timer
       if (this.longPressTimer) {
         clearTimeout(this.longPressTimer);
         this.longPressTimer = null;
       }

       if (touchCount === 1) {
         const touch = touches[0];
         
         // Check for double tap
         if (this.isDoubleTap(touch, timestamp)) {
           this.triggerDoubleTap(touch);
           return;
         }

         // Start single touch gesture
         this.currentGesture = {
           type: 'tap',
           startTime: timestamp,
           duration: 0,
           touches: [touch],
           center: { x: touch.x, y: touch.y },
           scale: 1,
           rotation: 0,
           velocity: { x: 0, y: 0 },
           distance: 0,
           isActive: true
         };

         this.lastPanPoint = { x: touch.x, y: touch.y };
         this.velocityTracker = [{ x: touch.x, y: touch.y, time: timestamp }];

         // Setup long press detection
         this.longPressTimer = window.setTimeout(() => {
           if (this.currentGesture && this.currentGesture.type === 'tap') {
             this.triggerLongPress(touch);
           }
         }, this.config.longPressDelay);

       } else if (touchCount === 2 && this.config.enableMultiTouch) {
         // Two finger gestures (pinch, rotate, pan)
         const [touch1, touch2] = touches;
         const center = this.calculateCenter([touch1, touch2]);
         const distance = this.calculateDistance(touch1, touch2);
         const rotation = this.calculateRotation(touch1, touch2);

         this.currentGesture = {
           type: 'none',
           startTime: timestamp,
           duration: 0,
           touches: [touch1, touch2],
           center,
           scale: 1,
           rotation: 0,
           velocity: { x: 0, y: 0 },
           distance,
           isActive: true
         };

         this.initialDistance = distance;
         this.initialRotation = rotation;
         this.initialCenter = center;
         this.lastPanPoint = center;

       } else if (touchCount > 2 && this.config.enableMultiTouch) {
         // Multi-touch selection
         this.currentGesture = {
           type: 'multi-select',
           startTime: timestamp,
           duration: 0,
           touches,
           center: this.calculateCenter(touches),
           scale: 1,
           rotation: 0,
           velocity: { x: 0, y: 0 },
           distance: 0,
           isActive: true
         };

         if (this.callbacks.onMultiSelect) {
           this.callbacks.onMultiSelect(touches);
         }
       }
     }

     private updateGesture(timestamp: number): void {
       if (!this.currentGesture) return;

       const touches = Array.from(this.activeTouches.values());
       const touchCount = touches.length;

       this.currentGesture.duration = timestamp - this.currentGesture.startTime;
       this.currentGesture.touches = touches;

       if (touchCount === 1) {
         this.updateSingleTouchGesture(touches[0], timestamp);
       } else if (touchCount === 2) {
         this.updateTwoFingerGesture(touches, timestamp);
       }
     }

     private updateSingleTouchGesture(touch: TouchPoint, timestamp: number): void {
       if (!this.currentGesture) return;

       const deltaX = touch.x - this.lastPanPoint.x;
       const deltaY = touch.y - this.lastPanPoint.y;
       const distance = Math.sqrt(deltaX * deltaX + deltaY * deltaY);

       // Update velocity tracking
       this.velocityTracker.push({ x: touch.x, y: touch.y, time: timestamp });
       if (this.velocityTracker.length > 5) {
         this.velocityTracker.shift();
       }

       // Calculate velocity
       if (this.velocityTracker.length >= 2) {
         const recent = this.velocityTracker[this.velocityTracker.length - 1];
         const older = this.velocityTracker[this.velocityTracker.length - 2];
         const dt = recent.time - older.time;
         
         if (dt > 0) {
           this.currentGesture.velocity = {
             x: (recent.x - older.x) / dt,
             y: (recent.y - older.y) / dt
           };
         }
       }

       // Determine gesture type based on movement
       if (this.currentGesture.type === 'tap' && distance > this.config.panThreshold) {
         this.currentGesture.type = 'pan';
         
         // Cancel long press
         if (this.longPressTimer) {
           clearTimeout(this.longPressTimer);
           this.longPressTimer = null;
         }

         if (this.callbacks.onPanStart) {
           this.callbacks.onPanStart(this.currentGesture);
         }
       }

       if (this.currentGesture.type === 'pan') {
         this.currentGesture.center = { x: touch.x, y: touch.y };
         this.currentGesture.distance += distance;
         this.lastPanPoint = { x: touch.x, y: touch.y };

         if (this.callbacks.onPanMove) {
           this.callbacks.onPanMove(this.currentGesture);
         }
       }
     }

     private updateTwoFingerGesture(touches: TouchPoint[], timestamp: number): void {
       if (!this.currentGesture || touches.length !== 2) return;

       const [touch1, touch2] = touches;
       const center = this.calculateCenter([touch1, touch2]);
       const distance = this.calculateDistance(touch1, touch2);
       const rotation = this.calculateRotation(touch1, touch2);

       const scale = distance / this.initialDistance;
       const rotationDelta = rotation - this.initialRotation;
       const panDelta = {
         x: center.x - this.initialCenter.x,
         y: center.y - this.initialCenter.y
       };

       this.currentGesture.center = center;
       this.currentGesture.scale = scale;
       this.currentGesture.rotation = rotationDelta;

       // Determine primary gesture
       const scaleChange = Math.abs(scale - 1);
       const rotationChange = Math.abs(rotationDelta);
       const panDistance = Math.sqrt(panDelta.x * panDelta.x + panDelta.y * panDelta.y);

       if (this.currentGesture.type === 'none') {
         if (scaleChange > this.config.pinchThreshold) {
           this.currentGesture.type = 'pinch';
           if (this.callbacks.onPinchStart) {
             this.callbacks.onPinchStart(this.currentGesture);
           }
         } else if (rotationChange > this.config.rotationThreshold) {
           this.currentGesture.type = 'rotate';
           if (this.callbacks.onRotateStart) {
             this.callbacks.onRotateStart(this.currentGesture);
           }
         } else if (panDistance > this.config.panThreshold) {
           this.currentGesture.type = 'pan';
           if (this.callbacks.onPanStart) {
             this.callbacks.onPanStart(this.currentGesture);
           }
         }
       }

       // Trigger appropriate callbacks
       if (this.currentGesture.type === 'pinch' && this.callbacks.onPinchMove) {
         this.callbacks.onPinchMove(this.currentGesture);
       } else if (this.currentGesture.type === 'rotate' && this.callbacks.onRotateMove) {
         this.callbacks.onRotateMove(this.currentGesture);
       } else if (this.currentGesture.type === 'pan' && this.callbacks.onPanMove) {
         this.callbacks.onPanMove(this.currentGesture);
       }
     }

     private finalizeGesture(timestamp: number): void {
       if (!this.currentGesture) return;

       this.currentGesture.duration = timestamp - this.currentGesture.startTime;
       this.currentGesture.isActive = false;

       // Clear long press timer
       if (this.longPressTimer) {
         clearTimeout(this.longPressTimer);
         this.longPressTimer = null;
       }

       const gestureType = this.currentGesture.type;
       const touches = this.currentGesture.touches;

       // Handle gesture completion
       if (gestureType === 'tap' && touches.length === 1) {
         this.triggerTap(touches[0]);
       } else if (gestureType === 'pan') {
         // Check for swipe
         const velocity = this.currentGesture.velocity;
         const speed = Math.sqrt(velocity.x * velocity.x + velocity.y * velocity.y);
         
         if (speed > this.config.velocityThreshold && this.currentGesture.distance > this.config.swipeThreshold) {
           this.currentGesture.type = 'swipe';
           this.currentGesture.direction = this.getSwipeDirection(velocity);
           
           if (this.callbacks.onSwipe) {
             this.callbacks.onSwipe(this.currentGesture);
           }
         } else if (this.callbacks.onPanEnd) {
           this.callbacks.onPanEnd(this.currentGesture);
         }
       } else if (gestureType === 'pinch' && this.callbacks.onPinchEnd) {
         this.callbacks.onPinchEnd(this.currentGesture);
       } else if (gestureType === 'rotate' && this.callbacks.onRotateEnd) {
         this.callbacks.onRotateEnd(this.currentGesture);
       }

       // Store in history
       this.gestureHistory.push({ ...this.currentGesture });
       if (this.gestureHistory.length > 10) {
         this.gestureHistory.shift();
       }

       this.currentGesture = null;
     }

     private isDoubleTap(touch: TouchPoint, timestamp: number): boolean {
       if (!this.lastTapPoint || timestamp - this.lastTapTime > this.config.doubleTapDelay) {
         return false;
       }

       const distance = Math.sqrt(
         Math.pow(touch.x - this.lastTapPoint.x, 2) + 
         Math.pow(touch.y - this.lastTapPoint.y, 2)
       );

       return distance < this.config.tapThreshold;
     }

     private triggerTap(touch: TouchPoint): void {
       this.lastTapTime = touch.timestamp;
       this.lastTapPoint = touch;

       if (this.callbacks.onTap) {
         this.callbacks.onTap(touch);
       }
     }

     private triggerDoubleTap(touch: TouchPoint): void {
       this.lastTapTime = 0;
       this.lastTapPoint = null;

       if (this.callbacks.onDoubleTap) {
         this.callbacks.onDoubleTap(touch);
       }
     }

     private triggerLongPress(touch: TouchPoint): void {
       if (this.callbacks.onLongPress) {
         this.callbacks.onLongPress(touch);
       }
     }

     private calculateCenter(touches: TouchPoint[]): { x: number; y: number } {
       const sum = touches.reduce((acc, touch) => ({
         x: acc.x + touch.x,
         y: acc.y + touch.y
       }), { x: 0, y: 0 });

       return {
         x: sum.x / touches.length,
         y: sum.y / touches.length
       };
     }

     private calculateDistance(touch1: TouchPoint, touch2: TouchPoint): number {
       return Math.sqrt(
         Math.pow(touch2.x - touch1.x, 2) + 
         Math.pow(touch2.y - touch1.y, 2)
       );
     }

     private calculateRotation(touch1: TouchPoint, touch2: TouchPoint): number {
       return Math.atan2(touch2.y - touch1.y, touch2.x - touch1.x) * 180 / Math.PI;
     }

     private getSwipeDirection(velocity: { x: number; y: number }): 'up' | 'down' | 'left' | 'right' {
       const absX = Math.abs(velocity.x);
       const absY = Math.abs(velocity.y);

       if (absX > absY) {
         return velocity.x > 0 ? 'right' : 'left';
       } else {
         return velocity.y > 0 ? 'down' : 'up';
       }
     }

     private clearGesture(): void {
       this.currentGesture = null;
       this.activeTouches.clear();
       
       if (this.longPressTimer) {
         clearTimeout(this.longPressTimer);
         this.longPressTimer = null;
       }
     }

     // Mouse event handlers for desktop testing
     private handleMouseDown(event: MouseEvent): void {
       const touch: TouchPoint = {
         id: 0,
         x: event.clientX,
         y: event.clientY,
         timestamp: performance.now()
       };

       this.activeTouches.set(0, touch);
       this.analyzeGesture(touch.timestamp);
     }

     private handleMouseMove(event: MouseEvent): void {
       if (this.activeTouches.has(0)) {
         const touch: TouchPoint = {
           id: 0,
           x: event.clientX,
           y: event.clientY,
           timestamp: performance.now()
         };

         this.activeTouches.set(0, touch);
         this.updateGesture(touch.timestamp);
       }
     }

     private handleMouseUp(event: MouseEvent): void {
       if (this.activeTouches.has(0)) {
         this.activeTouches.delete(0);
         this.finalizeGesture(performance.now());
       }
     }

     // Pointer event handlers
     private handlePointerDown(event: PointerEvent): void {
       const touch: TouchPoint = {
         id: event.pointerId,
         x: event.clientX,
         y: event.clientY,
         timestamp: performance.now(),
         pressure: event.pressure
       };

       this.activeTouches.set(event.pointerId, touch);
       this.analyzeGesture(touch.timestamp);
     }

     private handlePointerMove(event: PointerEvent): void {
       if (this.activeTouches.has(event.pointerId)) {
         const touch: TouchPoint = {
           id: event.pointerId,
           x: event.clientX,
           y: event.clientY,
           timestamp: performance.now(),
           pressure: event.pressure
         };

         this.activeTouches.set(event.pointerId, touch);
         this.updateGesture(touch.timestamp);
       }
     }

     private handlePointerUp(event: PointerEvent): void {
       if (this.activeTouches.has(event.pointerId)) {
         this.activeTouches.delete(event.pointerId);
         this.finalizeGesture(performance.now());
       }
     }

     private handlePointerCancel(event: PointerEvent): void {
       this.clearGesture();
     }

     // Public API
     public enable(): void {
       this.isEnabled = true;
     }

     public disable(): void {
       this.isEnabled = false;
       this.clearGesture();
     }

     public getCurrentGesture(): GestureState | null {
       return this.currentGesture;
     }

     public getGestureHistory(): GestureState[] {
       return [...this.gestureHistory];
     }

     public updateConfig(newConfig: Partial<GestureConfig>): void {
       this.config = { ...this.config, ...newConfig };
     }

     public getActiveTouches(): TouchPoint[] {
       return Array.from(this.activeTouches.values());
     }

     public dispose(): void {
       this.clearGesture();
       // Remove event listeners would go here
     }
   }
   ```

## Expected Outputs
- Multi-touch gesture recognition supporting tap, pan, pinch, rotate, and swipe
- Smooth gesture feedback with velocity tracking and inertia
- Configurable gesture thresholds optimized for different device types
- Cross-platform compatibility with touch, mouse, and pointer events
- Real-time gesture state tracking with history management

## Validation
1. Recognizes all gesture types accurately with <5% false positives
2. Multi-touch gestures work smoothly with up to 5 simultaneous touches
3. Gesture response latency remains under 16ms for 60fps interaction
4. Velocity tracking enables natural swipe and momentum behaviors
5. Touch event handling works across iOS Safari, Android Chrome, and desktop browsers

## Next Steps
- Mobile UI adaptation system (micro-phase 9.33)
- Performance throttling integration (micro-phase 9.34)