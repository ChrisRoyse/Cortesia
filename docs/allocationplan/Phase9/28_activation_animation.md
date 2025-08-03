# Micro-Phase 9.28: Activation Animation Component

## Objective
Create sophisticated activation animation system with smooth transitions, easing functions, and spreading activation effects for neuromorphic visualizations.

## Prerequisites
- Completed micro-phase 9.27 (Column Rendering)
- ColumnRenderer class available
- Understanding of animation timing and easing functions

## Task Description
Implement a comprehensive animation system that handles activation spreading, smooth transitions, and complex visual effects with performance optimization and configurable parameters.

## Specific Actions

1. **Create ActivationAnimator class structure**:
   ```typescript
   // src/ui/ActivationAnimator.ts
   import { ColumnRenderer, ColumnState } from './ColumnRenderer';
   
   export interface AnimationConfig {
     columnRenderer: ColumnRenderer;
     defaultDuration?: number;
     defaultEasing?: EasingFunction;
     enableSpreadAnimation?: boolean;
     enableParticleEffects?: boolean;
     maxConcurrentAnimations?: number;
   }
   
   export type EasingFunction = 'linear' | 'ease-in' | 'ease-out' | 'ease-in-out' | 'cubic-bezier';
   
   export interface AnimationKeyframe {
     time: number; // 0 to 1
     activation: number;
     position?: [number, number];
     size?: number;
     color?: [number, number, number, number];
     opacity?: number;
   }
   
   export interface ActivationAnimation {
     id: string;
     columnId: number;
     startTime: number;
     duration: number;
     keyframes: AnimationKeyframe[];
     easing: EasingFunction;
     onComplete?: () => void;
     onUpdate?: (progress: number, values: any) => void;
   }
   
   export interface SpreadingActivation {
     id: string;
     sourceColumnId: number;
     targetColumns: number[];
     spreadDelay: number;
     waveSpeed: number;
     intensity: number;
     decayRate: number;
     startTime: number;
   }
   
   export interface ParticleEffect {
     id: string;
     position: [number, number];
     velocity: [number, number];
     life: number;
     maxLife: number;
     size: number;
     color: [number, number, number, number];
     type: 'spark' | 'pulse' | 'trail';
   }
   
   export class ActivationAnimator {
     private columnRenderer: ColumnRenderer;
     private config: Required<AnimationConfig>;
     private activeAnimations: Map<string, ActivationAnimation> = new Map();
     private spreadingActivations: Map<string, SpreadingActivation> = new Map();
     private particleEffects: ParticleEffect[] = [];
     private animationFrame: number | null = null;
     private isRunning = false;
     private lastFrameTime = 0;
     private deltaTime = 0;
     
     constructor(config: AnimationConfig) {
       this.columnRenderer = config.columnRenderer;
       this.config = {
         columnRenderer: config.columnRenderer,
         defaultDuration: config.defaultDuration ?? 1000,
         defaultEasing: config.defaultEasing ?? 'ease-out',
         enableSpreadAnimation: config.enableSpreadAnimation ?? true,
         enableParticleEffects: config.enableParticleEffects ?? true,
         maxConcurrentAnimations: config.maxConcurrentAnimations ?? 50
       };
     }
   }
   ```

2. **Implement core animation system**:
   ```typescript
   public start(): void {
     if (this.isRunning) return;
     
     this.isRunning = true;
     this.lastFrameTime = performance.now();
     this.animate();
   }
   
   public stop(): void {
     this.isRunning = false;
     if (this.animationFrame) {
       cancelAnimationFrame(this.animationFrame);
       this.animationFrame = null;
     }
   }
   
   private animate(): void {
     if (!this.isRunning) return;
     
     const currentTime = performance.now();
     this.deltaTime = currentTime - this.lastFrameTime;
     this.lastFrameTime = currentTime;
     
     // Update all active animations
     this.updateAnimations(currentTime);
     this.updateSpreadingActivations(currentTime);
     
     if (this.config.enableParticleEffects) {
       this.updateParticleEffects();
     }
     
     // Continue animation loop
     this.animationFrame = requestAnimationFrame(() => this.animate());
   }
   
   private updateAnimations(currentTime: number): void {
     const completedAnimations: string[] = [];
     
     for (const [id, animation] of this.activeAnimations) {
       const elapsed = currentTime - animation.startTime;
       const progress = Math.min(elapsed / animation.duration, 1);
       
       if (progress >= 1) {
         // Animation complete
         this.applyAnimationFrame(animation, 1);
         completedAnimations.push(id);
         
         if (animation.onComplete) {
           animation.onComplete();
         }
       } else {
         // Apply current frame
         const easedProgress = this.applyEasing(progress, animation.easing);
         this.applyAnimationFrame(animation, easedProgress);
         
         if (animation.onUpdate) {
           animation.onUpdate(progress, this.getCurrentAnimationValues(animation, easedProgress));
         }
       }
     }
     
     // Remove completed animations
     for (const id of completedAnimations) {
       this.activeAnimations.delete(id);
     }
   }
   
   private applyAnimationFrame(animation: ActivationAnimation, progress: number): void {
     const values = this.getCurrentAnimationValues(animation, progress);
     
     // Update column state
     this.columnRenderer.updateColumn(animation.columnId, {
       activation: values.activation,
       position: values.position,
       size: values.size
     });
   }
   
   private getCurrentAnimationValues(animation: ActivationAnimation, progress: number): any {
     const keyframes = animation.keyframes;
     
     // Find surrounding keyframes
     let beforeFrame = keyframes[0];
     let afterFrame = keyframes[keyframes.length - 1];
     
     for (let i = 0; i < keyframes.length - 1; i++) {
       if (progress >= keyframes[i].time && progress <= keyframes[i + 1].time) {
         beforeFrame = keyframes[i];
         afterFrame = keyframes[i + 1];
         break;
       }
     }
     
     // Interpolate between keyframes
     const frameProgress = afterFrame.time === beforeFrame.time ? 0 : 
       (progress - beforeFrame.time) / (afterFrame.time - beforeFrame.time);
     
     return {
       activation: this.lerp(beforeFrame.activation, afterFrame.activation, frameProgress),
       position: beforeFrame.position && afterFrame.position ? [
         this.lerp(beforeFrame.position[0], afterFrame.position[0], frameProgress),
         this.lerp(beforeFrame.position[1], afterFrame.position[1], frameProgress)
       ] : undefined,
       size: beforeFrame.size !== undefined && afterFrame.size !== undefined ?
         this.lerp(beforeFrame.size, afterFrame.size, frameProgress) : undefined,
       color: beforeFrame.color && afterFrame.color ? [
         this.lerp(beforeFrame.color[0], afterFrame.color[0], frameProgress),
         this.lerp(beforeFrame.color[1], afterFrame.color[1], frameProgress),
         this.lerp(beforeFrame.color[2], afterFrame.color[2], frameProgress),
         this.lerp(beforeFrame.color[3], afterFrame.color[3], frameProgress)
       ] : undefined,
       opacity: beforeFrame.opacity !== undefined && afterFrame.opacity !== undefined ?
         this.lerp(beforeFrame.opacity, afterFrame.opacity, frameProgress) : undefined
     };
   }
   
   private applyEasing(t: number, easing: EasingFunction): number {
     switch (easing) {
       case 'linear':
         return t;
       case 'ease-in':
         return t * t;
       case 'ease-out':
         return 1 - Math.pow(1 - t, 2);
       case 'ease-in-out':
         return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
       case 'cubic-bezier':
         // Custom cubic-bezier approximation
         return this.cubicBezier(t, 0.25, 0.46, 0.45, 0.94);
       default:
         return t;
     }
   }
   
   private cubicBezier(t: number, x1: number, y1: number, x2: number, y2: number): number {
     // Simplified cubic-bezier calculation
     const cx = 3 * x1;
     const bx = 3 * (x2 - x1) - cx;
     const ax = 1 - cx - bx;
     
     const cy = 3 * y1;
     const by = 3 * (y2 - y1) - cy;
     const ay = 1 - cy - by;
     
     return ((ay * t + by) * t + cy) * t;
   }
   
   private lerp(a: number, b: number, t: number): number {
     return a + (b - a) * t;
   }
   ```

3. **Implement activation animations and presets**:
   ```typescript
   public animateActivation(
     columnId: number,
     targetActivation: number,
     duration?: number,
     easing?: EasingFunction
   ): string {
     const currentColumn = this.columnRenderer.getColumn(columnId);
     if (!currentColumn) return '';
     
     const animationId = this.generateAnimationId();
     const startActivation = currentColumn.activation;
     
     const animation: ActivationAnimation = {
       id: animationId,
       columnId,
       startTime: performance.now(),
       duration: duration ?? this.config.defaultDuration,
       easing: easing ?? this.config.defaultEasing,
       keyframes: [
         { time: 0, activation: startActivation },
         { time: 1, activation: targetActivation }
       ]
     };
     
     this.activeAnimations.set(animationId, animation);
     return animationId;
   }
   
   public animatePulse(
     columnId: number,
     intensity: number = 1.0,
     cycles: number = 1,
     duration?: number
   ): string {
     const currentColumn = this.columnRenderer.getColumn(columnId);
     if (!currentColumn) return '';
     
     const animationId = this.generateAnimationId();
     const baseActivation = currentColumn.activation;
     const peakActivation = Math.min(baseActivation + intensity, 1.0);
     
     // Create keyframes for pulse cycles
     const keyframes: AnimationKeyframe[] = [];
     const cycleTime = 1 / cycles;
     
     for (let i = 0; i < cycles; i++) {
       const cycleStart = i * cycleTime;
       const cyclePeak = cycleStart + cycleTime * 0.5;
       const cycleEnd = (i + 1) * cycleTime;
       
       keyframes.push(
         { time: cycleStart, activation: baseActivation },
         { time: cyclePeak, activation: peakActivation },
         { time: cycleEnd, activation: baseActivation }
       );
     }
     
     const animation: ActivationAnimation = {
       id: animationId,
       columnId,
       startTime: performance.now(),
       duration: duration ?? this.config.defaultDuration,
       easing: 'ease-in-out',
       keyframes
     };
     
     this.activeAnimations.set(animationId, animation);
     return animationId;
   }
   
   public animateGlow(
     columnId: number,
     glowIntensity: number = 0.5,
     duration?: number
   ): string {
     const currentColumn = this.columnRenderer.getColumn(columnId);
     if (!currentColumn) return '';
     
     const animationId = this.generateAnimationId();
     const baseSize = currentColumn.size;
     const glowSize = baseSize * (1 + glowIntensity);
     
     const animation: ActivationAnimation = {
       id: animationId,
       columnId,
       startTime: performance.now(),
       duration: duration ?? this.config.defaultDuration,
       easing: 'ease-out',
       keyframes: [
         { time: 0, size: baseSize, opacity: 1.0 },
         { time: 0.3, size: glowSize, opacity: 0.8 },
         { time: 1, size: baseSize, opacity: 1.0 }
       ]
     };
     
     this.activeAnimations.set(animationId, animation);
     return animationId;
   }
   
   public animateSparkle(
     columnId: number,
     sparkCount: number = 5,
     duration?: number
   ): string {
     const currentColumn = this.columnRenderer.getColumn(columnId);
     if (!currentColumn || !this.config.enableParticleEffects) return '';
     
     const animationId = this.generateAnimationId();
     const position = currentColumn.position;
     const size = currentColumn.size;
     
     // Create sparkle particles
     for (let i = 0; i < sparkCount; i++) {
       const angle = (i / sparkCount) * Math.PI * 2;
       const velocity = 50 + Math.random() * 30; // pixels per second
       
       const particle: ParticleEffect = {
         id: this.generateAnimationId(),
         position: [position[0], position[1]],
         velocity: [
           Math.cos(angle) * velocity,
           Math.sin(angle) * velocity
         ],
         life: duration ?? 1000,
         maxLife: duration ?? 1000,
         size: size * 0.1 + Math.random() * size * 0.1,
         color: [1, 1, 0.5, 1],
         type: 'spark'
       };
       
       this.particleEffects.push(particle);
     }
     
     return animationId;
   }
   ```

4. **Implement spreading activation system**:
   ```typescript
   public animateSpreadingActivation(
     sourceColumnId: number,
     targetColumns: number[],
     waveSpeed: number = 200,
     intensity: number = 0.8,
     decayRate: number = 0.1
   ): string {
     if (!this.config.enableSpreadAnimation) return '';
     
     const spreadId = this.generateAnimationId();
     const sourceColumn = this.columnRenderer.getColumn(sourceColumnId);
     
     if (!sourceColumn) return '';
     
     const spreading: SpreadingActivation = {
       id: spreadId,
       sourceColumnId,
       targetColumns: [...targetColumns],
       spreadDelay: 50, // ms between columns
       waveSpeed,
       intensity,
       decayRate,
       startTime: performance.now()
     };
     
     this.spreadingActivations.set(spreadId, spreading);
     
     // Start source animation
     this.animatePulse(sourceColumnId, intensity * 0.5, 1, 500);
     
     return spreadId;
   }
   
   private updateSpreadingActivations(currentTime: number): void {
     const completedSpreads: string[] = [];
     
     for (const [id, spread] of this.spreadingActivations) {
       const elapsed = currentTime - spread.startTime;
       const sourceColumn = this.columnRenderer.getColumn(spread.sourceColumnId);
       
       if (!sourceColumn) {
         completedSpreads.push(id);
         continue;
       }
       
       // Calculate which columns should be activating
       const activeColumnIndex = Math.floor(elapsed / spread.spreadDelay);
       
       if (activeColumnIndex >= spread.targetColumns.length) {
         completedSpreads.push(id);
         continue;
       }
       
       // Activate columns in sequence
       for (let i = 0; i <= activeColumnIndex && i < spread.targetColumns.length; i++) {
         const columnId = spread.targetColumns[i];
         const columnElapsed = elapsed - (i * spread.spreadDelay);
         
         if (columnElapsed >= 0) {
           const targetColumn = this.columnRenderer.getColumn(columnId);
           if (targetColumn) {
             // Calculate distance-based intensity
             const distance = this.calculateDistance(sourceColumn.position, targetColumn.position);
             const distanceDecay = Math.exp(-distance / spread.waveSpeed);
             const timeDecay = Math.exp(-columnElapsed / 1000 * spread.decayRate);
             const intensity = spread.intensity * distanceDecay * timeDecay;
             
             // Create wave effect
             this.createActivationWave(columnId, intensity, columnElapsed);
             
             // Add particle trail if enabled
             if (this.config.enableParticleEffects && i > 0) {
               this.createConnectionParticles(
                 sourceColumn.position,
                 targetColumn.position,
                 intensity
               );
             }
           }
         }
       }
     }
     
     // Remove completed spreads
     for (const id of completedSpreads) {
       this.spreadingActivations.delete(id);
     }
   }
   
   private createActivationWave(columnId: number, intensity: number, elapsed: number): void {
     // Create a wave-like activation animation
     const waveDuration = 800;
     const wavePhase = (elapsed % waveDuration) / waveDuration;
     const waveActivation = intensity * Math.sin(wavePhase * Math.PI * 2) * 0.5 + 0.5;
     
     this.columnRenderer.updateColumn(columnId, {
       activation: Math.max(0, waveActivation)
     });
   }
   
   private calculateDistance(pos1: [number, number], pos2: [number, number]): number {
     const dx = pos2[0] - pos1[0];
     const dy = pos2[1] - pos1[1];
     return Math.sqrt(dx * dx + dy * dy);
   }
   
   private createConnectionParticles(
     start: [number, number],
     end: [number, number],
     intensity: number
   ): void {
     const particleCount = Math.floor(intensity * 5);
     
     for (let i = 0; i < particleCount; i++) {
       const t = i / particleCount;
       const position: [number, number] = [
         start[0] + (end[0] - start[0]) * t,
         start[1] + (end[1] - start[1]) * t
       ];
       
       const particle: ParticleEffect = {
         id: this.generateAnimationId(),
         position,
         velocity: [
           (Math.random() - 0.5) * 20,
           (Math.random() - 0.5) * 20
         ],
         life: 500 + Math.random() * 300,
         maxLife: 800,
         size: 2 + Math.random() * 3,
         color: [0.4, 0.8, 1, intensity],
         type: 'trail'
       };
       
       this.particleEffects.push(particle);
     }
   }
   ```

5. **Implement particle effects and utilities**:
   ```typescript
   private updateParticleEffects(): void {
     for (let i = this.particleEffects.length - 1; i >= 0; i--) {
       const particle = this.particleEffects[i];
       
       // Update particle life
       particle.life -= this.deltaTime;
       
       if (particle.life <= 0) {
         this.particleEffects.splice(i, 1);
         continue;
       }
       
       // Update particle position
       particle.position[0] += particle.velocity[0] * (this.deltaTime / 1000);
       particle.position[1] += particle.velocity[1] * (this.deltaTime / 1000);
       
       // Apply physics based on particle type
       switch (particle.type) {
         case 'spark':
           // Apply gravity and drag
           particle.velocity[1] += 100 * (this.deltaTime / 1000); // gravity
           particle.velocity[0] *= 0.98; // drag
           particle.velocity[1] *= 0.98;
           break;
           
         case 'pulse':
           // Expand and fade
           particle.size *= 1.02;
           break;
           
         case 'trail':
           // Fade based on life
           const lifeRatio = particle.life / particle.maxLife;
           particle.color[3] = lifeRatio;
           particle.size *= 0.99;
           break;
       }
     }
   }
   
   public renderParticles(ctx: CanvasRenderingContext2D): void {
     if (!this.config.enableParticleEffects) return;
     
     ctx.save();
     
     for (const particle of this.particleEffects) {
       const alpha = particle.color[3];
       if (alpha <= 0) continue;
       
       ctx.globalAlpha = alpha;
       ctx.fillStyle = `rgb(${particle.color[0] * 255}, ${particle.color[1] * 255}, ${particle.color[2] * 255})`;
       
       ctx.beginPath();
       ctx.arc(particle.position[0], particle.position[1], particle.size, 0, Math.PI * 2);
       ctx.fill();
     }
     
     ctx.restore();
   }
   
   public stopAnimation(animationId: string): boolean {
     if (this.activeAnimations.has(animationId)) {
       this.activeAnimations.delete(animationId);
       return true;
     }
     
     if (this.spreadingActivations.has(animationId)) {
       this.spreadingActivations.delete(animationId);
       return true;
     }
     
     return false;
   }
   
   public stopAllAnimations(): void {
     this.activeAnimations.clear();
     this.spreadingActivations.clear();
     this.particleEffects = [];
   }
   
   public pauseAnimation(animationId: string): void {
     const animation = this.activeAnimations.get(animationId);
     if (animation) {
       // Store current progress to resume later
       const elapsed = performance.now() - animation.startTime;
       (animation as any).pausedAt = elapsed;
     }
   }
   
   public resumeAnimation(animationId: string): void {
     const animation = this.activeAnimations.get(animationId);
     if (animation && (animation as any).pausedAt !== undefined) {
       // Adjust start time to account for pause
       animation.startTime = performance.now() - (animation as any).pausedAt;
       delete (animation as any).pausedAt;
     }
   }
   
   public getActiveAnimationCount(): number {
     return this.activeAnimations.size + this.spreadingActivations.size;
   }
   
   public getParticleCount(): number {
     return this.particleEffects.length;
   }
   
   private generateAnimationId(): string {
     return `anim_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
   }
   
   public dispose(): void {
     this.stop();
     this.stopAllAnimations();
   }
   
   // Animation presets for common patterns
   public createActivationBurst(
     centerColumnId: number,
     radius: number,
     intensity: number = 0.8
   ): string[] {
     const centerColumn = this.columnRenderer.getColumn(centerColumnId);
     if (!centerColumn) return [];
     
     // Find columns within radius
     const nearbyColumns: number[] = [];
     // This would typically query the column manager for nearby columns
     // For now, simulate with a simple pattern
     
     const animationIds: string[] = [];
     
     // Animate center column first
     animationIds.push(this.animatePulse(centerColumnId, intensity, 2, 1000));
     
     // Animate nearby columns with delay
     for (let i = 0; i < nearbyColumns.length; i++) {
       setTimeout(() => {
         const distance = i; // simplified distance calculation
         const delayedIntensity = intensity * Math.exp(-distance / radius);
         animationIds.push(this.animateActivation(nearbyColumns[i], delayedIntensity, 800));
       }, i * 100);
     }
     
     return animationIds;
   }
   ```

## Expected Outputs
- Smooth activation transitions with configurable easing functions
- Spreading activation animations across connected columns
- Particle effects for visual enhancement
- Multiple concurrent animation support
- Performance-optimized animation loop
- Preset animation patterns for common use cases

## Validation
1. Animations maintain 60 FPS with 50+ concurrent animations
2. Spreading activation follows realistic timing patterns
3. Particle effects don't cause memory leaks
4. Animation interruption and resumption works correctly
5. Easing functions provide smooth visual transitions

## Next Steps
- Create touch interactions component (micro-phase 9.29)
- Integrate activation animator with cortical visualizer