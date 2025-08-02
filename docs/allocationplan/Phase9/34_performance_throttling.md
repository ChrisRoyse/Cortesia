# Micro-Phase 9.34: Performance Throttling and Battery Optimization

## Objective
Implement intelligent performance throttling system that dynamically adjusts computational intensity based on device capabilities, battery status, and thermal conditions to ensure optimal user experience.

## Prerequisites
- Completed micro-phase 9.33 (Mobile UI Adaptation)
- MobileUIAdapter class available
- MobileDetector providing device capabilities
- Understanding of performance monitoring and adaptive algorithms

## Task Description
Create comprehensive performance throttling system that monitors device conditions in real-time and automatically adjusts WASM computation intensity, rendering quality, and update frequencies to maintain smooth performance while preserving battery life.

## Specific Actions

1. **Create PerformanceThrottler class with adaptive algorithms**:
   ```typescript
   // src/mobile/PerformanceThrottler.ts
   import { MobileDetector, DeviceCapabilities } from './MobileDetector';
   import { MobileUIAdapter } from './MobileUIAdapter';

   export interface PerformanceProfile {
     name: 'battery-saver' | 'balanced' | 'performance' | 'custom';
     wasmThreads: number;
     updateFrequency: number; // Hz
     renderQuality: 'low' | 'medium' | 'high';
     maxColumns: number;
     enableAnimations: boolean;
     enableEffects: boolean;
     enableParallelProcessing: boolean;
     memoryLimit: number; // MB
     cpuUsageLimit: number; // 0-1
     thermalThreshold: number; // 0-1
   }

   export interface SystemMetrics {
     frameRate: number;
     frameTime: number;
     memoryUsage: number;
     cpuUsage: number;
     batteryLevel: number;
     batteryCharging: boolean;
     thermalState: 'nominal' | 'fair' | 'serious' | 'critical';
     networkSpeed: 'slow' | 'medium' | 'fast';
     deviceTemperature?: number;
   }

   export interface ThrottlingConfig {
     enableAdaptiveThrottling: boolean;
     enableBatteryOptimization: boolean;
     enableThermalManagement: boolean;
     enableMemoryManagement: boolean;
     enableNetworkOptimization: boolean;
     aggressiveMode: boolean;
     monitoringInterval: number;
     adaptationSensitivity: number;
   }

   export interface PerformanceBudget {
     frameTimeTarget: number; // ms
     memoryBudget: number; // MB
     cpuBudget: number; // 0-1
     batteryDrainRate: number; // %/hour
     thermalBudget: number; // 0-1
   }

   export class PerformanceThrottler {
     private mobileDetector: MobileDetector;
     private uiAdapter: MobileUIAdapter;
     private currentProfile: PerformanceProfile;
     private config: ThrottlingConfig;
     private budget: PerformanceBudget;

     private metrics: SystemMetrics = {
       frameRate: 60,
       frameTime: 16.67,
       memoryUsage: 0,
       cpuUsage: 0,
       batteryLevel: 1.0,
       batteryCharging: false,
       thermalState: 'nominal',
       networkSpeed: 'medium'
     };

     private metricHistory: Array<{
       timestamp: number;
       metrics: SystemMetrics;
     }> = [];

     private performanceObserver: PerformanceObserver | null = null;
     private monitoringInterval: number | null = null;
     private adaptationTimer: number | null = null;
     
     // Performance monitoring
     private frameTimings: number[] = [];
     private lastFrameTime = 0;
     private memoryPressureCount = 0;
     private thermalPressureCount = 0;
     private batteryDrainSamples: Array<{ time: number; level: number }> = [];

     // Adaptive algorithms
     private adaptationState = {
       direction: 0, // -1 for throttling down, 1 for throttling up
       magnitude: 0, // How aggressive the changes should be
       confidence: 0, // Confidence in current adaptation
       stabilityCounter: 0 // Frames of stable performance
     };

     // Event callbacks
     public onProfileChange?: (profile: PerformanceProfile) => void;
     public onThrottleChange?: (throttleLevel: number) => void;
     public onBatteryOptimization?: (enabled: boolean) => void;
     public onThermalAlert?: (state: string) => void;

     constructor(
       mobileDetector: MobileDetector,
       uiAdapter: MobileUIAdapter,
       config?: Partial<ThrottlingConfig>
     ) {
       this.mobileDetector = mobileDetector;
       this.uiAdapter = uiAdapter;
       
       this.config = {
         enableAdaptiveThrottling: true,
         enableBatteryOptimization: true,
         enableThermalManagement: true,
         enableMemoryManagement: true,
         enableNetworkOptimization: true,
         aggressiveMode: false,
         monitoringInterval: 1000,
         adaptationSensitivity: 0.5,
         ...config
       };

       this.currentProfile = this.generateInitialProfile();
       this.budget = this.calculatePerformanceBudget();

       this.initializeMonitoring();
       this.startAdaptiveLoop();
     }

     private generateInitialProfile(): PerformanceProfile {
       const capabilities = this.mobileDetector.getCapabilities();
       if (!capabilities) {
         throw new Error('Device capabilities not available');
       }

       const adaptiveConfig = this.mobileDetector.getAdaptiveConfig();
       if (!adaptiveConfig) {
         throw new Error('Adaptive config not available');
       }

       // Start with balanced profile based on device capabilities
       if (capabilities.isMobile && capabilities.performance.estimatedGPUTier === 'low') {
         return {
           name: 'battery-saver',
           wasmThreads: Math.min(2, capabilities.performance.hardwareConcurrency),
           updateFrequency: 30,
           renderQuality: 'low',
           maxColumns: 100,
           enableAnimations: false,
           enableEffects: false,
           enableParallelProcessing: false,
           memoryLimit: 128,
           cpuUsageLimit: 0.4,
           thermalThreshold: 0.6
         };
       } else if (capabilities.isTablet || (capabilities.isMobile && capabilities.performance.estimatedGPUTier === 'medium')) {
         return {
           name: 'balanced',
           wasmThreads: Math.min(4, capabilities.performance.hardwareConcurrency),
           updateFrequency: 60,
           renderQuality: 'medium',
           maxColumns: 300,
           enableAnimations: true,
           enableEffects: false,
           enableParallelProcessing: true,
           memoryLimit: 256,
           cpuUsageLimit: 0.6,
           thermalThreshold: 0.7
         };
       } else {
         return {
           name: 'performance',
           wasmThreads: capabilities.performance.hardwareConcurrency,
           updateFrequency: 60,
           renderQuality: 'high',
           maxColumns: 500,
           enableAnimations: true,
           enableEffects: true,
           enableParallelProcessing: true,
           memoryLimit: 512,
           cpuUsageLimit: 0.8,
           thermalThreshold: 0.8
         };
       }
     }

     private calculatePerformanceBudget(): PerformanceBudget {
       const capabilities = this.mobileDetector.getCapabilities();
       if (!capabilities) {
         throw new Error('Device capabilities not available');
       }

       return {
         frameTimeTarget: capabilities.isMobile ? 33.33 : 16.67, // 30fps vs 60fps
         memoryBudget: Math.min(
           (capabilities.performance.deviceMemory || 4) * 1024 * 0.25, // 25% of device memory
           this.currentProfile.memoryLimit
         ),
         cpuBudget: this.currentProfile.cpuUsageLimit,
         batteryDrainRate: capabilities.isMobile ? 5 : 10, // %/hour
         thermalBudget: this.currentProfile.thermalThreshold
       };
     }

     private initializeMonitoring(): void {
       // Initialize Performance Observer for frame timing
       if ('PerformanceObserver' in window) {
         this.performanceObserver = new PerformanceObserver((list) => {
           for (const entry of list.getEntries()) {
             if (entry.entryType === 'measure' || entry.entryType === 'navigation') {
               this.recordFrameTiming(entry.duration);
             }
           }
         });

         try {
           this.performanceObserver.observe({ entryTypes: ['measure', 'navigation'] });
         } catch (error) {
           console.warn('Performance monitoring not fully supported:', error);
         }
       }

       // Initialize memory monitoring
       this.startMemoryMonitoring();

       // Initialize battery monitoring
       this.startBatteryMonitoring();

       // Initialize thermal monitoring
       this.startThermalMonitoring();

       // Initialize network monitoring
       this.startNetworkMonitoring();
     }

     private startMemoryMonitoring(): void {
       if ('memory' in performance) {
         setInterval(() => {
           const memory = (performance as any).memory;
           const usedMB = memory.usedJSHeapSize / (1024 * 1024);
           const totalMB = memory.totalJSHeapSize / (1024 * 1024);
           
           this.metrics.memoryUsage = usedMB / totalMB;

           // Detect memory pressure
           if (usedMB > this.budget.memoryBudget * 0.8) {
             this.memoryPressureCount++;
           } else {
             this.memoryPressureCount = Math.max(0, this.memoryPressureCount - 1);
           }
         }, this.config.monitoringInterval);
       }
     }

     private startBatteryMonitoring(): void {
       if ('getBattery' in navigator) {
         (navigator as any).getBattery().then((battery: any) => {
           const updateBatteryInfo = () => {
             this.metrics.batteryLevel = battery.level;
             this.metrics.batteryCharging = battery.charging;

             // Track battery drain rate
             const now = performance.now();
             this.batteryDrainSamples.push({
               time: now,
               level: battery.level
             });

             // Keep only last hour of samples
             const oneHour = 60 * 60 * 1000;
             this.batteryDrainSamples = this.batteryDrainSamples.filter(
               sample => now - sample.time < oneHour
             );
           };

           updateBatteryInfo();
           battery.addEventListener('chargingchange', updateBatteryInfo);
           battery.addEventListener('levelchange', updateBatteryInfo);
         });
       }
     }

     private startThermalMonitoring(): void {
       // Use CPU usage as thermal proxy if thermal API not available
       let cpuSamples: number[] = [];

       setInterval(() => {
         // Estimate CPU usage from frame timing stability
         const recentFrameTimes = this.frameTimings.slice(-10);
         if (recentFrameTimes.length > 5) {
           const variance = this.calculateVariance(recentFrameTimes);
           const avgFrameTime = recentFrameTimes.reduce((a, b) => a + b, 0) / recentFrameTimes.length;
           
           // High variance + high frame time = high CPU usage
           const estimatedCpuUsage = Math.min(1, (avgFrameTime / 16.67) * (variance / 5));
           this.metrics.cpuUsage = estimatedCpuUsage;

           cpuSamples.push(estimatedCpuUsage);
           if (cpuSamples.length > 10) cpuSamples.shift();

           // Estimate thermal state
           const avgCpuUsage = cpuSamples.reduce((a, b) => a + b, 0) / cpuSamples.length;
           
           if (avgCpuUsage > 0.9) {
             this.metrics.thermalState = 'critical';
             this.thermalPressureCount += 3;
           } else if (avgCpuUsage > 0.7) {
             this.metrics.thermalState = 'serious';
             this.thermalPressureCount += 2;
           } else if (avgCpuUsage > 0.5) {
             this.metrics.thermalState = 'fair';
             this.thermalPressureCount += 1;
           } else {
             this.metrics.thermalState = 'nominal';
             this.thermalPressureCount = Math.max(0, this.thermalPressureCount - 1);
           }
         }
       }, this.config.monitoringInterval);
     }

     private startNetworkMonitoring(): void {
       const connection = (navigator as any).connection || 
                         (navigator as any).mozConnection || 
                         (navigator as any).webkitConnection;

       if (connection) {
         const updateNetworkInfo = () => {
           const effectiveType = connection.effectiveType;
           
           if (effectiveType === '4g' || effectiveType === '3g') {
             this.metrics.networkSpeed = 'fast';
           } else if (effectiveType === '2g') {
             this.metrics.networkSpeed = 'slow';
           } else {
             this.metrics.networkSpeed = 'medium';
           }
         };

         updateNetworkInfo();
         connection.addEventListener('change', updateNetworkInfo);
       }
     }

     private startAdaptiveLoop(): void {
       this.monitoringInterval = window.setInterval(() => {
         this.updateMetrics();
         this.analyzePerformance();
         this.adaptPerformance();
         this.recordMetricsHistory();
       }, this.config.monitoringInterval);
     }

     private updateMetrics(): void {
       // Calculate current frame rate
       const now = performance.now();
       if (this.lastFrameTime > 0) {
         const frameTime = now - this.lastFrameTime;
         this.recordFrameTiming(frameTime);
         this.metrics.frameRate = 1000 / frameTime;
         this.metrics.frameTime = frameTime;
       }
       this.lastFrameTime = now;
     }

     private recordFrameTiming(frameTime: number): void {
       this.frameTimings.push(frameTime);
       if (this.frameTimings.length > 60) {
         this.frameTimings.shift();
       }
     }

     private analyzePerformance(): void {
       if (!this.config.enableAdaptiveThrottling) return;

       // Calculate performance score (0-1, higher is better)
       const frameScore = Math.min(1, this.budget.frameTimeTarget / this.metrics.frameTime);
       const memoryScore = Math.max(0, 1 - (this.metrics.memoryUsage / this.budget.memoryBudget));
       const cpuScore = Math.max(0, 1 - (this.metrics.cpuUsage / this.budget.cpuBudget));
       const thermalScore = this.getThermalScore();
       const batteryScore = this.getBatteryScore();

       const overallScore = (frameScore + memoryScore + cpuScore + thermalScore + batteryScore) / 5;

       // Determine adaptation direction and magnitude
       if (overallScore < 0.7) {
         // Performance issues - throttle down
         this.adaptationState.direction = -1;
         this.adaptationState.magnitude = Math.min(1, (0.7 - overallScore) * 2);
         this.adaptationState.stabilityCounter = 0;
       } else if (overallScore > 0.85 && this.adaptationState.stabilityCounter > 5) {
         // Good performance - can throttle up
         this.adaptationState.direction = 1;
         this.adaptationState.magnitude = Math.min(0.5, (overallScore - 0.85) * 2);
       } else {
         // Stable performance
         this.adaptationState.direction = 0;
         this.adaptationState.stabilityCounter++;
       }

       // Update confidence based on consistency
       const recentScores = this.metricHistory.slice(-5).map(h => this.calculateScore(h.metrics));
       if (recentScores.length >= 3) {
         const variance = this.calculateVariance(recentScores);
         this.adaptationState.confidence = Math.max(0, 1 - variance);
       }
     }

     private getThermalScore(): number {
       switch (this.metrics.thermalState) {
         case 'nominal': return 1.0;
         case 'fair': return 0.8;
         case 'serious': return 0.5;
         case 'critical': return 0.2;
         default: return 1.0;
       }
     }

     private getBatteryScore(): number {
       if (!this.config.enableBatteryOptimization) return 1.0;
       
       if (this.metrics.batteryCharging) return 1.0;
       
       if (this.metrics.batteryLevel < 0.2) return 0.3;
       if (this.metrics.batteryLevel < 0.5) return 0.7;
       return 1.0;
     }

     private calculateScore(metrics: SystemMetrics): number {
       const frameScore = Math.min(1, this.budget.frameTimeTarget / metrics.frameTime);
       const memoryScore = Math.max(0, 1 - (metrics.memoryUsage / this.budget.memoryBudget));
       const cpuScore = Math.max(0, 1 - (metrics.cpuUsage / this.budget.cpuBudget));
       return (frameScore + memoryScore + cpuScore) / 3;
     }

     private adaptPerformance(): void {
       if (this.adaptationState.direction === 0) return;
       if (this.adaptationState.confidence < 0.5) return;

       const profile = { ...this.currentProfile };
       const adaptationStrength = this.adaptationState.magnitude * this.config.adaptationSensitivity;

       if (this.adaptationState.direction < 0) {
         // Throttle down
         this.throttleDown(profile, adaptationStrength);
       } else {
         // Throttle up
         this.throttleUp(profile, adaptationStrength);
       }

       this.applyProfile(profile);
     }

     private throttleDown(profile: PerformanceProfile, strength: number): void {
       // Reduce computational load
       if (strength > 0.7) {
         profile.updateFrequency = Math.max(15, profile.updateFrequency * 0.5);
         profile.renderQuality = 'low';
         profile.enableAnimations = false;
         profile.enableEffects = false;
         profile.maxColumns = Math.floor(profile.maxColumns * 0.5);
         profile.wasmThreads = Math.max(1, Math.floor(profile.wasmThreads * 0.5));
       } else if (strength > 0.4) {
         profile.updateFrequency = Math.max(30, profile.updateFrequency * 0.8);
         profile.renderQuality = profile.renderQuality === 'high' ? 'medium' : 'low';
         profile.enableEffects = false;
         profile.maxColumns = Math.floor(profile.maxColumns * 0.8);
       } else {
         profile.updateFrequency = Math.max(45, profile.updateFrequency * 0.9);
         if (profile.renderQuality === 'high') {
           profile.renderQuality = 'medium';
         }
       }

       // Adjust memory and CPU limits
       profile.memoryLimit = Math.floor(profile.memoryLimit * (1 - strength * 0.3));
       profile.cpuUsageLimit = Math.max(0.2, profile.cpuUsageLimit * (1 - strength * 0.2));
     }

     private throttleUp(profile: PerformanceProfile, strength: number): void {
       // Increase computational load cautiously
       if (strength > 0.3) {
         profile.updateFrequency = Math.min(60, profile.updateFrequency * 1.1);
         if (profile.renderQuality === 'low') {
           profile.renderQuality = 'medium';
         } else if (profile.renderQuality === 'medium') {
           profile.renderQuality = 'high';
         }
         profile.maxColumns = Math.floor(profile.maxColumns * 1.2);
       }

       // Only enable expensive features if performance is very good
       if (strength > 0.5 && this.adaptationState.stabilityCounter > 10) {
         profile.enableAnimations = true;
         if (strength > 0.7) {
           profile.enableEffects = true;
         }
       }
     }

     private applyProfile(newProfile: PerformanceProfile): void {
       const oldProfile = this.currentProfile;
       this.currentProfile = newProfile;

       // Apply WASM settings
       if (newProfile.wasmThreads !== oldProfile.wasmThreads) {
         // Signal WASM to adjust thread count
         this.notifyWasmThreadChange(newProfile.wasmThreads);
       }

       // Apply UI settings
       const uiOptimizations = this.uiAdapter.getOptimizations();
       this.uiAdapter.updateOptimizations({
         ...uiOptimizations,
         enableVirtualScrolling: newProfile.renderQuality === 'low',
         maxRenderItems: newProfile.maxColumns,
         enableReducedMotion: !newProfile.enableAnimations
       });

       // Update performance budget
       this.budget = this.calculatePerformanceBudget();

       if (this.onProfileChange) {
         this.onProfileChange(newProfile);
       }

       if (this.onThrottleChange) {
         const throttleLevel = this.calculateThrottleLevel();
         this.onThrottleChange(throttleLevel);
       }
     }

     private notifyWasmThreadChange(threadCount: number): void {
       // This would integrate with the WASM module to adjust thread usage
       // Implementation depends on WASM module API
       console.log(`Adjusting WASM threads to: ${threadCount}`);
     }

     private calculateThrottleLevel(): number {
       // Return throttle level 0-1 (0 = no throttling, 1 = maximum throttling)
       const baselineProfile = this.generateInitialProfile();
       
       const updateRatio = this.currentProfile.updateFrequency / baselineProfile.updateFrequency;
       const columnRatio = this.currentProfile.maxColumns / baselineProfile.maxColumns;
       const threadRatio = this.currentProfile.wasmThreads / baselineProfile.wasmThreads;
       
       const avgRatio = (updateRatio + columnRatio + threadRatio) / 3;
       return Math.max(0, 1 - avgRatio);
     }

     private calculateVariance(values: number[]): number {
       if (values.length < 2) return 0;
       
       const mean = values.reduce((a, b) => a + b, 0) / values.length;
       const squaredDiffs = values.map(value => Math.pow(value - mean, 2));
       return squaredDiffs.reduce((a, b) => a + b, 0) / values.length;
     }

     private recordMetricsHistory(): void {
       this.metricHistory.push({
         timestamp: performance.now(),
         metrics: { ...this.metrics }
       });

       // Keep only last 5 minutes of history
       const fiveMinutes = 5 * 60 * 1000;
       const cutoff = performance.now() - fiveMinutes;
       this.metricHistory = this.metricHistory.filter(entry => entry.timestamp > cutoff);
     }

     // Public API
     public setProfile(profileName: PerformanceProfile['name']): void {
       let newProfile: PerformanceProfile;

       switch (profileName) {
         case 'battery-saver':
           newProfile = this.createBatterySaverProfile();
           break;
         case 'balanced':
           newProfile = this.createBalancedProfile();
           break;
         case 'performance':
           newProfile = this.createPerformanceProfile();
           break;
         default:
           return;
       }

       this.applyProfile(newProfile);
     }

     private createBatterySaverProfile(): PerformanceProfile {
       return {
         name: 'battery-saver',
         wasmThreads: 1,
         updateFrequency: 15,
         renderQuality: 'low',
         maxColumns: 50,
         enableAnimations: false,
         enableEffects: false,
         enableParallelProcessing: false,
         memoryLimit: 64,
         cpuUsageLimit: 0.3,
         thermalThreshold: 0.5
       };
     }

     private createBalancedProfile(): PerformanceProfile {
       const capabilities = this.mobileDetector.getCapabilities()!;
       return {
         name: 'balanced',
         wasmThreads: Math.min(4, capabilities.performance.hardwareConcurrency),
         updateFrequency: 30,
         renderQuality: 'medium',
         maxColumns: 200,
         enableAnimations: true,
         enableEffects: false,
         enableParallelProcessing: true,
         memoryLimit: 256,
         cpuUsageLimit: 0.6,
         thermalThreshold: 0.7
       };
     }

     private createPerformanceProfile(): PerformanceProfile {
       const capabilities = this.mobileDetector.getCapabilities()!;
       return {
         name: 'performance',
         wasmThreads: capabilities.performance.hardwareConcurrency,
         updateFrequency: 60,
         renderQuality: 'high',
         maxColumns: 500,
         enableAnimations: true,
         enableEffects: true,
         enableParallelProcessing: true,
         memoryLimit: 512,
         cpuUsageLimit: 0.8,
         thermalThreshold: 0.8
       };
     }

     public getCurrentProfile(): PerformanceProfile {
       return { ...this.currentProfile };
     }

     public getMetrics(): SystemMetrics {
       return { ...this.metrics };
     }

     public getMetricsHistory(): Array<{ timestamp: number; metrics: SystemMetrics }> {
       return [...this.metricHistory];
     }

     public updateConfig(newConfig: Partial<ThrottlingConfig>): void {
       this.config = { ...this.config, ...newConfig };
     }

     public forceAdaptation(): void {
       this.analyzePerformance();
       this.adaptPerformance();
     }

     public isThrottling(): boolean {
       return this.calculateThrottleLevel() > 0.1;
     }

     public getBatteryDrainRate(): number {
       if (this.batteryDrainSamples.length < 2) return 0;
       
       const recent = this.batteryDrainSamples[this.batteryDrainSamples.length - 1];
       const older = this.batteryDrainSamples[0];
       const timeDiff = (recent.time - older.time) / (1000 * 60 * 60); // hours
       const levelDiff = older.level - recent.level;
       
       return timeDiff > 0 ? (levelDiff / timeDiff) * 100 : 0;
     }

     public dispose(): void {
       if (this.monitoringInterval) {
         clearInterval(this.monitoringInterval);
       }

       if (this.adaptationTimer) {
         clearInterval(this.adaptationTimer);
       }

       if (this.performanceObserver) {
         this.performanceObserver.disconnect();
       }

       this.metricHistory = [];
       this.frameTimings = [];
       this.batteryDrainSamples = [];
     }
   }
   ```

## Expected Outputs
- Intelligent performance adaptation based on device capabilities and current conditions
- Real-time battery optimization extending device usage time by 20-40%
- Thermal management preventing device overheating during intensive operations
- Memory pressure detection and automatic garbage collection triggering
- Network-aware computation scheduling for data-sensitive operations

## Validation
1. Performance profiles adapt smoothly without causing visual stuttering
2. Battery optimization extends usage time measurably on mobile devices
3. Thermal throttling prevents device temperature from exceeding safe limits
4. Memory management keeps heap usage within 80% of available memory
5. Frame rate maintains target thresholds (30fps mobile, 60fps desktop) under load

## Next Steps
- Mobile memory management system (micro-phase 9.35)
- Integration with complete Phase 9 WASM system