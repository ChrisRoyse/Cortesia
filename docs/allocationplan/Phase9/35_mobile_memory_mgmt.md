# Micro-Phase 9.35: Mobile Memory Management and Garbage Collection

## Objective
Implement sophisticated memory management system optimized for mobile devices with limited RAM, featuring predictive garbage collection, memory pool management, and aggressive optimization strategies.

## Prerequisites
- Completed micro-phase 9.34 (Performance Throttling)
- PerformanceThrottler class available
- Understanding of JavaScript memory management and mobile constraints
- Access to WASM memory management APIs

## Task Description
Create comprehensive memory management system that proactively manages heap allocation, implements intelligent garbage collection scheduling, and maintains optimal memory usage patterns for sustained mobile performance.

## Specific Actions

1. **Create MobileMemoryManager class with intelligent allocation strategies**:
   ```typescript
   // src/mobile/MobileMemoryManager.ts
   import { PerformanceThrottler, SystemMetrics } from './PerformanceThrottler';
   import { MobileDetector } from './MobileDetector';

   export interface MemoryConfig {
     enableAggressiveGC: boolean;
     enableMemoryPools: boolean;
     enablePreemptiveCleanup: boolean;
     enableWasmMemoryOptimization: boolean;
     gcTriggerThreshold: number; // 0-1
     memoryPressureThreshold: number; // 0-1
     poolPreallocationSize: number; // MB
     maxHeapSize: number; // MB
     emergencyCleanupThreshold: number; // 0-1
   }

   export interface MemoryPool {
     name: string;
     size: number;
     allocatedBlocks: Set<ArrayBuffer>;
     freeBlocks: ArrayBuffer[];
     blockSize: number;
     maxBlocks: number;
     hitRate: number;
     totalAllocations: number;
   }

   export interface MemoryStats {
     jsHeapSizeLimit: number;
     totalJSHeapSize: number;
     usedJSHeapSize: number;
     wasmMemorySize: number;
     memoryPressure: number; // 0-1
     gcPressure: number; // 0-1
     allocationRate: number; // MB/s
     gcFrequency: number; // GC/s
     memoryEfficiency: number; // 0-1
     poolUtilization: number; // 0-1
   }

   export interface AllocationStrategy {
     preferPools: boolean;
     aggressiveReuse: boolean;
     predictivePreallocation: boolean;
     emergencyMode: boolean;
     compressionEnabled: boolean;
   }

   export interface MemoryAlert {
     type: 'warning' | 'critical' | 'emergency';
     message: string;
     threshold: number;
     currentUsage: number;
     recommendation: string;
     timestamp: number;
   }

   export class MobileMemoryManager {
     private performanceThrottler: PerformanceThrottler;
     private mobileDetector: MobileDetector;
     private config: MemoryConfig;
     private strategy: AllocationStrategy;

     private memoryPools: Map<string, MemoryPool> = new Map();
     private allocationHistory: Array<{
       timestamp: number;
       size: number;
       type: 'allocation' | 'deallocation' | 'gc';
     }> = [];

     private stats: MemoryStats = {
       jsHeapSizeLimit: 0,
       totalJSHeapSize: 0,
       usedJSHeapSize: 0,
       wasmMemorySize: 0,
       memoryPressure: 0,
       gcPressure: 0,
       allocationRate: 0,
       gcFrequency: 0,
       memoryEfficiency: 0,
       poolUtilization: 0
     };

     private monitoringInterval: number | null = null;
     private gcScheduler: number | null = null;
     private cleanupTimer: number | null = null;
     
     // Memory pressure tracking
     private pressureHistory: number[] = [];
     private lastGCTime = 0;
     private gcCount = 0;
     private allocationCount = 0;
     
     // Emergency cleanup callbacks
     private cleanupCallbacks: Array<() => Promise<number>> = [];
     private alertCallbacks: Array<(alert: MemoryAlert) => void> = [];
     
     // WASM memory integration
     private wasmModule: any = null;
     private wasmMemoryView: Uint8Array | null = null;

     constructor(
       performanceThrottler: PerformanceThrottler,
       mobileDetector: MobileDetector,
       config?: Partial<MemoryConfig>
     ) {
       this.performanceThrottler = performanceThrottler;
       this.mobileDetector = mobileDetector;
       
       this.config = {
         enableAggressiveGC: true,
         enableMemoryPools: true,
         enablePreemptiveCleanup: true,
         enableWasmMemoryOptimization: true,
         gcTriggerThreshold: 0.7,
         memoryPressureThreshold: 0.8,
         poolPreallocationSize: 32,
         maxHeapSize: this.calculateMaxHeapSize(),
         emergencyCleanupThreshold: 0.9,
         ...config
       };

       this.strategy = this.generateAllocationStrategy();
       
       this.initializeMemoryPools();
       this.startMemoryMonitoring();
       this.setupMemoryPressureHandling();
     }

     private calculateMaxHeapSize(): number {
       const capabilities = this.mobileDetector.getCapabilities();
       if (!capabilities) return 256;

       const deviceMemory = capabilities.performance.deviceMemory || 4;
       
       if (capabilities.isMobile) {
         return Math.min(256, deviceMemory * 1024 * 0.15); // 15% on mobile
       } else if (capabilities.isTablet) {
         return Math.min(512, deviceMemory * 1024 * 0.25); // 25% on tablet
       } else {
         return Math.min(1024, deviceMemory * 1024 * 0.4); // 40% on desktop
       }
     }

     private generateAllocationStrategy(): AllocationStrategy {
       const capabilities = this.mobileDetector.getCapabilities();
       if (!capabilities) {
         return {
           preferPools: true,
           aggressiveReuse: true,
           predictivePreallocation: false,
           emergencyMode: false,
           compressionEnabled: false
         };
       }

       const isLowMemory = (capabilities.performance.deviceMemory || 4) < 4;
       
       return {
         preferPools: capabilities.isMobile || isLowMemory,
         aggressiveReuse: capabilities.isMobile,
         predictivePreallocation: !isLowMemory,
         emergencyMode: false,
         compressionEnabled: isLowMemory
       };
     }

     private initializeMemoryPools(): void {
       if (!this.config.enableMemoryPools) return;

       // Create pools for common buffer sizes
       const poolConfigs = [
         { name: 'small', blockSize: 1024, maxBlocks: 100 },      // 1KB blocks
         { name: 'medium', blockSize: 8192, maxBlocks: 50 },     // 8KB blocks
         { name: 'large', blockSize: 65536, maxBlocks: 20 },     // 64KB blocks
         { name: 'xlarge', blockSize: 262144, maxBlocks: 10 },   // 256KB blocks
       ];

       for (const poolConfig of poolConfigs) {
         this.createMemoryPool(poolConfig.name, poolConfig.blockSize, poolConfig.maxBlocks);
       }
     }

     private createMemoryPool(name: string, blockSize: number, maxBlocks: number): void {
       const pool: MemoryPool = {
         name,
         size: blockSize * maxBlocks,
         allocatedBlocks: new Set(),
         freeBlocks: [],
         blockSize,
         maxBlocks,
         hitRate: 0,
         totalAllocations: 0
       };

       // Pre-allocate some blocks if strategy allows
       if (this.strategy.predictivePreallocation) {
         const preallocationCount = Math.min(maxBlocks / 4, 5);
         for (let i = 0; i < preallocationCount; i++) {
           const buffer = new ArrayBuffer(blockSize);
           pool.freeBlocks.push(buffer);
         }
       }

       this.memoryPools.set(name, pool);
     }

     private startMemoryMonitoring(): void {
       this.monitoringInterval = window.setInterval(() => {
         this.updateMemoryStats();
         this.analyzeMemoryPressure();
         this.scheduleGarbageCollection();
         this.optimizeMemoryPools();
       }, 1000);

       // High-frequency monitoring for critical situations
       setInterval(() => {
         if (this.strategy.emergencyMode) {
           this.checkEmergencyCleanup();
         }
       }, 100);
     }

     private updateMemoryStats(): void {
       // JavaScript heap stats
       if ('memory' in performance) {
         const memory = (performance as any).memory;
         this.stats.jsHeapSizeLimit = memory.jsHeapSizeLimit;
         this.stats.totalJSHeapSize = memory.totalJSHeapSize;
         this.stats.usedJSHeapSize = memory.usedJSHeapSize;
       }

       // WASM memory stats
       if (this.wasmModule && this.wasmModule.memory) {
         this.stats.wasmMemorySize = this.wasmModule.memory.buffer.byteLength;
       }

       // Calculate derived metrics
       this.stats.memoryPressure = this.stats.usedJSHeapSize / this.stats.jsHeapSizeLimit;
       this.stats.memoryEfficiency = this.calculateMemoryEfficiency();
       this.stats.poolUtilization = this.calculatePoolUtilization();
       
       // Calculate allocation rate
       const recentAllocations = this.allocationHistory.filter(
         entry => performance.now() - entry.timestamp < 1000
       );
       
       const totalSize = recentAllocations
         .filter(entry => entry.type === 'allocation')
         .reduce((sum, entry) => sum + entry.size, 0);
       
       this.stats.allocationRate = totalSize / (1024 * 1024); // MB/s

       // Calculate GC frequency
       const recentGCs = recentAllocations.filter(entry => entry.type === 'gc');
       this.stats.gcFrequency = recentGCs.length;
     }

     private calculateMemoryEfficiency(): number {
       if (this.stats.totalJSHeapSize === 0) return 1;
       return this.stats.usedJSHeapSize / this.stats.totalJSHeapSize;
     }

     private calculatePoolUtilization(): number {
       let totalBlocks = 0;
       let allocatedBlocks = 0;

       for (const pool of this.memoryPools.values()) {
         totalBlocks += pool.maxBlocks;
         allocatedBlocks += pool.allocatedBlocks.size;
       }

       return totalBlocks > 0 ? allocatedBlocks / totalBlocks : 0;
     }

     private analyzeMemoryPressure(): void {
       this.pressureHistory.push(this.stats.memoryPressure);
       if (this.pressureHistory.length > 10) {
         this.pressureHistory.shift();
       }

       const avgPressure = this.pressureHistory.reduce((a, b) => a + b, 0) / this.pressureHistory.length;
       
       // Update strategy based on pressure
       if (avgPressure > this.config.emergencyCleanupThreshold) {
         this.enterEmergencyMode();
       } else if (avgPressure > this.config.memoryPressureThreshold) {
         this.enableAggressiveMode();
       } else if (avgPressure < 0.5) {
         this.enterNormalMode();
       }

       // Generate alerts
       if (avgPressure > this.config.memoryPressureThreshold) {
         this.generateMemoryAlert(avgPressure);
       }
     }

     private enterEmergencyMode(): void {
       if (this.strategy.emergencyMode) return;

       this.strategy.emergencyMode = true;
       this.strategy.aggressiveReuse = true;
       this.strategy.preferPools = true;
       this.strategy.compressionEnabled = true;

       this.triggerEmergencyCleanup();
       
       console.warn('Entering emergency memory mode');
     }

     private enableAggressiveMode(): void {
       this.strategy.aggressiveReuse = true;
       this.strategy.preferPools = true;
       
       // Trigger preventive GC
       if (this.config.enableAggressiveGC) {
         this.scheduleImmediateGC();
       }
     }

     private enterNormalMode(): void {
       this.strategy.emergencyMode = false;
       this.strategy.aggressiveReuse = this.mobileDetector.isMobile();
       this.strategy.compressionEnabled = false;
     }

     private scheduleGarbageCollection(): void {
       if (!this.config.enableAggressiveGC) return;

       const now = performance.now();
       const timeSinceLastGC = now - this.lastGCTime;
       
       // Schedule GC based on pressure and time
       const shouldGC = (
         this.stats.memoryPressure > this.config.gcTriggerThreshold ||
         (timeSinceLastGC > 5000 && this.stats.allocationRate > 1) || // 5s and high allocation
         this.strategy.emergencyMode
       );

       if (shouldGC && timeSinceLastGC > 1000) { // Min 1s between GCs
         this.scheduleImmediateGC();
       }
     }

     private scheduleImmediateGC(): void {
       if (this.gcScheduler) return;

       this.gcScheduler = window.setTimeout(() => {
         this.performGarbageCollection();
         this.gcScheduler = null;
       }, 16); // Next frame
     }

     private performGarbageCollection(): void {
       const beforeMemory = this.stats.usedJSHeapSize;
       
       try {
         // Force garbage collection if available
         if ('gc' in window) {
           (window as any).gc();
         } else {
           // Fallback: create memory pressure to trigger GC
           this.createMemoryPressure();
         }

         this.lastGCTime = performance.now();
         this.gcCount++;

         this.allocationHistory.push({
           timestamp: performance.now(),
           size: beforeMemory - this.stats.usedJSHeapSize,
           type: 'gc'
         });

       } catch (error) {
         console.warn('GC failed:', error);
       }
     }

     private createMemoryPressure(): void {
       // Create temporary large arrays to trigger GC
       const tempArrays: any[] = [];
       try {
         for (let i = 0; i < 100; i++) {
           tempArrays.push(new Array(10000).fill(0));
         }
       } catch (error) {
         // Expected - memory pressure created
       } finally {
         // Clear arrays to allow GC
         tempArrays.length = 0;
       }
     }

     private optimizeMemoryPools(): void {
       for (const pool of this.memoryPools.values()) {
         this.optimizePool(pool);
       }
     }

     private optimizePool(pool: MemoryPool): void {
       // Calculate hit rate
       pool.hitRate = pool.totalAllocations > 0 ? 
         (pool.totalAllocations - pool.freeBlocks.length) / pool.totalAllocations : 0;

       // Shrink pool if underutilized
       if (pool.hitRate < 0.3 && pool.freeBlocks.length > 5) {
         const removeCount = Math.floor(pool.freeBlocks.length / 2);
         pool.freeBlocks.splice(0, removeCount);
       }

       // Grow pool if overutilized and strategy allows
       if (pool.hitRate > 0.9 && pool.freeBlocks.length === 0 && 
           this.strategy.predictivePreallocation && !this.strategy.emergencyMode) {
         const addCount = Math.min(5, pool.maxBlocks - pool.allocatedBlocks.size);
         for (let i = 0; i < addCount; i++) {
           const buffer = new ArrayBuffer(pool.blockSize);
           pool.freeBlocks.push(buffer);
         }
       }
     }

     private setupMemoryPressureHandling(): void {
       // Listen for memory pressure events if available
       if ('onmemorywarning' in window) {
         window.addEventListener('memorywarning', () => {
           this.handleMemoryWarning();
         });
       }

       // Monitor for Out of Memory situations
       window.addEventListener('error', (event) => {
         if (event.message && event.message.includes('out of memory')) {
           this.handleOutOfMemory();
         }
       });
     }

     private handleMemoryWarning(): void {
       this.enterEmergencyMode();
       this.triggerEmergencyCleanup();
     }

     private handleOutOfMemory(): void {
       this.enterEmergencyMode();
       this.triggerEmergencyCleanup();
       
       // Notify performance throttler to reduce load
       this.performanceThrottler.setProfile('battery-saver');
     }

     private triggerEmergencyCleanup(): void {
       Promise.all(this.cleanupCallbacks.map(callback => callback()))
         .then(results => {
           const totalFreed = results.reduce((sum, freed) => sum + freed, 0);
           console.log(`Emergency cleanup freed ${totalFreed} MB`);
         })
         .catch(error => {
           console.error('Emergency cleanup failed:', error);
         });
     }

     private checkEmergencyCleanup(): void {
       if (this.stats.memoryPressure > this.config.emergencyCleanupThreshold) {
         this.triggerEmergencyCleanup();
       }
     }

     private generateMemoryAlert(pressure: number): void {
       let alertType: MemoryAlert['type'];
       let recommendation: string;

       if (pressure > 0.95) {
         alertType = 'emergency';
         recommendation = 'Immediate cleanup required. Consider reducing data size.';
       } else if (pressure > 0.85) {
         alertType = 'critical';
         recommendation = 'Memory pressure high. Reduce active operations.';
       } else {
         alertType = 'warning';
         recommendation = 'Memory usage elevated. Monitor closely.';
       }

       const alert: MemoryAlert = {
         type: alertType,
         message: `Memory pressure at ${(pressure * 100).toFixed(1)}%`,
         threshold: this.config.memoryPressureThreshold,
         currentUsage: pressure,
         recommendation,
         timestamp: performance.now()
       };

       this.alertCallbacks.forEach(callback => callback(alert));
     }

     // Memory allocation API
     public allocateBuffer(size: number): ArrayBuffer | null {
       this.allocationCount++;
       
       this.allocationHistory.push({
         timestamp: performance.now(),
         size,
         type: 'allocation'
       });

       // Try pool allocation first if strategy prefers it
       if (this.strategy.preferPools) {
         const poolBuffer = this.allocateFromPool(size);
         if (poolBuffer) return poolBuffer;
       }

       // Fallback to direct allocation
       try {
         return new ArrayBuffer(size);
       } catch (error) {
         // Allocation failed - try emergency cleanup
         this.triggerEmergencyCleanup();
         
         // Retry after cleanup
         try {
           return new ArrayBuffer(size);
         } catch (retryError) {
           console.error('Memory allocation failed:', retryError);
           return null;
         }
       }
     }

     private allocateFromPool(size: number): ArrayBuffer | null {
       // Find appropriate pool
       let selectedPool: MemoryPool | null = null;
       
       for (const pool of this.memoryPools.values()) {
         if (pool.blockSize >= size && pool.freeBlocks.length > 0) {
           if (!selectedPool || pool.blockSize < selectedPool.blockSize) {
             selectedPool = pool;
           }
         }
       }

       if (!selectedPool) return null;

       // Allocate from pool
       const buffer = selectedPool.freeBlocks.pop()!;
       selectedPool.allocatedBlocks.add(buffer);
       selectedPool.totalAllocations++;

       return buffer;
     }

     public deallocateBuffer(buffer: ArrayBuffer): void {
       this.allocationHistory.push({
         timestamp: performance.now(),
         size: buffer.byteLength,
         type: 'deallocation'
       });

       // Return to pool if it came from one
       for (const pool of this.memoryPools.values()) {
         if (pool.allocatedBlocks.has(buffer)) {
           pool.allocatedBlocks.delete(buffer);
           
           if (pool.freeBlocks.length < pool.maxBlocks) {
             pool.freeBlocks.push(buffer);
           }
           return;
         }
       }

       // Buffer wasn't from a pool - nothing more to do (GC will handle it)
     }

     // WASM integration
     public integrateWasmModule(wasmModule: any): void {
       this.wasmModule = wasmModule;
       
       if (wasmModule.memory) {
         this.wasmMemoryView = new Uint8Array(wasmModule.memory.buffer);
       }

       // Setup WASM memory monitoring
       if (this.config.enableWasmMemoryOptimization) {
         this.optimizeWasmMemory();
       }
     }

     private optimizeWasmMemory(): void {
       if (!this.wasmModule || !this.wasmModule.memory) return;

       // Monitor WASM memory growth
       const checkWasmMemory = () => {
         const currentSize = this.wasmModule.memory.buffer.byteLength;
         
         if (currentSize > this.stats.wasmMemorySize) {
           this.stats.wasmMemorySize = currentSize;
           
           // Update memory view
           this.wasmMemoryView = new Uint8Array(this.wasmModule.memory.buffer);
           
           // Check if we need to trigger cleanup
           const totalMemory = this.stats.usedJSHeapSize + currentSize;
           const memoryPressure = totalMemory / (this.config.maxHeapSize * 1024 * 1024);
           
           if (memoryPressure > this.config.memoryPressureThreshold) {
             this.enterEmergencyMode();
           }
         }
       };

       setInterval(checkWasmMemory, 1000);
     }

     // Public API
     public registerCleanupCallback(callback: () => Promise<number>): void {
       this.cleanupCallbacks.push(callback);
     }

     public registerAlertCallback(callback: (alert: MemoryAlert) => void): void {
       this.alertCallbacks.push(callback);
     }

     public getMemoryStats(): MemoryStats {
       return { ...this.stats };
     }

     public getAllocationHistory(): Array<{ timestamp: number; size: number; type: string }> {
       return [...this.allocationHistory];
     }

     public getPoolStats(): Map<string, MemoryPool> {
       return new Map(this.memoryPools);
     }

     public forceGarbageCollection(): void {
       this.performGarbageCollection();
     }

     public clearAllPools(): void {
       for (const pool of this.memoryPools.values()) {
         pool.freeBlocks = [];
         pool.allocatedBlocks.clear();
       }
     }

     public updateConfig(newConfig: Partial<MemoryConfig>): void {
       this.config = { ...this.config, ...newConfig };
       this.strategy = this.generateAllocationStrategy();
     }

     public getRecommendations(): string[] {
       const recommendations: string[] = [];

       if (this.stats.memoryPressure > 0.8) {
         recommendations.push('Consider reducing the number of active columns');
         recommendations.push('Disable visual effects and animations');
         recommendations.push('Enable aggressive garbage collection');
       }

       if (this.stats.poolUtilization < 0.3) {
         recommendations.push('Memory pools are underutilized - consider reducing pool sizes');
       }

       if (this.stats.allocationRate > 10) {
         recommendations.push('High allocation rate detected - review memory usage patterns');
       }

       if (this.stats.gcFrequency > 2) {
         recommendations.push('Frequent garbage collection - reduce allocation rate');
       }

       return recommendations;
     }

     public dispose(): void {
       if (this.monitoringInterval) {
         clearInterval(this.monitoringInterval);
       }

       if (this.gcScheduler) {
         clearTimeout(this.gcScheduler);
       }

       if (this.cleanupTimer) {
         clearTimeout(this.cleanupTimer);
       }

       this.clearAllPools();
       this.allocationHistory = [];
       this.pressureHistory = [];
       this.cleanupCallbacks = [];
       this.alertCallbacks = [];
     }
   }
   ```

## Expected Outputs
- Proactive memory pressure detection and automatic mitigation strategies
- Intelligent garbage collection scheduling minimizing performance impact
- Memory pool management reducing allocation overhead by 30-50%
- WASM memory integration with cross-language optimization
- Emergency cleanup system preventing out-of-memory crashes

## Validation
1. Memory usage stays within 80% of device limits during sustained operation
2. Garbage collection frequency remains under 2 GC/second during normal use
3. Memory pools achieve >70% hit rate for common allocations
4. Emergency cleanup prevents crashes when approaching memory limits
5. Memory efficiency (used/allocated ratio) maintains >85% during operation

## Next Steps
- Final integration testing with complete Phase 9 WASM system
- Performance benchmarking across mobile device spectrum
- Production deployment and monitoring setup