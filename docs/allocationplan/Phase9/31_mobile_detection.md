# Micro-Phase 9.31: Mobile Device Detection and Capabilities

## Objective
Implement comprehensive device detection system that identifies mobile devices, assesses their capabilities, and optimizes system configuration accordingly.

## Prerequisites
- Completed micro-phase 9.30 (Real-time Updates)
- RealtimeUpdateManager class available
- Understanding of mobile device APIs and feature detection

## Task Description
Create a robust mobile detection system that identifies device types, screen characteristics, performance capabilities, and hardware features to enable adaptive optimization strategies throughout the application.

## Specific Actions

1. **Create MobileDetector class with device identification**:
   ```typescript
   // src/mobile/MobileDetector.ts
   export interface DeviceCapabilities {
     isMobile: boolean;
     isTablet: boolean;
     isDesktop: boolean;
     screenSize: {
       width: number;
       height: number;
       diagonal: number;
       pixelRatio: number;
     };
     performance: {
       deviceMemory?: number;
       hardwareConcurrency: number;
       maxTouchPoints: number;
       estimatedGPUTier: 'high' | 'medium' | 'low';
     };
     features: {
       touchSupport: boolean;
       orientationSupport: boolean;
       batteryAPI: boolean;
       vibrationAPI: boolean;
       deviceMotion: boolean;
       deviceOrientation: boolean;
       visualViewport: boolean;
       webGL: boolean;
       webGL2: boolean;
       offscreenCanvas: boolean;
       sharedArrayBuffer: boolean;
       webAssembly: boolean;
       simd: boolean;
     };
     network: {
       connection?: any;
       effectiveType?: string;
       downlink?: number;
       saveData?: boolean;
     };
     browser: {
       userAgent: string;
       vendor: string;
       name: string;
       version: string;
       engine: string;
     };
     operatingSystem: {
       name: string;
       version?: string;
       architecture?: string;
     };
   }

   export interface AdaptiveConfig {
     renderingMode: 'high' | 'medium' | 'low';
     updateInterval: number;
     batchSize: number;
     maxColumns: number;
     enableAnimations: boolean;
     enableEffects: boolean;
     memoryThreshold: number;
     cpuThreshold: number;
   }

   export class MobileDetector {
     private capabilities: DeviceCapabilities | null = null;
     private adaptiveConfig: AdaptiveConfig | null = null;
     private orientationChangeListeners: (() => void)[] = [];
     private resizeObserver: ResizeObserver | null = null;
     
     // Performance monitoring
     private performanceMetrics = {
       frameTime: 0,
       memoryUsage: 0,
       cpuUsage: 0,
       batteryLevel: 1.0,
       isCharging: true
     };

     constructor() {
       this.detectCapabilities();
       this.setupEventListeners();
     }

     private detectCapabilities(): void {
       this.capabilities = {
         isMobile: this.detectMobile(),
         isTablet: this.detectTablet(),
         isDesktop: this.detectDesktop(),
         screenSize: this.getScreenInfo(),
         performance: this.getPerformanceInfo(),
         features: this.getFeatureSupport(),
         network: this.getNetworkInfo(),
         browser: this.getBrowserInfo(),
         operatingSystem: this.getOSInfo()
       };

       this.adaptiveConfig = this.generateAdaptiveConfig();
     }

     private detectMobile(): boolean {
       const userAgent = navigator.userAgent.toLowerCase();
       const mobileRegex = /android|webos|iphone|ipad|ipod|blackberry|iemobile|opera mini/i;
       
       return (
         mobileRegex.test(userAgent) ||
         ('ontouchstart' in window) ||
         (navigator.maxTouchPoints > 0) ||
         (window.innerWidth <= 768)
       );
     }

     private detectTablet(): boolean {
       const userAgent = navigator.userAgent.toLowerCase();
       const tabletRegex = /ipad|tablet|android(?!.*mobile)/i;
       
       return (
         tabletRegex.test(userAgent) ||
         (navigator.maxTouchPoints > 0 && window.innerWidth >= 768 && window.innerWidth <= 1024)
       );
     }

     private detectDesktop(): boolean {
       return !this.detectMobile() && !this.detectTablet();
     }

     private getScreenInfo() {
       const screen = window.screen;
       const viewport = window.visualViewport || {
         width: window.innerWidth,
         height: window.innerHeight,
         scale: 1
       };

       return {
         width: viewport.width,
         height: viewport.height,
         diagonal: Math.sqrt(Math.pow(screen.width, 2) + Math.pow(screen.height, 2)),
         pixelRatio: window.devicePixelRatio || 1
       };
     }

     private getPerformanceInfo() {
       const performance = navigator as any;
       
       return {
         deviceMemory: performance.deviceMemory,
         hardwareConcurrency: navigator.hardwareConcurrency || 4,
         maxTouchPoints: navigator.maxTouchPoints || 0,
         estimatedGPUTier: this.estimateGPUPerformance()
       };
     }

     private estimateGPUPerformance(): 'high' | 'medium' | 'low' {
       const canvas = document.createElement('canvas');
       const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
       
       if (!gl) return 'low';
       
       const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
       if (debugInfo) {
         const renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL).toLowerCase();
         
         if (renderer.includes('intel') || renderer.includes('integrated')) {
           return 'medium';
         } else if (renderer.includes('nvidia') || renderer.includes('amd') || renderer.includes('radeon')) {
           return 'high';
         }
       }
       
       // Fallback performance test
       const extension = gl.getExtension('WEBGL_lose_context');
       if (extension) extension.loseContext();
       
       return 'medium';
     }

     private getFeatureSupport() {
       return {
         touchSupport: 'ontouchstart' in window || navigator.maxTouchPoints > 0,
         orientationSupport: 'orientation' in window || 'onorientationchange' in window,
         batteryAPI: 'getBattery' in navigator,
         vibrationAPI: 'vibrate' in navigator,
         deviceMotion: 'DeviceMotionEvent' in window,
         deviceOrientation: 'DeviceOrientationEvent' in window,
         visualViewport: 'visualViewport' in window,
         webGL: this.hasWebGL(),
         webGL2: this.hasWebGL2(),
         offscreenCanvas: 'OffscreenCanvas' in window,
         sharedArrayBuffer: 'SharedArrayBuffer' in window,
         webAssembly: 'WebAssembly' in window,
         simd: this.hasSIMDSupport()
       };
     }

     private hasWebGL(): boolean {
       try {
         const canvas = document.createElement('canvas');
         return !!(canvas.getContext('webgl') || canvas.getContext('experimental-webgl'));
       } catch {
         return false;
       }
     }

     private hasWebGL2(): boolean {
       try {
         const canvas = document.createElement('canvas');
         return !!canvas.getContext('webgl2');
       } catch {
         return false;
       }
     }

     private hasSIMDSupport(): boolean {
       try {
         return 'WebAssembly' in window && 'instantiate' in WebAssembly;
       } catch {
         return false;
       }
     }

     private getNetworkInfo() {
       const connection = (navigator as any).connection || 
                         (navigator as any).mozConnection || 
                         (navigator as any).webkitConnection;

       return {
         connection,
         effectiveType: connection?.effectiveType,
         downlink: connection?.downlink,
         saveData: connection?.saveData
       };
     }

     private getBrowserInfo() {
       const userAgent = navigator.userAgent;
       const vendor = navigator.vendor;

       return {
         userAgent,
         vendor,
         name: this.getBrowserName(userAgent),
         version: this.getBrowserVersion(userAgent),
         engine: this.getBrowserEngine(userAgent)
       };
     }

     private getBrowserName(userAgent: string): string {
       if (userAgent.includes('Firefox')) return 'Firefox';
       if (userAgent.includes('Chrome')) return 'Chrome';
       if (userAgent.includes('Safari') && !userAgent.includes('Chrome')) return 'Safari';
       if (userAgent.includes('Edge')) return 'Edge';
       return 'Unknown';
     }

     private getBrowserVersion(userAgent: string): string {
       const match = userAgent.match(/(firefox|chrome|safari|edge)\/(\d+)/i);
       return match ? match[2] : 'Unknown';
     }

     private getBrowserEngine(userAgent: string): string {
       if (userAgent.includes('Webkit')) return 'Webkit';
       if (userAgent.includes('Gecko')) return 'Gecko';
       if (userAgent.includes('Blink')) return 'Blink';
       return 'Unknown';
     }

     private getOSInfo() {
       const userAgent = navigator.userAgent;
       
       if (userAgent.includes('Windows')) {
         return { name: 'Windows', version: this.getWindowsVersion(userAgent) };
       } else if (userAgent.includes('Mac')) {
         return { name: 'macOS', version: this.getMacVersion(userAgent) };
       } else if (userAgent.includes('Android')) {
         return { name: 'Android', version: this.getAndroidVersion(userAgent) };
       } else if (userAgent.includes('iPhone') || userAgent.includes('iPad')) {
         return { name: 'iOS', version: this.getiOSVersion(userAgent) };
       } else if (userAgent.includes('Linux')) {
         return { name: 'Linux' };
       }
       
       return { name: 'Unknown' };
     }

     private getWindowsVersion(userAgent: string): string {
       if (userAgent.includes('Windows NT 10.0')) return '10';
       if (userAgent.includes('Windows NT 6.3')) return '8.1';
       if (userAgent.includes('Windows NT 6.2')) return '8';
       if (userAgent.includes('Windows NT 6.1')) return '7';
       return 'Unknown';
     }

     private getMacVersion(userAgent: string): string {
       const match = userAgent.match(/Mac OS X (\d+_\d+_\d+)/);
       return match ? match[1].replace(/_/g, '.') : 'Unknown';
     }

     private getAndroidVersion(userAgent: string): string {
       const match = userAgent.match(/Android (\d+\.?\d*)/);
       return match ? match[1] : 'Unknown';
     }

     private getiOSVersion(userAgent: string): string {
       const match = userAgent.match(/OS (\d+_\d+_?\d*)/);
       return match ? match[1].replace(/_/g, '.') : 'Unknown';
     }
   }
   ```

2. **Implement adaptive configuration generation**:
   ```typescript
   private generateAdaptiveConfig(): AdaptiveConfig {
     if (!this.capabilities) {
       throw new Error('Capabilities not detected');
     }

     const caps = this.capabilities;
     let config: AdaptiveConfig;

     // Base configuration based on device type
     if (caps.isDesktop) {
       config = {
         renderingMode: 'high',
         updateInterval: 16, // 60fps
         batchSize: 100,
         maxColumns: 2000,
         enableAnimations: true,
         enableEffects: true,
         memoryThreshold: 0.8,
         cpuThreshold: 0.7
       };
     } else if (caps.isTablet) {
       config = {
         renderingMode: 'medium',
         updateInterval: 20, // 50fps
         batchSize: 75,
         maxColumns: 1000,
         enableAnimations: true,
         enableEffects: true,
         memoryThreshold: 0.7,
         cpuThreshold: 0.6
       };
     } else {
       config = {
         renderingMode: 'low',
         updateInterval: 33, // 30fps
         batchSize: 50,
         maxColumns: 500,
         enableAnimations: false,
         enableEffects: false,
         memoryThreshold: 0.6,
         cpuThreshold: 0.5
       };
     }

     // Adjust based on performance capabilities
     if (caps.performance.estimatedGPUTier === 'low') {
       config.renderingMode = 'low';
       config.enableEffects = false;
       config.maxColumns = Math.min(config.maxColumns, 300);
     } else if (caps.performance.estimatedGPUTier === 'high') {
       config.renderingMode = 'high';
       config.enableEffects = true;
     }

     // Adjust based on memory
     if (caps.performance.deviceMemory && caps.performance.deviceMemory < 4) {
       config.batchSize = Math.min(config.batchSize, 30);
       config.maxColumns = Math.min(config.maxColumns, 200);
       config.memoryThreshold = 0.5;
     }

     // Adjust based on CPU
     if (caps.performance.hardwareConcurrency <= 2) {
       config.updateInterval = Math.max(config.updateInterval, 50);
       config.cpuThreshold = 0.4;
     }

     // Network considerations
     if (caps.network.saveData || caps.network.effectiveType === 'slow-2g' || caps.network.effectiveType === '2g') {
       config.enableAnimations = false;
       config.enableEffects = false;
       config.updateInterval = Math.max(config.updateInterval, 100);
     }

     return config;
   }

   private setupEventListeners(): void {
     // Orientation change
     window.addEventListener('orientationchange', () => {
       setTimeout(() => {
         this.handleOrientationChange();
       }, 100);
     });

     // Resize
     window.addEventListener('resize', () => {
       this.handleResize();
     });

     // Visual viewport changes (mobile browsers)
     if ('visualViewport' in window) {
       window.visualViewport!.addEventListener('resize', () => {
         this.handleViewportChange();
       });
     }

     // Performance monitoring
     this.startPerformanceMonitoring();
   }

   private handleOrientationChange(): void {
     this.capabilities!.screenSize = this.getScreenInfo();
     this.adaptiveConfig = this.generateAdaptiveConfig();
     
     this.orientationChangeListeners.forEach(listener => listener());
   }

   private handleResize(): void {
     this.capabilities!.screenSize = this.getScreenInfo();
   }

   private handleViewportChange(): void {
     this.capabilities!.screenSize = this.getScreenInfo();
   }

   private startPerformanceMonitoring(): void {
     setInterval(() => {
       this.updatePerformanceMetrics();
     }, 1000);
   }

   private updatePerformanceMetrics(): void {
     // Monitor frame time
     let frameStart = performance.now();
     requestAnimationFrame(() => {
       this.performanceMetrics.frameTime = performance.now() - frameStart;
     });

     // Monitor memory (if available)
     if ('memory' in performance) {
       const memory = (performance as any).memory;
       this.performanceMetrics.memoryUsage = memory.usedJSHeapSize / memory.totalJSHeapSize;
     }

     // Monitor battery (if available)
     if (this.capabilities?.features.batteryAPI) {
       (navigator as any).getBattery().then((battery: any) => {
         this.performanceMetrics.batteryLevel = battery.level;
         this.performanceMetrics.isCharging = battery.charging;
       });
     }
   }

   // Public API
   public getCapabilities(): DeviceCapabilities | null {
     return this.capabilities;
   }

   public getAdaptiveConfig(): AdaptiveConfig | null {
     return this.adaptiveConfig;
   }

   public isMobile(): boolean {
     return this.capabilities?.isMobile || false;
   }

   public isTablet(): boolean {
     return this.capabilities?.isTablet || false;
   }

   public isDesktop(): boolean {
     return this.capabilities?.isDesktop || false;
   }

   public getPerformanceMetrics() {
     return { ...this.performanceMetrics };
   }

   public onOrientationChange(callback: () => void): void {
     this.orientationChangeListeners.push(callback);
   }

   public removeOrientationListener(callback: () => void): void {
     const index = this.orientationChangeListeners.indexOf(callback);
     if (index > -1) {
       this.orientationChangeListeners.splice(index, 1);
     }
   }

   public recalibrate(): void {
     this.detectCapabilities();
   }
   ```

## Expected Outputs
- Comprehensive device capability assessment and classification
- Adaptive configuration generation optimized for detected hardware
- Real-time performance monitoring and adjustment capabilities
- Cross-platform mobile, tablet, and desktop detection accuracy
- Network-aware optimization settings for low-bandwidth scenarios

## Validation
1. Correctly identifies mobile vs tablet vs desktop across major platforms
2. Accurately detects GPU performance tier and adjusts rendering accordingly
3. Adaptive config reduces resource usage by >50% on low-end devices
4. Performance monitoring tracks frame time and memory usage reliably
5. Orientation and viewport changes trigger immediate reconfiguration

## Next Steps
- Integration with touch gesture system (micro-phase 9.32)
- UI adaptation based on detected capabilities (micro-phase 9.33)