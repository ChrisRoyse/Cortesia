# Micro-Phase 9.26: Canvas Setup Component

## Objective
Create a high-performance canvas rendering system with WebGL fallback and optimized rendering pipeline for neuromorphic visualizations.

## Prerequisites
- Completed micro-phase 9.25 (Allocation Interface)
- Understanding of Canvas 2D and WebGL APIs
- Knowledge of rendering optimization techniques

## Task Description
Implement a robust canvas setup system that automatically detects capabilities, manages multiple rendering contexts, and provides optimized drawing operations.

## Specific Actions

1. **Create CanvasManager class structure**:
   ```typescript
   // src/ui/CanvasManager.ts
   export interface CanvasConfig {
     container: HTMLElement;
     preferWebGL?: boolean;
     enableOffscreenCanvas?: boolean;
     pixelRatio?: number;
     antialias?: boolean;
     preserveDrawingBuffer?: boolean;
     enableDebugInfo?: boolean;
   }
   
   export interface RenderingCapabilities {
     webgl: boolean;
     webgl2: boolean;
     offscreenCanvas: boolean;
     hardwareAcceleration: boolean;
     maxTextureSize: number;
     maxViewportDims: [number, number];
     supportedExtensions: string[];
   }
   
   export interface ViewportConfig {
     x: number;
     y: number;
     width: number;
     height: number;
     devicePixelRatio: number;
   }
   
   export type RenderingContext = CanvasRenderingContext2D | WebGLRenderingContext | WebGL2RenderingContext;
   
   export class CanvasManager {
     private container: HTMLElement;
     private canvas: HTMLCanvasElement;
     private offscreenCanvas: OffscreenCanvas | null = null;
     private context: RenderingContext | null = null;
     private webglContext: WebGLRenderingContext | WebGL2RenderingContext | null = null;
     private config: Required<CanvasConfig>;
     private capabilities: RenderingCapabilities;
     private viewport: ViewportConfig;
     private frameBuffer: WebGLFramebuffer | null = null;
     private debugInfo: WebGLDebugInfo | null = null;
     private resizeObserver: ResizeObserver | null = null;
     private isWebGL = false;
     private isWebGL2 = false;
     
     constructor(config: CanvasConfig) {
       this.container = config.container;
       this.config = {
         container: config.container,
         preferWebGL: config.preferWebGL ?? true,
         enableOffscreenCanvas: config.enableOffscreenCanvas ?? true,
         pixelRatio: config.pixelRatio ?? window.devicePixelRatio || 1,
         antialias: config.antialias ?? true,
         preserveDrawingBuffer: config.preserveDrawingBuffer ?? false,
         enableDebugInfo: config.enableDebugInfo ?? false
       };
       
       this.capabilities = this.detectCapabilities();
       this.setupCanvas();
       this.setupContext();
       this.setupViewport();
       this.setupResizeHandling();
       
       if (this.config.enableDebugInfo) {
         this.setupDebugInfo();
       }
     }
   }
   
   interface WebGLDebugInfo {
     renderer: string;
     vendor: string;
     version: string;
     shadingLanguageVersion: string;
     extensions: string[];
   }
   ```

2. **Implement capability detection and canvas setup**:
   ```typescript
   private detectCapabilities(): RenderingCapabilities {
     const testCanvas = document.createElement('canvas');
     const capabilities: RenderingCapabilities = {
       webgl: false,
       webgl2: false,
       offscreenCanvas: typeof OffscreenCanvas !== 'undefined',
       hardwareAcceleration: false,
       maxTextureSize: 0,
       maxViewportDims: [0, 0],
       supportedExtensions: []
     };
     
     // Test WebGL support
     try {
       const webgl2Context = testCanvas.getContext('webgl2');
       if (webgl2Context) {
         capabilities.webgl2 = true;
         capabilities.webgl = true;
         capabilities.maxTextureSize = webgl2Context.getParameter(webgl2Context.MAX_TEXTURE_SIZE);
         capabilities.maxViewportDims = webgl2Context.getParameter(webgl2Context.MAX_VIEWPORT_DIMS);
         capabilities.supportedExtensions = webgl2Context.getSupportedExtensions() || [];
         
         // Check for hardware acceleration
         const debugInfo = webgl2Context.getExtension('WEBGL_debug_renderer_info');
         if (debugInfo) {
           const renderer = webgl2Context.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
           capabilities.hardwareAcceleration = !renderer.toLowerCase().includes('software');
         }
       } else {
         const webglContext = testCanvas.getContext('webgl');
         if (webglContext) {
           capabilities.webgl = true;
           capabilities.maxTextureSize = webglContext.getParameter(webglContext.MAX_TEXTURE_SIZE);
           capabilities.maxViewportDims = webglContext.getParameter(webglContext.MAX_VIEWPORT_DIMS);
           capabilities.supportedExtensions = webglContext.getSupportedExtensions() || [];
           
           const debugInfo = webglContext.getExtension('WEBGL_debug_renderer_info');
           if (debugInfo) {
             const renderer = webglContext.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
             capabilities.hardwareAcceleration = !renderer.toLowerCase().includes('software');
           }
         }
       }
     } catch (error) {
       console.warn('WebGL capability detection failed:', error);
     }
     
     return capabilities;
   }
   
   private setupCanvas(): void {
     this.canvas = document.createElement('canvas');
     this.canvas.style.display = 'block';
     this.canvas.style.width = '100%';
     this.canvas.style.height = '100%';
     this.canvas.style.touchAction = 'none'; // Prevent scrolling on touch
     
     // Add accessibility attributes
     this.canvas.setAttribute('role', 'img');
     this.canvas.setAttribute('aria-label', 'Neuromorphic network visualization');
     
     this.container.appendChild(this.canvas);
     
     // Setup offscreen canvas if available and enabled
     if (this.capabilities.offscreenCanvas && this.config.enableOffscreenCanvas) {
       try {
         this.offscreenCanvas = this.canvas.transferControlToOffscreen();
       } catch (error) {
         console.warn('Failed to create offscreen canvas:', error);
         this.offscreenCanvas = null;
       }
     }
   }
   
   private setupContext(): void {
     const canvas = this.offscreenCanvas || this.canvas;
     
     // Try to get the best available context
     if (this.config.preferWebGL && this.capabilities.webgl) {
       this.context = this.setupWebGLContext(canvas);
       if (this.context) {
         this.webglContext = this.context as WebGLRenderingContext | WebGL2RenderingContext;
         this.isWebGL = true;
         this.isWebGL2 = this.capabilities.webgl2;
         return;
       }
     }
     
     // Fallback to 2D context
     this.context = this.setup2DContext(canvas);
     if (!this.context) {
       throw new Error('Failed to obtain any rendering context');
     }
   }
   
   private setupWebGLContext(canvas: HTMLCanvasElement | OffscreenCanvas): WebGLRenderingContext | WebGL2RenderingContext | null {
     const contextOptions: WebGLContextAttributes = {
       antialias: this.config.antialias,
       preserveDrawingBuffer: this.config.preserveDrawingBuffer,
       alpha: true,
       depth: true,
       stencil: false,
       powerPreference: 'high-performance',
       failIfMajorPerformanceCaveat: false
     };
     
     // Try WebGL2 first
     if (this.capabilities.webgl2) {
       const context = canvas.getContext('webgl2', contextOptions) as WebGL2RenderingContext;
       if (context) {
         this.setupWebGLExtensions(context);
         return context;
       }
     }
     
     // Fallback to WebGL1
     if (this.capabilities.webgl) {
       const context = canvas.getContext('webgl', contextOptions) as WebGLRenderingContext;
       if (context) {
         this.setupWebGLExtensions(context);
         return context;
       }
     }
     
     return null;
   }
   
   private setup2DContext(canvas: HTMLCanvasElement | OffscreenCanvas): CanvasRenderingContext2D | null {
     const contextOptions: CanvasRenderingContext2DSettings = {
       alpha: true,
       desynchronized: true
     };
     
     const context = canvas.getContext('2d', contextOptions) as CanvasRenderingContext2D;
     if (context) {
       // Optimize 2D context
       context.imageSmoothingEnabled = this.config.antialias;
       context.imageSmoothingQuality = 'high';
     }
     
     return context;
   }
   
   private setupWebGLExtensions(context: WebGLRenderingContext | WebGL2RenderingContext): void {
     // Enable useful extensions
     const extensions = [
       'OES_vertex_array_object',
       'WEBGL_debug_renderer_info',
       'EXT_texture_filter_anisotropic',
       'OES_standard_derivatives',
       'EXT_shader_texture_lod'
     ];
     
     extensions.forEach(ext => {
       try {
         context.getExtension(ext);
       } catch (error) {
         // Extension not available, continue
       }
     });
   }
   ```

3. **Implement viewport management and responsive sizing**:
   ```typescript
   private setupViewport(): void {
     this.updateViewport();
     
     // Set initial WebGL viewport if using WebGL
     if (this.isWebGL && this.webglContext) {
       this.webglContext.viewport(0, 0, this.viewport.width, this.viewport.height);
     }
   }
   
   private updateViewport(): void {
     const rect = this.container.getBoundingClientRect();
     const dpr = this.config.pixelRatio;
     
     // Update canvas size
     this.canvas.width = rect.width * dpr;
     this.canvas.height = rect.height * dpr;
     this.canvas.style.width = rect.width + 'px';
     this.canvas.style.height = rect.height + 'px';
     
     // Update viewport config
     this.viewport = {
       x: 0,
       y: 0,
       width: this.canvas.width,
       height: this.canvas.height,
       devicePixelRatio: dpr
     };
     
     // Scale 2D context for device pixel ratio
     if (!this.isWebGL && this.context) {
       const ctx = this.context as CanvasRenderingContext2D;
       ctx.scale(dpr, dpr);
     }
     
     // Update WebGL viewport
     if (this.isWebGL && this.webglContext) {
       this.webglContext.viewport(0, 0, this.viewport.width, this.viewport.height);
     }
   }
   
   private setupResizeHandling(): void {
     if (typeof ResizeObserver !== 'undefined') {
       this.resizeObserver = new ResizeObserver((entries) => {
         for (const entry of entries) {
           if (entry.target === this.container) {
             this.handleResize();
           }
         }
       });
       
       this.resizeObserver.observe(this.container);
     } else {
       // Fallback to window resize
       window.addEventListener('resize', () => {
         this.handleResize();
       });
     }
   }
   
   private handleResize(): void {
     // Debounce resize handling
     clearTimeout(this.resizeTimeout);
     this.resizeTimeout = window.setTimeout(() => {
       this.updateViewport();
       this.onResize();
     }, 16); // ~60fps
   }
   
   private resizeTimeout: number = 0;
   
   protected onResize(): void {
     // Override in subclasses for custom resize handling
     this.container.dispatchEvent(new CustomEvent('canvasResize', {
       detail: { viewport: this.viewport }
     }));
   }
   ```

4. **Implement rendering utilities and optimizations**:
   ```typescript
   public clear(color: [number, number, number, number] = [0, 0, 0, 1]): void {
     if (this.isWebGL && this.webglContext) {
       const gl = this.webglContext;
       gl.clearColor(color[0], color[1], color[2], color[3]);
       gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
     } else if (this.context) {
       const ctx = this.context as CanvasRenderingContext2D;
       ctx.save();
       ctx.setTransform(1, 0, 0, 1, 0, 0);
       ctx.fillStyle = `rgba(${color[0] * 255}, ${color[1] * 255}, ${color[2] * 255}, ${color[3]})`;
       ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
       ctx.restore();
     }
   }
   
   public createFrameBuffer(width: number, height: number): WebGLFramebuffer | null {
     if (!this.isWebGL || !this.webglContext) return null;
     
     const gl = this.webglContext;
     const framebuffer = gl.createFramebuffer();
     const texture = gl.createTexture();
     
     if (!framebuffer || !texture) return null;
     
     // Setup texture
     gl.bindTexture(gl.TEXTURE_2D, texture);
     gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
     gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
     gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
     gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
     gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
     
     // Attach texture to framebuffer
     gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
     gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
     
     // Check completeness
     if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
       gl.deleteFramebuffer(framebuffer);
       gl.deleteTexture(texture);
       return null;
     }
     
     gl.bindFramebuffer(gl.FRAMEBUFFER, null);
     gl.bindTexture(gl.TEXTURE_2D, null);
     
     return framebuffer;
   }
   
   public createShaderProgram(vertexSource: string, fragmentSource: string): WebGLProgram | null {
     if (!this.isWebGL || !this.webglContext) return null;
     
     const gl = this.webglContext;
     
     const vertexShader = this.compileShader(gl.VERTEX_SHADER, vertexSource);
     const fragmentShader = this.compileShader(gl.FRAGMENT_SHADER, fragmentSource);
     
     if (!vertexShader || !fragmentShader) return null;
     
     const program = gl.createProgram();
     if (!program) return null;
     
     gl.attachShader(program, vertexShader);
     gl.attachShader(program, fragmentShader);
     gl.linkProgram(program);
     
     if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
       console.error('Shader program link error:', gl.getProgramInfoLog(program));
       gl.deleteProgram(program);
       return null;
     }
     
     // Clean up shaders
     gl.deleteShader(vertexShader);
     gl.deleteShader(fragmentShader);
     
     return program;
   }
   
   private compileShader(type: number, source: string): WebGLShader | null {
     if (!this.webglContext) return null;
     
     const gl = this.webglContext;
     const shader = gl.createShader(type);
     
     if (!shader) return null;
     
     gl.shaderSource(shader, source);
     gl.compileShader(shader);
     
     if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
       console.error('Shader compile error:', gl.getShaderInfoLog(shader));
       gl.deleteShader(shader);
       return null;
     }
     
     return shader;
   }
   
   public createVertexBuffer(data: Float32Array): WebGLBuffer | null {
     if (!this.isWebGL || !this.webglContext) return null;
     
     const gl = this.webglContext;
     const buffer = gl.createBuffer();
     
     if (!buffer) return null;
     
     gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
     gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
     gl.bindBuffer(gl.ARRAY_BUFFER, null);
     
     return buffer;
   }
   
   public measureText(text: string, font: string): TextMetrics | null {
     if (this.isWebGL) {
       // Create temporary 2D context for text measurement
       const tempCanvas = document.createElement('canvas');
       const tempCtx = tempCanvas.getContext('2d');
       if (!tempCtx) return null;
       
       tempCtx.font = font;
       return tempCtx.measureText(text);
     } else {
       const ctx = this.context as CanvasRenderingContext2D;
       const oldFont = ctx.font;
       ctx.font = font;
       const metrics = ctx.measureText(text);
       ctx.font = oldFont;
       return metrics;
     }
   }
   ```

5. **Implement debug information and performance monitoring**:
   ```typescript
   private setupDebugInfo(): void {
     if (!this.isWebGL || !this.webglContext) return;
     
     const gl = this.webglContext;
     const debugExtension = gl.getExtension('WEBGL_debug_renderer_info');
     
     if (debugExtension) {
       this.debugInfo = {
         renderer: gl.getParameter(debugExtension.UNMASKED_RENDERER_WEBGL),
         vendor: gl.getParameter(debugExtension.UNMASKED_VENDOR_WEBGL),
         version: gl.getParameter(gl.VERSION),
         shadingLanguageVersion: gl.getParameter(gl.SHADING_LANGUAGE_VERSION),
         extensions: gl.getSupportedExtensions() || []
       };
       
       console.log('WebGL Debug Info:', this.debugInfo);
     }
   }
   
   public getPerformanceInfo(): any {
     const info: any = {
       context: this.isWebGL ? (this.isWebGL2 ? 'WebGL2' : 'WebGL') : '2D',
       capabilities: this.capabilities,
       viewport: this.viewport
     };
     
     if (this.debugInfo) {
       info.debug = this.debugInfo;
     }
     
     if (this.isWebGL && this.webglContext) {
       const gl = this.webglContext;
       info.webgl = {
         maxTextureSize: gl.getParameter(gl.MAX_TEXTURE_SIZE),
         maxViewportDims: gl.getParameter(gl.MAX_VIEWPORT_DIMS),
         maxVertexAttribs: gl.getParameter(gl.MAX_VERTEX_ATTRIBS),
         maxVaryingVectors: gl.getParameter(gl.MAX_VARYING_VECTORS),
         maxFragmentUniforms: gl.getParameter(gl.MAX_FRAGMENT_UNIFORM_VECTORS),
         maxVertexUniforms: gl.getParameter(gl.MAX_VERTEX_UNIFORM_VECTORS)
       };
     }
     
     return info;
   }
   
   public enableDebugMode(): void {
     if (!this.isWebGL || !this.webglContext) return;
     
     // Wrap WebGL calls for debugging
     const gl = this.webglContext;
     const originalGetError = gl.getError.bind(gl);
     
     const checkError = (functionName: string) => {
       const error = originalGetError();
       if (error !== gl.NO_ERROR) {
         console.error(`WebGL Error in ${functionName}:`, this.getErrorString(error));
       }
     };
     
     // Override key WebGL functions to add error checking
     const functionsToWrap = ['drawArrays', 'drawElements', 'useProgram', 'bindBuffer'];
     
     functionsToWrap.forEach(funcName => {
       const original = (gl as any)[funcName].bind(gl);
       (gl as any)[funcName] = (...args: any[]) => {
         const result = original(...args);
         checkError(funcName);
         return result;
       };
     });
   }
   
   private getErrorString(error: number): string {
     if (!this.webglContext) return 'Unknown error';
     
     const gl = this.webglContext;
     switch (error) {
       case gl.NO_ERROR: return 'NO_ERROR';
       case gl.INVALID_ENUM: return 'INVALID_ENUM';
       case gl.INVALID_VALUE: return 'INVALID_VALUE';
       case gl.INVALID_OPERATION: return 'INVALID_OPERATION';
       case gl.OUT_OF_MEMORY: return 'OUT_OF_MEMORY';
       case gl.CONTEXT_LOST_WEBGL: return 'CONTEXT_LOST_WEBGL';
       default: return `Unknown error: ${error}`;
     }
   }
   
   public exportAsImage(format: 'png' | 'jpeg' = 'png', quality: number = 0.9): string {
     return this.canvas.toDataURL(`image/${format}`, quality);
   }
   
   public dispose(): void {
     // Clean up resources
     if (this.resizeObserver) {
       this.resizeObserver.disconnect();
       this.resizeObserver = null;
     }
     
     if (this.frameBuffer && this.webglContext) {
       this.webglContext.deleteFramebuffer(this.frameBuffer);
       this.frameBuffer = null;
     }
     
     // Remove canvas from DOM
     if (this.canvas.parentNode) {
       this.canvas.parentNode.removeChild(this.canvas);
     }
   }
   
   // Getters
   public get canvasElement(): HTMLCanvasElement { return this.canvas; }
   public get renderingContext(): RenderingContext | null { return this.context; }
   public get isWebGLEnabled(): boolean { return this.isWebGL; }
   public get isWebGL2Enabled(): boolean { return this.isWebGL2; }
   public get viewportConfig(): ViewportConfig { return { ...this.viewport }; }
   public get renderingCapabilities(): RenderingCapabilities { return { ...this.capabilities }; }
   ```

## Expected Outputs
- Automated WebGL/Canvas 2D context selection
- Responsive viewport management with device pixel ratio support
- Performance-optimized rendering pipeline
- Debug mode with error checking and performance monitoring
- Utility methods for common rendering operations
- Proper resource cleanup and memory management

## Validation
1. Context creation succeeds on all major browsers
2. Viewport updates smoothly during window resize
3. WebGL fallback to 2D works correctly when WebGL unavailable
4. Performance monitoring provides accurate metrics
5. Memory usage remains stable during extended use

## Next Steps
- Create column rendering component (micro-phase 9.27)
- Integrate canvas manager with cortical visualizer