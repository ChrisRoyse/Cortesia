# Micro-Phase 9.17: WASM Module Loader Implementation

## Objective
Create a robust WASM module loader with initialization, memory management, and error handling for the CortexKG WebAssembly interface.

## Prerequisites
- Completed micro-phase 9.16 (JS project setup)
- WASM module built and available in pkg directory
- TypeScript project structure initialized

## Task Description
Implement the core WASM loader that handles module initialization, memory allocation, and provides a clean interface for loading the CortexKG WebAssembly module with proper error handling and performance monitoring.

## Specific Actions

1. **Create WASM loader utility**:
   ```typescript
   // src/core/WasmLoader.ts
   export interface WasmLoadResult {
     module: WebAssembly.Module;
     instance: WebAssembly.Instance;
     memory: WebAssembly.Memory;
     exports: any;
     loadTimeMs: number;
   }

   export interface WasmLoadOptions {
     wasmPath: string;
     memoryInitialPages?: number;
     memoryMaximumPages?: number;
     enableSharedMemory?: boolean;
     timeout?: number;
   }

   export class WasmLoader {
     private static instance: WasmLoader;
     private loadedModules: Map<string, WasmLoadResult> = new Map();
     
     static getInstance(): WasmLoader {
       if (!WasmLoader.instance) {
         WasmLoader.instance = new WasmLoader();
       }
       return WasmLoader.instance;
     }

     async loadModule(options: WasmLoadOptions): Promise<WasmLoadResult> {
       const startTime = performance.now();
       
       try {
         // Check cache first
         const cached = this.loadedModules.get(options.wasmPath);
         if (cached) {
           return cached;
         }

         // Fetch WASM bytes
         const wasmBytes = await this.fetchWasmBytes(options.wasmPath, options.timeout);
         
         // Create memory if specified
         const memory = this.createMemory(options);
         
         // Compile and instantiate
         const module = await WebAssembly.compile(wasmBytes);
         const instance = await WebAssembly.instantiate(module, {
           env: {
             memory: memory,
             abort: this.handleAbort.bind(this),
           }
         });

         const result: WasmLoadResult = {
           module,
           instance,
           memory,
           exports: instance.exports,
           loadTimeMs: performance.now() - startTime
         };

         // Cache the result
         this.loadedModules.set(options.wasmPath, result);
         
         return result;
       } catch (error) {
         throw new Error(`Failed to load WASM module: ${error.message}`);
       }
     }

     private async fetchWasmBytes(wasmPath: string, timeout: number = 10000): Promise<ArrayBuffer> {
       const controller = new AbortController();
       const timeoutId = setTimeout(() => controller.abort(), timeout);

       try {
         const response = await fetch(wasmPath, {
           signal: controller.signal,
           headers: {
             'Accept': 'application/wasm'
           }
         });

         if (!response.ok) {
           throw new Error(`HTTP ${response.status}: ${response.statusText}`);
         }

         return await response.arrayBuffer();
       } finally {
         clearTimeout(timeoutId);
       }
     }

     private createMemory(options: WasmLoadOptions): WebAssembly.Memory | undefined {
       if (options.memoryInitialPages !== undefined) {
         return new WebAssembly.Memory({
           initial: options.memoryInitialPages,
           maximum: options.memoryMaximumPages,
           shared: options.enableSharedMemory || false
         });
       }
       return undefined;
     }

     private handleAbort(message: number, fileName: number, line: number, column: number): void {
       console.error('WASM abort:', { message, fileName, line, column });
       throw new Error(`WASM module aborted at ${line}:${column}`);
     }

     clearCache(): void {
       this.loadedModules.clear();
     }

     getCacheSize(): number {
       return this.loadedModules.size;
     }
   }
   ```

2. **Create initialization manager**:
   ```typescript
   // src/core/WasmInitializer.ts
   import { WasmLoader, WasmLoadOptions } from './WasmLoader';

   export interface InitializationConfig {
     wasmPath: string;
     memorySize?: number;
     enableSIMD?: boolean;
     enableThreads?: boolean;
     enableBulkMemory?: boolean;
     debugMode?: boolean;
   }

   export interface InitializationResult {
     success: boolean;
     loadTimeMs: number;
     memoryUsageMB: number;
     features: {
       simd: boolean;
       threads: boolean;
       bulkMemory: boolean;
     };
     error?: string;
   }

   export class WasmInitializer {
     private loader: WasmLoader;
     private initialized: boolean = false;
     private wasmResult: any = null;

     constructor() {
       this.loader = WasmLoader.getInstance();
     }

     async initialize(config: InitializationConfig): Promise<InitializationResult> {
       const startTime = performance.now();

       try {
         // Check WebAssembly support
         if (!this.checkWasmSupport()) {
           throw new Error('WebAssembly is not supported in this environment');
         }

         // Detect available features
         const features = await this.detectWasmFeatures();
         
         // Configure load options
         const loadOptions: WasmLoadOptions = {
           wasmPath: config.wasmPath,
           memoryInitialPages: config.memorySize ? Math.ceil(config.memorySize / 64) : 16,
           memoryMaximumPages: 1024, // 64MB max
           enableSharedMemory: config.enableThreads && features.threads,
           timeout: 15000
         };

         // Load the WASM module
         this.wasmResult = await this.loader.loadModule(loadOptions);
         
         // Verify exports
         this.verifyExports(this.wasmResult.exports);
         
         this.initialized = true;

         return {
           success: true,
           loadTimeMs: performance.now() - startTime,
           memoryUsageMB: this.getMemoryUsage(),
           features: {
             simd: features.simd && (config.enableSIMD ?? true),
             threads: features.threads && (config.enableThreads ?? false),
             bulkMemory: features.bulkMemory && (config.enableBulkMemory ?? true)
           }
         };

       } catch (error) {
         return {
           success: false,
           loadTimeMs: performance.now() - startTime,
           memoryUsageMB: 0,
           features: { simd: false, threads: false, bulkMemory: false },
           error: error.message
         };
       }
     }

     private checkWasmSupport(): boolean {
       return typeof WebAssembly === 'object' && 
              typeof WebAssembly.instantiate === 'function';
     }

     private async detectWasmFeatures(): Promise<{ simd: boolean; threads: boolean; bulkMemory: boolean }> {
       const features = {
         simd: false,
         threads: false,
         bulkMemory: false
       };

       try {
         // Test SIMD support
         const simdTest = new Uint8Array([
           0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, // WASM header
           0x01, 0x04, 0x01, 0x60, 0x00, 0x00,             // Type section
           0x03, 0x02, 0x01, 0x00,                         // Function section
           0x0a, 0x0b, 0x01, 0x09, 0x00, 0xfd, 0x00, 0x0b // Code section with SIMD
         ]);
         await WebAssembly.instantiate(simdTest);
         features.simd = true;
       } catch (e) {
         // SIMD not supported
       }

       try {
         // Test shared memory (threads) support
         new WebAssembly.Memory({ initial: 1, maximum: 1, shared: true });
         features.threads = true;
       } catch (e) {
         // Threads not supported
       }

       // Bulk memory is widely supported, assume true
       features.bulkMemory = true;

       return features;
     }

     private verifyExports(exports: any): void {
       const requiredExports = [
         'memory',
         'cortexkg_new',
         'cortexkg_initialize',
         'cortexkg_allocate_concept',
         'cortexkg_query'
       ];

       for (const exportName of requiredExports) {
         if (!(exportName in exports)) {
           throw new Error(`Missing required WASM export: ${exportName}`);
         }
       }
     }

     private getMemoryUsage(): number {
       if (this.wasmResult && this.wasmResult.memory) {
         return (this.wasmResult.memory.buffer.byteLength / (1024 * 1024));
       }
       return 0;
     }

     getWasmExports(): any {
       if (!this.initialized || !this.wasmResult) {
         throw new Error('WASM module not initialized');
       }
       return this.wasmResult.exports;
     }

     isInitialized(): boolean {
       return this.initialized;
     }

     dispose(): void {
       this.wasmResult = null;
       this.initialized = false;
       this.loader.clearCache();
     }
   }
   ```

3. **Create memory management utilities**:
   ```typescript
   // src/utils/MemoryManager.ts
   export class MemoryManager {
     private wasmMemory: WebAssembly.Memory;
     private allocatedPointers: Set<number> = new Set();

     constructor(memory: WebAssembly.Memory) {
       this.wasmMemory = memory;
     }

     allocateString(str: string): number {
       const encoder = new TextEncoder();
       const bytes = encoder.encode(str);
       const ptr = this.allocateBytes(bytes.length + 1);
       
       const view = new Uint8Array(this.wasmMemory.buffer, ptr, bytes.length + 1);
       view.set(bytes);
       view[bytes.length] = 0; // null terminator
       
       return ptr;
     }

     readString(ptr: number, length?: number): string {
       const buffer = this.wasmMemory.buffer;
       const view = new Uint8Array(buffer, ptr);
       
       if (length === undefined) {
         // Find null terminator
         length = 0;
         while (view[length] !== 0 && ptr + length < buffer.byteLength) {
           length++;
         }
       }

       const decoder = new TextDecoder();
       return decoder.decode(new Uint8Array(buffer, ptr, length));
     }

     allocateBytes(size: number): number {
       // Simple allocator - in practice, this would use WASM's allocator
       const pages = Math.ceil(size / 65536);
       const currentPages = this.wasmMemory.buffer.byteLength / 65536;
       
       if (currentPages * 65536 < size) {
         this.wasmMemory.grow(pages);
       }

       // Return a mock pointer (in practice, call WASM's malloc)
       const ptr = Math.floor(Math.random() * 1000000) + 65536;
       this.allocatedPointers.add(ptr);
       return ptr;
     }

     deallocate(ptr: number): void {
       this.allocatedPointers.delete(ptr);
       // In practice, call WASM's free function
     }

     getMemoryStats(): { totalBytes: number; allocatedPointers: number } {
       return {
         totalBytes: this.wasmMemory.buffer.byteLength,
         allocatedPointers: this.allocatedPointers.size
       };
     }

     cleanup(): void {
       this.allocatedPointers.clear();
     }
   }
   ```

## Expected Outputs
- Complete WASM loader with caching and error handling
- Initialization manager with feature detection
- Memory management utilities for string/byte allocation
- Performance monitoring and diagnostics
- Support for SIMD, threads, and bulk memory features

## Validation
1. WASM module loads successfully with proper error handling
2. Feature detection correctly identifies browser capabilities
3. Memory allocation and deallocation work without leaks
4. Initialization times are logged and reasonable (<2 seconds)
5. Cache mechanisms improve subsequent load times

## Next Steps
- Build JavaScript API wrapper (micro-phase 9.18)
- Implement promise-based interfaces (micro-phase 9.19)