# Micro-Phase 9.18: JavaScript API Wrapper Implementation

## Objective
Build a comprehensive JavaScript API wrapper that provides a clean, promise-based interface to the CortexKG WASM module with proper type safety and error handling.

## Prerequisites
- Completed micro-phase 9.17 (WASM loader)
- WASM module initialization working
- TypeScript project structure with proper type definitions

## Task Description
Create the main JavaScript API wrapper class that encapsulates all WASM functionality and provides a modern, promise-based interface for web developers to interact with the CortexKG knowledge system.

## Specific Actions

1. **Create main API wrapper class**:
   ```typescript
   // src/core/CortexKGWeb.ts
   import { WasmInitializer, InitializationConfig, InitializationResult } from './WasmInitializer';
   import { MemoryManager } from '../utils/MemoryManager';
   import { 
     AllocationResult, 
     QueryResult, 
     PerformanceMetrics, 
     CortexConfig 
   } from '../types/cortexkg';

   export interface CortexKGWebConfig extends CortexConfig {
     wasmPath?: string;
     autoInitialize?: boolean;
     enablePersistence?: boolean;
     databaseName?: string;
     debugMode?: boolean;
   }

   export interface ColumnState {
     id: number;
     allocated: boolean;
     activation: number;
     conceptCount: number;
     lastAccessed: Date;
   }

   export class CortexKGWeb {
     private initializer: WasmInitializer;
     private wasmExports: any = null;
     private memoryManager: MemoryManager | null = null;
     private config: CortexKGWebConfig;
     private isReady: boolean = false;
     private initializationPromise: Promise<void> | null = null;
     private performanceCollector: PerformanceCollector;

     constructor(config: CortexKGWebConfig = {}) {
       this.config = {
         wasmPath: './wasm/cortexkg_wasm.wasm',
         column_count: 100,
         max_connections_per_column: 50,
         enable_simd: true,
         cache_size_mb: 64,
         autoInitialize: true,
         enablePersistence: false,
         debugMode: false,
         ...config
       };

       this.initializer = new WasmInitializer();
       this.performanceCollector = new PerformanceCollector();

       if (this.config.autoInitialize) {
         this.initializationPromise = this.initialize();
       }
     }

     async initialize(): Promise<void> {
       if (this.isReady) {
         return;
       }

       if (this.initializationPromise) {
         return this.initializationPromise;
       }

       this.initializationPromise = this._doInitialize();
       return this.initializationPromise;
     }

     private async _doInitialize(): Promise<void> {
       try {
         const initConfig: InitializationConfig = {
           wasmPath: this.config.wasmPath!,
           memorySize: this.config.cache_size_mb,
           enableSIMD: this.config.enable_simd,
           enableThreads: false, // Keep false for now
           enableBulkMemory: true,
           debugMode: this.config.debugMode
         };

         const result: InitializationResult = await this.initializer.initialize(initConfig);
         
         if (!result.success) {
           throw new Error(`Initialization failed: ${result.error}`);
         }

         this.wasmExports = this.initializer.getWasmExports();
         this.memoryManager = new MemoryManager(this.wasmExports.memory);

         // Initialize the CortexKG instance in WASM
         const cortexPtr = this.wasmExports.cortexkg_new(
           this.config.column_count,
           this.config.max_connections_per_column,
           this.config.enable_simd ? 1 : 0
         );

         if (cortexPtr === 0) {
           throw new Error('Failed to create CortexKG instance in WASM');
         }

         // Store the pointer for future calls
         (this as any)._cortexPtr = cortexPtr;

         // Initialize with persistence if enabled
         if (this.config.enablePersistence && this.config.databaseName) {
           await this.initializeWithStorage(this.config.databaseName);
         } else {
           const initResult = this.wasmExports.cortexkg_initialize(cortexPtr);
           if (initResult !== 0) {
             throw new Error('WASM CortexKG initialization failed');
           }
         }

         this.isReady = true;

         if (this.config.debugMode) {
           console.log('CortexKG Web initialized:', {
             loadTime: result.loadTimeMs,
             memoryUsage: result.memoryUsageMB,
             features: result.features,
             config: this.config
           });
         }

       } catch (error) {
         this.isReady = false;
         throw new Error(`CortexKG initialization failed: ${error.message}`);
       }
     }

     async allocateConcept(content: string): Promise<AllocationResult> {
       await this.ensureReady();
       const startTime = performance.now();

       try {
         const contentPtr = this.memoryManager!.allocateString(content);
         const resultPtr = this.wasmExports.cortexkg_allocate_concept(
           (this as any)._cortexPtr,
           contentPtr
         );

         if (resultPtr === 0) {
           throw new Error('Concept allocation failed in WASM');
         }

         // Read result from WASM memory
         const result = this.readAllocationResult(resultPtr);
         
         // Clean up memory
         this.memoryManager!.deallocate(contentPtr);
         this.wasmExports.deallocate_result(resultPtr);

         const processingTime = performance.now() - startTime;
         this.performanceCollector.recordAllocation(processingTime);

         return {
           ...result,
           processing_time_ms: processingTime
         };

       } catch (error) {
         throw new Error(`Concept allocation failed: ${error.message}`);
       }
     }

     async query(queryText: string, maxResults: number = 10): Promise<QueryResult[]> {
       await this.ensureReady();
       const startTime = performance.now();

       try {
         const queryPtr = this.memoryManager!.allocateString(queryText);
         const resultsPtr = this.wasmExports.cortexkg_query(
           (this as any)._cortexPtr,
           queryPtr,
           maxResults
         );

         if (resultsPtr === 0) {
           throw new Error('Query failed in WASM');
         }

         // Read results from WASM memory
         const results = this.readQueryResults(resultsPtr);
         
         // Clean up memory
         this.memoryManager!.deallocate(queryPtr);
         this.wasmExports.deallocate_query_results(resultsPtr);

         const processingTime = performance.now() - startTime;
         this.performanceCollector.recordQuery(processingTime, results.length);

         return results;

       } catch (error) {
         throw new Error(`Query failed: ${error.message}`);
       }
     }

     async getPerformanceMetrics(): Promise<PerformanceMetrics> {
       await this.ensureReady();

       const metricsPtr = this.wasmExports.cortexkg_get_performance_metrics(
         (this as any)._cortexPtr
       );

       if (metricsPtr === 0) {
         throw new Error('Failed to get performance metrics');
       }

       const metrics = this.readPerformanceMetrics(metricsPtr);
       this.wasmExports.deallocate_metrics(metricsPtr);

       return {
         ...metrics,
         ...this.performanceCollector.getMetrics()
       };
     }

     async getColumnStates(): Promise<ColumnState[]> {
       await this.ensureReady();

       const statesPtr = this.wasmExports.cortexkg_get_column_states(
         (this as any)._cortexPtr
       );

       if (statesPtr === 0) {
         throw new Error('Failed to get column states');
       }

       const states = this.readColumnStates(statesPtr);
       this.wasmExports.deallocate_column_states(statesPtr);

       return states;
     }

     async initializeWithStorage(databaseName: string): Promise<void> {
       await this.ensureReady();

       const dbNamePtr = this.memoryManager!.allocateString(databaseName);
       const result = this.wasmExports.cortexkg_initialize_with_storage(
         (this as any)._cortexPtr,
         dbNamePtr
       );

       this.memoryManager!.deallocate(dbNamePtr);

       if (result !== 0) {
         throw new Error('Storage initialization failed');
       }
     }

     private async ensureReady(): Promise<void> {
       if (!this.isReady) {
         if (!this.initializationPromise) {
           this.initializationPromise = this.initialize();
         }
         await this.initializationPromise;
       }

       if (!this.isReady) {
         throw new Error('CortexKG is not ready. Initialization may have failed.');
       }
     }

     private readAllocationResult(ptr: number): AllocationResult {
       // Read structured data from WASM memory
       const view = new DataView(this.wasmExports.memory.buffer, ptr);
       return {
         column_id: view.getUint32(0, true),
         confidence: view.getFloat64(8, true),
         processing_time_ms: 0 // Will be set by caller
       };
     }

     private readQueryResults(ptr: number): QueryResult[] {
       // Read array of query results from WASM memory
       const view = new DataView(this.wasmExports.memory.buffer, ptr);
       const count = view.getUint32(0, true);
       const results: QueryResult[] = [];

       for (let i = 0; i < count; i++) {
         const offset = 4 + (i * 32); // Assuming 32 bytes per result
         results.push({
           concept_id: view.getUint32(offset, true),
           content: this.memoryManager!.readString(view.getUint32(offset + 4, true)),
           relevance_score: view.getFloat64(offset + 8, true),
           activation_path: [] // TODO: Implement path reading
         });
       }

       return results;
     }

     private readPerformanceMetrics(ptr: number): PerformanceMetrics {
       const view = new DataView(this.wasmExports.memory.buffer, ptr);
       return {
         total_allocations: view.getUint32(0, true),
         average_allocation_time_ms: view.getFloat64(8, true),
         memory_usage_bytes: view.getUint32(16, true),
         cache_hit_rate: view.getFloat64(24, true)
       };
     }

     private readColumnStates(ptr: number): ColumnState[] {
       const view = new DataView(this.wasmExports.memory.buffer, ptr);
       const count = view.getUint32(0, true);
       const states: ColumnState[] = [];

       for (let i = 0; i < count; i++) {
         const offset = 4 + (i * 24); // Assuming 24 bytes per state
         states.push({
           id: view.getUint32(offset, true),
           allocated: view.getUint8(offset + 4) === 1,
           activation: view.getFloat64(offset + 8, true),
           conceptCount: view.getUint32(offset + 16, true),
           lastAccessed: new Date() // TODO: Read timestamp from WASM
         });
       }

       return states;
     }

     get isInitialized(): boolean {
       return this.isReady;
     }

     get columnCount(): number {
       return this.config.column_count || 100;
     }

     dispose(): void {
       if (this.memoryManager) {
         this.memoryManager.cleanup();
       }

       if (this.wasmExports && (this as any)._cortexPtr) {
         this.wasmExports.cortexkg_dispose((this as any)._cortexPtr);
       }

       this.initializer.dispose();
       this.isReady = false;
       this.initializationPromise = null;
     }
   }
   ```

2. **Create performance collector**:
   ```typescript
   // src/utils/PerformanceCollector.ts
   interface PerformanceData {
     allocationTimes: number[];
     queryTimes: number[];
     queryCounts: number[];
     startTime: number;
   }

   export class PerformanceCollector {
     private data: PerformanceData;
     private maxSamples: number = 1000;

     constructor() {
       this.data = {
         allocationTimes: [],
         queryTimes: [],
         queryCounts: [],
         startTime: performance.now()
       };
     }

     recordAllocation(timeMs: number): void {
       this.data.allocationTimes.push(timeMs);
       if (this.data.allocationTimes.length > this.maxSamples) {
         this.data.allocationTimes.shift();
       }
     }

     recordQuery(timeMs: number, resultCount: number): void {
       this.data.queryTimes.push(timeMs);
       this.data.queryCounts.push(resultCount);
       
       if (this.data.queryTimes.length > this.maxSamples) {
         this.data.queryTimes.shift();
         this.data.queryCounts.shift();
       }
     }

     getMetrics(): Partial<PerformanceMetrics> {
       return {
         average_allocation_time_ms: this.calculateAverage(this.data.allocationTimes),
         total_allocations: this.data.allocationTimes.length
       };
     }

     private calculateAverage(values: number[]): number {
       if (values.length === 0) return 0;
       return values.reduce((sum, val) => sum + val, 0) / values.length;
     }

     reset(): void {
       this.data = {
         allocationTimes: [],
         queryTimes: [],
         queryCounts: [],
         startTime: performance.now()
       };
     }
   }
   ```

3. **Create batch operations support**:
   ```typescript
   // src/core/BatchOperations.ts
   import { CortexKGWeb } from './CortexKGWeb';
   import { AllocationResult, QueryResult } from '../types/cortexkg';

   export interface BatchAllocationRequest {
     concepts: string[];
     batchSize?: number;
     onProgress?: (completed: number, total: number) => void;
   }

   export interface BatchAllocationResult {
     results: AllocationResult[];
     errors: Array<{ index: number; concept: string; error: string }>;
     totalTime: number;
   }

   export class BatchOperations {
     constructor(private cortex: CortexKGWeb) {}

     async allocateConceptsBatch(request: BatchAllocationRequest): Promise<BatchAllocationResult> {
       const { concepts, batchSize = 10, onProgress } = request;
       const startTime = performance.now();
       const results: AllocationResult[] = [];
       const errors: Array<{ index: number; concept: string; error: string }> = [];

       for (let i = 0; i < concepts.length; i += batchSize) {
         const batch = concepts.slice(i, i + batchSize);
         const batchPromises = batch.map(async (concept, batchIndex) => {
           const globalIndex = i + batchIndex;
           try {
             const result = await this.cortex.allocateConcept(concept);
             results[globalIndex] = result;
           } catch (error) {
             errors.push({
               index: globalIndex,
               concept,
               error: error.message
             });
           }
         });

         await Promise.all(batchPromises);

         if (onProgress) {
           onProgress(Math.min(i + batchSize, concepts.length), concepts.length);
         }

         // Small delay to prevent overwhelming the system
         if (i + batchSize < concepts.length) {
           await new Promise(resolve => setTimeout(resolve, 10));
         }
       }

       return {
         results: results.filter(r => r !== undefined),
         errors,
         totalTime: performance.now() - startTime
       };
     }

     async queryMultiple(queries: string[], maxResultsPerQuery: number = 10): Promise<QueryResult[][]> {
       const promises = queries.map(query => 
         this.cortex.query(query, maxResultsPerQuery)
       );

       return Promise.all(promises);
     }
   }
   ```

## Expected Outputs
- Complete CortexKGWeb class with full API coverage
- Performance monitoring and metrics collection
- Batch operation support for multiple concepts/queries
- Proper memory management and cleanup
- Type-safe interfaces with comprehensive error handling

## Validation
1. All WASM functions are properly wrapped with type safety
2. Promise-based API works correctly with async/await
3. Memory allocation and deallocation work without leaks
4. Performance metrics are accurately collected and reported
5. Batch operations handle large datasets efficiently

## Next Steps
- Implement promise-based interfaces (micro-phase 9.19)
- Add comprehensive error handling (micro-phase 9.20)