# Micro-Phase 9.19: Promise-Based Interface Implementation

## Objective
Implement comprehensive promise-based interfaces with proper async handling, cancellation support, and timeout management for all CortexKG operations.

## Prerequisites
- Completed micro-phase 9.18 (JS API wrapper)
- WASM module fully operational
- Performance monitoring system in place

## Task Description
Create advanced promise-based interfaces that provide cancellation, timeout, retry mechanisms, and proper async error handling for all CortexKG operations to ensure robust web application integration.

## Specific Actions

1. **Create promise wrapper utilities**:
   ```typescript
   // src/core/PromiseManager.ts
   export interface PromiseOptions {
     timeout?: number;
     retries?: number;
     retryDelay?: number;
     signal?: AbortSignal;
     onProgress?: (progress: ProgressEvent) => void;
   }

   export interface ProgressEvent {
     type: 'start' | 'progress' | 'complete' | 'error';
     progress: number; // 0-1
     message?: string;
     data?: any;
   }

   export interface CancellablePromise<T> extends Promise<T> {
     cancel: () => void;
     timeout: (ms: number) => CancellablePromise<T>;
     retry: (attempts: number, delay?: number) => CancellablePromise<T>;
   }

   export class PromiseManager {
     private activePromises: Set<CancellablePromise<any>> = new Set();
     private defaultTimeout: number = 30000; // 30 seconds

     createCancellablePromise<T>(
       executor: (
         resolve: (value: T) => void,
         reject: (reason?: any) => void,
         onCancel: (callback: () => void) => void,
         reportProgress: (event: ProgressEvent) => void
       ) => void,
       options: PromiseOptions = {}
     ): CancellablePromise<T> {
       let cancelled = false;
       let cancelCallbacks: (() => void)[] = [];
       let timeoutId: NodeJS.Timeout | null = null;

       const promise = new Promise<T>((resolve, reject) => {
         const onCancel = (callback: () => void) => {
           cancelCallbacks.push(callback);
         };

         const reportProgress = (event: ProgressEvent) => {
           if (options.onProgress && !cancelled) {
             options.onProgress(event);
           }
         };

         // Handle external abort signal
         if (options.signal) {
           options.signal.addEventListener('abort', () => {
             if (!cancelled) {
               cancelled = true;
               cancelCallbacks.forEach(cb => cb());
               reject(new Error('Operation was aborted'));
             }
           });
         }

         // Set up timeout
         const timeout = options.timeout || this.defaultTimeout;
         if (timeout > 0) {
           timeoutId = setTimeout(() => {
             if (!cancelled) {
               cancelled = true;
               cancelCallbacks.forEach(cb => cb());
               reject(new Error(`Operation timed out after ${timeout}ms`));
             }
           }, timeout);
         }

         // Execute the main function
         try {
           executor(
             (value) => {
               if (!cancelled) {
                 if (timeoutId) clearTimeout(timeoutId);
                 resolve(value);
               }
             },
             (reason) => {
               if (!cancelled) {
                 if (timeoutId) clearTimeout(timeoutId);
                 reject(reason);
               }
             },
             onCancel,
             reportProgress
           );
         } catch (error) {
           if (!cancelled) {
             if (timeoutId) clearTimeout(timeoutId);
             reject(error);
           }
         }
       }) as CancellablePromise<T>;

       // Add cancellation method
       promise.cancel = () => {
         if (!cancelled) {
           cancelled = true;
           if (timeoutId) clearTimeout(timeoutId);
           cancelCallbacks.forEach(cb => cb());
           this.activePromises.delete(promise);
         }
       };

       // Add timeout method
       promise.timeout = (ms: number) => {
         return this.createCancellablePromise(executor, { ...options, timeout: ms });
       };

       // Add retry method
       promise.retry = (attempts: number, delay: number = 1000) => {
         return this.createRetryablePromise(executor, attempts, delay, options);
       };

       this.activePromises.add(promise);

       // Clean up when promise completes
       promise.finally(() => {
         this.activePromises.delete(promise);
       });

       return promise;
     }

     private createRetryablePromise<T>(
       executor: any,
       maxAttempts: number,
       delay: number,
       options: PromiseOptions
     ): CancellablePromise<T> {
       let attempt = 0;

       const tryOperation = (): CancellablePromise<T> => {
         attempt++;
         
         return this.createCancellablePromise<T>((resolve, reject, onCancel, reportProgress) => {
           reportProgress({
             type: 'progress',
             progress: attempt / maxAttempts,
             message: `Attempt ${attempt} of ${maxAttempts}`
           });

           executor(
             resolve,
             (error: any) => {
               if (attempt < maxAttempts) {
                 setTimeout(() => {
                   tryOperation().then(resolve).catch(reject);
                 }, delay);
               } else {
                 reject(new Error(`Operation failed after ${maxAttempts} attempts: ${error.message}`));
               }
             },
             onCancel,
             reportProgress
           );
         }, options);
       };

       return tryOperation();
     }

     async all<T>(promises: CancellablePromise<T>[]): Promise<T[]> {
       try {
         return await Promise.all(promises);
       } catch (error) {
         // Cancel all remaining promises if one fails
         promises.forEach(p => p.cancel());
         throw error;
       }
     }

     async race<T>(promises: CancellablePromise<T>[]): Promise<T> {
       try {
         return await Promise.race(promises);
       } finally {
         // Cancel all remaining promises
         promises.forEach(p => p.cancel());
       }
     }

     cancelAll(): void {
       this.activePromises.forEach(promise => promise.cancel());
       this.activePromises.clear();
     }

     getActiveCount(): number {
       return this.activePromises.size;
     }
   }
   ```

2. **Create async API wrapper**:
   ```typescript
   // src/core/AsyncCortexKG.ts
   import { CortexKGWeb, CortexKGWebConfig } from './CortexKGWeb';
   import { PromiseManager, CancellablePromise, PromiseOptions } from './PromiseManager';
   import { AllocationResult, QueryResult, PerformanceMetrics } from '../types/cortexkg';

   export interface AsyncAllocationOptions extends PromiseOptions {
     priority?: 'low' | 'normal' | 'high';
     batch?: boolean;
   }

   export interface AsyncQueryOptions extends PromiseOptions {
     maxResults?: number;
     threshold?: number;
     streaming?: boolean;
   }

   export interface StreamingQueryResult {
     results: AsyncIterableIterator<QueryResult>;
     total: Promise<number>;
     cancel: () => void;
   }

   export class AsyncCortexKG {
     private cortex: CortexKGWeb;
     private promiseManager: PromiseManager;
     private operationQueue: Map<string, CancellablePromise<any>> = new Map();

     constructor(config: CortexKGWebConfig = {}) {
       this.cortex = new CortexKGWeb(config);
       this.promiseManager = new PromiseManager();
     }

     async initialize(options: PromiseOptions = {}): Promise<void> {
       return this.promiseManager.createCancellablePromise(
         async (resolve, reject, onCancel, reportProgress) => {
           reportProgress({ type: 'start', progress: 0, message: 'Initializing CortexKG...' });

           onCancel(() => {
             // Cannot really cancel WASM initialization, but we can abort the promise
           });

           try {
             await this.cortex.initialize();
             reportProgress({ type: 'complete', progress: 1, message: 'Initialization complete' });
             resolve();
           } catch (error) {
             reportProgress({ type: 'error', progress: 0, message: error.message });
             reject(error);
           }
         },
         options
       );
     }

     allocateConcept(
       content: string, 
       options: AsyncAllocationOptions = {}
     ): CancellablePromise<AllocationResult> {
       const operationId = `allocate_${Date.now()}_${Math.random()}`;

       const promise = this.promiseManager.createCancellablePromise<AllocationResult>(
         async (resolve, reject, onCancel, reportProgress) => {
           reportProgress({ 
             type: 'start', 
             progress: 0, 
             message: `Allocating concept: ${content.substring(0, 50)}...` 
           });

           let cancelled = false;
           onCancel(() => {
             cancelled = true;
           });

           try {
             // Simulate progress for longer operations
             const progressInterval = setInterval(() => {
               if (!cancelled) {
                 reportProgress({ 
                   type: 'progress', 
                   progress: 0.5, 
                   message: 'Processing allocation...' 
                 });
               }
             }, 100);

             onCancel(() => clearInterval(progressInterval));

             const result = await this.cortex.allocateConcept(content);
             
             clearInterval(progressInterval);

             if (cancelled) {
               reject(new Error('Operation was cancelled'));
               return;
             }

             reportProgress({ 
               type: 'complete', 
               progress: 1, 
               message: `Allocated to column ${result.column_id}` 
             });

             resolve(result);
           } catch (error) {
             if (!cancelled) {
               reportProgress({ type: 'error', progress: 0, message: error.message });
               reject(error);
             }
           } finally {
             this.operationQueue.delete(operationId);
           }
         },
         options
       );

       this.operationQueue.set(operationId, promise);
       return promise;
     }

     query(
       queryText: string, 
       options: AsyncQueryOptions = {}
     ): CancellablePromise<QueryResult[]> {
       const operationId = `query_${Date.now()}_${Math.random()}`;

       const promise = this.promiseManager.createCancellablePromise<QueryResult[]>(
         async (resolve, reject, onCancel, reportProgress) => {
           reportProgress({ 
             type: 'start', 
             progress: 0, 
             message: `Querying: ${queryText.substring(0, 50)}...` 
           });

           let cancelled = false;
           onCancel(() => {
             cancelled = true;
           });

           try {
             const maxResults = options.maxResults || 10;
             const results = await this.cortex.query(queryText, maxResults);
             
             if (cancelled) {
               reject(new Error('Query was cancelled'));
               return;
             }

             // Filter by threshold if specified
             const filteredResults = options.threshold 
               ? results.filter(r => r.relevance_score >= options.threshold!)
               : results;

             reportProgress({ 
               type: 'complete', 
               progress: 1, 
               message: `Found ${filteredResults.length} results` 
             });

             resolve(filteredResults);
           } catch (error) {
             if (!cancelled) {
               reportProgress({ type: 'error', progress: 0, message: error.message });
               reject(error);
             }
           } finally {
             this.operationQueue.delete(operationId);
           }
         },
         options
       );

       this.operationQueue.set(operationId, promise);
       return promise;
     }

     streamingQuery(
       queryText: string, 
       options: AsyncQueryOptions = {}
     ): StreamingQueryResult {
       let cancelled = false;
       const results: QueryResult[] = [];

       const asyncIterator = async function* (this: AsyncCortexKG) {
         const batchSize = 5;
         let offset = 0;

         while (!cancelled) {
           try {
             // Simulate streaming by querying in batches
             const batch = await this.query(queryText, {
               ...options,
               maxResults: batchSize
             });

             if (batch.length === 0) break;

             for (const result of batch) {
               if (cancelled) return;
               results.push(result);
               yield result;
             }

             offset += batchSize;
             if (batch.length < batchSize) break;

             // Small delay between batches
             await new Promise(resolve => setTimeout(resolve, 10));
           } catch (error) {
             throw error;
           }
         }
       }.bind(this);

       const totalPromise = new Promise<number>((resolve, reject) => {
         (async () => {
           try {
             for await (const _ of asyncIterator()) {
               // Consume the iterator to get total count
             }
             resolve(results.length);
           } catch (error) {
             reject(error);
           }
         })();
       });

       return {
         results: asyncIterator(),
         total: totalPromise,
         cancel: () => {
           cancelled = true;
         }
       };
     }

     async batchAllocate(
       concepts: string[], 
       options: AsyncAllocationOptions = {}
     ): Promise<AllocationResult[]> {
       const batchPromises = concepts.map(concept => 
         this.allocateConcept(concept, { ...options, batch: true })
       );

       return this.promiseManager.all(batchPromises);
     }

     async getPerformanceMetrics(options: PromiseOptions = {}): Promise<PerformanceMetrics> {
       return this.promiseManager.createCancellablePromise(
         async (resolve, reject) => {
           try {
             const metrics = await this.cortex.getPerformanceMetrics();
             resolve(metrics);
           } catch (error) {
             reject(error);
           }
         },
         options
       );
     }

     cancelOperation(operationId?: string): void {
       if (operationId && this.operationQueue.has(operationId)) {
         this.operationQueue.get(operationId)!.cancel();
       } else {
         this.promiseManager.cancelAll();
       }
     }

     getActiveOperationCount(): number {
       return this.promiseManager.getActiveCount();
     }

     async dispose(): Promise<void> {
       this.promiseManager.cancelAll();
       this.operationQueue.clear();
       this.cortex.dispose();
     }

     // Getter to access the underlying cortex for advanced operations
     get underlying(): CortexKGWeb {
       return this.cortex;
     }
   }
   ```

3. **Create queue management system**:
   ```typescript
   // src/core/OperationQueue.ts
   export interface QueuedOperation<T> {
     id: string;
     operation: () => CancellablePromise<T>;
     priority: number;
     timeout: number;
     retries: number;
     createdAt: Date;
   }

   export interface QueueStats {
     pending: number;
     running: number;
     completed: number;
     failed: number;
     avgWaitTime: number;
     avgExecutionTime: number;
   }

   export class OperationQueue {
     private queue: QueuedOperation<any>[] = [];
     private running: Map<string, CancellablePromise<any>> = new Map();
     private completed: Map<string, { success: boolean; duration: number }> = new Map();
     private maxConcurrent: number = 5;
     private isProcessing: boolean = false;

     constructor(maxConcurrent: number = 5) {
       this.maxConcurrent = maxConcurrent;
     }

     enqueue<T>(operation: QueuedOperation<T>): Promise<T> {
       return new Promise((resolve, reject) => {
         const queuedOp = {
           ...operation,
           operation: () => {
             const promise = operation.operation();
             promise.then(resolve).catch(reject);
             return promise;
           }
         };

         // Insert in priority order
         const insertIndex = this.queue.findIndex(op => op.priority < queuedOp.priority);
         if (insertIndex === -1) {
           this.queue.push(queuedOp);
         } else {
           this.queue.splice(insertIndex, 0, queuedOp);
         }

         this.processQueue();
       });
     }

     private async processQueue(): Promise<void> {
       if (this.isProcessing || this.running.size >= this.maxConcurrent) {
         return;
       }

       this.isProcessing = true;

       while (this.queue.length > 0 && this.running.size < this.maxConcurrent) {
         const operation = this.queue.shift()!;
         const startTime = performance.now();

         try {
           const promise = operation.operation();
           this.running.set(operation.id, promise);

           promise
             .then(() => {
               this.completed.set(operation.id, {
                 success: true,
                 duration: performance.now() - startTime
               });
             })
             .catch(() => {
               this.completed.set(operation.id, {
                 success: false,
                 duration: performance.now() - startTime
               });
             })
             .finally(() => {
               this.running.delete(operation.id);
               this.processQueue(); // Process next operations
             });

         } catch (error) {
           this.completed.set(operation.id, {
             success: false,
             duration: performance.now() - startTime
           });
         }
       }

       this.isProcessing = false;
     }

     cancel(operationId: string): boolean {
       // Remove from queue if not started
       const queueIndex = this.queue.findIndex(op => op.id === operationId);
       if (queueIndex !== -1) {
         this.queue.splice(queueIndex, 1);
         return true;
       }

       // Cancel if running
       const runningOp = this.running.get(operationId);
       if (runningOp) {
         runningOp.cancel();
         return true;
       }

       return false;
     }

     getStats(): QueueStats {
       const completedOps = Array.from(this.completed.values());
       const successful = completedOps.filter(op => op.success);
       const failed = completedOps.filter(op => !op.success);

       return {
         pending: this.queue.length,
         running: this.running.size,
         completed: successful.length,
         failed: failed.length,
         avgWaitTime: 0, // TODO: Calculate based on queue times
         avgExecutionTime: successful.reduce((sum, op) => sum + op.duration, 0) / successful.length || 0
       };
     }

     clear(): void {
       this.queue.length = 0;
       this.running.forEach(promise => promise.cancel());
       this.running.clear();
     }
   }
   ```

## Expected Outputs
- Comprehensive promise-based API with cancellation support
- Progress reporting and timeout management
- Retry mechanisms with exponential backoff
- Streaming query interface for large result sets
- Operation queue management with priority handling

## Validation
1. All operations can be cancelled gracefully without memory leaks
2. Timeout mechanisms work correctly and don't leave hanging operations
3. Progress reporting provides meaningful feedback during long operations
4. Retry logic handles transient failures appropriately
5. Batch operations maintain proper error isolation

## Next Steps
- Add comprehensive error handling (micro-phase 9.20)
- Create HTML structure for web interface (micro-phase 9.21)