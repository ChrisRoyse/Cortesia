/**
 * Performance utility functions for Phase 5 visualization
 */

/**
 * Performance monitor class for tracking rendering metrics
 */
export class PerformanceMonitor {
  private frameCount: number = 0;
  private lastTime: number = performance.now();
  private fps: number = 0;
  private frameTimes: number[] = [];
  private maxSamples: number = 60;
  
  /**
   * Updates the performance metrics
   */
  update(): void {
    const currentTime = performance.now();
    const deltaTime = currentTime - this.lastTime;
    
    this.frameTimes.push(deltaTime);
    if (this.frameTimes.length > this.maxSamples) {
      this.frameTimes.shift();
    }
    
    this.frameCount++;
    
    // Calculate FPS every second
    if (this.frameCount % 60 === 0) {
      const avgFrameTime = this.frameTimes.reduce((a, b) => a + b, 0) / this.frameTimes.length;
      this.fps = 1000 / avgFrameTime;
    }
    
    this.lastTime = currentTime;
  }
  
  /**
   * Gets the current FPS
   */
  getFPS(): number {
    return Math.round(this.fps);
  }
  
  /**
   * Gets the average frame time
   */
  getAverageFrameTime(): number {
    if (this.frameTimes.length === 0) return 0;
    return this.frameTimes.reduce((a, b) => a + b, 0) / this.frameTimes.length;
  }
  
  /**
   * Resets the performance monitor
   */
  reset(): void {
    this.frameCount = 0;
    this.lastTime = performance.now();
    this.fps = 0;
    this.frameTimes = [];
  }
}

/**
 * Throttles a function to run at most once per specified interval
 * @param func - Function to throttle
 * @param limit - Time limit in milliseconds
 * @returns Throttled function
 */
export function throttle<T extends (...args: any[]) => any>(
  func: T,
  limit: number
): (...args: Parameters<T>) => void {
  let inThrottle: boolean = false;
  let lastResult: ReturnType<T>;
  
  return function(this: any, ...args: Parameters<T>) {
    if (!inThrottle) {
      lastResult = func.apply(this, args);
      inThrottle = true;
      setTimeout(() => {
        inThrottle = false;
      }, limit);
    }
    return lastResult;
  };
}

/**
 * Debounces a function to run after a specified delay
 * @param func - Function to debounce
 * @param wait - Wait time in milliseconds
 * @param immediate - Execute immediately on first call
 * @returns Debounced function
 */
export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number,
  immediate: boolean = false
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout | null = null;
  
  return function(this: any, ...args: Parameters<T>) {
    const later = () => {
      timeout = null;
      if (!immediate) func.apply(this, args);
    };
    
    const callNow = immediate && !timeout;
    
    if (timeout) clearTimeout(timeout);
    timeout = setTimeout(later, wait);
    
    if (callNow) func.apply(this, args);
  };
}

/**
 * Measures the execution time of a function
 * @param name - Name for the measurement
 * @param func - Function to measure
 * @returns Function result
 */
export function measureTime<T>(name: string, func: () => T): T {
  const start = performance.now();
  const result = func();
  const end = performance.now();
  console.log(`${name} took ${(end - start).toFixed(2)}ms`);
  return result;
}

/**
 * Creates a memoized version of a function
 * @param func - Function to memoize
 * @param keyGenerator - Optional function to generate cache key
 * @returns Memoized function
 */
export function memoize<T extends (...args: any[]) => any>(
  func: T,
  keyGenerator?: (...args: Parameters<T>) => string
): T {
  const cache = new Map<string, ReturnType<T>>();
  
  return function(this: any, ...args: Parameters<T>): ReturnType<T> {
    const key = keyGenerator ? keyGenerator(...args) : JSON.stringify(args);
    
    if (cache.has(key)) {
      return cache.get(key)!;
    }
    
    const result = func.apply(this, args);
    cache.set(key, result);
    
    // Limit cache size
    if (cache.size > 100) {
      const firstKey = cache.keys().next().value;
      cache.delete(firstKey);
    }
    
    return result;
  } as T;
}

/**
 * Batches multiple calls into a single execution
 * @param func - Function to batch
 * @param wait - Wait time before execution
 * @returns Batched function
 */
export function batch<T extends (items: any[]) => void>(
  func: T,
  wait: number = 0
): (item: any) => void {
  let items: any[] = [];
  let timeout: NodeJS.Timeout | null = null;
  
  const flush = () => {
    if (items.length > 0) {
      func(items);
      items = [];
    }
    timeout = null;
  };
  
  return (item: any) => {
    items.push(item);
    
    if (timeout) clearTimeout(timeout);
    
    if (wait === 0) {
      Promise.resolve().then(flush);
    } else {
      timeout = setTimeout(flush, wait);
    }
  };
}

/**
 * Creates a worker pool for parallel processing
 */
export class WorkerPool {
  private workers: Worker[] = [];
  private queue: Array<{ data: any; resolve: (value: any) => void; reject: (error: any) => void }> = [];
  private busyWorkers = new Set<Worker>();
  
  constructor(workerScript: string, poolSize: number = navigator.hardwareConcurrency || 4) {
    for (let i = 0; i < poolSize; i++) {
      const worker = new Worker(workerScript);
      worker.onmessage = (e) => this.handleWorkerMessage(worker, e);
      worker.onerror = (e) => this.handleWorkerError(worker, e);
      this.workers.push(worker);
    }
  }
  
  /**
   * Executes a task in the worker pool
   * @param data - Data to process
   * @returns Promise with result
   */
  execute(data: any): Promise<any> {
    return new Promise((resolve, reject) => {
      const availableWorker = this.workers.find(w => !this.busyWorkers.has(w));
      
      if (availableWorker) {
        this.busyWorkers.add(availableWorker);
        availableWorker.postMessage(data);
        this.queue.push({ data, resolve, reject });
      } else {
        this.queue.push({ data, resolve, reject });
      }
    });
  }
  
  private handleWorkerMessage(worker: Worker, event: MessageEvent): void {
    const task = this.queue.shift();
    if (task) {
      task.resolve(event.data);
    }
    
    this.busyWorkers.delete(worker);
    
    // Process next task in queue
    const nextTask = this.queue.find(t => !this.busyWorkers.has(worker));
    if (nextTask) {
      this.busyWorkers.add(worker);
      worker.postMessage(nextTask.data);
    }
  }
  
  private handleWorkerError(worker: Worker, error: ErrorEvent): void {
    const task = this.queue.shift();
    if (task) {
      task.reject(error);
    }
    this.busyWorkers.delete(worker);
  }
  
  /**
   * Terminates all workers
   */
  terminate(): void {
    this.workers.forEach(worker => worker.terminate());
    this.workers = [];
    this.queue = [];
    this.busyWorkers.clear();
  }
}

/**
 * Implements a least-recently-used cache
 */
export class LRUCache<K, V> {
  private capacity: number;
  private cache = new Map<K, V>();
  
  constructor(capacity: number) {
    this.capacity = capacity;
  }
  
  get(key: K): V | undefined {
    const value = this.cache.get(key);
    if (value !== undefined) {
      // Move to end (most recently used)
      this.cache.delete(key);
      this.cache.set(key, value);
    }
    return value;
  }
  
  set(key: K, value: V): void {
    if (this.cache.has(key)) {
      this.cache.delete(key);
    } else if (this.cache.size >= this.capacity) {
      // Remove least recently used (first item)
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }
    this.cache.set(key, value);
  }
  
  clear(): void {
    this.cache.clear();
  }
  
  size(): number {
    return this.cache.size;
  }
}

/**
 * Detects if the browser supports WebGL
 */
export function isWebGLSupported(): boolean {
  try {
    const canvas = document.createElement('canvas');
    return !!(
      window.WebGLRenderingContext &&
      (canvas.getContext('webgl') || canvas.getContext('experimental-webgl'))
    );
  } catch (e) {
    return false;
  }
}

/**
 * Gets memory usage information (if available)
 */
export function getMemoryInfo(): { used: number; total: number } | null {
  if ('memory' in performance) {
    const memory = (performance as any).memory;
    return {
      used: memory.usedJSHeapSize,
      total: memory.totalJSHeapSize
    };
  }
  return null;
}

/**
 * Optimizes a function for animation frames
 * @param func - Function to optimize
 * @returns Optimized function
 */
export function optimizeForAnimation<T extends (...args: any[]) => void>(
  func: T
): (...args: Parameters<T>) => void {
  let ticking = false;
  let args: Parameters<T> | null = null;
  
  return function(...newArgs: Parameters<T>) {
    args = newArgs;
    
    if (!ticking) {
      requestAnimationFrame(() => {
        if (args) {
          func(...args);
          args = null;
        }
        ticking = false;
      });
      ticking = true;
    }
  };
}