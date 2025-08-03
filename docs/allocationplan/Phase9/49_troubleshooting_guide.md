# Micro-Phase 9.49: Troubleshooting Guide

## Objective
Write comprehensive troubleshooting guide covering common issues and solutions that developers might encounter when integrating and using CortexKG in their applications.

## Prerequisites
- Completed micro-phase 9.48 (Example Apps)
- All WASM implementation components tested and documented
- API documentation and integration guides available

## Task Description
Create detailed troubleshooting documentation covering installation issues, runtime errors, performance problems, browser compatibility issues, and debugging techniques. Provide systematic approaches to diagnosing and resolving common problems.

## Specific Actions

1. **Create comprehensive troubleshooting guide structure**
```markdown
# CortexKG Troubleshooting Guide

## Table of Contents
1. [Quick Diagnostics](#quick-diagnostics)
2. [Installation Issues](#installation-issues)
3. [WASM Loading Problems](#wasm-loading-problems)
4. [Memory Management Issues](#memory-management-issues)
5. [Performance Problems](#performance-problems)
6. [Browser Compatibility](#browser-compatibility)
7. [API Usage Errors](#api-usage-errors)
8. [Debugging Tools](#debugging-tools)
9. [Common Error Messages](#common-error-messages)
10. [Advanced Troubleshooting](#advanced-troubleshooting)

## Quick Diagnostics

### System Health Check
Run this diagnostic script to quickly identify common issues:

```javascript
// CortexKG Health Check Script
async function performHealthCheck() {
    const results = {
        browser: getBrowserInfo(),
        webAssembly: await checkWebAssemblySupport(),
        memory: checkMemoryCapabilities(),
        storage: await checkStorageSupport(),
        cortexKG: await checkCortexKGStatus()
    };
    
    console.table(results);
    return results;
}

function getBrowserInfo() {
    const ua = navigator.userAgent;
    const browser = {
        name: 'Unknown',
        version: 'Unknown',
        mobile: /Mobi|Android/i.test(ua)
    };
    
    if (ua.includes('Chrome')) {
        browser.name = 'Chrome';
        browser.version = ua.match(/Chrome\/([0-9.]+)/)[1];
    } else if (ua.includes('Firefox')) {
        browser.name = 'Firefox';
        browser.version = ua.match(/Firefox\/([0-9.]+)/)[1];
    } else if (ua.includes('Safari') && !ua.includes('Chrome')) {
        browser.name = 'Safari';
        browser.version = ua.match(/Version\/([0-9.]+)/)[1];
    } else if (ua.includes('Edge')) {
        browser.name = 'Edge';
        browser.version = ua.match(/Edge\/([0-9.]+)/)[1];
    }
    
    return browser;
}

async function checkWebAssemblySupport() {
    const support = {
        basic: typeof WebAssembly !== 'undefined',
        streaming: typeof WebAssembly.instantiateStreaming !== 'undefined',
        simd: false,
        threads: false,
        bulkMemory: false
    };
    
    if (support.basic) {
        // Test SIMD support
        try {
            const simdWasm = new Uint8Array([
                0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
                0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7b,
                0x03, 0x02, 0x01, 0x00,
                0x0a, 0x0a, 0x01, 0x08, 0x00, 0xfd, 0x0c, 0x00, 0x00, 0x00, 0x0b
            ]);
            await WebAssembly.validate(simdWasm);
            support.simd = true;
        } catch (e) {
            support.simd = false;
        }
        
        // Test threads support
        try {
            const threadsWasm = new Uint8Array([
                0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
                0x05, 0x04, 0x01, 0x03, 0x01, 0x01,
                0x0a, 0x09, 0x01, 0x07, 0x00, 0xfe, 0x10, 0x02, 0x00, 0x0b
            ]);
            await WebAssembly.validate(threadsWasm);
            support.threads = true;
        } catch (e) {
            support.threads = false;
        }
        
        // Test bulk memory operations
        try {
            const bulkWasm = new Uint8Array([
                0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
                0x05, 0x03, 0x01, 0x00, 0x01,
                0x0a, 0x09, 0x01, 0x07, 0x00, 0xfc, 0x0a, 0x00, 0x00, 0x0b
            ]);
            await WebAssembly.validate(bulkWasm);
            support.bulkMemory = true;
        } catch (e) {
            support.bulkMemory = false;
        }
    }
    
    return support;
}

function checkMemoryCapabilities() {
    const memory = {
        deviceMemory: navigator.deviceMemory || 'Unknown',
        hardwareConcurrency: navigator.hardwareConcurrency || 'Unknown',
        maxHeapSize: 'Unknown'
    };
    
    // Estimate max heap size
    try {
        let size = 16;
        while (size <= 1024) {
            try {
                new ArrayBuffer(size * 1024 * 1024);
                memory.maxHeapSize = `${size}MB+`;
                size *= 2;
            } catch (e) {
                break;
            }
        }
    } catch (e) {
        memory.maxHeapSize = 'Limited';
    }
    
    return memory;
}

async function checkStorageSupport() {
    const storage = {
        indexedDB: typeof indexedDB !== 'undefined',
        localStorage: typeof localStorage !== 'undefined',
        sessionStorage: typeof sessionStorage !== 'undefined',
        quotaEstimate: null
    };
    
    if ('storage' in navigator && 'estimate' in navigator.storage) {
        try {
            const estimate = await navigator.storage.estimate();
            storage.quotaEstimate = {
                usage: Math.round(estimate.usage / (1024 * 1024)),
                quota: Math.round(estimate.quota / (1024 * 1024))
            };
        } catch (e) {
            storage.quotaEstimate = 'Unavailable';
        }
    }
    
    return storage;
}

async function checkCortexKGStatus() {
    const status = {
        loaded: false,
        initialized: false,
        version: 'Unknown',
        memoryUsage: null,
        lastError: null
    };
    
    try {
        if (typeof wasm_loader !== 'undefined') {
            status.loaded = true;
            
            if (wasm_loader.isInitialized && wasm_loader.isInitialized()) {
                status.initialized = true;
                status.memoryUsage = wasm_loader.getMemoryUsage();
            }
            
            if (wasm_loader.getVersion) {
                status.version = wasm_loader.getVersion();
            }
        }
    } catch (error) {
        status.lastError = error.message;
    }
    
    return status;
}

// Run health check
performHealthCheck().then(results => {
    console.log('🔍 CortexKG Health Check Complete');
    
    // Provide recommendations based on results
    const recommendations = [];
    
    if (!results.webAssembly.basic) {
        recommendations.push('❌ WebAssembly not supported - upgrade browser');
    }
    
    if (!results.webAssembly.simd) {
        recommendations.push('⚠️ SIMD not supported - performance may be reduced');
    }
    
    if (results.memory.deviceMemory && results.memory.deviceMemory < 2) {
        recommendations.push('⚠️ Low device memory detected - reduce memory allocation');
    }
    
    if (!results.storage.indexedDB) {
        recommendations.push('❌ IndexedDB not supported - persistent storage unavailable');
    }
    
    if (!results.cortexKG.loaded) {
        recommendations.push('❌ CortexKG not loaded - check script inclusion');
    }
    
    if (results.cortexKG.loaded && !results.cortexKG.initialized) {
        recommendations.push('⚠️ CortexKG not initialized - call wasm_loader.init()');
    }
    
    if (recommendations.length > 0) {
        console.log('📋 Recommendations:');
        recommendations.forEach(rec => console.log(rec));
    } else {
        console.log('✅ All systems operational');
    }
});
```

## Installation Issues

### Problem: CortexKG files not loading

**Symptoms:**
- "Failed to fetch" errors in console
- 404 errors for WASM or JS files
- Scripts not executing

**Solutions:**

1. **Check file paths:**
```javascript
// Verify file structure
/*
your-project/
├── index.html
├── js/
│   ├── cortex-kg.js
│   └── cortex-kg.wasm
└── your-app.js
*/

// Correct script inclusion
<script src="js/cortex-kg.js"></script>
```

2. **Configure web server for WASM:**
```nginx
# Nginx configuration
location ~* \.wasm$ {
    add_header Content-Type application/wasm;
    add_header Cross-Origin-Embedder-Policy require-corp;
    add_header Cross-Origin-Opener-Policy same-origin;
}
```

```apache
# Apache configuration
<Files "*.wasm">
    Header set Content-Type "application/wasm"
    Header set Cross-Origin-Embedder-Policy "require-corp"
    Header set Cross-Origin-Opener-Policy "same-origin"
</Files>
```

3. **Local development server setup:**
```bash
# Python HTTP server with WASM support
python3 -m http.server 8000

# Node.js with Express
npm install express
node -e "
const express = require('express');
const app = express();
app.use(express.static('.', {
  setHeaders: (res, path) => {
    if (path.endsWith('.wasm')) {
      res.setHeader('Content-Type', 'application/wasm');
    }
  }
}));
app.listen(8000, () => console.log('Server running on http://localhost:8000'));
"
```

### Problem: CDN loading failures

**Symptoms:**
- Intermittent loading failures
- Slow loading times
- Version mismatch errors

**Solutions:**

1. **Use specific version pins:**
```html
<!-- Instead of @latest -->
<script src="https://cdn.jsdelivr.net/npm/cortex-kg@1.2.3/dist/cortex-kg.min.js"></script>

<!-- With integrity check -->
<script src="https://cdn.jsdelivr.net/npm/cortex-kg@1.2.3/dist/cortex-kg.min.js" 
        integrity="sha384-..." 
        crossorigin="anonymous"></script>
```

2. **Implement fallback loading:**
```javascript
async function loadCortexKG() {
    const cdnUrls = [
        'https://cdn.jsdelivr.net/npm/cortex-kg@1.2.3/dist/cortex-kg.min.js',
        'https://unpkg.com/cortex-kg@1.2.3/dist/cortex-kg.min.js',
        './js/cortex-kg.min.js' // Local fallback
    ];
    
    for (const url of cdnUrls) {
        try {
            await loadScript(url);
            console.log(`✅ Loaded CortexKG from: ${url}`);
            return;
        } catch (error) {
            console.warn(`⚠️ Failed to load from ${url}:`, error);
        }
    }
    
    throw new Error('Failed to load CortexKG from all sources');
}

function loadScript(url) {
    return new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = url;
        script.onload = resolve;
        script.onerror = reject;
        script.timeout = 10000; // 10 second timeout
        document.head.appendChild(script);
    });
}
```

## WASM Loading Problems

### Problem: WebAssembly initialization fails

**Symptoms:**
- "WebAssembly.instantiate(): Invalid module" errors
- "CompileError" or "LinkError" messages
- Initialization timeout

**Solutions:**

1. **Check WASM file integrity:**
```javascript
async function verifyWasmFile(url) {
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const arrayBuffer = await response.arrayBuffer();
        const uint8Array = new Uint8Array(arrayBuffer);
        
        // Check WASM magic number
        const magicNumber = [0x00, 0x61, 0x73, 0x6d];
        for (let i = 0; i < 4; i++) {
            if (uint8Array[i] !== magicNumber[i]) {
                throw new Error('Invalid WASM file: Magic number mismatch');
            }
        }
        
        // Validate WASM module
        const isValid = await WebAssembly.validate(arrayBuffer);
        if (!isValid) {
            throw new Error('Invalid WASM module');
        }
        
        console.log('✅ WASM file verified successfully');
        return arrayBuffer;
        
    } catch (error) {
        console.error('❌ WASM verification failed:', error);
        throw error;
    }
}

// Usage
verifyWasmFile('./js/cortex-kg.wasm')
    .then(() => wasm_loader.init())
    .catch(error => console.error('Initialization failed:', error));
```

2. **Implement progressive loading:**
```javascript
async function initializeWithFallback() {
    const configs = [
        { memorySize: 64 * 1024 * 1024, enableSIMD: true, enableThreads: true },
        { memorySize: 32 * 1024 * 1024, enableSIMD: true, enableThreads: false },
        { memorySize: 16 * 1024 * 1024, enableSIMD: false, enableThreads: false },
        { memorySize: 8 * 1024 * 1024, enableSIMD: false, enableThreads: false }
    ];
    
    for (const config of configs) {
        try {
            console.log('Attempting initialization with config:', config);
            await wasm_loader.init(config);
            console.log('✅ Initialization successful');
            return;
        } catch (error) {
            console.warn('⚠️ Config failed:', config, error.message);
        }
    }
    
    throw new Error('All initialization configurations failed');
}
```

3. **Handle browser-specific issues:**
```javascript
function getBrowserSpecificConfig() {
    const ua = navigator.userAgent;
    const config = {
        memorySize: 32 * 1024 * 1024,
        enableSIMD: true,
        enableThreads: false
    };
    
    if (ua.includes('Safari') && !ua.includes('Chrome')) {
        // Safari limitations
        config.enableSIMD = false;
        config.enableThreads = false;
        config.memorySize = 16 * 1024 * 1024;
    } else if (ua.includes('Firefox')) {
        // Firefox optimizations
        config.enableThreads = true;
    } else if (/Mobi|Android/i.test(ua)) {
        // Mobile limitations
        config.memorySize = 16 * 1024 * 1024;
        config.enableSIMD = false;
    }
    
    return config;
}

// Usage
const config = getBrowserSpecificConfig();
await wasm_loader.init(config);
```

### Problem: Memory allocation errors

**Symptoms:**
- "RangeError: Maximum call stack size exceeded"
- "Out of memory" errors
- Slow performance or crashes

**Solutions:**

1. **Implement memory monitoring:**
```javascript
class MemoryMonitor {
    constructor(threshold = 0.8) {
        this.threshold = threshold;
        this.monitoring = false;
        this.cleanup handlers = [];
    }
    
    startMonitoring() {
        if (this.monitoring) return;
        
        this.monitoring = true;
        this.monitorInterval = setInterval(() => {
            this.checkMemoryPressure();
        }, 5000);
    }
    
    stopMonitoring() {
        this.monitoring = false;
        if (this.monitorInterval) {
            clearInterval(this.monitorInterval);
        }
    }
    
    checkMemoryPressure() {
        try {
            const stats = wasm_loader.getMemoryUsage();
            
            if (stats.pressure > this.threshold) {
                console.warn(`⚠️ High memory pressure: ${(stats.pressure * 100).toFixed(1)}%`);
                this.triggerCleanup();
            }
            
            // Log memory stats
            console.debug('Memory stats:', {
                used: `${(stats.usedSize / (1024 * 1024)).toFixed(1)}MB`,
                free: `${(stats.freeSize / (1024 * 1024)).toFixed(1)}MB`,
                pressure: `${(stats.pressure * 100).toFixed(1)}%`
            });
            
        } catch (error) {
            console.error('Memory monitoring error:', error);
        }
    }
    
    triggerCleanup() {
        console.log('🧹 Triggering memory cleanup...');
        
        for (const handler of this.cleanupHandlers) {
            try {
                handler();
            } catch (error) {
                console.error('Cleanup handler error:', error);
            }
        }
    }
    
    addCleanupHandler(handler) {
        this.cleanupHandlers.push(handler);
    }
}

// Usage
const memoryMonitor = new MemoryMonitor(0.8);

memoryMonitor.addCleanupHandler(() => {
    // Cleanup unused concepts
    const unusedConcepts = getUnusedConcepts();
    for (const concept of unusedConcepts) {
        cortexWrapper.deallocateConcept(concept.id);
    }
});

memoryMonitor.startMonitoring();
```

2. **Implement concept lifecycle management:**
```javascript
class ConceptManager {
    constructor(maxConcepts = 100) {
        this.concepts = new Map();
        this.maxConcepts = maxConcepts;
        this.accessTimes = new Map();
    }
    
    async allocateConcept(name, size, metadata = {}) {
        // Check if we need to cleanup
        if (this.concepts.size >= this.maxConcepts) {
            await this.cleanupOldConcepts();
        }
        
        const concept = await cortexWrapper.allocateConcept(name, size, metadata);
        
        this.concepts.set(concept.id, concept);
        this.accessTimes.set(concept.id, Date.now());
        
        return concept;
    }
    
    async deallocateConcept(conceptId) {
        await cortexWrapper.deallocateConcept(conceptId);
        this.concepts.delete(conceptId);
        this.accessTimes.delete(conceptId);
    }
    
    markAccessed(conceptId) {
        this.accessTimes.set(conceptId, Date.now());
    }
    
    async cleanupOldConcepts() {
        const now = Date.now();
        const maxAge = 5 * 60 * 1000; // 5 minutes
        
        const conceptsToRemove = [];
        
        for (const [conceptId, lastAccess] of this.accessTimes) {
            if (now - lastAccess > maxAge) {
                conceptsToRemove.push(conceptId);
            }
        }
        
        // Sort by age (oldest first) and remove up to 20% of concepts
        conceptsToRemove.sort((a, b) => this.accessTimes.get(a) - this.accessTimes.get(b));
        const toRemove = conceptsToRemove.slice(0, Math.ceil(this.concepts.size * 0.2));
        
        for (const conceptId of toRemove) {
            await this.deallocateConcept(conceptId);
        }
        
        console.log(`🧹 Cleaned up ${toRemove.length} old concepts`);
    }
}
```

## Performance Problems

### Problem: Slow spatial pooling operations

**Symptoms:**
- Operations taking > 100ms consistently
- UI freezing during processing
- Poor responsiveness

**Solutions:**

1. **Optimize input patterns:**
```javascript
// Bad: Dense patterns
const badPattern = new Array(2048).fill(1); // 100% density

// Good: Sparse patterns (2-5% density)
const goodPattern = new Array(2048).fill(0);
for (let i = 0; i < 40; i++) { // ~2% density
    goodPattern[Math.floor(Math.random() * 2048)] = 1;
}

// Optimize pattern generation
function createOptimizedPattern(inputData) {
    const pattern = new Array(2048).fill(0);
    const targetDensity = 0.02; // 2%
    const targetBits = Math.floor(pattern.length * targetDensity);
    
    // Hash-based bit selection for consistency
    const hash = simpleHash(inputData);
    for (let i = 0; i < targetBits; i++) {
        const index = (hash + i * 17) % pattern.length;
        pattern[index] = 1;
    }
    
    return pattern;
}
```

2. **Implement batching and chunking:**
```javascript
class BatchProcessor {
    constructor(batchSize = 10, delayMs = 10) {
        this.batchSize = batchSize;
        this.delayMs = delayMs;
        this.queue = [];
        this.processing = false;
    }
    
    async processBatch(patterns, options = {}) {
        return new Promise((resolve, reject) => {
            this.queue.push(...patterns.map(pattern => ({
                pattern,
                options,
                resolve,
                reject
            })));
            
            if (!this.processing) {
                this.processQueue();
            }
        });
    }
    
    async processQueue() {
        this.processing = true;
        
        while (this.queue.length > 0) {
            const batch = this.queue.splice(0, this.batchSize);
            const results = [];
            
            for (const item of batch) {
                try {
                    const result = await cortexWrapper.spatialPooling(item.pattern, item.options);
                    results.push(result);
                    item.resolve(result);
                } catch (error) {
                    item.reject(error);
                }
            }
            
            // Allow UI updates between batches
            if (this.queue.length > 0) {
                await new Promise(resolve => setTimeout(resolve, this.delayMs));
            }
        }
        
        this.processing = false;
    }
}

// Usage
const processor = new BatchProcessor(5, 16); // Process 5 at a time with 16ms delay
const results = await processor.processBatch(patterns);
```

3. **Use Web Workers for heavy operations:**
```javascript
// worker.js
importScripts('./cortex-kg.js');

let initialized = false;

self.onmessage = async function(e) {
    const { type, data, id } = e.data;
    
    try {
        if (type === 'init') {
            if (!initialized) {
                await wasm_loader.init(data);
                initialized = true;
            }
            self.postMessage({ type: 'init-complete', id });
            
        } else if (type === 'spatial-pooling') {
            if (!initialized) {
                throw new Error('Worker not initialized');
            }
            
            const result = await cortexWrapper.spatialPooling(data.pattern, data.options);
            self.postMessage({ type: 'spatial-pooling-result', id, result });
            
        } else if (type === 'allocate-concept') {
            const concept = await cortexWrapper.allocateConcept(data.name, data.size, data.metadata);
            self.postMessage({ type: 'concept-allocated', id, concept });
        }
        
    } catch (error) {
        self.postMessage({ type: 'error', id, error: error.message });
    }
};

// main.js
class WorkerPool {
    constructor(workerCount = navigator.hardwareConcurrency || 4) {
        this.workers = [];
        this.workQueue = [];
        this.idCounter = 0;
        
        for (let i = 0; i < workerCount; i++) {
            const worker = new Worker('./worker.js');
            worker.onmessage = this.handleWorkerMessage.bind(this);
            this.workers.push({
                worker,
                busy: false,
                pendingTasks: new Map()
            });
        }
    }
    
    async initializeWorkers(config) {
        const promises = this.workers.map(({ worker }) => 
            this.sendTask(worker, 'init', config)
        );
        await Promise.all(promises);
    }
    
    async spatialPooling(pattern, options) {
        const worker = this.getAvailableWorker();
        return this.sendTask(worker, 'spatial-pooling', { pattern, options });
    }
    
    getAvailableWorker() {
        return this.workers.find(w => !w.busy) || this.workers[0];
    }
    
    sendTask(workerInfo, type, data) {
        return new Promise((resolve, reject) => {
            const id = ++this.idCounter;
            
            workerInfo.pendingTasks.set(id, { resolve, reject });
            workerInfo.busy = true;
            workerInfo.worker.postMessage({ type, data, id });
        });
    }
    
    handleWorkerMessage(e) {
        const { type, id, result, error } = e.data;
        const workerInfo = this.workers.find(w => 
            w.pendingTasks.has(id)
        );
        
        if (!workerInfo) return;
        
        const { resolve, reject } = workerInfo.pendingTasks.get(id);
        workerInfo.pendingTasks.delete(id);
        workerInfo.busy = workerInfo.pendingTasks.size > 0;
        
        if (error) {
            reject(new Error(error));
        } else {
            resolve(result);
        }
    }
}

// Usage
const workerPool = new WorkerPool(4);
await workerPool.initializeWorkers({ memorySize: 32 * 1024 * 1024 });

const results = await Promise.all(
    patterns.map(pattern => workerPool.spatialPooling(pattern))
);
```

### Problem: Browser freezing during operations

**Symptoms:**
- UI becomes unresponsive
- Browser shows "Page Unresponsive" warnings
- Animation stuttering

**Solutions:**

1. **Implement yielding mechanisms:**
```javascript
async function processWithYielding(items, processor, yieldInterval = 50) {
    const results = [];
    
    for (let i = 0; i < items.length; i++) {
        results.push(await processor(items[i]));
        
        // Yield control every yieldInterval operations
        if (i % yieldInterval === 0) {
            await new Promise(resolve => setTimeout(resolve, 0));
        }
    }
    
    return results;
}

// Usage
const results = await processWithYielding(patterns, async (pattern) => {
    return await cortexWrapper.spatialPooling(pattern);
}, 10);
```

2. **Use RequestIdleCallback for non-critical operations:**
```javascript
class IdleProcessor {
    constructor() {
        this.queue = [];
        this.processing = false;
    }
    
    addTask(task) {
        this.queue.push(task);
        this.scheduleProcessing();
    }
    
    scheduleProcessing() {
        if (this.processing) return;
        
        if ('requestIdleCallback' in window) {
            requestIdleCallback((deadline) => {
                this.processUntilDeadline(deadline);
            }, { timeout: 1000 });
        } else {
            // Fallback for browsers without requestIdleCallback
            setTimeout(() => this.processUntilDeadline({ timeRemaining: () => 5 }), 0);
        }
    }
    
    processUntilDeadline(deadline) {
        this.processing = true;
        
        while (this.queue.length > 0 && deadline.timeRemaining() > 1) {
            const task = this.queue.shift();
            try {
                task();
            } catch (error) {
                console.error('Idle task error:', error);
            }
        }
        
        this.processing = false;
        
        if (this.queue.length > 0) {
            this.scheduleProcessing();
        }
    }
}

// Usage
const idleProcessor = new IdleProcessor();

// Queue non-critical operations
idleProcessor.addTask(() => {
    // Cleanup old data
    cleanupOldConcepts();
});

idleProcessor.addTask(() => {
    // Update statistics
    updatePerformanceMetrics();
});
```

## Browser Compatibility

### Problem: Safari-specific issues

**Symptoms:**
- WASM fails to load in Safari
- SharedArrayBuffer not available
- Performance significantly worse

**Solutions:**

1. **Safari-specific configuration:**
```javascript
function getSafariConfig() {
    const isSafari = /^((?!chrome|android).)*safari/i.test(navigator.userAgent);
    
    if (isSafari) {
        return {
            memorySize: 16 * 1024 * 1024, // Smaller memory allocation
            enableSIMD: false,             // SIMD support is limited
            enableThreads: false,          // No SharedArrayBuffer
            enableStreaming: false,        // No streaming compilation
            wasmLoadMethod: 'fetch'        // Use fetch instead of streaming
        };
    }
    
    return {
        memorySize: 64 * 1024 * 1024,
        enableSIMD: true,
        enableThreads: true,
        enableStreaming: true,
        wasmLoadMethod: 'streaming'
    };
}

// Custom WASM loading for Safari
async function loadWasmForSafari(url) {
    const response = await fetch(url);
    const wasmBytes = await response.arrayBuffer();
    const wasmModule = await WebAssembly.compile(wasmBytes);
    const wasmInstance = await WebAssembly.instantiate(wasmModule);
    return wasmInstance;
}
```

2. **Feature detection and polyfills:**
```javascript
// Polyfill for missing features
if (!window.SharedArrayBuffer) {
    // Provide fallback for SharedArrayBuffer
    window.SharedArrayBuffer = ArrayBuffer;
    console.warn('SharedArrayBuffer not available, using ArrayBuffer fallback');
}

if (!WebAssembly.instantiateStreaming) {
    // Polyfill for browsers without streaming support
    WebAssembly.instantiateStreaming = async function(response, importObject) {
        const source = await (await response).arrayBuffer();
        return await WebAssembly.instantiate(source, importObject);
    };
}

// Check for required features
function checkBrowserCapabilities() {
    const capabilities = {
        webAssembly: typeof WebAssembly !== 'undefined',
        simd: false,
        threads: typeof SharedArrayBuffer !== 'undefined',
        streaming: typeof WebAssembly.instantiateStreaming !== 'undefined',
        indexedDB: typeof indexedDB !== 'undefined'
    };
    
    // Test SIMD support safely
    try {
        const simdTest = new Uint8Array([
            0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
            0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7b,
            0x03, 0x02, 0x01, 0x00,
            0x0a, 0x0a, 0x01, 0x08, 0x00, 0xfd, 0x0c, 0x00, 0x00, 0x00, 0x0b
        ]);
        WebAssembly.validate(simdTest);
        capabilities.simd = true;
    } catch (e) {
        capabilities.simd = false;
    }
    
    return capabilities;
}
```

### Problem: Mobile browser limitations

**Symptoms:**
- Out of memory errors on mobile
- Poor performance on mobile devices
- Touch events not working properly

**Solutions:**

1. **Mobile-optimized configuration:**
```javascript
function getMobileConfig() {
    const isMobile = /Mobi|Android/i.test(navigator.userAgent);
    const isTablet = /iPad|Android.*(?!Mobile)/i.test(navigator.userAgent);
    
    if (isMobile) {
        return {
            memorySize: 8 * 1024 * 1024,   // 8MB for phones
            enableSIMD: false,
            enableThreads: false,
            batchSize: 5,                   // Smaller batch sizes
            yieldInterval: 10,              // More frequent yielding
            visualizationQuality: 'low'     // Reduced visualization complexity
        };
    } else if (isTablet) {
        return {
            memorySize: 16 * 1024 * 1024,  // 16MB for tablets
            enableSIMD: false,
            enableThreads: false,
            batchSize: 10,
            yieldInterval: 20,
            visualizationQuality: 'medium'
        };
    }
    
    // Desktop configuration
    return {
        memorySize: 64 * 1024 * 1024,
        enableSIMD: true,
        enableThreads: true,
        batchSize: 50,
        yieldInterval: 100,
        visualizationQuality: 'high'
    };
}
```

2. **Touch event handling:**
```javascript
class TouchHandler {
    constructor(element) {
        this.element = element;
        this.touches = new Map();
        this.setupEventListeners();
    }
    
    setupEventListeners() {
        // Prevent default touch behaviors that interfere with canvas
        this.element.addEventListener('touchstart', this.handleTouchStart.bind(this), { passive: false });
        this.element.addEventListener('touchmove', this.handleTouchMove.bind(this), { passive: false });
        this.element.addEventListener('touchend', this.handleTouchEnd.bind(this), { passive: false });
        this.element.addEventListener('touchcancel', this.handleTouchCancel.bind(this), { passive: false });
        
        // Prevent context menu on long press
        this.element.addEventListener('contextmenu', (e) => e.preventDefault());
    }
    
    handleTouchStart(e) {
        e.preventDefault(); // Prevent scrolling
        
        for (const touch of e.changedTouches) {
            const rect = this.element.getBoundingClientRect();
            const x = touch.clientX - rect.left;
            const y = touch.clientY - rect.top;
            
            this.touches.set(touch.identifier, { x, y, startTime: Date.now() });
            
            // Emit custom event
            this.element.dispatchEvent(new CustomEvent('canvastouchstart', {
                detail: { x, y, touchId: touch.identifier }
            }));
        }
    }
    
    handleTouchMove(e) {
        e.preventDefault();
        
        for (const touch of e.changedTouches) {
            if (this.touches.has(touch.identifier)) {
                const rect = this.element.getBoundingClientRect();
                const x = touch.clientX - rect.left;
                const y = touch.clientY - rect.top;
                
                this.touches.set(touch.identifier, { 
                    ...this.touches.get(touch.identifier), 
                    x, y 
                });
                
                this.element.dispatchEvent(new CustomEvent('canvastouchmove', {
                    detail: { x, y, touchId: touch.identifier }
                }));
            }
        }
    }
    
    handleTouchEnd(e) {
        e.preventDefault();
        
        for (const touch of e.changedTouches) {
            if (this.touches.has(touch.identifier)) {
                const touchData = this.touches.get(touch.identifier);
                const duration = Date.now() - touchData.startTime;
                
                this.element.dispatchEvent(new CustomEvent('canvastouchend', {
                    detail: { 
                        x: touchData.x, 
                        y: touchData.y, 
                        touchId: touch.identifier,
                        duration 
                    }
                }));
                
                this.touches.delete(touch.identifier);
            }
        }
    }
    
    handleTouchCancel(e) {
        for (const touch of e.changedTouches) {
            this.touches.delete(touch.identifier);
        }
    }
}

// Usage
const canvas = document.getElementById('myCanvas');
const touchHandler = new TouchHandler(canvas);

canvas.addEventListener('canvastouchstart', (e) => {
    console.log('Touch started at:', e.detail.x, e.detail.y);
});
```

## API Usage Errors

### Problem: Incorrect concept allocation

**Symptoms:**
- "Invalid concept size" errors
- Memory allocation failures
- Concept ID conflicts

**Solutions:**

1. **Validate parameters before allocation:**
```javascript
class ConceptValidator {
    static validateAllocation(name, size, metadata = {}) {
        const errors = [];
        
        // Validate name
        if (!name || typeof name !== 'string') {
            errors.push('Name must be a non-empty string');
        } else if (name.length > 100) {
            errors.push('Name must be 100 characters or less');
        } else if (!/^[a-zA-Z0-9_-]+$/.test(name)) {
            errors.push('Name can only contain letters, numbers, underscores, and hyphens');
        }
        
        // Validate size
        if (!Number.isInteger(size) || size <= 0) {
            errors.push('Size must be a positive integer');
        } else if (size > 100 * 1024 * 1024) { // 100MB limit
            errors.push('Size cannot exceed 100MB');
        } else if (size < 64) {
            errors.push('Size must be at least 64 bytes');
        }
        
        // Validate metadata
        if (metadata && typeof metadata !== 'object') {
            errors.push('Metadata must be an object');
        } else if (metadata) {
            try {
                const serialized = JSON.stringify(metadata);
                if (serialized.length > 10000) {
                    errors.push('Metadata is too large (max 10KB when serialized)');
                }
            } catch (e) {
                errors.push('Metadata must be JSON serializable');
            }
        }
        
        if (errors.length > 0) {
            throw new Error(`Concept validation failed: ${errors.join(', ')}`);
        }
    }
    
    static validatePattern(pattern) {
        if (!Array.isArray(pattern)) {
            throw new Error('Pattern must be an array');
        }
        
        if (pattern.length === 0) {
            throw new Error('Pattern cannot be empty');
        }
        
        if (pattern.length > 10000) {
            throw new Error('Pattern too large (max 10,000 elements)');
        }
        
        const allBinary = pattern.every(bit => bit === 0 || bit === 1);
        if (!allBinary) {
            throw new Error('Pattern must contain only 0s and 1s');
        }
        
        const density = pattern.reduce((sum, bit) => sum + bit, 0) / pattern.length;
        if (density > 0.1) {
            console.warn(`High pattern density (${(density * 100).toFixed(1)}%) may reduce performance`);
        }
    }
}

// Safe concept allocation wrapper
async function safeAllocateConcept(name, size, metadata) {
    try {
        ConceptValidator.validateAllocation(name, size, metadata);
        
        // Check if concept with this name already exists
        const existing = await findConceptByName(name);
        if (existing) {
            throw new Error(`Concept with name "${name}" already exists`);
        }
        
        // Check memory availability
        const memoryStats = wasm_loader.getMemoryUsage();
        const availableMemory = memoryStats.freeSize;
        
        if (size > availableMemory * 0.8) { // Leave 20% buffer
            throw new Error(`Insufficient memory: ${size} bytes requested, ${Math.floor(availableMemory * 0.8)} bytes available`);
        }
        
        const concept = await cortexWrapper.allocateConcept(name, size, metadata);
        console.log(`✅ Allocated concept "${name}" (${size} bytes)`);
        
        return concept;
        
    } catch (error) {
        console.error(`❌ Failed to allocate concept "${name}":`, error.message);
        throw error;
    }
}
```

2. **Implement retry mechanisms:**
```javascript
async function allocateConceptWithRetry(name, size, metadata, maxRetries = 3) {
    let lastError;
    
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
            return await safeAllocateConcept(name, size, metadata);
            
        } catch (error) {
            lastError = error;
            
            if (error.message.includes('Insufficient memory')) {
                // Try to free up memory
                console.log(`Attempt ${attempt}: Memory full, triggering cleanup...`);
                await performMemoryCleanup();
                
                // Reduce size on subsequent attempts
                if (attempt > 1) {
                    size = Math.floor(size * 0.8);
                    console.log(`Reducing size to ${size} bytes`);
                }
                
            } else if (error.message.includes('already exists')) {
                // Append suffix to make name unique
                name = `${name}_${attempt}`;
                console.log(`Name conflict, trying "${name}"`);
                
            } else {
                // Other errors are not retryable
                throw error;
            }
            
            // Wait before retry
            await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
        }
    }
    
    throw new Error(`Failed to allocate concept after ${maxRetries} attempts: ${lastError.message}`);
}
```

### Problem: Spatial pooling parameter errors

**Symptoms:**
- "Invalid sparsity value" errors
- Poor spatial pooling results
- Unexpected behavior

**Solutions:**

1. **Parameter validation and optimization:**
```javascript
class SpatialPoolingOptimizer {
    static validateOptions(options = {}) {
        const validated = {
            sparsity: 0.02,
            boostStrength: 2.0,
            dutyCyclePeriod: 1000,
            ...options
        };
        
        // Validate sparsity
        if (validated.sparsity <= 0 || validated.sparsity >= 1) {
            throw new Error('Sparsity must be between 0 and 1 (exclusive)');
        }
        if (validated.sparsity > 0.1) {
            console.warn(`High sparsity (${validated.sparsity}) may reduce efficiency`);
        }
        
        // Validate boost strength
        if (validated.boostStrength < 0) {
            throw new Error('Boost strength must be non-negative');
        }
        if (validated.boostStrength > 10) {
            console.warn(`Very high boost strength (${validated.boostStrength}) may cause instability`);
        }
        
        // Validate duty cycle period
        if (validated.dutyCyclePeriod < 100) {
            throw new Error('Duty cycle period must be at least 100');
        }
        
        return validated;
    }
    
    static getOptimalParameters(patternSize, targetColumns = null) {
        // Estimate optimal parameters based on pattern size
        const columns = targetColumns || Math.floor(patternSize / 4);
        
        return {
            sparsity: Math.max(0.01, Math.min(0.05, 40 / columns)),
            boostStrength: patternSize > 1000 ? 2.5 : 2.0,
            dutyCyclePeriod: Math.min(2000, Math.max(500, columns))
        };
    }
    
    static analyzePattern(pattern) {
        const totalBits = pattern.length;
        const activeBits = pattern.reduce((sum, bit) => sum + bit, 0);
        const density = activeBits / totalBits;
        
        return {
            size: totalBits,
            activeBits,
            density,
            sparse: density < 0.1,
            recommended: this.getOptimalParameters(totalBits)
        };
    }
}

// Safe spatial pooling wrapper
async function safeSpatialPooling(pattern, options = {}) {
    try {
        // Validate and analyze pattern
        ConceptValidator.validatePattern(pattern);
        const analysis = SpatialPoolingOptimizer.analyzePattern(pattern);
        
        console.debug('Pattern analysis:', analysis);
        
        // Validate and optimize options
        const validatedOptions = SpatialPoolingOptimizer.validateOptions(options);
        
        // Use recommended parameters if not specified
        if (!options.sparsity) {
            validatedOptions.sparsity = analysis.recommended.sparsity;
        }
        if (!options.boostStrength) {
            validatedOptions.boostStrength = analysis.recommended.boostStrength;
        }
        if (!options.dutyCyclePeriod) {
            validatedOptions.dutyCyclePeriod = analysis.recommended.dutyCyclePeriod;
        }
        
        console.debug('Using spatial pooling options:', validatedOptions);
        
        const startTime = performance.now();
        const result = await cortexWrapper.spatialPooling(pattern, validatedOptions);
        const duration = performance.now() - startTime;
        
        console.debug(`Spatial pooling completed in ${duration.toFixed(1)}ms`);
        
        return result;
        
    } catch (error) {
        console.error('Spatial pooling failed:', error.message);
        throw error;
    }
}
```

## Debugging Tools

### Comprehensive debugging suite

```javascript
class CortexKGDebugger {
    constructor() {
        this.logs = [];
        this.metrics = new Map();
        this.breakpoints = new Set();
        this.watchedVariables = new Map();
        this.enabled = false;
    }
    
    enable() {
        this.enabled = true;
        this.interceptAPICalls();
        this.startMetricsCollection();
        console.log('🐛 CortexKG Debugger enabled');
    }
    
    disable() {
        this.enabled = false;
        this.restoreAPICalls();
        this.stopMetricsCollection();
        console.log('🐛 CortexKG Debugger disabled');
    }
    
    interceptAPICalls() {
        // Store original methods
        this.originalMethods = {
            allocateConcept: cortexWrapper.allocateConcept,
            deallocateConcept: cortexWrapper.deallocateConcept,
            spatialPooling: cortexWrapper.spatialPooling,
            temporalMemory: cortexWrapper.temporalMemory
        };
        
        // Wrap methods with debugging
        cortexWrapper.allocateConcept = this.wrapMethod('allocateConcept', this.originalMethods.allocateConcept);
        cortexWrapper.deallocateConcept = this.wrapMethod('deallocateConcept', this.originalMethods.deallocateConcept);
        cortexWrapper.spatialPooling = this.wrapMethod('spatialPooling', this.originalMethods.spatialPooling);
        cortexWrapper.temporalMemory = this.wrapMethod('temporalMemory', this.originalMethods.temporalMemory);
    }
    
    wrapMethod(methodName, originalMethod) {
        return async (...args) => {
            const callId = this.generateCallId();
            const startTime = performance.now();
            
            this.log('info', `📞 ${methodName} called`, { callId, args: this.sanitizeArgs(args) });
            
            // Check breakpoints
            if (this.breakpoints.has(methodName)) {
                debugger; // This will trigger browser debugger
            }
            
            try {
                const result = await originalMethod.apply(cortexWrapper, args);
                const endTime = performance.now();
                const duration = endTime - startTime;
                
                this.recordMetric(methodName, duration);
                this.log('info', `✅ ${methodName} completed`, { callId, duration: `${duration.toFixed(1)}ms`, result: this.sanitizeResult(result) });
                
                return result;
                
            } catch (error) {
                const endTime = performance.now();
                const duration = endTime - startTime;
                
                this.log('error', `❌ ${methodName} failed`, { callId, duration: `${duration.toFixed(1)}ms`, error: error.message });
                throw error;
            }
        };
    }
    
    log(level, message, data = {}) {
        if (!this.enabled) return;
        
        const logEntry = {
            timestamp: new Date().toISOString(),
            level,
            message,
            ...data
        };
        
        this.logs.push(logEntry);
        
        // Keep only last 1000 logs
        if (this.logs.length > 1000) {
            this.logs.shift();
        }
        
        // Console output with styling
        const style = {
            error: 'color: #f44336; font-weight: bold;',
            warn: 'color: #ff9800; font-weight: bold;',
            info: 'color: #2196f3;',
            debug: 'color: #9e9e9e;'
        };
        
        console.log(`%c[CortexKG Debug] ${message}`, style[level] || '', data);
    }
    
    sanitizeArgs(args) {
        return args.map(arg => {
            if (Array.isArray(arg) && arg.length > 10) {
                return `Array(${arg.length}) [${arg.slice(0, 5).join(',')}, ...]`;
            }
            return arg;
        });
    }
    
    sanitizeResult(result) {
        if (result && typeof result === 'object') {
            const sanitized = {};
            for (const [key, value] of Object.entries(result)) {
                if (Array.isArray(value) && value.length > 10) {
                    sanitized[key] = `Array(${value.length})`;
                } else {
                    sanitized[key] = value;
                }
            }
            return sanitized;
        }
        return result;
    }
    
    recordMetric(methodName, duration) {
        if (!this.metrics.has(methodName)) {
            this.metrics.set(methodName, {
                calls: 0,
                totalDuration: 0,
                minDuration: Infinity,
                maxDuration: 0,
                recentDurations: []
            });
        }
        
        const metric = this.metrics.get(methodName);
        metric.calls++;
        metric.totalDuration += duration;
        metric.minDuration = Math.min(metric.minDuration, duration);
        metric.maxDuration = Math.max(metric.maxDuration, duration);
        metric.recentDurations.push(duration);
        
        // Keep only last 100 durations
        if (metric.recentDurations.length > 100) {
            metric.recentDurations.shift();
        }
    }
    
    generateCallId() {
        return Math.random().toString(36).substr(2, 9);
    }
    
    setBreakpoint(methodName) {
        this.breakpoints.add(methodName);
        this.log('info', `🔴 Breakpoint set for ${methodName}`);
    }
    
    removeBreakpoint(methodName) {
        this.breakpoints.delete(methodName);
        this.log('info', `⚪ Breakpoint removed for ${methodName}`);
    }
    
    watchVariable(name, getValue) {
        this.watchedVariables.set(name, {
            getValue,
            lastValue: getValue(),
            changes: []
        });
        this.log('info', `👁️ Watching variable: ${name}`);
    }
    
    checkWatchedVariables() {
        for (const [name, watch] of this.watchedVariables) {
            try {
                const currentValue = watch.getValue();
                if (currentValue !== watch.lastValue) {
                    const change = {
                        timestamp: Date.now(),
                        oldValue: watch.lastValue,
                        newValue: currentValue
                    };
                    
                    watch.changes.push(change);
                    watch.lastValue = currentValue;
                    
                    this.log('info', `🔄 Variable changed: ${name}`, change);
                }
            } catch (error) {
                this.log('warn', `⚠️ Error watching variable ${name}:`, error.message);
            }
        }
    }
    
    startMetricsCollection() {
        this.metricsInterval = setInterval(() => {
            this.checkWatchedVariables();
            this.collectSystemMetrics();
        }, 1000);
    }
    
    stopMetricsCollection() {
        if (this.metricsInterval) {
            clearInterval(this.metricsInterval);
        }
    }
    
    collectSystemMetrics() {
        try {
            const memoryStats = wasm_loader.getMemoryUsage();
            this.recordSystemMetric('memory', memoryStats);
        } catch (error) {
            // Ignore if WASM not loaded
        }
    }
    
    recordSystemMetric(type, data) {
        if (!this.systemMetrics) {
            this.systemMetrics = new Map();
        }
        
        if (!this.systemMetrics.has(type)) {
            this.systemMetrics.set(type, []);
        }
        
        const metrics = this.systemMetrics.get(type);
        metrics.push({
            timestamp: Date.now(),
            ...data
        });
        
        // Keep only last 100 measurements
        if (metrics.length > 100) {
            metrics.shift();
        }
    }
    
    getReport() {
        const report = {
            overview: {
                debuggerEnabled: this.enabled,
                totalLogs: this.logs.length,
                activeBreakpoints: Array.from(this.breakpoints),
                watchedVariables: Array.from(this.watchedVariables.keys())
            },
            methodMetrics: {},
            recentLogs: this.logs.slice(-20),
            systemMetrics: this.systemMetrics ? Object.fromEntries(this.systemMetrics) : {}
        };
        
        for (const [methodName, metric] of this.metrics) {
            report.methodMetrics[methodName] = {
                calls: metric.calls,
                avgDuration: (metric.totalDuration / metric.calls).toFixed(1),
                minDuration: metric.minDuration.toFixed(1),
                maxDuration: metric.maxDuration.toFixed(1)
            };
        }
        
        return report;
    }
    
    exportLogs() {
        const data = {
            exportTime: new Date().toISOString(),
            logs: this.logs,
            metrics: Object.fromEntries(this.metrics),
            systemMetrics: this.systemMetrics ? Object.fromEntries(this.systemMetrics) : {}
        };
        
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `cortexkg-debug-${Date.now()}.json`;
        a.click();
        
        URL.revokeObjectURL(url);
    }
    
    restoreAPICalls() {
        if (this.originalMethods) {
            Object.assign(cortexWrapper, this.originalMethods);
        }
    }
}

// Global debugger instance
window.cortexDebugger = new CortexKGDebugger();

// Debugging utilities
window.debugCortexKG = {
    enable: () => window.cortexDebugger.enable(),
    disable: () => window.cortexDebugger.disable(),
    breakpoint: (method) => window.cortexDebugger.setBreakpoint(method),
    watch: (name, getValue) => window.cortexDebugger.watchVariable(name, getValue),
    report: () => console.table(window.cortexDebugger.getReport()),
    export: () => window.cortexDebugger.exportLogs(),
    clear: () => window.cortexDebugger.logs = []
};

console.log('🐛 CortexKG Debugging tools loaded. Use debugCortexKG.enable() to start debugging.');
```

## Common Error Messages

### Complete error reference guide

```javascript
const ERROR_MESSAGES = {
    'WebAssembly.instantiate(): Invalid module': {
        description: 'The WASM file is corrupted or incompatible',
        causes: [
            'WASM file download was interrupted',
            'Wrong WASM file version for this browser',
            'MIME type not set correctly on server'
        ],
        solutions: [
            'Clear browser cache and reload',
            'Check server MIME type configuration',
            'Verify WASM file integrity',
            'Try a different browser or update current browser'
        ]
    },
    
    'Memory allocation failed': {
        description: 'Not enough memory available for the requested allocation',
        causes: [
            'Device has insufficient RAM',
            'Too many concepts already allocated',
            'Memory fragmentation',
            'Browser memory limits exceeded'
        ],
        solutions: [
            'Reduce memorySize in initialization config',
            'Implement memory cleanup strategies',
            'Deallocate unused concepts',
            'Use smaller concept sizes'
        ]
    },
    
    'Concept not found': {
        description: 'Attempted to access a concept that doesn\'t exist',
        causes: [
            'Concept was already deallocated',
            'Incorrect concept ID',
            'Concept allocation failed silently'
        ],
        solutions: [
            'Check concept exists before accessing',
            'Implement concept lifecycle tracking',
            'Validate concept IDs before use',
            'Add error handling for allocation failures'
        ]
    },
    
    'Invalid spatial pooling parameters': {
        description: 'Spatial pooling options contain invalid values',
        causes: [
            'Sparsity value outside valid range (0-1)',
            'Negative boost strength',
            'Invalid duty cycle period'
        ],
        solutions: [
            'Validate parameters before calling spatialPooling',
            'Use recommended parameter values',
            'Check parameter documentation',
            'Use SpatialPoolingOptimizer.validateOptions()'
        ]
    }
};

function explainError(errorMessage) {
    const explanation = ERROR_MESSAGES[errorMessage];
    
    if (explanation) {
        console.group(`❌ Error: ${errorMessage}`);
        console.log(`📝 Description: ${explanation.description}`);
        console.log(`🔍 Possible causes:`);
        explanation.causes.forEach(cause => console.log(`  • ${cause}`));
        console.log(`💡 Solutions:`);
        explanation.solutions.forEach(solution => console.log(`  • ${solution}`));
        console.groupEnd();
    } else {
        console.log(`❓ Unknown error: ${errorMessage}`);
    }
}

// Auto-explain errors
window.addEventListener('error', (event) => {
    if (event.error && event.error.message) {
        explainError(event.error.message);
    }
});
```

## Advanced Troubleshooting

### Network and connectivity issues

```javascript
class NetworkDiagnostics {
    static async checkConnectivity() {
        const tests = [
            { name: 'Internet Connection', test: () => this.testInternetConnection() },
            { name: 'WASM File Access', test: () => this.testWasmFileAccess() },
            { name: 'CDN Availability', test: () => this.testCDNAvailability() },
            { name: 'CORS Configuration', test: () => this.testCORSConfiguration() }
        ];
        
        const results = {};
        
        for (const { name, test } of tests) {
            try {
                results[name] = await test();
            } catch (error) {
                results[name] = { success: false, error: error.message };
            }
        }
        
        return results;
    }
    
    static async testInternetConnection() {
        const start = performance.now();
        const response = await fetch('https://httpbin.org/get', { 
            method: 'GET',
            cache: 'no-cache'
        });
        const duration = performance.now() - start;
        
        return {
            success: response.ok,
            latency: `${duration.toFixed(0)}ms`,
            status: response.status
        };
    }
    
    static async testWasmFileAccess() {
        const wasmUrl = './js/cortex-kg.wasm';
        const response = await fetch(wasmUrl, { method: 'HEAD' });
        
        return {
            success: response.ok,
            contentType: response.headers.get('content-type'),
            contentLength: response.headers.get('content-length'),
            cacheControl: response.headers.get('cache-control')
        };
    }
    
    static async testCDNAvailability() {
        const cdnUrls = [
            'https://cdn.jsdelivr.net/npm/cortex-kg@latest/package.json',
            'https://unpkg.com/cortex-kg@latest/package.json'
        ];
        
        const results = {};
        
        for (const url of cdnUrls) {
            try {
                const start = performance.now();
                const response = await fetch(url);
                const duration = performance.now() - start;
                
                results[url] = {
                    success: response.ok,
                    latency: `${duration.toFixed(0)}ms`,
                    status: response.status
                };
            } catch (error) {
                results[url] = { success: false, error: error.message };
            }
        }
        
        return results;
    }
    
    static async testCORSConfiguration() {
        try {
            const response = await fetch(window.location.href, {
                method: 'OPTIONS'
            });
            
            const headers = {
                'Access-Control-Allow-Origin': response.headers.get('access-control-allow-origin'),
                'Access-Control-Allow-Methods': response.headers.get('access-control-allow-methods'),
                'Access-Control-Allow-Headers': response.headers.get('access-control-allow-headers')
            };
            
            return {
                success: true,
                corsHeaders: headers,
                originAllowed: headers['Access-Control-Allow-Origin'] === '*' || 
                              headers['Access-Control-Allow-Origin'] === window.location.origin
            };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
}

// Usage
NetworkDiagnostics.checkConnectivity().then(results => {
    console.log('🌐 Network Diagnostics:', results);
});
```

### Performance profiling tools

```javascript
class PerformanceProfiler {
    constructor() {
        this.profiles = new Map();
        this.currentProfile = null;
    }
    
    startProfile(name) {
        if (this.currentProfile) {
            console.warn(`Profile "${this.currentProfile}" still active. Stopping it first.`);
            this.stopProfile();
        }
        
        this.currentProfile = name;
        this.profiles.set(name, {
            startTime: performance.now(),
            markers: [],
            memorySnapshots: [],
            operations: []
        });
        
        // Start memory monitoring
        this.startMemoryMonitoring();
        
        console.log(`📊 Started profiling: ${name}`);
    }
    
    stopProfile() {
        if (!this.currentProfile) {
            console.warn('No active profile to stop');
            return null;
        }
        
        const profile = this.profiles.get(this.currentProfile);
        profile.endTime = performance.now();
        profile.duration = profile.endTime - profile.startTime;
        
        this.stopMemoryMonitoring();
        
        console.log(`📊 Stopped profiling: ${this.currentProfile} (${profile.duration.toFixed(1)}ms)`);
        
        const result = this.generateReport(this.currentProfile);
        this.currentProfile = null;
        
        return result;
    }
    
    mark(label) {
        if (!this.currentProfile) return;
        
        const profile = this.profiles.get(this.currentProfile);
        const timestamp = performance.now();
        
        profile.markers.push({
            label,
            timestamp,
            relativeTime: timestamp - profile.startTime
        });
        
        console.log(`📍 Mark: ${label} (+${(timestamp - profile.startTime).toFixed(1)}ms)`);
    }
    
    recordOperation(operation, duration, metadata = {}) {
        if (!this.currentProfile) return;
        
        const profile = this.profiles.get(this.currentProfile);
        profile.operations.push({
            operation,
            duration,
            timestamp: performance.now(),
            ...metadata
        });
    }
    
    startMemoryMonitoring() {
        this.memoryInterval = setInterval(() => {
            if (this.currentProfile) {
                try {
                    const stats = wasm_loader.getMemoryUsage();
                    const profile = this.profiles.get(this.currentProfile);
                    
                    profile.memorySnapshots.push({
                        timestamp: performance.now(),
                        ...stats
                    });
                } catch (error) {
                    // Ignore if WASM not loaded
                }
            }
        }, 100); // Sample every 100ms
    }
    
    stopMemoryMonitoring() {
        if (this.memoryInterval) {
            clearInterval(this.memoryInterval);
        }
    }
    
    generateReport(profileName) {
        const profile = this.profiles.get(profileName);
        
        if (!profile) {
            throw new Error(`Profile "${profileName}" not found`);
        }
        
        const report = {
            name: profileName,
            duration: profile.duration,
            markers: profile.markers,
            operations: this.analyzeOperations(profile.operations),
            memory: this.analyzeMemory(profile.memorySnapshots),
            timeline: this.generateTimeline(profile)
        };
        
        return report;
    }
    
    analyzeOperations(operations) {
        const byType = new Map();
        
        for (const op of operations) {
            if (!byType.has(op.operation)) {
                byType.set(op.operation, {
                    count: 0,
                    totalDuration: 0,
                    minDuration: Infinity,
                    maxDuration: 0,
                    durations: []
                });
            }
            
            const stats = byType.get(op.operation);
            stats.count++;
            stats.totalDuration += op.duration;
            stats.minDuration = Math.min(stats.minDuration, op.duration);
            stats.maxDuration = Math.max(stats.maxDuration, op.duration);
            stats.durations.push(op.duration);
        }
        
        // Calculate averages and percentiles
        const result = {};
        for (const [operation, stats] of byType) {
            const sorted = stats.durations.sort((a, b) => a - b);
            
            result[operation] = {
                count: stats.count,
                avgDuration: stats.totalDuration / stats.count,
                minDuration: stats.minDuration,
                maxDuration: stats.maxDuration,
                p50: sorted[Math.floor(sorted.length * 0.5)],
                p95: sorted[Math.floor(sorted.length * 0.95)],
                p99: sorted[Math.floor(sorted.length * 0.99)]
            };
        }
        
        return result;
    }
    
    analyzeMemory(snapshots) {
        if (snapshots.length === 0) return null;
        
        const usedSizes = snapshots.map(s => s.usedSize);
        const pressures = snapshots.map(s => s.pressure);
        
        return {
            initial: snapshots[0],
            final: snapshots[snapshots.length - 1],
            peak: Math.max(...usedSizes),
            minUsed: Math.min(...usedSizes),
            maxPressure: Math.max(...pressures),
            averagePressure: pressures.reduce((a, b) => a + b, 0) / pressures.length,
            growthRate: (snapshots[snapshots.length - 1].usedSize - snapshots[0].usedSize) / snapshots.length
        };
    }
    
    generateTimeline(profile) {
        const events = [];
        
        // Add markers
        for (const marker of profile.markers) {
            events.push({
                type: 'marker',
                time: marker.relativeTime,
                label: marker.label
            });
        }
        
        // Add operations
        for (const op of profile.operations) {
            events.push({
                type: 'operation',
                time: op.timestamp - profile.startTime,
                operation: op.operation,
                duration: op.duration
            });
        }
        
        // Sort by time
        events.sort((a, b) => a.time - b.time);
        
        return events;
    }
    
    exportProfile(profileName, format = 'json') {
        const report = this.generateReport(profileName);
        
        if (format === 'json') {
            const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            
            const a = document.createElement('a');
            a.href = url;
            a.download = `cortexkg-profile-${profileName}-${Date.now()}.json`;
            a.click();
            
            URL.revokeObjectURL(url);
        }
    }
}

// Global profiler instance
window.cortexProfiler = new PerformanceProfiler();

// Profiling utilities
window.profileCortexKG = {
    start: (name) => window.cortexProfiler.startProfile(name),
    stop: () => window.cortexProfiler.stopProfile(),
    mark: (label) => window.cortexProfiler.mark(label),
    export: (name) => window.cortexProfiler.exportProfile(name)
};

console.log('📊 CortexKG Performance Profiler loaded. Use profileCortexKG.start("name") to begin profiling.');
```
```

## Expected Outputs
- Complete troubleshooting guide covering all common issues and their solutions
- Systematic diagnostic tools for identifying problems quickly
- Step-by-step solutions for installation, configuration, and runtime errors
- Browser-specific compatibility guides and workarounds
- Advanced debugging tools and performance profiling utilities
- Comprehensive error message reference with explanations and solutions

## Validation
1. Verify troubleshooting guide covers all major error scenarios encountered during development
2. Test diagnostic tools accuracy in identifying and resolving actual issues
3. Confirm solutions work across different browsers and deployment environments
4. Validate debugging tools provide actionable insights for performance optimization
5. Ensure guide is accessible to developers with varying levels of experience

## Next Steps
- Complete Phase 9 WASM implementation documentation
- Deploy troubleshooting guide to developer portal
- Create interactive troubleshooting assistant tool