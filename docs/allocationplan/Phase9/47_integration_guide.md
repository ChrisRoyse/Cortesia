# Micro-Phase 9.47: Integration Guide

## Objective
Build comprehensive integration guide for developers using CortexKG in their projects, covering setup, configuration, common patterns, and deployment scenarios.

## Prerequisites
- Completed micro-phase 9.46 (API Documentation)
- TypeScript definitions and API reference available
- Testing framework validated (phases 9.41-9.44)

## Task Description
Create detailed integration documentation covering project setup, dependency management, configuration options, common integration patterns, and troubleshooting. Provide step-by-step guides for different frameworks and deployment environments.

## Specific Actions

1. **Create comprehensive installation and setup guide**
```markdown
# CortexKG Integration Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [Installation Methods](#installation-methods)
3. [Framework Integration](#framework-integration)
4. [Configuration Options](#configuration-options)
5. [Common Patterns](#common-patterns)
6. [Performance Optimization](#performance-optimization)
7. [Deployment Scenarios](#deployment-scenarios)
8. [Troubleshooting](#troubleshooting)

## Quick Start

### Basic HTML Integration
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CortexKG Quick Start</title>
</head>
<body>
    <canvas id="cortical-canvas" width="800" height="600"></canvas>
    
    <!-- Include CortexKG scripts -->
    <script src="./cortex-kg-wasm.js"></script>
    <script src="./cortex-kg.js"></script>
    
    <script>
        async function initializeCortexKG() {
            try {
                // Initialize WASM module
                await wasmLoader.init({
                    memorySize: 32 * 1024 * 1024, // 32MB
                    enableSIMD: true
                });
                
                // Set up visualization
                const canvas = document.getElementById('cortical-canvas');
                cortexVisualizer.init(canvas);
                
                // Allocate first concept
                const concept = await cortexWrapper.allocateConcept('hello-world', 1024);
                console.log('CortexKG initialized successfully!', concept);
                
            } catch (error) {
                console.error('Failed to initialize CortexKG:', error);
            }
        }
        
        // Initialize when page loads
        window.addEventListener('load', initializeCortexKG);
    </script>
</body>
</html>
```

### NPM Package Integration
```bash
# Install CortexKG via npm
npm install cortex-kg

# For TypeScript projects
npm install @types/cortex-kg
```

```javascript
// ES6 Module Import
import { wasmLoader, cortexWrapper, storageManager, cortexVisualizer } from 'cortex-kg';

// CommonJS Import
const { wasmLoader, cortexWrapper } = require('cortex-kg');

// TypeScript with full type definitions
import {
  CortexKG,
  WasmConfig,
  Concept,
  ConceptMetadata
} from 'cortex-kg';
```

## Installation Methods

### Method 1: CDN Integration
```html
<!-- Production CDN -->
<script src="https://cdn.jsdelivr.net/npm/cortex-kg@latest/dist/cortex-kg.min.js"></script>

<!-- Development CDN with source maps -->
<script src="https://cdn.jsdelivr.net/npm/cortex-kg@latest/dist/cortex-kg.js"></script>

<!-- Specific version -->
<script src="https://cdn.jsdelivr.net/npm/cortex-kg@1.0.0/dist/cortex-kg.min.js"></script>
```

### Method 2: Local Build Integration
```bash
# Clone repository
git clone https://github.com/your-org/cortex-kg.git
cd cortex-kg

# Install dependencies
npm install

# Build for production
npm run build:release

# Copy files to your project
cp dist/* /path/to/your/project/js/
```

### Method 3: Package Manager Integration
```json
// package.json
{
  "dependencies": {
    "cortex-kg": "^1.0.0"
  },
  "devDependencies": {
    "@types/cortex-kg": "^1.0.0"
  }
}
```

### Method 4: Bundler Integration
```javascript
// webpack.config.js
module.exports = {
  // ... other config
  resolve: {
    alias: {
      'cortex-kg': path.resolve(__dirname, 'node_modules/cortex-kg/dist/cortex-kg.js')
    }
  },
  module: {
    rules: [
      {
        test: /\.wasm$/,
        type: 'webassembly/async'
      }
    ]
  },
  experiments: {
    asyncWebAssembly: true
  }
};
```

## Framework Integration

### React Integration
```jsx
// CortexKGProvider.jsx
import React, { createContext, useContext, useEffect, useState } from 'react';
import { wasmLoader, cortexWrapper } from 'cortex-kg';

const CortexKGContext = createContext();

export function CortexKGProvider({ children, config = {} }) {
  const [isLoaded, setIsLoaded] = useState(false);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    async function initializeCortexKG() {
      try {
        await wasmLoader.init({
          memorySize: 32 * 1024 * 1024,
          enableSIMD: true,
          ...config
        });
        setIsLoaded(true);
      } catch (err) {
        setError(err.message);
      }
    }
    
    initializeCortexKG();
  }, [config]);
  
  const value = {
    isLoaded,
    error,
    cortexWrapper,
    wasmLoader
  };
  
  return (
    <CortexKGContext.Provider value={value}>
      {children}
    </CortexKGContext.Provider>
  );
}

export function useCortexKG() {
  const context = useContext(CortexKGContext);
  if (!context) {
    throw new Error('useCortexKG must be used within CortexKGProvider');
  }
  return context;
}

// CorticalVisualization.jsx
import React, { useRef, useEffect } from 'react';
import { useCortexKG } from './CortexKGProvider';
import { cortexVisualizer } from 'cortex-kg';

export function CorticalVisualization({ width = 800, height = 600 }) {
  const canvasRef = useRef(null);
  const { isLoaded } = useCortexKG();
  
  useEffect(() => {
    if (isLoaded && canvasRef.current) {
      cortexVisualizer.init(canvasRef.current, {
        width,
        height,
        columnSize: 4
      });
    }
  }, [isLoaded, width, height]);
  
  return (
    <div className="cortical-visualization">
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        style={{ border: '1px solid #ccc' }}
      />
    </div>
  );
}

// App.jsx
import React from 'react';
import { CortexKGProvider } from './CortexKGProvider';
import { CorticalVisualization } from './CorticalVisualization';

function App() {
  return (
    <CortexKGProvider config={{ memorySize: 64 * 1024 * 1024 }}>
      <div className="App">
        <h1>CortexKG React Integration</h1>
        <CorticalVisualization />
      </div>
    </CortexKGProvider>
  );
}

export default App;
```

### Vue.js Integration
```vue
<!-- CortexKG.vue -->
<template>
  <div class="cortex-kg-container">
    <div v-if="loading" class="loading">
      Initializing CortexKG...
    </div>
    <div v-else-if="error" class="error">
      Error: {{ error }}
    </div>
    <div v-else>
      <canvas 
        ref="canvas"
        :width="width"
        :height="height"
        @click="handleCanvasClick"
      ></canvas>
      <div class="controls">
        <button @click="allocateRandomConcept">Add Concept</button>
        <button @click="clearAllConcepts">Clear All</button>
      </div>
    </div>
  </div>
</template>

<script>
import { wasmLoader, cortexWrapper, cortexVisualizer } from 'cortex-kg';

export default {
  name: 'CortexKG',
  props: {
    width: { type: Number, default: 800 },
    height: { type: Number, default: 600 },
    config: { type: Object, default: () => ({}) }
  },
  data() {
    return {
      loading: true,
      error: null,
      concepts: []
    };
  },
  async mounted() {
    try {
      await wasmLoader.init({
        memorySize: 32 * 1024 * 1024,
        enableSIMD: true,
        ...this.config
      });
      
      cortexVisualizer.init(this.$refs.canvas, {
        width: this.width,
        height: this.height
      });
      
      this.loading = false;
    } catch (err) {
      this.error = err.message;
      this.loading = false;
    }
  },
  methods: {
    async allocateRandomConcept() {
      try {
        const concept = await cortexWrapper.allocateConcept(
          `concept-${Date.now()}`,
          1024 + Math.random() * 1024
        );
        this.concepts.push(concept);
        this.$emit('concept-allocated', concept);
      } catch (err) {
        this.error = err.message;
      }
    },
    
    async clearAllConcepts() {
      try {
        for (const concept of this.concepts) {
          await cortexWrapper.deallocateConcept(concept.id);
        }
        this.concepts = [];
        this.$emit('concepts-cleared');
      } catch (err) {
        this.error = err.message;
      }
    },
    
    handleCanvasClick(event) {
      const rect = this.$refs.canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      this.$emit('canvas-click', { x, y });
    }
  }
};
</script>

<style scoped>
.cortex-kg-container {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.loading, .error {
  padding: 20px;
  text-align: center;
}

.error {
  color: #d32f2f;
  background: #ffebee;
  border-radius: 4px;
}

.controls {
  margin-top: 10px;
}

.controls button {
  margin: 0 5px;
  padding: 8px 16px;
  background: #1976d2;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.controls button:hover {
  background: #1565c0;
}
</style>
```

### Angular Integration
```typescript
// cortex-kg.service.ts
import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';
import { wasmLoader, cortexWrapper, Concept } from 'cortex-kg';

@Injectable({
  providedIn: 'root'
})
export class CortexKGService {
  private loadedSubject = new BehaviorSubject<boolean>(false);
  private errorSubject = new BehaviorSubject<string | null>(null);
  
  public loaded$ = this.loadedSubject.asObservable();
  public error$ = this.errorSubject.asObservable();
  
  constructor() {
    this.initialize();
  }
  
  private async initialize(): Promise<void> {
    try {
      await wasmLoader.init({
        memorySize: 32 * 1024 * 1024,
        enableSIMD: true
      });
      this.loadedSubject.next(true);
    } catch (error) {
      this.errorSubject.next(error.message);
    }
  }
  
  async allocateConcept(name: string, size: number, metadata?: any): Promise<Concept> {
    if (!this.loadedSubject.value) {
      throw new Error('CortexKG not initialized');
    }
    return cortexWrapper.allocateConcept(name, size, metadata);
  }
  
  async deallocateConcept(conceptId: string): Promise<void> {
    return cortexWrapper.deallocateConcept(conceptId);
  }
  
  async queryConcepts(query: string): Promise<Concept[]> {
    return cortexWrapper.queryConcepts(query);
  }
}

// cortical-visualization.component.ts
import { Component, ElementRef, ViewChild, OnInit, Input } from '@angular/core';
import { CortexKGService } from './cortex-kg.service';
import { cortexVisualizer } from 'cortex-kg';

@Component({
  selector: 'app-cortical-visualization',
  template: `
    <div class="visualization-container">
      <canvas #canvas [width]="width" [height]="height"></canvas>
      <div class="status" *ngIf="error">{{ error }}</div>
    </div>
  `,
  styles: [`
    .visualization-container {
      display: flex;
      flex-direction: column;
      align-items: center;
    }
    canvas {
      border: 1px solid #ccc;
    }
    .status {
      margin-top: 10px;
      padding: 10px;
      background: #ffebee;
      color: #d32f2f;
      border-radius: 4px;
    }
  `]
})
export class CorticalVisualizationComponent implements OnInit {
  @ViewChild('canvas', { static: true }) canvasRef!: ElementRef<HTMLCanvasElement>;
  @Input() width = 800;
  @Input() height = 600;
  
  error: string | null = null;
  
  constructor(private cortexKGService: CortexKGService) {}
  
  ngOnInit(): void {
    this.cortexKGService.loaded$.subscribe(loaded => {
      if (loaded) {
        this.initializeVisualization();
      }
    });
    
    this.cortexKGService.error$.subscribe(error => {
      this.error = error;
    });
  }
  
  private initializeVisualization(): void {
    cortexVisualizer.init(this.canvasRef.nativeElement, {
      width: this.width,
      height: this.height,
      columnSize: 4
    });
  }
}
```

### Node.js Integration (Server-side)
```javascript
// server.js - Node.js backend integration
const { wasmLoader, cortexWrapper } = require('cortex-kg/dist/cortex-kg-node');
const express = require('express');
const cors = require('cors');

class CortexKGServer {
  constructor() {
    this.app = express();
    this.isInitialized = false;
    this.setupMiddleware();
    this.setupRoutes();
  }
  
  setupMiddleware() {
    this.app.use(cors());
    this.app.use(express.json());
    this.app.use(express.static('public'));
  }
  
  setupRoutes() {
    // Health check
    this.app.get('/health', (req, res) => {
      res.json({
        status: 'ok',
        cortexKGInitialized: this.isInitialized,
        timestamp: new Date().toISOString()
      });
    });
    
    // Initialize CortexKG
    this.app.post('/api/initialize', async (req, res) => {
      try {
        if (!this.isInitialized) {
          await wasmLoader.init({
            memorySize: 64 * 1024 * 1024, // Server can handle more memory
            enableSIMD: true
          });
          this.isInitialized = true;
        }
        res.json({ success: true, message: 'CortexKG initialized' });
      } catch (error) {
        res.status(500).json({ success: false, error: error.message });
      }
    });
    
    // Allocate concept
    this.app.post('/api/concepts', async (req, res) => {
      try {
        this.ensureInitialized();
        const { name, size, metadata } = req.body;
        const concept = await cortexWrapper.allocateConcept(name, size, metadata);
        res.json({ success: true, concept });
      } catch (error) {
        res.status(500).json({ success: false, error: error.message });
      }
    });
    
    // Query concepts
    this.app.get('/api/concepts/search', async (req, res) => {
      try {
        this.ensureInitialized();
        const { q, limit = 10 } = req.query;
        const concepts = await cortexWrapper.queryConcepts(q, { maxResults: parseInt(limit) });
        res.json({ success: true, concepts });
      } catch (error) {
        res.status(500).json({ success: false, error: error.message });
      }
    });
    
    // Memory usage
    this.app.get('/api/memory', (req, res) => {
      try {
        this.ensureInitialized();
        const memoryStats = wasmLoader.getMemoryUsage();
        res.json({ success: true, memory: memoryStats });
      } catch (error) {
        res.status(500).json({ success: false, error: error.message });
      }
    });
  }
  
  ensureInitialized() {
    if (!this.isInitialized) {
      throw new Error('CortexKG not initialized. Call /api/initialize first.');
    }
  }
  
  async start(port = 3000) {
    // Initialize CortexKG on server start
    try {
      await wasmLoader.init({
        memorySize: 64 * 1024 * 1024,
        enableSIMD: true
      });
      this.isInitialized = true;
      console.log('✓ CortexKG initialized successfully');
    } catch (error) {
      console.error('❌ Failed to initialize CortexKG:', error);
    }
    
    this.app.listen(port, () => {
      console.log(`🚀 CortexKG server running on port ${port}`);
    });
  }
}

// Start server
const server = new CortexKGServer();
server.start(process.env.PORT || 3000);

module.exports = CortexKGServer;
```

## Configuration Options

### Basic Configuration
```javascript
const config = {
  // Memory allocation (default: 16MB)
  memorySize: 32 * 1024 * 1024, // 32MB
  
  // SIMD optimization (default: true if supported)
  enableSIMD: true,
  
  // Threading support (default: false)
  enableThreads: false,
  
  // Debug mode (default: false)
  debug: process.env.NODE_ENV === 'development',
  
  // Custom WASM file path (default: auto-detected)
  wasmPath: './custom-cortex-kg.wasm',
  
  // Initialization timeout (default: 30000ms)
  timeout: 30000
};

await wasmLoader.init(config);
```

### Advanced Configuration
```javascript
const advancedConfig = {
  memory: {
    initialSize: 16 * 1024 * 1024,      // 16MB initial
    maxSize: 128 * 1024 * 1024,         // 128MB maximum
    growthIncrement: 8 * 1024 * 1024,   // 8MB growth increments
    pressureThreshold: 0.8               // Trigger cleanup at 80% usage
  },
  
  performance: {
    enableSIMD: true,
    enableThreads: false,
    enableBulkMemory: true,
    simdChunkSize: 1024,
    batchSize: 100
  },
  
  storage: {
    enablePersistence: true,
    databaseName: 'cortex-kg-storage',
    version: 1,
    quotaLimit: 50 * 1024 * 1024,       // 50MB IndexedDB quota
    syncInterval: 5000                   // Sync every 5 seconds
  },
  
  visualization: {
    enableRealtime: true,
    frameRate: 60,
    columnSize: 4,
    animationDuration: 500,
    theme: 'dark'
  },
  
  debugging: {
    enableLogging: true,
    logLevel: 'info', // 'debug', 'info', 'warn', 'error'
    enableProfiling: false,
    memoryTracking: true
  }
};

// Apply advanced configuration
await wasmLoader.init(advancedConfig);
```

### Environment-specific Configuration
```javascript
// Development configuration
const devConfig = {
  memorySize: 16 * 1024 * 1024,
  debug: true,
  enableProfiling: true,
  logLevel: 'debug',
  enableHotReload: true
};

// Production configuration
const prodConfig = {
  memorySize: 64 * 1024 * 1024,
  debug: false,
  enableProfiling: false,
  logLevel: 'error',
  enableCompression: true,
  enableCaching: true
};

// Testing configuration
const testConfig = {
  memorySize: 8 * 1024 * 1024,
  debug: true,
  enableMocking: true,
  timeout: 5000
};

const config = process.env.NODE_ENV === 'production' ? prodConfig :
               process.env.NODE_ENV === 'test' ? testConfig : devConfig;
```

## Common Patterns

### Pattern 1: Lazy Loading
```javascript
class LazyCorRexKG {
  constructor() {
    this.initialized = false;
    this.initPromise = null;
  }
  
  async ensureInitialized() {
    if (this.initialized) return;
    
    if (!this.initPromise) {
      this.initPromise = this.initialize();
    }
    
    await this.initPromise;
  }
  
  async initialize() {
    await wasmLoader.init({
      memorySize: 32 * 1024 * 1024,
      enableSIMD: true
    });
    this.initialized = true;
  }
  
  async allocateConcept(name, size, metadata) {
    await this.ensureInitialized();
    return cortexWrapper.allocateConcept(name, size, metadata);
  }
}

const lazyCortex = new LazyCortexKG();
export default lazyCortex;
```

### Pattern 2: Memory Management
```javascript
class ManagedCortexKG {
  constructor(config = {}) {
    this.concepts = new Map();
    this.memoryThreshold = config.memoryThreshold || 0.8;
    this.cleanupInterval = config.cleanupInterval || 10000;
    this.startMonitoring();
  }
  
  async allocateConcept(name, size, metadata = {}) {
    const concept = await cortexWrapper.allocateConcept(name, size, {
      ...metadata,
      lastAccessed: Date.now()
    });
    
    this.concepts.set(concept.id, concept);
    return concept;
  }
  
  async deallocateConcept(conceptId) {
    await cortexWrapper.deallocateConcept(conceptId);
    this.concepts.delete(conceptId);
  }
  
  startMonitoring() {
    setInterval(() => {
      this.checkMemoryPressure();
    }, this.cleanupInterval);
  }
  
  async checkMemoryPressure() {
    const stats = wasmLoader.getMemoryUsage();
    
    if (stats.pressure > this.memoryThreshold) {
      console.warn('High memory pressure, starting cleanup');
      await this.performCleanup();
    }
  }
  
  async performCleanup() {
    const now = Date.now();
    const staleThreshold = 5 * 60 * 1000; // 5 minutes
    
    const staleConcepts = Array.from(this.concepts.values())
      .filter(concept => now - concept.metadata.lastAccessed > staleThreshold)
      .sort((a, b) => a.metadata.lastAccessed - b.metadata.lastAccessed);
    
    const conceptsToRemove = staleConcepts.slice(0, Math.ceil(staleConcepts.length * 0.3));
    
    for (const concept of conceptsToRemove) {
      await this.deallocateConcept(concept.id);
    }
    
    console.log(`Cleaned up ${conceptsToRemove.length} stale concepts`);
  }
}
```

### Pattern 3: Event-driven Architecture
```javascript
class EventDrivenCortexKG extends EventTarget {
  constructor() {
    super();
    this.concepts = new Map();
  }
  
  async allocateConcept(name, size, metadata) {
    const concept = await cortexWrapper.allocateConcept(name, size, metadata);
    this.concepts.set(concept.id, concept);
    
    this.dispatchEvent(new CustomEvent('conceptAllocated', {
      detail: { concept }
    }));
    
    return concept;
  }
  
  async performSpatialPooling(inputPattern, options) {
    this.dispatchEvent(new CustomEvent('spatialPoolingStarted', {
      detail: { inputPattern, options }
    }));
    
    const result = await cortexWrapper.spatialPooling(inputPattern, options);
    
    this.dispatchEvent(new CustomEvent('spatialPoolingCompleted', {
      detail: { result, inputPattern }
    }));
    
    return result;
  }
  
  onConceptAllocated(callback) {
    this.addEventListener('conceptAllocated', callback);
  }
  
  onSpatialPoolingCompleted(callback) {
    this.addEventListener('spatialPoolingCompleted', callback);
  }
}

// Usage
const cortex = new EventDrivenCortexKG();

cortex.onConceptAllocated((event) => {
  console.log('New concept allocated:', event.detail.concept);
});

cortex.onSpatialPoolingCompleted((event) => {
  console.log('Spatial pooling result:', event.detail.result);
});
```

### Pattern 4: Batch Processing
```javascript
class BatchCortexKG {
  constructor(batchSize = 10) {
    this.batchSize = batchSize;
    this.allocationQueue = [];
    this.dealocationQueue = [];
    this.processingBatch = false;
  }
  
  async allocateConcept(name, size, metadata) {
    return new Promise((resolve, reject) => {
      this.allocationQueue.push({
        name, size, metadata, resolve, reject
      });
      
      this.processBatch();
    });
  }
  
  async deallocateConcept(conceptId) {
    return new Promise((resolve, reject) => {
      this.dealocationQueue.push({
        conceptId, resolve, reject
      });
      
      this.processBatch();
    });
  }
  
  async processBatch() {
    if (this.processingBatch) return;
    
    if (this.allocationQueue.length >= this.batchSize ||
        this.dealocationQueue.length >= this.batchSize) {
      this.processingBatch = true;
      
      try {
        await this.processAllocations();
        await this.processDeallocations();
      } finally {
        this.processingBatch = false;
      }
    }
  }
  
  async processAllocations() {
    const batch = this.allocationQueue.splice(0, this.batchSize);
    
    const promises = batch.map(async (item) => {
      try {
        const concept = await cortexWrapper.allocateConcept(
          item.name, item.size, item.metadata
        );
        item.resolve(concept);
      } catch (error) {
        item.reject(error);
      }
    });
    
    await Promise.all(promises);
  }
  
  async processDeallocations() {
    const batch = this.dealocationQueue.splice(0, this.batchSize);
    
    const promises = batch.map(async (item) => {
      try {
        await cortexWrapper.deallocateConcept(item.conceptId);
        item.resolve();
      } catch (error) {
        item.reject(error);
      }
    });
    
    await Promise.all(promises);
  }
}
```
```

2. **Create framework-specific integration templates**
```bash
# integration-templates/create-templates.sh
#!/bin/bash

echo "Creating CortexKG integration templates..."

# Create React template
mkdir -p templates/react-cortexkg
cat > templates/react-cortexkg/package.json << 'EOF'
{
  "name": "react-cortexkg-template",
  "version": "1.0.0",
  "private": true,
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1",
    "cortex-kg": "^1.0.0",
    "@types/cortex-kg": "^1.0.0"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject"
  },
  "eslintConfig": {
    "extends": [
      "react-app",
      "react-app/jest"
    ]
  },
  "browserslist": {
    "production": [
      ">0.2%",
      "not dead",
      "not op_mini all"
    ],
    "development": [
      "last 1 chrome version",
      "last 1 firefox version",
      "last 1 safari version"
    ]
  }
}
EOF

# Create Vue template
mkdir -p templates/vue-cortexkg
cat > templates/vue-cortexkg/package.json << 'EOF'
{
  "name": "vue-cortexkg-template",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "serve": "vue-cli-service serve",
    "build": "vue-cli-service build",
    "test:unit": "vue-cli-service test:unit",
    "lint": "vue-cli-service lint"
  },
  "dependencies": {
    "core-js": "^3.8.3",
    "vue": "^3.2.13",
    "cortex-kg": "^1.0.0"
  },
  "devDependencies": {
    "@babel/core": "^7.12.16",
    "@babel/eslint-parser": "^7.12.16",
    "@vue/cli-plugin-babel": "~5.0.0",
    "@vue/cli-plugin-eslint": "~5.0.0",
    "@vue/cli-plugin-unit-jest": "~5.0.0",
    "@vue/cli-service": "~5.0.0",
    "@vue/test-utils": "^2.0.0-0",
    "@vue/vue3-jest": "^27.0.0-alpha.1",
    "babel-jest": "^27.0.6",
    "eslint": "^7.32.0",
    "eslint-plugin-vue": "^8.0.3",
    "jest": "^27.0.5"
  }
}
EOF

# Create Angular template
mkdir -p templates/angular-cortexkg
cat > templates/angular-cortexkg/package.json << 'EOF'
{
  "name": "angular-cortexkg-template",
  "version": "1.0.0",
  "scripts": {
    "ng": "ng",
    "start": "ng serve",
    "build": "ng build",
    "watch": "ng build --watch --configuration development",
    "test": "ng test"
  },
  "private": true,
  "dependencies": {
    "@angular/animations": "^15.2.0",
    "@angular/common": "^15.2.0",
    "@angular/compiler": "^15.2.0",
    "@angular/core": "^15.2.0",
    "@angular/forms": "^15.2.0",
    "@angular/platform-browser": "^15.2.0",
    "@angular/platform-browser-dynamic": "^15.2.0",
    "@angular/router": "^15.2.0",
    "rxjs": "~7.8.0",
    "tslib": "^2.3.0",
    "zone.js": "~0.12.0",
    "cortex-kg": "^1.0.0",
    "@types/cortex-kg": "^1.0.0"
  },
  "devDependencies": {
    "@angular-devkit/build-angular": "^15.2.4",
    "@angular/cli": "~15.2.4",
    "@angular/compiler-cli": "^15.2.0",
    "@types/jasmine": "~4.3.0",
    "jasmine-core": "~4.5.0",
    "karma": "~6.4.0",
    "karma-chrome-launcher": "~3.1.0",
    "karma-coverage": "~2.2.0",
    "karma-jasmine": "~5.1.0",
    "karma-jasmine-html-reporter": "~2.0.0",
    "typescript": "~4.9.4"
  }
}
EOF

# Create Node.js template
mkdir -p templates/nodejs-cortexkg
cat > templates/nodejs-cortexkg/package.json << 'EOF'
{
  "name": "nodejs-cortexkg-template",
  "version": "1.0.0",
  "description": "Node.js server with CortexKG integration",
  "main": "server.js",
  "scripts": {
    "start": "node server.js",
    "dev": "nodemon server.js",
    "test": "jest"
  },
  "dependencies": {
    "express": "^4.18.2",
    "cors": "^2.8.5",
    "helmet": "^7.0.0",
    "compression": "^1.7.4",
    "cortex-kg": "^1.0.0"
  },
  "devDependencies": {
    "nodemon": "^3.0.1",
    "jest": "^29.5.0",
    "supertest": "^6.3.3"
  },
  "keywords": ["cortexkg", "nodejs", "server"],
  "author": "Your Name",
  "license": "MIT"
}
EOF

echo "✓ Integration templates created successfully"
```

3. **Create deployment configuration examples**
```dockerfile
# Docker configuration example
# Dockerfile.cortexkg
FROM node:18-alpine

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm ci --only=production

# Copy application files
COPY . .

# Create directory for WASM files
RUN mkdir -p public/wasm

# Copy WASM binaries
COPY dist/*.wasm public/wasm/

# Set environment variables
ENV NODE_ENV=production
ENV CORTEXKG_MEMORY_SIZE=67108864
ENV CORTEXKG_ENABLE_SIMD=true

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:3000/health || exit 1

# Start application
CMD ["npm", "start"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  cortexkg-app:
    build:
      context: .
      dockerfile: Dockerfile.cortexkg
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - CORTEXKG_MEMORY_SIZE=67108864
      - CORTEXKG_ENABLE_SIMD=true
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
  
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl/certs
    depends_on:
      - cortexkg-app
    restart: unless-stopped
```

```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream cortexkg_backend {
        server cortexkg-app:3000;
    }
    
    # WASM MIME type
    location ~* \.wasm$ {
        add_header Content-Type application/wasm;
        add_header Cross-Origin-Embedder-Policy require-corp;
        add_header Cross-Origin-Opener-Policy same-origin;
    }
    
    server {
        listen 80;
        server_name your-domain.com;
        
        # Redirect to HTTPS
        return 301 https://$server_name$request_uri;
    }
    
    server {
        listen 443 ssl http2;
        server_name your-domain.com;
        
        ssl_certificate /etc/ssl/certs/cert.pem;
        ssl_certificate_key /etc/ssl/certs/key.pem;
        
        # Security headers for WASM
        add_header Cross-Origin-Embedder-Policy require-corp;
        add_header Cross-Origin-Opener-Policy same-origin;
        
        location /api/ {
            proxy_pass http://cortexkg_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        location / {
            proxy_pass http://cortexkg_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

4. **Create testing integration examples**
```javascript
// tests/integration/cortexkg-integration.test.js
const { wasmLoader, cortexWrapper } = require('cortex-kg');

describe('CortexKG Integration Tests', () => {
  beforeAll(async () => {
    // Initialize CortexKG before tests
    await wasmLoader.init({
      memorySize: 16 * 1024 * 1024,
      enableSIMD: true,
      debug: true
    });
  });

  afterAll(async () => {
    // Cleanup after tests
    const stats = wasmLoader.getMemoryUsage();
    console.log('Final memory usage:', stats);
  });

  describe('Basic Integration', () => {
    test('should allocate and deallocate concepts', async () => {
      const concept = await cortexWrapper.allocateConcept('test-concept', 1024);
      expect(concept).toBeDefined();
      expect(concept.id).toBeTruthy();
      expect(concept.name).toBe('test-concept');
      expect(concept.size).toBe(1024);

      await cortexWrapper.deallocateConcept(concept.id);
    });

    test('should perform spatial pooling', async () => {
      const inputPattern = new Array(1000).fill(0);
      // Set 20 random bits to 1
      for (let i = 0; i < 20; i++) {
        inputPattern[Math.floor(Math.random() * 1000)] = 1;
      }

      const result = await cortexWrapper.spatialPooling(inputPattern);
      expect(result).toBeDefined();
      expect(result.activeColumns).toBeDefined();
      expect(Array.isArray(result.activeColumns)).toBe(true);
    });
  });

  describe('Error Handling', () => {
    test('should handle invalid concept allocation', async () => {
      await expect(
        cortexWrapper.allocateConcept('', -1)
      ).rejects.toThrow();
    });

    test('should handle non-existent concept deallocation', async () => {
      await expect(
        cortexWrapper.deallocateConcept('non-existent-id')
      ).rejects.toThrow();
    });
  });

  describe('Performance', () => {
    test('should handle batch operations efficiently', async () => {
      const startTime = performance.now();
      
      const promises = [];
      for (let i = 0; i < 100; i++) {
        promises.push(
          cortexWrapper.allocateConcept(`batch-${i}`, 512)
        );
      }
      
      const concepts = await Promise.all(promises);
      const allocationTime = performance.now() - startTime;
      
      expect(concepts).toHaveLength(100);
      expect(allocationTime).toBeLessThan(5000); // Should complete in 5 seconds
      
      // Cleanup
      for (const concept of concepts) {
        await cortexWrapper.deallocateConcept(concept.id);
      }
    });
  });
});

// tests/e2e/browser-integration.test.js
const puppeteer = require('puppeteer');

describe('Browser Integration E2E Tests', () => {
  let browser;
  let page;

  beforeAll(async () => {
    browser = await puppeteer.launch({
      headless: false, // Set to true for CI
      args: [
        '--enable-features=SharedArrayBuffer',
        '--enable-unsafe-webassembly',
        '--disable-web-security'
      ]
    });
    page = await browser.newPage();
  });

  afterAll(async () => {
    await browser.close();
  });

  test('should load CortexKG in browser', async () => {
    await page.goto('http://localhost:3000');
    
    // Wait for CortexKG to initialize
    await page.waitForFunction(() => window.wasmLoaded === true, { timeout: 30000 });
    
    const isLoaded = await page.evaluate(() => window.wasmLoaded);
    expect(isLoaded).toBe(true);
  });

  test('should create and visualize concepts', async () => {
    await page.goto('http://localhost:3000');
    await page.waitForFunction(() => window.wasmLoaded === true);
    
    // Create a concept through the UI
    await page.click('[data-testid="add-concept-button"]');
    await page.type('[data-testid="concept-name-input"]', 'e2e-test-concept');
    await page.type('[data-testid="concept-size-input"]', '1024');
    await page.click('[data-testid="create-concept-button"]');
    
    // Check if concept appears in the list
    await page.waitForSelector('[data-testid="concept-list-item"]');
    
    const conceptCount = await page.$$eval('[data-testid="concept-list-item"]', 
      items => items.length
    );
    expect(conceptCount).toBeGreaterThan(0);
  });

  test('should handle mobile responsive design', async () => {
    // Test mobile viewport
    await page.setViewport({ width: 375, height: 667 });
    await page.goto('http://localhost:3000');
    await page.waitForFunction(() => window.wasmLoaded === true);
    
    // Check if mobile layout is active
    const isMobileLayout = await page.evaluate(() => {
      return window.getComputedStyle(document.body)
        .getPropertyValue('--mobile-layout') === 'true';
    });
    
    expect(isMobileLayout).toBe(true);
    
    // Test touch interactions
    const canvas = await page.$('[data-testid="cortical-canvas"]');
    await canvas.tap();
    
    // Verify touch interaction was registered
    const touchHandled = await page.evaluate(() => 
      window.lastTouchInteraction !== null
    );
    expect(touchHandled).toBe(true);
  });
});
```

5. **Create troubleshooting and debugging guides**
```javascript
// debug/cortexkg-debugger.js
class CortexKGDebugger {
  constructor() {
    this.memoryHistory = [];
    this.performanceMetrics = new Map();
    this.errorLog = [];
    this.isRecording = false;
  }
  
  startRecording() {
    this.isRecording = true;
    this.setupMemoryMonitoring();
    this.setupPerformanceMonitoring();
    this.setupErrorTracking();
    console.log('🐛 CortexKG debugging started');
  }
  
  stopRecording() {
    this.isRecording = false;
    console.log('🐛 CortexKG debugging stopped');
    return this.generateReport();
  }
  
  setupMemoryMonitoring() {
    const originalInit = wasmLoader.init;
    const originalAllocate = cortexWrapper.allocateConcept;
    const originalDeallocate = cortexWrapper.deallocateConcept;
    
    wasmLoader.init = async (...args) => {
      const result = await originalInit.apply(this, args);
      this.recordMemorySnapshot('init');
      return result;
    };
    
    cortexWrapper.allocateConcept = async (...args) => {
      const startTime = performance.now();
      const result = await originalAllocate.apply(this, args);
      const endTime = performance.now();
      
      this.recordPerformanceMetric('allocateConcept', endTime - startTime);
      this.recordMemorySnapshot('allocate', result.id);
      return result;
    };
    
    cortexWrapper.deallocateConcept = async (...args) => {
      const startTime = performance.now();
      const result = await originalDeallocate.apply(this, args);
      const endTime = performance.now();
      
      this.recordPerformanceMetric('deallocateConcept', endTime - startTime);
      this.recordMemorySnapshot('deallocate', args[0]);
      return result;
    };
  }
  
  setupPerformanceMonitoring() {
    // Monitor spatial pooling performance
    const originalSpatialPooling = cortexWrapper.spatialPooling;
    
    cortexWrapper.spatialPooling = async (...args) => {
      const startTime = performance.now();
      const result = await originalSpatialPooling.apply(this, args);
      const endTime = performance.now();
      
      this.recordPerformanceMetric('spatialPooling', endTime - startTime, {
        inputSize: args[0]?.length || 0
      });
      
      return result;
    };
  }
  
  setupErrorTracking() {
    const originalError = console.error;
    
    console.error = (...args) => {
      if (this.isRecording) {
        this.errorLog.push({
          timestamp: Date.now(),
          message: args.join(' '),
          stack: new Error().stack
        });
      }
      originalError.apply(console, args);
    };
  }
  
  recordMemorySnapshot(operation, conceptId = null) {
    if (!this.isRecording) return;
    
    const stats = wasmLoader.getMemoryUsage();
    this.memoryHistory.push({
      timestamp: Date.now(),
      operation,
      conceptId,
      ...stats
    });
  }
  
  recordPerformanceMetric(operation, duration, metadata = {}) {
    if (!this.isRecording) return;
    
    if (!this.performanceMetrics.has(operation)) {
      this.performanceMetrics.set(operation, []);
    }
    
    this.performanceMetrics.get(operation).push({
      timestamp: Date.now(),
      duration,
      ...metadata
    });
  }
  
  generateReport() {
    const report = {
      summary: this.generateSummary(),
      memoryAnalysis: this.analyzeMemoryUsage(),
      performanceAnalysis: this.analyzePerformance(),
      errors: this.errorLog,
      recommendations: this.generateRecommendations()
    };
    
    console.table(report.summary);
    return report;
  }
  
  generateSummary() {
    return {
      recordingDuration: this.memoryHistory.length > 0 ? 
        this.memoryHistory[this.memoryHistory.length - 1].timestamp - this.memoryHistory[0].timestamp : 0,
      memorySnapshots: this.memoryHistory.length,
      operationsTracked: Array.from(this.performanceMetrics.keys()).length,
      errorsLogged: this.errorLog.length,
      peakMemoryUsage: Math.max(...this.memoryHistory.map(m => m.usedSize)),
      averageMemoryPressure: this.memoryHistory.length > 0 ?
        this.memoryHistory.reduce((sum, m) => sum + m.pressure, 0) / this.memoryHistory.length : 0
    };
  }
  
  analyzeMemoryUsage() {
    if (this.memoryHistory.length === 0) return {};
    
    const memoryLeaks = this.detectMemoryLeaks();
    const memoryPressureSpikes = this.detectMemoryPressureSpikes();
    
    return {
      memoryLeaks,
      memoryPressureSpikes,
      memoryGrowthRate: this.calculateMemoryGrowthRate()
    };
  }
  
  analyzePerformance() {
    const analysis = {};
    
    for (const [operation, metrics] of this.performanceMetrics) {
      const durations = metrics.map(m => m.duration);
      analysis[operation] = {
        count: metrics.length,
        averageDuration: durations.reduce((a, b) => a + b, 0) / durations.length,
        minDuration: Math.min(...durations),
        maxDuration: Math.max(...durations),
        slowOperations: metrics.filter(m => m.duration > 100) // Operations over 100ms
      };
    }
    
    return analysis;
  }
  
  detectMemoryLeaks() {
    // Simple leak detection: memory that never gets freed
    const allocations = this.memoryHistory.filter(m => m.operation === 'allocate');
    const deallocations = this.memoryHistory.filter(m => m.operation === 'deallocate');
    
    const allocatedConcepts = new Set(allocations.map(a => a.conceptId));
    const deallocatedConcepts = new Set(deallocations.map(d => d.conceptId));
    
    const potentialLeaks = Array.from(allocatedConcepts).filter(id => 
      !deallocatedConcepts.has(id)
    );
    
    return potentialLeaks;
  }
  
  detectMemoryPressureSpikes() {
    return this.memoryHistory.filter(m => m.pressure > 0.8);
  }
  
  calculateMemoryGrowthRate() {
    if (this.memoryHistory.length < 2) return 0;
    
    const first = this.memoryHistory[0];
    const last = this.memoryHistory[this.memoryHistory.length - 1];
    const timeDiff = last.timestamp - first.timestamp;
    const memoryDiff = last.usedSize - first.usedSize;
    
    return timeDiff > 0 ? memoryDiff / timeDiff : 0; // bytes per millisecond
  }
  
  generateRecommendations() {
    const recommendations = [];
    
    const summary = this.generateSummary();
    
    if (summary.averageMemoryPressure > 0.7) {
      recommendations.push({
        type: 'memory',
        priority: 'high',
        message: 'High average memory pressure detected. Consider increasing memory allocation or implementing more aggressive cleanup.'
      });
    }
    
    if (summary.errorsLogged > 0) {
      recommendations.push({
        type: 'error',
        priority: 'high',
        message: `${summary.errorsLogged} errors detected. Review error log for issues.`
      });
    }
    
    const performanceAnalysis = this.analyzePerformance();
    for (const [operation, metrics] of Object.entries(performanceAnalysis)) {
      if (metrics.averageDuration > 100) {
        recommendations.push({
          type: 'performance',
          priority: 'medium',
          message: `${operation} operations averaging ${metrics.averageDuration.toFixed(2)}ms. Consider optimization.`
        });
      }
    }
    
    return recommendations;
  }
}

// Usage example
const debugger = new CortexKGDebugger();
debugger.startRecording();

// ... run your CortexKG operations ...

const report = debugger.stopRecording();
console.log('Debug Report:', report);
```

## Expected Outputs
- Complete integration guide with framework-specific examples and templates
- Step-by-step setup instructions for different deployment scenarios
- Configuration reference with environment-specific examples
- Common integration patterns and best practices documentation
- Debugging tools and troubleshooting guides for development teams

## Validation
1. Verify integration templates work correctly with popular frameworks (React, Vue, Angular, Node.js)
2. Test deployment configurations in containerized and cloud environments
3. Confirm configuration options produce expected behavior across different scenarios
4. Validate debugging tools provide actionable insights for performance optimization
5. Ensure integration guide covers all major use cases and edge cases developers might encounter

## Next Steps
- Proceed to micro-phase 9.48 (Example Apps)
- Publish integration guide to developer documentation portal
- Create video tutorials for complex integration scenarios