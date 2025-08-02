# Micro-Phase 9.46: API Documentation

## Objective
Create comprehensive API documentation with TypeScript definitions, method signatures, and usage examples for all CortexKG WASM bindings and JavaScript interfaces.

## Prerequisites
- Completed micro-phase 9.44 (Browser Compatibility Tests)
- WASM bindings fully implemented (phases 9.04-9.08)
- JavaScript API wrapper finalized (phases 9.16-9.20)

## Task Description
Generate complete API documentation covering all public interfaces, including TypeScript type definitions, method signatures, parameter descriptions, return values, and practical usage examples. Create auto-generated documentation from source code comments and manual examples for complex workflows.

## Specific Actions

1. **Generate TypeScript definition files**
```typescript
// cortex-kg.d.ts - Main API definitions
declare namespace CortexKG {
  interface WasmLoader {
    /**
     * Initialize the WASM module with optional configuration
     * @param config - Optional configuration object
     * @returns Promise that resolves when WASM is loaded
     */
    init(config?: WasmConfig): Promise<void>;
    
    /**
     * Get the current memory usage statistics
     * @returns Memory usage information
     */
    getMemoryUsage(): MemoryStats;
    
    /**
     * Reload the WASM module (for debugging/testing)
     * @returns Promise that resolves when reload is complete
     */
    reload(): Promise<void>;
    
    /**
     * Get WASM exports for direct low-level access
     * @returns Object containing all WASM exported functions
     */
    getExports(): Record<string, Function>;
  }

  interface CortexWrapper {
    /**
     * Allocate a new concept in the cortical structure
     * @param name - Unique identifier for the concept
     * @param size - Size allocation for the concept data
     * @param metadata - Optional metadata object
     * @returns Promise resolving to allocated concept
     */
    allocateConcept(name: string, size: number, metadata?: ConceptMetadata): Promise<Concept>;
    
    /**
     * Deallocate an existing concept and free its memory
     * @param conceptId - ID of the concept to deallocate
     * @returns Promise resolving when deallocation is complete
     */
    deallocateConcept(conceptId: string): Promise<void>;
    
    /**
     * Perform spatial pooling operation on input pattern
     * @param inputPattern - Binary array representing input pattern
     * @param options - Optional pooling configuration
     * @returns Promise resolving to spatial pooling result
     */
    spatialPooling(inputPattern: number[], options?: SpatialPoolingOptions): Promise<SpatialResult>;
    
    /**
     * Execute temporal memory sequence processing
     * @param sequence - Array of spatial patterns over time
     * @param prediction - Whether to return predictions
     * @returns Promise resolving to temporal memory result
     */
    temporalMemory(sequence: number[][], prediction?: boolean): Promise<TemporalResult>;
    
    /**
     * Query concepts using semantic search
     * @param query - Search query string or pattern
     * @param options - Optional query configuration
     * @returns Promise resolving to array of matching concepts
     */
    queryConcepts(query: string | number[], options?: QueryOptions): Promise<Concept[]>;
    
    /**
     * Store arbitrary data in WASM memory
     * @param data - Data to store (Uint8Array)
     * @returns Memory address where data was stored
     */
    storeInMemory(data: Uint8Array): number;
    
    /**
     * Retrieve data from WASM memory
     * @param address - Memory address to read from
     * @param length - Number of bytes to read
     * @returns Retrieved data as Uint8Array
     */
    retrieveFromMemory(address: number, length: number): Uint8Array;
  }

  interface StorageManager {
    /**
     * Store concept data persistently in IndexedDB
     * @param concept - Concept object to store
     * @returns Promise resolving when storage is complete
     */
    storeConcept(concept: Concept): Promise<void>;
    
    /**
     * Retrieve concept data from persistent storage
     * @param conceptId - ID of concept to retrieve
     * @returns Promise resolving to concept or null if not found
     */
    retrieveConcept(conceptId: string): Promise<Concept | null>;
    
    /**
     * Delete concept from persistent storage
     * @param conceptId - ID of concept to delete
     * @returns Promise resolving when deletion is complete
     */
    deleteConcept(conceptId: string): Promise<void>;
    
    /**
     * Get storage usage statistics
     * @returns Promise resolving to storage stats
     */
    getStorageStats(): Promise<StorageStats>;
    
    /**
     * Clear all stored data (use with caution)
     * @returns Promise resolving when clear is complete
     */
    clearAllData(): Promise<void>;
  }

  interface CorticalVisualizer {
    /**
     * Initialize the cortical visualization canvas
     * @param canvas - HTML canvas element
     * @param options - Optional visualization configuration
     */
    init(canvas: HTMLCanvasElement, options?: VisualizationOptions): void;
    
    /**
     * Render the current cortical state
     * @param force - Force redraw even if state hasn't changed
     */
    render(force?: boolean): void;
    
    /**
     * Update visualization with new cortical data
     * @param corticalData - New cortical state data
     */
    updateCorticalData(corticalData: CorticalData): void;
    
    /**
     * Set zoom level for the visualization
     * @param zoomLevel - Zoom level (1.0 = 100%)
     */
    setZoom(zoomLevel: number): void;
    
    /**
     * Pan the visualization to specific coordinates
     * @param x - X coordinate to pan to
     * @param y - Y coordinate to pan to
     */
    panTo(x: number, y: number): void;
    
    /**
     * Highlight specific concepts in the visualization
     * @param conceptIds - Array of concept IDs to highlight
     */
    highlightConcepts(conceptIds: string[]): void;
  }

  // Type definitions
  interface WasmConfig {
    memorySize?: number;
    enableSIMD?: boolean;
    enableThreads?: boolean;
    debug?: boolean;
  }

  interface MemoryStats {
    totalSize: number;
    usedSize: number;
    freeSize: number;
    pressure: number; // 0-1 indicating memory pressure
  }

  interface Concept {
    id: string;
    name: string;
    size: number;
    metadata?: ConceptMetadata;
    createdAt: Date;
    lastAccessed: Date;
    activationLevel: number;
  }

  interface ConceptMetadata {
    description?: string;
    tags?: string[];
    priority?: number;
    version?: number;
  }

  interface SpatialPoolingOptions {
    sparsity?: number;
    boostStrength?: number;
    dutyCyclePeriod?: number;
  }

  interface SpatialResult {
    activeColumns: number[];
    overlap: number[];
    boost: number[];
  }

  interface TemporalResult {
    activeCells: number[];
    predictiveCells: number[];
    prediction: number[];
  }

  interface QueryOptions {
    maxResults?: number;
    threshold?: number;
    includeMetadata?: boolean;
  }

  interface StorageStats {
    totalConcepts: number;
    totalSize: number;
    lastSync: Date;
  }

  interface VisualizationOptions {
    width?: number;
    height?: number;
    columnSize?: number;
    showActivations?: boolean;
    animationSpeed?: number;
  }

  interface CorticalData {
    columns: Column[];
    activations: number[];
    predictions: number[];
    timestamp: number;
  }

  interface Column {
    id: number;
    x: number;
    y: number;
    isActive: boolean;
    isPredictive: boolean;
    boost: number;
  }
}

// Global API access
declare const wasmLoader: CortexKG.WasmLoader;
declare const cortexWrapper: CortexKG.CortexWrapper;
declare const storageManager: CortexKG.StorageManager;
declare const cortexVisualizer: CortexKG.CorticalVisualizer;
```

2. **Create comprehensive API reference documentation**
```markdown
# CortexKG API Reference

## Overview
CortexKG provides a WebAssembly-based cortical computing engine with JavaScript bindings for building intelligent applications. The API is organized into four main modules:

- **WasmLoader**: Low-level WASM module management
- **CortexWrapper**: High-level cortical operations
- **StorageManager**: Persistent data management  
- **CorticalVisualizer**: Real-time visualization

## Getting Started

### Basic Initialization
```javascript
// Initialize the WASM module
await wasmLoader.init({
  memorySize: 64 * 1024 * 1024, // 64MB
  enableSIMD: true,
  enableThreads: false,
  debug: false
});

// Verify initialization
console.log('WASM loaded:', wasmLoader.getMemoryUsage());
```

### Simple Concept Allocation
```javascript
// Allocate a new concept
const concept = await cortexWrapper.allocateConcept('user-preference', 1000, {
  description: 'User preference patterns',
  tags: ['user', 'behavior'],
  priority: 5
});

console.log('Concept allocated:', concept.id);
```

## WasmLoader Module

### Methods

#### `init(config?: WasmConfig): Promise<void>`
Initializes the WebAssembly module with optional configuration.

**Parameters:**
- `config` (optional): Configuration object
  - `memorySize`: Initial memory size in bytes (default: 16MB)
  - `enableSIMD`: Enable SIMD optimizations (default: true)
  - `enableThreads`: Enable threading support (default: false)
  - `debug`: Enable debug mode (default: false)

**Returns:** Promise that resolves when initialization is complete

**Example:**
```javascript
await wasmLoader.init({
  memorySize: 32 * 1024 * 1024, // 32MB
  enableSIMD: true,
  debug: process.env.NODE_ENV === 'development'
});
```

#### `getMemoryUsage(): MemoryStats`
Returns current memory usage statistics.

**Returns:** MemoryStats object containing:
- `totalSize`: Total allocated memory
- `usedSize`: Currently used memory
- `freeSize`: Available free memory
- `pressure`: Memory pressure level (0-1)

**Example:**
```javascript
const stats = wasmLoader.getMemoryUsage();
if (stats.pressure > 0.8) {
  console.warn('High memory pressure:', stats);
}
```

## CortexWrapper Module

### Core Operations

#### `allocateConcept(name: string, size: number, metadata?: ConceptMetadata): Promise<Concept>`
Allocates a new concept in the cortical structure.

**Parameters:**
- `name`: Unique identifier for the concept
- `size`: Memory allocation size for concept data
- `metadata` (optional): Additional concept metadata

**Returns:** Promise resolving to the allocated Concept object

**Example:**
```javascript
const concept = await cortexWrapper.allocateConcept('product-category', 2048, {
  description: 'E-commerce product categorization',
  tags: ['commerce', 'ml'],
  priority: 8
});
```

#### `spatialPooling(inputPattern: number[], options?: SpatialPoolingOptions): Promise<SpatialResult>`
Performs spatial pooling on an input pattern.

**Parameters:**
- `inputPattern`: Binary array representing the input pattern
- `options` (optional): Spatial pooling configuration
  - `sparsity`: Target sparsity level (default: 0.02)
  - `boostStrength`: Boost strength for inactive columns (default: 2.0)
  - `dutyCyclePeriod`: Period for duty cycle calculation (default: 1000)

**Returns:** Promise resolving to SpatialResult with active columns and overlaps

**Example:**
```javascript
const inputPattern = new Array(2048).fill(0);
// Set some bits to 1 to represent active inputs
for (let i = 0; i < 40; i++) {
  inputPattern[Math.floor(Math.random() * 2048)] = 1;
}

const result = await cortexWrapper.spatialPooling(inputPattern, {
  sparsity: 0.02,
  boostStrength: 2.0
});

console.log('Active columns:', result.activeColumns.length);
```

### Query Operations

#### `queryConcepts(query: string | number[], options?: QueryOptions): Promise<Concept[]>`
Searches for concepts using semantic similarity.

**Parameters:**
- `query`: Search query (string or pattern array)
- `options` (optional): Query configuration
  - `maxResults`: Maximum number of results (default: 10)
  - `threshold`: Similarity threshold (default: 0.7)
  - `includeMetadata`: Include full metadata (default: true)

**Returns:** Promise resolving to array of matching concepts

**Example:**
```javascript
// Text-based query
const textResults = await cortexWrapper.queryConcepts('machine learning', {
  maxResults: 5,
  threshold: 0.8
});

// Pattern-based query
const pattern = [1, 0, 1, 1, 0, 0, 1, 0]; // Binary pattern
const patternResults = await cortexWrapper.queryConcepts(pattern, {
  maxResults: 3,
  threshold: 0.6
});
```

## StorageManager Module

### Persistence Operations

#### `storeConcept(concept: Concept): Promise<void>`
Stores a concept persistently in IndexedDB.

**Parameters:**
- `concept`: Concept object to store

**Returns:** Promise that resolves when storage is complete

**Example:**
```javascript
const concept = await cortexWrapper.allocateConcept('user-session', 1024);
await storageManager.storeConcept(concept);
console.log('Concept stored persistently');
```

#### `retrieveConcept(conceptId: string): Promise<Concept | null>`
Retrieves a concept from persistent storage.

**Parameters:**
- `conceptId`: Unique identifier of the concept

**Returns:** Promise resolving to Concept object or null if not found

**Example:**
```javascript
const stored = await storageManager.retrieveConcept('user-session');
if (stored) {
  console.log('Retrieved concept:', stored.name);
} else {
  console.log('Concept not found');
}
```

## CorticalVisualizer Module

### Visualization Setup

#### `init(canvas: HTMLCanvasElement, options?: VisualizationOptions): void`
Initializes the cortical visualization on a canvas element.

**Parameters:**
- `canvas`: HTML canvas element for rendering
- `options` (optional): Visualization configuration
  - `width`: Canvas width (default: canvas.width)
  - `height`: Canvas height (default: canvas.height)
  - `columnSize`: Size of individual columns (default: 4)
  - `showActivations`: Show activation animations (default: true)
  - `animationSpeed`: Animation speed multiplier (default: 1.0)

**Example:**
```javascript
const canvas = document.getElementById('cortical-canvas');
cortexVisualizer.init(canvas, {
  columnSize: 6,
  showActivations: true,
  animationSpeed: 1.5
});
```

#### `updateCorticalData(corticalData: CorticalData): void`
Updates the visualization with new cortical state data.

**Parameters:**
- `corticalData`: Current cortical state including columns and activations

**Example:**
```javascript
const corticalData = {
  columns: [
    { id: 0, x: 10, y: 10, isActive: true, isPredictive: false, boost: 1.0 },
    { id: 1, x: 16, y: 10, isActive: false, isPredictive: true, boost: 1.2 }
  ],
  activations: [0.8, 0.0, 0.6, 0.3],
  predictions: [0.2, 0.9, 0.1, 0.7],
  timestamp: Date.now()
};

cortexVisualizer.updateCorticalData(corticalData);
cortexVisualizer.render();
```

## Error Handling

### Common Error Types
```javascript
try {
  await cortexWrapper.allocateConcept('test', 1000000000); // Too large
} catch (error) {
  if (error.name === 'MemoryError') {
    console.error('Not enough memory:', error.message);
  } else if (error.name === 'ValidationError') {
    console.error('Invalid parameters:', error.message);
  } else {
    console.error('Unexpected error:', error);
  }
}
```

### Memory Management Best Practices
```javascript
// Monitor memory usage
setInterval(() => {
  const stats = wasmLoader.getMemoryUsage();
  if (stats.pressure > 0.9) {
    // Clean up unused concepts
    console.warn('High memory pressure, cleaning up...');
    // Implement cleanup logic
  }
}, 5000);

// Proper concept lifecycle
async function processUserData(userData) {
  const concept = await cortexWrapper.allocateConcept('temp-data', 1024);
  try {
    // Process data...
    await cortexWrapper.spatialPooling(userData);
  } finally {
    // Always clean up
    await cortexWrapper.deallocateConcept(concept.id);
  }
}
```

## Performance Tips

### Batch Operations
```javascript
// Instead of individual allocations
const concepts = await Promise.all([
  cortexWrapper.allocateConcept('concept1', 1024),
  cortexWrapper.allocateConcept('concept2', 1024),
  cortexWrapper.allocateConcept('concept3', 1024)
]);

// Use bulk operations when available
await cortexWrapper.batchAllocate([
  { name: 'concept1', size: 1024 },
  { name: 'concept2', size: 1024 },
  { name: 'concept3', size: 1024 }
]);
```

### Optimize Visualization Updates
```javascript
// Throttle visualization updates
let lastRender = 0;
function updateVisualization(data) {
  const now = Date.now();
  if (now - lastRender > 16) { // ~60fps
    cortexVisualizer.updateCorticalData(data);
    cortexVisualizer.render();
    lastRender = now;
  }
}
```
```

3. **Create JSDoc documentation generation script**
```javascript
// scripts/generate-docs.js - Documentation generation script
const fs = require('fs');
const path = require('path');

/**
 * Extract JSDoc comments from source files and generate API documentation
 */
class DocGenerator {
  constructor() {
    this.apiMethods = new Map();
    this.typeDefinitions = new Map();
  }

  /**
   * Parse a source file and extract documentation
   * @param {string} filePath - Path to source file
   */
  parseFile(filePath) {
    const content = fs.readFileSync(filePath, 'utf8');
    
    // Extract JSDoc comments
    const jsdocRegex = /\/\*\*([\s\S]*?)\*\//g;
    const methodRegex = /(?:async\s+)?(\w+)\s*\([^)]*\)\s*[:{]/g;
    
    let match;
    const docs = [];
    
    while ((match = jsdocRegex.exec(content)) !== null) {
      const comment = match[1];
      const docInfo = this.parseJSDocComment(comment);
      docs.push(docInfo);
    }
    
    return docs;
  }

  /**
   * Parse individual JSDoc comment
   * @param {string} comment - JSDoc comment text
   * @returns {Object} Parsed documentation object
   */
  parseJSDocComment(comment) {
    const lines = comment.split('\n').map(line => line.trim().replace(/^\*\s?/, ''));
    
    const doc = {
      description: '',
      params: [],
      returns: null,
      examples: [],
      since: null,
      deprecated: false
    };

    let currentSection = 'description';
    
    for (const line of lines) {
      if (line.startsWith('@param')) {
        const paramMatch = line.match(/@param\s+\{([^}]+)\}\s+(\w+)\s*-?\s*(.*)/);
        if (paramMatch) {
          doc.params.push({
            type: paramMatch[1],
            name: paramMatch[2],
            description: paramMatch[3]
          });
        }
        currentSection = 'param';
      } else if (line.startsWith('@returns')) {
        const returnMatch = line.match(/@returns?\s+\{([^}]+)\}\s*(.*)/);
        if (returnMatch) {
          doc.returns = {
            type: returnMatch[1],
            description: returnMatch[2]
          };
        }
        currentSection = 'returns';
      } else if (line.startsWith('@example')) {
        currentSection = 'example';
        doc.examples.push('');
      } else if (line.startsWith('@since')) {
        doc.since = line.replace('@since', '').trim();
      } else if (line.startsWith('@deprecated')) {
        doc.deprecated = true;
      } else if (currentSection === 'description' && line) {
        doc.description += line + ' ';
      } else if (currentSection === 'example' && doc.examples.length > 0) {
        doc.examples[doc.examples.length - 1] += line + '\n';
      }
    }

    doc.description = doc.description.trim();
    return doc;
  }

  /**
   * Generate markdown documentation
   * @param {string} outputPath - Output file path
   */
  generateMarkdown(outputPath) {
    const sourceFiles = [
      './src/js/wasm-loader.js',
      './src/js/cortex-wrapper.js',
      './src/js/storage-manager.js',
      './src/js/cortical-visualizer.js'
    ];

    let markdown = '# CortexKG API Documentation\n\n';
    markdown += '> Auto-generated from source code JSDoc comments\n\n';

    for (const file of sourceFiles) {
      if (fs.existsSync(file)) {
        const docs = this.parseFile(file);
        const moduleName = path.basename(file, '.js');
        
        markdown += `## ${moduleName}\n\n`;
        
        for (const doc of docs) {
          if (doc.description) {
            markdown += `### ${doc.methodName || 'Method'}\n\n`;
            markdown += `${doc.description}\n\n`;
            
            if (doc.params.length > 0) {
              markdown += '**Parameters:**\n';
              for (const param of doc.params) {
                markdown += `- \`${param.name}\` (${param.type}): ${param.description}\n`;
              }
              markdown += '\n';
            }
            
            if (doc.returns) {
              markdown += `**Returns:** ${doc.returns.type} - ${doc.returns.description}\n\n`;
            }
            
            if (doc.examples.length > 0) {
              markdown += '**Example:**\n```javascript\n';
              markdown += doc.examples[0].trim();
              markdown += '\n```\n\n';
            }
            
            if (doc.deprecated) {
              markdown += '> ⚠️ **Deprecated**: This method is deprecated and may be removed in future versions.\n\n';
            }
          }
        }
      }
    }

    fs.writeFileSync(outputPath, markdown);
    console.log(`Documentation generated: ${outputPath}`);
  }
}

// Generate documentation
const generator = new DocGenerator();
generator.generateMarkdown('./docs/api-reference.md');
```

4. **Create interactive API explorer**
```html
<!-- docs/api-explorer.html - Interactive API documentation -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CortexKG API Explorer</title>
    <style>
        body {
            font-family: 'Monaco', 'Menlo', monospace;
            margin: 0;
            padding: 20px;
            background: #1e1e1e;
            color: #d4d4d4;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 300px 1fr;
            gap: 20px;
        }
        .sidebar {
            background: #252526;
            padding: 20px;
            border-radius: 8px;
            height: fit-content;
        }
        .main {
            background: #252526;
            padding: 20px;
            border-radius: 8px;
        }
        .method-list {
            list-style: none;
            padding: 0;
            margin: 0;
        }
        .method-list li {
            padding: 8px 12px;
            cursor: pointer;
            border-radius: 4px;
            margin-bottom: 4px;
        }
        .method-list li:hover {
            background: #2d2d30;
        }
        .method-list li.active {
            background: #0e639c;
        }
        .method-details {
            display: none;
        }
        .method-details.active {
            display: block;
        }
        .code-block {
            background: #1e1e1e;
            padding: 15px;
            border-radius: 4px;
            margin: 10px 0;
            overflow-x: auto;
        }
        .try-button {
            background: #0e639c;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            margin: 10px 0;
        }
        .try-button:hover {
            background: #1177bb;
        }
        .result {
            background: #0f3460;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
            min-height: 100px;
            white-space: pre-wrap;
        }
        .param-input {
            background: #1e1e1e;
            color: #d4d4d4;
            border: 1px solid #3e3e3e;
            padding: 8px;
            border-radius: 4px;
            margin: 5px 0;
            width: 100%;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <h3>API Methods</h3>
            <ul class="method-list" id="methodList">
                <!-- Populated by JavaScript -->
            </ul>
        </div>
        <div class="main">
            <div id="methodDetails">
                <h2>Select a method to explore</h2>
                <p>Choose a method from the sidebar to see its documentation and try it interactively.</p>
            </div>
        </div>
    </div>

    <script>
        // API method definitions for interactive exploration
        const apiMethods = {
            wasmLoader: {
                init: {
                    description: 'Initialize the WASM module with optional configuration',
                    parameters: [
                        { name: 'config', type: 'WasmConfig', optional: true, default: '{}' }
                    ],
                    returns: 'Promise<void>',
                    example: `await wasmLoader.init({
  memorySize: 32 * 1024 * 1024,
  enableSIMD: true,
  debug: false
});`
                },
                getMemoryUsage: {
                    description: 'Get current memory usage statistics',
                    parameters: [],
                    returns: 'MemoryStats',
                    example: `const stats = wasmLoader.getMemoryUsage();
console.log('Memory pressure:', stats.pressure);`
                }
            },
            cortexWrapper: {
                allocateConcept: {
                    description: 'Allocate a new concept in the cortical structure',
                    parameters: [
                        { name: 'name', type: 'string', default: '"test-concept"' },
                        { name: 'size', type: 'number', default: '1024' },
                        { name: 'metadata', type: 'ConceptMetadata', optional: true, default: '{}' }
                    ],
                    returns: 'Promise<Concept>',
                    example: `const concept = await cortexWrapper.allocateConcept(
  'user-preference',
  2048,
  { description: 'User behavior patterns' }
);`
                },
                spatialPooling: {
                    description: 'Perform spatial pooling on input pattern',
                    parameters: [
                        { name: 'inputPattern', type: 'number[]', default: 'new Array(100).fill(0).map(() => Math.random() > 0.8 ? 1 : 0)' },
                        { name: 'options', type: 'SpatialPoolingOptions', optional: true, default: '{}' }
                    ],
                    returns: 'Promise<SpatialResult>',
                    example: `const pattern = new Array(2048).fill(0);
for (let i = 0; i < 40; i++) {
  pattern[Math.floor(Math.random() * 2048)] = 1;
}
const result = await cortexWrapper.spatialPooling(pattern);`
                },
                queryConcepts: {
                    description: 'Query concepts using semantic search',
                    parameters: [
                        { name: 'query', type: 'string | number[]', default: '"machine learning"' },
                        { name: 'options', type: 'QueryOptions', optional: true, default: '{ maxResults: 5 }' }
                    ],
                    returns: 'Promise<Concept[]>',
                    example: `const results = await cortexWrapper.queryConcepts(
  'artificial intelligence',
  { maxResults: 10, threshold: 0.8 }
);`
                }
            },
            storageManager: {
                storeConcept: {
                    description: 'Store concept persistently in IndexedDB',
                    parameters: [
                        { name: 'concept', type: 'Concept', default: 'concept' }
                    ],
                    returns: 'Promise<void>',
                    example: `const concept = await cortexWrapper.allocateConcept('data', 1024);
await storageManager.storeConcept(concept);`
                },
                retrieveConcept: {
                    description: 'Retrieve concept from persistent storage',
                    parameters: [
                        { name: 'conceptId', type: 'string', default: '"test-concept"' }
                    ],
                    returns: 'Promise<Concept | null>',
                    example: `const stored = await storageManager.retrieveConcept('user-preference');
if (stored) {
  console.log('Found concept:', stored.name);
}`
                }
            }
        };

        // Populate method list
        function populateMethodList() {
            const methodList = document.getElementById('methodList');
            
            for (const [module, methods] of Object.entries(apiMethods)) {
                const moduleHeader = document.createElement('li');
                moduleHeader.textContent = module;
                moduleHeader.style.fontWeight = 'bold';
                moduleHeader.style.color = '#4ec9b0';
                methodList.appendChild(moduleHeader);
                
                for (const [methodName, methodInfo] of Object.entries(methods)) {
                    const methodItem = document.createElement('li');
                    methodItem.textContent = `  ${methodName}()`;
                    methodItem.dataset.module = module;
                    methodItem.dataset.method = methodName;
                    methodItem.addEventListener('click', () => showMethodDetails(module, methodName, methodInfo));
                    methodList.appendChild(methodItem);
                }
            }
        }

        // Show method details
        function showMethodDetails(module, methodName, methodInfo) {
            // Clear active state
            document.querySelectorAll('.method-list li').forEach(li => li.classList.remove('active'));
            event.target.classList.add('active');

            const detailsContainer = document.getElementById('methodDetails');
            
            let parametersHtml = '';
            if (methodInfo.parameters.length > 0) {
                parametersHtml = '<h4>Parameters:</h4>';
                methodInfo.parameters.forEach((param, index) => {
                    const required = param.optional ? ' (optional)' : ' (required)';
                    parametersHtml += `
                        <div>
                            <label>${param.name} (${param.type})${required}:</label>
                            <input type="text" class="param-input" id="param-${index}" 
                                   value="${param.default || ''}" 
                                   placeholder="${param.type}">
                        </div>
                    `;
                });
            }

            detailsContainer.innerHTML = `
                <h2>${module}.${methodName}()</h2>
                <p>${methodInfo.description}</p>
                
                ${parametersHtml}
                
                <h4>Returns:</h4>
                <p><code>${methodInfo.returns}</code></p>
                
                <h4>Example:</h4>
                <div class="code-block">${methodInfo.example}</div>
                
                <button class="try-button" onclick="tryMethod('${module}', '${methodName}')">
                    Try This Method
                </button>
                
                <h4>Result:</h4>
                <div class="result" id="result"></div>
            `;
        }

        // Try method execution
        async function tryMethod(module, methodName) {
            const resultDiv = document.getElementById('result');
            resultDiv.textContent = 'Executing...';
            
            try {
                // Get parameter values from inputs
                const params = [];
                const paramInputs = document.querySelectorAll('.param-input');
                paramInputs.forEach(input => {
                    if (input.value) {
                        try {
                            // Try to parse as JSON, fallback to string
                            const value = input.value.startsWith('{') || input.value.startsWith('[') || input.value.startsWith('"') 
                                ? JSON.parse(input.value)
                                : input.value;
                            params.push(value);
                        } catch (e) {
                            params.push(input.value);
                        }
                    }
                });

                // Mock execution (in real implementation, would call actual API)
                const mockResult = {
                    success: true,
                    module: module,
                    method: methodName,
                    parameters: params,
                    timestamp: new Date().toISOString(),
                    result: `Mock result for ${module}.${methodName}()`,
                    executionTime: Math.random() * 100 + 'ms'
                };

                resultDiv.textContent = JSON.stringify(mockResult, null, 2);
                
            } catch (error) {
                resultDiv.textContent = `Error: ${error.message}`;
            }
        }

        // Initialize the page
        populateMethodList();
    </script>
</body>
</html>
```

5. **Create API usage examples collection**
```javascript
// docs/examples/api-usage-examples.js - Comprehensive API usage examples

/**
 * Basic API Usage Examples
 * Demonstrates fundamental operations with CortexKG
 */
class BasicUsageExamples {
  
  /**
   * Example 1: Simple initialization and concept allocation
   */
  static async example1_BasicSetup() {
    console.log('=== Example 1: Basic Setup ===');
    
    try {
      // Initialize WASM module
      await wasmLoader.init({
        memorySize: 16 * 1024 * 1024, // 16MB
        enableSIMD: true,
        debug: true
      });
      
      console.log('✓ WASM loaded successfully');
      console.log('Memory stats:', wasmLoader.getMemoryUsage());
      
      // Allocate a simple concept
      const concept = await cortexWrapper.allocateConcept('hello-world', 1024, {
        description: 'First concept allocation',
        tags: ['example', 'basic']
      });
      
      console.log('✓ Concept allocated:', concept);
      
      // Store the concept persistently
      await storageManager.storeConcept(concept);
      console.log('✓ Concept stored persistently');
      
      return { success: true, conceptId: concept.id };
      
    } catch (error) {
      console.error('❌ Error in basic setup:', error);
      return { success: false, error: error.message };
    }
  }

  /**
   * Example 2: Spatial pooling with real data patterns
   */
  static async example2_SpatialPooling() {
    console.log('=== Example 2: Spatial Pooling ===');
    
    try {
      // Create a meaningful input pattern (simulating image data)
      const imageWidth = 32;
      const imageHeight = 32;
      const inputPattern = new Array(imageWidth * imageHeight).fill(0);
      
      // Create a simple vertical line pattern
      for (let y = 8; y < 24; y++) {
        for (let x = 14; x < 18; x++) {
          inputPattern[y * imageWidth + x] = 1;
        }
      }
      
      console.log('Input pattern sparsity:', 
        inputPattern.filter(x => x === 1).length / inputPattern.length);
      
      // Perform spatial pooling
      const result = await cortexWrapper.spatialPooling(inputPattern, {
        sparsity: 0.02,
        boostStrength: 2.0,
        dutyCyclePeriod: 1000
      });
      
      console.log('✓ Spatial pooling result:');
      console.log('  Active columns:', result.activeColumns.length);
      console.log('  Column indices:', result.activeColumns.slice(0, 10));
      console.log('  Average overlap:', 
        result.overlap.reduce((a, b) => a + b, 0) / result.overlap.length);
      
      return { success: true, activeColumns: result.activeColumns.length };
      
    } catch (error) {
      console.error('❌ Error in spatial pooling:', error);
      return { success: false, error: error.message };
    }
  }

  /**
   * Example 3: Temporal sequence processing
   */
  static async example3_TemporalMemory() {
    console.log('=== Example 3: Temporal Memory ===');
    
    try {
      // Create a sequence of patterns (ABC pattern)
      const patternA = new Array(100).fill(0);
      patternA.fill(1, 0, 20);  // First 20 bits active
      
      const patternB = new Array(100).fill(0);
      patternB.fill(1, 20, 40); // Next 20 bits active
      
      const patternC = new Array(100).fill(0);
      patternC.fill(1, 40, 60); // Next 20 bits active
      
      // Create sequence: A->B->C->A->B->C
      const sequence = [patternA, patternB, patternC, patternA, patternB, patternC];
      
      console.log('Processing sequence of', sequence.length, 'patterns');
      
      // Process temporal sequence
      const result = await cortexWrapper.temporalMemory(sequence, true);
      
      console.log('✓ Temporal memory result:');
      console.log('  Active cells:', result.activeCells.length);
      console.log('  Predictive cells:', result.predictiveCells.length);
      console.log('  Prediction accuracy:', 
        result.prediction.filter((p, i) => p > 0.5 && sequence[i % sequence.length][i % 100] === 1).length / result.prediction.length);
      
      return { success: true, predictions: result.prediction.length };
      
    } catch (error) {
      console.error('❌ Error in temporal memory:', error);
      return { success: false, error: error.message };
    }
  }
}

/**
 * Advanced API Usage Examples
 * Demonstrates complex workflows and optimization techniques
 */
class AdvancedUsageExamples {
  
  /**
   * Example 4: Batch concept management
   */
  static async example4_BatchOperations() {
    console.log('=== Example 4: Batch Operations ===');
    
    try {
      const startTime = performance.now();
      
      // Allocate multiple concepts in parallel
      const conceptPromises = [];
      for (let i = 0; i < 50; i++) {
        conceptPromises.push(
          cortexWrapper.allocateConcept(`batch-concept-${i}`, 512 + (i * 10), {
            description: `Batch allocated concept ${i}`,
            priority: Math.floor(i / 10),
            version: 1
          })
        );
      }
      
      const concepts = await Promise.all(conceptPromises);
      
      console.log('✓ Allocated', concepts.length, 'concepts in parallel');
      
      // Store all concepts persistently
      const storagePromises = concepts.map(concept => 
        storageManager.storeConcept(concept)
      );
      await Promise.all(storagePromises);
      
      console.log('✓ All concepts stored persistently');
      
      // Query concepts by priority
      const highPriorityConcepts = concepts.filter(c => c.metadata.priority >= 4);
      console.log('✓ High priority concepts:', highPriorityConcepts.length);
      
      const endTime = performance.now();
      console.log(`✓ Batch operation completed in ${endTime - startTime}ms`);
      
      return { 
        success: true, 
        conceptsCreated: concepts.length,
        executionTime: endTime - startTime 
      };
      
    } catch (error) {
      console.error('❌ Error in batch operations:', error);
      return { success: false, error: error.message };
    }
  }

  /**
   * Example 5: Memory management and optimization
   */
  static async example5_MemoryManagement() {
    console.log('=== Example 5: Memory Management ===');
    
    try {
      const initialStats = wasmLoader.getMemoryUsage();
      console.log('Initial memory:', initialStats);
      
      // Gradually increase memory usage
      const concepts = [];
      let memoryPressure = 0;
      
      while (memoryPressure < 0.7) {
        const concept = await cortexWrapper.allocateConcept(
          `memory-test-${concepts.length}`,
          1024 * (concepts.length + 1) // Increasing size
        );
        concepts.push(concept);
        
        const stats = wasmLoader.getMemoryUsage();
        memoryPressure = stats.pressure;
        
        console.log(`Concept ${concepts.length}: Memory pressure at ${(memoryPressure * 100).toFixed(1)}%`);
        
        if (concepts.length > 100) break; // Safety break
      }
      
      console.log('✓ Reached memory pressure threshold');
      
      // Implement memory cleanup strategy
      const conceptsToRemove = concepts.filter((_, index) => index % 3 === 0); // Remove every 3rd concept
      
      for (const concept of conceptsToRemove) {
        await cortexWrapper.deallocateConcept(concept.id);
      }
      
      const finalStats = wasmLoader.getMemoryUsage();
      console.log('✓ Memory cleanup completed');
      console.log('Final memory:', finalStats);
      
      return {
        success: true,
        peakConcepts: concepts.length,
        memoryRecovered: initialStats.freeSize - finalStats.freeSize
      };
      
    } catch (error) {
      console.error('❌ Error in memory management:', error);
      return { success: false, error: error.message };
    }
  }

  /**
   * Example 6: Real-time visualization integration
   */
  static async example6_VisualizationIntegration() {
    console.log('=== Example 6: Visualization Integration ===');
    
    try {
      // Create canvas element (or get existing one)
      let canvas = document.getElementById('demo-canvas');
      if (!canvas) {
        canvas = document.createElement('canvas');
        canvas.id = 'demo-canvas';
        canvas.width = 800;
        canvas.height = 600;
        document.body.appendChild(canvas);
      }
      
      // Initialize visualizer
      cortexVisualizer.init(canvas, {
        columnSize: 4,
        showActivations: true,
        animationSpeed: 1.2
      });
      
      console.log('✓ Visualizer initialized');
      
      // Generate dynamic cortical data
      const columns = [];
      for (let i = 0; i < 2048; i++) {
        columns.push({
          id: i,
          x: (i % 64) * 12,
          y: Math.floor(i / 64) * 12,
          isActive: Math.random() > 0.98,
          isPredictive: Math.random() > 0.95,
          boost: 1.0 + (Math.random() * 0.5)
        });
      }
      
      // Simulate real-time updates
      for (let frame = 0; frame < 60; frame++) {
        // Update activations
        columns.forEach(column => {
          if (Math.random() > 0.99) {
            column.isActive = !column.isActive;
          }
          if (Math.random() > 0.98) {
            column.isPredictive = !column.isPredictive;
          }
        });
        
        const corticalData = {
          columns: columns,
          activations: columns.map(c => c.isActive ? 1.0 : 0.0),
          predictions: columns.map(c => c.isPredictive ? 1.0 : 0.0),
          timestamp: Date.now()
        };
        
        cortexVisualizer.updateCorticalData(corticalData);
        cortexVisualizer.render();
        
        // Wait for next frame
        await new Promise(resolve => requestAnimationFrame(resolve));
      }
      
      console.log('✓ Real-time visualization demo completed');
      
      return { success: true, framesRendered: 60 };
      
    } catch (error) {
      console.error('❌ Error in visualization integration:', error);
      return { success: false, error: error.message };
    }
  }
}

/**
 * Error Handling and Edge Cases Examples
 */
class ErrorHandlingExamples {
  
  /**
   * Example 7: Comprehensive error handling
   */
  static async example7_ErrorHandling() {
    console.log('=== Example 7: Error Handling ===');
    
    const errors = [];
    
    // Test 1: Invalid memory allocation
    try {
      await cortexWrapper.allocateConcept('too-large', Number.MAX_SAFE_INTEGER);
    } catch (error) {
      errors.push({ test: 'large-allocation', error: error.message });
      console.log('✓ Caught large allocation error:', error.message);
    }
    
    // Test 2: Invalid concept ID
    try {
      await cortexWrapper.deallocateConcept('non-existent-id');
    } catch (error) {
      errors.push({ test: 'invalid-id', error: error.message });
      console.log('✓ Caught invalid ID error:', error.message);
    }
    
    // Test 3: Invalid spatial pooling pattern
    try {
      await cortexWrapper.spatialPooling([1, 2, 3, 'invalid']);
    } catch (error) {
      errors.push({ test: 'invalid-pattern', error: error.message });
      console.log('✓ Caught invalid pattern error:', error.message);
    }
    
    // Test 4: Storage failure simulation
    try {
      // Simulate IndexedDB quota exceeded
      const largeConcept = await cortexWrapper.allocateConcept('large-storage-test', 1024);
      // This would normally fail in a quota-limited environment
      await storageManager.storeConcept(largeConcept);
    } catch (error) {
      errors.push({ test: 'storage-quota', error: error.message });
      console.log('✓ Caught storage error:', error.message);
    }
    
    console.log('✓ Error handling tests completed');
    return { success: true, errorsHandled: errors.length, errors };
  }
}

// Export examples for use in documentation
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    BasicUsageExamples,
    AdvancedUsageExamples,
    ErrorHandlingExamples
  };
}
```

## Expected Outputs
- Complete TypeScript definition files with all API interfaces and types
- Auto-generated API reference documentation from JSDoc comments
- Interactive API explorer for testing methods in the browser
- Comprehensive usage examples covering basic to advanced scenarios
- Error handling patterns and best practice documentation

## Validation
1. Verify TypeScript definitions compile without errors and provide accurate IntelliSense
2. Confirm API documentation accurately reflects actual method signatures and behavior
3. Test interactive API explorer functions correctly with mock and real API calls
4. Validate usage examples execute successfully and demonstrate intended functionality
5. Ensure documentation covers all public API methods with clear, actionable examples

## Next Steps
- Proceed to micro-phase 9.47 (Integration Guide)
- Publish API documentation to developer portal
- Set up automated documentation generation in CI/CD pipeline