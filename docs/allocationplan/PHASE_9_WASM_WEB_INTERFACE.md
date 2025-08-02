# Phase 9: WASM & Web Interface
## Duration: Week 10 | Universal Access Platform

### AI-Verifiable Success Criteria

#### Performance Metrics
- **WASM Bundle Size**: <2MB optimized build
- **Initial Load Time**: <3 seconds on 3G connection
- **Memory Usage**: <50MB peak for 10K concepts
- **Query Response Time**: <100ms for local operations
- **Cross-Platform Compatibility**: 100% core features across major browsers

#### Functional Requirements
- **Universal Access**: Run CortexKG in any modern browser
- **Offline Capability**: Core functionality without network
- **Progressive Enhancement**: Graceful degradation on limited devices
- **Real-time Visualization**: Interactive cortical column display
- **Touch Interface**: Mobile-optimized interaction patterns

### SPARC Implementation Methodology

#### S - Specification
Transform CortexKG into a universal web platform:

```yaml
WASM Platform Goals:
  - Browser Native: Zero installation, instant access
  - Offline First: Local storage with sync capabilities
  - Mobile Ready: Touch-optimized responsive design
  - Performance: Near-native speed for knowledge operations
  - Accessibility: WCAG 2.1 AA compliance
```

#### P - Pseudocode

**WASM Core Architecture**:
```rust
// Main WASM entry point
#[wasm_bindgen]
pub struct CortexKGWasm {
    cortical_map: CorticalMap,
    allocation_engine: AllocationEngine,
    query_processor: QueryProcessor,
    local_storage: IndexedDBStorage,
}

#[wasm_bindgen]
impl CortexKGWasm {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<CortexKGWasm, JsValue> {
        let instance = CortexKGWasm::initialize().await?;
        Ok(instance)
    }
    
    #[wasm_bindgen]
    pub fn allocate_concept(&mut self, content: &str) -> Promise {
        future_to_promise(self.internal_allocate(content))
    }
    
    #[wasm_bindgen]
    pub fn query(&self, query: &str) -> Promise {
        future_to_promise(self.internal_query(query))
    }
}
```

**Web Interface Architecture**:
```typescript
// TypeScript web application layer
class CortexKGWebApp {
    private wasmModule: CortexKGWasm;
    private visualizer: CorticalVisualizer;
    private interface: ResponsiveInterface;
    
    async initialize(): Promise<void> {
        this.wasmModule = new CortexKGWasm();
        this.visualizer = new CorticalVisualizer(this.wasmModule);
        this.interface = new ResponsiveInterface();
        
        await this.setupOfflineStorage();
        await this.initializeVisualization();
    }
}
```

#### R - Refinement Architecture

**WASM-Optimized Data Structures**:
```rust
// Memory-efficient structures for WASM
#[wasm_bindgen]
pub struct WasmCorticalColumn {
    id: u32,
    allocated_concept_id: Option<u32>,
    activation_level: f32,
    lateral_connections: Box<[u32]>, // Fixed array for efficiency
}

// Optimized sparse graph for web
pub struct WasmSparseGraph {
    vertices: Vec<u32>,
    edges: Vec<(u32, u32, f32)>,
    csr_row_ptr: Vec<usize>,
    csr_col_indices: Vec<u32>,
    csr_values: Vec<f32>,
}

// IndexedDB integration
#[wasm_bindgen]
pub struct IndexedDBStorage {
    db_name: String,
    version: u32,
    stores: HashMap<String, ObjectStore>,
}
```

#### C - Completion Tasks

### London School TDD Implementation

#### Test Suite 1: WASM Core Functionality
```rust
#[cfg(test)]
mod wasm_core_tests {
    use super::*;
    use wasm_bindgen_test::*;
    
    #[wasm_bindgen_test]
    async fn test_wasm_initialization() {
        let cortex = CortexKGWasm::new().await;
        assert!(cortex.is_ok());
        
        let cortex = cortex.unwrap();
        assert_eq!(cortex.get_column_count(), 1024); // Default size
        assert_eq!(cortex.get_allocated_concepts_count(), 0);
    }
    
    #[wasm_bindgen_test]
    async fn test_concept_allocation_wasm() {
        let mut cortex = CortexKGWasm::new().await.unwrap();
        let concept_content = "Test concept for WASM allocation";
        
        let result = cortex.allocate_concept(concept_content).await;
        assert!(result.is_ok());
        
        let allocation_result = result.unwrap();
        assert!(allocation_result.column_id > 0);
        assert!(allocation_result.confidence > 0.5);
    }
    
    #[wasm_bindgen_test]
    async fn test_query_processing_wasm() {
        let mut cortex = setup_test_cortex().await;
        populate_test_concepts(&mut cortex).await;
        
        let query_result = cortex.query("machine learning").await.unwrap();
        
        assert!(!query_result.results.is_empty());
        assert!(query_result.processing_time_ms < 100.0);
    }
}
```

#### Test Suite 2: Browser Storage Integration
```rust
#[cfg(test)]
mod storage_tests {
    use super::*;
    use wasm_bindgen_test::*;
    
    #[wasm_bindgen_test]
    async fn test_indexeddb_storage() {
        let storage = IndexedDBStorage::new("cortex_test").await.unwrap();
        
        let concept = TestConcept::new("test_concept", "test content");
        storage.store_concept(&concept).await.unwrap();
        
        let retrieved = storage.get_concept("test_concept").await.unwrap();
        assert_eq!(retrieved.id, concept.id);
        assert_eq!(retrieved.content, concept.content);
    }
    
    #[wasm_bindgen_test]
    async fn test_offline_synchronization() {
        let storage = IndexedDBStorage::new("cortex_sync_test").await.unwrap();
        
        // Add concepts while "offline"
        storage.add_pending_sync(create_test_concept()).await.unwrap();
        storage.add_pending_sync(create_test_concept()).await.unwrap();
        
        // Simulate coming back online
        let pending_items = storage.get_pending_sync().await.unwrap();
        assert_eq!(pending_items.len(), 2);
        
        storage.clear_pending_sync().await.unwrap();
        let remaining = storage.get_pending_sync().await.unwrap();
        assert!(remaining.is_empty());
    }
}
```

#### Test Suite 3: Web Interface Components
```typescript
// Jest tests for web components
describe('CorticalVisualizer', () => {
    let visualizer: CorticalVisualizer;
    let mockWasm: jest.Mocked<CortexKGWasm>;
    
    beforeEach(() => {
        mockWasm = createMockWasm();
        visualizer = new CorticalVisualizer(mockWasm);
    });
    
    test('renders cortical columns correctly', async () => {
        const columns = await mockWasm.getCorticalColumns();
        visualizer.render(columns);
        
        const columnElements = document.querySelectorAll('.cortical-column');
        expect(columnElements.length).toBe(columns.length);
    });
    
    test('handles touch interactions on mobile', () => {
        const touchEvent = createTouchEvent();
        visualizer.handleTouch(touchEvent);
        
        expect(mockWasm.selectColumn).toHaveBeenCalledWith(touchEvent.columnId);
    });
    
    test('updates visualization in real-time', async () => {
        visualizer.startRealTimeUpdates();
        
        // Simulate allocation
        await mockWasm.allocateConcept('test concept');
        
        // Check visualization updated
        const activeColumns = document.querySelectorAll('.column-active');
        expect(activeColumns.length).toBeGreaterThan(0);
    });
});
```

### Task Breakdown

#### Task 9.1: WASM Core Implementation
**Duration**: 2 days
**Deliverable**: High-performance WASM module

```rust
// Cargo.toml WASM configuration
[lib]
crate-type = ["cdylib"]

[dependencies]
wasm-bindgen = "0.2"
wasm-bindgen-futures = "0.4"
js-sys = "0.3"
web-sys = "0.3"
serde = { version = "1.0", features = ["derive"] }
serde-wasm-bindgen = "0.4"

// Main WASM implementation
#[wasm_bindgen]
impl CortexKGWasm {
    #[wasm_bindgen]
    pub async fn allocate_concept(&mut self, content: &str) -> Result<JsValue, JsValue> {
        let concept = Concept::from_text(content)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
            
        let allocation_result = self.allocation_engine
            .allocate(&concept)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
            
        // Convert to JS-friendly format
        let js_result = AllocationResult {
            column_id: allocation_result.column_id.0,
            confidence: allocation_result.confidence,
            processing_time_ms: allocation_result.processing_time.as_millis() as f64,
            lateral_activations: allocation_result.lateral_activations
                .into_iter()
                .map(|(id, activation)| (id.0, activation))
                .collect(),
        };
        
        Ok(serde_wasm_bindgen::to_value(&js_result)?)
    }
    
    #[wasm_bindgen]
    pub async fn query(&self, query_text: &str) -> Result<JsValue, JsValue> {
        let query = Query::parse(query_text)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
            
        let results = self.query_processor
            .process(&query)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
            
        let js_results = QueryResults {
            results: results.into_iter()
                .map(|r| QueryResult {
                    concept_id: r.concept.id.0,
                    content: r.concept.content,
                    relevance_score: r.relevance_score,
                    activation_path: r.activation_path.into_iter()
                        .map(|id| id.0)
                        .collect(),
                })
                .collect(),
            processing_time_ms: query.processing_time.as_millis() as f64,
        };
        
        Ok(serde_wasm_bindgen::to_value(&js_results)?)
    }
}
```

#### Task 9.2: IndexedDB Storage Layer
**Duration**: 2 days
**Deliverable**: Offline-capable browser storage

```rust
#[wasm_bindgen]
impl IndexedDBStorage {
    #[wasm_bindgen(constructor)]
    pub fn new(db_name: &str) -> Promise {
        future_to_promise(async move {
            let storage = IndexedDBStorage::initialize(db_name).await?;
            Ok(JsValue::from(storage))
        })
    }
    
    async fn initialize(db_name: &str) -> Result<IndexedDBStorage, JsError> {
        let window = web_sys::window().unwrap();
        let indexed_db = window.indexed_db()?.unwrap();
        
        let open_request = indexed_db.open_with_u32(db_name, 1)?;
        
        let db = JsFuture::from(open_request).await?;
        let db: web_sys::IdbDatabase = db.dyn_into()?;
        
        // Create object stores
        if !db.object_store_names().contains("concepts") {
            let store = db.create_object_store("concepts")?;
            store.create_index("content_index", &JsValue::from_str("content"), 
                              &web_sys::IdbIndexParameters::new())?;
        }
        
        Ok(IndexedDBStorage {
            db,
            concepts_store: "concepts".to_string(),
            columns_store: "columns".to_string(),
            graph_store: "graph".to_string(),
        })
    }
    
    #[wasm_bindgen]
    pub fn store_concept(&self, concept: &JsValue) -> Promise {
        let concept_data = concept.clone();
        let store_name = self.concepts_store.clone();
        let db = self.db.clone();
        
        future_to_promise(async move {
            let transaction = db.transaction_with_str_and_mode(
                &store_name,
                web_sys::IdbTransactionMode::Readwrite
            )?;
            
            let store = transaction.object_store(&store_name)?;
            let request = store.add(&concept_data)?;
            
            JsFuture::from(request).await?;
            Ok(JsValue::from(true))
        })
    }
}
```

#### Task 9.3: Web Interface Implementation
**Duration**: 2 days
**Deliverable**: Responsive web application

```typescript
// Main application class
export class CortexKGWebApp {
    private wasmModule: CortexKGWasm | null = null;
    private visualizer: CorticalVisualizer | null = null;
    private queryInterface: QueryInterface;
    private allocationInterface: AllocationInterface;
    
    async initialize(): Promise<void> {
        // Load WASM module
        this.wasmModule = await CortexKGWasm.new();
        
        // Initialize visualization
        this.visualizer = new CorticalVisualizer(
            document.getElementById('cortical-canvas') as HTMLCanvasElement,
            this.wasmModule
        );
        
        // Setup interfaces
        this.queryInterface = new QueryInterface(this.wasmModule);
        this.allocationInterface = new AllocationInterface(this.wasmModule);
        
        // Setup event listeners
        this.setupEventListeners();
        
        // Start real-time updates
        this.startRealTimeUpdates();
    }
    
    private setupEventListeners(): void {
        // Query input handling
        const queryInput = document.getElementById('query-input') as HTMLInputElement;
        queryInput.addEventListener('input', debounce(async (event) => {
            const query = (event.target as HTMLInputElement).value;
            if (query.length > 2) {
                await this.performQuery(query);
            }
        }, 300));
        
        // Allocation button
        const allocateButton = document.getElementById('allocate-button') as HTMLButtonElement;
        allocateButton.addEventListener('click', async () => {
            await this.allocateConcept();
        });
        
        // Touch handling for mobile
        if ('ontouchstart' in window) {
            this.setupTouchHandlers();
        }
    }
    
    private async performQuery(query: string): Promise<void> {
        try {
            const startTime = performance.now();
            const results = await this.wasmModule!.query(query);
            const endTime = performance.now();
            
            this.displayQueryResults(results, endTime - startTime);
            this.visualizer!.highlightActivatedColumns(results.activation_path);
        } catch (error) {
            this.displayError(`Query failed: ${error}`);
        }
    }
    
    private async allocateConcept(): Promise<void> {
        const content = (document.getElementById('concept-input') as HTMLTextAreaElement).value;
        
        if (!content.trim()) {
            this.displayError('Please enter concept content');
            return;
        }
        
        try {
            const allocation = await this.wasmModule!.allocate_concept(content);
            this.displayAllocationResult(allocation);
            this.visualizer!.animateAllocation(allocation.column_id);
        } catch (error) {
            this.displayError(`Allocation failed: ${error}`);
        }
    }
}

// Cortical visualization component
export class CorticalVisualizer {
    private canvas: HTMLCanvasElement;
    private ctx: CanvasRenderingContext2D;
    private wasmModule: CortexKGWasm;
    private animationFrame: number | null = null;
    
    constructor(canvas: HTMLCanvasElement, wasmModule: CortexKGWasm) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d')!;
        this.wasmModule = wasmModule;
        
        this.setupCanvas();
        this.startRenderLoop();
    }
    
    private setupCanvas(): void {
        // Make canvas responsive
        const resizeCanvas = () => {
            const container = this.canvas.parentElement!;
            this.canvas.width = container.clientWidth * window.devicePixelRatio;
            this.canvas.height = container.clientHeight * window.devicePixelRatio;
            this.canvas.style.width = container.clientWidth + 'px';
            this.canvas.style.height = container.clientHeight + 'px';
            
            this.ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        };
        
        window.addEventListener('resize', resizeCanvas);
        resizeCanvas();
    }
    
    private async renderCorticalColumns(): Promise<void> {
        const columns = await this.wasmModule.getCorticalColumns();
        
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        const cols = Math.ceil(Math.sqrt(columns.length));
        const cellSize = Math.min(this.canvas.width, this.canvas.height) / cols;
        
        columns.forEach((column, index) => {
            const x = (index % cols) * cellSize;
            const y = Math.floor(index / cols) * cellSize;
            
            this.renderColumn(column, x, y, cellSize);
        });
    }
    
    private renderColumn(column: any, x: number, y: number, size: number): void {
        // Color based on activation level
        const activation = column.activation_level;
        const hue = activation > 0 ? 120 : 0; // Green for active, red for inactive
        const saturation = Math.abs(activation) * 100;
        const lightness = 50 + activation * 30;
        
        this.ctx.fillStyle = `hsl(${hue}, ${saturation}%, ${lightness}%)`;
        this.ctx.fillRect(x + 1, y + 1, size - 2, size - 2);
        
        // Draw concept if allocated
        if (column.allocated_concept_id) {
            this.ctx.fillStyle = 'white';
            this.ctx.font = `${size * 0.1}px Arial`;
            this.ctx.textAlign = 'center';
            this.ctx.fillText('C', x + size/2, y + size/2);
        }
    }
    
    public animateAllocation(columnId: number): void {
        // Add visual feedback for allocation
        const startTime = performance.now();
        const duration = 1000; // 1 second animation
        
        const animate = (currentTime: number) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            // Pulsing effect
            const intensity = Math.sin(progress * Math.PI * 4) * (1 - progress);
            this.highlightColumn(columnId, intensity);
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };
        
        requestAnimationFrame(animate);
    }
}
```

#### Task 9.4: Performance Optimization
**Duration**: 1 day
**Deliverable**: Optimized bundle and runtime performance

```rust
// Build optimizations in Cargo.toml
[profile.release]
opt-level = 's'  # Optimize for size
lto = true       # Link time optimization
codegen-units = 1
panic = 'abort'

// WASM-specific optimizations
#[cfg(target_arch = "wasm32")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

// Memory-efficient data structures
impl WasmOptimizations {
    fn compress_graph_for_wasm(graph: &SparseGraph) -> WasmSparseGraph {
        // Use u32 instead of u64 for IDs to save memory
        // Pack edge data efficiently
        // Use typed arrays for better JS interop
        WasmSparseGraph {
            vertices: graph.vertices.iter().map(|&v| v as u32).collect(),
            edges: graph.edges.iter()
                .map(|(u, v, w)| (*u as u32, *v as u32, *w))
                .collect(),
            csr_row_ptr: graph.csr_row_ptr.clone(),
            csr_col_indices: graph.csr_col_indices.iter().map(|&i| i as u32).collect(),
            csr_values: graph.csr_values.clone(),
        }
    }
}
```

### Performance Benchmarks

#### Benchmark 9.1: WASM Load Time
```typescript
// Performance measurement for initial load
async function benchmarkLoadTime(): Promise<number> {
    const startTime = performance.now();
    
    const wasmModule = await CortexKGWasm.new();
    await wasmModule.initialize();
    
    const endTime = performance.now();
    return endTime - startTime;
}

// Target: <3 seconds on 3G connection
```

#### Benchmark 9.2: Memory Usage
```rust
#[wasm_bindgen]
impl CortexKGWasm {
    #[wasm_bindgen]
    pub fn get_memory_usage(&self) -> JsValue {
        let memory_info = MemoryInfo {
            heap_size: self.get_heap_size(),
            allocated_concepts: self.cortical_map.allocated_count(),
            graph_size: self.sparse_graph.memory_usage(),
            cache_size: self.query_cache.memory_usage(),
        };
        
        serde_wasm_bindgen::to_value(&memory_info).unwrap()
    }
}
```

### Deliverables

#### 9.1 WASM Core Module
- Optimized Rust-to-WASM compilation
- Complete cortical column simulation
- IndexedDB integration for persistence
- Sub-100ms query response times

#### 9.2 Web Application
- Responsive design for all devices
- Real-time cortical visualization
- Touch-optimized mobile interface
- Progressive web app capabilities

#### 9.3 Performance Package
- <2MB optimized bundle size
- <50MB memory usage for 10K concepts
- Offline-first architecture
- Cross-browser compatibility

#### 9.4 Documentation and Examples
- Integration guide for web developers
- Performance tuning recommendations
- Mobile optimization strategies
- Accessibility compliance documentation

### Integration Points

#### Browser Storage Integration
```rust
impl CortexKGWasm {
    async fn sync_with_server(&mut self) -> Result<(), JsError> {
        // Sync local IndexedDB with remote server when online
        let local_changes = self.storage.get_pending_changes().await?;
        
        if self.is_online().await? {
            for change in local_changes {
                self.upload_change(&change).await?;
            }
            self.storage.clear_pending_changes().await?;
        }
        
        Ok(())
    }
}
```

#### Mobile Performance
```typescript
class MobileOptimizations {
    setupPerformanceMonitoring(): void {
        // Monitor memory pressure
        if ('memory' in performance) {
            const memInfo = (performance as any).memory;
            if (memInfo.usedJSHeapSize > memInfo.jsHeapSizeLimit * 0.8) {
                this.triggerGarbageCollection();
            }
        }
        
        // Reduce visualization quality on low-end devices
        if (this.isLowEndDevice()) {
            this.visualizer.setLowQualityMode();
        }
    }
}
```

This phase makes CortexKG universally accessible through web browsers while maintaining high performance and providing an intuitive interface for cortical column visualization and interaction.