# Micro-Phase 9.06: Implement Main CortexKGWasm Struct

## Objective
Implement the complete CortexKGWasm struct that integrates all subsystems and provides the main API surface for JavaScript interaction.

## Prerequisites
- Completed micro-phases 9.01-9.05 (memory structures ready)
- Core bindings infrastructure in place

## Task Description
Build the main WASM struct that coordinates allocation engine, query processor, and storage systems with proper initialization and state management.

## Specific Actions

1. **Update CortexKGWasm struct with full implementation**:
   ```rust
   // src/lib.rs (updated)
   use wasm_bindgen::prelude::*;
   use wasm_bindgen_futures::future_to_promise;
   use js_sys::Promise;
   
   use crate::memory::column::WasmCorticalColumn;
   use crate::memory::sparse::WasmSparseConnections;
   use crate::memory::pool::MemoryPool;
   use crate::allocation::AllocationEngine;
   use crate::query::QueryProcessor;
   use crate::storage::IndexedDBStorage;
   
   #[wasm_bindgen]
   pub struct CortexKGWasm {
       // Core components
       columns: Vec<WasmCorticalColumn>,
       connections: WasmSparseConnections,
       allocation_engine: AllocationEngine,
       query_processor: QueryProcessor,
       storage: Option<IndexedDBStorage>,
       
       // Configuration
       config: CortexConfig,
       
       // State
       initialized: bool,
       total_allocations: u32,
   }
   
   #[wasm_bindgen]
   pub struct CortexConfig {
       pub column_count: u32,
       pub max_connections_per_column: u32,
       pub enable_simd: bool,
       pub cache_size_mb: u32,
   }
   
   #[wasm_bindgen]
   impl CortexConfig {
       #[wasm_bindgen(constructor)]
       pub fn new() -> Self {
           Self {
               column_count: 1024,
               max_connections_per_column: 10,
               enable_simd: true,
               cache_size_mb: 10,
           }
       }
   }
   
   #[wasm_bindgen]
   impl CortexKGWasm {
       #[wasm_bindgen(constructor)]
       pub fn new_with_config(config: CortexConfig) -> Result<CortexKGWasm, JsValue> {
           utils::set_panic_hook();
           
           // Initialize columns
           let mut columns = Vec::with_capacity(config.column_count as usize);
           for i in 0..config.column_count {
               columns.push(WasmCorticalColumn::new(i));
           }
           
           // Initialize sparse connections
           let connections = WasmSparseConnections::new(config.column_count);
           
           // Initialize engines
           let allocation_engine = AllocationEngine::new(
               config.column_count,
               config.enable_simd
           );
           let query_processor = QueryProcessor::new(config.cache_size_mb);
           
           Ok(CortexKGWasm {
               columns,
               connections,
               allocation_engine,
               query_processor,
               storage: None,
               config,
               initialized: false,
               total_allocations: 0,
           })
       }
       
       #[wasm_bindgen]
       pub fn new() -> Result<CortexKGWasm, JsValue> {
           Self::new_with_config(CortexConfig::new())
       }
   }
   ```

2. **Implement initialization with storage**:
   ```rust
   #[wasm_bindgen]
   impl CortexKGWasm {
       #[wasm_bindgen]
       pub async fn initialize_with_storage(&mut self, db_name: &str) -> Result<(), JsValue> {
           if self.initialized {
               return Err(JsValue::from_str("Already initialized"));
           }
           
           // Initialize IndexedDB storage
           let storage = IndexedDBStorage::new(db_name).await
               .map_err(|e| JsValue::from_str(&format!("Storage init failed: {}", e)))?;
           
           // Load persisted state if available
           if let Ok(state) = storage.load_cortex_state().await {
               self.restore_from_state(state)?;
           }
           
           self.storage = Some(storage);
           self.initialized = true;
           
           Ok(())
       }
       
       #[wasm_bindgen]
       pub async fn initialize(&mut self) -> Result<(), JsValue> {
           self.initialize_with_storage("cortexkg_default").await
       }
       
       fn restore_from_state(&mut self, state: PersistedState) -> Result<(), JsValue> {
           // Restore allocations
           for (column_id, concept_id) in state.allocations {
               if let Some(column) = self.columns.get_mut(column_id as usize) {
                   column.allocated_concept_id = concept_id;
               }
           }
           
           // Restore connections
           for (from, to, weight) in state.connections {
               self.connections.add_connection(from, to, weight);
           }
           
           self.total_allocations = state.total_allocations;
           Ok(())
       }
   }
   ```

3. **Implement state persistence**:
   ```rust
   // src/storage/state.rs
   use serde::{Serialize, Deserialize};
   
   #[derive(Serialize, Deserialize)]
   pub struct PersistedState {
       pub allocations: Vec<(u32, u32)>, // (column_id, concept_id)
       pub connections: Vec<(u32, u32, f32)>, // (from, to, weight)
       pub total_allocations: u32,
       pub timestamp: f64,
   }
   
   impl CortexKGWasm {
       pub async fn save_state(&self) -> Result<(), JsValue> {
           if let Some(storage) = &self.storage {
               let state = self.create_persisted_state();
               storage.save_cortex_state(&state).await
                   .map_err(|e| JsValue::from_str(&format!("Save failed: {}", e)))?;
           }
           Ok(())
       }
       
       fn create_persisted_state(&self) -> PersistedState {
           let allocations: Vec<(u32, u32)> = self.columns.iter()
               .filter(|c| c.is_allocated())
               .map(|c| (c.id, c.allocated_concept_id))
               .collect();
           
           // Extract connections (simplified)
           let connections = vec![]; // TODO: Implement connection extraction
           
           PersistedState {
               allocations,
               connections,
               total_allocations: self.total_allocations,
               timestamp: js_sys::Date::now(),
           }
       }
   }
   ```

4. **Add performance monitoring**:
   ```rust
   #[wasm_bindgen]
   pub struct PerformanceMetrics {
       pub total_allocations: u32,
       pub average_allocation_time_ms: f64,
       pub memory_usage_bytes: usize,
       pub cache_hit_rate: f32,
   }
   
   #[wasm_bindgen]
   impl CortexKGWasm {
       #[wasm_bindgen]
       pub fn get_performance_metrics(&self) -> PerformanceMetrics {
           PerformanceMetrics {
               total_allocations: self.total_allocations,
               average_allocation_time_ms: self.allocation_engine.get_average_time(),
               memory_usage_bytes: self.estimate_memory_usage(),
               cache_hit_rate: self.query_processor.get_cache_hit_rate(),
           }
       }
       
       fn estimate_memory_usage(&self) -> usize {
           let column_memory = self.columns.len() * std::mem::size_of::<WasmCorticalColumn>();
           let connection_memory = self.connections.estimate_memory();
           let cache_memory = (self.config.cache_size_mb as usize) * 1024 * 1024;
           
           column_memory + connection_memory + cache_memory
       }
   }
   ```

5. **Add column state queries**:
   ```rust
   #[wasm_bindgen]
   impl CortexKGWasm {
       #[wasm_bindgen]
       pub fn get_column_states(&self) -> js_sys::Array {
           let array = js_sys::Array::new();
           
           for column in &self.columns {
               let obj = js_sys::Object::new();
               js_sys::Reflect::set(
                   &obj,
                   &JsValue::from_str("id"),
                   &JsValue::from(column.id)
               ).unwrap();
               js_sys::Reflect::set(
                   &obj,
                   &JsValue::from_str("allocated"),
                   &JsValue::from(column.is_allocated())
               ).unwrap();
               js_sys::Reflect::set(
                   &obj,
                   &JsValue::from_str("activation"),
                   &JsValue::from(column.activation_level)
               ).unwrap();
               
               array.push(&obj);
           }
           
           array
       }
       
       #[wasm_bindgen]
       pub fn get_allocated_concepts_count(&self) -> u32 {
           self.columns.iter()
               .filter(|c| c.is_allocated())
               .count() as u32
       }
   }
   ```

## Expected Outputs
- Complete CortexKGWasm implementation with all subsystems
- Configuration support for customization
- State persistence and restoration
- Performance metrics collection
- Column state inspection API

## Validation
1. Initialization completes successfully
2. State persists and restores correctly
3. Performance metrics are accurate
4. Memory usage stays within bounds
5. All APIs are accessible from JavaScript

## Next Steps
- Port allocation methods to WASM (micro-phase 9.07)
- Port query processing to WASM (micro-phase 9.08)