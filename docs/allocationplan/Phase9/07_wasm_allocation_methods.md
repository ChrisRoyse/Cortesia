# Micro-Phase 9.07: Port Allocation Engine Methods to WASM

## Objective
Port the core allocation engine from Rust to WASM bindings with optimized memory management and JavaScript-accessible interfaces.

## Prerequisites
- Completed micro-phases 9.01-9.06 (CortexKGWasm struct ready)
- AllocationEngine struct foundation in place
- Memory layout structures implemented

## Task Description
Implement complete allocation engine functionality with WASM bindings, including concept allocation, deallocation, and optimization algorithms with proper error handling and performance monitoring.

## Specific Actions

1. **Implement core AllocationEngine with WASM bindings**:
   ```rust
   // src/allocation/engine.rs
   use wasm_bindgen::prelude::*;
   use std::collections::HashMap;
   
   #[wasm_bindgen]
   pub struct AllocationEngine {
       // Internal state
       column_count: u32,
       allocation_map: HashMap<u32, u32>, // concept_id -> column_id
       free_columns: Vec<u32>,
       allocation_history: Vec<AllocationRecord>,
       
       // Performance tracking
       simd_enabled: bool,
       allocation_times: Vec<f64>,
       optimization_cycles: u32,
   }
   
   #[derive(Clone)]
   struct AllocationRecord {
       concept_id: u32,
       column_id: u32,
       timestamp: f64,
       confidence: f32,
   }
   
   #[wasm_bindgen]
   impl AllocationEngine {
       #[wasm_bindgen(constructor)]
       pub fn new(column_count: u32, enable_simd: bool) -> Self {
           let free_columns: Vec<u32> = (0..column_count).collect();
           
           Self {
               column_count,
               allocation_map: HashMap::new(),
               free_columns,
               allocation_history: Vec::new(),
               simd_enabled: enable_simd,
               allocation_times: Vec::new(),
               optimization_cycles: 0,
           }
       }
   }
   ```

2. **Implement allocation methods with performance tracking**:
   ```rust
   #[wasm_bindgen]
   impl AllocationEngine {
       #[wasm_bindgen]
       pub fn allocate_concept(&mut self, concept_id: u32) -> Result<u32, JsValue> {
           let start_time = js_sys::performance::now();
           
           // Check if already allocated
           if let Some(&existing_column) = self.allocation_map.get(&concept_id) {
               return Ok(existing_column);
           }
           
           // Find optimal column
           let column_id = self.find_optimal_column(concept_id)?;
           
           // Perform allocation
           self.allocation_map.insert(concept_id, column_id);
           self.free_columns.retain(|&id| id != column_id);
           
           // Record allocation
           let record = AllocationRecord {
               concept_id,
               column_id,
               timestamp: js_sys::Date::now(),
               confidence: 1.0,
           };
           self.allocation_history.push(record);
           
           // Track performance
           let elapsed = js_sys::performance::now() - start_time;
           self.allocation_times.push(elapsed);
           if self.allocation_times.len() > 1000 {
               self.allocation_times.remove(0);
           }
           
           Ok(column_id)
       }
       
       #[wasm_bindgen]
       pub fn deallocate_concept(&mut self, concept_id: u32) -> Result<bool, JsValue> {
           if let Some(column_id) = self.allocation_map.remove(&concept_id) {
               self.free_columns.push(column_id);
               self.free_columns.sort_unstable();
               
               // Remove from history
               self.allocation_history.retain(|r| r.concept_id != concept_id);
               
               Ok(true)
           } else {
               Ok(false)
           }
       }
       
       fn find_optimal_column(&self, concept_id: u32) -> Result<u32, JsValue> {
           if self.free_columns.is_empty() {
               return Err(JsValue::from_str("No free columns available"));
           }
           
           // Simple strategy: use first free column
           // TODO: Implement sophisticated allocation algorithm
           if self.simd_enabled {
               self.find_optimal_column_simd(concept_id)
           } else {
               Ok(self.free_columns[0])
           }
       }
       
       fn find_optimal_column_simd(&self, _concept_id: u32) -> Result<u32, JsValue> {
           // SIMD-optimized column selection
           // For now, return first available
           Ok(self.free_columns[0])
       }
   }
   ```

3. **Implement batch allocation operations**:
   ```rust
   #[wasm_bindgen]
   impl AllocationEngine {
       #[wasm_bindgen]
       pub fn allocate_batch(&mut self, concept_ids: Vec<u32>) -> Result<js_sys::Array, JsValue> {
           let results = js_sys::Array::new();
           
           for concept_id in concept_ids {
               match self.allocate_concept(concept_id) {
                   Ok(column_id) => {
                       let allocation_result = js_sys::Object::new();
                       js_sys::Reflect::set(
                           &allocation_result,
                           &JsValue::from_str("conceptId"),
                           &JsValue::from(concept_id)
                       ).unwrap();
                       js_sys::Reflect::set(
                           &allocation_result,
                           &JsValue::from_str("columnId"),
                           &JsValue::from(column_id)
                       ).unwrap();
                       js_sys::Reflect::set(
                           &allocation_result,
                           &JsValue::from_str("success"),
                           &JsValue::from(true)
                       ).unwrap();
                       results.push(&allocation_result);
                   },
                   Err(e) => {
                       let allocation_result = js_sys::Object::new();
                       js_sys::Reflect::set(
                           &allocation_result,
                           &JsValue::from_str("conceptId"),
                           &JsValue::from(concept_id)
                       ).unwrap();
                       js_sys::Reflect::set(
                           &allocation_result,
                           &JsValue::from_str("success"),
                           &JsValue::from(false)
                       ).unwrap();
                       js_sys::Reflect::set(
                           &allocation_result,
                           &JsValue::from_str("error"),
                           &e
                       ).unwrap();
                       results.push(&allocation_result);
                   }
               }
           }
           
           Ok(results)
       }
       
       #[wasm_bindgen]
       pub fn optimize_allocations(&mut self) -> Result<u32, JsValue> {
           let start_time = js_sys::performance::now();
           let mut optimizations = 0;
           
           // Run defragmentation
           optimizations += self.defragment_allocations()?;
           
           // Rebalance based on usage patterns
           optimizations += self.rebalance_allocations()?;
           
           self.optimization_cycles += 1;
           
           let elapsed = js_sys::performance::now() - start_time;
           web_sys::console::log_1(&JsValue::from_str(&format!(
               "Optimization completed in {:.2}ms, {} changes made", 
               elapsed, optimizations
           )));
           
           Ok(optimizations)
       }
       
       fn defragment_allocations(&mut self) -> Result<u32, JsValue> {
           // Move allocations to lower-numbered columns to create contiguous free space
           let mut moves = 0;
           let mut allocations: Vec<_> = self.allocation_map.iter().collect();
           allocations.sort_by_key(|(_, &column_id)| column_id);
           
           let mut target_column = 0u32;
           for (&concept_id, &current_column) in allocations {
               // Find next available target column
               while self.allocation_map.values().any(|&col| col == target_column) {
                   target_column += 1;
               }
               
               if target_column < current_column {
                   self.allocation_map.insert(concept_id, target_column);
                   moves += 1;
               }
               target_column += 1;
           }
           
           // Update free columns list
           self.update_free_columns_list();
           
           Ok(moves)
       }
       
       fn rebalance_allocations(&mut self) -> Result<u32, JsValue> {
           // Implement load balancing logic
           // For now, return 0 (no rebalancing performed)
           Ok(0)
       }
       
       fn update_free_columns_list(&mut self) {
           let allocated_columns: std::collections::HashSet<_> = 
               self.allocation_map.values().cloned().collect();
           
           self.free_columns = (0..self.column_count)
               .filter(|col| !allocated_columns.contains(col))
               .collect();
       }
   }
   ```

4. **Implement allocation queries and statistics**:
   ```rust
   #[wasm_bindgen]
   impl AllocationEngine {
       #[wasm_bindgen]
       pub fn get_allocation_info(&self, concept_id: u32) -> Option<u32> {
           self.allocation_map.get(&concept_id).copied()
       }
       
       #[wasm_bindgen]
       pub fn get_free_column_count(&self) -> u32 {
           self.free_columns.len() as u32
       }
       
       #[wasm_bindgen]
       pub fn get_allocated_count(&self) -> u32 {
           self.allocation_map.len() as u32
       }
       
       #[wasm_bindgen]
       pub fn get_average_time(&self) -> f64 {
           if self.allocation_times.is_empty() {
               0.0
           } else {
               self.allocation_times.iter().sum::<f64>() / self.allocation_times.len() as f64
           }
       }
       
       #[wasm_bindgen]
       pub fn get_allocation_efficiency(&self) -> f32 {
           let total_columns = self.column_count as f32;
           let allocated_columns = self.allocation_map.len() as f32;
           
           if total_columns > 0.0 {
               allocated_columns / total_columns
           } else {
               0.0
           }
       }
       
       #[wasm_bindgen]
       pub fn get_allocation_history(&self) -> js_sys::Array {
           let array = js_sys::Array::new();
           
           for record in &self.allocation_history {
               let obj = js_sys::Object::new();
               js_sys::Reflect::set(&obj, &"conceptId".into(), &record.concept_id.into()).unwrap();
               js_sys::Reflect::set(&obj, &"columnId".into(), &record.column_id.into()).unwrap();
               js_sys::Reflect::set(&obj, &"timestamp".into(), &record.timestamp.into()).unwrap();
               js_sys::Reflect::set(&obj, &"confidence".into(), &record.confidence.into()).unwrap();
               array.push(&obj);
           }
           
           array
       }
   }
   ```

5. **Add integration with CortexKGWasm**:
   ```rust
   // src/lib.rs (additions to CortexKGWasm impl)
   #[wasm_bindgen]
   impl CortexKGWasm {
       #[wasm_bindgen]
       pub fn allocate_concept(&mut self, concept_id: u32) -> Result<u32, JsValue> {
           if !self.initialized {
               return Err(JsValue::from_str("CortexKG not initialized"));
           }
           
           let column_id = self.allocation_engine.allocate_concept(concept_id)?;
           
           // Update corresponding column
           if let Some(column) = self.columns.get_mut(column_id as usize) {
               column.allocated_concept_id = concept_id;
               column.activation_level = 1.0;
           }
           
           self.total_allocations += 1;
           
           Ok(column_id)
       }
       
       #[wasm_bindgen]
       pub fn deallocate_concept(&mut self, concept_id: u32) -> Result<bool, JsValue> {
           let success = self.allocation_engine.deallocate_concept(concept_id)?;
           
           if success {
               // Find and reset column
               for column in &mut self.columns {
                   if column.allocated_concept_id == concept_id {
                       column.allocated_concept_id = 0;
                       column.activation_level = 0.0;
                       break;
                   }
               }
           }
           
           Ok(success)
       }
       
       #[wasm_bindgen]
       pub async fn optimize_allocations(&mut self) -> Result<u32, JsValue> {
           let optimizations = self.allocation_engine.optimize_allocations()?;
           
           // Persist optimized state if storage available
           if optimizations > 0 {
               self.save_state().await?;
           }
           
           Ok(optimizations)
       }
   }
   ```

## Expected Outputs
- Complete AllocationEngine with WASM bindings
- Batch allocation and deallocation operations
- Performance monitoring and optimization algorithms
- JavaScript-accessible allocation statistics
- Integration with CortexKGWasm main struct

## Validation
1. Single concept allocation completes successfully
2. Batch operations handle multiple concepts efficiently
3. Optimization algorithms reduce fragmentation
4. Performance metrics are accurate and updated
5. JavaScript can access all allocation methods

## Next Steps
- Port query processing methods to WASM (micro-phase 9.08)
- Setup SIMD intrinsics for optimization (micro-phase 9.09)