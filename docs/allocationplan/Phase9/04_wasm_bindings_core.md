# Micro-Phase 9.04: Core WASM Bindings

## Objective
Create the core WASM bindings that expose CortexKG functionality to JavaScript with proper type conversion and error handling.

## Prerequisites
- Completed micro-phases 9.01-9.03 (setup and configuration)
- Understanding of wasm-bindgen syntax

## Task Description
Implement the main WASM bindings structure with proper JavaScript interop, including type definitions and basic API surface.

## Specific Actions

1. **Create core types module**:
   ```rust
   // src/types.rs
   use serde::{Deserialize, Serialize};
   use wasm_bindgen::prelude::*;
   
   #[wasm_bindgen]
   #[derive(Serialize, Deserialize)]
   pub struct ConceptId(pub u32);
   
   #[wasm_bindgen]
   #[derive(Serialize, Deserialize)]
   pub struct ColumnId(pub u32);
   
   #[wasm_bindgen]
   #[derive(Serialize, Deserialize)]
   pub struct AllocationResult {
       pub column_id: u32,
       pub confidence: f32,
       pub processing_time_ms: f64,
   }
   
   #[wasm_bindgen]
   impl AllocationResult {
       #[wasm_bindgen(constructor)]
       pub fn new(column_id: u32, confidence: f32, processing_time_ms: f64) -> Self {
           Self {
               column_id,
               confidence,
               processing_time_ms,
           }
       }
       
       #[wasm_bindgen(getter)]
       pub fn column_id(&self) -> u32 {
           self.column_id
       }
       
       #[wasm_bindgen(getter)]
       pub fn confidence(&self) -> f32 {
           self.confidence
       }
       
       #[wasm_bindgen(getter)]
       pub fn processing_time_ms(&self) -> f64 {
           self.processing_time_ms
       }
   }
   ```

2. **Create error handling module**:
   ```rust
   // src/error.rs
   use wasm_bindgen::prelude::*;
   
   #[wasm_bindgen]
   pub struct WasmError {
       message: String,
   }
   
   #[wasm_bindgen]
   impl WasmError {
       pub fn new(msg: &str) -> Self {
           Self {
               message: msg.to_string(),
           }
       }
       
       #[wasm_bindgen(getter)]
       pub fn message(&self) -> String {
           self.message.clone()
       }
   }
   
   pub type WasmResult<T> = Result<T, JsValue>;
   
   pub fn to_js_error<E: std::fmt::Display>(e: E) -> JsValue {
       JsValue::from_str(&format!("CortexKG Error: {}", e))
   }
   ```

3. **Update main lib.rs with core structure**:
   ```rust
   // src/lib.rs
   mod types;
   mod error;
   mod utils;
   
   use wasm_bindgen::prelude::*;
   use wasm_bindgen_futures::future_to_promise;
   use js_sys::Promise;
   
   use crate::types::*;
   use crate::error::*;
   
   #[wasm_bindgen]
   pub struct CortexKGWasm {
       // Internal state (will be expanded in later phases)
       initialized: bool,
       column_count: u32,
   }
   
   #[wasm_bindgen]
   impl CortexKGWasm {
       #[wasm_bindgen(constructor)]
       pub fn new() -> Result<CortexKGWasm, JsValue> {
           utils::set_panic_hook();
           
           Ok(CortexKGWasm {
               initialized: false,
               column_count: 1024, // Default
           })
       }
       
       #[wasm_bindgen]
       pub async fn initialize(&mut self) -> Result<JsValue, JsValue> {
           if self.initialized {
               return Err(to_js_error("Already initialized"));
           }
           
           // Initialization logic (placeholder)
           self.initialized = true;
           
           Ok(JsValue::from_bool(true))
       }
       
       #[wasm_bindgen(getter)]
       pub fn is_initialized(&self) -> bool {
           self.initialized
       }
       
       #[wasm_bindgen(getter)]
       pub fn column_count(&self) -> u32 {
           self.column_count
       }
       
       #[wasm_bindgen]
       pub fn allocate_concept(&mut self, content: &str) -> Promise {
           let content = content.to_string();
           
           future_to_promise(async move {
               // Placeholder for allocation logic
               let result = AllocationResult::new(
                   42, // column_id
                   0.85, // confidence
                   1.5, // processing_time_ms
               );
               
               Ok(JsValue::from(result))
           })
       }
       
       #[wasm_bindgen]
       pub fn query(&self, query_text: &str) -> Promise {
           let query = query_text.to_string();
           
           future_to_promise(async move {
               // Placeholder for query logic
               Ok(JsValue::from_str("Query results placeholder"))
           })
       }
   }
   ```

4. **Create JavaScript API types**:
   ```typescript
   // types.d.ts (for reference, not generated yet)
   export interface AllocationResult {
       readonly column_id: number;
       readonly confidence: number;
       readonly processing_time_ms: number;
   }
   
   export interface QueryResult {
       readonly concept_id: number;
       readonly content: string;
       readonly relevance_score: number;
   }
   
   export class CortexKGWasm {
       constructor();
       initialize(): Promise<boolean>;
       allocate_concept(content: string): Promise<AllocationResult>;
       query(query_text: string): Promise<QueryResult[]>;
       readonly is_initialized: boolean;
       readonly column_count: number;
   }
   ```

5. **Create test for bindings**:
   ```rust
   // tests/web.rs
   #![cfg(target_arch = "wasm32")]
   
   use wasm_bindgen_test::*;
   use cortexkg_wasm::*;
   
   wasm_bindgen_test_configure!(run_in_browser);
   
   #[wasm_bindgen_test]
   async fn test_initialization() {
       let mut cortex = CortexKGWasm::new().unwrap();
       assert!(!cortex.is_initialized());
       
       cortex.initialize().await.unwrap();
       assert!(cortex.is_initialized());
   }
   
   #[wasm_bindgen_test]
   async fn test_basic_allocation() {
       let mut cortex = CortexKGWasm::new().unwrap();
       cortex.initialize().await.unwrap();
       
       let result = cortex.allocate_concept("test concept").await;
       assert!(result.is_ok());
   }
   ```

## Expected Outputs
- Core WASM bindings with proper type definitions
- Error handling infrastructure
- Basic CortexKGWasm struct with initialization
- Promise-based async API for JavaScript
- Test infrastructure for WASM bindings

## Validation
1. Run `wasm-pack build` - compiles without errors
2. Generated pkg/ contains proper JS bindings
3. Tests pass: `wasm-pack test --headless --firefox`
4. JavaScript types align with Rust definitions

## Next Steps
- Define memory-efficient data structures (micro-phase 9.05)
- Implement CortexKGWasm struct fully (micro-phase 9.06)