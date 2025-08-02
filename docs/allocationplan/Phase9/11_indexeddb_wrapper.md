# Micro-Phase 9.11: Create IndexedDB Wrapper

## Objective
Create a comprehensive IndexedDB wrapper for browser-based persistence, enabling offline functionality and state management for CortexKG.

## Prerequisites
- Completed WASM core implementation
- Understanding of IndexedDB API
- web-sys features configured for IDB

## Task Description
Implement a Rust wrapper around IndexedDB using web-sys, providing async storage operations for concepts, cortical states, and query caches.

## Specific Actions

1. **Create IndexedDB wrapper struct**:
   ```rust
   // src/storage/indexeddb.rs
   use wasm_bindgen::prelude::*;
   use wasm_bindgen_futures::JsFuture;
   use web_sys::{IdbDatabase, IdbObjectStore, IdbRequest, IdbTransaction};
   use serde::{Serialize, Deserialize};
   
   #[wasm_bindgen]
   pub struct IndexedDBStorage {
       db: IdbDatabase,
       db_name: String,
       version: u32,
   }
   
   #[wasm_bindgen]
   impl IndexedDBStorage {
       pub async fn new(db_name: &str) -> Result<IndexedDBStorage, JsValue> {
           let window = web_sys::window()
               .ok_or_else(|| JsValue::from_str("No window object"))?;
           
           let idb_factory = window.indexed_db()
               .map_err(|_| JsValue::from_str("IndexedDB not supported"))?
               .ok_or_else(|| JsValue::from_str("IndexedDB factory not available"))?;
           
           let version = 1;
           let open_request = idb_factory.open_with_u32(db_name, version)?;
           
           // Handle upgrade needed
           let db_name_clone = db_name.to_string();
           let onupgradeneeded = Closure::wrap(Box::new(move |event: web_sys::Event| {
               let request = event.target()
                   .unwrap()
                   .dyn_into::<IdbRequest>()
                   .unwrap();
               
               let db = request.result()
                   .unwrap()
                   .dyn_into::<IdbDatabase>()
                   .unwrap();
               
               // Create object stores
               Self::create_object_stores(&db).unwrap();
           }) as Box<dyn FnMut(_)>);
           
           open_request.set_onupgradeneeded(Some(onupgradeneeded.as_ref().unchecked_ref()));
           onupgradeneeded.forget();
           
           // Wait for database to open
           let db = JsFuture::from(open_request).await?;
           let db: IdbDatabase = db.dyn_into()?;
           
           Ok(IndexedDBStorage {
               db,
               db_name: db_name.to_string(),
               version,
           })
       }
       
       fn create_object_stores(db: &IdbDatabase) -> Result<(), JsValue> {
           // Concepts store
           if !db.object_store_names().contains("concepts") {
               let store = db.create_object_store("concepts")?;
               store.create_index("content_hash", &JsValue::from_str("content_hash"))?;
               store.create_index("timestamp", &JsValue::from_str("timestamp"))?;
           }
           
           // Columns store
           if !db.object_store_names().contains("columns") {
               let store = db.create_object_store("columns")?;
               store.create_index("allocated", &JsValue::from_str("allocated"))?;
           }
           
           // Connections store
           if !db.object_store_names().contains("connections") {
               db.create_object_store("connections")?;
           }
           
           // Query cache store
           if !db.object_store_names().contains("query_cache") {
               let store = db.create_object_store("query_cache")?;
               store.create_index("expiry", &JsValue::from_str("expiry"))?;
           }
           
           Ok(())
       }
   }
   ```

2. **Implement concept storage operations**:
   ```rust
   #[derive(Serialize, Deserialize)]
   pub struct StoredConcept {
       pub id: u32,
       pub content: String,
       pub content_hash: u64,
       pub column_id: u32,
       pub timestamp: f64,
       pub metadata: ConceptMetadata,
   }
   
   #[derive(Serialize, Deserialize)]
   pub struct ConceptMetadata {
       pub complexity: f32,
       pub category: String,
       pub connections: Vec<u32>,
   }
   
   impl IndexedDBStorage {
       pub async fn store_concept(&self, concept: &StoredConcept) -> Result<(), JsValue> {
           let transaction = self.db.transaction_with_str_and_mode(
               "concepts",
               web_sys::IdbTransactionMode::Readwrite
           )?;
           
           let store = transaction.object_store("concepts")?;
           
           let value = serde_wasm_bindgen::to_value(concept)?;
           let key = JsValue::from(concept.id);
           
           let request = store.put_with_key(&value, &key)?;
           JsFuture::from(request).await?;
           
           Ok(())
       }
       
       pub async fn get_concept(&self, id: u32) -> Result<Option<StoredConcept>, JsValue> {
           let transaction = self.db.transaction_with_str("concepts")?;
           let store = transaction.object_store("concepts")?;
           
           let request = store.get(&JsValue::from(id))?;
           let result = JsFuture::from(request).await?;
           
           if result.is_undefined() {
               Ok(None)
           } else {
               let concept: StoredConcept = serde_wasm_bindgen::from_value(result)?;
               Ok(Some(concept))
           }
       }
       
       pub async fn get_all_concepts(&self) -> Result<Vec<StoredConcept>, JsValue> {
           let transaction = self.db.transaction_with_str("concepts")?;
           let store = transaction.object_store("concepts")?;
           
           let request = store.get_all()?;
           let result = JsFuture::from(request).await?;
           
           let array: js_sys::Array = result.dyn_into()?;
           let mut concepts = Vec::new();
           
           for item in array.iter() {
               let concept: StoredConcept = serde_wasm_bindgen::from_value(item)?;
               concepts.push(concept);
           }
           
           Ok(concepts)
       }
   }
   ```

3. **Implement cortical state persistence**:
   ```rust
   #[derive(Serialize, Deserialize)]
   pub struct CorticalState {
       pub columns: Vec<ColumnState>,
       pub connections: Vec<ConnectionState>,
       pub timestamp: f64,
       pub version: u32,
   }
   
   #[derive(Serialize, Deserialize)]
   pub struct ColumnState {
       pub id: u32,
       pub allocated_concept_id: Option<u32>,
       pub activation_level: f32,
       pub state: u8,
   }
   
   #[derive(Serialize, Deserialize)]
   pub struct ConnectionState {
       pub from: u32,
       pub to: u32,
       pub weight: f32,
   }
   
   impl IndexedDBStorage {
       pub async fn save_cortical_state(&self, state: &CorticalState) -> Result<(), JsValue> {
           let transaction = self.db.transaction_with_str_sequence_and_mode(
               &js_sys::Array::of2(&"columns".into(), &"connections".into()),
               web_sys::IdbTransactionMode::Readwrite
           )?;
           
           // Save columns
           let columns_store = transaction.object_store("columns")?;
           for column in &state.columns {
               let value = serde_wasm_bindgen::to_value(column)?;
               let key = JsValue::from(column.id);
               columns_store.put_with_key(&value, &key)?;
           }
           
           // Save connections
           let connections_store = transaction.object_store("connections")?;
           connections_store.clear()?; // Clear old connections
           
           for (idx, connection) in state.connections.iter().enumerate() {
               let value = serde_wasm_bindgen::to_value(connection)?;
               let key = JsValue::from(idx as u32);
               connections_store.put_with_key(&value, &key)?;
           }
           
           Ok(())
       }
       
       pub async fn load_cortical_state(&self) -> Result<CorticalState, JsValue> {
           // Load columns
           let columns = self.load_all_columns().await?;
           
           // Load connections
           let connections = self.load_all_connections().await?;
           
           Ok(CorticalState {
               columns,
               connections,
               timestamp: js_sys::Date::now(),
               version: self.version,
           })
       }
   }
   ```

4. **Implement query cache operations**:
   ```rust
   #[derive(Serialize, Deserialize)]
   pub struct CachedQuery {
       pub query: String,
       pub results: Vec<QueryResult>,
       pub timestamp: f64,
       pub expiry: f64,
   }
   
   #[derive(Serialize, Deserialize)]
   pub struct QueryResult {
       pub concept_id: u32,
       pub relevance: f32,
       pub activation_path: Vec<u32>,
   }
   
   impl IndexedDBStorage {
       pub async fn cache_query(&self, query: &str, results: Vec<QueryResult>) -> Result<(), JsValue> {
           let cached = CachedQuery {
               query: query.to_string(),
               results,
               timestamp: js_sys::Date::now(),
               expiry: js_sys::Date::now() + 3600000.0, // 1 hour cache
           };
           
           let transaction = self.db.transaction_with_str_and_mode(
               "query_cache",
               web_sys::IdbTransactionMode::Readwrite
           )?;
           
           let store = transaction.object_store("query_cache")?;
           let value = serde_wasm_bindgen::to_value(&cached)?;
           let key = JsValue::from_str(query);
           
           store.put_with_key(&value, &key)?;
           
           Ok(())
       }
       
       pub async fn get_cached_query(&self, query: &str) -> Result<Option<Vec<QueryResult>>, JsValue> {
           let transaction = self.db.transaction_with_str("query_cache")?;
           let store = transaction.object_store("query_cache")?;
           
           let request = store.get(&JsValue::from_str(query))?;
           let result = JsFuture::from(request).await?;
           
           if result.is_undefined() {
               return Ok(None);
           }
           
           let cached: CachedQuery = serde_wasm_bindgen::from_value(result)?;
           
           // Check expiry
           if cached.expiry < js_sys::Date::now() {
               // Delete expired entry
               self.delete_cached_query(query).await?;
               Ok(None)
           } else {
               Ok(Some(cached.results))
           }
       }
   }
   ```

5. **Add utility methods**:
   ```rust
   #[wasm_bindgen]
   impl IndexedDBStorage {
       #[wasm_bindgen]
       pub async fn clear_all(&self) -> Result<(), JsValue> {
           let stores = ["concepts", "columns", "connections", "query_cache"];
           
           let transaction = self.db.transaction_with_str_sequence_and_mode(
               &stores.iter().map(|s| JsValue::from_str(s)).collect::<js_sys::Array>(),
               web_sys::IdbTransactionMode::Readwrite
           )?;
           
           for store_name in stores.iter() {
               let store = transaction.object_store(store_name)?;
               store.clear()?;
           }
           
           Ok(())
       }
       
       #[wasm_bindgen]
       pub async fn get_storage_size(&self) -> Result<f64, JsValue> {
           if let Some(navigator) = web_sys::window().and_then(|w| w.navigator()) {
               if let Ok(storage) = navigator.storage() {
                   if let Ok(estimate) = storage.estimate() {
                       let estimate_future = JsFuture::from(estimate).await?;
                       if let Ok(usage) = js_sys::Reflect::get(&estimate_future, &"usage".into()) {
                           return Ok(usage.as_f64().unwrap_or(0.0));
                       }
                   }
               }
           }
           Ok(0.0)
       }
   }
   ```

## Expected Outputs
- Complete IndexedDB wrapper with async operations
- Object stores for concepts, columns, connections, and cache
- Serialization/deserialization with serde-wasm-bindgen
- Cache expiry handling
- Storage size estimation
- Clear and utility functions

## Validation
1. Database opens and creates stores correctly
2. Concepts store and retrieve accurately
3. State persistence works across sessions
4. Query cache expires properly
5. No memory leaks with closures

## Next Steps
- Define storage schema (micro-phase 9.12)
- Implement concept storage operations (micro-phase 9.13)