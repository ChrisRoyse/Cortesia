# Micro-Phase 9.12: Define Storage Schema

## Objective
Define comprehensive storage schema for concepts, connections, cache, and metadata with optimized indexing for high-performance browser storage operations.

## Prerequisites
- Completed micro-phase 9.11 (IndexedDB wrapper)
- Understanding of IndexedDB schema design
- CortexKG data structures from Phase 1

## Task Description
Create detailed storage schemas for all CortexKG data types with proper indexing, versioning, and migration support for scalable browser-based persistence.

## Specific Actions

1. **Define core concept storage schema**:
   ```rust
   // src/storage/schema.rs
   use serde::{Serialize, Deserialize};
   use std::collections::HashMap;
   
   #[derive(Serialize, Deserialize, Clone, Debug)]
   pub struct ConceptSchema {
       pub id: u32,
       pub content: String,
       pub content_hash: u64,
       pub column_id: u32,
       pub created_timestamp: f64,
       pub last_accessed: f64,
       pub access_count: u32,
       pub metadata: ConceptMetadata,
       pub embeddings: ConceptEmbeddings,
       pub version: u32,
   }
   
   #[derive(Serialize, Deserialize, Clone, Debug)]
   pub struct ConceptMetadata {
       pub complexity: f32,
       pub category: String,
       pub source: ConceptSource,
       pub tags: Vec<String>,
       pub priority: u8,
       pub confidence: f32,
       pub connections: Vec<ConnectionReference>,
   }
   
   #[derive(Serialize, Deserialize, Clone, Debug)]
   pub struct ConceptSource {
       pub origin: String,        // "user_input", "inference", "import"
       pub timestamp: f64,
       pub session_id: String,
       pub parent_concept_id: Option<u32>,
   }
   
   #[derive(Serialize, Deserialize, Clone, Debug)]
   pub struct ConceptEmbeddings {
       pub semantic_vector: Vec<f32>,      // 256-dim semantic embedding
       pub neural_signature: Vec<f32>,     // 128-dim neural state
       pub context_vector: Vec<f32>,       // 64-dim context embedding
       pub last_computed: f64,
   }
   
   #[derive(Serialize, Deserialize, Clone, Debug)]
   pub struct ConnectionReference {
       pub target_id: u32,
       pub connection_type: ConnectionType,
       pub strength: f32,
       pub created: f64,
   }
   
   #[derive(Serialize, Deserialize, Clone, Debug)]
   pub enum ConnectionType {
       Semantic,
       Temporal,
       Causal,
       Spatial,
       Hierarchical,
       Inferential,
   }
   ```

2. **Define cortical column schema**:
   ```rust
   #[derive(Serialize, Deserialize, Clone, Debug)]
   pub struct ColumnSchema {
       pub id: u32,
       pub allocated_concept_id: Option<u32>,
       pub state: ColumnState,
       pub position: ColumnPosition,
       pub connections: ColumnConnections,
       pub performance_metrics: ColumnMetrics,
       pub last_updated: f64,
       pub version: u32,
   }
   
   #[derive(Serialize, Deserialize, Clone, Debug)]
   pub struct ColumnState {
       pub activation_level: f32,
       pub inhibition_level: f32,
       pub spike_time: f32,
       pub refractory_period: f32,
       pub adaptation_state: f32,
       pub plasticity_window: f32,
   }
   
   #[derive(Serialize, Deserialize, Clone, Debug)]
   pub struct ColumnPosition {
       pub x: u32,
       pub y: u32,
       pub layer: u8,
       pub region: String,
       pub neighbors: Vec<u32>,
   }
   
   #[derive(Serialize, Deserialize, Clone, Debug)]
   pub struct ColumnConnections {
       pub excitatory: Vec<SynapticConnection>,
       pub inhibitory: Vec<SynapticConnection>,
       pub modulatory: Vec<SynapticConnection>,
       pub feedforward: Vec<SynapticConnection>,
       pub feedback: Vec<SynapticConnection>,
   }
   
   #[derive(Serialize, Deserialize, Clone, Debug)]
   pub struct SynapticConnection {
       pub target_id: u32,
       pub weight: f32,
       pub delay: f32,
       pub plasticity_factor: f32,
       pub connection_type: SynapseType,
       pub last_activated: f64,
   }
   
   #[derive(Serialize, Deserialize, Clone, Debug)]
   pub enum SynapseType {
       AMPA,
       NMDA, 
       GABA_A,
       GABA_B,
       Modulatory,
   }
   
   #[derive(Serialize, Deserialize, Clone, Debug)]
   pub struct ColumnMetrics {
       pub allocation_efficiency: f32,
       pub spike_frequency: f32,
       pub plasticity_rate: f32,
       pub energy_consumption: f32,
       pub computational_load: f32,
   }
   ```

3. **Define query cache schema**:
   ```rust
   #[derive(Serialize, Deserialize, Clone, Debug)]
   pub struct QueryCacheSchema {
       pub query_hash: u64,
       pub query_text: String,
       pub query_type: QueryType,
       pub results: QueryResults,
       pub metadata: QueryMetadata,
       pub performance: QueryPerformance,
       pub created: f64,
       pub last_accessed: f64,
       pub expiry: f64,
       pub version: u32,
   }
   
   #[derive(Serialize, Deserialize, Clone, Debug)]
   pub enum QueryType {
       Semantic,
       Neural,
       Hybrid,
       Similarity,
       Temporal,
       Pattern,
   }
   
   #[derive(Serialize, Deserialize, Clone, Debug)]
   pub struct QueryResults {
       pub concepts: Vec<ConceptResult>,
       pub activations: Vec<ActivationResult>,
       pub paths: Vec<ActivationPath>,
       pub confidence: f32,
       pub total_results: u32,
       pub truncated: bool,
   }
   
   #[derive(Serialize, Deserialize, Clone, Debug)]
   pub struct ConceptResult {
       pub concept_id: u32,
       pub relevance_score: f32,
       pub activation_level: f32,
       pub match_type: MatchType,
       pub context_bonus: f32,
   }
   
   #[derive(Serialize, Deserialize, Clone, Debug)]
   pub struct ActivationResult {
       pub column_id: u32,
       pub activation_strength: f32,
       pub spike_timing: f32,
       pub propagation_delay: f32,
   }
   
   #[derive(Serialize, Deserialize, Clone, Debug)]
   pub struct ActivationPath {
       pub column_sequence: Vec<u32>,
       pub activation_times: Vec<f32>,
       pub total_path_strength: f32,
       pub propagation_time: f32,
   }
   
   #[derive(Serialize, Deserialize, Clone, Debug)]
   pub enum MatchType {
       Exact,
       Semantic,
       Partial,
       Inferred,
       Contextual,
   }
   
   #[derive(Serialize, Deserialize, Clone, Debug)]
   pub struct QueryMetadata {
       pub user_context: String,
       pub session_id: String,
       pub processing_mode: ProcessingMode,
       pub parameters: HashMap<String, f32>,
   }
   
   #[derive(Serialize, Deserialize, Clone, Debug)]
   pub enum ProcessingMode {
       Fast,
       Accurate,
       Balanced,
       Exhaustive,
   }
   
   #[derive(Serialize, Deserialize, Clone, Debug)]
   pub struct QueryPerformance {
       pub processing_time_ms: f64,
       pub memory_usage_mb: f64,
       pub cache_hit_rate: f32,
       pub neural_ops_count: u32,
       pub simd_utilization: f32,
   }
   ```

4. **Define system state schema**:
   ```rust
   #[derive(Serialize, Deserialize, Clone, Debug)]
   pub struct SystemStateSchema {
       pub system_id: String,
       pub cortical_state: CorticalStateSnapshot,
       pub configuration: SystemConfiguration,
       pub statistics: SystemStatistics,
       pub created: f64,
       pub last_checkpoint: f64,
       pub version: u32,
   }
   
   #[derive(Serialize, Deserialize, Clone, Debug)]
   pub struct CorticalStateSnapshot {
       pub total_columns: u32,
       pub allocated_columns: u32,
       pub total_concepts: u32,
       pub total_connections: u32,
       pub global_inhibition: f32,
       pub plasticity_rate: f32,
       pub learning_rate: f32,
       pub column_states_checksum: u64,
   }
   
   #[derive(Serialize, Deserialize, Clone, Debug)]
   pub struct SystemConfiguration {
       pub cortex_size: (u32, u32),
       pub column_count: u32,
       pub max_concepts: u32,
       pub cache_size_mb: u32,
       pub simd_enabled: bool,
       pub neural_parameters: NeuralParameters,
   }
   
   #[derive(Serialize, Deserialize, Clone, Debug)]
   pub struct NeuralParameters {
       pub spike_threshold: f32,
       pub refractory_period: f32,
       pub decay_rate: f32,
       pub plasticity_window: f32,
       pub inhibition_radius: f32,
       pub ttfs_base_time: f32,
   }
   
   #[derive(Serialize, Deserialize, Clone, Debug)]
   pub struct SystemStatistics {
       pub uptime_ms: u64,
       pub total_queries: u32,
       pub cache_hits: u32,
       pub cache_misses: u32,
       pub avg_query_time_ms: f64,
       pub memory_usage_mb: f64,
       pub storage_size_mb: f64,
       pub last_gc_time: f64,
   }
   ```

5. **Define database schema manager**:
   ```rust
   use wasm_bindgen::prelude::*;
   use web_sys::{IdbDatabase, IdbObjectStore};
   
   #[wasm_bindgen]
   pub struct SchemaManager {
       current_version: u32,
       migration_handlers: Vec<Box<dyn Fn(&IdbDatabase) -> Result<(), JsValue>>>,
   }
   
   #[wasm_bindgen]
   impl SchemaManager {
       #[wasm_bindgen(constructor)]
       pub fn new() -> Self {
           Self {
               current_version: 1,
               migration_handlers: Vec::new(),
           }
       }
       
       pub fn create_schema_v1(db: &IdbDatabase) -> Result<(), JsValue> {
           // Concepts store with optimized indexes
           if !db.object_store_names().contains("concepts") {
               let concepts_store = db.create_object_store_with_optional_parameters(
                   "concepts",
                   &js_sys::Object::new()
               )?;
               
               // Primary indexes
               concepts_store.create_index("content_hash", &"content_hash".into())?;
               concepts_store.create_index("column_id", &"column_id".into())?;
               concepts_store.create_index("category", &"metadata.category".into())?;
               concepts_store.create_index("last_accessed", &"last_accessed".into())?;
               concepts_store.create_index("complexity", &"metadata.complexity".into())?;
               
               // Compound indexes for complex queries
               let compound_idx = js_sys::Array::new();
               compound_idx.push(&"metadata.category".into());
               compound_idx.push(&"metadata.complexity".into());
               concepts_store.create_index("category_complexity", &compound_idx)?;
               
               // Timestamp-based indexes for cleanup
               concepts_store.create_index("created_timestamp", &"created_timestamp".into())?;
               concepts_store.create_index("access_count", &"access_count".into())?;
           }
           
           // Columns store with spatial indexing
           if !db.object_store_names().contains("columns") {
               let columns_store = db.create_object_store("columns")?;
               columns_store.create_index("allocated", &"allocated_concept_id".into())?;
               columns_store.create_index("activation", &"state.activation_level".into())?;
               columns_store.create_index("position_x", &"position.x".into())?;
               columns_store.create_index("position_y", &"position.y".into())?;
               columns_store.create_index("layer", &"position.layer".into())?;
               columns_store.create_index("last_updated", &"last_updated".into())?;
           }
           
           // Connections store for synaptic data
           if !db.object_store_names().contains("connections") {
               let connections_store = db.create_object_store("connections")?;
               connections_store.create_index("source_id", &"source_id".into())?;
               connections_store.create_index("target_id", &"target_id".into())?;
               connections_store.create_index("connection_type", &"connection_type".into())?;
               connections_store.create_index("weight", &"weight".into())?;
               connections_store.create_index("last_activated", &"last_activated".into())?;
           }
           
           // Query cache with expiry management
           if !db.object_store_names().contains("query_cache") {
               let cache_store = db.create_object_store("query_cache")?;
               cache_store.create_index("query_hash", &"query_hash".into())?;
               cache_store.create_index("query_type", &"query_type".into())?;
               cache_store.create_index("expiry", &"expiry".into())?;
               cache_store.create_index("last_accessed", &"last_accessed".into())?;
               cache_store.create_index("confidence", &"results.confidence".into())?;
           }
           
           // System state for snapshots
           if !db.object_store_names().contains("system_state") {
               let state_store = db.create_object_store("system_state")?;
               state_store.create_index("created", &"created".into())?;
               state_store.create_index("last_checkpoint", &"last_checkpoint".into())?;
               state_store.create_index("version", &"version".into())?;
           }
           
           // Embeddings store for vector operations
           if !db.object_store_names().contains("embeddings") {
               let embeddings_store = db.create_object_store("embeddings")?;
               embeddings_store.create_index("concept_id", &"concept_id".into())?;
               embeddings_store.create_index("embedding_type", &"embedding_type".into())?;
               embeddings_store.create_index("last_computed", &"last_computed".into())?;
           }
           
           Ok(())
       }
       
       #[wasm_bindgen]
       pub fn get_current_version(&self) -> u32 {
           self.current_version
       }
       
       #[wasm_bindgen]
       pub fn validate_schema(&self, db: &IdbDatabase) -> bool {
           let required_stores = [
               "concepts", "columns", "connections", 
               "query_cache", "system_state", "embeddings"
           ];
           
           for store_name in required_stores.iter() {
               if !db.object_store_names().contains(store_name) {
                   return false;
               }
           }
           
           true
       }
   }
   ```

## Expected Outputs
- Complete schema definitions for all data types
- Optimized indexing strategy for fast queries
- Version management and migration support
- Compound indexes for complex operations
- Memory-efficient serialization structures
- Schema validation utilities

## Validation
1. All required object stores and indexes are created
2. Schema supports efficient range queries
3. Serialization/deserialization works correctly
4. Index utilization improves query performance
5. Schema validation passes for all stores

## Next Steps
- Implement concept storage operations (micro-phase 9.13)
- Create offline sync queue (micro-phase 9.14)