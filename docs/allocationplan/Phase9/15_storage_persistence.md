# Micro-Phase 9.15: Add Storage Persistence Layer

## Objective
Add comprehensive data persistence layer with backup/restore functionality, data compression, integrity checks, and migration support for long-term storage reliability.

## Prerequisites
- Completed micro-phase 9.14 (Offline sync queue)
- Understanding of data compression algorithms
- Knowledge of integrity checking mechanisms

## Task Description
Implement robust persistence layer supporting automatic backups, data compression, integrity validation, and seamless migration between storage versions.

## Specific Actions

1. **Create persistence manager**:
   ```rust
   // src/storage/persistence.rs
   use wasm_bindgen::prelude::*;
   use serde::{Serialize, Deserialize};
   use web_sys::{IdbDatabase, Blob, File, FileReader};
   use wasm_bindgen_futures::JsFuture;
   use std::collections::HashMap;
   use js_sys::{Uint8Array, Date};
   
   #[wasm_bindgen]
   pub struct PersistenceManager {
       db: IdbDatabase,
       compression_enabled: bool,
       integrity_checking: bool,
       auto_backup_interval: u32, // milliseconds
       backup_retention_days: u32,
       last_backup_time: f64,
       backup_counter: u32,
   }
   
   #[derive(Serialize, Deserialize, Clone, Debug)]
   pub struct BackupMetadata {
       pub backup_id: String,
       pub created_at: f64,
       pub cortex_version: String,
       pub data_version: u32,
       pub total_concepts: u32,
       pub total_columns: u32,
       pub total_connections: u32,
       pub compression_used: bool,
       pub integrity_hash: String,
       pub backup_size_bytes: u32,
       pub description: String,
   }
   
   #[derive(Serialize, Deserialize, Clone, Debug)]
   pub struct BackupData {
       pub metadata: BackupMetadata,
       pub concepts: Vec<ConceptSchema>,
       pub columns: Vec<ColumnSchema>,
       pub connections: Vec<ConnectionSchema>,
       pub system_state: SystemStateSchema,
       pub query_cache: Vec<QueryCacheSchema>,
       pub sync_queue: Vec<SyncQueueEntry>,
   }
   
   #[derive(Serialize, Deserialize, Clone, Debug)]
   pub struct IntegrityReport {
       pub is_valid: bool,
       pub total_checks: u32,
       pub failed_checks: u32,
       pub issues: Vec<IntegrityIssue>,
       pub last_check_time: f64,
   }
   
   #[derive(Serialize, Deserialize, Clone, Debug)]
   pub struct IntegrityIssue {
       pub issue_type: IntegrityIssueType,
       pub entity_type: String,
       pub entity_id: u32,
       pub description: String,
       pub severity: IssueSeverity,
   }
   
   #[derive(Serialize, Deserialize, Clone, Debug)]
   pub enum IntegrityIssueType {
       MissingReference,
       InvalidData,
       CorruptedEntry,
       VersionMismatch,
       HashMismatch,
       OrphanedData,
   }
   
   #[derive(Serialize, Deserialize, Clone, Debug)]
   pub enum IssueSeverity {
       Low,
       Medium,
       High,
       Critical,
   }
   
   #[wasm_bindgen]
   impl PersistenceManager {
       #[wasm_bindgen(constructor)]
       pub fn new(
           db: IdbDatabase,
           compression_enabled: bool,
           auto_backup_interval_hours: u32
       ) -> Self {
           Self {
               db,
               compression_enabled,
               integrity_checking: true,
               auto_backup_interval: auto_backup_interval_hours * 3600 * 1000, // Convert to ms
               backup_retention_days: 30,
               last_backup_time: 0.0,
               backup_counter: 0,
           }
       }
   }
   ```

2. **Implement backup functionality**:
   ```rust
   impl PersistenceManager {
       #[wasm_bindgen]
       pub async fn create_backup(&mut self, description: Option<String>) -> Result<String, JsValue> {
           let backup_id = self.generate_backup_id();
           let start_time = Date::now();
           
           // Collect all data from stores
           let backup_data = self.collect_backup_data().await?;
           
           // Create metadata
           let metadata = BackupMetadata {
               backup_id: backup_id.clone(),
               created_at: start_time,
               cortex_version: env!("CARGO_PKG_VERSION").to_string(),
               data_version: 1,
               total_concepts: backup_data.concepts.len() as u32,
               total_columns: backup_data.columns.len() as u32,
               total_connections: backup_data.connections.len() as u32,
               compression_used: self.compression_enabled,
               integrity_hash: String::new(), // Will be set after serialization
               backup_size_bytes: 0, // Will be set after compression
               description: description.unwrap_or_else(|| "Automatic backup".to_string()),
           };
           
           let mut backup = BackupData {
               metadata,
               concepts: backup_data.concepts,
               columns: backup_data.columns,
               connections: backup_data.connections,
               system_state: backup_data.system_state,
               query_cache: backup_data.query_cache,
               sync_queue: backup_data.sync_queue,
           };
           
           // Serialize backup data
           let serialized = serde_json::to_string(&backup)
               .map_err(|e| JsValue::from_str(&format!("Serialization failed: {}", e)))?;
           
           // Calculate integrity hash
           let hash = self.calculate_hash(&serialized);
           backup.metadata.integrity_hash = hash;
           
           // Compress if enabled
           let final_data = if self.compression_enabled {
               self.compress_data(&serialized)?
           } else {
               serialized.into_bytes()
           };
           
           backup.metadata.backup_size_bytes = final_data.len() as u32;
           
           // Store backup
           self.store_backup(&backup_id, &final_data, &backup.metadata).await?;
           
           // Update last backup time
           self.last_backup_time = start_time;
           self.backup_counter += 1;
           
           // Clean old backups
           self.cleanup_old_backups().await?;
           
           Ok(backup_id)
       }
       
       async fn collect_backup_data(&self) -> Result<BackupData, JsValue> {
           let transaction = self.db.transaction_with_str_sequence(&js_sys::Array::of6(
               &"concepts".into(),
               &"columns".into(),
               &"connections".into(),
               &"system_state".into(),
               &"query_cache".into(),
               &"sync_queue".into(),
           ))?;
           
           // Collect concepts
           let concepts = self.collect_all_from_store::<ConceptSchema>(&transaction, "concepts").await?;
           
           // Collect columns
           let columns = self.collect_all_from_store::<ColumnSchema>(&transaction, "columns").await?;
           
           // Collect connections
           let connections = self.collect_all_from_store::<ConnectionSchema>(&transaction, "connections").await?;
           
           // Collect system state
           let system_states = self.collect_all_from_store::<SystemStateSchema>(&transaction, "system_state").await?;
           let system_state = system_states.into_iter().next()
               .unwrap_or_else(|| SystemStateSchema::default());
           
           // Collect query cache
           let query_cache = self.collect_all_from_store::<QueryCacheSchema>(&transaction, "query_cache").await?;
           
           // Collect sync queue
           let sync_queue = self.collect_all_from_store::<SyncQueueEntry>(&transaction, "sync_queue").await?;
           
           Ok(BackupData {
               metadata: BackupMetadata::default(), // Will be filled later
               concepts,
               columns,
               connections,
               system_state,
               query_cache,
               sync_queue,
           })
       }
       
       async fn collect_all_from_store<T>(&self, transaction: &IdbTransaction, store_name: &str) -> Result<Vec<T>, JsValue>
       where
           T: for<'de> Deserialize<'de>,
       {
           let store = transaction.object_store(store_name)?;
           let request = store.get_all()?;
           let result = JsFuture::from(request).await?;
           
           let array: js_sys::Array = result.dyn_into()?;
           let mut items = Vec::new();
           
           for value in array.iter() {
               let item: T = serde_wasm_bindgen::from_value(value)?;
               items.push(item);
           }
           
           Ok(items)
       }
       
       fn generate_backup_id(&self) -> String {
           let timestamp = Date::now() as u64;
           let counter = self.backup_counter;
           format!("backup_{}_{}", timestamp, counter)
       }
       
       fn calculate_hash(&self, data: &str) -> String {
           // Simple hash implementation (in production, use crypto API)
           use std::collections::hash_map::DefaultHasher;
           use std::hash::{Hash, Hasher};
           
           let mut hasher = DefaultHasher::new();
           data.hash(&mut hasher);
           format!("{:x}", hasher.finish())
       }
       
       fn compress_data(&self, data: &str) -> Result<Vec<u8>, JsValue> {
           // Placeholder for compression (use flate2 or similar in real implementation)
           // For now, just encode as UTF-8
           Ok(data.as_bytes().to_vec())
       }
       
       async fn store_backup(&self, backup_id: &str, data: &[u8], metadata: &BackupMetadata) -> Result<(), JsValue> {
           let transaction = self.db.transaction_with_str_and_mode(
               "backups",
               web_sys::IdbTransactionMode::Readwrite
           )?;
           
           let store = transaction.object_store("backups")?;
           
           // Store backup data
           let backup_entry = js_sys::Object::new();
           js_sys::Reflect::set(&backup_entry, &"id".into(), &JsValue::from_str(backup_id))?;
           js_sys::Reflect::set(&backup_entry, &"data".into(), &Uint8Array::from(data))?;
           js_sys::Reflect::set(&backup_entry, &"metadata".into(), &serde_wasm_bindgen::to_value(metadata)?)?;
           
           let request = store.put_with_key(&backup_entry, &JsValue::from_str(backup_id))?;
           JsFuture::from(request).await?;
           
           Ok(())
       }
   }
   ```

3. **Implement restore functionality**:
   ```rust
   impl PersistenceManager {
       #[wasm_bindgen]
       pub async fn restore_backup(&self, backup_id: &str) -> Result<bool, JsValue> {
           // Load backup data
           let backup_data = self.load_backup(backup_id).await?;
           
           // Verify integrity
           if !self.verify_backup_integrity(&backup_data).await? {
               return Err(JsValue::from_str("Backup integrity check failed"));
           }
           
           // Clear existing data
           self.clear_all_stores().await?;
           
           // Restore data
           self.restore_concepts(&backup_data.concepts).await?;
           self.restore_columns(&backup_data.columns).await?;
           self.restore_connections(&backup_data.connections).await?;
           self.restore_system_state(&backup_data.system_state).await?;
           self.restore_query_cache(&backup_data.query_cache).await?;
           self.restore_sync_queue(&backup_data.sync_queue).await?;
           
           Ok(true)
       }
       
       async fn load_backup(&self, backup_id: &str) -> Result<BackupData, JsValue> {
           let transaction = self.db.transaction_with_str("backups")?;
           let store = transaction.object_store("backups")?;
           
           let request = store.get(&JsValue::from_str(backup_id))?;
           let result = JsFuture::from(request).await?;
           
           if result.is_undefined() {
               return Err(JsValue::from_str("Backup not found"));
           }
           
           let backup_entry = result.dyn_into::<js_sys::Object>()?;
           let data_array = js_sys::Reflect::get(&backup_entry, &"data".into())?
               .dyn_into::<Uint8Array>()?;
           let metadata: BackupMetadata = serde_wasm_bindgen::from_value(
               js_sys::Reflect::get(&backup_entry, &"metadata".into())?
           )?;
           
           // Convert Uint8Array to Vec<u8>
           let data_bytes: Vec<u8> = data_array.to_vec();
           
           // Decompress if needed
           let data_string = if metadata.compression_used {
               self.decompress_data(&data_bytes)?
           } else {
               String::from_utf8(data_bytes)
                   .map_err(|e| JsValue::from_str(&format!("UTF-8 decode error: {}", e)))?
           };
           
           // Deserialize
           let backup_data: BackupData = serde_json::from_str(&data_string)
               .map_err(|e| JsValue::from_str(&format!("Deserialization error: {}", e)))?;
           
           Ok(backup_data)
       }
       
       async fn verify_backup_integrity(&self, backup_data: &BackupData) -> Result<bool, JsValue> {
           if !self.integrity_checking {
               return Ok(true);
           }
           
           // Re-serialize without metadata to verify hash
           let data_for_hash = BackupData {
               metadata: BackupMetadata::default(),
               concepts: backup_data.concepts.clone(),
               columns: backup_data.columns.clone(),
               connections: backup_data.connections.clone(),
               system_state: backup_data.system_state.clone(),
               query_cache: backup_data.query_cache.clone(),
               sync_queue: backup_data.sync_queue.clone(),
           };
           
           let serialized = serde_json::to_string(&data_for_hash)
               .map_err(|e| JsValue::from_str(&format!("Integrity check serialization failed: {}", e)))?;
           
           let calculated_hash = self.calculate_hash(&serialized);
           
           Ok(calculated_hash == backup_data.metadata.integrity_hash)
       }
       
       async fn restore_concepts(&self, concepts: &[ConceptSchema]) -> Result<(), JsValue> {
           let transaction = self.db.transaction_with_str_and_mode(
               "concepts",
               web_sys::IdbTransactionMode::Readwrite
           )?;
           
           let store = transaction.object_store("concepts")?;
           
           for concept in concepts {
               let value = serde_wasm_bindgen::to_value(concept)?;
               let key = JsValue::from(concept.id);
               store.put_with_key(&value, &key)?;
           }
           
           Ok(())
       }
       
       fn decompress_data(&self, data: &[u8]) -> Result<String, JsValue> {
           // Placeholder for decompression
           String::from_utf8(data.to_vec())
               .map_err(|e| JsValue::from_str(&format!("Decompression failed: {}", e)))
       }
   }
   ```

4. **Implement integrity checking**:
   ```rust
   impl PersistenceManager {
       #[wasm_bindgen]
       pub async fn check_data_integrity(&self) -> Result<IntegrityReport, JsValue> {
           let start_time = Date::now();
           let mut issues = Vec::new();
           let mut total_checks = 0u32;
           
           // Check concept integrity
           let concept_issues = self.check_concept_integrity().await?;
           total_checks += concept_issues.len() as u32;
           issues.extend(concept_issues);
           
           // Check column integrity
           let column_issues = self.check_column_integrity().await?;
           total_checks += column_issues.len() as u32;
           issues.extend(column_issues);
           
           // Check connection integrity
           let connection_issues = self.check_connection_integrity().await?;
           total_checks += connection_issues.len() as u32;
           issues.extend(connection_issues);
           
           // Check referential integrity
           let referential_issues = self.check_referential_integrity().await?;
           total_checks += referential_issues.len() as u32;
           issues.extend(referential_issues);
           
           let failed_checks = issues.len() as u32;
           let is_valid = failed_checks == 0;
           
           Ok(IntegrityReport {
               is_valid,
               total_checks,
               failed_checks,
               issues,
               last_check_time: start_time,
           })
       }
       
       async fn check_concept_integrity(&self) -> Result<Vec<IntegrityIssue>, JsValue> {
           let mut issues = Vec::new();
           
           let transaction = self.db.transaction_with_str("concepts")?;
           let store = transaction.object_store("concepts")?;
           let request = store.get_all()?;
           let result = JsFuture::from(request).await?;
           
           let array: js_sys::Array = result.dyn_into()?;
           
           for value in array.iter() {
               let concept: ConceptSchema = serde_wasm_bindgen::from_value(value)?;
               
               // Check for required fields
               if concept.content.is_empty() {
                   issues.push(IntegrityIssue {
                       issue_type: IntegrityIssueType::InvalidData,
                       entity_type: "concept".to_string(),
                       entity_id: concept.id,
                       description: "Concept has empty content".to_string(),
                       severity: IssueSeverity::High,
                   });
               }
               
               // Check content hash
               let calculated_hash = self.calculate_content_hash(&concept.content);
               if calculated_hash != concept.content_hash {
                   issues.push(IntegrityIssue {
                       issue_type: IntegrityIssueType::HashMismatch,
                       entity_type: "concept".to_string(),
                       entity_id: concept.id,
                       description: "Content hash mismatch".to_string(),
                       severity: IssueSeverity::Medium,
                   });
               }
               
               // Check metadata consistency
               if concept.metadata.complexity < 0.0 || concept.metadata.complexity > 1.0 {
                   issues.push(IntegrityIssue {
                       issue_type: IntegrityIssueType::InvalidData,
                       entity_type: "concept".to_string(),
                       entity_id: concept.id,
                       description: "Invalid complexity value".to_string(),
                       severity: IssueSeverity::Medium,
                   });
               }
           }
           
           Ok(issues)
       }
       
       async fn check_referential_integrity(&self) -> Result<Vec<IntegrityIssue>, JsValue> {
           let mut issues = Vec::new();
           
           // Load all concept IDs
           let concept_ids = self.get_all_concept_ids().await?;
           let column_ids = self.get_all_column_ids().await?;
           
           // Check concept-column references
           let transaction = self.db.transaction_with_str("concepts")?;
           let store = transaction.object_store("concepts")?;
           let request = store.get_all()?;
           let result = JsFuture::from(request).await?;
           
           let array: js_sys::Array = result.dyn_into()?;
           
           for value in array.iter() {
               let concept: ConceptSchema = serde_wasm_bindgen::from_value(value)?;
               
               // Check if assigned column exists
               if !column_ids.contains(&concept.column_id) {
                   issues.push(IntegrityIssue {
                       issue_type: IntegrityIssueType::MissingReference,
                       entity_type: "concept".to_string(),
                       entity_id: concept.id,
                       description: format!("References non-existent column {}", concept.column_id),
                       severity: IssueSeverity::High,
                   });
               }
               
               // Check connection references
               for connection in &concept.metadata.connections {
                   if !concept_ids.contains(&connection.target_id) {
                       issues.push(IntegrityIssue {
                           issue_type: IntegrityIssueType::MissingReference,
                           entity_type: "concept".to_string(),
                           entity_id: concept.id,
                           description: format!("References non-existent concept {}", connection.target_id),
                           severity: IssueSeverity::Medium,
                       });
                   }
               }
           }
           
           Ok(issues)
       }
       
       fn calculate_content_hash(&self, content: &str) -> u64 {
           use std::collections::hash_map::DefaultHasher;
           use std::hash::{Hash, Hasher};
           
           let mut hasher = DefaultHasher::new();
           content.hash(&mut hasher);
           hasher.finish()
       }
   }
   ```

5. **Add automatic maintenance**:
   ```rust
   #[wasm_bindgen]
   impl PersistenceManager {
       #[wasm_bindgen]
       pub async fn start_automatic_maintenance(&mut self) -> Result<(), JsValue> {
           // Schedule periodic integrity checks
           let integrity_callback = Closure::wrap(Box::new(move || {
               wasm_bindgen_futures::spawn_local(async move {
                   // Perform integrity check
                   if let Err(e) = self.check_data_integrity().await {
                       web_sys::console::error_1(&JsValue::from_str(&format!("Integrity check failed: {:?}", e)));
                   }
               });
           }) as Box<dyn FnMut()>);
           
           // Schedule every 24 hours
           web_sys::window()
               .unwrap()
               .set_interval_with_callback_and_timeout_and_arguments_0(
                   integrity_callback.as_ref().unchecked_ref(),
                   24 * 60 * 60 * 1000 // 24 hours in ms
               )?;
           
           integrity_callback.forget();
           
           // Schedule automatic backups
           if self.auto_backup_interval > 0 {
               let backup_callback = Closure::wrap(Box::new(move || {
                   wasm_bindgen_futures::spawn_local(async move {
                       if let Err(e) = self.create_backup(Some("Automatic backup".to_string())).await {
                           web_sys::console::error_1(&JsValue::from_str(&format!("Auto backup failed: {:?}", e)));
                       }
                   });
               }) as Box<dyn FnMut()>);
               
               web_sys::window()
                   .unwrap()
                   .set_interval_with_callback_and_timeout_and_arguments_0(
                       backup_callback.as_ref().unchecked_ref(),
                       self.auto_backup_interval as i32
                   )?;
               
               backup_callback.forget();
           }
           
           Ok(())
       }
       
       #[wasm_bindgen]
       pub async fn repair_data_issues(&self, issues: Vec<IntegrityIssue>) -> Result<u32, JsValue> {
           let mut repaired_count = 0u32;
           
           for issue in issues {
               match issue.issue_type {
                   IntegrityIssueType::HashMismatch => {
                       if self.repair_hash_mismatch(&issue).await? {
                           repaired_count += 1;
                       }
                   },
                   IntegrityIssueType::OrphanedData => {
                       if self.repair_orphaned_data(&issue).await? {
                           repaired_count += 1;
                       }
                   },
                   IntegrityIssueType::MissingReference => {
                       if self.repair_missing_reference(&issue).await? {
                           repaired_count += 1;
                       }
                   },
                   _ => {
                       // Some issues require manual intervention
                       web_sys::console::warn_1(&JsValue::from_str(&format!(
                           "Issue requires manual repair: {}", issue.description
                       )));
                   }
               }
           }
           
           Ok(repaired_count)
       }
       
       async fn cleanup_old_backups(&self) -> Result<u32, JsValue> {
           let cutoff_time = Date::now() - (self.backup_retention_days as f64 * 24.0 * 60.0 * 60.0 * 1000.0);
           let mut deleted_count = 0u32;
           
           let transaction = self.db.transaction_with_str_and_mode(
               "backups",
               web_sys::IdbTransactionMode::Readwrite
           )?;
           
           let store = transaction.object_store("backups")?;
           let request = store.get_all()?;
           let result = JsFuture::from(request).await?;
           
           let array: js_sys::Array = result.dyn_into()?;
           
           for value in array.iter() {
               let entry = value.dyn_into::<js_sys::Object>()?;
               let metadata: BackupMetadata = serde_wasm_bindgen::from_value(
                   js_sys::Reflect::get(&entry, &"metadata".into())?
               )?;
               
               if metadata.created_at < cutoff_time {
                   store.delete(&JsValue::from_str(&metadata.backup_id))?;
                   deleted_count += 1;
               }
           }
           
           Ok(deleted_count)
       }
   }
   ```

## Expected Outputs
- Comprehensive backup/restore system with compression
- Automatic integrity checking and validation
- Data corruption detection and repair capabilities
- Scheduled maintenance and cleanup operations
- Version migration support for schema updates
- Storage optimization and defragmentation

## Validation
1. Backups create and restore correctly with integrity verification
2. Data compression reduces storage size significantly
3. Integrity checks detect and report data issues accurately
4. Automatic maintenance runs without impacting performance
5. Old backups are cleaned up according to retention policy

## Next Steps
- Implement browser performance optimization (micro-phase 9.16)
- Add JavaScript integration layer (micro-phase 9.17)