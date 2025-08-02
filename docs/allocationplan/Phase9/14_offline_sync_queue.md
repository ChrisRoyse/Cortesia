# Micro-Phase 9.14: Create Offline Sync Queue

## Objective
Create robust offline sync queue with conflict resolution, ensuring data consistency between browser storage and external systems during network interruptions.

## Prerequisites
- Completed micro-phase 9.13 (Concept storage)
- Understanding of offline-first design patterns
- Knowledge of conflict resolution strategies

## Task Description
Implement comprehensive offline sync queue supporting operation queuing, conflict detection, resolution strategies, and automatic synchronization with robust error handling.

## Specific Actions

1. **Create sync operation types**:
   ```rust
   // src/storage/sync_queue.rs
   use wasm_bindgen::prelude::*;
   use serde::{Serialize, Deserialize};
   use std::collections::{HashMap, VecDeque};
   use web_sys::{IdbDatabase, IdbObjectStore, IdbTransaction};
   use wasm_bindgen_futures::JsFuture;
   
   #[derive(Serialize, Deserialize, Clone, Debug)]
   pub enum SyncOperation {
       CreateConcept {
           concept_id: u32,
           concept_data: ConceptData,
           timestamp: f64,
       },
       UpdateConcept {
           concept_id: u32,
           changes: ConceptChanges,
           version: u32,
           timestamp: f64,
       },
       DeleteConcept {
           concept_id: u32,
           timestamp: f64,
       },
       CreateConnection {
           connection_id: u32,
           from_id: u32,
           to_id: u32,
           connection_type: ConnectionType,
           timestamp: f64,
       },
       UpdateConnection {
           connection_id: u32,
           changes: ConnectionChanges,
           version: u32,
           timestamp: f64,
       },
       DeleteConnection {
           connection_id: u32,
           timestamp: f64,
       },
       SystemStateCheckpoint {
           state_data: SystemStateData,
           timestamp: f64,
       },
   }
   
   #[derive(Serialize, Deserialize, Clone, Debug)]
   pub struct SyncQueueEntry {
       pub id: u32,
       pub operation: SyncOperation,
       pub status: SyncStatus,
       pub retry_count: u8,
       pub created_at: f64,
       pub last_attempt: Option<f64>,
       pub error_message: Option<String>,
       pub priority: SyncPriority,
       pub dependencies: Vec<u32>, // Other entry IDs this depends on
   }
   
   #[derive(Serialize, Deserialize, Clone, Debug)]
   pub enum SyncStatus {
       Pending,
       InProgress,
       Completed,
       Failed,
       Cancelled,
       Conflicted,
   }
   
   #[derive(Serialize, Deserialize, Clone, Debug)]
   pub enum SyncPriority {
       Low = 1,
       Normal = 2,
       High = 3,
       Critical = 4,
   }
   
   #[derive(Serialize, Deserialize, Clone, Debug)]
   pub struct ConceptData {
       pub content: String,
       pub metadata: ConceptMetadata,
       pub embeddings: Option<ConceptEmbeddings>,
   }
   
   #[derive(Serialize, Deserialize, Clone, Debug)]
   pub struct ConceptChanges {
       pub content: Option<String>,
       pub metadata_changes: HashMap<String, serde_json::Value>,
       pub embedding_updates: Option<ConceptEmbeddings>,
   }
   ```

2. **Implement sync queue manager**:
   ```rust
   #[wasm_bindgen]
   pub struct SyncQueue {
       db: IdbDatabase,
       queue: VecDeque<SyncQueueEntry>,
       next_id: u32,
       sync_in_progress: bool,
       network_available: bool,
       max_retry_attempts: u8,
       retry_delay_ms: u32,
       conflict_resolver: ConflictResolver,
   }
   
   #[wasm_bindgen]
   impl SyncQueue {
       #[wasm_bindgen(constructor)]
       pub fn new(db: IdbDatabase) -> Self {
           Self {
               db,
               queue: VecDeque::new(),
               next_id: 1,
               sync_in_progress: false,
               network_available: true,
               max_retry_attempts: 3,
               retry_delay_ms: 5000,
               conflict_resolver: ConflictResolver::new(),
           }
       }
       
       #[wasm_bindgen]
       pub async fn enqueue_operation(
           &mut self, 
           operation: SyncOperation, 
           priority: SyncPriority
       ) -> Result<u32, JsValue> {
           let entry = SyncQueueEntry {
               id: self.next_id,
               operation,
               status: SyncStatus::Pending,
               retry_count: 0,
               created_at: js_sys::Date::now(),
               last_attempt: None,
               error_message: None,
               priority,
               dependencies: Vec::new(),
           };
           
           // Store in IndexedDB for persistence
           self.persist_queue_entry(&entry).await?;
           
           // Add to in-memory queue in priority order
           let insert_position = self.queue.iter()
               .position(|e| (e.priority as u8) < (priority as u8))
               .unwrap_or(self.queue.len());
           
           self.queue.insert(insert_position, entry);
           
           let id = self.next_id;
           self.next_id += 1;
           
           // Trigger sync if network is available
           if self.network_available && !self.sync_in_progress {
               self.start_sync_process().await?;
           }
           
           Ok(id)
       }
       
       async fn persist_queue_entry(&self, entry: &SyncQueueEntry) -> Result<(), JsValue> {
           let transaction = self.db.transaction_with_str_and_mode(
               "sync_queue",
               web_sys::IdbTransactionMode::Readwrite
           )?;
           
           let store = transaction.object_store("sync_queue")?;
           let value = serde_wasm_bindgen::to_value(entry)?;
           let key = JsValue::from(entry.id);
           
           let request = store.put_with_key(&value, &key)?;
           JsFuture::from(request).await?;
           
           Ok(())
       }
       
       #[wasm_bindgen]
       pub async fn load_queue_from_storage(&mut self) -> Result<u32, JsValue> {
           let transaction = self.db.transaction_with_str("sync_queue")?;
           let store = transaction.object_store("sync_queue")?;
           
           let cursor_request = store.open_cursor()?;
           let cursor_future = JsFuture::from(cursor_request).await?;
           
           let mut cursor = if cursor_future.is_null() {
               None
           } else {
               Some(cursor_future.dyn_into::<web_sys::IdbCursorWithValue>()?)
           };
           
           let mut loaded_count = 0u32;
           self.queue.clear();
           
           while let Some(cursor_ref) = cursor.as_ref() {
               let value = cursor_ref.value()?;
               let entry: SyncQueueEntry = serde_wasm_bindgen::from_value(value)?;
               
               // Only load pending and failed entries
               if matches!(entry.status, SyncStatus::Pending | SyncStatus::Failed) {
                   self.queue.push_back(entry);
                   loaded_count += 1;
               }
               
               // Update next_id
               if entry.id >= self.next_id {
                   self.next_id = entry.id + 1;
               }
               
               let continue_request = cursor_ref.continue_()?;
               let continue_future = JsFuture::from(continue_request).await;
               cursor = if continue_future.is_ok() && !continue_future.unwrap().is_null() {
                   Some(continue_future.unwrap().dyn_into()?)
               } else {
                   None
               };
           }
           
           // Sort queue by priority
           self.queue.make_contiguous().sort_by(|a, b| {
               (b.priority as u8).cmp(&(a.priority as u8))
           });
           
           Ok(loaded_count)
       }
   }
   ```

3. **Implement sync processing**:
   ```rust
   impl SyncQueue {
       async fn start_sync_process(&mut self) -> Result<(), JsValue> {
           if self.sync_in_progress || !self.network_available {
               return Ok(());
           }
           
           self.sync_in_progress = true;
           
           while let Some(mut entry) = self.queue.pop_front() {
               if !self.network_available {
                   // Put entry back and stop
                   self.queue.push_front(entry);
                   break;
               }
               
               entry.status = SyncStatus::InProgress;
               entry.last_attempt = Some(js_sys::Date::now());
               self.persist_queue_entry(&entry).await?;
               
               let result = self.execute_sync_operation(&entry.operation).await;
               
               match result {
                   Ok(_) => {
                       entry.status = SyncStatus::Completed;
                       self.persist_queue_entry(&entry).await?;
                   },
                   Err(sync_error) => {
                       entry.retry_count += 1;
                       entry.error_message = Some(sync_error.error_message);
                       
                       if sync_error.is_conflict {
                           entry.status = SyncStatus::Conflicted;
                           let resolved = self.resolve_conflict(&entry).await?;
                           if resolved {
                               entry.status = SyncStatus::Pending;
                               entry.retry_count = 0;
                               self.queue.push_back(entry);
                           }
                       } else if entry.retry_count >= self.max_retry_attempts {
                           entry.status = SyncStatus::Failed;
                       } else {
                           entry.status = SyncStatus::Pending;
                           // Add back to queue with delay
                           self.schedule_retry(&entry).await?;
                       }
                       
                       self.persist_queue_entry(&entry).await?;
                   }
               }
           }
           
           self.sync_in_progress = false;
           Ok(())
       }
       
       async fn execute_sync_operation(&self, operation: &SyncOperation) -> Result<(), SyncError> {
           match operation {
               SyncOperation::CreateConcept { concept_id, concept_data, .. } => {
                   self.sync_create_concept(*concept_id, concept_data).await
               },
               SyncOperation::UpdateConcept { concept_id, changes, version, .. } => {
                   self.sync_update_concept(*concept_id, changes, *version).await
               },
               SyncOperation::DeleteConcept { concept_id, .. } => {
                   self.sync_delete_concept(*concept_id).await
               },
               SyncOperation::CreateConnection { connection_id, from_id, to_id, connection_type, .. } => {
                   self.sync_create_connection(*connection_id, *from_id, *to_id, connection_type).await
               },
               SyncOperation::UpdateConnection { connection_id, changes, version, .. } => {
                   self.sync_update_connection(*connection_id, changes, *version).await
               },
               SyncOperation::DeleteConnection { connection_id, .. } => {
                   self.sync_delete_connection(*connection_id).await
               },
               SyncOperation::SystemStateCheckpoint { state_data, .. } => {
                   self.sync_system_checkpoint(state_data).await
               },
           }
       }
       
       async fn sync_create_concept(&self, concept_id: u32, concept_data: &ConceptData) -> Result<(), SyncError> {
           // Simulate API call to external service
           let request_data = js_sys::Object::new();
           js_sys::Reflect::set(&request_data, &"id".into(), &JsValue::from(concept_id))?;
           js_sys::Reflect::set(&request_data, &"content".into(), &JsValue::from_str(&concept_data.content))?;
           
           // Check if concept already exists (conflict detection)
           let exists = self.check_concept_exists_remotely(concept_id).await?;
           if exists {
               return Err(SyncError {
                   error_message: "Concept already exists remotely".to_string(),
                   is_conflict: true,
                   retry_after: None,
               });
           }
           
           // Perform actual sync (placeholder for real API call)
           let success = self.perform_remote_api_call("create_concept", &request_data).await?;
           
           if success {
               Ok(())
           } else {
               Err(SyncError {
                   error_message: "Remote creation failed".to_string(),
                   is_conflict: false,
                   retry_after: Some(self.retry_delay_ms),
               })
           }
       }
       
       async fn schedule_retry(&mut self, entry: &SyncQueueEntry) -> Result<(), JsValue> {
           // Calculate exponential backoff delay
           let delay = self.retry_delay_ms * (2_u32.pow(entry.retry_count as u32));
           
           // Schedule re-insertion into queue
           let entry_clone = entry.clone();
           let callback = Closure::wrap(Box::new(move || {
               // This would re-insert the entry after delay
               // In a real implementation, use setTimeout or similar
           }) as Box<dyn FnMut()>);
           
           web_sys::window()
               .unwrap()
               .set_timeout_with_callback_and_timeout_and_arguments_0(
                   callback.as_ref().unchecked_ref(),
                   delay as i32
               )?;
           
           callback.forget();
           Ok(())
       }
   }
   
   #[derive(Debug)]
   struct SyncError {
       error_message: String,
       is_conflict: bool,
       retry_after: Option<u32>,
   }
   ```

4. **Implement conflict resolution**:
   ```rust
   pub struct ConflictResolver {
       resolution_strategies: HashMap<String, ConflictResolutionStrategy>,
   }
   
   #[derive(Clone)]
   pub enum ConflictResolutionStrategy {
       LastWriteWins,
       FirstWriteWins,
       MergeChanges,
       UserChoice,
       CustomLogic(fn(&ConflictData) -> ResolutionResult),
   }
   
   #[derive(Serialize, Deserialize)]
   pub struct ConflictData {
       pub local_version: u32,
       pub remote_version: u32,
       pub local_timestamp: f64,
       pub remote_timestamp: f64,
       pub local_changes: serde_json::Value,
       pub remote_changes: serde_json::Value,
       pub base_data: serde_json::Value,
   }
   
   #[derive(Serialize, Deserialize)]
   pub enum ResolutionResult {
       UseLocal,
       UseRemote,
       UseMerged(serde_json::Value),
       RequiresUserInput,
       Failed(String),
   }
   
   impl ConflictResolver {
       pub fn new() -> Self {
           let mut strategies = HashMap::new();
           strategies.insert("concept".to_string(), ConflictResolutionStrategy::LastWriteWins);
           strategies.insert("connection".to_string(), ConflictResolutionStrategy::MergeChanges);
           strategies.insert("system_state".to_string(), ConflictResolutionStrategy::FirstWriteWins);
           
           Self {
               resolution_strategies: strategies,
           }
       }
       
       pub fn resolve_conflict(&self, entity_type: &str, conflict_data: &ConflictData) -> ResolutionResult {
           let strategy = self.resolution_strategies
               .get(entity_type)
               .unwrap_or(&ConflictResolutionStrategy::LastWriteWins);
           
           match strategy {
               ConflictResolutionStrategy::LastWriteWins => {
                   if conflict_data.local_timestamp > conflict_data.remote_timestamp {
                       ResolutionResult::UseLocal
                   } else {
                       ResolutionResult::UseRemote
                   }
               },
               ConflictResolutionStrategy::FirstWriteWins => {
                   if conflict_data.local_timestamp < conflict_data.remote_timestamp {
                       ResolutionResult::UseLocal
                   } else {
                       ResolutionResult::UseRemote
                   }
               },
               ConflictResolutionStrategy::MergeChanges => {
                   self.merge_changes(conflict_data)
               },
               ConflictResolutionStrategy::UserChoice => {
                   ResolutionResult::RequiresUserInput
               },
               ConflictResolutionStrategy::CustomLogic(resolver_fn) => {
                   resolver_fn(conflict_data)
               },
           }
       }
       
       fn merge_changes(&self, conflict_data: &ConflictData) -> ResolutionResult {
           // Implement three-way merge
           let mut merged = conflict_data.base_data.clone();
           
           // Apply non-conflicting changes from both local and remote
           if let (Some(local_obj), Some(remote_obj), Some(merged_obj)) = (
               conflict_data.local_changes.as_object(),
               conflict_data.remote_changes.as_object(),
               merged.as_object_mut()
           ) {
               // Apply local changes
               for (key, value) in local_obj {
                   if !remote_obj.contains_key(key) {
                       merged_obj.insert(key.clone(), value.clone());
                   }
               }
               
               // Apply remote changes
               for (key, value) in remote_obj {
                   if !local_obj.contains_key(key) {
                       merged_obj.insert(key.clone(), value.clone());
                   } else if local_obj[key] == remote_obj[key] {
                       // Same change, use it
                       merged_obj.insert(key.clone(), value.clone());
                   } else {
                       // Conflict on this field - need user input
                       return ResolutionResult::RequiresUserInput;
                   }
               }
               
               ResolutionResult::UseMerged(merged)
           } else {
               ResolutionResult::Failed("Invalid conflict data format".to_string())
           }
       }
   }
   
   impl SyncQueue {
       async fn resolve_conflict(&self, entry: &SyncQueueEntry) -> Result<bool, JsValue> {
           // Extract conflict information
           let conflict_data = self.extract_conflict_data(entry).await?;
           let entity_type = self.get_entity_type(&entry.operation);
           
           let resolution = self.conflict_resolver.resolve_conflict(&entity_type, &conflict_data);
           
           match resolution {
               ResolutionResult::UseLocal => {
                   // Local version wins, proceed with original operation
                   Ok(true)
               },
               ResolutionResult::UseRemote => {
                   // Remote version wins, update local data
                   self.apply_remote_changes(entry, &conflict_data).await?;
                   Ok(false) // Don't retry original operation
               },
               ResolutionResult::UseMerged(merged_data) => {
                   // Apply merged changes
                   self.apply_merged_changes(entry, &merged_data).await?;
                   Ok(false) // Don't retry original operation
               },
               ResolutionResult::RequiresUserInput => {
                   // Store conflict for user resolution
                   self.store_user_conflict(entry, &conflict_data).await?;
                   Ok(false)
               },
               ResolutionResult::Failed(error) => {
                   web_sys::console::error_1(&JsValue::from_str(&format!("Conflict resolution failed: {}", error)));
                   Ok(false)
               },
           }
       }
   }
   ```

5. **Add network state monitoring**:
   ```rust
   #[wasm_bindgen]
   impl SyncQueue {
       #[wasm_bindgen]
       pub fn set_network_state(&mut self, available: bool) {
           self.network_available = available;
           
           if available && !self.sync_in_progress && !self.queue.is_empty() {
               // Trigger sync when network becomes available
               wasm_bindgen_futures::spawn_local(async move {
                   if let Err(e) = self.start_sync_process().await {
                       web_sys::console::error_1(&JsValue::from_str(&format!("Sync failed: {:?}", e)));
                   }
               });
           }
       }
       
       #[wasm_bindgen]
       pub fn get_sync_status(&self) -> js_sys::Object {
           let status = js_sys::Object::new();
           js_sys::Reflect::set(&status, &"queue_size".into(), &JsValue::from(self.queue.len()))?;
           js_sys::Reflect::set(&status, &"sync_in_progress".into(), &JsValue::from(self.sync_in_progress))?;
           js_sys::Reflect::set(&status, &"network_available".into(), &JsValue::from(self.network_available))?;
           
           let pending_count = self.queue.iter()
               .filter(|e| matches!(e.status, SyncStatus::Pending))
               .count();
           js_sys::Reflect::set(&status, &"pending_operations".into(), &JsValue::from(pending_count))?;
           
           let failed_count = self.queue.iter()
               .filter(|e| matches!(e.status, SyncStatus::Failed))
               .count();
           js_sys::Reflect::set(&status, &"failed_operations".into(), &JsValue::from(failed_count))?;
           
           status
       }
       
       #[wasm_bindgen]
       pub async fn retry_failed_operations(&mut self) -> Result<u32, JsValue> {
           let mut retry_count = 0u32;
           
           for entry in &mut self.queue {
               if matches!(entry.status, SyncStatus::Failed) && entry.retry_count < self.max_retry_attempts {
                   entry.status = SyncStatus::Pending;
                   entry.retry_count = 0;
                   entry.error_message = None;
                   self.persist_queue_entry(entry).await?;
                   retry_count += 1;
               }
           }
           
           if retry_count > 0 && self.network_available && !self.sync_in_progress {
               self.start_sync_process().await?;
           }
           
           Ok(retry_count)
       }
   }
   ```

## Expected Outputs
- Robust offline sync queue with persistence
- Comprehensive conflict resolution strategies
- Priority-based operation processing
- Automatic retry with exponential backoff
- Network state monitoring and adaptive behavior
- Real-time sync status reporting

## Validation
1. Operations queue correctly during offline periods
2. Sync resumes automatically when network returns
3. Conflicts are detected and resolved appropriately
4. Failed operations retry with proper backoff
5. Queue state persists across browser sessions

## Next Steps
- Add storage persistence layer (micro-phase 9.15)
- Implement browser performance optimization (micro-phase 9.16)