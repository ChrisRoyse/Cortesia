# Task 113: Implement Repair Execution Engine

## Prerequisites Check
- [ ] Task 112 completed: automatic repair scheduler implemented
- [ ] Job scheduling and tracking is working
- [ ] Run: `cargo check` (should pass)

## Context
Implement the execution engine that actually performs repair operations on inconsistent documents.

## Task Objective
Create the RepairExecutionEngine that handles the actual repair work with timeout and error handling.

## Steps
1. Add repair execution engine struct:
   ```rust
   /// Engine that executes repair operations
   pub struct RepairExecutionEngine {
       /// Configuration
       config: AutoRepairConfig,
       /// Active repair operations
       active_operations: Arc<RwLock<HashMap<String, ActiveRepair>>>,
   }
   
   /// Active repair operation tracking
   #[derive(Debug, Clone)]
   struct ActiveRepair {
       /// Job being executed
       job: RepairJob,
       /// Start time
       started_at: Instant,
       /// Timeout deadline
       timeout_at: Instant,
   }
   ```
2. Add constructor and basic methods:
   ```rust
   impl RepairExecutionEngine {
       /// Create new repair execution engine
       pub fn new(config: AutoRepairConfig) -> Self {
           Self {
               config,
               active_operations: Arc::new(RwLock::new(HashMap::new())),
           }
       }
       
       /// Check for timed out operations
       pub async fn cleanup_timed_out_operations(&self) -> Vec<String> {
           let mut active = self.active_operations.write().await;
           let now = Instant::now();
           let mut timed_out = Vec::new();
           
           active.retain(|job_id, active_repair| {
               if now > active_repair.timeout_at {
                   timed_out.push(job_id.clone());
                   false
               } else {
                   true
               }
           });
           
           timed_out
       }
   }
   ```
3. Add repair execution method:
   ```rust
   impl RepairExecutionEngine {
       /// Execute a repair job
       pub async fn execute_repair_job(&self, job: RepairJob) -> RepairResult {
           let start_time = Instant::now();
           let timeout_at = start_time + Duration::from_secs(self.config.job_timeout_seconds);
           
           // Track active operation
           {
               let mut active = self.active_operations.write().await;
               active.insert(job.id.clone(), ActiveRepair {
                   job: job.clone(),
                   started_at: start_time,
                   timeout_at,
               });
           }
           
           // Execute repair with timeout
           let result = tokio::time::timeout(
               Duration::from_secs(self.config.job_timeout_seconds),
               self.perform_repair(&job)
           ).await;
           
           // Clean up active operation
           {
               let mut active = self.active_operations.write().await;
               active.remove(&job.id);
           }
           
           match result {
               Ok(repair_result) => repair_result,
               Err(_) => RepairResult {
                   doc_id: job.doc_id,
                   success: false,
                   updated_systems: Vec::new(),
                   error: Some("Repair operation timed out".to_string()),
               }
           }
       }
       
       /// Perform the actual repair operation
       async fn perform_repair(&self, job: &RepairJob) -> RepairResult {
           match job.strategy {
               SyncStrategy::TextSourceOfTruth => {
                   self.repair_from_text_source(&job.doc_id).await
               }
               SyncStrategy::VectorSourceOfTruth => {
                   self.repair_from_vector_source(&job.doc_id).await
               }
               SyncStrategy::MostRecent => {
                   self.repair_from_most_recent(&job.doc_id).await
               }
               SyncStrategy::ManualResolution => {
                   RepairResult {
                       doc_id: job.doc_id.clone(),
                       success: false,
                       updated_systems: Vec::new(),
                       error: Some("Manual resolution required".to_string()),
                   }
               }
           }
       }
   }
   ```
4. Add specific repair strategy implementations:
   ```rust
   impl RepairExecutionEngine {
       /// Repair document using text search as source of truth
       async fn repair_from_text_source(&self, doc_id: &str) -> RepairResult {
           // Simulate repair process - actual implementation would:
           // 1. Fetch document from text search system
           // 2. Update vector store with document
           // 3. Update cache if needed
           // 4. Verify consistency
           
           let mut updated_systems = Vec::new();
           let mut success = true;
           let mut error = None;
           
           // Simulate vector store update
           match self.update_vector_store(doc_id).await {
               Ok(_) => updated_systems.push("vector".to_string()),
               Err(e) => {
                   success = false;
                   error = Some(format!("Vector store update failed: {}", e));
               }
           }
           
           // Simulate cache update if vector update succeeded
           if success {
               if let Err(e) = self.update_cache(doc_id).await {
                   // Cache update failure is not critical
                   eprintln!("Cache update warning: {}", e);
               } else {
                   updated_systems.push("cache".to_string());
               }
           }
           
           RepairResult {
               doc_id: doc_id.to_string(),
               success,
               updated_systems,
               error,
           }
       }
       
       /// Repair document using vector store as source of truth
       async fn repair_from_vector_source(&self, doc_id: &str) -> RepairResult {
           let mut updated_systems = Vec::new();
           let mut success = true;
           let mut error = None;
           
           // Simulate text search update
           match self.update_text_search(doc_id).await {
               Ok(_) => updated_systems.push("text".to_string()),
               Err(e) => {
                   success = false;
                   error = Some(format!("Text search update failed: {}", e));
               }
           }
           
           // Simulate cache update if text update succeeded
           if success {
               if let Err(e) = self.update_cache(doc_id).await {
                   eprintln!("Cache update warning: {}", e);
               } else {
                   updated_systems.push("cache".to_string());
               }
           }
           
           RepairResult {
               doc_id: doc_id.to_string(),
               success,
               updated_systems,
               error,
           }
       }
       
       /// Repair document using most recent version
       async fn repair_from_most_recent(&self, doc_id: &str) -> RepairResult {
           // Simplified implementation - would need version comparison logic
           self.repair_from_text_source(doc_id).await
       }
   }
   ```
5. Add system update helpers (simulation):
   ```rust
   impl RepairExecutionEngine {
       /// Update vector store (simulated)
       async fn update_vector_store(&self, _doc_id: &str) -> Result<(), String> {
           // Simulate update operation
           tokio::time::sleep(Duration::from_millis(100)).await;
           Ok(())
       }
       
       /// Update text search (simulated)
       async fn update_text_search(&self, _doc_id: &str) -> Result<(), String> {
           tokio::time::sleep(Duration::from_millis(150)).await;
           Ok(())
       }
       
       /// Update cache (simulated)
       async fn update_cache(&self, _doc_id: &str) -> Result<(), String> {
           tokio::time::sleep(Duration::from_millis(50)).await;
           Ok(())
       }
   }
   ```
6. Verify compilation

## Success Criteria
- [ ] RepairExecutionEngine with timeout handling
- [ ] Active operation tracking with cleanup
- [ ] Repair strategy implementations for each sync type
- [ ] System update simulation methods
- [ ] Timeout protection for repair operations
- [ ] Error handling with descriptive messages
- [ ] Non-critical cache update handling
- [ ] Compiles without errors

## Time: 7 minutes

## Next Task
Task 114 will implement health monitoring and automatic trigger detection.

## Notes
Execution engine provides robust repair operations with timeout protection and proper error handling for different repair strategies.