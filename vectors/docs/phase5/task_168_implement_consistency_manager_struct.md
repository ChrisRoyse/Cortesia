# Task 107: Implement ConsistencyManager Struct

## Prerequisites Check
- [ ] Task 106 completed: consistency module foundation created
- [ ] DocumentVersion and ConsistencyState are defined
- [ ] Run: `cargo check` (should pass)

## Context
Implement the main ConsistencyManager struct that monitors and maintains consistency across systems.

## Task Objective
Create the ConsistencyManager struct with version tracking and monitoring capabilities.

## Steps
1. Add ConsistencyManager struct:
   ```rust
   /// Cross-system consistency manager
   pub struct ConsistencyManager {
       /// Document version tracking
       versions: Arc<RwLock<HashMap<String, DocumentVersion>>>,
       /// Configuration
       config: ConsistencyConfig,
       /// Last consistency check timestamp
       last_check: Arc<RwLock<Instant>>,
       /// Inconsistent documents queue
       inconsistent_queue: Arc<RwLock<Vec<String>>>,
       /// System references (will be added later)
       // text_engine: Arc<TextSearchEngine>,
       // vector_store: Arc<VectorStore>,
       // cache: Arc<RwLock<MemoryEfficientCache>>,
   }
   ```
2. Add constructor and basic methods:
   ```rust
   impl ConsistencyManager {
       /// Create new consistency manager
       pub fn new(config: ConsistencyConfig) -> Self {
           Self {
               versions: Arc::new(RwLock::new(HashMap::new())),
               config,
               last_check: Arc::new(RwLock::new(Instant::now())),
               inconsistent_queue: Arc::new(RwLock::new(Vec::new())),
           }
       }
       
       /// Get current configuration
       pub fn config(&self) -> &ConsistencyConfig {
           &self.config
       }
       
       /// Update configuration
       pub fn update_config(&mut self, config: ConsistencyConfig) {
           self.config = config;
       }
   }
   ```
3. Add version tracking methods:
   ```rust
   impl ConsistencyManager {
       /// Track document version update
       pub async fn track_document_update(
           &self,
           doc_id: String,
           system: &str,
           version: String,
       ) {
           let mut versions = self.versions.write().await;
           let doc_version = versions.entry(doc_id.clone()).or_insert_with(|| {
               DocumentVersion {
                   id: doc_id,
                   text_version: None,
                   vector_version: None,
                   cache_version: None,
                   last_updated: Instant::now(),
                   state: ConsistencyState::Unknown,
               }
           });
           
           match system {
               "text" => doc_version.text_version = Some(version),
               "vector" => doc_version.vector_version = Some(version),
               "cache" => doc_version.cache_version = Some(version),
               _ => {} // Unknown system
           }
           
           doc_version.last_updated = Instant::now();
           doc_version.state = ConsistencyState::Unknown; // Will be checked later
       }
       
       /// Get document version info
       pub async fn get_document_version(&self, doc_id: &str) -> Option<DocumentVersion> {
           let versions = self.versions.read().await;
           versions.get(doc_id).cloned()
       }
   }
   ```
4. Add consistency checking preparation:
   ```rust
   impl ConsistencyManager {
       /// Check if consistency check is needed
       pub async fn needs_consistency_check(&self) -> bool {
           let last_check = self.last_check.read().await;
           last_check.elapsed().as_secs() >= self.config.check_interval
       }
       
       /// Get list of documents to check
       pub async fn get_documents_to_check(&self) -> Vec<String> {
           let versions = self.versions.read().await;
           versions.keys()
               .take(self.config.batch_size)
               .cloned()
               .collect()
       }
       
       /// Mark document as inconsistent
       pub async fn mark_inconsistent(&self, doc_id: String) {
           let mut queue = self.inconsistent_queue.write().await;
           if !queue.contains(&doc_id) {
               queue.push(doc_id);
           }
       }
   }
   ```
5. Verify compilation

## Success Criteria
- [ ] ConsistencyManager struct with version tracking
- [ ] Document version update tracking per system
- [ ] Inconsistent document queue management
- [ ] Batch consistency checking support
- [ ] Configuration management methods
- [ ] Thread-safe operations with RwLock
- [ ] Compiles without errors

## Time: 5 minutes

## Next Task
Task 108 will implement consistency verification algorithms.

## Notes
ConsistencyManager provides foundation for tracking document versions across systems and queuing inconsistent documents for repair.