# Task 108: Implement Consistency Verification Algorithms

## Prerequisites Check
- [ ] Task 107 completed: ConsistencyManager struct implemented
- [ ] Version tracking methods are working
- [ ] Run: `cargo check` (should pass)

## Context
Implement algorithms to verify consistency between text search, vector store, and cache systems.

## Task Objective
Add methods to check and verify consistency across different systems.

## Steps
1. Add consistency verification method:
   ```rust
   impl ConsistencyManager {
       /// Verify consistency for a document across systems
       pub async fn verify_document_consistency(
           &self,
           doc_id: &str,
       ) -> ConsistencyState {
           let versions = self.versions.read().await;
           
           if let Some(doc_version) = versions.get(doc_id) {
               // Check if versions exist and match
               let has_text = doc_version.text_version.is_some();
               let has_vector = doc_version.vector_version.is_some();
               let has_cache = doc_version.cache_version.is_some();
               
               match (has_text, has_vector, has_cache) {
                   (true, true, true) => {
                       // All systems have the document - check version consistency
                       if self.versions_match(doc_version) {
                           ConsistencyState::Consistent
                       } else {
                           ConsistencyState::Inconsistent
                       }
                   }
                   (true, true, false) => {
                       // Cache miss is acceptable
                       if self.core_versions_match(doc_version) {
                           ConsistencyState::Consistent
                       } else {
                           ConsistencyState::Inconsistent
                       }
                   }
                   (false, false, false) => {
                       // Document doesn't exist anywhere - consistent absence
                       ConsistencyState::Consistent
                   }
                   _ => {
                       // Partial presence indicates inconsistency
                       ConsistencyState::Inconsistent
                   }
               }
           } else {
               ConsistencyState::Unknown
           }
       }
       
       /// Check if all versions match
       fn versions_match(&self, doc_version: &DocumentVersion) -> bool {
           match (&doc_version.text_version, &doc_version.vector_version, &doc_version.cache_version) {
               (Some(text), Some(vector), Some(cache)) => {
                   text == vector && vector == cache
               }
               _ => false,
           }
       }
       
       /// Check if core versions (text and vector) match
       fn core_versions_match(&self, doc_version: &DocumentVersion) -> bool {
           match (&doc_version.text_version, &doc_version.vector_version) {
               (Some(text), Some(vector)) => text == vector,
               _ => false,
           }
       }
   }
   ```
2. Add batch consistency checking:
   ```rust
   impl ConsistencyManager {
       /// Run consistency check on batch of documents
       pub async fn run_consistency_check(&self) -> ConsistencyCheckResult {
           let mut consistent_count = 0;
           let mut inconsistent_count = 0;
           let mut unknown_count = 0;
           let mut inconsistent_docs = Vec::new();
           
           let docs_to_check = self.get_documents_to_check().await;
           
           for doc_id in docs_to_check {
               let state = self.verify_document_consistency(&doc_id).await;
               
               // Update document state
               {
                   let mut versions = self.versions.write().await;
                   if let Some(doc_version) = versions.get_mut(&doc_id) {
                       doc_version.state = state.clone();
                   }
               }
               
               match state {
                   ConsistencyState::Consistent => consistent_count += 1,
                   ConsistencyState::Inconsistent => {
                       inconsistent_count += 1;
                       inconsistent_docs.push(doc_id.clone());
                       self.mark_inconsistent(doc_id).await;
                   }
                   _ => unknown_count += 1,
               }
           }
           
           // Update last check time
           {
               let mut last_check = self.last_check.write().await;
               *last_check = Instant::now();
           }
           
           ConsistencyCheckResult {
               consistent_count,
               inconsistent_count,
               unknown_count,
               inconsistent_documents: inconsistent_docs,
               check_timestamp: Instant::now(),
           }
       }
   }
   ```
3. Add consistency check result structure:
   ```rust
   /// Result of consistency checking operation
   #[derive(Debug, Clone)]
   pub struct ConsistencyCheckResult {
       /// Number of consistent documents
       pub consistent_count: usize,
       /// Number of inconsistent documents
       pub inconsistent_count: usize,
       /// Number of unknown state documents
       pub unknown_count: usize,
       /// List of inconsistent document IDs
       pub inconsistent_documents: Vec<String>,
       /// Timestamp of the check
       pub check_timestamp: Instant,
   }
   ```
4. Verify compilation

## Success Criteria
- [ ] Document consistency verification algorithm
- [ ] Version matching logic for all systems
- [ ] Batch consistency checking with results
- [ ] Inconsistent document identification and queuing
- [ ] Core system consistency checking (text + vector)
- [ ] ConsistencyCheckResult with comprehensive metrics
- [ ] Compiles without errors

## Time: 6 minutes

## Next Task
Task 109 will implement synchronization and repair strategies.

## Notes
Verification algorithms handle different consistency scenarios including partial presence and cache misses appropriately.