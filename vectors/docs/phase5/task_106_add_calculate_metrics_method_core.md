# Task 108: Add Calculate Metrics Method Core

## Prerequisites Check
- [ ] Task 107 completed: Record repair operation method added
- [ ] Run: `cargo check` (should pass)

## Context
Add core metrics calculation method to ConsistencyManager.

## Task Objective
Implement calculate_metrics method with document state counting.

## Steps
1. Add calculate_metrics method skeleton to ConsistencyManager:
   ```rust
   impl ConsistencyManager {
       /// Calculate current consistency metrics
       pub async fn calculate_metrics(&self) -> ConsistencyMetrics {
           let versions = self.versions.read().await;
           let last_check = *self.last_check.read().await;
           
           let total_documents = versions.len();
           let mut consistent_count = 0;
           let mut inconsistent_count = 0;
           let mut unknown_count = 0;
           
           for doc_version in versions.values() {
               match doc_version.state {
                   ConsistencyState::Consistent => consistent_count += 1,
                   ConsistencyState::Inconsistent => inconsistent_count += 1,
                   ConsistencyState::Synchronizing => inconsistent_count += 1,
                   ConsistencyState::Failed(_) => inconsistent_count += 1,
                   ConsistencyState::Unknown => unknown_count += 1,
               }
           }
           
           // Calculate consistency ratio
           let consistency_ratio = if total_documents > 0 {
               consistent_count as f64 / total_documents as f64
           } else {
               1.0
           };
           
           // Basic metrics structure (health score calculation will be added later)
           ConsistencyMetrics {
               total_documents,
               consistent_documents: consistent_count,
               inconsistent_documents: inconsistent_count,
               unknown_documents: unknown_count,
               consistency_ratio,
               avg_repair_time_ms: 0.0,
               total_repairs_attempted: 0,
               successful_repairs: 0,
               failed_repairs: 0,
               repair_success_ratio: 0.0,
               last_check,
               health_score: 0.0,
           }
       }
   }
   ```

## Success Criteria
- [ ] Core metrics calculation implemented
- [ ] Document state counting logic
- [ ] Consistency ratio calculation
- [ ] Compiles without errors

## Time: 5 minutes