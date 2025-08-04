# Task 116: Integrate Health Score in Metrics

## Prerequisites Check
- [ ] Task 109 completed: Health score calculation added
- [ ] Run: `cargo check` (should pass)

## Context
Integrate health score calculation into the main calculate_metrics method.

## Task Objective
Update calculate_metrics to use calculate_health_score method.

## Steps
1. Update calculate_metrics method to include health score:
   ```rust
   impl ConsistencyManager {
       /// Calculate current consistency metrics (updated)
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
           
           let consistency_ratio = if total_documents > 0 {
               consistent_count as f64 / total_documents as f64
           } else {
               1.0
           };
           
           // Calculate health score based on consistency ratio and recent activity
           let health_score = self.calculate_health_score(consistency_ratio, &last_check);
           
           ConsistencyMetrics {
               total_documents,
               consistent_documents: consistent_count,
               inconsistent_documents: inconsistent_count,
               unknown_documents: unknown_count,
               consistency_ratio,
               avg_repair_time_ms: 0.0, // Would be calculated from repair history
               total_repairs_attempted: 0, // Would be tracked
               successful_repairs: 0, // Would be tracked
               failed_repairs: 0, // Would be tracked
               repair_success_ratio: 0.0, // Would be calculated
               last_check,
               health_score,
           }
       }
   }
   ```

## Success Criteria
- [ ] Health score integrated into metrics calculation
- [ ] Method properly calls calculate_health_score
- [ ] Complete metrics structure returned
- [ ] Compiles without errors

## Time: 3 minutes