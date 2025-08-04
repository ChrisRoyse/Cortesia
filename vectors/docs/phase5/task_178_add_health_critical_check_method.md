# Task 111: Add Health Critical Check Method

## Prerequisites Check
- [ ] Task 110 completed: Health score integrated in metrics
- [ ] Run: `cargo check` (should pass)

## Context
Add method to determine if system health is in critical state.

## Task Objective
Implement is_health_critical method for alerting triggers.

## Steps
1. Add health critical check method to ConsistencyManager:
   ```rust
   impl ConsistencyManager {
       /// Check if system health is critical
       pub async fn is_health_critical(&self) -> bool {
           let metrics = self.calculate_metrics().await;
           metrics.health_score < 0.5 || metrics.consistency_ratio < 0.8
       }
   }
   ```

## Success Criteria
- [ ] Health critical check method added
- [ ] Proper thresholds for health score and consistency ratio
- [ ] Uses calculate_metrics method
- [ ] Compiles without errors

## Time: 2 minutes