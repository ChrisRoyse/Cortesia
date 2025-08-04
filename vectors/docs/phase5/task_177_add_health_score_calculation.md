# Task 109: Add Health Score Calculation

## Prerequisites Check
- [ ] Task 108 completed: Calculate metrics method core added
- [ ] Run: `cargo check` (should pass)

## Context
Add health score calculation method with consistency and freshness factors.

## Task Objective
Implement calculate_health_score method with weighted scoring.

## Steps
1. Add health score calculation method to ConsistencyManager:
   ```rust
   impl ConsistencyManager {
       /// Calculate system health score
       fn calculate_health_score(&self, consistency_ratio: f64, last_check: &Instant) -> f64 {
           let consistency_weight = 0.7;
           let freshness_weight = 0.3;
           
           // Freshness score based on how recent the last check was
           let check_age_hours = last_check.elapsed().as_secs_f64() / 3600.0;
           let freshness_score = if check_age_hours < 1.0 {
               1.0
           } else if check_age_hours < 6.0 {
               1.0 - (check_age_hours - 1.0) / 5.0 * 0.5
           } else {
               0.5
           };
           
           consistency_ratio * consistency_weight + freshness_score * freshness_weight
       }
   }
   ```

## Success Criteria
- [ ] Health score calculation method added
- [ ] Weighted scoring with consistency and freshness
- [ ] Time-based freshness scoring
- [ ] Compiles without errors

## Time: 4 minutes