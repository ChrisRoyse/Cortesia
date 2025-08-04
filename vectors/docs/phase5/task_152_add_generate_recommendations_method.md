# Task 152: Add Generate Recommendations Method

## Prerequisites Check
- [ ] Task 151 completed: mode comparison logic added
- [ ] Run: `cargo check` (should pass)

## Context
Add method to generate optimization recommendations based on strategies and current performance.

## Task Objective
Implement generate_optimization_recommendations method to evaluate strategies and create recommendations.

## Steps
1. Add generate_optimization_recommendations method to QueryOptimizer impl block:
   ```rust
   /// Generate optimization recommendations
   pub async fn generate_optimization_recommendations(&self) -> Vec<OptimizationRecommendation> {
       if !self.config.enabled {
           return Vec::new();
       }
       
       let mut recommendations = Vec::new();
       
       // Analyze each strategy
       for strategy in &self.strategies {
           if let Some(recommendation) = self.evaluate_strategy(strategy).await {
               recommendations.push(recommendation);
           }
       }
       
       recommendations
   }
   ```

## Success Criteria
- [ ] generate_optimization_recommendations method added
- [ ] Strategy evaluation loop implemented
- [ ] Configuration check for enabled state
- [ ] Compiles without errors

## Time: 4 minutes