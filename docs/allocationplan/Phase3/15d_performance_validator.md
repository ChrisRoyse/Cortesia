# Task 15d: Implement Performance Validator

**Time**: 6 minutes (1 min read, 4 min implement, 1 min verify)
**Dependencies**: 15c_semantic_validator.md
**Stage**: Inheritance System

## Objective
Create validator for performance characteristics of inheritance operations.

## Implementation
Create `src/inheritance/validation/performance_validator.rs`:

```rust
use std::sync::Arc;
use std::time::{Duration, Instant};
use crate::inheritance::validation::rules::*;
use crate::inheritance::cache::InheritanceCacheManager;

pub struct PerformanceValidator {
    cache_manager: Arc<InheritanceCacheManager>,
    rules: PerformanceRules,
}

impl PerformanceValidator {
    pub fn new(cache_manager: Arc<InheritanceCacheManager>, rules: PerformanceRules) -> Self {
        Self {
            cache_manager,
            rules,
        }
    }

    pub async fn validate_resolution_performance(&self, concept_id: &str) -> Result<Vec<ValidationResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();
        
        let start_time = Instant::now();
        
        // Simulate property resolution (in real implementation, this would be actual resolution)
        tokio::time::sleep(Duration::from_millis(10)).await;
        
        let resolution_time = start_time.elapsed();
        let resolution_time_ms = resolution_time.as_millis() as u64;
        
        if resolution_time_ms > self.rules.max_resolution_time_ms {
            results.push(
                ValidationResult::new(
                    "slow_resolution",
                    ValidationSeverity::Warning,
                    &format!("Property resolution took {}ms, exceeding limit of {}ms", 
                           resolution_time_ms, self.rules.max_resolution_time_ms)
                )
                .with_concept(concept_id)
                .with_suggestion("Consider caching or optimizing the inheritance chain")
            );
        }
        
        Ok(results)
    }

    pub async fn validate_cache_usage(&self) -> Result<Vec<ValidationResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();
        
        let metrics = self.cache_manager.get_detailed_metrics().await;
        
        // Check memory usage
        if metrics.memory_usage_mb > self.rules.max_cache_memory_mb as f64 {
            results.push(
                ValidationResult::new(
                    "high_cache_memory",
                    ValidationSeverity::Warning,
                    &format!("Cache memory usage {:.1}MB exceeds limit of {}MB", 
                           metrics.memory_usage_mb, self.rules.max_cache_memory_mb)
                )
                .with_suggestion("Consider reducing cache size or clearing old entries")
            );
        }
        
        // Check cache hit rate
        if metrics.hit_rate < 0.7 {
            results.push(
                ValidationResult::new(
                    "low_cache_hit_rate",
                    ValidationSeverity::Info,
                    &format!("Cache hit rate {:.1}% is below optimal threshold", metrics.hit_rate * 100.0)
                )
                .with_suggestion("Review caching strategy or increase cache size")
            );
        }
        
        Ok(results)
    }

    pub async fn validate_inheritance_chain_depth(&self, concept_id: &str) -> Result<Vec<ValidationResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();
        
        if self.rules.warn_on_deep_chains {
            if let Some(cached_chain) = self.cache_manager.get_cached_chain(concept_id).await {
                let depth = cached_chain.chain.total_depth;
                
                if depth > 10 {
                    results.push(
                        ValidationResult::new(
                            "deep_inheritance_chain",
                            ValidationSeverity::Warning,
                            &format!("Inheritance chain depth {} may impact performance", depth)
                        )
                        .with_concept(concept_id)
                        .with_suggestion("Consider flattening the inheritance hierarchy")
                    );
                }
            }
        }
        
        Ok(results)
    }

    pub async fn validate_concurrent_operations(&self) -> Result<Vec<ValidationResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();
        
        // In a real implementation, this would track active operations
        let current_operations = self.get_current_operation_count().await;
        
        if current_operations > self.rules.max_concurrent_operations {
            results.push(
                ValidationResult::new(
                    "high_concurrent_operations",
                    ValidationSeverity::Warning,
                    &format!("Currently {} operations running, exceeding limit of {}", 
                           current_operations, self.rules.max_concurrent_operations)
                )
                .with_suggestion("Consider implementing rate limiting or queuing")
            );
        }
        
        Ok(results)
    }

    pub async fn validate_database_query_performance(&self, concept_id: &str) -> Result<Vec<ValidationResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();
        
        let start_time = Instant::now();
        
        // Simulate database query (in real implementation, this would be actual query)
        tokio::time::sleep(Duration::from_millis(5)).await;
        
        let query_time = start_time.elapsed();
        let query_time_ms = query_time.as_millis() as u64;
        
        if query_time_ms > 100 {
            results.push(
                ValidationResult::new(
                    "slow_database_query",
                    ValidationSeverity::Warning,
                    &format!("Database query took {}ms for concept '{}'", query_time_ms, concept_id)
                )
                .with_concept(concept_id)
                .with_suggestion("Consider adding database indexes or optimizing queries")
            );
        }
        
        Ok(results)
    }

    pub async fn validate_memory_usage_pattern(&self, concept_id: &str) -> Result<Vec<ValidationResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();
        
        // Check if concept causes memory spikes
        let memory_usage = self.estimate_concept_memory_usage(concept_id).await;
        
        if memory_usage > 10 * 1024 * 1024 { // 10MB
            results.push(
                ValidationResult::new(
                    "high_memory_usage",
                    ValidationSeverity::Warning,
                    &format!("Concept '{}' uses approximately {:.1}MB of memory", 
                           concept_id, memory_usage as f64 / 1024.0 / 1024.0)
                )
                .with_concept(concept_id)
                .with_suggestion("Consider simplifying the concept or its properties")
            );
        }
        
        Ok(results)
    }

    async fn get_current_operation_count(&self) -> u32 {
        // In a real implementation, this would track active operations
        // For now, return a simulated count
        5
    }

    async fn estimate_concept_memory_usage(&self, _concept_id: &str) -> usize {
        // In a real implementation, this would calculate actual memory usage
        // For now, return a simulated value
        1024 * 1024 // 1MB
    }

    pub async fn generate_performance_report(&self) -> PerformanceReport {
        let cache_metrics = self.cache_manager.get_detailed_metrics().await;
        
        PerformanceReport {
            cache_hit_rate: cache_metrics.hit_rate,
            memory_usage_mb: cache_metrics.memory_usage_mb,
            average_resolution_time_ms: cache_metrics.basic_stats.average_access_time_ms,
            total_operations: cache_metrics.basic_stats.hit_count + cache_metrics.basic_stats.miss_count,
            recommendations: self.generate_recommendations(&cache_metrics).await,
        }
    }

    async fn generate_recommendations(&self, metrics: &crate::inheritance::cache::DetailedCacheMetrics) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        if metrics.hit_rate < 0.8 {
            recommendations.push("Consider increasing cache size to improve hit rate".to_string());
        }
        
        if metrics.memory_usage_mb > self.rules.max_cache_memory_mb as f64 * 0.9 {
            recommendations.push("Cache memory usage is near limit, consider optimization".to_string());
        }
        
        if metrics.average_chain_depth > 15.0 {
            recommendations.push("Average inheritance depth is high, consider flattening hierarchies".to_string());
        }
        
        recommendations
    }
}

#[derive(Debug)]
pub struct PerformanceReport {
    pub cache_hit_rate: f64,
    pub memory_usage_mb: f64,
    pub average_resolution_time_ms: f64,
    pub total_operations: u64,
    pub recommendations: Vec<String>,
}
```

## Success Criteria
- Validates resolution time performance
- Monitors cache usage effectively
- Provides actionable recommendations

## Next Task
15e_custom_rule_engine.md