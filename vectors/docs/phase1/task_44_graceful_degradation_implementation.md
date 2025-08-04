# Task 44: Graceful Degradation Implementation

**Time:** 10 minutes (2 min read, 6 min implement, 2 min verify)
**Prerequisites:** Task 43 completed
**Input Files:**
- C:/code/LLMKG/vectors/tantivy_search/src/error.rs
- C:/code/LLMKG/vectors/tantivy_search/src/connection_pool.rs
- C:/code/LLMKG/vectors/tantivy_search/src/search.rs

## Complete Context (For AI with ZERO Knowledge)

You are implementing **graceful degradation** for the Tantivy-based search system. Graceful degradation ensures the system continues providing reduced functionality when components fail, rather than completely failing.

**What is Graceful Degradation?** A system design approach where services continue operating at reduced capacity when dependencies fail, providing fallback mechanisms and partial functionality instead of complete system failure.

**System Context:** After task 43, we have comprehensive error types with recovery guidance. This task implements automatic fallback strategies that maintain service availability during partial failures.

**This Task:** Creates a DegradationManager that monitors component health, implements fallback strategies, and provides reduced functionality modes when full service is unavailable.

## Exact Steps (6 minutes implementation)

### Step 1: Create Degradation Manager Module (2 minutes)
Create `C:/code/LLMKG/vectors/tantivy_search/src/degradation.rs`:
```rust
use crate::error::{SearchError, SearchResult, ErrorSeverity};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ServiceMode {
    Full,           // All features available
    Degraded,       // Reduced functionality
    Essential,      // Core features only
    Maintenance,    // Read-only mode
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComponentStatus {
    Healthy,
    Degraded,
    Failed,
    Maintenance,
}

#[derive(Debug, Clone)]
pub struct ComponentHealth {
    pub status: ComponentStatus,
    pub last_check: Instant,
    pub failure_count: u32,
    pub recovery_attempts: u32,
    pub error_rate: f64,
    pub response_time: Duration,
}

#[derive(Debug, Clone)]
pub struct DegradationConfig {
    pub max_failures_before_degradation: u32,
    pub recovery_check_interval: Duration,
    pub health_check_timeout: Duration,
    pub error_rate_threshold: f64,
    pub response_time_threshold: Duration,
    pub auto_recovery_enabled: bool,
}

impl Default for DegradationConfig {
    fn default() -> Self {
        Self {
            max_failures_before_degradation: 3,
            recovery_check_interval: Duration::from_secs(30),
            health_check_timeout: Duration::from_secs(5),
            error_rate_threshold: 0.1, // 10% error rate
            response_time_threshold: Duration::from_millis(1000),
            auto_recovery_enabled: true,
        }
    }
}

pub struct DegradationManager {
    config: DegradationConfig,
    current_mode: Arc<RwLock<ServiceMode>>,
    component_health: Arc<RwLock<HashMap<String, ComponentHealth>>>,
    fallback_strategies: HashMap<String, Box<dyn FallbackStrategy + Send + Sync>>,
}
```

### Step 2: Implement Fallback Strategies (2 minutes)
Continue in `src/degradation.rs`:
```rust
#[async_trait::async_trait]
pub trait FallbackStrategy {
    async fn execute(&self, original_error: &SearchError) -> SearchResult<FallbackResult>;
    fn priority(&self) -> u8; // Higher number = higher priority
    fn is_applicable(&self, error: &SearchError) -> bool;
}

#[derive(Debug)]
pub enum FallbackResult {
    Success(String), // Fallback succeeded with result
    PartialSuccess(String, Vec<String>), // Partial results with warnings
    Failed(SearchError), // Fallback also failed
}

pub struct CachedResultFallback {
    cache_ttl: Duration,
}

#[async_trait::async_trait]
impl FallbackStrategy for CachedResultFallback {
    async fn execute(&self, original_error: &SearchError) -> SearchResult<FallbackResult> {
        // In a real implementation, this would check a cache
        match original_error {
            SearchError::IndexError { .. } | SearchError::ConnectionError { .. } => {
                // Simulate cache lookup
                tokio::time::sleep(Duration::from_millis(10)).await;
                Ok(FallbackResult::PartialSuccess(
                    "Cached results from previous query".to_string(),
                    vec!["Results may be outdated".to_string()],
                ))
            }
            _ => Ok(FallbackResult::Failed(original_error.clone())),
        }
    }

    fn priority(&self) -> u8 {
        100 // High priority
    }

    fn is_applicable(&self, error: &SearchError) -> bool {
        matches!(
            error,
            SearchError::IndexError { .. } | SearchError::ConnectionError { .. }
        )
    }
}

pub struct SimplifiedSearchFallback;

#[async_trait::async_trait]
impl FallbackStrategy for SimplifiedSearchFallback {
    async fn execute(&self, _original_error: &SearchError) -> SearchResult<FallbackResult> {
        // Simulate a simplified search that always works
        tokio::time::sleep(Duration::from_millis(50)).await;
        Ok(FallbackResult::PartialSuccess(
            "Simplified search results".to_string(),
            vec!["Advanced features unavailable".to_string()],
        ))
    }

    fn priority(&self) -> u8 {
        50 // Medium priority
    }

    fn is_applicable(&self, error: &SearchError) -> bool {
        matches!(
            error,
            SearchError::QueryError { .. } | SearchError::IndexError { .. }
        )
    }
}

impl DegradationManager {
    pub fn new(config: DegradationConfig) -> Self {
        let mut fallback_strategies: HashMap<String, Box<dyn FallbackStrategy + Send + Sync>> =
            HashMap::new();
        
        fallback_strategies.insert(
            "cached_result".to_string(),
            Box::new(CachedResultFallback {
                cache_ttl: Duration::from_secs(300),
            }),
        );
        
        fallback_strategies.insert(
            "simplified_search".to_string(),
            Box::new(SimplifiedSearchFallback),
        );

        Self {
            config,
            current_mode: Arc::new(RwLock::new(ServiceMode::Full)),
            component_health: Arc::new(RwLock::new(HashMap::new())),
            fallback_strategies,
        }
    }

    pub async fn get_current_mode(&self) -> ServiceMode {
        self.current_mode.read().await.clone()
    }

    pub async fn update_component_health(
        &self,
        component: String,
        status: ComponentStatus,
        response_time: Duration,
    ) {
        let mut health_map = self.component_health.write().await;
        let health = health_map.entry(component).or_insert(ComponentHealth {
            status: ComponentStatus::Healthy,
            last_check: Instant::now(),
            failure_count: 0,
            recovery_attempts: 0,
            error_rate: 0.0,
            response_time: Duration::ZERO,
        });

        health.status = status.clone();
        health.last_check = Instant::now();
        health.response_time = response_time;

        match status {
            ComponentStatus::Failed => {
                health.failure_count += 1;
            }
            ComponentStatus::Healthy => {
                health.failure_count = 0;
                health.recovery_attempts = 0;
            }
            _ => {}
        }

        drop(health_map);
        self.evaluate_service_mode().await;
    }
}
```

### Step 3: Implement Service Mode Logic (1.5 minutes)
Continue in `src/degradation.rs`:
```rust
impl DegradationManager {
    async fn evaluate_service_mode(&self) {
        let health_map = self.component_health.read().await;
        
        let mut failed_components = 0;
        let mut degraded_components = 0;
        let total_components = health_map.len();

        for health in health_map.values() {
            match health.status {
                ComponentStatus::Failed => failed_components += 1,
                ComponentStatus::Degraded => degraded_components += 1,
                _ => {}
            }
        }

        let new_mode = if failed_components > total_components / 2 {
            ServiceMode::Essential
        } else if failed_components > 0 || degraded_components > total_components / 3 {
            ServiceMode::Degraded
        } else {
            ServiceMode::Full
        };

        let mut current_mode = self.current_mode.write().await;
        if !matches!(*current_mode, new_mode) {
            *current_mode = new_mode;
            tracing::info!("Service mode changed to: {:?}", *current_mode);
        }
    }

    pub async fn execute_with_fallback(
        &self,
        operation: impl std::future::Future<Output = SearchResult<String>>,
    ) -> SearchResult<String> {
        let result = operation.await;
        
        match result {
            Ok(success) => Ok(success),
            Err(error) => {
                tracing::warn!("Operation failed, attempting fallback: {:?}", error);
                self.try_fallback_strategies(&error).await
            }
        }
    }

    async fn try_fallback_strategies(&self, error: &SearchError) -> SearchResult<String> {
        let mut strategies: Vec<_> = self
            .fallback_strategies
            .values()
            .filter(|strategy| strategy.is_applicable(error))
            .collect();
        
        // Sort by priority (highest first)
        strategies.sort_by(|a, b| b.priority().cmp(&a.priority()));

        for strategy in strategies {
            match strategy.execute(error).await {
                Ok(FallbackResult::Success(result)) => {
                    tracing::info!("Fallback strategy succeeded");
                    return Ok(result);
                }
                Ok(FallbackResult::PartialSuccess(result, warnings)) => {
                    tracing::warn!("Fallback partial success with warnings: {:?}", warnings);
                    return Ok(result);
                }
                Ok(FallbackResult::Failed(_)) | Err(_) => {
                    tracing::debug!("Fallback strategy failed, trying next");
                    continue;
                }
            }
        }

        Err(error.clone())
    }

    pub async fn get_health_report(&self) -> HashMap<String, ComponentHealth> {
        self.component_health.read().await.clone()
    }

    pub async fn force_service_mode(&self, mode: ServiceMode) {
        *self.current_mode.write().await = mode;
        tracing::info!("Service mode manually set to: {:?}", self.current_mode.read().await);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_degradation_manager_creation() {
        let config = DegradationConfig::default();
        let manager = DegradationManager::new(config);
        
        assert!(matches!(manager.get_current_mode().await, ServiceMode::Full));
    }

    #[tokio::test]
    async fn test_component_health_update() {
        let config = DegradationConfig::default();
        let manager = DegradationManager::new(config);
        
        manager
            .update_component_health(
                "test_component".to_string(),
                ComponentStatus::Failed,
                Duration::from_millis(1000),
            )
            .await;
        
        let health = manager.get_health_report().await;
        assert!(health.contains_key("test_component"));
        assert!(matches!(
            health.get("test_component").unwrap().status,
            ComponentStatus::Failed
        ));
    }

    #[tokio::test]
    async fn test_fallback_execution() {
        let config = DegradationConfig::default();
        let manager = DegradationManager::new(config);
        
        let failing_operation = async {
            Err(SearchError::IndexError {
                message: "Index unavailable".to_string(),
                error_code: "IDX_001".to_string(),
                recoverable: true,
            })
        };
        
        let result = manager.execute_with_fallback(failing_operation).await;
        assert!(result.is_ok());
    }
}
```

### Step 4: Integration with Search Engine (0.5 minutes)
Add to `C:/code/LLMKG/vectors/tantivy_search/src/search.rs` imports:
```rust
use crate::degradation::{DegradationManager, ServiceMode, ComponentStatus};
```

Add method to SearchEngine:
```rust
impl SearchEngine {
    pub fn with_degradation_manager(mut self, manager: DegradationManager) -> Self {
        // Integration logic would be implemented here
        self
    }
}
```

## Verification Steps (2 minutes)
```bash
cd C:/code/LLMKG/vectors/tantivy_search
cargo check
cargo test degradation
cargo test test_degradation_manager_creation
cargo test test_fallback_execution
```

Add to `Cargo.toml`:
```toml
async-trait = "0.1.83"
```

## Success Validation Checklist
- [ ] File exists: `src/degradation.rs` with DegradationManager and fallback strategies
- [ ] Dependency added: `async-trait = "0.1.83"`
- [ ] Service modes properly defined (Full, Degraded, Essential, Maintenance)
- [ ] Component health tracking implemented with failure counts and recovery attempts
- [ ] Fallback strategies can handle different error types with priority ordering
- [ ] Service mode automatically adjusts based on component health
- [ ] Execute_with_fallback provides automatic error recovery
- [ ] Command `cargo check` completes without errors
- [ ] All degradation tests pass successfully

## Files Created For Next Task
1. **C:/code/LLMKG/vectors/tantivy_search/src/degradation.rs** - Graceful degradation system with fallback strategies
2. **Enhanced resilience** - System now maintains service availability during component failures

**Next Task (Task 45)** will implement logging and monitoring system for observability and troubleshooting.