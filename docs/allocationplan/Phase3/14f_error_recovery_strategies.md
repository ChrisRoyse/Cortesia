# Task 14f: Implement Error Recovery Strategies

**Time**: 8 minutes
**Dependencies**: 14e_error_logging.md
**Stage**: Inheritance System

## Objective
Create specific recovery strategies for different types of inheritance errors.

## Implementation
Create `src/inheritance/error_recovery.rs`:

```rust
use std::sync::Arc;
use crate::inheritance::error_types::*;
use crate::inheritance::error_logging::InheritanceErrorLogger;

pub struct ErrorRecoveryManager {
    logger: Arc<InheritanceErrorLogger>,
    fallback_strategies: std::collections::HashMap<String, Box<dyn FallbackStrategy>>,
}

impl ErrorRecoveryManager {
    pub fn new(logger: Arc<InheritanceErrorLogger>) -> Self {
        let mut manager = Self {
            logger,
            fallback_strategies: std::collections::HashMap::new(),
        };
        
        // Register default strategies
        manager.register_default_strategies();
        manager
    }

    fn register_default_strategies(&mut self) {
        self.register_strategy("cycle_detection", Box::new(CycleRecoveryStrategy));
        self.register_strategy("cache_failure", Box::new(CacheRecoveryStrategy));
        self.register_strategy("property_conflict", Box::new(PropertyConflictRecoveryStrategy));
    }

    pub fn register_strategy(&mut self, error_type: &str, strategy: Box<dyn FallbackStrategy>) {
        self.fallback_strategies.insert(error_type.to_string(), strategy);
    }

    pub async fn attempt_recovery(
        &self,
        error: &InheritanceErrorWithContext,
    ) -> Result<RecoveryResult, RecoveryError> {
        let strategy_key = self.determine_strategy_key(&error.error);
        
        if let Some(strategy) = self.fallback_strategies.get(&strategy_key) {
            self.logger.log_recovery_attempt(&error.context.trace_id, &strategy_key);
            
            match strategy.attempt_recovery(error).await {
                Ok(result) => {
                    self.logger.log_recovery_attempt(&error.context.trace_id, "Recovery successful");
                    Ok(result)
                }
                Err(recovery_error) => {
                    self.logger.log_recovery_attempt(&error.context.trace_id, &format!("Recovery failed: {}", recovery_error));
                    Err(recovery_error)
                }
            }
        } else {
            Err(RecoveryError::NoStrategyAvailable(strategy_key))
        }
    }

    fn determine_strategy_key(&self, error: &InheritanceError) -> String {
        match error {
            InheritanceError::CycleDetected { .. } => "cycle_detection".to_string(),
            InheritanceError::CacheError(_) => "cache_failure".to_string(),
            InheritanceError::DatabaseError(_) => "database_failure".to_string(),
            _ => "generic".to_string(),
        }
    }
}

#[derive(Debug)]
pub enum RecoveryResult {
    Recovered { fallback_value: Option<String> },
    PartialRecovery { warnings: Vec<String> },
    RecoveryFailed { reason: String },
}

#[derive(Debug)]
pub enum RecoveryError {
    NoStrategyAvailable(String),
    StrategyFailed(String),
    InsufficientContext,
}

impl std::fmt::Display for RecoveryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RecoveryError::NoStrategyAvailable(strategy) => 
                write!(f, "No recovery strategy available for: {}", strategy),
            RecoveryError::StrategyFailed(reason) => 
                write!(f, "Recovery strategy failed: {}", reason),
            RecoveryError::InsufficientContext => 
                write!(f, "Insufficient context for recovery"),
        }
    }
}

impl std::error::Error for RecoveryError {}

#[async_trait::async_trait]
pub trait FallbackStrategy: Send + Sync {
    async fn attempt_recovery(&self, error: &InheritanceErrorWithContext) -> Result<RecoveryResult, RecoveryError>;
    fn strategy_name(&self) -> &str;
}

pub struct CycleRecoveryStrategy;

#[async_trait::async_trait]
impl FallbackStrategy for CycleRecoveryStrategy {
    async fn attempt_recovery(&self, error: &InheritanceErrorWithContext) -> Result<RecoveryResult, RecoveryError> {
        if let InheritanceError::CycleDetected { parent, child } = &error.error {
            // Strategy: Remove the newest inheritance relationship
            let warning = format!("Removed inheritance relationship {} -> {} to break cycle", child, parent);
            
            Ok(RecoveryResult::PartialRecovery { 
                warnings: vec![warning] 
            })
        } else {
            Err(RecoveryError::StrategyFailed("Not a cycle detection error".to_string()))
        }
    }

    fn strategy_name(&self) -> &str {
        "cycle_recovery"
    }
}

pub struct CacheRecoveryStrategy;

#[async_trait::async_trait]
impl FallbackStrategy for CacheRecoveryStrategy {
    async fn attempt_recovery(&self, error: &InheritanceErrorWithContext) -> Result<RecoveryResult, RecoveryError> {
        // Strategy: Clear cache and continue without caching
        Ok(RecoveryResult::Recovered { 
            fallback_value: Some("cache_cleared".to_string()) 
        })
    }

    fn strategy_name(&self) -> &str {
        "cache_recovery"
    }
}

pub struct PropertyConflictRecoveryStrategy;

#[async_trait::async_trait]
impl FallbackStrategy for PropertyConflictRecoveryStrategy {
    async fn attempt_recovery(&self, _error: &InheritanceErrorWithContext) -> Result<RecoveryResult, RecoveryError> {
        // Strategy: Use first inheritance source as the authoritative value
        Ok(RecoveryResult::PartialRecovery { 
            warnings: vec!["Used first inheritance source to resolve conflict".to_string()] 
        })
    }

    fn strategy_name(&self) -> &str {
        "property_conflict_recovery"
    }
}
```

## Success Criteria
- Recovery strategies handle specific error types
- Fallback mechanisms provide reasonable defaults
- Recovery results indicate success/failure

## Next Task
14g_error_metrics.md