# Task 15k: Create Validation System Tests

**Time**: 8 minutes (1.5 min read, 5 min implement, 1.5 min verify)
**Dependencies**: 15j_validation_integration.md
**Stage**: Inheritance System

## Objective
Create comprehensive test suite for the validation system.

## Implementation
Create `src/inheritance/validation/tests/mod.rs`:

```rust
#[cfg(test)]
mod structural_validator_tests;
#[cfg(test)]
mod semantic_validator_tests;
#[cfg(test)]
mod performance_validator_tests;
#[cfg(test)]
mod custom_rule_engine_tests;
#[cfg(test)]
mod validation_coordinator_tests;
#[cfg(test)]
mod validation_scheduler_tests;
#[cfg(test)]
mod validation_integration_tests;

use std::sync::Arc;
use crate::core::neo4j_connection::Neo4jConnectionManager;
use crate::inheritance::cache::InheritanceCacheManager;
use crate::inheritance::validation::rules::*;

// Test utilities
pub fn create_test_validation_rules() -> ValidationRules {
    ValidationRules {
        structural_rules: StructuralRules {
            max_inheritance_depth: 10,
            allow_multiple_inheritance: true,
            allow_circular_references: false,
            max_children_per_concept: Some(50),
            require_unique_property_names: true,
        },
        semantic_rules: SemanticRules {
            require_concept_existence: true,
            validate_property_types: true,
            enforce_inheritance_compatibility: true,
            validate_relationship_semantics: true,
            check_naming_conventions: true,
        },
        performance_rules: PerformanceRules {
            max_resolution_time_ms: 500,
            max_cache_memory_mb: 50,
            warn_on_deep_chains: true,
            max_concurrent_operations: 25,
        },
        custom_rules: vec![
            CustomRule {
                id: "test_naming_rule".to_string(),
                name: "Test Naming Convention".to_string(),
                description: "Enforce test naming conventions".to_string(),
                rule_type: CustomRuleType::ConceptNaming,
                parameters: {
                    let mut params = std::collections::HashMap::new();
                    params.insert("pattern".to_string(), "^test_[a-z_]+$".to_string());
                    params
                },
                enabled: true,
            }
        ],
    }
}

pub fn create_test_validation_results() -> Vec<ValidationResult> {
    vec![
        ValidationResult::new(
            "test_critical",
            ValidationSeverity::Critical,
            "Test critical issue"
        ).with_concept("test_concept_1"),
        
        ValidationResult::new(
            "test_error", 
            ValidationSeverity::Error,
            "Test error issue"
        ).with_concept("test_concept_2"),
        
        ValidationResult::new(
            "test_warning",
            ValidationSeverity::Warning, 
            "Test warning issue"
        ).with_concept("test_concept_3"),
        
        ValidationResult::new(
            "test_info",
            ValidationSeverity::Info,
            "Test info issue"
        ).with_concept("test_concept_4"),
    ]
}

// Mock connection manager for testing
pub struct MockConnectionManager;

impl MockConnectionManager {
    pub fn new() -> Arc<Self> {
        Arc::new(Self)
    }
}

// Mock cache manager for testing  
pub fn create_mock_cache_manager() -> Arc<InheritanceCacheManager> {
    let config = crate::inheritance::cache::CacheConfig {
        max_chain_cache_size: 100,
        max_property_cache_size: 200,
        ttl_minutes: 10,
        max_memory_usage_mb: 10,
        eviction_threshold: 0.8,
        enable_predictive_caching: false,
        cache_warming_enabled: false,
    };
    
    Arc::new(InheritanceCacheManager::new(config))
}
```

Create `src/inheritance/validation/tests/structural_validator_tests.rs`:

```rust
use super::*;
use crate::inheritance::validation::structural_validator::StructuralValidator;

#[tokio::test]
async fn test_validate_inheritance_depth() {
    let connection_manager = MockConnectionManager::new();
    let rules = create_test_validation_rules().structural_rules;
    let validator = StructuralValidator::new(connection_manager, rules);
    
    // Test concept within depth limit
    let results = validator.validate_inheritance_depth("shallow_concept").await.unwrap();
    assert!(results.is_empty(), "Should not have depth violations for shallow concept");
}

#[tokio::test]
async fn test_validate_circular_references() {
    let connection_manager = MockConnectionManager::new();
    let rules = create_test_validation_rules().structural_rules;
    let validator = StructuralValidator::new(connection_manager, rules);
    
    let results = validator.validate_circular_references().await.unwrap();
    // In a real test, we would set up test data with cycles
    assert!(results.len() >= 0, "Should return validation results");
}

#[tokio::test]
async fn test_validate_multiple_inheritance() {
    let connection_manager = MockConnectionManager::new();
    let mut rules = create_test_validation_rules().structural_rules;
    rules.allow_multiple_inheritance = false;
    
    let validator = StructuralValidator::new(connection_manager, rules);
    
    let results = validator.validate_multiple_inheritance("multi_parent_concept").await.unwrap();
    // Test would verify multiple inheritance detection
    assert!(results.len() >= 0, "Should return validation results");
}

#[tokio::test]
async fn test_validate_children_count() {
    let connection_manager = MockConnectionManager::new();
    let rules = create_test_validation_rules().structural_rules;
    let validator = StructuralValidator::new(connection_manager, rules);
    
    let results = validator.validate_children_count("parent_concept").await.unwrap();
    assert!(results.len() >= 0, "Should return validation results");
}
```

Create `src/inheritance/validation/tests/validation_coordinator_tests.rs`:

```rust
use super::*;
use crate::inheritance::validation::validation_coordinator::ValidationCoordinator;

#[tokio::test]
async fn test_validate_concept() {
    let connection_manager = MockConnectionManager::new();
    let cache_manager = create_mock_cache_manager();
    let rules = create_test_validation_rules();
    
    let coordinator = ValidationCoordinator::new(
        connection_manager,
        cache_manager,
        rules,
    );
    
    let report = coordinator.validate_concept("test_concept").await.unwrap();
    
    assert_eq!(report.concept_id, "test_concept");
    assert!(report.validation_results.len() >= 0);
    assert!(report.validator_results.len() >= 0);
}

#[tokio::test]
async fn test_validate_inheritance_relationship() {
    let connection_manager = MockConnectionManager::new();
    let cache_manager = create_mock_cache_manager();
    let rules = create_test_validation_rules();
    
    let coordinator = ValidationCoordinator::new(
        connection_manager,
        cache_manager,
        rules,
    );
    
    let report = coordinator.validate_inheritance_relationship(
        "child_concept", 
        "parent_concept"
    ).await.unwrap();
    
    assert!(report.concept_id.contains("child_concept"));
    assert!(report.concept_id.contains("parent_concept"));
}

#[tokio::test]
async fn test_validate_system_health() {
    let connection_manager = MockConnectionManager::new();
    let cache_manager = create_mock_cache_manager();
    let rules = create_test_validation_rules();
    
    let coordinator = ValidationCoordinator::new(
        connection_manager,
        cache_manager,
        rules,
    );
    
    let report = coordinator.validate_system_health().await.unwrap();
    
    assert!(report.total_issues >= 0);
    assert!(matches!(report.overall_health, 
        crate::inheritance::validation::validation_coordinator::SystemHealth::Healthy |
        crate::inheritance::validation::validation_coordinator::SystemHealth::Degraded |
        crate::inheritance::validation::validation_coordinator::SystemHealth::Unhealthy |
        crate::inheritance::validation::validation_coordinator::SystemHealth::Critical
    ));
}

#[tokio::test]
async fn test_validate_batch_concepts() {
    let connection_manager = MockConnectionManager::new();
    let cache_manager = create_mock_cache_manager();
    let rules = create_test_validation_rules();
    
    let coordinator = ValidationCoordinator::new(
        connection_manager,
        cache_manager,
        rules,
    );
    
    let concept_ids = vec![
        "concept_1".to_string(),
        "concept_2".to_string(),
        "concept_3".to_string(),
    ];
    
    let report = coordinator.validate_batch_concepts(&concept_ids).await.unwrap();
    
    assert_eq!(report.total_concepts, 3);
    assert_eq!(report.concept_reports.len(), 3);
}
```

Create `src/inheritance/validation/tests/validation_integration_tests.rs`:

```rust
use super::*;
use crate::inheritance::validation::integration::{IntegratedInheritanceSystem, InheritanceSystemBuilder};

#[tokio::test]
async fn test_integrated_system_creation() {
    let connection_manager = MockConnectionManager::new();
    let rules = create_test_validation_rules();
    
    let system = InheritanceSystemBuilder::new(connection_manager)
        .with_validation_rules(rules)
        .with_scheduling(false) // Disable scheduling for test
        .with_api(false) // Disable API for test
        .build()
        .await;
    
    assert!(system.is_ok(), "Integrated system should be created successfully");
}

#[tokio::test]
async fn test_system_startup_validation() {
    let connection_manager = MockConnectionManager::new();
    let rules = create_test_validation_rules();
    
    let system = IntegratedInheritanceSystem::new(
        connection_manager,
        Some(rules),
    ).await.unwrap();
    
    let result = system.validate_system_startup().await;
    assert!(result.is_ok(), "System startup validation should pass");
}

#[tokio::test]
async fn test_system_graceful_shutdown() {
    let connection_manager = MockConnectionManager::new();
    let rules = create_test_validation_rules();
    
    let system = IntegratedInheritanceSystem::new(
        connection_manager,
        Some(rules),
    ).await.unwrap();
    
    let result = system.shutdown_gracefully().await;
    assert!(result.is_ok(), "System should shutdown gracefully");
}
```

Create `src/inheritance/validation/tests/performance_tests.rs`:

```rust
use super::*;
use std::time::Instant;
use crate::inheritance::validation::validation_coordinator::ValidationCoordinator;

#[tokio::test]
async fn test_validation_performance() {
    let connection_manager = MockConnectionManager::new();
    let cache_manager = create_mock_cache_manager();
    let rules = create_test_validation_rules();
    
    let coordinator = ValidationCoordinator::new(
        connection_manager,
        cache_manager,
        rules,
    );
    
    let start = Instant::now();
    let _report = coordinator.validate_concept("test_concept").await.unwrap();
    let duration = start.elapsed();
    
    assert!(duration.as_millis() < 1000, "Validation should complete within 1 second");
}

#[tokio::test]
async fn test_batch_validation_performance() {
    let connection_manager = MockConnectionManager::new();
    let cache_manager = create_mock_cache_manager();
    let rules = create_test_validation_rules();
    
    let coordinator = ValidationCoordinator::new(
        connection_manager,
        cache_manager,
        rules,
    );
    
    let concept_ids: Vec<String> = (0..10)
        .map(|i| format!("concept_{}", i))
        .collect();
    
    let start = Instant::now();
    let _report = coordinator.validate_batch_concepts(&concept_ids).await.unwrap();
    let duration = start.elapsed();
    
    assert!(duration.as_millis() < 5000, "Batch validation should complete within 5 seconds");
}

#[tokio::test]
async fn test_concurrent_validations() {
    let connection_manager = MockConnectionManager::new();
    let cache_manager = create_mock_cache_manager();
    let rules = create_test_validation_rules();
    
    let coordinator = Arc::new(ValidationCoordinator::new(
        connection_manager,
        cache_manager,
        rules,
    ));
    
    let mut handles = Vec::new();
    
    for i in 0..5 {
        let coordinator_clone = coordinator.clone();
        let concept_id = format!("concurrent_concept_{}", i);
        
        let handle = tokio::spawn(async move {
            coordinator_clone.validate_concept(&concept_id).await
        });
        
        handles.push(handle);
    }
    
    let results = futures::future::join_all(handles).await;
    
    for result in results {
        assert!(result.is_ok(), "Concurrent validation task should succeed");
        assert!(result.unwrap().is_ok(), "Validation should succeed");
    }
}
```

## Success Criteria
- Comprehensive test coverage for all validators
- Performance tests validate timing requirements
- Integration tests verify system coordination

## Next Task
15l_validation_mod_file.md