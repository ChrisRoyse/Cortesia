# Task 15f: Create Validation Coordinator

**Time**: 8 minutes (1.5 min read, 5 min implement, 1.5 min verify)
**Dependencies**: 15e_custom_rule_engine.md
**Stage**: Inheritance System

## Objective
Create central coordinator that orchestrates all validation activities.

## Implementation
Create `src/inheritance/validation/validation_coordinator.rs`:

```rust
use std::sync::Arc;
use std::collections::HashMap;
use tokio::task::JoinSet;
use crate::core::neo4j_connection::Neo4jConnectionManager;
use crate::inheritance::cache::InheritanceCacheManager;
use crate::inheritance::validation::rules::*;
use crate::inheritance::validation::structural_validator::StructuralValidator;
use crate::inheritance::validation::semantic_validator::SemanticValidator;
use crate::inheritance::validation::performance_validator::PerformanceValidator;
use crate::inheritance::validation::custom_rule_engine::{CustomRuleEngine, ValidationContext};

pub struct ValidationCoordinator {
    structural_validator: Arc<StructuralValidator>,
    semantic_validator: Arc<SemanticValidator>,
    performance_validator: Arc<PerformanceValidator>,
    custom_rule_engine: Arc<CustomRuleEngine>,
    rules: ValidationRules,
}

impl ValidationCoordinator {
    pub fn new(
        connection_manager: Arc<Neo4jConnectionManager>,
        cache_manager: Arc<InheritanceCacheManager>,
        rules: ValidationRules,
    ) -> Self {
        let structural_validator = Arc::new(StructuralValidator::new(
            connection_manager.clone(),
            rules.structural_rules.clone()
        ));
        
        let semantic_validator = Arc::new(SemanticValidator::new(
            connection_manager.clone(),
            rules.semantic_rules.clone()
        ));
        
        let performance_validator = Arc::new(PerformanceValidator::new(
            cache_manager,
            rules.performance_rules.clone()
        ));
        
        let mut custom_rule_engine = CustomRuleEngine::new();
        for rule in &rules.custom_rules {
            custom_rule_engine.add_rule(rule.clone());
        }
        let custom_rule_engine = Arc::new(custom_rule_engine);
        
        Self {
            structural_validator,
            semantic_validator,
            performance_validator,
            custom_rule_engine,
            rules,
        }
    }

    pub async fn validate_concept(&self, concept_id: &str) -> Result<ValidationReport, Box<dyn std::error::Error>> {
        let mut join_set = JoinSet::new();
        
        // Run all validators in parallel
        let structural_validator = self.structural_validator.clone();
        let concept_id_clone = concept_id.to_string();
        join_set.spawn(async move {
            ("structural", structural_validator.validate_inheritance_depth(&concept_id_clone).await)
        });
        
        let semantic_validator = self.semantic_validator.clone();
        let concept_id_clone = concept_id.to_string();
        join_set.spawn(async move {
            ("semantic", semantic_validator.validate_concept_existence(&concept_id_clone).await)
        });
        
        let performance_validator = self.performance_validator.clone();
        let concept_id_clone = concept_id.to_string();
        join_set.spawn(async move {
            ("performance", performance_validator.validate_resolution_performance(&concept_id_clone).await)
        });
        
        let custom_rule_engine = self.custom_rule_engine.clone();
        let concept_id_clone = concept_id.to_string();
        join_set.spawn(async move {
            let context = ValidationContext {
                concept_id: Some(concept_id_clone),
                property_name: None,
                operation: "concept_validation".to_string(),
                parameters: HashMap::new(),
            };
            ("custom", custom_rule_engine.execute_rules(&context).await)
        });
        
        // Collect all results
        let mut all_results = Vec::new();
        let mut validator_results = HashMap::new();
        
        while let Some(result) = join_set.join_next().await {
            match result {
                Ok((validator_type, validation_result)) => {
                    match validation_result {
                        Ok(mut results) => {
                            validator_results.insert(validator_type.to_string(), results.len());
                            all_results.append(&mut results);
                        }
                        Err(e) => {
                            all_results.push(ValidationResult::new(
                                &format!("{}_error", validator_type),
                                ValidationSeverity::Error,
                                &format!("Validator '{}' failed: {}", validator_type, e)
                            ));
                        }
                    }
                }
                Err(e) => {
                    all_results.push(ValidationResult::new(
                        "coordinator_error",
                        ValidationSeverity::Error,
                        &format!("Validation task failed: {}", e)
                    ));
                }
            }
        }
        
        Ok(ValidationReport::new(concept_id, all_results, validator_results))
    }

    pub async fn validate_inheritance_relationship(
        &self,
        child_id: &str,
        parent_id: &str,
    ) -> Result<ValidationReport, Box<dyn std::error::Error>> {
        let mut all_results = Vec::new();
        let mut validator_results = HashMap::new();
        
        // Validate multiple inheritance
        let mut results = self.structural_validator.validate_multiple_inheritance(child_id).await?;
        validator_results.insert("structural_multiple".to_string(), results.len());
        all_results.append(&mut results);
        
        // Validate inheritance compatibility
        let mut results = self.semantic_validator.validate_inheritance_compatibility(child_id, parent_id).await?;
        validator_results.insert("semantic_compatibility".to_string(), results.len());
        all_results.append(&mut results);
        
        // Validate relationship semantics
        let mut results = self.semantic_validator.validate_relationship_semantics(child_id, parent_id).await?;
        validator_results.insert("semantic_relationship".to_string(), results.len());
        all_results.append(&mut results);
        
        // Custom rules for inheritance
        let context = ValidationContext {
            concept_id: Some(child_id.to_string()),
            property_name: None,
            operation: format!("inheritance_{}_{}", child_id, parent_id),
            parameters: {
                let mut params = HashMap::new();
                params.insert("parent_id".to_string(), parent_id.to_string());
                params
            },
        };
        
        let mut results = self.custom_rule_engine.execute_rules(&context).await?;
        validator_results.insert("custom_inheritance".to_string(), results.len());
        all_results.append(&mut results);
        
        Ok(ValidationReport::new(&format!("{}_{}", child_id, parent_id), all_results, validator_results))
    }

    pub async fn validate_system_health(&self) -> Result<SystemValidationReport, Box<dyn std::error::Error>> {
        let mut all_results = Vec::new();
        
        // Validate circular references
        let mut results = self.structural_validator.validate_circular_references().await?;
        all_results.append(&mut results);
        
        // Validate cache usage
        let mut results = self.performance_validator.validate_cache_usage().await?;
        all_results.append(&mut results);
        
        // Validate concurrent operations
        let mut results = self.performance_validator.validate_concurrent_operations().await?;
        all_results.append(&mut results);
        
        let critical_count = all_results.iter()
            .filter(|r| matches!(r.severity, ValidationSeverity::Critical))
            .count();
        
        let error_count = all_results.iter()
            .filter(|r| matches!(r.severity, ValidationSeverity::Error))
            .count();
        
        let warning_count = all_results.iter()
            .filter(|r| matches!(r.severity, ValidationSeverity::Warning))
            .count();
        
        let overall_health = if critical_count > 0 {
            SystemHealth::Critical
        } else if error_count > 5 {
            SystemHealth::Unhealthy
        } else if warning_count > 10 {
            SystemHealth::Degraded
        } else {
            SystemHealth::Healthy
        };
        
        Ok(SystemValidationReport {
            overall_health,
            total_issues: all_results.len(),
            critical_issues: critical_count,
            error_issues: error_count,
            warning_issues: warning_count,
            validation_results: all_results,
            timestamp: chrono::Utc::now(),
        })
    }

    pub async fn validate_batch_concepts(&self, concept_ids: &[String]) -> Result<BatchValidationReport, Box<dyn std::error::Error>> {
        let mut join_set = JoinSet::new();
        
        for concept_id in concept_ids {
            let coordinator = self.clone();
            let concept_id = concept_id.clone();
            join_set.spawn(async move {
                (concept_id.clone(), coordinator.validate_concept(&concept_id).await)
            });
        }
        
        let mut concept_reports = HashMap::new();
        let mut total_issues = 0;
        
        while let Some(result) = join_set.join_next().await {
            match result {
                Ok((concept_id, validation_result)) => {
                    match validation_result {
                        Ok(report) => {
                            total_issues += report.total_issues();
                            concept_reports.insert(concept_id, report);
                        }
                        Err(e) => {
                            let error_report = ValidationReport::new(
                                &concept_id,
                                vec![ValidationResult::new(
                                    "batch_validation_error",
                                    ValidationSeverity::Error,
                                    &format!("Batch validation failed: {}", e)
                                )],
                                HashMap::new()
                            );
                            concept_reports.insert(concept_id, error_report);
                            total_issues += 1;
                        }
                    }
                }
                Err(e) => {
                    total_issues += 1;
                    eprintln!("Batch validation task failed: {}", e);
                }
            }
        }
        
        Ok(BatchValidationReport {
            concept_reports,
            total_concepts: concept_ids.len(),
            total_issues,
            timestamp: chrono::Utc::now(),
        })
    }
}

impl Clone for ValidationCoordinator {
    fn clone(&self) -> Self {
        Self {
            structural_validator: self.structural_validator.clone(),
            semantic_validator: self.semantic_validator.clone(),
            performance_validator: self.performance_validator.clone(),
            custom_rule_engine: self.custom_rule_engine.clone(),
            rules: self.rules.clone(),
        }
    }
}

#[derive(Debug)]
pub struct ValidationReport {
    pub concept_id: String,
    pub validation_results: Vec<ValidationResult>,
    pub validator_results: HashMap<String, usize>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl ValidationReport {
    pub fn new(concept_id: &str, results: Vec<ValidationResult>, validator_results: HashMap<String, usize>) -> Self {
        Self {
            concept_id: concept_id.to_string(),
            validation_results: results,
            validator_results,
            timestamp: chrono::Utc::now(),
        }
    }

    pub fn total_issues(&self) -> usize {
        self.validation_results.len()
    }

    pub fn has_critical_issues(&self) -> bool {
        self.validation_results.iter()
            .any(|r| matches!(r.severity, ValidationSeverity::Critical))
    }
}

#[derive(Debug)]
pub struct SystemValidationReport {
    pub overall_health: SystemHealth,
    pub total_issues: usize,
    pub critical_issues: usize,
    pub error_issues: usize,
    pub warning_issues: usize,
    pub validation_results: Vec<ValidationResult>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug)]
pub enum SystemHealth {
    Healthy,
    Degraded,
    Unhealthy,
    Critical,
}

#[derive(Debug)]
pub struct BatchValidationReport {
    pub concept_reports: HashMap<String, ValidationReport>,
    pub total_concepts: usize,
    pub total_issues: usize,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}
```

## Success Criteria
- Coordinates all validators effectively
- Runs validations in parallel for performance
- Generates comprehensive reports

## Next Task
15g_validation_scheduler.md