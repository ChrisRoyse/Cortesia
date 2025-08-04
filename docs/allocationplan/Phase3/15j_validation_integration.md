# Task 15j: Integrate Validation with Inheritance System

**Time**: 7 minutes (1.5 min read, 4 min implement, 1.5 min verify)
**Dependencies**: 15i_validation_api.md
**Stage**: Inheritance System

## Objective
Integrate validation system with existing inheritance components.

## Implementation
Modify existing inheritance components to include validation:

Update `src/inheritance/hierarchy_manager.rs`:

```rust
use crate::inheritance::validation::validation_coordinator::ValidationCoordinator;

// Add validation to the hierarchy manager
impl InheritanceHierarchyManager {
    pub async fn new_with_validation(
        connection_manager: Arc<Neo4jConnectionManager>,
        validation_coordinator: Arc<ValidationCoordinator>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut manager = Self::new(connection_manager).await?;
        manager.validation_coordinator = Some(validation_coordinator);
        Ok(manager)
    }

    // Update create_inheritance_relationship to include validation
    pub async fn create_inheritance_relationship(
        &self,
        parent_concept_id: &str,
        child_concept_id: &str,
        inheritance_type: InheritanceType,
        inheritance_weight: f32,
    ) -> Result<String, Box<dyn std::error::Error>> {
        // Validate before creating relationship
        if let Some(validator) = &self.validation_coordinator {
            let validation_report = validator.validate_inheritance_relationship(
                child_concept_id, 
                parent_concept_id
            ).await?;
            
            // Check for critical issues
            if validation_report.has_critical_issues() {
                return Err(format!("Cannot create inheritance relationship due to critical validation issues").into());
            }
        }
        
        // Proceed with original logic if validation passes
        // ... existing implementation ...
        Ok("relationship_id".to_string()) // Placeholder
    }
}

// Add validation coordinator field
pub struct InheritanceHierarchyManager {
    connection_manager: Arc<Neo4jConnectionManager>,
    cache_manager: Arc<InheritanceCacheManager>,
    validation_coordinator: Option<Arc<ValidationCoordinator>>,
}
```

Update `src/inheritance/property_inheritance_engine.rs`:

```rust
// Add validation to property inheritance engine
impl PropertyInheritanceEngine {
    pub async fn resolve_properties_with_validation(
        &self,
        concept_id: &str,
        include_inherited: bool,
    ) -> Result<ResolvedProperties, Box<dyn std::error::Error>> {
        // Validate concept before resolution
        if let Some(validator) = &self.validation_coordinator {
            let validation_report = validator.validate_concept(concept_id).await?;
            
            // Log warnings but don't fail for non-critical issues
            for result in &validation_report.validation_results {
                match result.severity {
                    crate::inheritance::validation::rules::ValidationSeverity::Critical => {
                        return Err(format!("Critical validation issue: {}", result.message).into());
                    }
                    crate::inheritance::validation::rules::ValidationSeverity::Warning => {
                        tracing::warn!(
                            concept_id = %concept_id,
                            rule_id = %result.rule_id,
                            message = %result.message,
                            "Validation warning during property resolution"
                        );
                    }
                    _ => {}
                }
            }
        }
        
        // Proceed with normal property resolution
        self.resolve_properties(concept_id, include_inherited).await
    }

    // Add validation coordinator field
    validation_coordinator: Option<Arc<ValidationCoordinator>>,
}
```

Create integration factory in `src/inheritance/validation/integration.rs`:

```rust
use std::sync::Arc;
use crate::core::neo4j_connection::Neo4jConnectionManager;
use crate::inheritance::cache::InheritanceCacheManager;
use crate::inheritance::validation::validation_coordinator::ValidationCoordinator;
use crate::inheritance::validation::validation_scheduler::{ValidationScheduler, SchedulerConfig};
use crate::inheritance::validation::validation_reporting::ValidationReportGenerator;
use crate::inheritance::validation::validation_api::ValidationApiState;
use crate::inheritance::validation::rules::ValidationRules;

pub struct IntegratedInheritanceSystem {
    pub hierarchy_manager: Arc<crate::inheritance::hierarchy_manager::InheritanceHierarchyManager>,
    pub property_engine: Arc<crate::inheritance::property_inheritance_engine::PropertyInheritanceEngine>,
    pub cache_manager: Arc<InheritanceCacheManager>,
    pub validation_coordinator: Arc<ValidationCoordinator>,
    pub validation_scheduler: Arc<ValidationScheduler>,
    pub validation_api_state: ValidationApiState,
}

impl IntegratedInheritanceSystem {
    pub async fn new(
        connection_manager: Arc<Neo4jConnectionManager>,
        validation_rules: Option<ValidationRules>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Create cache manager
        let cache_config = crate::inheritance::cache::CacheConfig::default();
        let cache_manager = Arc::new(InheritanceCacheManager::new(cache_config));
        
        // Create validation coordinator
        let rules = validation_rules.unwrap_or_default();
        let validation_coordinator = Arc::new(ValidationCoordinator::new(
            connection_manager.clone(),
            cache_manager.clone(),
            rules,
        ));
        
        // Create hierarchy manager with validation
        let hierarchy_manager = Arc::new(
            crate::inheritance::hierarchy_manager::InheritanceHierarchyManager::new_with_validation(
                connection_manager.clone(),
                validation_coordinator.clone(),
            ).await?
        );
        
        // Create property inheritance engine with validation
        let inheritance_config = crate::inheritance::property_inheritance_engine::InheritanceConfig {
            max_inheritance_depth: 20,
            cache_ttl_minutes: 30,
            enable_property_exceptions: true,
        };
        
        let property_engine = Arc::new(
            crate::inheritance::property_inheritance_engine::PropertyInheritanceEngine::new_with_validation(
                connection_manager.clone(),
                inheritance_config,
                validation_coordinator.clone(),
            ).await?
        );
        
        // Create validation scheduler
        let scheduler_config = SchedulerConfig::default();
        let (validation_scheduler, _report_receiver) = ValidationScheduler::new(
            validation_coordinator.clone(),
            scheduler_config,
        );
        let validation_scheduler = Arc::new(validation_scheduler);
        
        // Create report generator
        let report_generator = Arc::new(tokio::sync::Mutex::new(ValidationReportGenerator::new()));
        
        // Create API state
        let validation_api_state = ValidationApiState {
            coordinator: validation_coordinator.clone(),
            scheduler: validation_scheduler.clone(),
            report_generator,
        };
        
        Ok(Self {
            hierarchy_manager,
            property_engine,
            cache_manager,
            validation_coordinator,
            validation_scheduler,
            validation_api_state,
        })
    }

    pub async fn start_validation_scheduling(&self) -> Vec<tokio::task::JoinHandle<()>> {
        self.validation_scheduler.start()
    }

    pub async fn validate_system_startup(&self) -> Result<(), Box<dyn std::error::Error>> {
        tracing::info!("Starting system validation check...");
        
        let validation_report = self.validation_coordinator.validate_system_health().await?;
        
        if validation_report.critical_issues > 0 {
            tracing::error!(
                critical_issues = validation_report.critical_issues,
                "Critical validation issues detected during startup"
            );
            return Err("System startup validation failed due to critical issues".into());
        }
        
        if validation_report.error_issues > 0 {
            tracing::warn!(
                error_issues = validation_report.error_issues,
                "Error-level validation issues detected during startup"
            );
        }
        
        tracing::info!(
            total_issues = validation_report.total_issues,
            warning_issues = validation_report.warning_issues,
            "System validation completed"
        );
        
        Ok(())
    }

    pub fn get_api_router(&self) -> axum::Router {
        crate::inheritance::validation::validation_api::create_validation_router(
            self.validation_api_state.clone()
        )
    }

    pub async fn shutdown_gracefully(&self) -> Result<(), Box<dyn std::error::Error>> {
        tracing::info!("Shutting down inheritance system...");
        
        // Run final validation before shutdown
        let final_report = self.validation_coordinator.validate_system_health().await?;
        
        tracing::info!(
            total_issues = final_report.total_issues,
            "Final system validation completed during shutdown"
        );
        
        Ok(())
    }
}

// Configuration builder for easier setup
pub struct InheritanceSystemBuilder {
    connection_manager: Arc<Neo4jConnectionManager>,
    validation_rules: Option<ValidationRules>,
    enable_scheduling: bool,
    enable_api: bool,
}

impl InheritanceSystemBuilder {
    pub fn new(connection_manager: Arc<Neo4jConnectionManager>) -> Self {
        Self {
            connection_manager,
            validation_rules: None,
            enable_scheduling: true,
            enable_api: true,
        }
    }

    pub fn with_validation_rules(mut self, rules: ValidationRules) -> Self {
        self.validation_rules = Some(rules);
        self
    }

    pub fn with_scheduling(mut self, enable: bool) -> Self {
        self.enable_scheduling = enable;
        self
    }

    pub fn with_api(mut self, enable: bool) -> Self {
        self.enable_api = enable;
        self
    }

    pub async fn build(self) -> Result<IntegratedInheritanceSystem, Box<dyn std::error::Error>> {
        let mut system = IntegratedInheritanceSystem::new(
            self.connection_manager,
            self.validation_rules,
        ).await?;
        
        if self.enable_scheduling {
            let _handles = system.start_validation_scheduling().await;
            // Store handles for cleanup if needed
        }
        
        // Validate system after setup
        system.validate_system_startup().await?;
        
        Ok(system)
    }
}
```

## Success Criteria
- Validation is integrated into inheritance operations
- System startup validation works
- Builder pattern simplifies configuration

## Next Task
15k_validation_testing.md