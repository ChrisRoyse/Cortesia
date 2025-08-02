# Task 15: Inheritance Validation and Consistency
**Estimated Time**: 10-15 minutes
**Dependencies**: 14_exception_handling.md
**Stage**: Inheritance System

## Objective
Implement comprehensive validation and consistency checking for the inheritance system to ensure data integrity, detect anomalies, and maintain system coherence across inheritance operations.

## Specific Requirements

### 1. Structural Validation
- Inheritance hierarchy consistency checking
- Cycle detection and prevention
- Depth limit validation
- Orphaned concept detection

### 2. Semantic Validation
- Property type consistency across inheritance chains
- Exception validity and coherence
- Inheritance relationship validity
- Context condition consistency

### 3. Performance Validation
- Inheritance performance monitoring
- Cache consistency verification
- Resolution time anomaly detection
- Memory usage validation

## Implementation Steps

### 1. Create Validation Framework
```rust
// src/inheritance/validation/validation_types.rs
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    pub validation_id: String,
    pub validation_timestamp: DateTime<Utc>,
    pub validation_scope: ValidationScope,
    pub overall_status: ValidationStatus,
    pub structural_issues: Vec<StructuralIssue>,
    pub semantic_issues: Vec<SemanticIssue>,
    pub performance_issues: Vec<PerformanceIssue>,
    pub recommendations: Vec<ValidationRecommendation>,
    pub validation_duration: std::time::Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationScope {
    FullSystem,
    ConceptSubtree(String),
    InheritanceChain(String),
    PropertyValidation(String, String),
    ExceptionValidation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationStatus {
    Passed,
    PassedWithWarnings,
    Failed,
    PartialFailure,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralIssue {
    pub issue_id: String,
    pub issue_type: StructuralIssueType,
    pub severity: IssueSeverity,
    pub affected_concepts: Vec<String>,
    pub description: String,
    pub auto_fixable: bool,
    pub fix_recommendation: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StructuralIssueType {
    CycleDetected,
    MaxDepthExceeded,
    OrphanedConcept,
    InvalidRelationship,
    MissingParent,
    DuplicateRelationship,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticIssue {
    pub issue_id: String,
    pub issue_type: SemanticIssueType,
    pub severity: IssueSeverity,
    pub concept_id: String,
    pub property_name: Option<String>,
    pub description: String,
    pub context: String,
    pub suggested_resolution: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SemanticIssueType {
    PropertyTypeInconsistency,
    InvalidExceptionValue,
    ConflictingExceptions,
    MissingInheritedProperty,
    InvalidContextCondition,
    InconsistentPrecedence,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceIssue {
    pub issue_id: String,
    pub issue_type: PerformanceIssueType,
    pub severity: IssueSeverity,
    pub metric_name: String,
    pub current_value: f64,
    pub threshold_value: f64,
    pub affected_operations: Vec<String>,
    pub optimization_suggestions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IssueSeverity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}
```

### 2. Implement Core Validator
```rust
// src/inheritance/validation/inheritance_validator.rs
pub struct InheritanceValidator {
    hierarchy_manager: Arc<InheritanceHierarchyManager>,
    property_engine: Arc<PropertyInheritanceEngine>,
    exception_manager: Arc<ExceptionManager>,
    cache_manager: Arc<InheritanceCacheManager>,
    validation_config: ValidationConfig,
    performance_monitor: Arc<ValidationPerformanceMonitor>,
}

impl InheritanceValidator {
    pub async fn new(
        hierarchy_manager: Arc<InheritanceHierarchyManager>,
        property_engine: Arc<PropertyInheritanceEngine>,
        exception_manager: Arc<ExceptionManager>,
        cache_manager: Arc<InheritanceCacheManager>,
        validation_config: ValidationConfig,
    ) -> Self {
        Self {
            hierarchy_manager,
            property_engine,
            exception_manager,
            cache_manager,
            validation_config,
            performance_monitor: Arc::new(ValidationPerformanceMonitor::new()),
        }
    }
    
    pub async fn validate_full_system(&self) -> Result<ValidationReport, ValidationError> {
        let validation_start = Instant::now();
        let validation_id = Uuid::new_v4().to_string();
        
        info!("Starting full system inheritance validation: {}", validation_id);
        
        // Perform structural validation
        let structural_issues = self.validate_structural_integrity().await?;
        
        // Perform semantic validation
        let semantic_issues = self.validate_semantic_consistency().await?;
        
        // Perform performance validation
        let performance_issues = self.validate_performance_characteristics().await?;
        
        // Generate recommendations
        let recommendations = self.generate_recommendations(
            &structural_issues,
            &semantic_issues,
            &performance_issues,
        ).await?;
        
        // Determine overall status
        let overall_status = self.determine_overall_status(
            &structural_issues,
            &semantic_issues,
            &performance_issues,
        );
        
        let validation_duration = validation_start.elapsed();
        
        let report = ValidationReport {
            validation_id,
            validation_timestamp: Utc::now(),
            validation_scope: ValidationScope::FullSystem,
            overall_status,
            structural_issues,
            semantic_issues,
            performance_issues,
            recommendations,
            validation_duration,
        };
        
        // Record validation metrics
        self.performance_monitor.record_validation_completion(&report).await;
        
        Ok(report)
    }
    
    pub async fn validate_concept_inheritance(
        &self,
        concept_id: &str,
    ) -> Result<ValidationReport, ValidationError> {
        let validation_start = Instant::now();
        let validation_id = format!("concept_{}_{}", concept_id, Uuid::new_v4().simple());
        
        // Validate concept's inheritance chain
        let chain_issues = self.validate_inheritance_chain(concept_id).await?;
        
        // Validate concept's properties
        let property_issues = self.validate_concept_properties(concept_id).await?;
        
        // Validate concept's exceptions
        let exception_issues = self.validate_concept_exceptions(concept_id).await?;
        
        // Combine issues
        let mut structural_issues = chain_issues;
        let mut semantic_issues = property_issues;
        semantic_issues.extend(exception_issues);
        
        // Generate targeted recommendations
        let recommendations = self.generate_concept_recommendations(
            concept_id,
            &structural_issues,
            &semantic_issues,
        ).await?;
        
        let overall_status = self.determine_overall_status(
            &structural_issues,
            &semantic_issues,
            &Vec::new(), // No performance issues for single concept
        );
        
        Ok(ValidationReport {
            validation_id,
            validation_timestamp: Utc::now(),
            validation_scope: ValidationScope::ConceptSubtree(concept_id.to_string()),
            overall_status,
            structural_issues,
            semantic_issues,
            performance_issues: Vec::new(),
            recommendations,
            validation_duration: validation_start.elapsed(),
        })
    }
    
    async fn validate_structural_integrity(&self) -> Result<Vec<StructuralIssue>, ValidationError> {
        let mut issues = Vec::new();
        
        // Check for cycles
        let cycle_issues = self.detect_inheritance_cycles().await?;
        issues.extend(cycle_issues);
        
        // Check depth limits
        let depth_issues = self.validate_inheritance_depths().await?;
        issues.extend(depth_issues);
        
        // Check for orphaned concepts
        let orphan_issues = self.detect_orphaned_concepts().await?;
        issues.extend(orphan_issues);
        
        // Validate relationship integrity
        let relationship_issues = self.validate_relationship_integrity().await?;
        issues.extend(relationship_issues);
        
        Ok(issues)
    }
    
    async fn detect_inheritance_cycles(&self) -> Result<Vec<StructuralIssue>, CycleDetectionError> {
        let session = self.hierarchy_manager.connection_manager.get_session().await?;
        
        // Query to detect cycles using graph algorithms
        let query = r#"
            MATCH path = (c:Concept)-[:INHERITS_FROM*1..50]->(c)
            WHERE ALL(r IN relationships(path) WHERE r.is_active = true)
            WITH c, path, length(path) as cycle_length
            RETURN c.id as concept_id, 
                   [n IN nodes(path) | n.id] as cycle_path,
                   cycle_length
            ORDER BY cycle_length
        "#;
        
        let result = session.run(query, None).await?;
        
        let mut cycle_issues = Vec::new();
        for record in result {
            let concept_id: String = record.get("concept_id")?;
            let cycle_path: Vec<String> = record.get("cycle_path")?;
            let cycle_length: i64 = record.get("cycle_length")?;
            
            let issue = StructuralIssue {
                issue_id: format!("cycle_{}", concept_id),
                issue_type: StructuralIssueType::CycleDetected,
                severity: IssueSeverity::Critical,
                affected_concepts: cycle_path,
                description: format!(
                    "Inheritance cycle detected starting from concept '{}' with length {}",
                    concept_id, cycle_length
                ),
                auto_fixable: false,
                fix_recommendation: Some(
                    "Remove one of the inheritance relationships in the cycle to break it".to_string()
                ),
            };
            
            cycle_issues.push(issue);
        }
        
        Ok(cycle_issues)
    }
    
    async fn validate_inheritance_depths(&self) -> Result<Vec<StructuralIssue>, DepthValidationError> {
        let session = self.hierarchy_manager.connection_manager.get_session().await?;
        
        // Find concepts exceeding maximum depth
        let query = r#"
            MATCH path = (c:Concept)-[:INHERITS_FROM*]->(root:Concept)
            WHERE NOT (root)-[:INHERITS_FROM]->() AND length(path) > $max_depth
            WITH c, path, length(path) as depth
            ORDER BY depth DESC
            RETURN c.id as concept_id, depth
            LIMIT 100
        "#;
        
        let parameters = hashmap![
            "max_depth".to_string() => (self.validation_config.max_inheritance_depth as i64).into(),
        ];
        
        let result = session.run(query, Some(parameters)).await?;
        
        let mut depth_issues = Vec::new();
        for record in result {
            let concept_id: String = record.get("concept_id")?;
            let depth: i64 = record.get("depth")?;
            
            let issue = StructuralIssue {
                issue_id: format!("depth_{}", concept_id),
                issue_type: StructuralIssueType::MaxDepthExceeded,
                severity: if depth > self.validation_config.max_inheritance_depth + 5 {
                    IssueSeverity::High
                } else {
                    IssueSeverity::Medium
                },
                affected_concepts: vec![concept_id.clone()],
                description: format!(
                    "Concept '{}' has inheritance depth {} which exceeds maximum allowed depth {}",
                    concept_id, depth, self.validation_config.max_inheritance_depth
                ),
                auto_fixable: false,
                fix_recommendation: Some(
                    "Consider flattening the inheritance hierarchy or restructuring relationships".to_string()
                ),
            };
            
            depth_issues.push(issue);
        }
        
        Ok(depth_issues)
    }
    
    async fn validate_semantic_consistency(&self) -> Result<Vec<SemanticIssue>, ValidationError> {
        let mut issues = Vec::new();
        
        // Validate property type consistency
        let type_issues = self.validate_property_type_consistency().await?;
        issues.extend(type_issues);
        
        // Validate exception coherence
        let exception_issues = self.validate_exception_coherence().await?;
        issues.extend(exception_issues);
        
        // Validate context conditions
        let context_issues = self.validate_context_conditions().await?;
        issues.extend(context_issues);
        
        Ok(issues)
    }
    
    async fn validate_property_type_consistency(&self) -> Result<Vec<SemanticIssue>, ValidationError> {
        let session = self.hierarchy_manager.connection_manager.get_session().await?;
        
        // Find properties with inconsistent types across inheritance chains
        let query = r#"
            MATCH (child:Concept)-[:INHERITS_FROM*]->(parent:Concept)
            MATCH (child)-[:HAS_PROPERTY]->(child_prop:Property)
            MATCH (parent)-[:HAS_PROPERTY]->(parent_prop:Property)
            WHERE child_prop.name = parent_prop.name 
              AND child_prop.type <> parent_prop.type
              AND child_prop.allow_type_override = false
            RETURN child.id as child_id,
                   parent.id as parent_id,
                   child_prop.name as property_name,
                   child_prop.type as child_type,
                   parent_prop.type as parent_type
        "#;
        
        let result = session.run(query, None).await?;
        
        let mut type_issues = Vec::new();
        for record in result {
            let child_id: String = record.get("child_id")?;
            let parent_id: String = record.get("parent_id")?;
            let property_name: String = record.get("property_name")?;
            let child_type: String = record.get("child_type")?;
            let parent_type: String = record.get("parent_type")?;
            
            let issue = SemanticIssue {
                issue_id: format!("type_inconsistency_{}_{}", child_id, property_name),
                issue_type: SemanticIssueType::PropertyTypeInconsistency,
                severity: IssueSeverity::High,
                concept_id: child_id.clone(),
                property_name: Some(property_name.clone()),
                description: format!(
                    "Property '{}' has type '{}' in concept '{}' but type '{}' in parent '{}'",
                    property_name, child_type, child_id, parent_type, parent_id
                ),
                context: format!("Inheritance chain: {} -> {}", child_id, parent_id),
                suggested_resolution: Some(
                    "Ensure property types are compatible or enable type override".to_string()
                ),
            };
            
            type_issues.push(issue);
        }
        
        Ok(type_issues)
    }
    
    pub async fn auto_fix_issues(
        &self,
        validation_report: &ValidationReport,
    ) -> Result<FixResult, AutoFixError> {
        let mut fixed_issues = Vec::new();
        let mut failed_fixes = Vec::new();
        
        // Attempt to fix auto-fixable structural issues
        for issue in &validation_report.structural_issues {
            if issue.auto_fixable {
                match self.fix_structural_issue(issue).await {
                    Ok(_) => fixed_issues.push(issue.issue_id.clone()),
                    Err(e) => failed_fixes.push((issue.issue_id.clone(), e.to_string())),
                }
            }
        }
        
        // Attempt to fix auto-fixable semantic issues
        for issue in &validation_report.semantic_issues {
            if let Some(resolution) = &issue.suggested_resolution {
                match self.fix_semantic_issue(issue, resolution).await {
                    Ok(_) => fixed_issues.push(issue.issue_id.clone()),
                    Err(e) => failed_fixes.push((issue.issue_id.clone(), e.to_string())),
                }
            }
        }
        
        Ok(FixResult {
            fixed_issues,
            failed_fixes,
            total_attempted: fixed_issues.len() + failed_fixes.len(),
        })
    }
}
```

### 3. Implement Continuous Validation
```rust
// src/inheritance/validation/continuous_validator.rs
pub struct ContinuousValidator {
    validator: Arc<InheritanceValidator>,
    validation_scheduler: ValidationScheduler,
    issue_tracker: Arc<RwLock<IssueTracker>>,
    alert_manager: AlertManager,
    config: ContinuousValidationConfig,
}

impl ContinuousValidator {
    pub async fn start_continuous_validation(&self) -> Result<(), ValidationError> {
        info!("Starting continuous inheritance validation");
        
        // Schedule regular full system validations
        self.schedule_system_validations().await?;
        
        // Set up real-time validation triggers
        self.setup_realtime_triggers().await?;
        
        // Start validation monitoring
        self.start_validation_monitoring().await?;
        
        Ok(())
    }
    
    async fn schedule_system_validations(&self) -> Result<(), SchedulingError> {
        // Schedule full system validation every hour
        let validator = self.validator.clone();
        let issue_tracker = self.issue_tracker.clone();
        let alert_manager = self.alert_manager.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(3600)); // 1 hour
            
            loop {
                interval.tick().await;
                
                match validator.validate_full_system().await {
                    Ok(report) => {
                        let critical_issues = report.structural_issues
                            .iter()
                            .chain(report.semantic_issues.iter())
                            .filter(|issue| matches!(issue.severity, IssueSeverity::Critical))
                            .count();
                        
                        if critical_issues > 0 {
                            alert_manager.send_critical_alert(
                                format!("Found {} critical inheritance issues", critical_issues)
                            ).await;
                        }
                        
                        issue_tracker.write().await.update_issues(&report);
                    },
                    Err(e) => {
                        error!("Continuous validation failed: {}", e);
                        alert_manager.send_error_alert(
                            format!("Validation system error: {}", e)
                        ).await;
                    }
                }
            }
        });
        
        Ok(())
    }
    
    pub async fn validate_on_change(
        &self,
        change_event: InheritanceChangeEvent,
    ) -> Result<(), ValidationError> {
        // Perform targeted validation based on change type
        let validation_scope = match change_event.change_type {
            ChangeType::RelationshipCreated => {
                ValidationScope::ConceptSubtree(change_event.affected_concept_id.clone())
            },
            ChangeType::PropertyAdded | ChangeType::PropertyModified => {
                ValidationScope::PropertyValidation(
                    change_event.affected_concept_id.clone(),
                    change_event.property_name.unwrap_or_default(),
                )
            },
            ChangeType::ExceptionCreated | ChangeType::ExceptionModified => {
                ValidationScope::ExceptionValidation
            },
        };
        
        // Run targeted validation
        let report = match validation_scope {
            ValidationScope::ConceptSubtree(concept_id) => {
                self.validator.validate_concept_inheritance(&concept_id).await?
            },
            _ => {
                // For other scopes, run a quick system check
                self.validator.validate_full_system().await?
            }
        };
        
        // Check for critical issues
        let has_critical_issues = report.structural_issues
            .iter()
            .chain(report.semantic_issues.iter())
            .any(|issue| matches!(issue.severity, IssueSeverity::Critical));
        
        if has_critical_issues {
            self.alert_manager.send_immediate_alert(
                format!("Critical inheritance issue detected after change: {:?}", change_event)
            ).await;
        }
        
        // Update issue tracker
        self.issue_tracker.write().await.update_issues(&report);
        
        Ok(())
    }
}
```

## Acceptance Criteria

### Functional Requirements
- [ ] Full system validation detects structural issues accurately
- [ ] Semantic validation identifies property type inconsistencies
- [ ] Performance validation monitors system health
- [ ] Auto-fix capabilities work for simple issues
- [ ] Continuous validation provides real-time monitoring

### Performance Requirements
- [ ] Full system validation completes within 30 seconds
- [ ] Concept validation completes within 500ms
- [ ] Change-triggered validation completes within 2 seconds
- [ ] Memory usage for validation < 100MB

### Testing Requirements
- [ ] Unit tests for validation logic
- [ ] Integration tests for validation workflows
- [ ] Performance tests for large inheritance hierarchies
- [ ] Auto-fix functionality tests

## Validation Steps

1. **Test structural validation**:
   ```rust
   let report = validator.validate_full_system().await?;
   assert_eq!(report.overall_status, ValidationStatus::Passed);
   ```

2. **Test auto-fix capabilities**:
   ```rust
   let fix_result = validator.auto_fix_issues(&validation_report).await?;
   assert!(fix_result.fixed_issues.len() > 0);
   ```

3. **Run validation tests**:
   ```bash
   cargo test inheritance_validation_tests
   ```

## Files to Create/Modify
- `src/inheritance/validation/validation_types.rs` - Validation data structures
- `src/inheritance/validation/inheritance_validator.rs` - Core validator
- `src/inheritance/validation/continuous_validator.rs` - Continuous validation
- `src/inheritance/validation/mod.rs` - Module exports
- `tests/inheritance/validation_tests.rs` - Validation test suite

## Success Metrics
- Validation accuracy: 100% for known issues
- False positive rate < 5%
- Auto-fix success rate > 80% for fixable issues
- Validation performance within targets

## Next Task
Upon completion, the Inheritance System stage is complete. Proceed to the next stage in the Phase 3 execution plan.