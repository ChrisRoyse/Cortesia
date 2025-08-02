# Task 24: Conflict Detection and Resolution System
**Estimated Time**: 15-20 minutes
**Dependencies**: 23_spreading_activation.md
**Stage**: Advanced Features

## Objective
Implement a comprehensive conflict detection and resolution system that identifies inconsistencies in the knowledge graph, resolves conflicts through intelligent strategies, and maintains data integrity across temporal versions and inheritance hierarchies.

## Specific Requirements

### 1. Multi-Level Conflict Detection
- Property-level conflicts between inherited and local values
- Temporal conflicts across version branches and merges
- Semantic conflicts in relationship definitions and constraints
- Cross-inheritance conflicts in property propagation chains

### 2. Intelligent Resolution Strategies
- Rule-based automatic conflict resolution with confidence scoring
- Machine learning-guided resolution recommendations
- User-guided manual resolution workflows with context preservation
- Consensus-based resolution for multi-source conflicts

### 3. Consistency Maintenance
- Real-time conflict monitoring and early detection
- Proactive validation during update operations
- Conflict prevention through constraint enforcement
- Audit trails for all conflict resolution decisions

## Implementation Steps

### 1. Create Conflict Detection Core System
```rust
// src/inheritance/validation/conflict_detector.rs
use std::collections::{HashMap, HashSet, BTreeMap};
use chrono::{DateTime, Utc};
use uuid::Uuid;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone)]
pub struct ConflictDetector {
    detection_engines: HashMap<ConflictType, Box<dyn ConflictDetectionEngine>>,
    conflict_registry: Arc<RwLock<ConflictRegistry>>,
    validation_rules: Arc<ValidationRuleEngine>,
    semantic_analyzer: Arc<SemanticConflictAnalyzer>,
    temporal_validator: Arc<TemporalConflictValidator>,
    config: ConflictDetectionConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictRegistry {
    pub active_conflicts: HashMap<String, DetectedConflict>,
    pub resolved_conflicts: HashMap<String, ResolvedConflict>,
    pub conflict_patterns: HashMap<String, ConflictPattern>,
    pub resolution_statistics: ResolutionStatistics,
    pub last_scan_timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedConflict {
    pub conflict_id: String,
    pub conflict_type: ConflictType,
    pub severity: ConflictSeverity,
    pub involved_entities: Vec<String>,
    pub conflict_details: ConflictDetails,
    pub detection_timestamp: DateTime<Utc>,
    pub detection_context: DetectionContext,
    pub suggested_resolutions: Vec<ResolutionSuggestion>,
    pub confidence_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictType {
    PropertyValueConflict,
    InheritanceConflict,
    TemporalConflict,
    SemanticConflict,
    ConstraintViolation,
    RelationshipConflict,
    TypeConflict,
    ReferentialIntegrity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictSeverity {
    Critical,    // Data integrity at risk
    High,        // Functional impact likely
    Medium,      // Minor inconsistency
    Low,         // Cosmetic or style issue
    Warning,     // Potential future issue
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictDetails {
    pub description: String,
    pub conflicting_values: HashMap<String, ConflictingValue>,
    pub affected_properties: Vec<String>,
    pub inheritance_chains: Vec<InheritanceChain>,
    pub temporal_context: Option<TemporalContext>,
    pub semantic_context: Option<SemanticContext>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictingValue {
    pub value: serde_json::Value,
    pub source: ValueSource,
    pub confidence: f64,
    pub timestamp: DateTime<Utc>,
    pub metadata: ValueMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValueSource {
    DirectAssignment { user: Option<String> },
    InheritanceChain { chain: Vec<String> },
    TemporalMerge { merge_id: String },
    SystemGenerated { algorithm: String },
    UserResolution { resolution_id: String },
}

impl ConflictDetector {
    pub async fn new(
        validation_rules: Arc<ValidationRuleEngine>,
        semantic_analyzer: Arc<SemanticConflictAnalyzer>,
        temporal_validator: Arc<TemporalConflictValidator>,
        config: ConflictDetectionConfig,
    ) -> Result<Self, ConflictDetectorError> {
        let mut detection_engines: HashMap<ConflictType, Box<dyn ConflictDetectionEngine>> = HashMap::new();
        
        // Initialize specific detection engines
        detection_engines.insert(
            ConflictType::PropertyValueConflict,
            Box::new(PropertyConflictDetector::new(config.property_config.clone())),
        );
        detection_engines.insert(
            ConflictType::InheritanceConflict,
            Box::new(InheritanceConflictDetector::new(config.inheritance_config.clone())),
        );
        detection_engines.insert(
            ConflictType::TemporalConflict,
            Box::new(TemporalConflictDetector::new(config.temporal_config.clone())),
        );
        detection_engines.insert(
            ConflictType::SemanticConflict,
            Box::new(SemanticConflictDetector::new(config.semantic_config.clone())),
        );
        
        let conflict_registry = Arc::new(RwLock::new(ConflictRegistry::new()));
        
        Ok(Self {
            detection_engines,
            conflict_registry,
            validation_rules,
            semantic_analyzer,
            temporal_validator,
            config,
        })
    }
    
    pub async fn detect_conflicts(
        &self,
        detection_scope: DetectionScope,
    ) -> Result<ConflictDetectionResult, ConflictDetectionError> {
        let detection_start = Instant::now();
        let scan_id = Uuid::new_v4().to_string();
        
        info!("Starting conflict detection scan: {} (scope: {:?})", scan_id, detection_scope);
        
        let mut all_detected_conflicts = Vec::new();
        let mut detection_statistics = HashMap::new();
        
        // Run each detection engine
        for (conflict_type, engine) in &self.detection_engines {
            let engine_start = Instant::now();
            
            let engine_results = engine.detect_conflicts(&detection_scope).await?;
            let engine_duration = engine_start.elapsed();
            
            detection_statistics.insert(conflict_type.clone(), DetectionEngineStats {
                conflicts_detected: engine_results.len(),
                detection_time: engine_duration,
                false_positive_rate: self.estimate_false_positive_rate(conflict_type).await?,
            });
            
            all_detected_conflicts.extend(engine_results);
            
            debug!(
                "Conflict detection engine {:?} completed in {:?}, found {} conflicts",
                conflict_type, engine_duration, all_detected_conflicts.len()
            );
        }
        
        // Deduplicate and merge related conflicts
        let merged_conflicts = self.merge_related_conflicts(&all_detected_conflicts).await?;
        
        // Generate resolution suggestions
        let conflicts_with_suggestions = self.generate_resolution_suggestions(&merged_conflicts).await?;
        
        // Update conflict registry
        self.update_conflict_registry(&conflicts_with_suggestions).await?;
        
        let total_duration = detection_start.elapsed();
        let result = ConflictDetectionResult {
            scan_id,
            detected_conflicts: conflicts_with_suggestions,
            detection_statistics,
            scan_duration: total_duration,
            scope: detection_scope,
            timestamp: Utc::now(),
        };
        
        info!(
            "Conflict detection completed in {:?}, found {} conflicts (merged from {} raw detections)",
            total_duration, result.detected_conflicts.len(), all_detected_conflicts.len()
        );
        
        Ok(result)
    }
    
    pub async fn detect_realtime_conflicts(
        &self,
        change_event: ChangeEvent,
    ) -> Result<Vec<DetectedConflict>, RealtimeDetectionError> {
        let detection_start = Instant::now();
        
        // Determine which detection engines are relevant for this change
        let relevant_engines = self.identify_relevant_engines(&change_event).await?;
        
        let mut detected_conflicts = Vec::new();
        
        for (conflict_type, engine) in relevant_engines {
            // Create focused detection scope for the change
            let focused_scope = DetectionScope::Focused {
                entity_ids: change_event.affected_entities.clone(),
                property_names: change_event.changed_properties.clone(),
                relationship_types: change_event.affected_relationships.clone(),
            };
            
            let conflicts = engine.detect_conflicts(&focused_scope).await?;
            
            for mut conflict in conflicts {
                // Mark as realtime detection
                conflict.detection_context.detection_mode = DetectionMode::Realtime;
                conflict.detection_context.triggering_change = Some(change_event.clone());
                detected_conflicts.push(conflict);
            }
        }
        
        // Apply immediate validation rules
        let validated_conflicts = self.apply_immediate_validation(&detected_conflicts).await?;
        
        debug!(
            "Realtime conflict detection completed in {:?}, found {} conflicts for change event {}",
            detection_start.elapsed(), validated_conflicts.len(), change_event.event_id
        );
        
        Ok(validated_conflicts)
    }
    
    async fn merge_related_conflicts(
        &self,
        conflicts: &[DetectedConflict],
    ) -> Result<Vec<DetectedConflict>, ConflictMergingError> {
        let mut merged_conflicts = Vec::new();
        let mut processed_conflicts = HashSet::new();
        
        for (i, conflict) in conflicts.iter().enumerate() {
            if processed_conflicts.contains(&i) {
                continue;
            }
            
            let mut related_conflicts = vec![conflict.clone()];
            processed_conflicts.insert(i);
            
            // Find related conflicts
            for (j, other_conflict) in conflicts.iter().enumerate().skip(i + 1) {
                if processed_conflicts.contains(&j) {
                    continue;
                }
                
                if self.are_conflicts_related(conflict, other_conflict).await? {
                    related_conflicts.push(other_conflict.clone());
                    processed_conflicts.insert(j);
                }
            }
            
            // Merge if multiple related conflicts found
            if related_conflicts.len() > 1 {
                let merged_conflict = self.merge_conflicts(&related_conflicts).await?;
                merged_conflicts.push(merged_conflict);
            } else {
                merged_conflicts.push(conflict.clone());
            }
        }
        
        Ok(merged_conflicts)
    }
    
    async fn generate_resolution_suggestions(
        &self,
        conflicts: &[DetectedConflict],
    ) -> Result<Vec<DetectedConflict>, ResolutionSuggestionError> {
        let mut conflicts_with_suggestions = Vec::new();
        
        for conflict in conflicts {
            let mut conflict_with_suggestions = conflict.clone();
            
            // Generate automatic resolution suggestions
            let auto_suggestions = self.generate_automatic_suggestions(conflict).await?;
            
            // Generate ML-based suggestions if enabled
            let ml_suggestions = if self.config.enable_ml_suggestions {
                self.generate_ml_suggestions(conflict).await?
            } else {
                Vec::new()
            };
            
            // Generate rule-based suggestions
            let rule_suggestions = self.generate_rule_based_suggestions(conflict).await?;
            
            // Combine and rank suggestions
            let all_suggestions = [auto_suggestions, ml_suggestions, rule_suggestions].concat();
            let ranked_suggestions = self.rank_resolution_suggestions(&all_suggestions, conflict).await?;
            
            conflict_with_suggestions.suggested_resolutions = ranked_suggestions;
            conflicts_with_suggestions.push(conflict_with_suggestions);
        }
        
        Ok(conflicts_with_suggestions)
    }
}

pub trait ConflictDetectionEngine: Send + Sync {
    async fn detect_conflicts(
        &self,
        scope: &DetectionScope,
    ) -> Result<Vec<DetectedConflict>, Box<dyn std::error::Error + Send + Sync>>;
    
    fn conflict_type(&self) -> ConflictType;
    fn engine_name(&self) -> &str;
}

#[derive(Debug)]
pub struct PropertyConflictDetector {
    inheritance_resolver: Arc<InheritanceResolver>,
    temporal_analyzer: Arc<TemporalAnalyzer>,
    config: PropertyConflictConfig,
}

impl ConflictDetectionEngine for PropertyConflictDetector {
    async fn detect_conflicts(
        &self,
        scope: &DetectionScope,
    ) -> Result<Vec<DetectedConflict>, Box<dyn std::error::Error + Send + Sync>> {
        let mut conflicts = Vec::new();
        
        // Get entities to check based on scope
        let entities_to_check = self.get_entities_from_scope(scope).await?;
        
        for entity_id in entities_to_check {
            // Check for property value conflicts
            let property_conflicts = self.detect_property_value_conflicts(&entity_id).await?;
            conflicts.extend(property_conflicts);
            
            // Check for inheritance conflicts
            let inheritance_conflicts = self.detect_inheritance_conflicts(&entity_id).await?;
            conflicts.extend(inheritance_conflicts);
        }
        
        Ok(conflicts)
    }
    
    fn conflict_type(&self) -> ConflictType {
        ConflictType::PropertyValueConflict
    }
    
    fn engine_name(&self) -> &str {
        "PropertyConflictDetector"
    }
}

impl PropertyConflictDetector {
    async fn detect_property_value_conflicts(
        &self,
        entity_id: &str,
    ) -> Result<Vec<DetectedConflict>, PropertyConflictError> {
        let mut conflicts = Vec::new();
        
        // Get all property values for entity (direct + inherited)
        let direct_properties = self.get_direct_properties(entity_id).await?;
        let inherited_properties = self.inheritance_resolver
            .resolve_inherited_properties(entity_id)
            .await?;
        
        // Check for conflicts between direct and inherited values
        for (property_name, direct_value) in &direct_properties {
            if let Some(inherited_value) = inherited_properties.get(property_name) {
                if self.values_conflict(&direct_value.value, &inherited_value.value) {
                    let conflict = DetectedConflict {
                        conflict_id: Uuid::new_v4().to_string(),
                        conflict_type: ConflictType::PropertyValueConflict,
                        severity: self.assess_conflict_severity(property_name, direct_value, inherited_value),
                        involved_entities: vec![entity_id.to_string()],
                        conflict_details: ConflictDetails {
                            description: format!(
                                "Property '{}' has conflicting values: direct value differs from inherited value",
                                property_name
                            ),
                            conflicting_values: {
                                let mut values = HashMap::new();
                                values.insert("direct".to_string(), ConflictingValue {
                                    value: direct_value.value.clone(),
                                    source: ValueSource::DirectAssignment { user: direct_value.last_modified_by.clone() },
                                    confidence: 1.0,
                                    timestamp: direct_value.timestamp,
                                    metadata: direct_value.metadata.clone(),
                                });
                                values.insert("inherited".to_string(), ConflictingValue {
                                    value: inherited_value.value.clone(),
                                    source: ValueSource::InheritanceChain { chain: inherited_value.inheritance_chain.clone() },
                                    confidence: inherited_value.confidence,
                                    timestamp: inherited_value.timestamp,
                                    metadata: inherited_value.metadata.clone(),
                                });
                                values
                            },
                            affected_properties: vec![property_name.clone()],
                            inheritance_chains: vec![inherited_value.inheritance_chain.clone()],
                            temporal_context: None,
                            semantic_context: None,
                        },
                        detection_timestamp: Utc::now(),
                        detection_context: DetectionContext::default(),
                        suggested_resolutions: Vec::new(), // Will be populated later
                        confidence_score: self.calculate_detection_confidence(direct_value, inherited_value),
                    };
                    
                    conflicts.push(conflict);
                }
            }
        }
        
        Ok(conflicts)
    }
    
    fn values_conflict(&self, value1: &serde_json::Value, value2: &serde_json::Value) -> bool {
        // Simple value comparison - can be made more sophisticated
        value1 != value2
    }
    
    fn assess_conflict_severity(
        &self,
        property_name: &str,
        direct_value: &PropertyValue,
        inherited_value: &ResolvedProperty,
    ) -> ConflictSeverity {
        // Assess severity based on property importance and value differences
        if self.config.critical_properties.contains(property_name) {
            ConflictSeverity::Critical
        } else if self.is_significant_difference(&direct_value.value, &inherited_value.value) {
            ConflictSeverity::High
        } else {
            ConflictSeverity::Medium
        }
    }
}
```

### 2. Create Conflict Resolution Engine
```rust
// src/inheritance/validation/conflict_resolver.rs
#[derive(Debug, Clone)]
pub struct ConflictResolver {
    resolution_strategies: HashMap<ConflictType, Box<dyn ResolutionStrategy>>,
    ml_resolver: Option<Arc<MlConflictResolver>>,
    user_interaction: Arc<UserInteractionHandler>,
    audit_logger: Arc<ConflictAuditLogger>,
    config: ConflictResolutionConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionResult {
    pub resolution_id: String,
    pub conflict_id: String,
    pub resolution_strategy: ResolutionStrategyType,
    pub resolution_actions: Vec<ResolutionAction>,
    pub confidence: f64,
    pub manual_intervention_required: bool,
    pub resolution_timestamp: DateTime<Utc>,
    pub resolved_by: Option<String>,
    pub validation_results: ValidationResults,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResolutionStrategyType {
    AutomaticRule,
    MlGuided,
    UserGuided,
    Consensus,
    TemporalPrecedence,
    SemanticCompatibility,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionAction {
    pub action_type: ActionType,
    pub target_entity: String,
    pub property_name: Option<String>,
    pub new_value: Option<serde_json::Value>,
    pub justification: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionType {
    UpdateProperty,
    RemoveProperty,
    AddProperty,
    UpdateInheritanceChain,
    CreateOverride,
    MergeValues,
    FlagForReview,
}

impl ConflictResolver {
    pub async fn resolve_conflict(
        &self,
        conflict: &DetectedConflict,
        resolution_preference: ResolutionPreference,
    ) -> Result<ResolutionResult, ConflictResolutionError> {
        let resolution_start = Instant::now();
        let resolution_id = Uuid::new_v4().to_string();
        
        info!(
            "Starting conflict resolution for conflict {} using preference {:?}",
            conflict.conflict_id, resolution_preference
        );
        
        // Select appropriate resolution strategy
        let strategy = self.select_resolution_strategy(conflict, &resolution_preference).await?;
        
        // Attempt automatic resolution first
        let resolution_result = match strategy {
            ResolutionStrategyType::AutomaticRule => {
                self.attempt_automatic_resolution(conflict).await?
            },
            ResolutionStrategyType::MlGuided => {
                self.attempt_ml_guided_resolution(conflict).await?
            },
            ResolutionStrategyType::UserGuided => {
                self.initiate_user_guided_resolution(conflict).await?
            },
            ResolutionStrategyType::Consensus => {
                self.attempt_consensus_resolution(conflict).await?
            },
            _ => {
                self.attempt_fallback_resolution(conflict, &strategy).await?
            }
        };
        
        // Validate resolution
        let validation_results = self.validate_resolution(conflict, &resolution_result).await?;
        
        // Apply resolution if validation passes
        let final_result = if validation_results.is_valid {
            self.apply_resolution(conflict, &resolution_result).await?
        } else {
            self.handle_validation_failure(conflict, &resolution_result, &validation_results).await?
        };
        
        // Log resolution
        self.audit_logger.log_resolution(conflict, &final_result).await?;
        
        let total_duration = resolution_start.elapsed();
        info!(
            "Conflict resolution completed in {:?} for conflict {} with strategy {:?}",
            total_duration, conflict.conflict_id, strategy
        );
        
        Ok(final_result)
    }
    
    async fn attempt_automatic_resolution(
        &self,
        conflict: &DetectedConflict,
    ) -> Result<ResolutionResult, AutomaticResolutionError> {
        let strategy = self.resolution_strategies
            .get(&conflict.conflict_type)
            .ok_or_else(|| AutomaticResolutionError::NoStrategyAvailable(conflict.conflict_type.clone()))?;
        
        let resolution_actions = strategy.generate_resolution_actions(conflict).await?;
        
        // Calculate confidence based on strategy reliability and conflict characteristics
        let confidence = self.calculate_resolution_confidence(conflict, &resolution_actions).await?;
        
        Ok(ResolutionResult {
            resolution_id: Uuid::new_v4().to_string(),
            conflict_id: conflict.conflict_id.clone(),
            resolution_strategy: ResolutionStrategyType::AutomaticRule,
            resolution_actions,
            confidence,
            manual_intervention_required: confidence < self.config.automatic_confidence_threshold,
            resolution_timestamp: Utc::now(),
            resolved_by: Some("system".to_string()),
            validation_results: ValidationResults::pending(),
        })
    }
    
    async fn attempt_ml_guided_resolution(
        &self,
        conflict: &DetectedConflict,
    ) -> Result<ResolutionResult, MlResolutionError> {
        let ml_resolver = self.ml_resolver.as_ref()
            .ok_or(MlResolutionError::MlResolverNotAvailable)?;
        
        // Extract features for ML model
        let conflict_features = ml_resolver.extract_conflict_features(conflict).await?;
        
        // Predict resolution actions
        let predicted_actions = ml_resolver.predict_resolution_actions(&conflict_features).await?;
        
        // Calculate ML confidence
        let ml_confidence = ml_resolver.calculate_prediction_confidence(&conflict_features, &predicted_actions).await?;
        
        Ok(ResolutionResult {
            resolution_id: Uuid::new_v4().to_string(),
            conflict_id: conflict.conflict_id.clone(),
            resolution_strategy: ResolutionStrategyType::MlGuided,
            resolution_actions: predicted_actions,
            confidence: ml_confidence,
            manual_intervention_required: ml_confidence < self.config.ml_confidence_threshold,
            resolution_timestamp: Utc::now(),
            resolved_by: Some("ml_system".to_string()),
            validation_results: ValidationResults::pending(),
        })
    }
}

pub trait ResolutionStrategy: Send + Sync {
    async fn generate_resolution_actions(
        &self,
        conflict: &DetectedConflict,
    ) -> Result<Vec<ResolutionAction>, Box<dyn std::error::Error + Send + Sync>>;
    
    fn strategy_name(&self) -> &str;
    fn applicable_conflict_types(&self) -> Vec<ConflictType>;
}
```

## Acceptance Criteria

### Functional Requirements
- [ ] Multi-level conflict detection across property, temporal, and semantic domains
- [ ] Automatic resolution strategies with confidence scoring
- [ ] Manual resolution workflows with user guidance
- [ ] Real-time conflict monitoring and prevention
- [ ] Comprehensive audit trails for all resolution decisions

### Performance Requirements
- [ ] Conflict detection scan completion < 2 seconds for 50,000 nodes
- [ ] Real-time conflict detection < 50ms for single entity changes
- [ ] Resolution suggestion generation < 100ms per conflict
- [ ] Conflict registry updates < 10ms per operation
- [ ] False positive rate < 5% for automatic detections

### Testing Requirements
- [ ] Unit tests for all conflict detection engines
- [ ] Integration tests with temporal versioning and inheritance systems
- [ ] Scenario tests for complex multi-domain conflicts
- [ ] Performance benchmarks for large-scale conflict detection

## Validation Steps

1. **Test conflict detection accuracy**:
   ```rust
   let detector = ConflictDetector::new(validation_rules, semantic_analyzer, temporal_validator, config).await?;
   let conflicts = detector.detect_conflicts(DetectionScope::Full).await?;
   assert!(conflicts.detected_conflicts.len() > 0);
   ```

2. **Test automatic resolution**:
   ```rust
   let resolver = ConflictResolver::new(strategies, ml_resolver, user_handler, audit_logger, config).await?;
   let result = resolver.resolve_conflict(&conflict, ResolutionPreference::Automatic).await?;
   assert!(result.confidence > 0.8);
   ```

3. **Run conflict resolution tests**:
   ```bash
   cargo test conflict_resolution_tests --release
   ```

## Files to Create/Modify
- `src/inheritance/validation/conflict_detector.rs` - Core conflict detection engine
- `src/inheritance/validation/conflict_resolver.rs` - Conflict resolution strategies
- `src/inheritance/validation/detection_engines.rs` - Specialized detection engines
- `src/inheritance/validation/resolution_strategies.rs` - Resolution strategy implementations
- `tests/inheritance/conflict_resolution_tests.rs` - Conflict resolution test suite

## Success Metrics
- Detection accuracy: >95% true positive rate for known conflicts
- Resolution success rate: >85% automatic resolution for low-risk conflicts
- Real-time detection latency: <50ms for entity change events
- User satisfaction: >90% approval rate for suggested resolutions

## Next Task
Upon completion, proceed to **25_compression_algorithms.md** to implement inheritance compression and optimization algorithms.