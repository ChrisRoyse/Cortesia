# Task 14: Exception and Override System
**Estimated Time**: 15-20 minutes
**Dependencies**: 13_inheritance_cache.md
**Stage**: Inheritance System

## Objective
Implement a sophisticated exception and override system that allows concepts to define specific exceptions to inherited properties and behavior, with conflict resolution and precedence rules.

## Specific Requirements

### 1. Exception Definition Framework
- Property value exceptions with reasoning
- Behavioral override mechanisms
- Conditional exception rules with context awareness
- Exception precedence and conflict resolution

### 2. Override Management System
- Multiple inheritance conflict resolution
- Override strength and confidence scoring
- Temporal override capabilities (time-based exceptions)
- Exception inheritance through chains

### 3. Conflict Resolution Engine
- Multi-source exception handling
- Precedence-based resolution algorithms
- Confidence-weighted decision making
- Exception validation and consistency checking

## Implementation Steps

### 1. Create Exception Data Structures
```rust
// src/inheritance/exceptions/exception_types.rs
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyException {
    pub exception_id: String,
    pub concept_id: String,
    pub property_name: String,
    pub original_value: PropertyValue,
    pub exception_value: PropertyValue,
    pub exception_type: ExceptionType,
    pub reason: String,
    pub confidence: f32,
    pub precedence: u32,
    pub context_conditions: Vec<ContextCondition>,
    pub created_at: DateTime<Utc>,
    pub effective_from: Option<DateTime<Utc>>,
    pub effective_until: Option<DateTime<Utc>>,
    pub created_by: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExceptionType {
    ValueOverride,        // Simple value replacement
    ConditionalOverride,  // Context-dependent override
    TemporalOverride,     // Time-based override
    InheritanceBlock,     // Block inheritance of this property
    TransformOverride,    // Apply transformation to inherited value
    Custom(String),       // Custom exception type
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextCondition {
    pub condition_id: String,
    pub condition_type: ConditionType,
    pub property_path: String,
    pub expected_value: PropertyValue,
    pub operator: ComparisonOperator,
    pub weight: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConditionType {
    PropertyEquals,
    PropertyGreaterThan,
    PropertyLessThan,
    PropertyContains,
    ConceptHasRelation,
    TemporalCondition,
    Custom(String),
}

#[derive(Debug, Clone)]
pub struct ExceptionResolutionResult {
    pub resolved_value: PropertyValue,
    pub applied_exceptions: Vec<AppliedException>,
    pub conflicts_detected: Vec<ExceptionConflict>,
    pub resolution_confidence: f32,
    pub resolution_method: ResolutionMethod,
}

#[derive(Debug, Clone)]
pub struct AppliedException {
    pub exception_id: String,
    pub exception_type: ExceptionType,
    pub original_value: PropertyValue,
    pub resolved_value: PropertyValue,
    pub application_confidence: f32,
    pub context_match_score: f32,
}

#[derive(Debug, Clone)]
pub struct ExceptionConflict {
    pub conflict_id: String,
    pub conflicting_exceptions: Vec<String>,
    pub conflict_type: ConflictType,
    pub resolution_strategy: ConflictResolutionStrategy,
    pub confidence_impact: f32,
}
```

### 2. Implement Exception Manager
```rust
// src/inheritance/exceptions/exception_manager.rs
pub struct ExceptionManager {
    connection_manager: Arc<Neo4jConnectionManager>,
    exception_cache: Arc<RwLock<HashMap<String, Vec<PropertyException>>>>,
    resolution_engine: Arc<ExceptionResolutionEngine>,
    conflict_resolver: Arc<ConflictResolver>,
    performance_monitor: Arc<ExceptionPerformanceMonitor>,
}

impl ExceptionManager {
    pub async fn new(
        connection_manager: Arc<Neo4jConnectionManager>,
    ) -> Result<Self, ExceptionManagerError> {
        Ok(Self {
            connection_manager,
            exception_cache: Arc::new(RwLock::new(HashMap::new())),
            resolution_engine: Arc::new(ExceptionResolutionEngine::new()),
            conflict_resolver: Arc::new(ConflictResolver::new()),
            performance_monitor: Arc::new(ExceptionPerformanceMonitor::new()),
        })
    }
    
    pub async fn create_property_exception(
        &self,
        concept_id: &str,
        property_name: &str,
        original_value: PropertyValue,
        exception_value: PropertyValue,
        exception_type: ExceptionType,
        reason: &str,
        confidence: f32,
        context_conditions: Vec<ContextCondition>,
    ) -> Result<String, ExceptionCreationError> {
        let creation_start = Instant::now();
        
        // Validate exception parameters
        self.validate_exception_parameters(
            concept_id,
            property_name,
            &original_value,
            &exception_value,
            confidence,
        ).await?;
        
        // Generate exception ID
        let exception_id = format!("exc_{}_{}_{}",
            concept_id,
            property_name,
            Utc::now().timestamp_millis()
        );
        
        // Calculate precedence based on existing exceptions
        let precedence = self.calculate_exception_precedence(
            concept_id,
            property_name,
            &exception_type,
        ).await?;
        
        let exception = PropertyException {
            exception_id: exception_id.clone(),
            concept_id: concept_id.to_string(),
            property_name: property_name.to_string(),
            original_value,
            exception_value,
            exception_type,
            reason: reason.to_string(),
            confidence,
            precedence,
            context_conditions,
            created_at: Utc::now(),
            effective_from: None,
            effective_until: None,
            created_by: "system".to_string(), // Should be from context
        };
        
        // Store exception in database
        self.store_exception_in_database(&exception).await?;
        
        // Update cache
        self.update_exception_cache(concept_id, exception).await;
        
        // Check for conflicts with existing exceptions
        let conflicts = self.check_exception_conflicts(concept_id, property_name).await?;
        if !conflicts.is_empty() {
            // Log conflicts but don't fail - let resolution engine handle them
            warn!("Exception conflicts detected for {}:{}: {:?}", concept_id, property_name, conflicts);
        }
        
        // Record performance metrics
        let creation_time = creation_start.elapsed();
        self.performance_monitor.record_exception_creation_time(creation_time).await;
        
        Ok(exception_id)
    }
    
    pub async fn resolve_property_with_exceptions(
        &self,
        concept_id: &str,
        property_name: &str,
        inherited_value: PropertyValue,
        context: &ResolutionContext,
    ) -> Result<ExceptionResolutionResult, ResolutionError> {
        let resolution_start = Instant::now();
        
        // Get applicable exceptions for this property
        let exceptions = self.get_applicable_exceptions(
            concept_id,
            property_name,
            context,
        ).await?;
        
        if exceptions.is_empty() {
            // No exceptions, return inherited value
            return Ok(ExceptionResolutionResult {
                resolved_value: inherited_value,
                applied_exceptions: Vec::new(),
                conflicts_detected: Vec::new(),
                resolution_confidence: 1.0,
                resolution_method: ResolutionMethod::NoExceptions,
            });
        }
        
        // Check for conflicts between exceptions
        let conflicts = self.detect_exception_conflicts(&exceptions).await?;
        
        // Resolve conflicts and apply exceptions
        let resolution_result = if conflicts.is_empty() {
            self.apply_exceptions_without_conflicts(
                inherited_value,
                exceptions,
                context,
            ).await?
        } else {
            self.resolve_conflicted_exceptions(
                inherited_value,
                exceptions,
                conflicts,
                context,
            ).await?
        };
        
        // Record performance metrics
        let resolution_time = resolution_start.elapsed();
        self.performance_monitor.record_resolution_time(resolution_time).await;
        
        Ok(resolution_result)
    }
    
    async fn get_applicable_exceptions(
        &self,
        concept_id: &str,
        property_name: &str,
        context: &ResolutionContext,
    ) -> Result<Vec<PropertyException>, ExceptionRetrievalError> {
        // Check cache first
        let cache_key = format!("{}:{}", concept_id, property_name);
        if let Some(cached_exceptions) = self.exception_cache.read().await.get(&cache_key) {
            // Filter by context conditions
            let applicable = self.filter_exceptions_by_context(cached_exceptions, context).await?;
            return Ok(applicable);
        }
        
        // Query database for exceptions
        let session = self.connection_manager.get_session().await?;
        let query = r#"
            MATCH (c:Concept {id: $concept_id})-[:HAS_EXCEPTION]->(e:Exception)
            WHERE e.property_name = $property_name 
              AND (e.effective_from IS NULL OR e.effective_from <= $current_time)
              AND (e.effective_until IS NULL OR e.effective_until >= $current_time)
            RETURN e
            ORDER BY e.precedence DESC, e.confidence DESC, e.created_at ASC
        "#;
        
        let parameters = hashmap![
            "concept_id".to_string() => concept_id.into(),
            "property_name".to_string() => property_name.into(),
            "current_time".to_string() => Utc::now().into(),
        ];
        
        let result = session.run(query, Some(parameters)).await?;
        
        let mut exceptions = Vec::new();
        for record in result {
            let exception_data: Value = record.get("e")?;
            let exception = PropertyException::from_neo4j_value(exception_data)?;
            exceptions.push(exception);
        }
        
        // Filter by context conditions
        let applicable_exceptions = self.filter_exceptions_by_context(&exceptions, context).await?;
        
        // Update cache
        self.exception_cache.write().await.insert(cache_key, exceptions);
        
        Ok(applicable_exceptions)
    }
    
    async fn apply_exceptions_without_conflicts(
        &self,
        inherited_value: PropertyValue,
        exceptions: Vec<PropertyException>,
        context: &ResolutionContext,
    ) -> Result<ExceptionResolutionResult, ApplicationError> {
        let mut current_value = inherited_value;
        let mut applied_exceptions = Vec::new();
        let mut resolution_confidence = 1.0;
        
        // Apply exceptions in precedence order
        for exception in exceptions {
            let application_result = self.apply_single_exception(
                &current_value,
                &exception,
                context,
            ).await?;
            
            if let Some(result) = application_result {
                current_value = result.resolved_value.clone();
                resolution_confidence *= result.application_confidence;
                applied_exceptions.push(result);
            }
        }
        
        Ok(ExceptionResolutionResult {
            resolved_value: current_value,
            applied_exceptions,
            conflicts_detected: Vec::new(),
            resolution_confidence,
            resolution_method: ResolutionMethod::SequentialApplication,
        })
    }
    
    async fn resolve_conflicted_exceptions(
        &self,
        inherited_value: PropertyValue,
        exceptions: Vec<PropertyException>,
        conflicts: Vec<ExceptionConflict>,
        context: &ResolutionContext,
    ) -> Result<ExceptionResolutionResult, ConflictResolutionError> {
        // Use conflict resolver to determine best resolution strategy
        let resolution_strategy = self.conflict_resolver.determine_best_strategy(
            &exceptions,
            &conflicts,
            context,
        ).await?;
        
        let resolution_result = match resolution_strategy {
            ConflictResolutionStrategy::HighestConfidence => {
                self.resolve_by_highest_confidence(inherited_value, exceptions, context).await?
            },
            ConflictResolutionStrategy::HighestPrecedence => {
                self.resolve_by_highest_precedence(inherited_value, exceptions, context).await?
            },
            ConflictResolutionStrategy::WeightedCombination => {
                self.resolve_by_weighted_combination(inherited_value, exceptions, context).await?
            },
            ConflictResolutionStrategy::TemporalPriority => {
                self.resolve_by_temporal_priority(inherited_value, exceptions, context).await?
            },
        };
        
        Ok(ExceptionResolutionResult {
            conflicts_detected: conflicts,
            resolution_method: ResolutionMethod::ConflictResolution(resolution_strategy),
            ..resolution_result
        })
    }
}
```

### 3. Implement Conflict Resolution Engine
```rust
// src/inheritance/exceptions/conflict_resolver.rs
pub struct ConflictResolver {
    resolution_strategies: HashMap<ConflictType, ConflictResolutionStrategy>,
    confidence_calculator: ConfidenceCalculator,
    temporal_resolver: TemporalResolver,
}

impl ConflictResolver {
    pub fn new() -> Self {
        let mut strategies = HashMap::new();
        strategies.insert(ConflictType::ValueConflict, ConflictResolutionStrategy::HighestConfidence);
        strategies.insert(ConflictType::TypeConflict, ConflictResolutionStrategy::HighestPrecedence);
        strategies.insert(ConflictType::TemporalConflict, ConflictResolutionStrategy::TemporalPriority);
        
        Self {
            resolution_strategies: strategies,
            confidence_calculator: ConfidenceCalculator::new(),
            temporal_resolver: TemporalResolver::new(),
        }
    }
    
    pub async fn determine_best_strategy(
        &self,
        exceptions: &[PropertyException],
        conflicts: &[ExceptionConflict],
        context: &ResolutionContext,
    ) -> Result<ConflictResolutionStrategy, StrategyDeterminationError> {
        // Analyze conflict types
        let primary_conflict_type = self.analyze_primary_conflict_type(conflicts).await?;
        
        // Get default strategy for conflict type
        let default_strategy = self.resolution_strategies
            .get(&primary_conflict_type)
            .cloned()
            .unwrap_or(ConflictResolutionStrategy::HighestConfidence);
        
        // Check if context suggests a different strategy
        let context_strategy = self.analyze_context_strategy(exceptions, context).await?;
        
        // Combine strategies based on confidence and context
        let final_strategy = self.combine_strategies(
            default_strategy,
            context_strategy,
            exceptions,
        ).await?;
        
        Ok(final_strategy)
    }
    
    pub async fn resolve_by_highest_confidence(
        &self,
        inherited_value: PropertyValue,
        exceptions: Vec<PropertyException>,
        context: &ResolutionContext,
    ) -> Result<ExceptionResolutionResult, ResolutionError> {
        // Find exception with highest confidence
        let highest_confidence_exception = exceptions
            .iter()
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
            .ok_or(ResolutionError::NoApplicableExceptions)?;
        
        // Apply the highest confidence exception
        let applied_exception = AppliedException {
            exception_id: highest_confidence_exception.exception_id.clone(),
            exception_type: highest_confidence_exception.exception_type.clone(),
            original_value: inherited_value.clone(),
            resolved_value: highest_confidence_exception.exception_value.clone(),
            application_confidence: highest_confidence_exception.confidence,
            context_match_score: self.calculate_context_match_score(
                highest_confidence_exception,
                context,
            ).await?,
        };
        
        Ok(ExceptionResolutionResult {
            resolved_value: highest_confidence_exception.exception_value.clone(),
            applied_exceptions: vec![applied_exception],
            conflicts_detected: Vec::new(), // Conflicts resolved
            resolution_confidence: highest_confidence_exception.confidence,
            resolution_method: ResolutionMethod::ConflictResolution(
                ConflictResolutionStrategy::HighestConfidence
            ),
        })
    }
    
    pub async fn resolve_by_weighted_combination(
        &self,
        inherited_value: PropertyValue,
        exceptions: Vec<PropertyException>,
        context: &ResolutionContext,
    ) -> Result<ExceptionResolutionResult, ResolutionError> {
        // Calculate weights for each exception
        let weighted_exceptions = self.calculate_exception_weights(&exceptions, context).await?;
        
        // Combine exception values using weights
        let combined_value = self.combine_values_with_weights(
            &inherited_value,
            &weighted_exceptions,
        ).await?;
        
        // Calculate combined confidence
        let combined_confidence = self.calculate_combined_confidence(&weighted_exceptions);
        
        // Create applied exceptions list
        let applied_exceptions: Vec<AppliedException> = weighted_exceptions
            .iter()
            .map(|(exception, weight)| AppliedException {
                exception_id: exception.exception_id.clone(),
                exception_type: exception.exception_type.clone(),
                original_value: inherited_value.clone(),
                resolved_value: exception.exception_value.clone(),
                application_confidence: exception.confidence * weight,
                context_match_score: *weight,
            })
            .collect();
        
        Ok(ExceptionResolutionResult {
            resolved_value: combined_value,
            applied_exceptions,
            conflicts_detected: Vec::new(),
            resolution_confidence: combined_confidence,
            resolution_method: ResolutionMethod::ConflictResolution(
                ConflictResolutionStrategy::WeightedCombination
            ),
        })
    }
}
```

## Acceptance Criteria

### Functional Requirements
- [ ] Property exceptions can be created with context conditions
- [ ] Exception resolution works for single and multiple conflicts
- [ ] Precedence-based conflict resolution functions correctly
- [ ] Temporal exceptions activate and deactivate properly
- [ ] Context-aware exception filtering works accurately

### Performance Requirements
- [ ] Exception creation time < 8ms
- [ ] Exception resolution time < 12ms for 10 exceptions
- [ ] Conflict resolution completes within 20ms
- [ ] Cache hit rate > 75% for exception queries

### Testing Requirements
- [ ] Unit tests for exception data structures
- [ ] Integration tests for conflict resolution scenarios
- [ ] Performance tests for large exception sets
- [ ] Context condition evaluation tests

## Validation Steps

1. **Test exception creation**:
   ```rust
   let exception_id = exception_manager.create_property_exception(
       "concept_id", "property_name", original_value, exception_value,
       ExceptionType::ValueOverride, "test reason", 0.9, vec![]
   ).await?;
   ```

2. **Test conflict resolution**:
   ```rust
   let resolution = exception_manager.resolve_property_with_exceptions(
       "concept_id", "property_name", inherited_value, &context
   ).await?;
   assert!(!resolution.conflicts_detected.is_empty());
   ```

3. **Run exception tests**:
   ```bash
   cargo test inheritance_exception_tests
   ```

## Files to Create/Modify
- `src/inheritance/exceptions/exception_types.rs` - Exception data structures
- `src/inheritance/exceptions/exception_manager.rs` - Core exception manager
- `src/inheritance/exceptions/conflict_resolver.rs` - Conflict resolution engine
- `src/inheritance/exceptions/mod.rs` - Module exports
- `tests/inheritance/exception_tests.rs` - Exception test suite

## Success Metrics
- Exception resolution accuracy: 100%
- Conflict resolution success rate: > 95%
- Average resolution time < 12ms
- Context condition evaluation accuracy: 100%

## Next Task
Upon completion, proceed to **15_inheritance_validation.md** to add inheritance validation and consistency checks.