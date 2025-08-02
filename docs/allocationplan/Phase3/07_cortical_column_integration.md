# Task 07: Cortical Column Integration

**Estimated Time**: 15-20 minutes  
**Dependencies**: 06_ttfs_encoding_integration.md, Phase 2 cortical columns  
**Stage**: Neural Integration  

## Objective
Connect knowledge graph operations to Phase 2 cortical column infrastructure for neural-guided data placement and retrieval patterns.

## Specific Requirements

### 1. Cortical Column Interface
- Integrate with Phase 2 cortical column architecture
- Map knowledge graph regions to cortical columns
- Support column-specific data placement strategies
- Enable cross-column communication for graph traversal

### 2. Neural-Guided Placement
- Use cortical column activation patterns for node placement
- Implement column-aware clustering
- Support dynamic column allocation based on content type
- Optimize for column-specific access patterns

### 3. Column State Management
- Track column activation states
- Monitor column capacity and usage
- Implement column load balancing
- Support column hibernation for unused regions

## Implementation Steps

### 1. Create Cortical Column Integration Service
```rust
// src/integration/cortical_column_integration.rs
use crate::phase2::cortical::{CorticalColumn, ColumnManager, ColumnActivationPattern};

pub struct CorticalColumnIntegration {
    column_manager: Arc<ColumnManager>,
    column_mappings: Arc<RwLock<HashMap<String, ColumnId>>>,
    activation_monitor: Arc<ActivationMonitor>,
    placement_optimizer: Arc<PlacementOptimizer>,
}

impl CorticalColumnIntegration {
    pub async fn new(column_manager: Arc<ColumnManager>) -> Result<Self, IntegrationError> {
        Ok(Self {
            column_manager,
            column_mappings: Arc::new(RwLock::new(HashMap::new())),
            activation_monitor: Arc::new(ActivationMonitor::new()),
            placement_optimizer: Arc::new(PlacementOptimizer::new()),
        })
    }
    
    pub async fn assign_concept_to_column(
        &self,
        concept_id: &str,
        concept_type: &str,
        content_features: &[f32],
    ) -> Result<ColumnId, ColumnAssignmentError> {
        let assignment_start = Instant::now();
        
        // Find optimal column based on content features
        let optimal_column = self.find_optimal_column(
            concept_type,
            content_features,
        ).await?;
        
        // Check column capacity
        if !self.column_manager.has_capacity(optimal_column).await? {
            // Find alternative column or trigger column expansion
            let alternative_column = self.find_alternative_column(
                concept_type,
                content_features,
            ).await?;
            
            if alternative_column.is_none() {
                return Err(ColumnAssignmentError::NoAvailableColumns);
            }
            
            optimal_column = alternative_column.unwrap();
        }
        
        // Assign concept to column
        self.column_manager.assign_concept(optimal_column, concept_id).await?;
        
        // Update mappings
        self.column_mappings.write().await.insert(
            concept_id.to_string(),
            optimal_column,
        );
        
        // Record performance metrics
        let assignment_time = assignment_start.elapsed();
        self.activation_monitor.record_assignment_time(assignment_time).await;
        
        Ok(optimal_column)
    }
    
    async fn find_optimal_column(
        &self,
        concept_type: &str,
        content_features: &[f32],
    ) -> Result<ColumnId, ColumnAssignmentError> {
        // Get activation patterns for content type
        let activation_pattern = self.column_manager
            .get_activation_pattern_for_type(concept_type)
            .await?;
        
        // Find column with highest activation for this pattern
        let optimal_column = self.placement_optimizer
            .find_best_column(activation_pattern, content_features)
            .await?;
        
        Ok(optimal_column)
    }
}
```

### 2. Implement Column-Aware Query Engine
```rust
// src/integration/column_aware_queries.rs
pub struct ColumnAwareQueryEngine {
    cortical_integration: Arc<CorticalColumnIntegration>,
    connection_manager: Arc<Neo4jConnectionManager>,
    query_optimizer: Arc<ColumnQueryOptimizer>,
}

impl ColumnAwareQueryEngine {
    pub async fn query_by_column_activation(
        &self,
        activation_pattern: &ColumnActivationPattern,
        query_type: QueryType,
        limit: usize,
    ) -> Result<Vec<ConceptNode>, ColumnQueryError> {
        let query_start = Instant::now();
        
        // Identify active columns for this pattern
        let active_columns = self.cortical_integration
            .get_active_columns(activation_pattern)
            .await?;
        
        // Build column-specific query
        let column_query = self.build_column_query(
            &active_columns,
            query_type,
            limit,
        )?;
        
        // Execute query with column optimization
        let session = self.connection_manager.get_session().await?;
        let result = session.run(&column_query.cypher, Some(column_query.parameters)).await?;
        
        // Parse results
        let mut concepts = Vec::new();
        for record in result {
            concepts.push(ConceptNode::from_neo4j_record(record)?);
        }
        
        // Record query performance
        let query_time = query_start.elapsed();
        self.query_optimizer.record_column_query_time(
            active_columns.len(),
            query_time,
        ).await;
        
        Ok(concepts)
    }
    
    fn build_column_query(
        &self,
        active_columns: &[ColumnId],
        query_type: QueryType,
        limit: usize,
    ) -> Result<ColumnQuery, QueryBuildError> {
        let column_filter = active_columns
            .iter()
            .map(|id| format!("c.column_id = '{}'", id))
            .collect::<Vec<_>>()
            .join(" OR ");
        
        let cypher = match query_type {
            QueryType::ByActivation => format!(
                r#"
                MATCH (c:Concept)
                WHERE ({})
                AND c.activation_level > $activation_threshold
                RETURN c
                ORDER BY c.activation_level DESC
                LIMIT $limit
                "#,
                column_filter
            ),
            QueryType::ByRecentAccess => format!(
                r#"
                MATCH (c:Concept)
                WHERE ({})
                AND c.last_accessed > $time_threshold
                RETURN c
                ORDER BY c.last_accessed DESC
                LIMIT $limit
                "#,
                column_filter
            ),
        };
        
        Ok(ColumnQuery {
            cypher,
            parameters: hashmap![
                "activation_threshold".to_string() => 0.5.into(),
                "time_threshold".to_string() => (Utc::now() - Duration::hours(1)).into(),
                "limit".to_string() => (limit as i64).into(),
            ],
        })
    }
}
```

### 3. Add Column State Monitoring
```rust
// src/integration/column_state_monitor.rs
pub struct ColumnStateMonitor {
    column_manager: Arc<ColumnManager>,
    state_cache: Arc<RwLock<HashMap<ColumnId, ColumnState>>>,
    performance_tracker: Arc<PerformanceTracker>,
}

impl ColumnStateMonitor {
    pub async fn monitor_column_states(&self) -> Result<(), MonitoringError> {
        let monitoring_start = Instant::now();
        
        // Get all active columns
        let active_columns = self.column_manager.get_active_columns().await?;
        
        for column_id in active_columns {
            let column_state = self.assess_column_state(column_id).await?;
            
            // Update state cache
            self.state_cache.write().await.insert(column_id, column_state.clone());
            
            // Check for state changes requiring action
            if column_state.requires_action() {
                self.handle_column_state_change(column_id, &column_state).await?;
            }
        }
        
        let monitoring_time = monitoring_start.elapsed();
        self.performance_tracker.record_monitoring_cycle(monitoring_time).await;
        
        Ok(())
    }
    
    async fn assess_column_state(
        &self,
        column_id: ColumnId,
    ) -> Result<ColumnState, StateAssessmentError> {
        let column_info = self.column_manager.get_column_info(column_id).await?;
        
        let state = ColumnState {
            id: column_id,
            activation_level: column_info.current_activation,
            capacity_used: column_info.capacity_used,
            capacity_total: column_info.capacity_total,
            last_accessed: column_info.last_accessed,
            concept_count: column_info.concept_count,
            performance_metrics: column_info.performance_metrics.clone(),
        };
        
        Ok(state)
    }
    
    async fn handle_column_state_change(
        &self,
        column_id: ColumnId,
        state: &ColumnState,
    ) -> Result<(), StateChangeError> {
        match state.get_required_action() {
            ColumnAction::LoadBalance => {
                self.trigger_load_balancing(column_id).await?;
            },
            ColumnAction::Hibernate => {
                self.request_hibernation(column_id).await?;
            },
            ColumnAction::Expand => {
                self.request_expansion(column_id).await?;
            },
            ColumnAction::None => {
                // No action required
            },
        }
        
        Ok(())
    }
}
```

## Acceptance Criteria

### Functional Requirements
- [ ] Cortical column integration connects to Phase 2 components
- [ ] Concept-to-column assignment works correctly
- [ ] Column-aware queries return appropriate results
- [ ] Column state monitoring detects capacity and performance issues
- [ ] Load balancing triggers automatically when needed

### Performance Requirements
- [ ] Column assignment time < 5ms
- [ ] Column-aware queries complete within 20ms
- [ ] State monitoring cycle < 100ms
- [ ] Column capacity utilization > 80%

### Testing Requirements
- [ ] Unit tests for column integration components
- [ ] Integration tests with Phase 2 cortical columns
- [ ] Performance tests for assignment and queries
- [ ] State monitoring accuracy tests

## Validation Steps

1. **Test column assignment**:
   ```rust
   let column_id = cortical_integration.assign_concept_to_column(
       "concept_123", "Entity", &content_features
   ).await?;
   ```

2. **Test column-aware queries**:
   ```rust
   let concepts = query_engine.query_by_column_activation(
       &activation_pattern, QueryType::ByActivation, 10
   ).await?;
   ```

3. **Run integration tests**:
   ```bash
   cargo test cortical_column_integration_tests
   ```

## Files to Create/Modify
- `src/integration/cortical_column_integration.rs` - Main integration service
- `src/integration/column_aware_queries.rs` - Column-aware query engine
- `src/integration/column_state_monitor.rs` - State monitoring
- `tests/integration/cortical_column_tests.rs` - Integration test suite

## Error Handling
- Column assignment failures
- Column capacity exceeded
- Phase 2 connectivity issues
- State monitoring failures
- Query optimization errors

## Success Metrics
- Column assignment success rate: 100%
- Average column utilization: 80-90%
- Query performance improvement: 25% over non-column-aware queries
- State monitoring accuracy: 95%

## Next Task
Upon completion, proceed to **08_neural_pathway_storage.md** to implement neural pathway metadata storage.