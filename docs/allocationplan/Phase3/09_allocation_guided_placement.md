# Task 09: Allocation-Guided Node Placement

**Estimated Time**: 30-35 minutes  
**Dependencies**: 08_neural_pathway_storage.md, Phase 2 Allocation Engine  
**Stage**: Neural Integration  

## Objective
Implement neural allocation-guided placement of nodes in the knowledge graph using Phase 2's allocation engine to determine optimal hierarchical positioning and relationships.

## Specific Requirements

### 1. Allocation-Driven Placement
- Use Phase 2 allocation engine for placement decisions
- Convert cortical column outputs to graph placement guidance
- Support hierarchical placement based on neural allocation
- Handle allocation conflicts and resolution

### 2. Placement Analysis
- Analyze allocation results for semantic grouping
- Determine parent-child relationships from allocation
- Handle exception detection from allocation patterns
- Support placement confidence scoring

### 3. Integration Points
- Connect to Phase 2 MultiColumnProcessor
- Use cortical consensus for placement decisions
- Store allocation metadata in graph nodes
- Support allocation-based retrieval optimization

## Implementation Steps

### 1. Create Allocation-Guided Placement Service
```rust
// src/integration/allocation_placement.rs
use crate::phase2::allocation::{AllocationEngine, CorticalConsensus, AllocationResult};
use crate::phase2::cortical::{MultiColumnProcessor, ColumnResult};

pub struct AllocationGuidedPlacement {
    allocation_engine: Arc<AllocationEngine>,
    multi_column_processor: Arc<MultiColumnProcessor>,
    connection_manager: Arc<Neo4jConnectionManager>,
    placement_cache: Arc<RwLock<LRUCache<String, PlacementDecision>>>,
    performance_monitor: Arc<PlacementPerformanceMonitor>,
}

impl AllocationGuidedPlacement {
    pub async fn new(
        allocation_engine: Arc<AllocationEngine>,
        multi_column_processor: Arc<MultiColumnProcessor>,
        connection_manager: Arc<Neo4jConnectionManager>,
    ) -> Result<Self, PlacementError> {
        Ok(Self {
            allocation_engine,
            multi_column_processor,
            connection_manager,
            placement_cache: Arc::new(RwLock::new(LRUCache::new(5000))),
            performance_monitor: Arc::new(PlacementPerformanceMonitor::new()),
        })
    }
    
    pub async fn determine_optimal_placement(
        &self,
        content: &str,
        spike_pattern: &TTFSSpikePattern,
    ) -> Result<PlacementDecision, PlacementError> {
        let placement_start = Instant::now();
        
        // Check cache first
        let cache_key = self.generate_placement_cache_key(content, spike_pattern);
        if let Some(cached_decision) = self.placement_cache.read().await.get(&cache_key) {
            return Ok(cached_decision.clone());
        }
        
        // Process through cortical columns for placement guidance
        let cortical_consensus = self.multi_column_processor
            .process_for_knowledge_graph(spike_pattern)
            .await?;
        
        // Use allocation engine to determine placement
        let allocation_result = self.allocation_engine
            .allocate_with_cortical_guidance(&cortical_consensus)
            .await?;
        
        // Analyze allocation result for graph placement
        let placement_analysis = self.analyze_allocation_for_placement(
            &allocation_result,
            &cortical_consensus,
        ).await?;
        
        // Determine hierarchical placement
        let hierarchy_placement = self.determine_hierarchy_placement(
            &placement_analysis,
            &cortical_consensus,
        ).await?;
        
        // Create placement decision
        let placement_decision = PlacementDecision {
            hierarchy_path: hierarchy_placement.path,
            parent_concept_id: hierarchy_placement.parent_id,
            confidence_score: cortical_consensus.overall_confidence,
            semantic_cluster: placement_analysis.semantic_cluster,
            structural_position: placement_analysis.structural_position,
            temporal_context: placement_analysis.temporal_context,
            detected_exceptions: placement_analysis.exceptions,
            allocation_metadata: AllocationMetadata::from(allocation_result),
            neural_pathway_reference: cortical_consensus.neural_pathway_id,
            placement_timestamp: Utc::now(),
        };
        
        // Cache the decision
        self.placement_cache.write().await.put(cache_key, placement_decision.clone());
        
        let placement_time = placement_start.elapsed();
        self.performance_monitor.record_placement_time(placement_time).await;
        
        Ok(placement_decision)
    }
    
    async fn analyze_allocation_for_placement(
        &self,
        allocation_result: &AllocationResult,
        cortical_consensus: &CorticalConsensus,
    ) -> Result<PlacementAnalysis, PlacementError> {
        // Analyze semantic column output for semantic grouping
        let semantic_analysis = self.analyze_semantic_allocation(
            &cortical_consensus.semantic_column_result,
            allocation_result,
        ).await?;
        
        // Analyze structural column output for hierarchical position
        let structural_analysis = self.analyze_structural_allocation(
            &cortical_consensus.structural_column_result,
            allocation_result,
        ).await?;
        
        // Analyze temporal column output for temporal context
        let temporal_analysis = self.analyze_temporal_allocation(
            &cortical_consensus.temporal_column_result,
            allocation_result,
        ).await?;
        
        // Analyze exception column output for conflict detection
        let exception_analysis = self.analyze_exception_allocation(
            &cortical_consensus.exception_column_result,
            allocation_result,
        ).await?;
        
        Ok(PlacementAnalysis {
            semantic_cluster: semantic_analysis.cluster_id,
            structural_position: structural_analysis.hierarchy_level,
            temporal_context: temporal_analysis.context,
            exceptions: exception_analysis.detected_exceptions,
            confidence_breakdown: ConfidenceBreakdown {
                semantic_confidence: semantic_analysis.confidence,
                structural_confidence: structural_analysis.confidence,
                temporal_confidence: temporal_analysis.confidence,
                exception_confidence: exception_analysis.confidence,
            },
        })
    }
    
    async fn determine_hierarchy_placement(
        &self,
        placement_analysis: &PlacementAnalysis,
        cortical_consensus: &CorticalConsensus,
    ) -> Result<HierarchyPlacement, PlacementError> {
        // Use semantic cluster to find appropriate parent concept
        let potential_parents = self.find_potential_parent_concepts(
            &placement_analysis.semantic_cluster,
            placement_analysis.structural_position,
        ).await?;
        
        // Rank potential parents based on cortical consensus
        let ranked_parents = self.rank_parent_candidates(
            &potential_parents,
            cortical_consensus,
        ).await?;
        
        // Select best parent based on allocation confidence
        let selected_parent = self.select_optimal_parent(
            &ranked_parents,
            &placement_analysis.confidence_breakdown,
        ).await?;
        
        // Generate hierarchy path
        let hierarchy_path = self.generate_hierarchy_path(
            &selected_parent,
            placement_analysis,
        ).await?;
        
        Ok(HierarchyPlacement {
            parent_id: selected_parent.map(|p| p.concept_id),
            path: hierarchy_path,
            depth: hierarchy_path.len() as i32,
            placement_confidence: placement_analysis.confidence_breakdown.overall_confidence(),
        })
    }
}
```

### 2. Implement Semantic Analysis for Placement
```rust
// src/integration/semantic_placement_analysis.rs
pub struct SemanticPlacementAnalyzer {
    connection_manager: Arc<Neo4jConnectionManager>,
    semantic_cache: Arc<RwLock<HashMap<String, SemanticCluster>>>,
}

impl SemanticPlacementAnalyzer {
    pub async fn analyze_semantic_allocation(
        &self,
        semantic_result: &ColumnResult,
        allocation_result: &AllocationResult,
    ) -> Result<SemanticAnalysis, SemanticAnalysisError> {
        // Extract semantic features from cortical column result
        let semantic_features = self.extract_semantic_features(semantic_result)?;
        
        // Find existing semantic clusters
        let existing_clusters = self.find_existing_semantic_clusters(
            &semantic_features,
            allocation_result.winning_column_id,
        ).await?;
        
        // Determine best cluster or create new one
        let cluster_assignment = if existing_clusters.is_empty() {
            self.create_new_semantic_cluster(&semantic_features, allocation_result).await?
        } else {
            self.select_best_cluster(&existing_clusters, &semantic_features).await?
        };
        
        Ok(SemanticAnalysis {
            cluster_id: cluster_assignment.cluster_id,
            confidence: cluster_assignment.confidence,
            semantic_features,
            cluster_centroid: cluster_assignment.centroid,
            cluster_size: cluster_assignment.size,
        })
    }
    
    async fn find_existing_semantic_clusters(
        &self,
        semantic_features: &SemanticFeatures,
        winning_column_id: &str,
    ) -> Result<Vec<SemanticCluster>, SemanticAnalysisError> {
        let session = self.connection_manager.get_session().await?;
        
        let query = r#"
            MATCH (c:Concept)
            WHERE c.semantic_cluster IS NOT NULL
              AND c.cortical_column_id = $winning_column_id
            WITH c.semantic_cluster as cluster_id, 
                 collect(c.semantic_embedding) as embeddings,
                 count(c) as cluster_size
            RETURN cluster_id, embeddings, cluster_size
        "#;
        
        let parameters = hashmap![
            "winning_column_id".to_string() => winning_column_id.into()
        ];
        
        let result = session.run(query, Some(parameters)).await?;
        
        let mut clusters = Vec::new();
        for record in result {
            let cluster_id: String = record.get("cluster_id")?;
            let embeddings: Vec<Vec<f32>> = record.get("embeddings")?;
            let size: i32 = record.get("cluster_size")?;
            
            // Calculate cluster centroid
            let centroid = self.calculate_centroid(&embeddings)?;
            
            // Calculate similarity to current semantic features
            let similarity = self.calculate_semantic_similarity(
                &semantic_features.embedding,
                &centroid,
            )?;
            
            clusters.push(SemanticCluster {
                cluster_id,
                centroid,
                size: size as usize,
                similarity_to_query: similarity,
            });
        }
        
        // Sort by similarity
        clusters.sort_by(|a, b| b.similarity_to_query.partial_cmp(&a.similarity_to_query).unwrap());
        
        Ok(clusters)
    }
    
    async fn create_new_semantic_cluster(
        &self,
        semantic_features: &SemanticFeatures,
        allocation_result: &AllocationResult,
    ) -> Result<ClusterAssignment, SemanticAnalysisError> {
        let cluster_id = format!("semantic_cluster_{}", Uuid::new_v4());
        
        // Store cluster metadata
        let session = self.connection_manager.get_session().await?;
        
        let query = r#"
            CREATE (sc:SemanticCluster {
                id: $cluster_id,
                centroid: $centroid,
                cortical_column_id: $column_id,
                allocation_pathway: $allocation_pathway,
                created_at: datetime(),
                size: 1
            })
            RETURN sc
        "#;
        
        let parameters = hashmap![
            "cluster_id".to_string() => cluster_id.clone().into(),
            "centroid".to_string() => semantic_features.embedding.clone().into(),
            "column_id".to_string() => allocation_result.winning_column_id.clone().into(),
            "allocation_pathway".to_string() => allocation_result.neural_pathway_id.clone().into()
        ];
        
        session.run(query, Some(parameters)).await?;
        
        Ok(ClusterAssignment {
            cluster_id,
            confidence: 1.0, // High confidence for new cluster
            centroid: semantic_features.embedding.clone(),
            size: 1,
        })
    }
}
```

### 3. Implement Node Placement with Allocation Guidance
```rust
// src/integration/node_placement_executor.rs
pub struct NodePlacementExecutor {
    allocation_placement: Arc<AllocationGuidedPlacement>,
    concept_crud: Arc<NodeCrudService<ConceptNode>>,
    relationship_ops: Arc<RelationshipOperations>,
}

impl NodePlacementExecutor {
    pub async fn place_concept_with_allocation(
        &self,
        concept_data: &ConceptCreationData,
        spike_pattern: &TTFSSpikePattern,
    ) -> Result<PlacedConcept, PlacementExecutionError> {
        // Determine optimal placement using allocation engine
        let placement_decision = self.allocation_placement
            .determine_optimal_placement(&concept_data.content, spike_pattern)
            .await?;
        
        // Create concept with placement metadata
        let concept = self.create_concept_with_placement(
            concept_data,
            &placement_decision,
        ).await?;
        
        // Establish hierarchical relationships
        if let Some(parent_id) = &placement_decision.parent_concept_id {
            self.create_inheritance_relationship(
                &concept.id,
                parent_id,
                &placement_decision,
            ).await?;
        }
        
        // Store allocation metadata
        self.store_allocation_metadata(
            &concept.id,
            &placement_decision.allocation_metadata,
        ).await?;
        
        // Create neural pathway relationship
        self.create_neural_pathway_relationship(
            &concept.id,
            &placement_decision.neural_pathway_reference,
        ).await?;
        
        // Handle detected exceptions
        for exception in &placement_decision.detected_exceptions {
            self.create_exception_relationship(
                &concept.id,
                exception,
            ).await?;
        }
        
        Ok(PlacedConcept {
            concept,
            placement_decision,
            relationships_created: self.count_created_relationships(&concept.id).await?,
        })
    }
    
    async fn create_concept_with_placement(
        &self,
        concept_data: &ConceptCreationData,
        placement_decision: &PlacementDecision,
    ) -> Result<ConceptNode, ConceptCreationError> {
        let concept = ConceptNodeBuilder::new(&concept_data.name, &concept_data.concept_type)
            .with_semantic_embedding(concept_data.semantic_embedding.clone())
            .with_ttfs_encoding(concept_data.ttfs_encoding)
            .with_confidence(placement_decision.confidence_score)
            .with_inheritance_depth(placement_decision.hierarchy_path.len() as i32)
            .with_allocation_metadata(placement_decision.allocation_metadata.clone())
            .with_semantic_cluster(placement_decision.semantic_cluster.clone())
            .with_neural_pathway_ref(placement_decision.neural_pathway_reference.clone())
            .build()?;
        
        let concept_id = self.concept_crud.create(&concept).await?;
        
        let mut final_concept = concept;
        final_concept.id = concept_id;
        
        Ok(final_concept)
    }
}
```

## Acceptance Criteria

### Functional Requirements
- [ ] Allocation engine integration works seamlessly with Phase 2
- [ ] Cortical column outputs guide placement decisions correctly
- [ ] Hierarchical placement reflects neural allocation patterns
- [ ] Exception detection from allocation patterns works
- [ ] Placement metadata is stored and retrievable

### Performance Requirements
- [ ] Placement decision generation < 10ms
- [ ] Allocation analysis < 5ms
- [ ] Cache hit rate for placement decisions > 60%
- [ ] Integration with Phase 2 has minimal latency overhead

### Testing Requirements
- [ ] Unit tests for placement analysis components
- [ ] Integration tests with Phase 2 allocation engine
- [ ] Performance tests for placement decision speed
- [ ] Accuracy tests for hierarchical placement

## Validation Steps

1. **Test allocation-guided placement**:
   ```rust
   let placement_service = AllocationGuidedPlacement::new(allocation_engine, processor, connection).await?;
   let decision = placement_service.determine_optimal_placement(content, &spike_pattern).await?;
   ```

2. **Test concept placement with hierarchy**:
   ```rust
   let placed_concept = placement_executor.place_concept_with_allocation(&concept_data, &spike_pattern).await?;
   ```

3. **Run integration tests**:
   ```bash
   cargo test allocation_placement_tests
   ```

## Files to Create/Modify
- `src/integration/allocation_placement.rs` - Main placement service
- `src/integration/semantic_placement_analysis.rs` - Semantic analysis
- `src/integration/node_placement_executor.rs` - Placement execution
- `tests/integration/allocation_placement_tests.rs` - Test suite

## Error Handling
- Phase 2 allocation engine connectivity issues
- Cortical consensus failures
- Invalid placement decisions
- Hierarchy constraint violations
- Cache inconsistencies

## Success Metrics
- Placement accuracy based on neural allocation: >90%
- Integration with Phase 2: Seamless
- Performance requirements met
- Exception detection accuracy: >85%

## Next Task
Upon completion, proceed to **10_spike_pattern_processing.md** to process spike patterns for graph operations.