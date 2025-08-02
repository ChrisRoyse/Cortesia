# Task 23: Spreading Activation Search Implementation
**Estimated Time**: 15-20 minutes
**Dependencies**: 22_branch_management.md
**Stage**: Advanced Features

## Objective
Implement a sophisticated spreading activation search algorithm that propagates activation through the knowledge graph based on semantic relationships and neural-inspired patterns, enabling intuitive discovery of related concepts and emergent knowledge patterns.

## Specific Requirements

### 1. Activation Propagation Engine
- Multi-level activation spreading with decay functions
- Weighted propagation based on relationship strength and types
- Parallel activation processing for performance optimization
- Adaptive threshold mechanisms for activation filtering

### 2. Semantic-Based Activation Rules
- Context-aware activation weights and propagation rules
- Inheritance-guided activation flow with property influence
- Dynamic activation patterns based on query context
- Attention mechanisms for focused activation spreading

### 3. Neural-Inspired Processing
- Spike-timing dependent propagation patterns
- Competitive inhibition between activation pathways
- Homeostatic regulation of overall activation levels
- Plasticity-based learning of activation patterns

## Implementation Steps

### 1. Create Spreading Activation Core Engine
```rust
// src/inheritance/search/spreading_activation.rs
use std::collections::{HashMap, HashSet, BinaryHeap};
use std::cmp::Ordering;
use chrono::{DateTime, Utc};
use uuid::Uuid;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone)]
pub struct SpreadingActivationEngine {
    activation_state: Arc<RwLock<ActivationState>>,
    propagation_rules: Arc<PropagationRuleEngine>,
    decay_function: Arc<dyn DecayFunction>,
    inhibition_processor: Arc<InhibitionProcessor>,
    plasticity_manager: Arc<PlasticityManager>,
    neural_processor: Arc<NeuralActivationProcessor>,
    config: SpreadingActivationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationState {
    pub node_activations: HashMap<String, NodeActivation>,
    pub relationship_activations: HashMap<String, RelationshipActivation>,
    pub global_activation_level: f64,
    pub activation_timestamp: DateTime<Utc>,
    pub activation_cycle: u64,
    pub inhibition_state: InhibitionState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeActivation {
    pub node_id: String,
    pub current_activation: f64,
    pub previous_activation: f64,
    pub activation_history: Vec<ActivationRecord>,
    pub source_activations: HashMap<String, f64>,
    pub propagated_activation: f64,
    pub threshold: f64,
    pub is_source: bool,
    pub activation_metadata: ActivationMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipActivation {
    pub relationship_id: String,
    pub source_node: String,
    pub target_node: String,
    pub activation_strength: f64,
    pub propagation_weight: f64,
    pub relationship_type: String,
    pub directional_bias: f64,
    pub activation_metadata: RelationshipActivationMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationRecord {
    pub timestamp: DateTime<Utc>,
    pub activation_value: f64,
    pub source: ActivationSource,
    pub propagation_path: Vec<String>,
    pub decay_applied: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationSource {
    Initial(String),
    Propagated { from_node: String, via_relationship: String },
    Reinforced { original_source: String, reinforcement_factor: f64 },
    Inhibited { inhibitor_node: String, inhibition_strength: f64 },
}

impl SpreadingActivationEngine {
    pub async fn new(
        config: SpreadingActivationConfig,
        neural_processor: Arc<NeuralActivationProcessor>,
    ) -> Result<Self, ActivationError> {
        let activation_state = Arc::new(RwLock::new(ActivationState::new()));
        let propagation_rules = Arc::new(PropagationRuleEngine::new(config.rule_config.clone()));
        let decay_function = create_decay_function(config.decay_config.clone())?;
        let inhibition_processor = Arc::new(InhibitionProcessor::new(config.inhibition_config.clone()));
        let plasticity_manager = Arc::new(PlasticityManager::new(config.plasticity_config.clone()));
        
        Ok(Self {
            activation_state,
            propagation_rules,
            decay_function,
            inhibition_processor,
            plasticity_manager,
            neural_processor,
            config,
        })
    }
    
    pub async fn initiate_spreading_activation(
        &self,
        search_query: SpreadingActivationQuery,
    ) -> Result<ActivationResult, ActivationError> {
        let activation_start = Instant::now();
        let session_id = Uuid::new_v4().to_string();
        
        info!(
            "Starting spreading activation search for query: {:?} (session: {})",
            search_query.query_terms, session_id
        );
        
        // Clear previous activation state
        self.reset_activation_state().await?;
        
        // Initialize source activations
        let source_nodes = self.identify_source_nodes(&search_query).await?;
        self.set_initial_activations(&source_nodes, &search_query).await?;
        
        // Perform spreading activation cycles
        let mut activation_results = Vec::new();
        for cycle in 0..search_query.max_cycles {
            let cycle_start = Instant::now();
            
            // Propagate activation
            let cycle_result = self.perform_activation_cycle(cycle).await?;
            activation_results.push(cycle_result.clone());
            
            // Check convergence criteria
            if self.check_convergence(&cycle_result).await? {
                info!("Activation converged after {} cycles", cycle + 1);
                break;
            }
            
            // Apply decay and inhibition
            self.apply_decay_and_inhibition().await?;
            
            // Log cycle completion
            debug!(
                "Completed activation cycle {} in {:?}, total activated nodes: {}",
                cycle, cycle_start.elapsed(), cycle_result.activated_nodes.len()
            );
        }
        
        // Collect and rank final results
        let final_result = self.collect_activation_results(&search_query).await?;
        
        // Apply plasticity learning
        self.plasticity_manager.update_plasticity(&final_result).await?;
        
        let total_duration = activation_start.elapsed();
        info!(
            "Spreading activation completed in {:?}, found {} relevant nodes",
            total_duration, final_result.ranked_nodes.len()
        );
        
        Ok(ActivationResult {
            session_id,
            query: search_query.clone(),
            ranked_nodes: final_result.ranked_nodes,
            activation_paths: final_result.activation_paths,
            cycles_performed: activation_results.len(),
            convergence_achieved: final_result.convergence_achieved,
            total_duration,
            activation_statistics: self.compute_activation_statistics(&activation_results).await,
        })
    }
    
    async fn identify_source_nodes(
        &self,
        query: &SpreadingActivationQuery,
    ) -> Result<Vec<SourceNode>, SourceIdentificationError> {
        let mut source_nodes = Vec::new();
        
        // Direct node matches
        for term in &query.query_terms {
            if let Some(direct_matches) = self.find_direct_node_matches(term).await? {
                for node_id in direct_matches {
                    source_nodes.push(SourceNode {
                        node_id: node_id.clone(),
                        initial_activation: query.initial_activation_strength,
                        source_type: SourceType::DirectMatch,
                        confidence: 1.0,
                        query_term: term.clone(),
                    });
                }
            }
        }
        
        // Semantic similarity matches
        for term in &query.query_terms {
            let similarity_matches = self.find_semantic_matches(
                term,
                query.semantic_similarity_threshold,
            ).await?;
            
            for (node_id, similarity_score) in similarity_matches {
                source_nodes.push(SourceNode {
                    node_id,
                    initial_activation: query.initial_activation_strength * similarity_score,
                    source_type: SourceType::SemanticMatch,
                    confidence: similarity_score,
                    query_term: term.clone(),
                });
            }
        }
        
        // Property-based matches
        if let Some(property_filters) = &query.property_filters {
            let property_matches = self.find_property_matches(property_filters).await?;
            for node_id in property_matches {
                source_nodes.push(SourceNode {
                    node_id,
                    initial_activation: query.initial_activation_strength * 0.8,
                    source_type: SourceType::PropertyMatch,
                    confidence: 0.8,
                    query_term: "property_filter".to_string(),
                });
            }
        }
        
        // Remove duplicates and sort by confidence
        source_nodes.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        source_nodes.dedup_by(|a, b| a.node_id == b.node_id);
        
        Ok(source_nodes)
    }
    
    async fn set_initial_activations(
        &self,
        source_nodes: &[SourceNode],
        query: &SpreadingActivationQuery,
    ) -> Result<(), ActivationError> {
        let mut activation_state = self.activation_state.write().await;
        
        for source_node in source_nodes {
            let node_activation = NodeActivation {
                node_id: source_node.node_id.clone(),
                current_activation: source_node.initial_activation,
                previous_activation: 0.0,
                activation_history: vec![ActivationRecord {
                    timestamp: Utc::now(),
                    activation_value: source_node.initial_activation,
                    source: ActivationSource::Initial(source_node.query_term.clone()),
                    propagation_path: vec![source_node.node_id.clone()],
                    decay_applied: 0.0,
                }],
                source_activations: HashMap::new(),
                propagated_activation: 0.0,
                threshold: query.activation_threshold,
                is_source: true,
                activation_metadata: ActivationMetadata {
                    created_at: Utc::now(),
                    source_type: source_node.source_type.clone(),
                    confidence: source_node.confidence,
                },
            };
            
            activation_state.node_activations.insert(
                source_node.node_id.clone(),
                node_activation,
            );
        }
        
        activation_state.activation_timestamp = Utc::now();
        activation_state.activation_cycle = 0;
        
        Ok(())
    }
    
    async fn perform_activation_cycle(
        &self,
        cycle: usize,
    ) -> Result<CycleResult, ActivationError> {
        let cycle_start = Instant::now();
        let mut newly_activated_nodes = Vec::new();
        let mut propagation_events = Vec::new();
        
        // Get current activation state
        let current_state = self.activation_state.read().await;
        let active_nodes: Vec<_> = current_state.node_activations
            .iter()
            .filter(|(_, activation)| activation.current_activation > activation.threshold)
            .map(|(node_id, activation)| (node_id.clone(), activation.clone()))
            .collect();
        drop(current_state);
        
        // Process each active node
        for (node_id, node_activation) in active_nodes {
            // Get outgoing relationships
            let relationships = self.get_outgoing_relationships(&node_id).await?;
            
            for relationship in relationships {
                // Calculate propagation strength
                let propagation_strength = self.calculate_propagation_strength(
                    &node_activation,
                    &relationship,
                    cycle,
                ).await?;
                
                if propagation_strength > self.config.minimum_propagation_threshold {
                    // Propagate activation to target node
                    let propagation_event = self.propagate_activation(
                        &node_id,
                        &relationship,
                        propagation_strength,
                        cycle,
                    ).await?;
                    
                    propagation_events.push(propagation_event.clone());
                    
                    // Check if target node is newly activated
                    if propagation_event.newly_activated {
                        newly_activated_nodes.push(propagation_event.target_node.clone());
                    }
                }
            }
        }
        
        // Apply neural processing (competitive inhibition, etc.)
        self.neural_processor.process_activation_cycle(
            &propagation_events,
            cycle,
        ).await?;
        
        // Update cycle counter
        let mut activation_state = self.activation_state.write().await;
        activation_state.activation_cycle = cycle as u64;
        
        Ok(CycleResult {
            cycle_number: cycle,
            newly_activated_nodes,
            propagation_events,
            cycle_duration: cycle_start.elapsed(),
            total_active_nodes: activation_state.node_activations.len(),
        })
    }
    
    async fn calculate_propagation_strength(
        &self,
        source_activation: &NodeActivation,
        relationship: &GraphRelationship,
        cycle: usize,
    ) -> Result<f64, PropagationError> {
        // Base propagation strength from source activation
        let base_strength = source_activation.current_activation;
        
        // Apply relationship weight
        let relationship_weight = self.propagation_rules
            .get_relationship_weight(&relationship.relationship_type)
            .await?;
        
        // Apply distance decay
        let distance_decay = self.decay_function.calculate_distance_decay(
            relationship.distance_from_source.unwrap_or(1),
        );
        
        // Apply temporal decay
        let temporal_decay = self.decay_function.calculate_temporal_decay(cycle);
        
        // Apply semantic compatibility
        let semantic_compatibility = self.calculate_semantic_compatibility(
            source_activation,
            relationship,
        ).await?;
        
        // Apply neural modulation
        let neural_modulation = self.neural_processor.calculate_neural_modulation(
            &source_activation.node_id,
            &relationship.target_node,
            cycle,
        ).await?;
        
        let final_strength = base_strength 
            * relationship_weight 
            * distance_decay 
            * temporal_decay 
            * semantic_compatibility 
            * neural_modulation;
        
        Ok(final_strength.max(0.0).min(1.0))
    }
    
    async fn propagate_activation(
        &self,
        source_node_id: &str,
        relationship: &GraphRelationship,
        propagation_strength: f64,
        cycle: usize,
    ) -> Result<PropagationEvent, PropagationError> {
        let mut activation_state = self.activation_state.write().await;
        let target_node_id = &relationship.target_node;
        
        // Get or create target node activation
        let was_newly_activated = !activation_state.node_activations.contains_key(target_node_id);
        
        let target_activation = activation_state.node_activations
            .entry(target_node_id.clone())
            .or_insert_with(|| NodeActivation {
                node_id: target_node_id.clone(),
                current_activation: 0.0,
                previous_activation: 0.0,
                activation_history: Vec::new(),
                source_activations: HashMap::new(),
                propagated_activation: 0.0,
                threshold: self.config.default_activation_threshold,
                is_source: false,
                activation_metadata: ActivationMetadata::default(),
            });
        
        // Update activation
        target_activation.previous_activation = target_activation.current_activation;
        target_activation.source_activations.insert(
            source_node_id.to_string(),
            propagation_strength,
        );
        
        // Calculate new total activation
        let total_incoming_activation: f64 = target_activation.source_activations.values().sum();
        target_activation.current_activation = (target_activation.current_activation + total_incoming_activation)
            .min(self.config.maximum_activation_level);
        
        // Record activation history
        target_activation.activation_history.push(ActivationRecord {
            timestamp: Utc::now(),
            activation_value: propagation_strength,
            source: ActivationSource::Propagated {
                from_node: source_node_id.to_string(),
                via_relationship: relationship.relationship_id.clone(),
            },
            propagation_path: vec![source_node_id.to_string(), target_node_id.clone()],
            decay_applied: 0.0,
        });
        
        Ok(PropagationEvent {
            source_node: source_node_id.to_string(),
            target_node: target_node_id.clone(),
            relationship_id: relationship.relationship_id.clone(),
            propagation_strength,
            newly_activated: was_newly_activated && target_activation.current_activation > target_activation.threshold,
            cycle,
            timestamp: Utc::now(),
        })
    }
    
    async fn collect_activation_results(
        &self,
        query: &SpreadingActivationQuery,
    ) -> Result<FinalActivationResult, CollectionError> {
        let activation_state = self.activation_state.read().await;
        
        // Collect all nodes with activation above threshold
        let mut activated_nodes: Vec<_> = activation_state.node_activations
            .iter()
            .filter(|(_, activation)| activation.current_activation > activation.threshold)
            .map(|(node_id, activation)| ActivatedNode {
                node_id: node_id.clone(),
                final_activation: activation.current_activation,
                activation_path: self.reconstruct_activation_path(node_id, &activation_state),
                source_contributions: activation.source_activations.clone(),
                is_source: activation.is_source,
                confidence: self.calculate_result_confidence(activation),
            })
            .collect();
        
        // Sort by activation strength
        activated_nodes.sort_by(|a, b| {
            b.final_activation.partial_cmp(&a.final_activation).unwrap()
        });
        
        // Apply result filtering and ranking
        let ranked_nodes = self.apply_result_filtering(&activated_nodes, query).await?;
        
        // Reconstruct activation paths
        let activation_paths = self.reconstruct_all_activation_paths(&activation_state);
        
        Ok(FinalActivationResult {
            ranked_nodes,
            activation_paths,
            convergence_achieved: self.check_final_convergence(&activation_state),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpreadingActivationQuery {
    pub query_terms: Vec<String>,
    pub initial_activation_strength: f64,
    pub activation_threshold: f64,
    pub semantic_similarity_threshold: f64,
    pub max_cycles: usize,
    pub max_results: usize,
    pub property_filters: Option<HashMap<String, serde_json::Value>>,
    pub relationship_constraints: Option<Vec<String>>,
    pub focus_areas: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationResult {
    pub session_id: String,
    pub query: SpreadingActivationQuery,
    pub ranked_nodes: Vec<ActivatedNode>,
    pub activation_paths: Vec<ActivationPath>,
    pub cycles_performed: usize,
    pub convergence_achieved: bool,
    pub total_duration: Duration,
    pub activation_statistics: ActivationStatistics,
}
```

### 2. Implement Neural Processing Components
```rust
// src/inheritance/search/neural_activation_processor.rs
#[derive(Debug)]
pub struct NeuralActivationProcessor {
    competitive_inhibition: Arc<CompetitiveInhibition>,
    spike_timing_processor: Arc<SpikeTimingProcessor>,
    homeostatic_regulator: Arc<HomeostaticRegulator>,
    attention_mechanism: Arc<AttentionMechanism>,
    config: NeuralProcessingConfig,
}

impl NeuralActivationProcessor {
    pub async fn process_activation_cycle(
        &self,
        propagation_events: &[PropagationEvent],
        cycle: usize,
    ) -> Result<(), NeuralProcessingError> {
        // Apply competitive inhibition
        self.competitive_inhibition.apply_inhibition(propagation_events).await?;
        
        // Process spike timing dependent effects
        self.spike_timing_processor.process_spike_timing(propagation_events, cycle).await?;
        
        // Apply homeostatic regulation
        self.homeostatic_regulator.regulate_activation_levels(propagation_events).await?;
        
        // Apply attention mechanisms
        self.attention_mechanism.focus_activation(propagation_events).await?;
        
        Ok(())
    }
    
    pub async fn calculate_neural_modulation(
        &self,
        source_node: &str,
        target_node: &str,
        cycle: usize,
    ) -> Result<f64, ModulationError> {
        // Calculate competitive inhibition factor
        let inhibition_factor = self.competitive_inhibition
            .calculate_inhibition_factor(source_node, target_node)
            .await?;
        
        // Calculate spike timing factor
        let timing_factor = self.spike_timing_processor
            .calculate_timing_factor(source_node, target_node, cycle)
            .await?;
        
        // Calculate attention factor
        let attention_factor = self.attention_mechanism
            .calculate_attention_factor(source_node, target_node)
            .await?;
        
        // Combine factors
        let combined_modulation = inhibition_factor * timing_factor * attention_factor;
        
        Ok(combined_modulation.max(0.0).min(2.0)) // Allow both inhibition and facilitation
    }
}
```

## Acceptance Criteria

### Functional Requirements
- [ ] Multi-cycle spreading activation with configurable parameters
- [ ] Weighted propagation based on relationship types and semantic similarity
- [ ] Neural-inspired competitive inhibition and attention mechanisms
- [ ] Adaptive thresholding and convergence detection
- [ ] Comprehensive result ranking and path reconstruction

### Performance Requirements
- [ ] Activation cycle completion < 100ms for graphs with <10,000 nodes
- [ ] Search result generation < 500ms for typical queries
- [ ] Memory usage scaling linearly with activated node count
- [ ] Parallel processing utilization > 80% for multi-core systems
- [ ] Convergence detection within 10 cycles for 90% of queries

### Testing Requirements
- [ ] Unit tests for activation propagation algorithms
- [ ] Integration tests with knowledge graph systems
- [ ] Performance benchmarks for large-scale activation
- [ ] Accuracy tests against ground truth datasets

## Validation Steps

1. **Test basic spreading activation**:
   ```rust
   let engine = SpreadingActivationEngine::new(config, neural_processor).await?;
   let query = SpreadingActivationQuery::new("artificial intelligence");
   let result = engine.initiate_spreading_activation(query).await?;
   assert!(!result.ranked_nodes.is_empty());
   ```

2. **Test convergence behavior**:
   ```rust
   let result = engine.initiate_spreading_activation(complex_query).await?;
   assert!(result.convergence_achieved);
   assert!(result.cycles_performed <= 10);
   ```

3. **Run spreading activation tests**:
   ```bash
   cargo test spreading_activation_tests --release
   ```

## Files to Create/Modify
- `src/inheritance/search/spreading_activation.rs` - Core spreading activation engine
- `src/inheritance/search/neural_activation_processor.rs` - Neural processing components
- `src/inheritance/search/propagation_rules.rs` - Activation propagation rules
- `src/inheritance/search/decay_functions.rs` - Activation decay algorithms
- `tests/inheritance/spreading_activation_tests.rs` - Spreading activation test suite

## Success Metrics
- Query response time: <500ms for typical searches
- Activation accuracy: >85% relevant results in top 10
- Convergence rate: >90% queries converge within 10 cycles
- Neural processing overhead: <20% of total computation time

## Next Task
Upon completion, proceed to **24_conflict_resolution.md** to implement conflict detection and resolution mechanisms.