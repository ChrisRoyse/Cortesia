# Phase 3: Associative Memory Networks

## Overview
**Duration**: 4 weeks  
**Goal**: Implement human-like associative memory with spreading activation  
**Priority**: HIGH  
**Dependencies**: Phase 1 & 2 completion  

## Week 9: Spreading Activation Framework

### Task 9.1: Activation Network Core
**File**: `src/associative/activation_network.rs` (new file)
```rust
pub struct ActivationNetwork {
    nodes: HashMap<NodeId, ActivationNode>,
    edges: HashMap<NodeId, Vec<Edge>>,
    activation_params: ActivationParams,
}

pub struct ActivationNode {
    id: NodeId,
    concept: String,
    base_activation: f32,
    current_activation: f32,
    decay_rate: f32,
    last_activated: Instant,
}

pub struct ActivationParams {
    spread_rate: f32,           // How much activation spreads
    decay_constant: f32,        // How fast activation decays
    threshold: f32,             // Minimum activation to spread
    max_iterations: usize,      // Prevent infinite loops
    spread_function: SpreadType,
}

impl ActivationNetwork {
    pub fn activate(&mut self, concept: &str, initial_strength: f32) -> ActivationResult {
        let mut activation_queue = BinaryHeap::new();
        let mut visited = HashSet::new();
        let mut activated_concepts = Vec::new();
        
        // Initialize activation
        if let Some(node_id) = self.find_node(concept) {
            self.nodes.get_mut(&node_id).unwrap().current_activation = initial_strength;
            activation_queue.push(ActivationItem { 
                node_id, 
                activation: initial_strength,
                depth: 0 
            });
        }
        
        // Spread activation
        while let Some(item) = activation_queue.pop() {
            if visited.contains(&item.node_id) || item.activation < self.activation_params.threshold {
                continue;
            }
            
            visited.insert(item.node_id);
            activated_concepts.push((
                self.nodes[&item.node_id].concept.clone(),
                item.activation,
                item.depth
            ));
            
            // Spread to neighbors
            if let Some(edges) = self.edges.get(&item.node_id) {
                for edge in edges {
                    let spread_activation = self.calculate_spread(item.activation, &edge);
                    if spread_activation > self.activation_params.threshold {
                        activation_queue.push(ActivationItem {
                            node_id: edge.target,
                            activation: spread_activation,
                            depth: item.depth + 1,
                        });
                    }
                }
            }
        }
        
        ActivationResult {
            activated_concepts,
            total_spread: visited.len(),
            max_depth_reached: activated_concepts.iter().map(|(_, _, d)| *d).max().unwrap_or(0),
        }
    }
    
    fn calculate_spread(&self, source_activation: f32, edge: &Edge) -> f32 {
        match self.activation_params.spread_function {
            SpreadType::Linear => source_activation * edge.weight * self.activation_params.spread_rate,
            SpreadType::Logarithmic => (source_activation * edge.weight).ln() * self.activation_params.spread_rate,
            SpreadType::Sigmoid => sigmoid(source_activation * edge.weight) * self.activation_params.spread_rate,
        }
    }
}
```

### Task 9.2: Dynamic Edge Weighting
**File**: `src/associative/edge_dynamics.rs` (new file)
```rust
pub struct EdgeDynamics {
    learning_rate: f32,
    hebbian_factor: f32,
    normalization: bool,
}

impl EdgeDynamics {
    pub fn update_edge_weight(&mut self, edge: &mut Edge, co_activation: f32) {
        // Hebbian learning: neurons that fire together, wire together
        let delta = self.learning_rate * co_activation * self.hebbian_factor;
        edge.weight = (edge.weight + delta).min(1.0);
        edge.last_strengthened = Instant::now();
        edge.activation_count += 1;
        
        if self.normalization {
            self.normalize_weights(edge.source);
        }
    }
    
    pub fn decay_unused_edges(&mut self, edges: &mut HashMap<NodeId, Vec<Edge>>) {
        for edge_list in edges.values_mut() {
            for edge in edge_list {
                let time_since_use = edge.last_strengthened.elapsed().as_secs_f32();
                edge.weight *= (-time_since_use / DECAY_TIME_CONSTANT).exp();
            }
        }
    }
}
```

### Task 9.3: Context-Sensitive Activation
**File**: `src/associative/contextual_activation.rs` (new file)
```rust
pub struct ContextualActivation {
    context_stack: Vec<Context>,
    context_weights: HashMap<(ContextType, NodeId), f32>,
}

pub struct Context {
    context_type: ContextType,
    active_concepts: HashSet<String>,
    mood: Option<EmotionalState>,
    task: Option<String>,
    timestamp: Instant,
}

impl ContextualActivation {
    pub fn modulate_activation(&self, 
        base_activation: f32, 
        node_id: NodeId, 
        current_context: &Context
    ) -> f32 {
        let mut modulated = base_activation;
        
        // Apply context-specific boosts
        if let Some(weight) = self.context_weights.get(&(current_context.context_type, node_id)) {
            modulated *= weight;
        }
        
        // Boost recently active concepts (recency effect)
        if current_context.active_concepts.contains(&self.get_concept(node_id)) {
            modulated *= RECENCY_BOOST;
        }
        
        // Apply mood congruence
        if let Some(mood) = &current_context.mood {
            modulated *= self.mood_congruence_factor(node_id, mood);
        }
        
        modulated
    }
}
```

## Week 10: Pattern Completion and Priming

### Task 10.1: Autoassociative Memory
**File**: `src/associative/pattern_completion.rs` (new file)
```rust
pub struct PatternCompletion {
    patterns: Vec<MemoryPattern>,
    completion_threshold: f32,
}

pub struct MemoryPattern {
    id: PatternId,
    components: Vec<(NodeId, f32)>,  // concept and typical activation
    frequency: u32,
    last_accessed: Instant,
}

impl PatternCompletion {
    pub fn complete_pattern(&self, partial: &[(String, f32)]) -> Option<CompletePattern> {
        let mut best_match = None;
        let mut best_score = 0.0;
        
        for pattern in &self.patterns {
            let score = self.calculate_match_score(partial, pattern);
            if score > best_score && score > self.completion_threshold {
                best_score = score;
                best_match = Some(pattern);
            }
        }
        
        best_match.map(|pattern| {
            self.generate_completion(partial, pattern)
        })
    }
    
    pub fn learn_pattern(&mut self, activated_concepts: &[(NodeId, f32)]) {
        // Check if this is a novel pattern
        if !self.is_known_pattern(activated_concepts) {
            self.patterns.push(MemoryPattern {
                id: PatternId::new(),
                components: activated_concepts.to_vec(),
                frequency: 1,
                last_accessed: Instant::now(),
            });
        }
    }
}
```

### Task 10.2: Semantic Priming System
**File**: `src/associative/priming_system.rs` (new file)
```rust
pub struct PrimingSystem {
    prime_decay_rate: f32,
    prime_spread_factor: f32,
    active_primes: HashMap<NodeId, PrimeState>,
}

pub struct PrimeState {
    strength: f32,
    source: PrimeSource,
    timestamp: Instant,
}

pub enum PrimeSource {
    Direct,           // Directly activated
    Associative,      // Activated through association
    Contextual,       // Activated by context
    Subliminal,       // Below-threshold activation
}

impl PrimingSystem {
    pub fn apply_prime(&mut self, concept: NodeId, strength: f32, source: PrimeSource) {
        self.active_primes.insert(concept, PrimeState {
            strength,
            source,
            timestamp: Instant::now(),
        });
        
        // Spread prime to related concepts
        self.spread_prime(concept, strength * self.prime_spread_factor);
    }
    
    pub fn get_priming_boost(&self, concept: NodeId) -> f32 {
        if let Some(prime) = self.active_primes.get(&concept) {
            let age = prime.timestamp.elapsed().as_secs_f32();
            prime.strength * (-age * self.prime_decay_rate).exp()
        } else {
            0.0
        }
    }
}
```

### Task 10.3: Tip-of-the-Tongue Retrieval
**File**: `src/associative/tot_retrieval.rs` (new file)
```rust
pub struct TipOfTongueRetrieval {
    phonetic_index: PhoneticIndex,
    semantic_hints: SemanticHintEngine,
}

impl TipOfTongueRetrieval {
    pub fn retrieve_from_partial(&self, hints: PartialMemoryHints) -> Vec<RetrievalCandidate> {
        let mut candidates = Vec::new();
        
        // Phonetic similarity
        if let Some(sounds_like) = &hints.sounds_like {
            candidates.extend(self.phonetic_index.find_similar(sounds_like));
        }
        
        // First letter
        if let Some(starts_with) = &hints.starts_with {
            candidates.extend(self.find_by_prefix(starts_with));
        }
        
        // Semantic features
        if let Some(features) = &hints.semantic_features {
            candidates.extend(self.semantic_hints.find_by_features(features));
        }
        
        // Combine and rank
        self.rank_candidates(candidates, &hints)
    }
}

pub struct PartialMemoryHints {
    sounds_like: Option<String>,
    starts_with: Option<String>,
    syllable_count: Option<usize>,
    semantic_features: Option<Vec<String>>,
    associated_concepts: Option<Vec<String>>,
}
```

## Week 11: Associative Learning

### Task 11.1: Hebbian Learning Implementation
**File**: `src/associative/hebbian_learning.rs` (new file)
```rust
pub struct HebbianLearning {
    learning_rate: f32,
    decay_factor: f32,
    max_weight: f32,
}

impl HebbianLearning {
    pub fn learn_association(&mut self, 
        network: &mut ActivationNetwork,
        concepts: &[(String, f32)],
        time_window: Duration
    ) {
        // Find co-activated concepts within time window
        let co_activations = self.find_coactivations(concepts, time_window);
        
        for (concept1, concept2, strength) in co_activations {
            // Strengthen bidirectional connection
            self.strengthen_connection(network, concept1, concept2, strength);
            
            // Create new connection if doesn't exist
            if !network.has_edge(concept1, concept2) {
                network.add_edge(concept1, concept2, strength * NEW_CONNECTION_WEIGHT);
            }
        }
    }
    
    pub fn consolidate_associations(&mut self, network: &mut ActivationNetwork) {
        // Strengthen frequently used paths
        for (node_id, edges) in network.edges.iter_mut() {
            for edge in edges {
                if edge.activation_count > CONSOLIDATION_THRESHOLD {
                    edge.weight = (edge.weight * CONSOLIDATION_BOOST).min(self.max_weight);
                    edge.consolidated = true;
                }
            }
        }
    }
}
```

### Task 11.2: Inhibitory Connections
**File**: `src/associative/inhibition.rs` (new file)
```rust
pub struct InhibitoryNetwork {
    inhibitory_edges: HashMap<NodeId, Vec<InhibitoryEdge>>,
    lateral_inhibition_strength: f32,
}

pub struct InhibitoryEdge {
    target: NodeId,
    strength: f32,
    edge_type: InhibitionType,
}

pub enum InhibitionType {
    Competitive,      // Mutually exclusive concepts
    Lateral,          // Same-level concepts
    Feedback,         // Top-down inhibition
}

impl InhibitoryNetwork {
    pub fn apply_inhibition(&self, 
        activations: &mut HashMap<NodeId, f32>,
        active_node: NodeId
    ) {
        if let Some(inhibitory_edges) = self.inhibitory_edges.get(&active_node) {
            for edge in inhibitory_edges {
                if let Some(target_activation) = activations.get_mut(&edge.target) {
                    *target_activation *= 1.0 - edge.strength;
                }
            }
        }
    }
}
```

## Week 12: Integration and Advanced Features

### Task 12.1: Memory Consolidation Engine
**File**: `src/associative/consolidation_engine.rs` (new file)
```rust
pub struct ConsolidationEngine {
    replay_buffer: VecDeque<ActivationTrace>,
    consolidation_cycles: usize,
}

impl ConsolidationEngine {
    pub fn run_consolidation_cycle(&mut self, network: &mut ActivationNetwork) {
        // Replay important activation patterns
        let important_traces = self.select_important_traces();
        
        for trace in important_traces {
            // Reactivate pattern with reduced strength
            network.replay_activation_trace(&trace, REPLAY_STRENGTH);
            
            // Strengthen connections in pattern
            self.strengthen_trace_connections(&trace, network);
        }
        
        // Prune weak connections
        self.prune_weak_connections(network);
    }
}
```

### Task 12.2: Associative API Endpoints
**File**: `src/mcp/llm_friendly_server/handlers/associative.rs` (new file)
```rust
pub async fn handle_activate_concept(params: Value) -> Result<Value> {
    let concept = params["concept"].as_str().unwrap();
    let strength = params["strength"].as_f64().unwrap_or(1.0) as f32;
    let include_path = params["include_activation_path"].as_bool().unwrap_or(false);
    
    let result = ACTIVATION_NETWORK.lock().unwrap().activate(concept, strength);
    
    Ok(json!({
        "concept": concept,
        "activated_concepts": result.activated_concepts,
        "total_activated": result.total_spread,
        "max_depth": result.max_depth_reached,
        "activation_path": if include_path { Some(result.path) } else { None }
    }))
}

pub async fn handle_find_associations(params: Value) -> Result<Value> {
    let concepts = params["concepts"].as_array().unwrap();
    let max_distance = params["max_distance"].as_u64().unwrap_or(3) as usize;
    
    let associations = find_concept_associations(concepts, max_distance);
    
    Ok(json!({
        "query_concepts": concepts,
        "associations": associations,
        "connection_strength": calculate_association_strength(associations)
    }))
}
```

### Task 12.3: Performance Optimization
```rust
// Optimizations:
1. Parallel activation spreading
2. Sparse matrix representation for edges
3. Activation caching for common queries
4. Batch updates for learning
5. GPU acceleration for large networks
```

### Task 12.4: Testing Suite
**File**: `tests/associative_memory_tests.rs`
```rust
#[test]
fn test_spreading_activation() {
    let mut network = create_test_network();
    network.add_edge("dog", "cat", 0.8);
    network.add_edge("cat", "mouse", 0.7);
    
    let result = network.activate("dog", 1.0);
    
    assert!(result.contains_concept("cat"));
    assert!(result.contains_concept("mouse"));
    assert!(result.get_activation("cat") > result.get_activation("mouse"));
}

#[test]
fn test_context_modulation() {
    // Test that context affects activation spreading
}

#[test]
fn test_pattern_completion() {
    // Test partial pattern completion
}
```

## Deliverables
1. **Spreading activation network** with configurable parameters
2. **Pattern completion** system for partial memories
3. **Priming effects** implementation
4. **Hebbian learning** for association strengthening
5. **Context-sensitive** activation modulation
6. **Consolidation engine** for memory organization

## Success Criteria
- [ ] Activation spreads correctly through network
- [ ] Pattern completion accuracy > 85%
- [ ] Priming effects measurable and consistent
- [ ] Context modulation shows 20%+ effect
- [ ] Learning strengthens co-activated concepts
- [ ] Performance < 50ms for typical activation

## Dependencies
- Graph processing library
- Parallel computation framework
- Phonetic matching library

## Risks & Mitigations
1. **Activation explosion**
   - Mitigation: Decay rates, activation limits
2. **Memory requirements for large networks**
   - Mitigation: Sparse representations, pruning
3. **Context switching overhead**
   - Mitigation: Context caching, fast switching