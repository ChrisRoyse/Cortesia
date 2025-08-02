# Directory Overview: Cognitive Module

## 1. High-Level Summary

The cognitive module implements a sophisticated, brain-inspired AI system with multiple thinking modes modeled after human cognitive processes. It provides a comprehensive framework for intelligent query processing, pattern recognition, memory management, and attention control. The module enables AI systems to engage in complex reasoning through various cognitive patterns including convergent, divergent, lateral, critical, abstract, systems, and adaptive thinking.

## 2. Tech Stack

*   **Languages:** Rust
*   **Frameworks:** 
    *   `tokio` - Async runtime
    *   `async-trait` - Async trait definitions
*   **Libraries:** 
    *   `ahash` - High-performance hashing
    *   `uuid` - Unique identifier generation
    *   `slotmap` - Entity management
    *   `serde` - Serialization/deserialization
    *   `rayon` - Parallel processing
*   **Internal Dependencies:**
    *   `crate::core::*` - Core graph and activation systems
    *   `crate::error::*` - Error handling
    *   `crate::monitoring::*` - Performance monitoring
    *   `crate::learning::*` - Learning components

## 3. Directory Structure

*   **Root Files:** Core cognitive patterns and infrastructure
*   **`divergent/`**: Specialized divergent thinking implementation
*   **`inhibitory/`**: Competitive inhibition and attention control
*   **`memory_integration/`**: Unified memory architecture

## 4. File Breakdown

### `mod.rs`

*   **Purpose:** Module declaration and public API exports
*   **Key Exports:**
    *   All cognitive pattern types
    *   Type definitions from `types.rs`
    *   Core components: `CognitiveOrchestrator`, `AttentionManager`, `WorkingMemorySystem`, etc.

### `types.rs`

*   **Purpose:** Core type definitions for the cognitive system
*   **Classes:**
    *   `QueryContext`
        *   **Description:** Context information for query processing
        *   **Fields:** `domain`, `confidence_threshold`, `max_depth`, `reasoning_trace`, etc.
    *   `PatternResult`
        *   **Description:** Generic result from cognitive pattern execution
        *   **Fields:** `pattern_type`, `answer`, `confidence`, `reasoning_trace`, `metadata`
    *   `ExplorationMap`
        *   **Description:** Track divergent exploration paths
        *   **Methods:**
            *   `add_node(entity_key, activation, depth)`: Add explored node
            *   `add_edge(from, to, relation_type, weight)`: Add exploration edge
            *   `get_high_activation_endpoints(count)`: Get most activated nodes
*   **Enums:**
    *   `CognitivePatternType`: Convergent, Divergent, Lateral, Systems, Critical, Abstract, Adaptive, ChainOfThought, TreeOfThoughts
    *   `AttentionType`: Selective, Divided, Sustained, Executive, Alternating
    *   `BufferType`: Phonological, Episodic, Visuospatial, Central
    *   `QueryIntent`: Factual, Relational, Causal, Comparative, Hypothetical, Procedural
*   **Traits:**
    *   `CognitivePattern`: Core trait for all cognitive patterns
        *   **Methods:** `execute()`, `get_pattern_type()`, `get_optimal_use_cases()`, `estimate_complexity()`

### `orchestrator.rs`

*   **Purpose:** Central coordinator for all cognitive patterns
*   **Classes:**
    *   `CognitiveOrchestrator`
        *   **Description:** Main orchestration system
        *   **Fields:** `patterns`, `adaptive_selector`, `performance_monitor`, `brain_graph`, `config`
        *   **Methods:**
            *   `new(brain_graph, config)`: Initialize orchestrator
            *   `execute_reasoning(query, context, strategy)`: Execute reasoning with strategy
            *   `execute_adaptive_reasoning(query, context)`: Adaptive pattern selection
            *   `get_statistics()`: Performance metrics
    *   `CognitiveOrchestratorConfig`
        *   **Description:** Configuration settings
        *   **Fields:** `enable_adaptive_selection`, `enable_ensemble_methods`, `default_timeout_ms`

### `convergent.rs`

*   **Purpose:** Focused, goal-directed thinking pattern
*   **Classes:**
    *   `ConvergentThinking`
        *   **Description:** Narrows down possibilities to find best answer
        *   **Methods:**
            *   `execute(query, context, parameters)`: Main convergent analysis
            *   `propagate_activation(seed_entities, context)`: Activation spreading
            *   `extract_best_answer(propagation_result)`: Extract final answer

### `divergent.rs`

*   **Purpose:** Expansive, creative thinking pattern
*   **Classes:**
    *   `DivergentThinking`
        *   **Description:** Generates multiple possibilities and alternatives
        *   **Methods:**
            *   `execute(query, context, parameters)`: Main divergent exploration
            *   `execute_divergent_exploration(concepts, parameters)`: Core exploration
            *   `activate_seed_concept(concept)`: Initial activation
            *   `spread_activation(entity_key, current_activation, wave_id)`: Activation propagation
            *   `graph_path_exploration(start_key, target_key, max_depth)`: Path finding

### `lateral.rs`

*   **Purpose:** Non-linear, creative problem-solving
*   **Classes:**
    *   `LateralThinking`
        *   **Description:** Finds unconventional connections between concepts
        *   **Methods:**
            *   `execute(query, context, parameters)`: Main lateral thinking
            *   `find_lateral_bridges(concepts, parameters)`: Bridge identification
            *   `analyze_novelty(bridges)`: Novelty assessment

### `systems.rs`

*   **Purpose:** Holistic system-level thinking
*   **Classes:**
    *   `SystemsThinking`
        *   **Description:** Analyzes complex interconnected systems
        *   **Methods:**
            *   `execute(query, context, parameters)`: System analysis
            *   `traverse_hierarchy(entity_key, reasoning_type)`: Hierarchy traversal
            *   `handle_exceptions(exceptions)`: Exception management

### `critical.rs`

*   **Purpose:** Analytical evaluation and judgment
*   **Classes:**
    *   `CriticalThinking`
        *   **Description:** Critical analysis and contradiction resolution
        *   **Methods:**
            *   `execute(query, context, parameters)`: Critical evaluation
            *   `find_contradictions(facts)`: Contradiction detection
            *   `resolve_contradictions(contradictions, strategy)`: Conflict resolution

### `abstract_pattern.rs`

*   **Purpose:** High-level pattern detection and abstraction
*   **Classes:**
    *   `AbstractThinking`
        *   **Description:** Identifies abstract patterns and refactoring opportunities
        *   **Methods:**
            *   `execute(query, context, parameters)`: Pattern abstraction
            *   `detect_patterns(scope, pattern_type)`: Pattern detection
            *   `identify_abstractions(patterns)`: Abstraction identification
            *   `find_refactoring_opportunities(patterns)`: Optimization identification

### `adaptive.rs`

*   **Purpose:** Dynamic strategy adaptation based on context
*   **Classes:**
    *   `AdaptiveThinking`
        *   **Description:** Selects and combines cognitive patterns adaptively
        *   **Methods:**
            *   `execute(query, context, parameters)`: Adaptive execution
            *   `analyze_query_characteristics(query)`: Query analysis
            *   `select_strategy(characteristics)`: Strategy selection
            *   `combine_pattern_results(results)`: Result combination

### `working_memory.rs`

*   **Purpose:** Short-term memory management system
*   **Classes:**
    *   `WorkingMemorySystem`
        *   **Description:** Multi-buffer working memory
        *   **Fields:** `buffers`, `central_executive`, `activation_engine`, `sdr_storage`
        *   **Methods:**
            *   `store_in_working_memory(content, buffer_type, importance)`: Store items
            *   `retrieve_from_working_memory(query, buffer_types)`: Query retrieval
            *   `get_attention_relevant_items(focus)`: Attention-based retrieval
            *   `maintain_buffer_limits()`: Capacity management
    *   `MemoryItem`
        *   **Description:** Individual memory item
        *   **Fields:** `content`, `importance`, `timestamp`, `access_count`, `decay_factor`

### `attention_manager.rs`

*   **Purpose:** Sophisticated attention control system
*   **Classes:**
    *   `AttentionManager`
        *   **Description:** Manages different types of attention
        *   **Fields:** `current_state`, `attention_history`, `cognitive_load_threshold`
        *   **Methods:**
            *   `focus_attention(targets, attention_type)`: Direct attention
            *   `shift_attention(new_targets, transition_type)`: Attention transition
            *   `manage_divided_attention(targets, weights)`: Multi-target attention
            *   `executive_control(command)`: High-level control
            *   `coordinate_with_cognitive_patterns(pattern_type)`: Pattern coordination
    *   `AttentionState`
        *   **Description:** Current attention configuration
        *   **Fields:** `active_focuses`, `cognitive_load`, `executive_state`

### `pattern_detector.rs`

*   **Purpose:** Advanced multi-dimensional pattern detection
*   **Classes:**
    *   `PatternDetector`
        *   **Description:** Detects various pattern types
        *   **Methods:**
            *   `detect_patterns(pattern_type, parameters)`: Main detection
            *   `detect_structural_patterns(min_frequency)`: Graph topology patterns
            *   `detect_temporal_patterns(time_window)`: Time-based patterns
            *   `detect_semantic_patterns(similarity_threshold)`: Concept clustering
            *   `detect_usage_patterns(min_frequency)`: Access patterns

### `basic_query.rs`

*   **Purpose:** Natural language query understanding
*   **Classes:**
    *   `BasicQueryProcessor`
        *   **Description:** Query parsing and intent extraction
        *   **Methods:**
            *   `understand_query(query)`: Main query processing
            *   `identify_intent(query)`: Intent classification
            *   `extract_concepts(query)`: Concept extraction
            *   `extract_relationships(query)`: Relationship identification
    *   `QueryUnderstanding`
        *   **Description:** Parsed query representation
        *   **Fields:** `intent`, `concepts`, `relationships`, `constraints`

### `graph_query_engine.rs`

*   **Purpose:** Graph-based query execution
*   **Classes:**
    *   `GraphQueryEngine`
        *   **Description:** Execute complex graph queries
        *   **Methods:**
            *   `execute_query(query, params)`: Main query execution
            *   `find_patterns(pattern_type, constraints)`: Pattern search
            *   `find_paths(start, end, constraints)`: Path finding
            *   `analyze_structure(scope)`: Structure analysis

### `tuned_parameters.rs`

*   **Purpose:** Optimized parameter configurations
*   **Constants:**
    *   Default activation thresholds
    *   Exploration breadth limits
    *   Confidence thresholds
    *   Time limits

### `attention_manager_traits.rs`

*   **Purpose:** Trait definitions for attention management
*   **Traits:**
    *   Core attention interfaces
    *   Extension traits

### `inhibitory_logic.rs`

*   **Purpose:** Basic inhibitory logic utilities
*   **Functions:**
    *   Inhibition calculation helpers

### `convergent_enhanced.rs`

*   **Purpose:** Enhanced convergent thinking implementation
*   **Classes:**
    *   `EnhancedConvergentThinking`
        *   **Description:** Advanced convergent analysis

## 5. Subdirectory Details

### `divergent/`

*   **`core_engine.rs`**
    *   **Purpose:** Core divergent exploration engine
    *   **Functions:**
        *   `execute_divergent_exploration()`: Main exploration process
        *   `activate_seed_concept()`: Initial concept activation
        *   `spread_activation()`: Activation spreading algorithm
        *   `graph_path_exploration()`: Advanced path exploration
*   **`constants.rs`**
    *   **Purpose:** Configuration constants for divergent thinking
    *   **Constants:** Activation thresholds, decay rates, exploration limits
*   **`utils.rs`**
    *   **Purpose:** Utility functions for divergent processing
    *   **Functions:** Scoring calculations, path utilities

### `inhibitory/`

*   **`mod.rs`**
    *   **Purpose:** Module exports and coordination
*   **`competition.rs`**
    *   **Purpose:** Competitive inhibition strategies
    *   **Classes:**
        *   `CompetitionGroup`: Group competition management
        *   **Functions:** `apply_group_competition()`, `calculate_inhibition_weights()`
*   **`exceptions.rs`**
    *   **Purpose:** Exception handling in inhibitory processing
    *   **Classes:**
        *   `InhibitionException`: Exception types
        *   **Functions:** `handle_inhibition_exceptions()`, `resolve_conflicts()`
*   **`dynamics.rs`**
    *   **Purpose:** Dynamic inhibition adjustments
*   **`hierarchical.rs`**
    *   **Purpose:** Hierarchical inhibition patterns
*   **`integration.rs`**
    *   **Purpose:** Integration with other cognitive systems
*   **`learning.rs`**
    *   **Purpose:** Learning inhibitory patterns
*   **`matrix.rs`**
    *   **Purpose:** Inhibition matrix calculations
*   **`metrics.rs`**
    *   **Purpose:** Performance metrics for inhibition
*   **`types.rs`**
    *   **Purpose:** Type definitions for inhibitory system

### `memory_integration/`

*   **`mod.rs`**
    *   **Purpose:** Module coordination and exports
*   **`system.rs`**
    *   **Purpose:** Main unified memory system
    *   **Classes:**
        *   `UnifiedMemorySystem`: Central memory integration
        *   **Methods:** `store_information()`, `retrieve_information()`, `consolidate_memories()`
*   **`coordinator.rs`**
    *   **Purpose:** Cross-memory coordination
    *   **Classes:**
        *   `MemoryCoordinator`: Coordinates different memory types
*   **`retrieval.rs`**
    *   **Purpose:** Integrated retrieval mechanisms
    *   **Functions:** `retrieve_across_memories()`, `rank_retrieval_results()`
*   **`consolidation.rs`**
    *   **Purpose:** Memory consolidation processes
    *   **Functions:** `consolidate_short_to_long_term()`, `strengthen_memory_traces()`
*   **`hierarchy.rs`**
    *   **Purpose:** Hierarchical memory organization
*   **`types.rs`**
    *   **Purpose:** Type definitions for memory integration

## 6. Key Algorithms and Logic

### Activation Spreading
The cognitive module uses activation spreading algorithms throughout:
1. Initial concept activation in seed nodes
2. Propagation through graph edges with decay
3. Accumulation of activation in target nodes
4. Threshold-based termination

### Attention Management
Multi-level attention control:
1. **Selective Attention**: Focus on specific targets
2. **Divided Attention**: Distribute across multiple targets
3. **Sustained Attention**: Maintain focus over time
4. **Executive Control**: High-level attention coordination

### Memory Hierarchy
Three-tier memory system:
1. **Working Memory**: Short-term, limited capacity buffers
2. **SDR Storage**: Pattern-based intermediate storage
3. **Knowledge Graph**: Long-term structured knowledge

### Inhibitory Control
Competitive inhibition prevents conflicting activations:
1. Group competition for resources
2. Lateral inhibition between competing concepts
3. Exception handling for special cases
4. Learning-based adaptation

## 7. Dependencies

### Internal Dependencies
*   **`crate::core::brain_enhanced_graph`**: Core knowledge graph
*   **`crate::core::activation_engine`**: Activation processing
*   **`crate::core::sdr_storage`**: Sparse distributed representations
*   **`crate::core::types`**: Core type definitions
*   **`crate::learning`**: Learning components
*   **`crate::monitoring`**: Performance monitoring
*   **`crate::error`**: Error handling

### External Dependencies
*   **`tokio`**: Async runtime for concurrent processing
*   **`async-trait`**: Async trait implementations
*   **`ahash`**: High-performance hashing
*   **`uuid`**: Unique identifiers
*   **`serde`**: Serialization support
*   **`rayon`**: Parallel processing

## 8. API Integration Points

### Cognitive Pattern Interface
All patterns implement the `CognitivePattern` trait:
```rust
async fn execute(
    &self,
    query: &str,
    context: Option<&str>,
    parameters: PatternParameters,
) -> Result<PatternResult>;
```

### Orchestrator API
Main entry point through `CognitiveOrchestrator`:
```rust
pub async fn execute_reasoning(
    &self,
    query: &str,
    context: Option<&str>,
    strategy: ReasoningStrategy,
) -> Result<ReasoningResult>
```

### Memory Integration API
Unified memory access:
```rust
pub async fn store_information(
    &self,
    content: &str,
    memory_type: MemoryType,
) -> Result<()>

pub async fn retrieve_information(
    &self,
    query: &str,
    strategy: RetrievalStrategy,
) -> Result<UnifiedRetrievalResult>
```

## 9. Design Patterns

### Strategy Pattern
- Cognitive patterns as interchangeable strategies
- Runtime pattern selection based on query characteristics

### Observer Pattern
- Statistics tracking and performance monitoring
- Event-based attention management

### Builder Pattern
- Configuration builders for complex parameters
- Fluent interfaces for pattern configuration

### State Pattern
- Exploration state tracking in divergent thinking
- Attention state management

### Decorator Pattern
- Enhanced versions of cognitive patterns
- Layered functionality additions

## 10. Performance Considerations

### Concurrency
- Async/await for non-blocking operations
- Parallel pattern execution in ensemble methods
- Lock-free data structures where possible

### Memory Management
- Buffer size limits in working memory
- Automatic garbage collection of expired items
- Efficient entity key representations

### Optimization Strategies
- Activation threshold cutoffs
- Early termination conditions
- Caching of frequently accessed paths
- Batch processing of related queries

## 11. Extension Points

### Adding New Cognitive Patterns
1. Implement the `CognitivePattern` trait
2. Register in `CognitiveOrchestrator`
3. Add to `CognitivePatternType` enum
4. Update adaptive selection logic

### Custom Attention Types
1. Extend `AttentionType` enum
2. Implement attention behavior in `AttentionManager`
3. Add coordination logic

### Memory Buffer Types
1. Add to `BufferType` enum
2. Implement buffer-specific storage logic
3. Update retrieval strategies

### Inhibitory Strategies
1. Create new competition group types
2. Implement inhibition calculation
3. Add to exception handling