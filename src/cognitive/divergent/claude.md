# Directory Overview: Divergent Thinking Module

## 1. High-Level Summary

The divergent thinking module implements a creative exploration pattern for the LLMKG (LLM Knowledge Graph) system. This cognitive pattern specializes in brainstorming, finding creative connections, discovering associations, and generating multiple alternative solutions. It explores broadly through a knowledge graph, identifying novel pathways and unexpected connections between concepts.

The module is designed to answer questions like:
- "What are examples of X?"
- "Find creative uses for Y"
- "What's related to Z?"
- "Brainstorm ideas about..."

## 2. Tech Stack

- **Language:** Rust
- **Async Runtime:** Tokio (via async_trait)
- **Key Dependencies:**
  - `crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph` - Core graph structure
  - `crate::core::brain_types` - Brain-inspired entity types and operations
  - `crate::cognitive::types` - Cognitive pattern definitions
  - `crate::error` - Error handling
- **Patterns:** Async/await, Arc for shared ownership, trait implementations

## 3. Directory Structure

```
./src/cognitive/divergent/
├── constants.rs      # Static configuration and data structures
├── core_engine.rs    # Alternative implementation with different type system
└── utils.rs          # Utility functions for the core engine
```

Note: The main implementation is in `../divergent.rs` at the parent level, while this directory contains an alternative engine implementation.

## 4. File Breakdown

### `constants.rs`

**Purpose:** Defines static configuration values and reference data structures for divergent thinking operations.

**Key Constants:**
- `DEFAULT_EXPLORATION_BREADTH: usize = 20` - Maximum nodes to explore at each depth
- `DEFAULT_CREATIVITY_THRESHOLD: f32 = 0.3` - Minimum creativity score to keep paths
- `DEFAULT_MAX_PATH_LENGTH: usize = 5` - Maximum exploration depth
- `DEFAULT_NOVELTY_WEIGHT: f32 = 0.4` - Weight for novelty in scoring
- `DEFAULT_RELEVANCE_WEIGHT: f32 = 0.6` - Weight for relevance in scoring
- `DEFAULT_ACTIVATION_DECAY: f32 = 0.9` - Activation reduction per hop
- `DEFAULT_MIN_ACTIVATION: f32 = 0.1` - Minimum activation to continue
- `DEFAULT_MAX_RESULTS: usize = 50` - Maximum results to return

**Key Functions:**
- `get_domain_hierarchy() -> HashMap<String, Vec<String>>` - Hierarchical domain relationships
- `get_semantic_fields() -> HashMap<String, Vec<String>>` - Semantic field groupings
- `get_domain_patterns() -> HashMap<String, Vec<String>>` - Domain-specific pattern keywords
- `get_stop_words() -> Vec<String>` - Common words to filter in queries
- `get_exploration_type_keywords() -> HashMap<String, Vec<String>>` - Keywords for exploration type detection
- `get_creativity_boost_factors() -> HashMap<String, f32>` - Creativity multipliers
- `get_relevance_weights() -> HashMap<String, f32>` - Component weights for relevance
- `get_novelty_parameters() -> HashMap<String, f32>` - Novelty calculation parameters

### `core_engine.rs`

**Purpose:** Alternative implementation of the divergent thinking processor using a different type system focused on detailed exploration tracking.

**Classes:**
- `DivergentThinking`
  - **Properties:**
    - `exploration_breadth: usize` - Width of exploration
    - `creativity_threshold: f32` - Minimum creativity score
    - `max_path_length: usize` - Maximum depth
    - `novelty_weight: f32` - Novelty importance
    - `relevance_weight: f32` - Relevance importance
    - `activation_decay: f32` - Decay factor
    - `min_activation: f32` - Cutoff threshold
    - `max_results: usize` - Result limit
  
  - **Methods:**
    - `new() -> Self` - Create with defaults
    - `new_with_params(...) -> Self` - Create with custom parameters
    - `execute_divergent_exploration(&self, graph, context) -> Result<ExplorationMap>` - Main execution
    - `activate_seed_concept(&self, graph, concept, state) -> Result<Vec<EntityKey>>` - Initialize exploration
    - `spread_activation(&self, graph, entities, state) -> Result<Vec<ExplorationPath>>` - Propagate activation
    - `graph_path_exploration(&self, graph, paths, type, state) -> Result<Vec<ExplorationPath>>` - Enhance paths
    - `find_typed_connections(&self, paths, type) -> Vec<ExplorationEdge>` - Extract connections
    - `calculate_exploration_statistics(&self, map, state) -> ExplorationStatistics` - Compute metrics

**Trait Implementations:**
- `CognitivePattern` - Standard cognitive pattern interface
- `Default` - Default parameter construction

### `utils.rs`

**Purpose:** Utility functions supporting the core engine implementation.

**Functions:**
- `find_concept_entities(graph, concept) -> Result<Vec<EntityKey>>`
  - Finds entities matching a concept using embedding similarity
  
- `generate_concept_embedding(concept) -> Vec<f32>`
  - Creates a 96-dimensional embedding vector for a concept
  - Uses simplified hash-based generation (placeholder for real embeddings)
  
- `simple_hash(s) -> u32`
  - Basic string hashing function
  
- `calculate_path_creativity(graph, path, type) -> Result<f32>`
  - Scores path creativity based on depth, activation, weights
  
- `calculate_path_novelty(graph, path) -> Result<f32>`
  - Scores path novelty based on length, connection strength
  
- `calculate_path_relevance(graph, path, type) -> Result<f32>`
  - Scores path relevance based on activation, connection strength
  
- `infer_exploration_type(query) -> String`
  - Detects exploration type from query keywords
  - Returns: "creative", "analytical", "associative", or "exploratory"
  
- `extract_seed_concept(query) -> String`
  - Extracts the main concept from a query by filtering stop words

### `../divergent.rs` (Main Implementation)

**Purpose:** Primary implementation of divergent thinking pattern with full feature set.

**Classes:**
- `DivergentThinking`
  - **Properties:**
    - `graph: Arc<BrainEnhancedKnowledgeGraph>` - Shared graph reference
    - `exploration_breadth: usize` - Exploration width (default: 20)
    - `creativity_threshold: f32` - Creativity cutoff (default: 0.3)
    - `max_exploration_depth: usize` - Max depth (default: 4)
    - `novelty_weight: f32` - Novelty importance (default: 0.4)
  
  - **Core Methods:**
    - `execute_divergent_exploration(seed_concept, exploration_type) -> Result<DivergentResult>`
      - Main entry point for divergent exploration
      - Steps: activate seed → spread activation → path exploration → rank results
    
    - `activate_seed_concept(concept) -> Result<ActivationPattern>`
      - Finds and activates entities matching the seed concept
    
    - `spread_activation(seed_activation, exploration_type) -> Result<ExplorationMap>`
      - Propagates activation through graph with breadth-first exploration
      - Respects creativity threshold and max depth
    
    - `graph_path_exploration(exploration_map) -> Result<Vec<ExplorationPath>>`
      - Generates creative paths between seed and endpoint nodes
    
    - `find_creative_path(start, end, exploration_map) -> Result<Option<ExplorationPath>>`
      - Breadth-first search for paths with creativity scoring
    
    - `rank_by_creativity(paths) -> Result<Vec<ExplorationPath>>`
      - Sorts paths by combined relevance/novelty score

  - **Relevance Calculation Methods:**
    - `calculate_concept_relevance(entity_concept, query_concept) -> f32`
      - Multi-layer relevance: exact match → substring → hierarchical → semantic → lexical → domain
    
    - `calculate_hierarchical_relevance(entity, query) -> f32`
      - Scores based on position in domain hierarchies
    
    - `calculate_semantic_relevance(entity, query) -> f32`
      - Scores based on shared semantic fields
    
    - `calculate_lexical_similarity(entity, query) -> f32`
      - Jaccard similarity and edit distance

**Helper Functions:**
- `calculate_concept_similarity(concept_a, concept_b) -> f32` - Semantic similarity score
- `infer_exploration_type(query) -> ExplorationType` - Detect exploration type from query
- `extract_seed_concept(query) -> Result<String>` - Extract main concept from natural language

## 5. Key Variables and Logic

### Exploration Types
- `ExplorationType::Instances` - Find examples of a concept
- `ExplorationType::Categories` - Find types/categories
- `ExplorationType::Properties` - Find attributes
- `ExplorationType::Associations` - Find related concepts
- `ExplorationType::Creative` - Brainstorm creative connections

### Core Data Structures

**ActivationPattern**
- Tracks activation levels for entities during exploration
- Key-value map of EntityKey → activation strength

**ExplorationMap**
- Complete exploration state including:
  - Starting entities
  - Exploration waves (entities at each depth)
  - Edges between entities
  - Statistics (total explored, depth, etc.)

**ExplorationPath**
- Single creative path through the graph:
  - `path: Vec<EntityKey>` - Sequence of entities
  - `concepts: Vec<String>` - Concept names
  - `relevance_score: f32` - How relevant to query
  - `novelty_score: f32` - How creative/unexpected

**DivergentResult**
- Final output containing:
  - Ranked exploration paths
  - Creativity scores
  - Total paths explored

### Scoring Algorithm

The system uses a weighted combination of relevance and novelty:
```
final_score = (1.0 - novelty_weight) * relevance + novelty_weight * novelty
```

Paths are ranked by this combined score and limited to `exploration_breadth` results.

## 6. Dependencies

### Internal Dependencies
- `crate::cognitive::types` - Core cognitive pattern types and traits
- `crate::core::brain_enhanced_graph` - Graph data structure and operations
- `crate::core::brain_types` - Entity types, activation patterns, relations
- `crate::core::types` - Base types like EntityKey
- `crate::error` - Error handling (Result, GraphError)

### External Dependencies
- `std::collections::{HashMap, HashSet}` - Data structures
- `std::sync::Arc` - Thread-safe reference counting
- `std::time::{SystemTime, Instant}` - Performance timing
- `async_trait` - Async trait definitions

## 7. Key Algorithms

### Activation Spreading
1. Start with seed entities at full activation (1.0)
2. For each depth level:
   - Find neighbors of current wave entities
   - Calculate neighbor activation = current × weight × decay
   - Add neighbors above threshold to next wave
   - Limit wave size to exploration_breadth
3. Continue until max_depth or no new activations

### Creative Path Finding
1. Use BFS between seed and high-activation endpoints
2. Track parent relationships for path reconstruction
3. Score paths based on:
   - Relevance: entity properties and activation levels
   - Novelty: path length, concept diversity, domain spanning
   - Creativity: embedding variance, connection rarity

### Concept Matching
Multi-level matching strategy:
1. Exact match (score 1.0)
2. Substring containment (score 0.8-0.9)
3. Hierarchical relationships (score 0.6-0.9)
4. Semantic field overlap (score based on overlap)
5. Lexical similarity (Jaccard/edit distance)
6. Domain pattern matching (score 0.8-0.9)

## 8. Usage Example

```rust
// Create divergent thinking processor
let graph = Arc::new(BrainEnhancedKnowledgeGraph::new());
let divergent = DivergentThinking::new(graph);

// Execute exploration
let result = divergent.execute_divergent_exploration(
    "machine learning",
    ExplorationType::Creative
).await?;

// Access results
for path in result.explorations {
    println!("Found creative connection: {} (novelty: {})", 
             path.concept, path.novelty_score);
}
```

## 9. Performance Considerations

- Exploration is limited by `exploration_breadth` at each level
- Activation decay prevents infinite spreading
- Results are capped at `max_results`
- Graph operations are the primary bottleneck
- Consider caching for repeated concept searches

## 10. Future Enhancements

1. Real embedding models instead of hash-based embeddings
2. Learned relevance/novelty weights
3. Query-specific parameter tuning
4. Path explanation generation
5. Integration with other cognitive patterns
6. Persistent creativity score learning