# Task 1: Remove Neural Server Dependencies

## Overview
The current Phase 3 implementation incorrectly assumes LLMKG makes LLM API calls. This task removes all neural/LLM dependencies and replaces them with proper graph operations.

## Files to Modify

### 1. Remove/Rename `NeuralProcessingServer` trait
**File**: `src/cognitive/neural_query.rs`
**Action**: Rename or repurpose this file

```rust
// REMOVE this trait:
pub trait NeuralProcessingServer {
    async fn neural_query(...);
    async fn neural_embed(...);
    // etc.
}

// REPLACE with graph operations:
pub trait GraphQueryEngine {
    async fn find_patterns(&self, query: GraphQuery) -> Result<Vec<Pattern>>;
    async fn traverse_paths(&self, start: EntityKey, params: TraversalParams) -> Result<Vec<Path>>;
    async fn analyze_structure(&self, subgraph: &Subgraph) -> Result<StructureAnalysis>;
    async fn compute_similarity(&self, entity_a: EntityKey, entity_b: EntityKey) -> Result<f32>;
}
```

### 2. Update Cognitive Patterns
Each cognitive pattern file needs to remove LLM calls and implement graph algorithms instead.

**Example - Convergent Thinking** (`src/cognitive/convergent.rs`):

```rust
// REMOVE:
async fn synthesize_with_llm(&self, themes: Vec<Theme>) -> Result<String> {
    let prompt = format!("Synthesize these themes: {:?}", themes);
    self.neural_server.neural_complete(prompt).await
}

// REPLACE with:
async fn synthesize_patterns(&self, themes: Vec<Theme>) -> Result<Synthesis> {
    // Find common patterns in graph
    let common_nodes = self.find_intersection_nodes(&themes)?;
    let shared_relationships = self.analyze_shared_edges(&themes)?;
    let convergence_point = self.calculate_convergence(&common_nodes, &shared_relationships)?;
    
    Ok(Synthesis {
        central_concept: convergence_point,
        supporting_patterns: common_nodes,
        confidence: self.calculate_confidence(&themes),
    })
}
```

### 3. Update Working Memory
**File**: `src/cognitive/working_memory.rs`

Remove any neural processing and implement as graph state:
```rust
// REMOVE:
let embedded = self.neural_server.neural_embed(item.content).await?;

// REPLACE with:
let graph_representation = self.graph.get_entity_subgraph(item.entity_key)?;
let activation_state = self.calculate_activation_state(&graph_representation)?;
```

## Implementation Steps

### Step 1: Audit All Files
Search for and document all occurrences of:
- `neural_`
- `NeuralProcessingServer`
- `llm`
- `embed`
- External API calls

### Step 2: Design Graph Operations
For each removed neural operation, design equivalent graph operation:

| Neural Operation | Graph Operation |
|-----------------|-----------------|
| neural_embed() | compute_entity_vector() using graph structure |
| neural_reason() | traverse_reasoning_path() through relationships |
| neural_complete() | generate_from_patterns() based on graph context |
| neural_classify() | classify_by_graph_topology() |

### Step 3: Implement Graph Algorithms
Create pure graph algorithms for each cognitive pattern:

```rust
// Example: Divergent thinking as graph expansion
pub async fn divergent_exploration(&self, start: EntityKey) -> Result<DivergentResult> {
    let mut explored = HashSet::new();
    let mut frontier = vec![start];
    let mut ideas = Vec::new();
    
    while !frontier.is_empty() && ideas.len() < self.max_ideas {
        let current = frontier.pop().unwrap();
        
        // Explore all relationships from current node
        let relationships = self.graph.get_relationships(current)?;
        
        for rel in relationships {
            if !explored.contains(&rel.target) {
                // Generate idea from this connection
                let idea = self.generate_idea_from_relationship(&rel)?;
                ideas.push(idea);
                
                frontier.push(rel.target);
                explored.insert(rel.target);
            }
        }
    }
    
    Ok(DivergentResult { ideas, exploration_depth: explored.len() })
}
```

### Step 4: Update Tests
Transform tests from LLM-dependent to graph-focused:

```rust
// REMOVE:
#[test]
async fn test_neural_reasoning() {
    let mock_server = MockNeuralServer::new();
    mock_server.expect_neural_reason()
        .returning(|_| Ok("Mocked response".to_string()));
    // ...
}

// REPLACE with:
#[test]
async fn test_graph_reasoning() {
    let graph = create_test_graph();
    graph.insert_entity(create_entity("Premise A"));
    graph.insert_entity(create_entity("Premise B"));
    graph.insert_relationship(create_logical_relation("A", "B"));
    
    let reasoning_result = convergent_thinking(&graph, "test query")?;
    assert!(reasoning_result.found_convergence);
}
```

## Verification Checklist

- [ ] No imports from `neural_query` module (unless renamed)
- [ ] No external HTTP client usage in cognitive modules
- [ ] No API keys or endpoints in cognitive code
- [ ] All cognitive patterns work with test graphs
- [ ] No mock LLM responses in tests
- [ ] Graph operations properly documented

## Files to Review and Update

1. `src/cognitive/convergent.rs` - Remove synthesis LLM calls
2. `src/cognitive/divergent.rs` - Remove generation LLM calls
3. `src/cognitive/lateral.rs` - Remove analogy LLM calls
4. `src/cognitive/systems.rs` - Remove analysis LLM calls
5. `src/cognitive/critical.rs` - Remove evaluation LLM calls
6. `src/cognitive/abstract_pattern.rs` - Remove abstraction LLM calls
7. `src/cognitive/adaptive.rs` - Remove adaptation LLM calls
8. `src/cognitive/working_memory.rs` - Remove embedding calls
9. `src/cognitive/attention_manager.rs` - Remove salience LLM calls
10. `src/cognitive/phase3_integration.rs` - Remove all neural references

## Expected Outcome

After this task:
- LLMKG operates purely on graph structures
- No external dependencies on any LLM
- All cognitive patterns are deterministic graph algorithms
- System can be tested without any external services
- Clear separation between MCP server (LLMKG) and clients (LLMs)