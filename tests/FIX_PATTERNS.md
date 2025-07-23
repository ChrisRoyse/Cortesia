# Test Compilation Fix Patterns

This document outlines common patterns for fixing compilation errors in the LLMKG test suite after recent API changes.

## 1. EntityData Field Renames

**OLD**:
```rust
EntityData {
    entity_type: 1,
    properties: "test".to_string(),
    embedding: vec![0.0; 512],
}
```

**NEW**:
```rust
EntityData {
    type_id: 1,
    properties: "test".to_string(),
    embedding: vec![0.0; 512],
}
```

## 2. KnowledgeGraph Constructor

**OLD**:
```rust
KnowledgeGraph::new()  // No arguments
```

**NEW**:
```rust
KnowledgeGraph::new(128).unwrap()  // Requires embedding dimension
```

## 3. Triple Struct Fields

The `Triple` struct no longer has `timestamp` or `metadata` fields.

**OLD**:
```rust
Triple {
    subject: "s".to_string(),
    predicate: "p".to_string(),
    object: "o".to_string(),
    confidence: 0.9,
    source: Some("test".to_string()),
    timestamp: std::time::SystemTime::now(),  // REMOVE THIS
    metadata: HashMap::new(),                 // REMOVE THIS
}
```

**NEW**:
```rust
Triple {
    subject: "s".to_string(),
    predicate: "p".to_string(),
    object: "o".to_string(),
    confidence: 0.9,
    source: Some("test".to_string()),
}
```

## 4. Method Signature Changes

### add_relationship
**OLD**:
```rust
graph.add_relationship(from, to, "relationship_name", 1.0)  // 4 arguments
```

**NEW**:
```rust
graph.add_relationship(from, to, 1.0)  // 3 arguments only
```

### ProductQuantizer::new
**OLD**:
```rust
ProductQuantizer::new(embedding_dim, num_subspaces, 256)  // 3 arguments
```

**NEW**:
```rust
ProductQuantizer::new(embedding_dim, num_subspaces)  // 2 arguments only
```

### optimize_graph_structure
**OLD**:
```rust
graph.optimize_graph_structure(0.1).await  // With threshold argument
```

**NEW**:
```rust
graph.optimize_graph_structure().await  // No arguments
```

## 5. Type Conversions (u32 vs EntityKey)

Many methods now expect `EntityKey` instead of `u32`.

**OLD**:
```rust
graph.set_entity_activation(10, 1.0).await;
```

**NEW**:
```rust
let entity_key = EntityKey::from_raw_parts(10, 0);
graph.set_entity_activation(entity_key, 1.0).await;
```

## 6. Removed/Renamed Methods

- `spread_activation` -> Use `propagate_activation_from_entity` instead
- `prune_weak_connections` -> Use `prune_weak_relationships` instead

## 7. Enum Value Changes

**OLD**:
```rust
CognitivePatternType::GraphOfThoughts  // Doesn't exist
```

**NEW**:
```rust
CognitivePatternType::TreeOfThoughts  // Use this instead
```

## 8. Struct Field Renames

### CognitiveQueryResult
- `total_time_ms` -> `execution_time_ms`

### QualityMetrics
- `coherence_score` -> `consistency_score`
- `relevance_score` -> `completeness_score`

## 9. Type Issues with HashMap Keys

When working with `HashMap<EntityKey, f32>`:

**OLD**:
```rust
let hub_id = 42u32;
stats.betweenness_centrality.get(&hub_id)  // Wrong type
```

**NEW**:
```rust
let hub_key = EntityKey::from_raw_parts(42, 0);
stats.betweenness_centrality.get(&hub_key)  // Correct type
```

## Common Import Additions

You may need to add these imports:
```rust
use llmkg::core::types::EntityKey;
```

## Notes

- Always check method signatures in the source code when in doubt
- The codebase is moving from raw numeric IDs (u32) to typed EntityKey
- Many APIs have been simplified to have fewer parameters
- Some fields have been renamed for clarity and consistency