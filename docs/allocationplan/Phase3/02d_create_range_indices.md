# Task 02d: Create Range Indices

**Estimated Time**: 7 minutes  
**Dependencies**: 02c_create_temporal_indices.md  
**Next Task**: 02e_create_relationship_indices.md  

## Objective
Create range indices for numeric properties to optimize range queries and sorting.

## Single Action
Execute Cypher commands to create range indices for numeric fields.

## Cypher Commands
```cypher
// Numeric range indices for performance queries
CREATE RANGE INDEX concept_confidence_range FOR (c:Concept) ON (c.confidence_score);
CREATE RANGE INDEX memory_decay_range FOR (m:Memory) ON (m.decay_rate);
CREATE RANGE INDEX exception_confidence_range FOR (e:Exception) ON (e.confidence);
CREATE RANGE INDEX inheritance_priority_range FOR (p:Property) ON (p.inheritance_priority);
CREATE RANGE INDEX neural_activation_range FOR (n:NeuralPathway) ON (n.activation_strength);
CREATE RANGE INDEX concept_embedding_range FOR (c:Concept) ON (c.semantic_embedding_norm);
```

## Batch Script
File: `scripts/create_range_indices.cypher`
```cypher
// Numeric Range Indices Script
CREATE RANGE INDEX concept_confidence_range FOR (c:Concept) ON (c.confidence_score);
CREATE RANGE INDEX memory_decay_range FOR (m:Memory) ON (c.decay_rate);
CREATE RANGE INDEX exception_confidence_range FOR (e:Exception) ON (e.confidence);
CREATE RANGE INDEX inheritance_priority_range FOR (p:Property) ON (p.inheritance_priority);
CREATE RANGE INDEX neural_activation_range FOR (n:NeuralPathway) ON (n.activation_strength);
CREATE RANGE INDEX concept_embedding_range FOR (c:Concept) ON (c.semantic_embedding_norm);
```

## Execution
```bash
# Execute range indices script
docker exec llmkg-neo4j cypher-shell -u neo4j -p knowledge123 -f /scripts/create_range_indices.cypher
```

## Success Check
```cypher
// Verify range indices
SHOW INDEXES WHERE type = 'RANGE';
```

Should show 6 range indices.

## Range Query Test
```cypher
// Test range query performance
EXPLAIN MATCH (c:Concept) WHERE c.confidence_score > 0.8 RETURN c.name ORDER BY c.confidence_score DESC;
// Should use range index scan
```

## Acceptance Criteria
- [ ] All 6 range indices created successfully
- [ ] Range queries show index usage in EXPLAIN
- [ ] ORDER BY queries on indexed fields use indices
- [ ] No range index creation errors

## Duration
5-7 minutes for range index creation and testing.