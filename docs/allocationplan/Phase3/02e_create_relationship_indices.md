# Task 02e: Create Relationship Indices

**Estimated Time**: 6 minutes  
**Dependencies**: 02d_create_range_indices.md  
**Next Task**: 03a_create_concept_node_struct.md  

## Objective
Create indices on relationship properties to optimize graph traversal and relationship queries.

## Single Action
Execute Cypher commands to create relationship property indices.

## Cypher Commands
```cypher
// Relationship property indices
CREATE INDEX inherits_depth_index FOR ()-[r:INHERITS_FROM]-() ON (r.inheritance_depth);
CREATE INDEX property_source_index FOR ()-[r:HAS_PROPERTY]-() ON (r.property_source);
CREATE INDEX semantic_similarity_index FOR ()-[r:SEMANTICALLY_RELATED]-() ON (r.similarity_score);
CREATE INDEX neural_connection_index FOR ()-[r:NEURAL_CONNECTION]-() ON (r.connection_strength);
CREATE INDEX temporal_relation_index FOR ()-[r:TEMPORAL_RELATION]-() ON (r.timestamp);
```

## Batch Script
File: `scripts/create_relationship_indices.cypher`
```cypher
// Relationship Indices Script
CREATE INDEX inherits_depth_index FOR ()-[r:INHERITS_FROM]-() ON (r.inheritance_depth);
CREATE INDEX property_source_index FOR ()-[r:HAS_PROPERTY]-() ON (r.property_source);
CREATE INDEX semantic_similarity_index FOR ()-[r:SEMANTICALLY_RELATED]-() ON (r.similarity_score);
CREATE INDEX neural_connection_index FOR ()-[r:NEURAL_CONNECTION]-() ON (r.connection_strength);
CREATE INDEX temporal_relation_index FOR ()-[r:TEMPORAL_RELATION]-() ON (r.timestamp);
```

## Execution
```bash
# Execute relationship indices script
docker exec llmkg-neo4j cypher-shell -u neo4j -p knowledge123 -f /scripts/create_relationship_indices.cypher
```

## Success Check
```cypher
// Verify relationship indices
SHOW INDEXES WHERE labelsOrTypes CONTAINS 'INHERITS_FROM' OR labelsOrTypes CONTAINS 'HAS_PROPERTY';
```

Should show 5 relationship indices.

## Relationship Query Test
```cypher
// Test relationship index usage
EXPLAIN MATCH (c1:Concept)-[r:INHERITS_FROM]->(c2:Concept) 
WHERE r.inheritance_depth < 3 
RETURN c1.name, c2.name;
// Should show relationship index usage
```

## Complete Schema Verification
```cypher
// Verify complete schema setup
SHOW CONSTRAINTS;
SHOW INDEXES;
```

Expected totals:
- 7 constraints (6 unique + 1 composite)
- ~27 indices (9 core + 7 temporal + 6 range + 5 relationship)

## Acceptance Criteria
- [ ] All 5 relationship indices created
- [ ] Relationship traversal queries use indices
- [ ] Complete schema verification shows all constraints and indices
- [ ] No relationship index creation errors

## Duration
4-6 minutes for relationship index creation and schema verification.