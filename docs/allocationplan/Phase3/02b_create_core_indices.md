# Task 02b: Create Core Performance Indices

**Estimated Time**: 8 minutes  
**Dependencies**: 02a_create_unique_constraints.md  
**Next Task**: 02c_create_temporal_indices.md  

## Objective
Create essential performance indices for name, type, and semantic properties.

## Single Action
Execute Cypher commands to create core performance indices.

## Cypher Commands
```cypher
// Core performance indices
CREATE INDEX concept_name_index FOR (c:Concept) ON (c.name);
CREATE INDEX concept_type_index FOR (c:Concept) ON (c.type);
CREATE INDEX concept_ttfs_index FOR (c:Concept) ON (c.ttfs_encoding);
CREATE INDEX memory_type_index FOR (m:Memory) ON (m.memory_type);
CREATE INDEX property_name_index FOR (p:Property) ON (p.name);
CREATE INDEX property_type_index FOR (p:Property) ON (p.property_type);

// Inheritance optimization indices
CREATE INDEX inheritance_depth_index FOR (c:Concept) ON (c.inheritance_depth);
CREATE INDEX property_inheritable_index FOR (p:Property) ON (p.is_inheritable);

// Neural pathway indices
CREATE INDEX pathway_type_index FOR (n:NeuralPathway) ON (n.pathway_type);
```

## Batch Execution Script
Create file: `scripts/create_core_indices.cypher`
```cypher
// Core Performance Indices Script
CREATE INDEX concept_name_index FOR (c:Concept) ON (c.name);
CREATE INDEX concept_type_index FOR (c:Concept) ON (c.type);
CREATE INDEX concept_ttfs_index FOR (c:Concept) ON (c.ttfs_encoding);
CREATE INDEX memory_type_index FOR (m:Memory) ON (m.memory_type);
CREATE INDEX property_name_index FOR (p:Property) ON (p.name);
CREATE INDEX property_type_index FOR (p:Property) ON (p.property_type);
CREATE INDEX inheritance_depth_index FOR (c:Concept) ON (c.inheritance_depth);
CREATE INDEX property_inheritable_index FOR (p:Property) ON (p.is_inheritable);
CREATE INDEX pathway_type_index FOR (n:NeuralPathway) ON (n.pathway_type);
```

## Execution Command
```bash
# Create script directory
mkdir -p scripts

# Execute the script
docker exec llmkg-neo4j cypher-shell -u neo4j -p knowledge123 \
  -f /scripts/create_core_indices.cypher
```

## Alternative (Manual execution)
```bash
# Execute each index individually
docker exec llmkg-neo4j cypher-shell -u neo4j -p knowledge123 \
  "CREATE INDEX concept_name_index FOR (c:Concept) ON (c.name);"
```

## Success Check
```cypher
// Verify indices exist
SHOW INDEXES;
```

Expected: 9 new indices should be listed.

## Performance Test
```cypher
// Test index usage
EXPLAIN MATCH (c:Concept) WHERE c.name = 'test' RETURN c;
// Should show index usage in query plan
```

## Acceptance Criteria
- [ ] All 9 core indices created successfully
- [ ] SHOW INDEXES displays all new indices
- [ ] Query plans show index usage
- [ ] No index creation errors

## Duration
6-8 minutes for index creation and verification.