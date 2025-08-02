# Task 02c: Create Temporal Indices

**Estimated Time**: 6 minutes  
**Dependencies**: 02b_create_core_indices.md  
**Next Task**: 02d_create_range_indices.md  

## Objective
Create indices for timestamp and temporal properties to optimize time-based queries.

## Single Action
Execute Cypher commands to create temporal and timestamp indices.

## Cypher Commands
```cypher
// Temporal timestamp indices
CREATE INDEX memory_timestamp_index FOR (m:Memory) ON (m.created_at);
CREATE INDEX concept_created_index FOR (c:Concept) ON (c.creation_timestamp);
CREATE INDEX concept_accessed_index FOR (c:Concept) ON (c.last_accessed);
CREATE INDEX version_timestamp_index FOR (v:Version) ON (v.created_at);
CREATE INDEX exception_timestamp_index FOR (e:Exception) ON (e.created_at);

// Frequency and access pattern indices  
CREATE INDEX concept_frequency_index FOR (c:Concept) ON (c.access_frequency);
CREATE INDEX memory_strength_index FOR (m:Memory) ON (m.strength);
```

## Batch Script
File: `scripts/create_temporal_indices.cypher`
```cypher
// Temporal Indices Script
CREATE INDEX memory_timestamp_index FOR (m:Memory) ON (m.created_at);
CREATE INDEX concept_created_index FOR (c:Concept) ON (c.creation_timestamp);
CREATE INDEX concept_accessed_index FOR (c:Concept) ON (c.last_accessed);
CREATE INDEX version_timestamp_index FOR (v:Version) ON (v.created_at);
CREATE INDEX exception_timestamp_index FOR (e:Exception) ON (e.created_at);
CREATE INDEX concept_frequency_index FOR (c:Concept) ON (c.access_frequency);
CREATE INDEX memory_strength_index FOR (m:Memory) ON (m.strength);
```

## Execution
```bash
# Execute temporal indices script
docker exec llmkg-neo4j cypher-shell -u neo4j -p knowledge123 -f /scripts/create_temporal_indices.cypher
```

## Success Check
```cypher
// Count temporal indices
SHOW INDEXES WHERE name CONTAINS 'timestamp' OR name CONTAINS 'created' OR name CONTAINS 'accessed';
```

Should return 7 temporal indices.

## Performance Validation
```cypher
// Test temporal query performance
EXPLAIN MATCH (m:Memory) WHERE m.created_at > datetime('2024-01-01') RETURN count(m);
// Should show index scan, not label scan
```

## Acceptance Criteria
- [ ] All 7 temporal indices created
- [ ] Timestamp queries use index scans
- [ ] No index creation errors
- [ ] Indices visible in SHOW INDEXES

## Duration
4-6 minutes for temporal index creation and verification.