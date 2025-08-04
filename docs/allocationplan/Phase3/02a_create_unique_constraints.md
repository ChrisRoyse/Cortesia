# Task 02a: Create Unique Constraints

**Estimated Time**: 5 minutes  
**Dependencies**: 01h_create_connection_health_check.md  
**Next Task**: 02b_create_core_indices.md  

## Objective
Create unique constraints for all core node types to prevent duplicate entities.

## Single Action
Execute Cypher commands to create unique ID constraints.

## Cypher Commands
Execute these constraints in Neo4j:
```cypher
// Core node type unique constraints
CREATE CONSTRAINT concept_id_unique FOR (c:Concept) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT memory_id_unique FOR (m:Memory) REQUIRE m.id IS UNIQUE;
CREATE CONSTRAINT property_id_unique FOR (p:Property) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT exception_id_unique FOR (e:Exception) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT version_id_unique FOR (v:Version) REQUIRE v.id IS UNIQUE;
CREATE CONSTRAINT neural_pathway_id_unique FOR (n:NeuralPathway) REQUIRE n.id IS UNIQUE;

// Composite constraint for version uniqueness
CREATE CONSTRAINT version_branch_unique FOR (v:Version) REQUIRE (v.branch_name, v.version_number) IS UNIQUE;
```

## Execution Methods
### Method 1: Using cypher-shell
```bash
docker exec llmkg-neo4j cypher-shell -u neo4j -p knowledge123 \
  "CREATE CONSTRAINT concept_id_unique FOR (c:Concept) REQUIRE c.id IS UNIQUE;"
```

### Method 2: Neo4j Browser
1. Open http://localhost:7474
2. Login with neo4j/knowledge123
3. Execute each constraint command

### Method 3: Using Rust (integration test)
Create file: `tests/schema_constraints_test.rs`
```rust
use llmkg::storage::{Neo4jConfig, Neo4jConnectionManager};

#[tokio::test]
async fn test_create_unique_constraints() {
    let config = Neo4jConfig::from_file("config/neo4j.toml").unwrap();
    let manager = Neo4jConnectionManager::new(config).await.unwrap();
    let session = manager.get_session().await.unwrap();
    
    let constraints = vec![
        "CREATE CONSTRAINT concept_id_unique FOR (c:Concept) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT memory_id_unique FOR (m:Memory) REQUIRE m.id IS UNIQUE",
    ];
    
    for constraint in constraints {
        let result = session.run(constraint).await;
        assert!(result.is_ok(), "Failed to create constraint: {}", constraint);
    }
}
```

## Success Check
```cypher
// Verify constraints exist
SHOW CONSTRAINTS;
```

Expected output should show the 7 constraints created.

## Acceptance Criteria
- [ ] All 6 unique ID constraints created
- [ ] Version branch composite constraint created
- [ ] No constraint creation errors
- [ ] Constraints are active and enforced

## Duration
3-5 minutes for constraint creation and verification.