# Task 01h: Create Connection Health Check Test

**Estimated Time**: 8 minutes  
**Dependencies**: 01g_implement_connection_methods.md  
**Next Task**: 02a_create_unique_constraints.md  

## Objective
Create a comprehensive test to verify Neo4j connection health and performance.

## Single Action
Create integration test file for connection verification.

## File to Create
File: `tests/neo4j_integration_test.rs`
```rust
use llmkg::storage::{Neo4jConfig, Neo4jConnectionManager};
use std::time::Instant;

#[tokio::test]
async fn test_neo4j_connection_health() {
    // Load config from file
    let config = match Neo4jConfig::from_file("config/neo4j.toml") {
        Ok(config) => config,
        Err(_) => {
            // Fallback to default config for testing
            Neo4jConfig {
                uri: "bolt://localhost:7687".to_string(),
                username: "neo4j".to_string(),
                password: "knowledge123".to_string(),
                database: "neo4j".to_string(),
                max_connections: 10,
                connection_timeout_secs: 30,
            }
        }
    };
    
    // Test connection establishment
    let start = Instant::now();
    let manager = Neo4jConnectionManager::new(config).await;
    let connection_time = start.elapsed();
    
    assert!(manager.is_ok(), "Failed to create connection manager");
    let manager = manager.unwrap();
    
    // Test health check
    let start = Instant::now();
    let health = manager.health_check().await;
    let health_check_time = start.elapsed();
    
    assert!(health.is_ok(), "Health check failed");
    assert!(health.unwrap(), "Database is not healthy");
    
    // Performance assertions
    assert!(connection_time.as_millis() < 1000, "Connection took too long: {:?}", connection_time);
    assert!(health_check_time.as_millis() < 100, "Health check took too long: {:?}", health_check_time);
    
    println!("✅ Connection established in {:?}", connection_time);
    println!("✅ Health check completed in {:?}", health_check_time);
}

#[tokio::test]
async fn test_session_creation() {
    let config = Neo4jConfig::from_file("config/neo4j.toml")
        .unwrap_or_else(|_| Neo4jConfig {
            uri: "bolt://localhost:7687".to_string(),
            username: "neo4j".to_string(),
            password: "knowledge123".to_string(),
            database: "neo4j".to_string(),
            max_connections: 10,
            connection_timeout_secs: 30,
        });
    
    let manager = Neo4jConnectionManager::new(config).await.unwrap();
    let session_result = manager.get_session().await;
    
    assert!(session_result.is_ok(), "Failed to create session");
    println!("✅ Session created successfully");
}
```

## Module Addition
Add to `src/lib.rs`:
```rust
pub mod storage;
```

## Success Check
```bash
# Run the integration test (requires Neo4j running)
cargo test test_neo4j_connection_health --test neo4j_integration_test

# If Neo4j is not running, just verify compilation
cargo check --tests
```

## Acceptance Criteria
- [ ] Integration test file created
- [ ] Tests verify connection performance (<1s)
- [ ] Tests verify health check performance (<100ms)
- [ ] Tests pass when Neo4j is available
- [ ] Tests compile even when Neo4j is unavailable

## Duration
6-8 minutes for test creation and verification.