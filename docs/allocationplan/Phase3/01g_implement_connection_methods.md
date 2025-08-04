# Task 01g: Implement Connection Methods

**Estimated Time**: 12 minutes  
**Dependencies**: 01f_implement_config_loading.md  
**Next Task**: 01h_create_connection_health_check.md  

## Objective
Implement the actual connection establishment and session management methods.

## Single Action
Replace TODO placeholders with real Neo4j driver implementation.

## Code to Replace
In `src/storage/neo4j_manager.rs`, replace the methods:

```rust
impl Neo4jConnectionManager {
    pub async fn new(config: Neo4jConfig) -> Result<Self> {
        let driver = Driver::new(&config.uri, neo4j::auth::basic(&config.username, &config.password));
        let driver_arc = Arc::new(driver);
        
        Ok(Neo4jConnectionManager {
            driver: driver_arc,
            config,
        })
    }
    
    pub async fn get_session(&self) -> Result<Session> {
        let session = self.driver.session(&self.config.database);
        Ok(session)
    }
    
    pub async fn health_check(&self) -> Result<bool> {
        match self.get_session().await {
            Ok(session) => {
                // Try a simple query to verify connection
                let result = session.run("RETURN 1 as health_check").await;
                match result {
                    Ok(_) => Ok(true),
                    Err(_) => Ok(false),
                }
            }
            Err(_) => Ok(false),
        }
    }
}
```

## Add Import Statements
Add to top of file:
```rust
use neo4j::auth;
```

## Success Check
```bash
# Verify compilation
cargo check
# Should compile without TODO warnings

# Quick syntax verification
cargo build --release --dry-run
```

## Integration Test (Optional)
If Neo4j is running, test connection:
```rust
#[tokio::test]
async fn test_connection() {
    let config = Neo4jConfig {
        uri: "bolt://localhost:7687".to_string(),
        username: "neo4j".to_string(),
        password: "knowledge123".to_string(),
        database: "neo4j".to_string(),
        max_connections: 10,
        connection_timeout_secs: 30,
    };
    
    let manager = Neo4jConnectionManager::new(config).await.unwrap();
    let health = manager.health_check().await.unwrap();
    assert!(health);
}
```

## Acceptance Criteria
- [ ] All TODO placeholders replaced with real implementations
- [ ] Driver initialization works
- [ ] Session creation works
- [ ] Health check implementation functional
- [ ] Code compiles without warnings

## Duration
10-12 minutes for implementation and verification.