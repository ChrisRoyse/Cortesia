# Task 01e: Create Connection Manager Struct

**Estimated Time**: 10 minutes  
**Dependencies**: 01d_add_neo4j_dependency.md  
**Next Task**: 01f_implement_config_loading.md  

## Objective
Create basic Rust struct for Neo4j connection management.

## Single Action
Create `src/storage/neo4j_manager.rs` with connection manager struct.

## File Structure
```bash
mkdir -p src/storage
```

## Code to Write
File: `src/storage/neo4j_manager.rs`
```rust
use anyhow::Result;
use neo4j::{Driver, Session};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Neo4jConfig {
    pub uri: String,
    pub username: String,
    pub password: String,
    pub database: String,
    pub max_connections: usize,
    pub connection_timeout_secs: u64,
}

#[derive(Debug)]
pub struct Neo4jConnectionManager {
    driver: Arc<Driver>,
    config: Neo4jConfig,
}

impl Neo4jConnectionManager {
    pub async fn new(config: Neo4jConfig) -> Result<Self> {
        // TODO: Implementation in next task
        todo!("Implementation in task 01f")
    }
    
    pub async fn get_session(&self) -> Result<Session> {
        // TODO: Implementation in next task  
        todo!("Implementation in task 01f")
    }
    
    pub async fn health_check(&self) -> Result<bool> {
        // TODO: Implementation in next task
        todo!("Implementation in task 01f")
    }
}
```

## Module Export
Add to `src/storage/mod.rs`:
```rust
pub mod neo4j_manager;

pub use neo4j_manager::{Neo4jConfig, Neo4jConnectionManager};
```

## Success Check
```bash
# Verify compilation
cargo check
# Should compile with warnings about unused code and todo!() macros
```

## Acceptance Criteria
- [ ] File `src/storage/neo4j_manager.rs` exists
- [ ] Struct definitions are correct
- [ ] Module exports work
- [ ] Code compiles without errors

## Duration
8-10 minutes for struct creation and verification.