# Task 20a: Create Basic Connection Pool Struct
**Time**: 4 minutes
**Dependencies**: None
**Stage**: Resource Management Foundation

## Objective
Create the foundational ConnectionPool struct with basic configuration.

## Implementation
Create file `src/inheritance/resources/connection_pool.rs`:
```rust
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::{Mutex, Semaphore};

#[derive(Debug)]
pub struct ConnectionPool {
    pub config: PoolConfig,
    pub available_connections: Arc<Mutex<VecDeque<Connection>>>,
    pub semaphore: Arc<Semaphore>,
    pub is_active: bool,
}

#[derive(Debug, Clone)]
pub struct PoolConfig {
    pub min_connections: usize,
    pub max_connections: usize,
    pub connection_timeout_ms: u64,
    pub idle_timeout_ms: u64,
}

#[derive(Debug)]
pub struct Connection {
    pub id: String,
    pub created_at: std::time::Instant,
    pub last_used: std::time::Instant,
    pub is_healthy: bool,
}

impl ConnectionPool {
    pub fn new(config: PoolConfig) -> Self {
        let max_connections = config.max_connections;
        Self {
            config,
            available_connections: Arc::new(Mutex::new(VecDeque::new())),
            semaphore: Arc::new(Semaphore::new(max_connections)),
            is_active: false,
        }
    }
}
```

## Success Criteria
- Basic struct compiles
- Configuration struct functional
- Connection tracking ready

**Next**: 20b