# Task 41: Connection Pooling Implementation

**Time:** 10 minutes (2 min read, 6 min implement, 2 min verify)
**Prerequisites:** Task 40 completed
**Input Files:** 
- C:/code/LLMKG/vectors/tantivy_search/src/lib.rs
- C:/code/LLMKG/vectors/tantivy_search/src/search.rs  
- C:/code/LLMKG/vectors/tantivy_search/Cargo.toml

## Complete Context (For AI with ZERO Knowledge)

You are implementing **connection pooling** for the Tantivy-based search system. Connection pooling manages multiple concurrent search readers efficiently, preventing resource exhaustion and improving performance under load.

**What is Connection Pooling?** A design pattern that maintains a pool of pre-initialized search readers that can be reused across multiple queries, eliminating the overhead of creating/destroying readers for each search operation.

**System Context:** After tasks 1-40, we have a complete DocumentIndexer with chunking, SearchEngine with dual-field support, and performance optimizations. This task adds production-grade connection management.

**This Task:** Creates a ConnectionPool that manages Tantivy IndexReader instances with automatic cleanup, health checks, and resource limits.

## Exact Steps (6 minutes implementation)

### Step 1: Add Connection Pool Dependencies (1 minute)
Edit `C:/code/LLMKG/vectors/tantivy_search/Cargo.toml`, add to `[dependencies]` section:
```toml
# Connection pooling for search readers  
dashmap = "5.5.3"
parking_lot = "0.12.1"
```

### Step 2: Create Connection Pool Module (2 minutes)
Create `C:/code/LLMKG/vectors/tantivy_search/src/connection_pool.rs`:
```rust
use anyhow::{anyhow, Result};
use dashmap::DashMap;
use parking_lot::RwLock;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tantivy::{Index, IndexReader};

#[derive(Debug, Clone)]
pub struct PoolConfig {
    pub max_connections: usize,
    pub connection_timeout: Duration,
    pub idle_timeout: Duration,
    pub health_check_interval: Duration,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_connections: 20,
            connection_timeout: Duration::from_secs(30),
            idle_timeout: Duration::from_secs(300), // 5 minutes
            health_check_interval: Duration::from_secs(60),
        }
    }
}

#[derive(Debug)]
struct PooledConnection {
    reader: IndexReader,
    created_at: Instant,
    last_used: RwLock<Instant>,
    is_healthy: RwLock<bool>,
}

impl PooledConnection {
    fn new(reader: IndexReader) -> Self {
        let now = Instant::now();
        Self {
            reader,
            created_at: now,
            last_used: RwLock::new(now),
            is_healthy: RwLock::new(true),
        }
    }

    fn mark_used(&self) {
        *self.last_used.write() = Instant::now();
    }

    fn is_expired(&self, idle_timeout: Duration) -> bool {
        self.last_used.read().elapsed() > idle_timeout
    }

    fn mark_unhealthy(&self) {
        *self.is_healthy.write() = false;
    }

    fn is_healthy(&self) -> bool {
        *self.is_healthy.read()
    }
}

pub struct ConnectionPool {
    index: Index,
    config: PoolConfig,
    connections: DashMap<String, Arc<PooledConnection>>,
    active_count: RwLock<usize>,
}
```

### Step 3: Implement Pool Methods (2 minutes)
Continue in `src/connection_pool.rs`:
```rust
impl ConnectionPool {
    pub fn new(index: Index, config: PoolConfig) -> Self {
        Self {
            index,
            config,
            connections: DashMap::new(),
            active_count: RwLock::new(0),
        }
    }

    pub fn get_reader(&self) -> Result<Arc<PooledConnection>> {
        // Try to find healthy, available connection
        for entry in self.connections.iter() {
            let connection = entry.value();
            if connection.is_healthy() && !connection.is_expired(self.config.idle_timeout) {
                connection.mark_used();
                return Ok(Arc::clone(connection));
            }
        }

        // Create new connection if under limit
        let active = *self.active_count.read();
        if active < self.config.max_connections {
            let reader = self.index.reader_builder()
                .reload_policy(tantivy::ReloadPolicy::OnCommit)
                .try_into()?;
            
            let connection = Arc::new(PooledConnection::new(reader));
            let id = format!("conn_{}", active);
            
            self.connections.insert(id, Arc::clone(&connection));
            *self.active_count.write() += 1;
            
            return Ok(connection);
        }

        Err(anyhow!("Connection pool exhausted. Max connections: {}", self.config.max_connections))
    }

    pub fn cleanup_expired(&self) {
        let mut to_remove = Vec::new();
        
        for entry in self.connections.iter() {
            let connection = entry.value();
            if connection.is_expired(self.config.idle_timeout) || !connection.is_healthy() {
                to_remove.push(entry.key().clone());
            }
        }

        for key in to_remove {
            if self.connections.remove(&key).is_some() {
                let mut active = self.active_count.write();
                if *active > 0 {
                    *active -= 1;
                }
            }
        }
    }

    pub fn get_stats(&self) -> PoolStats {
        PoolStats {
            active_connections: *self.active_count.read(),
            max_connections: self.config.max_connections,
            healthy_connections: self.connections.iter()
                .filter(|entry| entry.value().is_healthy())
                .count(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PoolStats {
    pub active_connections: usize,
    pub max_connections: usize,
    pub healthy_connections: usize,
}
```

### Step 4: Update Search Engine Integration (1 minute)
Add to `C:/code/LLMKG/vectors/tantivy_search/src/search.rs` imports:
```rust
use crate::connection_pool::{ConnectionPool, PoolConfig};
```

Add to SearchEngine struct:
```rust
pub struct SearchEngine {
    // ... existing fields
    connection_pool: Option<ConnectionPool>,
}
```

Add method to SearchEngine impl:
```rust
impl SearchEngine {
    pub fn with_connection_pool(mut self, config: PoolConfig) -> Result<Self> {
        if let Some(ref index) = self.index {
            self.connection_pool = Some(ConnectionPool::new(index.clone(), config));
        }
        Ok(self)
    }

    pub fn get_pool_stats(&self) -> Option<PoolStats> {
        self.connection_pool.as_ref().map(|pool| pool.get_stats())
    }
}
```

## Verification Steps (2 minutes)
```bash
cd C:/code/LLMKG/vectors/tantivy_search
cargo check
cargo test connection_pool
```

Create test in `src/connection_pool.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use tantivy::{schema::*, Index};

    #[test]
    fn test_connection_pool_basic() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let mut schema_builder = Schema::builder();
        schema_builder.add_text_field("content", TEXT);
        let schema = schema_builder.build();
        
        let index = Index::create_in_dir(&temp_dir.path(), schema)?;
        let config = PoolConfig::default();
        let pool = ConnectionPool::new(index, config);
        
        // Test getting a connection
        let conn1 = pool.get_reader()?;
        assert!(conn1.is_healthy());
        
        // Test stats
        let stats = pool.get_stats();
        assert_eq!(stats.active_connections, 1);
        assert_eq!(stats.healthy_connections, 1);
        
        Ok(())
    }
}
```

## Success Validation Checklist
- [ ] File exists: `src/connection_pool.rs` with PoolConfig, PooledConnection, and ConnectionPool
- [ ] Dependencies added: `dashmap = "5.5.3"`, `parking_lot = "0.12.1"`
- [ ] SearchEngine updated with connection pool integration methods
- [ ] Command `cargo check` completes without errors
- [ ] Test `test_connection_pool_basic` passes
- [ ] Pool enforces max connection limits correctly
- [ ] Expired connections are cleaned up automatically

## Files Created For Next Task
1. **C:/code/LLMKG/vectors/tantivy_search/src/connection_pool.rs** - Connection pool implementation with health checks
2. **Updated SearchEngine** - Now supports connection pooling for better resource management

**Next Task (Task 42)** will implement background processing system for index maintenance and cleanup operations.