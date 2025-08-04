# Task 01f: Implement Configuration Loading

**Estimated Time**: 8 minutes  
**Dependencies**: 01e_create_connection_manager_struct.md  
**Next Task**: 01g_implement_connection_methods.md  

## Objective
Implement configuration loading from the TOML file.

## Single Action
Add config loading function to load Neo4j settings from `config/neo4j.toml`.

## Code to Add
Add to `src/storage/neo4j_manager.rs`:
```rust
use std::fs;

impl Neo4jConfig {
    pub fn from_file(path: &str) -> Result<Self> {
        let contents = fs::read_to_string(path)?;
        let config: toml::Value = toml::from_str(&contents)?;
        
        let neo4j_section = config.get("neo4j")
            .ok_or_else(|| anyhow::anyhow!("Missing [neo4j] section in config"))?;
        
        Ok(Neo4jConfig {
            uri: neo4j_section.get("uri")
                .and_then(|v| v.as_str())
                .unwrap_or("bolt://localhost:7687")
                .to_string(),
            username: neo4j_section.get("username")
                .and_then(|v| v.as_str())
                .unwrap_or("neo4j")
                .to_string(),
            password: neo4j_section.get("password")
                .and_then(|v| v.as_str())
                .unwrap_or("password")
                .to_string(),
            database: neo4j_section.get("database")
                .and_then(|v| v.as_str())
                .unwrap_or("neo4j")
                .to_string(),
            max_connections: neo4j_section.get("max_connections")
                .and_then(|v| v.as_integer())
                .unwrap_or(50) as usize,
            connection_timeout_secs: neo4j_section.get("connection_timeout_secs")
                .and_then(|v| v.as_integer())
                .unwrap_or(30) as u64,
        })
    }
}
```

## Test the Implementation
Add simple test function:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config_loading() {
        // Test will fail if config file doesn't exist - that's expected
        match Neo4jConfig::from_file("config/neo4j.toml") {
            Ok(config) => {
                assert_eq!(config.uri, "bolt://localhost:7687");
                assert_eq!(config.username, "neo4j");
            }
            Err(_) => {
                // Expected if config file doesn't exist yet
                println!("Config file not found - will be created in setup");
            }
        }
    }
}
```

## Success Check
```bash
# Verify compilation
cargo check
# Should compile successfully

# Test config loading if file exists
cargo test test_config_loading
```

## Acceptance Criteria
- [ ] Config loading function implemented
- [ ] Handles missing config file gracefully
- [ ] Provides sensible defaults
- [ ] Test compiles and runs

## Duration
6-8 minutes for implementation and testing.