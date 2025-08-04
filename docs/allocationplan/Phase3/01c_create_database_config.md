# Task 01c: Create Database Configuration File

**Estimated Time**: 5 minutes  
**Dependencies**: 01b_verify_neo4j_connection.md  
**Next Task**: 01d_add_neo4j_dependency.md  

## Objective
Create a configuration file for Neo4j database connection settings.

## Single Action
Create `config/neo4j.toml` with connection parameters.

## File to Create
Create directory and file:
```bash
mkdir -p config
```

## Configuration Content
File: `config/neo4j.toml`
```toml
# Neo4j Database Configuration
[neo4j]
uri = "bolt://localhost:7687"
username = "neo4j"
password = "knowledge123"
database = "neo4j"  # Default database name
max_connections = 50
connection_timeout_secs = 30
session_timeout_secs = 300

[neo4j.memory]
heap_initial = "1G"
heap_max = "2G" 
pagecache = "1G"

[neo4j.performance]
query_timeout_secs = 60
transaction_timeout_secs = 120
retry_attempts = 3
retry_delay_ms = 1000
```

## Success Check
```bash
# Verify file exists and is readable
cat config/neo4j.toml | head -5
# Should show the [neo4j] section
```

## Acceptance Criteria
- [ ] File `config/neo4j.toml` exists
- [ ] Contains valid TOML syntax
- [ ] Includes all required connection parameters
- [ ] Configuration values match running Neo4j instance

## Duration
3-5 minutes to create and verify configuration.