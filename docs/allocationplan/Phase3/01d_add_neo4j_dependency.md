# Task 01d: Add Neo4j Rust Dependency

**Estimated Time**: 3 minutes  
**Dependencies**: 01c_create_database_config.md  
**Next Task**: 01e_create_connection_manager_struct.md  

## Objective
Add Neo4j Rust driver dependency to Cargo.toml.

## Single Action
Add the neo4j crate and required dependencies to Cargo.toml.

## Dependencies to Add
Add to `Cargo.toml` in `[dependencies]` section:
```toml
# Neo4j database driver
neo4j = "0.2"
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
toml = "0.8"
anyhow = "1.0"
uuid = { version = "1.0", features = ["v4"] }
```

## Command
```bash
# Add dependencies
cargo add neo4j@0.2
cargo add tokio --features full
cargo add serde --features derive  
cargo add toml@0.8
cargo add anyhow@1.0
cargo add uuid --features v4
```

## Success Check
```bash
# Verify dependencies were added
grep -A 10 "\[dependencies\]" Cargo.toml | grep neo4j
# Should show: neo4j = "0.2"

# Test compilation
cargo check
# Should complete without errors
```

## Acceptance Criteria
- [ ] Neo4j dependency added to Cargo.toml
- [ ] Required supporting dependencies added
- [ ] `cargo check` runs without errors
- [ ] Version constraints are compatible

## Duration
2-3 minutes for dependency addition and verification.