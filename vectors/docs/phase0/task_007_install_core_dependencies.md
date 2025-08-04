# Micro-Task 007: Install Core Dependencies

## Objective
Add essential shared dependencies to the workspace configuration.

## Context
The vector search system requires core dependencies like Tantivy for text search, async runtime for concurrency, and serialization libraries. This task adds these to the workspace level for sharing across crates.

## Prerequisites
- Task 006 completed (Cargo workspace created)
- Internet connection for dependency download

## Time Estimate
9 minutes

## Instructions
1. Edit `Cargo.toml` to add workspace dependencies section:
   ```toml
   [workspace.dependencies]
   tantivy = "0.21"
   tokio = { version = "1.0", features = ["full"] }
   serde = { version = "1.0", features = ["derive"] }
   serde_json = "1.0"
   anyhow = "1.0"
   thiserror = "1.0"
   uuid = { version = "1.0", features = ["v4"] }
   rayon = "1.7"
   ```
2. Verify dependencies can be resolved: `cargo tree` (will show empty, that's expected)
3. Add and commit changes: `git add Cargo.toml && git commit -m "Add core workspace dependencies"`

## Expected Output
- Workspace dependencies section added
- Core dependencies specified with versions
- Configuration committed to Git

## Success Criteria
- [ ] `[workspace.dependencies]` section exists in Cargo.toml
- [ ] Tantivy, Tokio, Serde, and other core deps listed
- [ ] No syntax errors in TOML file
- [ ] Changes committed to Git

## Next Task
task_008_setup_development_profile.md