# Micro-Task 006: Create Cargo Workspace

## Objective
Create the root Cargo.toml file to establish a Rust workspace for the vector search system.

## Context
A Cargo workspace allows multiple related crates to be managed together, sharing dependencies and build configurations. This is essential for the modular architecture of our vector search system.

## Prerequisites
- Task 005 completed (Git repository initialized)
- Currently in project root directory
- Basic understanding of TOML format

## Time Estimate
8 minutes

## Instructions
1. Create `Cargo.toml` in project root with workspace configuration:
   ```toml
   [workspace]
   members = [
       "crates/tantivy-core",
       "crates/lancedb-integration", 
       "crates/vector-indexing",
       "crates/search-api"
   ]
   resolver = "2"
   
   [workspace.package]
   version = "0.1.0"
   edition = "2021"
   ```
2. Create `crates` directory: `mkdir crates`
3. Verify workspace structure: `cargo check --workspace` (will show errors, that's expected)
4. Add and commit: `git add Cargo.toml && git commit -m "Add Cargo workspace configuration"`

## Expected Output
- Root `Cargo.toml` file created
- Workspace configuration established
- `crates` directory ready for member crates
- Changes committed to Git

## Success Criteria
- [ ] `Cargo.toml` exists in project root
- [ ] Workspace section correctly defines member crates
- [ ] `crates` directory exists
- [ ] Changes committed to Git repository

## Next Task
task_007_install_core_dependencies.md