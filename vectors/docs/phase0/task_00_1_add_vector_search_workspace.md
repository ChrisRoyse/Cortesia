# Task 00_1: Add Vector-Search Workspace Member to Neuromorphic Project

**Estimated Time:** 5-7 minutes  
**Prerequisites:** None (Foundation task)  
**Dependencies:** Must be completed before Task 00_2

## Objective
Add `vector-search` as a new workspace member to the existing neuromorphic Cargo.toml, properly integrating with the established codebase architecture.

## Context
You are creating the foundation for a vector search system that will integrate with the existing neuromorphic neural network codebase. The neuromorphic project already has 6 workspace members (neuromorphic-core, snn-allocation-engine, temporal-memory, neural-bridge, neuromorphic-wasm, snn-mocks), and you need to add vector-search as the 7th member.

## Task Details

### What You Need to Do
1. **Examine the existing workspace structure** in `/Cargo.toml`:
   - Note existing members array structure
   - Understand the crate organization pattern
   - Preserve existing dependencies and configurations

2. **Add vector-search to workspace members:**
   ```toml
   members = [
       "crates/neuromorphic-core",
       "crates/snn-allocation-engine", 
       "crates/temporal-memory",
       "crates/neural-bridge",
       "crates/neuromorphic-wasm",
       "crates/snn-mocks",
       "crates/vector-search",  # <- Add this line
   ]
   ```

3. **Add vector search dependencies to workspace.dependencies:**
   ```toml
   # Vector search and indexing (Phase 0 Foundation)
   tantivy = "0.22.0"               # Full-text search engine
   lancedb = "0.8.0"                # Vector database
   tree-sitter = "0.20.0"          # AST parsing for smart chunking
   tree-sitter-rust = "0.20.0"     # Rust language support
   tree-sitter-python = "0.20.0"   # Python language support
   walkdir = "2.4.0"               # Directory traversal
   regex = "1.10.0"                # Pattern matching
   ```

### Expected Output Files
- **Modified:** `Cargo.toml` (root workspace file)
- **Validation:** `cargo check` should succeed without errors

## Success Criteria
- [ ] vector-search added to workspace members array
- [ ] New vector search dependencies added to workspace.dependencies
- [ ] Existing workspace structure preserved exactly
- [ ] `cargo check` runs without dependency resolution errors
- [ ] No formatting or structural changes to existing sections

## Common Pitfalls to Avoid
- Don't modify existing member order or structure
- Don't change existing dependency versions 
- Don't add duplicate dependencies that already exist
- Ensure proper TOML formatting and indentation
- Don't run full compilation yet (just `cargo check`)

## Context for Next Task
Task 00_2 will create the actual `crates/vector-search` directory structure and basic Cargo.toml that uses these workspace dependencies.

## Integration Notes
This task establishes the workspace-level integration point. The vector-search crate will later integrate with:
- `neuromorphic-core` for neural processing
- `snn-allocation-engine` for cortical column allocation
- `temporal-memory` for memory persistence
- Existing error handling and logging infrastructure