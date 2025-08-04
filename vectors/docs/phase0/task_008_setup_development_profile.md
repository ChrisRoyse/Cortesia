# Micro-Task 008: Setup Development Profile

## Objective
Configure Cargo build profiles for development and release optimization.

## Context
Proper build profiles ensure fast compilation during development and optimal performance in release builds. This is crucial for the vector search system's performance characteristics.

## Prerequisites
- Task 007 completed (Core dependencies installed)
- Cargo.toml workspace file exists

## Time Estimate
7 minutes

## Instructions
1. Add profile configurations to `Cargo.toml`:
   ```toml
   [profile.dev]
   opt-level = 0
   debug = true
   debug-assertions = true
   overflow-checks = true
   incremental = true
   
   [profile.release]
   opt-level = 3
   debug = false
   debug-assertions = false
   overflow-checks = false
   lto = true
   codegen-units = 1
   ```
2. Verify configuration is valid: `cargo check --workspace`
3. Commit changes: `git add Cargo.toml && git commit -m "Configure build profiles"`

## Expected Output
- Development profile optimized for fast compilation
- Release profile optimized for performance
- Configuration validated and committed

## Success Criteria
- [ ] `[profile.dev]` section added to Cargo.toml
- [ ] `[profile.release]` section added to Cargo.toml
- [ ] `cargo check --workspace` runs without TOML errors
- [ ] Changes committed to Git

## Next Task
task_009_create_vscode_configuration.md