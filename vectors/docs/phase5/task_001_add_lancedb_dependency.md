# Task 001: Add lancedb Dependency to Cargo.toml

## Prerequisites Check
- [ ] Project workspace is properly initialized
- [ ] Cargo.toml exists in project root
- [ ] [dependencies] section exists in Cargo.toml
- [ ] Run: `cargo check` (should pass with existing dependencies)

## Context
Starting Phase 5 LanceDB integration. First atomic step is adding the core LanceDB dependency.

## Task Objective
Add exactly one line to Cargo.toml: `lancedb = "0.4"`

## Steps
1. Open Cargo.toml in editor
2. Find the [dependencies] section
3. Add the line: `lancedb = "0.4"`
4. Save file

## Success Criteria
- [ ] Line `lancedb = "0.4"` added to [dependencies] section
- [ ] File saves successfully
- [ ] No syntax errors in Cargo.toml

## Time: 3 minutes

## Next Task
Task 002: Add arrow-array dependency