# Task 002: Add arrow-array Dependency to Cargo.toml

## Prerequisites Check
- [ ] Task 001 completed: lancedb dependency added
- [ ] Cargo.toml contains lancedb = "0.4"
- [ ] [dependencies] section is properly formatted
- [ ] Run: `cargo check` (should download lancedb successfully)

## Context
Continuing LanceDB dependency setup. Arrow-array is required for handling vector data arrays.

## Task Objective
Add exactly one line to Cargo.toml: `arrow-array = "50.0"`

## Steps
1. Open Cargo.toml in editor
2. Find the [dependencies] section (where lancedb was just added)
3. Add the line: `arrow-array = "50.0"`
4. Save file

## Success Criteria
- [ ] Line `arrow-array = "50.0"` added to [dependencies] section
- [ ] File saves successfully
- [ ] Dependencies are properly formatted

## Time: 2 minutes

## Next Task
Task 003: Add arrow-schema dependency