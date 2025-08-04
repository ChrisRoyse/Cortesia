# Task 003: Add arrow-schema Dependency to Cargo.toml

## Prerequisites Check
- [ ] Task 002 completed: arrow-array dependency added
- [ ] Cargo.toml contains arrow-array = "50.0"
- [ ] Both lancedb and arrow-array dependencies present
- [ ] Run: `cargo check` (may show warnings but should not fail)

## Context
Continuing Arrow dependencies. Arrow-schema is needed for defining data structure schemas.

## Task Objective
Add exactly one line to Cargo.toml: `arrow-schema = "50.0"`

## Steps
1. Open Cargo.toml in editor
2. Find the [dependencies] section
3. Add the line: `arrow-schema = "50.0"`
4. Save file

## Success Criteria
- [ ] Line `arrow-schema = "50.0"` added to [dependencies] section
- [ ] File saves successfully
- [ ] Version matches arrow-array (50.0)

## Time: 2 minutes

## Next Task
Task 004: Add base arrow dependency