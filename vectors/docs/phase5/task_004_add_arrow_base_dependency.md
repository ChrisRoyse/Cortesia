# Task 004: Add Base Arrow Dependency to Cargo.toml

## Prerequisites Check
- [ ] Task 003 completed: arrow-schema dependency added
- [ ] Cargo.toml contains arrow-schema = "50.0"
- [ ] All Arrow dependencies have matching versions
- [ ] Run: `cargo check` (may have version conflicts to resolve)

## Context
Final Arrow dependency needed. Base arrow crate provides core functionality.

## Task Objective
Add exactly one line to Cargo.toml: `arrow = "50.0"`

## Steps
1. Open Cargo.toml in editor
2. Find the [dependencies] section
3. Add the line: `arrow = "50.0"`
4. Save file

## Success Criteria
- [ ] Line `arrow = "50.0"` added to [dependencies] section
- [ ] File saves successfully
- [ ] All Arrow versions match (50.0)

## Time: 2 minutes

## Next Task
Task 005: Verify tokio dependency exists