# Task 112: Add InconsistencyDetail Struct

## Prerequisites Check
- [ ] Task 111 completed: Health critical check method added
- [ ] Run: `cargo check` (should pass)

## Context
Add InconsistencyDetail struct for detailed inconsistency reporting.

## Task Objective
Define InconsistencyDetail to hold individual document inconsistency information.

## Steps
1. Add InconsistencyDetail struct:
   ```rust
   /// Individual inconsistency detail
   #[derive(Debug, Clone)]
   pub struct InconsistencyDetail {
       /// Document ID
       pub doc_id: String,
       /// Current consistency state
       pub state: ConsistencyState,
       /// Last update timestamp
       pub last_updated: Instant,
       /// Systems where document is present
       pub systems_present: Vec<String>,
   }
   ```

## Success Criteria
- [ ] InconsistencyDetail struct added
- [ ] All required fields included
- [ ] Proper field types
- [ ] Compiles without errors

## Time: 2 minutes