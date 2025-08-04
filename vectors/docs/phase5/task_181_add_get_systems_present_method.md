# Task 114: Add Get Systems Present Method

## Prerequisites Check
- [ ] Task 113 completed: InconsistencyReport struct added
- [ ] Run: `cargo check` (should pass)

## Context
Add helper method to determine which systems contain a specific document.

## Task Objective
Implement get_systems_present method for document system tracking.

## Steps
1. Add get_systems_present helper method to ConsistencyManager:
   ```rust
   impl ConsistencyManager {
       /// Get systems where document is present
       fn get_systems_present(&self, doc_version: &DocumentVersion) -> Vec<String> {
           let mut systems = Vec::new();
           if doc_version.text_version.is_some() {
               systems.push("text".to_string());
           }
           if doc_version.vector_version.is_some() {
               systems.push("vector".to_string());
           }
           if doc_version.cache_version.is_some() {
               systems.push("cache".to_string());
           }
           systems
       }
   }
   ```

## Success Criteria
- [ ] Get systems present method added
- [ ] Checks all system version fields
- [ ] Returns vector of system names
- [ ] Compiles without errors

## Time: 3 minutes