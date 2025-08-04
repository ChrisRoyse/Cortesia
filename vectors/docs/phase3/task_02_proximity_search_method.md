# Task 02: Implement Proximity Search Method

**Estimated Time: 10 minutes**

## Context
You have created the `ProximitySearchEngine` struct. Now you need to implement the core proximity search functionality that finds two terms within a specified distance of each other using Tantivy's proximity syntax.

## Current System State
- `ProximitySearchEngine` struct exists in `src/proximity.rs`
- Has `boolean_engine` field of type `BooleanSearchEngine`
- `new()` constructor is implemented

## Your Task
Add the `search_proximity` method to `ProximitySearchEngine` that searches for two terms within a maximum distance.

## Required Implementation

```rust
impl ProximitySearchEngine {
    pub fn search_proximity(&self, term1: &str, term2: &str, max_distance: u32) -> Result<Vec<SearchResult>> {
        // Use Tantivy's proximity syntax: "term1"~distance "term2"
        let proximity_query = format!("\"{}\"~{} \"{}\"", term1, max_distance, term2);
        self.boolean_engine.search_boolean(&proximity_query)
    }
}
```

## How Tantivy Proximity Works
- Syntax: `"term1"~distance "term2"` 
- Distance 0 means adjacent terms
- Distance 1 means one word can be between them
- Distance N means up to N words can be between them

## Success Criteria
- [ ] `search_proximity` method added to `ProximitySearchEngine`
- [ ] Method takes `term1`, `term2`, and `max_distance` parameters
- [ ] Uses correct Tantivy proximity syntax format
- [ ] Delegates to `boolean_engine.search_boolean()`
- [ ] Returns `Result<Vec<SearchResult>>`
- [ ] Code compiles without errors

## Example Usage
```rust
let results = proximity_engine.search_proximity("pub", "fn", 1)?;
// Finds: "pub fn", "pub static fn", but not "pub struct X { fn"
```

## Next Task
After this method, you'll implement phrase search functionality.