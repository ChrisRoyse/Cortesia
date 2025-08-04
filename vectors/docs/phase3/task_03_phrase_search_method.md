# Task 03: Implement Phrase Search Method

**Estimated Time: 10 minutes**

## Context
You have `ProximitySearchEngine` with proximity search. Now implement exact phrase matching using Tantivy's phrase query syntax with quotes.

## Current System State
- `ProximitySearchEngine` exists with `search_proximity` method
- Boolean engine handles underlying query execution
- Need exact phrase matching (terms in exact order)

## Your Task
Add the `search_phrase` method for exact phrase matching.

## Required Implementation

```rust
impl ProximitySearchEngine {
    pub fn search_phrase(&self, phrase: &str) -> Result<Vec<SearchResult>> {
        // Use Tantivy's phrase queries with quotes
        let phrase_query = format!("\"{}\"", phrase);
        self.boolean_engine.search_boolean(&phrase_query)
    }
}
```

## How Tantivy Phrase Search Works
- Syntax: `"exact phrase here"`
- Finds terms in exact order with no words between
- Case-sensitive matching
- Punctuation and spacing must match exactly

## Success Criteria
- [ ] `search_phrase` method added to `ProximitySearchEngine`
- [ ] Method takes single `phrase` parameter
- [ ] Wraps phrase in double quotes for Tantivy
- [ ] Delegates to `boolean_engine.search_boolean()`
- [ ] Returns `Result<Vec<SearchResult>>`
- [ ] Code compiles without errors

## Example Usage
```rust
let results = proximity_engine.search_phrase("pub fn initialize")?;
// Finds: "pub fn initialize()" 
// Does NOT find: "pub static fn initialize" or "fn pub initialize"
```

## Test Cases to Consider
- Exact phrase matches
- Wrong word order (should not match)
- Extra words between (should not match)
- Generic types like `Result<T, E>`

## Next Task
After phrase search, you'll implement NEAR query parsing for natural language queries.