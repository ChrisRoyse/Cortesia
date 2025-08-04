# Task 06: Implement Wildcard Search Method

**Estimated Time: 10 minutes**

## Context
You have `AdvancedPatternEngine` with proximity search delegation. Now implement wildcard pattern matching using Tantivy's native wildcard support.

## Current System State
- `AdvancedPatternEngine` exists in `src/patterns.rs`
- Has `proximity_engine` field and delegation methods
- Need wildcard pattern support (* and ? characters)

## Your Task
Add the `search_wildcard` method to `AdvancedPatternEngine`.

## Required Implementation

```rust
impl AdvancedPatternEngine {
    pub fn search_wildcard(&self, pattern: &str) -> Result<Vec<SearchResult>> {
        // Tantivy supports wildcards natively: "test*", "?ub", "struct*"
        // Pass pattern directly to boolean engine - Tantivy handles the wildcards
        self.proximity_engine.boolean_engine.search_boolean(pattern)
    }
}
```

**Note:** The `boolean_engine` field is already public in `ProximitySearchEngine` from Task 01, so this access will work correctly.

## How Tantivy Wildcards Work
- `*` matches zero or more characters
- `?` matches exactly one character
- Examples:
  - `test*` matches "test", "testing", "tests"
  - `?ub` matches "pub", "sub", but not "hub" or "stub"
  - `get_*_by_*` matches "get_user_by_id", "get_data_by_key"

## Success Criteria
- [ ] `search_wildcard` method added to `AdvancedPatternEngine`
- [ ] Method takes single `pattern` parameter with wildcards
- [ ] Passes pattern directly to boolean engine (Tantivy handles wildcards)
- [ ] Returns `Result<Vec<SearchResult>>`
- [ ] Update `ProximitySearchEngine.boolean_engine` to be public
- [ ] Code compiles without errors

## Example Usage
```rust
let results = pattern_engine.search_wildcard("Spike*Network")?;
// Finds: "SpikingNeuralNetwork", "SpikeNetwork", etc.

let results = pattern_engine.search_wildcard("test?")?; 
// Finds: "test1", "test2", "testa", but not "test" or "test10"
```

## Common Wildcard Patterns
- Function prefixes: `get_*`, `set_*`, `process_*`
- Type suffixes: `*Error`, `*Result`, `*Config`
- Mixed patterns: `*_by_*`, `handle_*_event`

## Next Task
After wildcard search, you'll implement regex pattern support.