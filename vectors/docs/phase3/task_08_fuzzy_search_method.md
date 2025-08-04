# Task 08: Implement Fuzzy Search Method

**Estimated Time: 10 minutes**

## Context
You have wildcard and regex search implemented. Now add fuzzy search for typo tolerance using Tantivy's edit distance capabilities.

## Current System State
- `AdvancedPatternEngine` has wildcard and regex methods
- Need fuzzy matching to handle typos and spelling variations
- Tantivy supports fuzzy search with edit distance

## Your Task
Add the `search_fuzzy` method to `AdvancedPatternEngine`.

## Required Implementation

```rust
impl AdvancedPatternEngine {
    pub fn search_fuzzy(&self, term: &str, max_edit_distance: u8) -> Result<Vec<SearchResult>> {
        // Tantivy fuzzy search: "term"~edit_distance
        let fuzzy_query = format!("\"{}\"~{}", term, max_edit_distance);
        self.proximity_engine.boolean_engine.search_boolean(&fuzzy_query)
    }
    
    // Convenience method with default edit distance of 1
    pub fn search_fuzzy_default(&self, term: &str) -> Result<Vec<SearchResult>> {
        self.search_fuzzy(term, 1)
    }
}
```

## How Tantivy Fuzzy Search Works
- Syntax: `"term"~N` where N is maximum edit distance
- Edit distance counts: insertions, deletions, substitutions
- Examples:
  - `"function"~1` matches "funcion", "functio", "functions"
  - `"initialize"~2` matches "initialise", "intialize", "init"

## Edit Distance Examples
- Distance 0: Exact match only
- Distance 1: 1 character difference (most common for typos)
- Distance 2: 2 character differences (more permissive)
- Distance 3+: Very permissive (may return many false positives)

## Success Criteria
- [ ] `search_fuzzy` method with configurable edit distance
- [ ] `search_fuzzy_default` convenience method (distance = 1)
- [ ] Uses Tantivy's `"term"~N` syntax
- [ ] Delegates to boolean engine
- [ ] Returns `Result<Vec<SearchResult>>`
- [ ] Code compiles without errors

## Example Usage
```rust
// Find "function" with 1 edit distance tolerance
let results = pattern_engine.search_fuzzy("function", 1)?;
// Matches: "function", "funcion", "functio", "functions"

// Use default distance of 1
let results = pattern_engine.search_fuzzy_default("initialize")?;
// Matches: "initialize", "initialise", "intialize"
```

## Performance Considerations
- Higher edit distances are more expensive
- Edit distance 1-2 is usually sufficient for typos
- Fuzzy search is slower than exact matching
- Consider limiting results for very permissive distances

## Common Use Cases
- Handling coding typos: "funcion" → "function"
- British vs American spelling: "initialise" → "initialize" 
- Common misspellings: "lenght" → "length"
- Variable name variations: "userId" → "user_id"

## Next Task
Now that all search methods are implemented, you'll create comprehensive proximity search tests.