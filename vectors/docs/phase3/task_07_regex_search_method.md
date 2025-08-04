# Task 07: Implement Regex Search Method

**Estimated Time: 10 minutes**

## Context
You have wildcard search implemented. Now add regex pattern support for complex pattern matching using Tantivy's regex capabilities.

## Current System State
- `AdvancedPatternEngine` has `search_wildcard` method
- `ProximitySearchEngine.boolean_engine` is public
- Need regex pattern support for advanced text matching

## Your Task
Add the `search_regex` method to `AdvancedPatternEngine`.

## Required Implementation

```rust
impl AdvancedPatternEngine {
    pub fn search_regex(&self, pattern: &str) -> Result<Vec<SearchResult>> {
        // For complex patterns, use Tantivy's regex support
        // Tantivy expects regex patterns wrapped in forward slashes
        let regex_query = format!("/{}/", pattern);
        self.proximity_engine.boolean_engine.search_boolean(&regex_query)
    }
}
```

## How Tantivy Regex Works
- Syntax: `/regex_pattern/`
- Uses standard regex syntax (Rust regex crate)
- Examples:
  - `/pub fn \w+\(\)/` matches "pub fn name()"
  - `/Result<.*?>/` matches "Result<String>", "Result<Vec<i32>>"
  - `/\w+Error$/` matches words ending in "Error"

## Security Considerations
- Regex can be computationally expensive
- Tantivy has built-in protections against ReDoS attacks
- Complex patterns may be slower than wildcards

## Success Criteria
- [ ] `search_regex` method added to `AdvancedPatternEngine`
- [ ] Method takes single `pattern` parameter (raw regex)
- [ ] Wraps pattern in `/` for Tantivy regex syntax
- [ ] Delegates to boolean engine
- [ ] Returns `Result<Vec<SearchResult>>`
- [ ] Code compiles without errors

## Example Usage
```rust
// Find function declarations
let results = pattern_engine.search_regex(r"pub fn \w+\(\)")?;

// Find generic Result types
let results = pattern_engine.search_regex(r"Result<.*?>")?;

// Find error types
let results = pattern_engine.search_regex(r"\w+Error")?;
```

## Common Regex Patterns for Code Search
- Function patterns: `r"fn \w+\(.*?\)"`
- Type patterns: `r"struct \w+"`
- Generic patterns: `r"<.*?>"`
- Error patterns: `r"\w+(Error|Exception)"`

## Performance Notes
- Simple wildcards are faster than regex
- Use regex for complex patterns that wildcards can't handle
- Consider query complexity vs. performance trade-offs

## Next Task
After regex search, you'll implement fuzzy search for typo tolerance.