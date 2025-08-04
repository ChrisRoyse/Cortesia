# Task 04: Implement NEAR Query Parsing

**Estimated Time: 10 minutes**

## Context
Users want natural language proximity queries like "pub NEAR/3 fn". You need to parse these queries and convert them to Tantivy syntax.

## Current System State
- `ProximitySearchEngine` has `search_proximity` and `search_phrase` methods
- Need to handle user-friendly NEAR syntax: "term1 NEAR/distance term2"

## Your Task
Add `parse_near_query` helper method and `search_near` method to handle NEAR syntax.

## Required Implementation

```rust
use regex::Regex;

impl ProximitySearchEngine {
    fn parse_near_query(&self, query: &str) -> Option<regex::Captures> {
        // Parse queries like: "pub NEAR/3 fn"
        let near_regex = Regex::new(r#"(\w+)\s+NEAR/(\d+)\s+(\w+)"#).ok()?;
        near_regex.captures(query)
    }
    
    pub fn search_near(&self, query: &str) -> Result<Vec<SearchResult>> {
        // Support queries like: "pub NEAR/3 fn" 
        // Convert to Tantivy syntax: "pub"~3 AND "fn"~3
        if let Some(captures) = self.parse_near_query(query) {
            let term1 = &captures[1];
            let distance = captures[2].parse::<u32>().unwrap_or(5);
            let term2 = &captures[3];
            
            let near_query = format!("(\"{}\"~{} AND \"{}\"~{})", term1, distance, term2, distance);
            self.boolean_engine.search_boolean(&near_query)
        } else {
            // If not a NEAR query, pass through to boolean search
            self.boolean_engine.search_boolean(query)
        }
    }
}
```

## Dependencies to Add
Add to top of file: `use regex::Regex;`

## How NEAR Queries Work
- Input: "pub NEAR/3 fn"
- Parsed: term1="pub", distance=3, term2="fn"  
- Output: `("pub"~3 AND "fn"~3)`
- Finds documents where both terms appear within 3 words of something

## Success Criteria
- [ ] `parse_near_query` helper method implemented
- [ ] Uses regex to parse "term1 NEAR/N term2" pattern
- [ ] `search_near` method converts NEAR syntax to Tantivy
- [ ] Falls back to boolean search for non-NEAR queries
- [ ] Handles invalid distance numbers (defaults to 5)
- [ ] Code compiles without errors

## Example Usage
```rust
let results = proximity_engine.search_near("pub NEAR/3 fn")?;
// Finds documents with "pub" and "fn" each within 3 words of other content
```

## Next Task
Create the `AdvancedPatternEngine` struct that wraps proximity search.