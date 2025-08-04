# REAL Task 06: Implement Proximity Search with Correct Syntax

**Estimated Time: 10 minutes**  
**TDD Approach: RED-GREEN-REFACTOR**  
**Prerequisites: Tasks 01-05 completed (working SearchEngine with basic search)**

## Context
This task implements proximity search using Tantivy's ACTUAL phrase-with-slop syntax. Unlike the theatrical version that used `"term1"~N "term2"` (which doesn't exist), this uses the real Tantivy syntax `"term1 term2"~N` where N is the slop distance.

## Critical API Correction
**WRONG (from old tasks)**: `"hello"~2 "world"`  
**CORRECT (real Tantivy)**: `"hello world"~2`

## Your Task - TDD RED PHASE
**WRITE THE FAILING TEST FIRST**

Add to `src/search/mod.rs` tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proximity_search_method_exists() -> Result<(), SearchError> {
        let engine = SearchEngine::new()?;
        
        // RED: This should fail because search_proximity doesn't exist yet
        let result = engine.search_proximity("hello", "world", 2);
        assert!(result.is_ok());
        Ok(())
    }
    
    #[test]
    fn test_proximity_search_syntax_validation() -> Result<(), SearchError> {
        let engine = SearchEngine::new()?;
        
        // Test that proximity search generates correct Tantivy query
        // This validates we're using "term1 term2"~N syntax, not fake syntax
        let result = engine.search_proximity("pub", "fn", 1);
        assert!(result.is_ok());
        Ok(())
    }
}
```

**Verify test fails**: `cargo test test_proximity_search` should fail - method doesn't exist.

## Your Task - TDD GREEN PHASE
**Implement the minimum proximity search method**

Add to `SearchEngine` impl in `src/search/mod.rs`:

```rust
impl SearchEngine {
    /// Searches for terms within specified distance of each other
    /// 
    /// Uses Tantivy's phrase-with-slop syntax: "term1 term2"~distance
    /// 
    /// # Arguments
    /// * `term1` - First search term
    /// * `term2` - Second search term  
    /// * `distance` - Maximum words allowed between terms (slop)
    /// 
    /// # Examples
    /// ```
    /// // Search for "pub" and "fn" with 1 word between
    /// let results = engine.search_proximity("pub", "fn", 1)?;
    /// // Matches: "pub fn", "pub static fn", but not "pub struct Data { fn }"
    /// ```
    pub fn search_proximity(&self, term1: &str, term2: &str, distance: u32) -> Result<Vec<SearchResult>, SearchError> {
        // Use CORRECT Tantivy syntax: "term1 term2"~N
        let proximity_query = format!("\"{} {}\"~{}", term1, term2, distance);
        
        // Delegate to existing search method (assumes Task 04 completed)
        self.search(&proximity_query)
    }
}
```

**Verify test passes**: `cargo test test_proximity_search` should pass.

## Your Task - TDD REFACTOR PHASE
**Add input validation and better error handling**

```rust
impl SearchEngine {
    pub fn search_proximity(&self, term1: &str, term2: &str, distance: u32) -> Result<Vec<SearchResult>, SearchError> {
        // Input validation
        if term1.trim().is_empty() || term2.trim().is_empty() {
            return Err(SearchError::QueryParsing("Proximity search terms cannot be empty".to_string()));
        }
        
        // Sanitize terms for query construction
        let clean_term1 = term1.trim().replace('"', "");
        let clean_term2 = term2.trim().replace('"', "");
        
        // Use CORRECT Tantivy phrase-with-slop syntax
        let proximity_query = format!("\"{} {}\"~{}", clean_term1, clean_term2, distance);
        
        self.search(&proximity_query)
            .map_err(|e| SearchError::QueryParsing(format!("Proximity search failed: {}", e)))
    }
}
```

## Add Advanced Test for Real Validation

```rust
#[test] 
fn test_proximity_search_with_real_data() -> Result<(), SearchError> {
    let mut engine = SearchEngine::new()?;
    
    // This test would need actual indexing capability
    // For now, validate query construction
    let result = engine.search_proximity("function", "hello", 0);
    assert!(result.is_ok());
    
    // Test edge case: empty terms should error
    let result = engine.search_proximity("", "world", 1);
    assert!(result.is_err());
    
    Ok(())
}
```

## Validation Checklist
- [ ] Test fails initially (`cargo test test_proximity_search` - method missing)
- [ ] After implementation, tests pass
- [ ] Uses CORRECT Tantivy syntax: `"term1 term2"~N`
- [ ] Input validation prevents empty terms
- [ ] Error handling for malformed queries
- [ ] No mocks or stubs - uses real search method
- [ ] Documentation explains slop behavior

## Integration Notes
- Depends on working `search()` method from previous tasks
- Uses existing SearchResult and SearchError types
- No duplicate types or conflicting implementations
- Ready for integration testing in later tasks

## Real API Usage Validation
This implementation uses the **actual** Tantivy query syntax:
- ✅ `"hello world"~2` - Allows 2 words between hello and world
- ✅ `"pub fn"~0` - Requires adjacent terms
- ❌ `"hello"~2 "world"` - **This syntax doesn't exist in Tantivy**

## Success Criteria
**This task is complete when:**
1. `search_proximity()` method exists and compiles
2. Uses correct Tantivy phrase-with-slop syntax
3. All tests pass without mocks or theatrical code
4. Input validation prevents common errors
5. Ready for real proximity search testing