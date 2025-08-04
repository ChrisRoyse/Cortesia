# Task 006: Create Test Case Builder Pattern

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This builds on Tasks 001-005. The GroundTruthCase struct needs a fluent builder pattern to make creating test cases more ergonomic and less error-prone.

## Project Structure
```
src/
  validation/
    ground_truth.rs  <- Extend this file
  lib.rs
Cargo.toml
```

## Task Description
Implement a builder pattern for GroundTruthCase that provides a fluent API for constructing test cases. This will make test case creation more readable, catch errors early, and provide sensible defaults.

## Requirements
1. Add to existing `src/validation/ground_truth.rs`
2. Create `GroundTruthCaseBuilder` struct with fluent methods
3. Implement validation in the builder to catch errors early
4. Provide automatic query type detection with override capability
5. Add convenience methods for common patterns
6. Ensure builder validates before building
7. Add example usage patterns

## Expected Code Structure to Add
```rust
use anyhow::{anyhow, Result, Context};

#[derive(Debug, Clone)]
pub struct GroundTruthCaseBuilder {
    query: Option<String>,
    expected_files: Vec<String>,
    query_type: Option<QueryType>,
    must_contain: Vec<String>,
    must_not_contain: Vec<String>,
    auto_detect_type: bool,
}

impl GroundTruthCaseBuilder {
    pub fn new() -> Self {
        Self {
            query: None,
            expected_files: Vec::new(),
            query_type: None,
            must_contain: Vec::new(),
            must_not_contain: Vec::new(),
            auto_detect_type: true,
        }
    }
    
    pub fn query<S: Into<String>>(mut self, query: S) -> Self {
        self.query = Some(query.into());
        self
    }
    
    pub fn expects_files<I, S>(mut self, files: I) -> Self 
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.expected_files = files.into_iter().map(|s| s.into()).collect();
        self
    }
    
    pub fn expects_file<S: Into<String>>(mut self, file: S) -> Self {
        self.expected_files.push(file.into());
        self
    }
    
    pub fn expects_count(mut self, count: usize) -> Self {
        // Pad or truncate expected_files to match count
        self.expected_files.resize(count, String::new());
        self
    }
    
    pub fn query_type(mut self, query_type: QueryType) -> Self {
        self.query_type = Some(query_type);
        self.auto_detect_type = false;
        self
    }
    
    pub fn must_contain<I, S>(mut self, terms: I) -> Self 
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.must_contain = terms.into_iter().map(|s| s.into()).collect();
        self
    }
    
    pub fn must_contain_term<S: Into<String>>(mut self, term: S) -> Self {
        self.must_contain.push(term.into());
        self
    }
    
    pub fn must_not_contain<I, S>(mut self, terms: I) -> Self 
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.must_not_contain = terms.into_iter().map(|s| s.into()).collect();
        self
    }
    
    pub fn must_not_contain_term<S: Into<String>>(mut self, term: S) -> Self {
        self.must_not_contain.push(term.into());
        self
    }
    
    pub fn disable_auto_detect_type(mut self) -> Self {
        self.auto_detect_type = false;
        self
    }
    
    pub fn build(self) -> Result<GroundTruthCase> {
        // Validate required fields
        let query = self.query
            .ok_or_else(|| anyhow!("Query is required"))?;
        
        if query.trim().is_empty() {
            return Err(anyhow!("Query cannot be empty"));
        }
        
        // Determine query type
        let query_type = if let Some(explicit_type) = self.query_type {
            explicit_type
        } else if self.auto_detect_type {
            QueryType::from_query(&query)
        } else {
            return Err(anyhow!("Query type must be specified when auto-detection is disabled"));
        };
        
        // Validate content requirements don't conflict
        for must_contain in &self.must_contain {
            if self.must_not_contain.contains(must_contain) {
                return Err(anyhow!(
                    "Content requirement conflict: '{}' appears in both must_contain and must_not_contain",
                    must_contain
                ));
            }
        }
        
        // Build the case
        let case = GroundTruthCase {
            query,
            expected_files: self.expected_files,
            expected_count: self.expected_files.len(),
            must_contain: self.must_contain,
            must_not_contain: self.must_not_contain,
            query_type,
        };
        
        // Final validation
        let validation_errors = case.validate()?;
        if !validation_errors.is_empty() {
            let error_messages: Vec<String> = validation_errors
                .into_iter()
                .map(|e| e.message)
                .collect();
            return Err(anyhow!(
                "Test case validation failed: {}",
                error_messages.join(", ")
            ));
        }
        
        Ok(case)
    }
}

impl Default for GroundTruthCaseBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// Convenience constructors for common patterns
impl GroundTruthCaseBuilder {
    pub fn special_chars_query<S: Into<String>>(query: S) -> Self {
        Self::new()
            .query(query)
            .query_type(QueryType::SpecialCharacters)
    }
    
    pub fn boolean_and_query<S: Into<String>>(query: S) -> Self {
        Self::new()
            .query(query)
            .query_type(QueryType::BooleanAnd)
    }
    
    pub fn boolean_or_query<S: Into<String>>(query: S) -> Self {
        Self::new()
            .query(query)
            .query_type(QueryType::BooleanOr)
    }
    
    pub fn boolean_not_query<S: Into<String>>(query: S) -> Self {
        Self::new()
            .query(query)
            .query_type(QueryType::BooleanNot)
    }
    
    pub fn wildcard_query<S: Into<String>>(query: S) -> Self {
        Self::new()
            .query(query)
            .query_type(QueryType::Wildcard)
    }
    
    pub fn phrase_query<S: Into<String>>(query: S) -> Self {
        Self::new()
            .query(query)
            .query_type(QueryType::Phrase)
    }
    
    pub fn proximity_query<S: Into<String>>(query: S) -> Self {
        Self::new()
            .query(query)
            .query_type(QueryType::Proximity)
    }
    
    pub fn regex_query<S: Into<String>>(query: S) -> Self {
        Self::new()
            .query(query)
            .query_type(QueryType::Regex)
    }
    
    pub fn vector_query<S: Into<String>>(query: S) -> Self {
        Self::new()
            .query(query)
            .query_type(QueryType::Vector)
    }
    
    pub fn hybrid_query<S: Into<String>>(query: S) -> Self {
        Self::new()
            .query(query)
            .query_type(QueryType::Hybrid)
    }
}

// Extension method for GroundTruthDataset
impl GroundTruthDataset {
    pub fn add_case_from_builder(&mut self, builder: GroundTruthCaseBuilder) -> Result<()> {
        let case = builder.build()?;
        self.add_test(case);
        Ok(())
    }
    
    pub fn builder() -> GroundTruthCaseBuilder {
        GroundTruthCaseBuilder::new()
    }
}

#[cfg(test)]
mod builder_tests {
    use super::*;
    
    #[test]
    fn test_basic_builder() {
        let case = GroundTruthCaseBuilder::new()
            .query("[workspace]")
            .expects_files(vec!["Cargo.toml", "src/lib.rs"])
            .must_contain_term("[workspace]")
            .build()
            .expect("Should build successfully");
        
        assert_eq!(case.query, "[workspace]");
        assert_eq!(case.expected_files.len(), 2);
        assert_eq!(case.expected_count, 2);
        assert!(case.must_contain.contains(&"[workspace]".to_string()));
    }
    
    #[test]
    fn test_convenience_constructors() {
        let case = GroundTruthCaseBuilder::special_chars_query("[test]")
            .expects_file("test.rs")
            .build()
            .expect("Should build successfully");
        
        assert_eq!(case.query_type, QueryType::SpecialCharacters);
    }
    
    #[test]
    fn test_validation_errors() {
        // Empty query should fail
        let result = GroundTruthCaseBuilder::new()
            .query("")
            .build();
        assert!(result.is_err());
        
        // Conflicting content requirements should fail
        let result = GroundTruthCaseBuilder::new()
            .query("test")
            .must_contain_term("foo")
            .must_not_contain_term("foo")
            .build();
        assert!(result.is_err());
    }
}
```

## Example Usage Patterns
```rust
// Basic usage
let case = GroundTruthCaseBuilder::new()
    .query("[workspace]")
    .expects_files(vec!["Cargo.toml", "src/lib.rs"])
    .must_contain_term("[workspace]")
    .build()?;

// Using convenience constructors
let boolean_case = GroundTruthCaseBuilder::boolean_and_query("rust AND vector")
    .expects_files(vec!["src/vector.rs", "README.md"])
    .must_contain(vec!["rust", "vector"])
    .build()?;

// Chaining with dataset
let mut dataset = GroundTruthDataset::new();
dataset.add_case_from_builder(
    GroundTruthCaseBuilder::wildcard_query("*.rs")
        .expects_files(vec!["main.rs", "lib.rs"])
)?;
```

## Dependencies
Same as previous tasks - should already be in Cargo.toml

## Success Criteria
- Builder pattern compiles without errors
- Fluent API is ergonomic and readable
- Early validation catches configuration errors
- Convenience constructors work for common query types
- Auto-detection of query types works correctly
- Builder integrates smoothly with existing GroundTruthDataset
- Unit tests demonstrate proper usage and error handling
- Examples show clear usage patterns

## Time Limit
10 minutes maximum