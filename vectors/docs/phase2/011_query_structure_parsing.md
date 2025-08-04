# Task 011: Query Structure Parsing Implementation

## Prerequisites
- Task 001-010 completed: DocumentLevelValidator structure created
- BooleanQueryStructure enum available from validator.rs (Task 002)
- DocumentLevelValidator struct with basic validation method

## Required Imports
```rust
// Add to src/validator.rs for this task
use anyhow::{Result, Context};
use std::collections::HashMap;
use regex::Regex;
use crate::boolean::{BooleanSearchEngine, SearchResult};
// BooleanQueryStructure already defined in this file from Task 002
```

## Context
You have a basic DocumentLevelValidator, but it only handles simple AND queries. Now you need to implement comprehensive query parsing that can handle OR, NOT, and nested expressions.

The goal is to parse query strings like:
- "pub AND struct" → BooleanQueryStructure::And([pub, struct])
- "fn OR impl" → BooleanQueryStructure::Or([fn, impl])  
- "pub NOT Error" → BooleanQueryStructure::Not{include: pub, exclude: Error}
- "(pub AND struct) OR fn" → BooleanQueryStructure::Complex([...])

## Your Task (10 minutes max)
Implement comprehensive query structure parsing for all boolean operators.

## Success Criteria
1. Write failing tests for OR and NOT query parsing
2. Implement OR query parsing functionality
3. Implement NOT query parsing functionality
4. Handle basic nested expressions with parentheses
5. All tests pass after implementation

## Implementation Steps

### 1. RED: Write failing tests for comprehensive parsing
```rust
// Add to src/validator.rs tests
#[test]
fn test_parse_or_queries() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    create_test_index(&index_path)?;
    let boolean_engine = BooleanSearchEngine::new(&index_path)?;
    let validator = DocumentLevelValidator::new(boolean_engine);
    
    // Test OR query parsing
    let parsed = validator.parse_boolean_query("struct OR fn")?;
    
    match parsed {
        BooleanQueryStructure::Or(terms) => {
            assert_eq!(terms.len(), 2);
            assert!(terms.contains(&"struct".to_string()));
            assert!(terms.contains(&"fn".to_string()));
        },
        _ => panic!("Should parse as OR query"),
    }
    
    Ok(())
}

#[test]
fn test_parse_not_queries() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    create_test_index(&index_path)?;
    let boolean_engine = BooleanSearchEngine::new(&index_path)?;
    let validator = DocumentLevelValidator::new(boolean_engine);
    
    // Test NOT query parsing
    let parsed = validator.parse_boolean_query("pub NOT Error")?;
    
    match parsed {
        BooleanQueryStructure::Not { include, exclude } => {
            assert_eq!(include, "pub");
            assert_eq!(exclude, "error"); // Should be lowercase
        },
        _ => panic!("Should parse as NOT query"),
    }
    
    Ok(())
}

#[test]
fn test_parse_complex_queries() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    create_test_index(&index_path)?;
    let boolean_engine = BooleanSearchEngine::new(&index_path)?;
    let validator = DocumentLevelValidator::new(boolean_engine);
    
    // Test simple parenthetical expression
    let parsed = validator.parse_boolean_query("(pub AND struct) OR fn")?;
    
    // Should parse as Complex structure
    match parsed {
        BooleanQueryStructure::Complex(_) => {
            // Basic validation - detailed validation in next task
        },
        _ => panic!("Should parse as Complex query for parenthetical expressions"),
    }
    
    Ok(())
}
```

### 2. GREEN: Implement comprehensive parse_boolean_query
```rust
// Update parse_boolean_query method in DocumentLevelValidator
impl DocumentLevelValidator {
    fn parse_boolean_query(&self, query: &str) -> Result<BooleanQueryStructure> {
        let query = query.trim();
        
        // Handle parentheses first (simple case)
        if query.contains('(') && query.contains(')') {
            return self.parse_complex_query(query);
        }
        
        // Handle NOT queries
        if query.contains(" NOT ") {
            return self.parse_not_query(query);
        }
        
        // Handle OR queries
        if query.contains(" OR ") {
            return self.parse_or_query(query);
        }
        
        // Handle AND queries (including implicit AND)
        if query.contains(" AND ") {
            return self.parse_and_query(query);
        }
        
        // Single term query - treat as AND with one term
        Ok(BooleanQueryStructure::And(vec![query.to_lowercase()]))
    }
    
    fn parse_and_query(&self, query: &str) -> Result<BooleanQueryStructure> {
        let terms: Vec<String> = query.split(" AND ")
            .map(|s| s.trim().to_lowercase())
            .collect();
        Ok(BooleanQueryStructure::And(terms))
    }
    
    fn parse_or_query(&self, query: &str) -> Result<BooleanQueryStructure> {
        let terms: Vec<String> = query.split(" OR ")
            .map(|s| s.trim().to_lowercase())
            .collect();
        Ok(BooleanQueryStructure::Or(terms))
    }
    
    fn parse_not_query(&self, query: &str) -> Result<BooleanQueryStructure> {
        let parts: Vec<&str> = query.split(" NOT ").collect();
        if parts.len() != 2 {
            return Err(anyhow::anyhow!("Invalid NOT query format: {}", query));
        }
        
        Ok(BooleanQueryStructure::Not {
            include: parts[0].trim().to_lowercase(),
            exclude: parts[1].trim().to_lowercase(),
        })
    }
    
    fn parse_complex_query(&self, query: &str) -> Result<BooleanQueryStructure> {
        // Simple implementation for basic parenthetical expressions
        // For now, just treat as complex and delegate to Tantivy's parsing
        
        // This is a simplified implementation - a full parser would be more complex
        if query.contains(") OR ") {
            // Handle "(something) OR something_else"
            let parts: Vec<&str> = query.split(" OR ").collect();
            let mut sub_queries = Vec::new();
            
            for part in parts {
                let clean_part = part.trim_matches(|c| c == '(' || c == ')').trim();
                let sub_query = self.parse_boolean_query(clean_part)?;
                sub_queries.push(sub_query);
            }
            
            return Ok(BooleanQueryStructure::Complex(sub_queries));
        }
        
        // Default: treat as complex but parse the inner content
        let inner = query.trim_matches(|c| c == '(' || c == ')')
            .trim();
        Ok(BooleanQueryStructure::Complex(vec![self.parse_boolean_query(inner)?]))
    }
}
```

### 3. REFACTOR: Update document_satisfies_query for all types
```rust
// Update document_satisfies_query method
impl DocumentLevelValidator {
    fn document_satisfies_query(&self, content: &str, query: &BooleanQueryStructure) -> Result<bool> {
        let content_lower = content.to_lowercase();
        
        match query {
            BooleanQueryStructure::And(terms) => {
                Ok(terms.iter().all(|term| content_lower.contains(term)))
            }
            BooleanQueryStructure::Or(terms) => {
                Ok(terms.iter().any(|term| content_lower.contains(term)))
            }
            BooleanQueryStructure::Not { include, exclude } => {
                Ok(content_lower.contains(include) && !content_lower.contains(exclude))
            }
            BooleanQueryStructure::Complex(sub_queries) => {
                // For complex queries, evaluate each sub-query
                // This is simplified - real implementation would handle OR/AND between sub-queries
                for sub_query in sub_queries {
                    if !self.document_satisfies_query(content, sub_query)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
        }
    }
}
```

### 4. Add comprehensive validation tests
```rust
#[test]
fn test_validation_with_parsed_queries() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    create_validation_test_index(&index_path)?;
    let boolean_engine = BooleanSearchEngine::new(&index_path)?;
    let validator = DocumentLevelValidator::new(boolean_engine);
    
    // Test various query types with validation
    let test_cases = vec![
        ("pub AND struct", true, "pub struct Data {}"),
        ("pub AND struct", false, "pub fn test() {}"), // Missing struct
        ("fn OR impl", true, "fn test() {}"),
        ("fn OR impl", true, "impl Display for T {}"),
        ("fn OR impl", false, "pub struct Data {}"), // Missing both
        ("pub NOT Error", true, "pub fn test() {}"),
        ("pub NOT Error", false, "pub fn test() -> Result<(), Error> {}"), // Has Error
    ];
    
    for (query, should_pass, content) in test_cases {
        let result = SearchResult {
            file_path: "test.rs".to_string(),
            content: content.to_string(),
            chunk_index: 0,
            score: 1.0,
        };
        
        let is_valid = validator.validate_boolean_results(query, &[result])?;
        assert_eq!(is_valid, should_pass, 
                  "Query '{}' with content '{}' should {}", 
                  query, content, if should_pass { "pass" } else { "fail" });
    }
    
    Ok(())
}

fn create_validation_test_index(index_path: &Path) -> Result<()> {
    // Similar to create_test_index but with more diverse content
    use tantivy::schema::{Schema, TEXT, STORED};
    use tantivy::{Index, doc};
    
    let mut schema_builder = Schema::builder();
    let file_path_field = schema_builder.add_text_field("file_path", TEXT | STORED);
    let content_field = schema_builder.add_text_field("content", TEXT | STORED);
    let raw_content_field = schema_builder.add_text_field("raw_content", TEXT | STORED);
    let chunk_index_field = schema_builder.add_u64_field("chunk_index", STORED);
    let schema = schema_builder.build();
    
    let index = Index::create_in_dir(index_path, schema.clone())?;
    let mut index_writer = index.writer(50_000_000)?;
    
    index_writer.add_document(doc!(
        file_path_field => "test.rs",
        content_field => "test content",
        raw_content_field => "test content",
        chunk_index_field => 0u64
    ))?;
    
    index_writer.commit()?;
    Ok(())
}
```

## Validation Checklist
- [ ] OR query parsing works correctly
- [ ] NOT query parsing works correctly  
- [ ] Basic complex query parsing handles parentheses
- [ ] document_satisfies_query handles all query types
- [ ] Validation tests pass for all query types

## Context for Next Task
Next task will implement the CrossChunkBooleanHandler to handle boolean logic that spans across multiple document chunks.