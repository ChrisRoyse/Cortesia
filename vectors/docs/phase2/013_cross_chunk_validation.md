# Task 013: Cross-Chunk Boolean Validation

## Prerequisites
- Task 001-012 completed: CrossChunkBooleanHandler structure created
- DocumentResult struct available from cross_chunk.rs (Task 012)
- SearchResult struct available from boolean.rs (Task 005)
- BooleanQueryStructure enum available from validator.rs (Task 002)

## Required Imports
```rust
// Add to src/cross_chunk.rs for this task
use anyhow::{Result, Context};
use crate::boolean::{BooleanSearchEngine, SearchResult};
use crate::validator::{DocumentLevelValidator, BooleanQueryStructure};
use std::collections::HashMap;
use tempfile::TempDir;
```

## Context
You have basic cross-chunk functionality, but you need to validate that the combined document content actually satisfies the boolean query requirements. Just because individual chunks matched doesn't guarantee the complete document satisfies the boolean logic.

This validation ensures:
- AND queries: ALL terms exist somewhere in the combined document
- OR queries: AT LEAST ONE term exists in the combined document  
- NOT queries: Include terms exist but exclude terms don't exist anywhere
- Complex queries: Nested logic is properly evaluated

## Your Task (10 minutes max)
Implement comprehensive boolean validation for cross-chunk results.

## Success Criteria
1. Write failing test for cross-chunk AND validation
2. Implement document_satisfies_boolean_query method
3. Write failing test for cross-chunk OR and NOT validation
4. Integrate validation into search_across_chunks
5. All tests pass with proper validation

## Implementation Steps

### 1. RED: Write failing test for cross-chunk AND validation
```rust
// Add to src/cross_chunk.rs tests
#[test]
fn test_cross_chunk_and_validation() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    create_cross_chunk_validation_index(&index_path)?;
    let boolean_engine = BooleanSearchEngine::new(&index_path)?;
    let handler = CrossChunkBooleanHandler::new(boolean_engine);
    
    // Search for terms that span chunks: "pub" in chunk 0, "Display" in chunk 2
    let results = handler.search_across_chunks("pub AND Display")?;
    
    // Should find document where both terms exist across chunks
    assert_eq!(results.len(), 1, "Should find document with terms across chunks");
    assert_eq!(results[0].file_path, "spanning_terms.rs");
    
    // Verify the combined content has both terms
    assert!(results[0].content.contains("pub"));
    assert!(results[0].content.contains("Display"));
    
    // Search for terms where AND fails - one term missing
    let results = handler.search_across_chunks("pub AND NonExistent")?;
    assert!(results.is_empty(), "Should not find documents missing required AND terms");
    
    Ok(())
}

fn create_cross_chunk_validation_index(index_path: &Path) -> Result<()> {
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
    
    // Document with terms spanning chunks
    index_writer.add_document(doc!(
        file_path_field => "spanning_terms.rs",
        content_field => "pub struct LargeStruct {",
        raw_content_field => "pub struct LargeStruct {",
        chunk_index_field => 0u64
    ))?;
    
    index_writer.add_document(doc!(
        file_path_field => "spanning_terms.rs",
        content_field => "    data: Vec<String>,",
        raw_content_field => "    data: Vec<String>,",
        chunk_index_field => 1u64
    ))?;
    
    index_writer.add_document(doc!(
        file_path_field => "spanning_terms.rs",
        content_field => "} impl Display for LargeStruct {",
        raw_content_field => "} impl Display for LargeStruct {",
        chunk_index_field => 2u64
    ))?;
    
    // Document with only some terms
    index_writer.add_document(doc!(
        file_path_field => "partial_match.rs",
        content_field => "pub fn test() {",
        raw_content_field => "pub fn test() {",
        chunk_index_field => 0u64
    ))?;
    
    index_writer.add_document(doc!(
        file_path_field => "partial_match.rs",
        content_field => "    println!(\"Hello\");",
        raw_content_field => "    println!(\"Hello\");",
        chunk_index_field => 1u64
    ))?;
    
    // Document with no matching terms
    index_writer.add_document(doc!(
        file_path_field => "no_match.rs",
        content_field => "fn helper() -> i32 { 42 }",
        raw_content_field => "fn helper() -> i32 { 42 }",
        chunk_index_field => 0u64
    ))?;
    
    index_writer.commit()?;
    Ok(())
}
```

### 2. GREEN: Implement document_satisfies_boolean_query method
```rust
// Add to CrossChunkBooleanHandler impl
use crate::validator::{DocumentLevelValidator, BooleanQueryStructure};

impl CrossChunkBooleanHandler {
    fn document_satisfies_boolean_query(&self, chunks: &[SearchResult], query: &str) -> Result<bool> {
        // Combine all chunk content from the document
        let full_content = chunks.iter()
            .map(|c| c.content.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        
        // Create a validator to check the boolean logic
        let validator = DocumentLevelValidator::new(self.boolean_engine.clone());
        
        // Create a temporary SearchResult with the combined content
        let combined_result = SearchResult {
            file_path: chunks[0].file_path.clone(),
            content: full_content,
            chunk_index: 0,
            score: 0.0,
        };
        
        // Validate the combined content against the query
        validator.validate_boolean_results(query, &[combined_result])
    }
}
```

### 3. RED: Write failing test for OR and NOT validation
```rust
#[test]
fn test_cross_chunk_or_validation() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    create_cross_chunk_validation_index(&index_path)?;
    let boolean_engine = BooleanSearchEngine::new(&index_path)?;
    let handler = CrossChunkBooleanHandler::new(boolean_engine);
    
    // Test OR across chunks - should find documents with either term
    let results = handler.search_across_chunks("Display OR fn")?;
    
    // Should find both spanning_terms.rs (has Display) and partial_match.rs (has fn)
    assert!(results.len() >= 2, "Should find documents with either term across chunks");
    
    let file_paths: Vec<&str> = results.iter().map(|r| r.file_path.as_str()).collect();
    assert!(file_paths.contains(&"spanning_terms.rs"));
    assert!(file_paths.contains(&"partial_match.rs"));
    
    Ok(())
}

#[test]
fn test_cross_chunk_not_validation() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    create_cross_chunk_not_index(&index_path)?;
    let boolean_engine = BooleanSearchEngine::new(&index_path)?;
    let handler = CrossChunkBooleanHandler::new(boolean_engine);
    
    // Test NOT across chunks - should exclude documents with forbidden terms
    let results = handler.search_across_chunks("pub NOT Error")?;
    
    // Should find documents with "pub" but not containing "Error" anywhere
    for result in &results {
        assert!(result.content.to_lowercase().contains("pub"));
        assert!(!result.content.to_lowercase().contains("error"));
    }
    
    // Should NOT find documents that have "Error" in any chunk
    let file_paths: Vec<&str> = results.iter().map(|r| r.file_path.as_str()).collect();
    assert!(!file_paths.contains(&"has_error.rs"));
    
    Ok(())
}

fn create_cross_chunk_not_index(index_path: &Path) -> Result<()> {
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
    
    // Document with pub but no Error - SHOULD MATCH
    index_writer.add_document(doc!(
        file_path_field => "good_pub.rs",
        content_field => "pub struct Good {",
        raw_content_field => "pub struct Good {",
        chunk_index_field => 0u64
    ))?;
    
    index_writer.add_document(doc!(
        file_path_field => "good_pub.rs",
        content_field => "    data: String,",
        raw_content_field => "    data: String,",
        chunk_index_field => 1u64
    ))?;
    
    // Document with pub AND Error - SHOULD NOT MATCH
    index_writer.add_document(doc!(
        file_path_field => "has_error.rs",
        content_field => "pub fn test() {",
        raw_content_field => "pub fn test() {",
        chunk_index_field => 0u64
    ))?;
    
    index_writer.add_document(doc!(
        file_path_field => "has_error.rs",
        content_field => "    return Err(Error::new(\"fail\"));",
        raw_content_field => "    return Err(Error::new(\"fail\"));",
        chunk_index_field => 1u64
    ))?;
    
    index_writer.commit()?;
    Ok(())
}
```

### 4. GREEN: Update process_document_chunks with validation
```rust
// Update process_document_chunks method in CrossChunkBooleanHandler
impl CrossChunkBooleanHandler {
    fn process_document_chunks(&self, file_path: &str, chunks: &[SearchResult], query: &str) -> Result<Option<DocumentResult>> {
        // Validate that the combined document actually satisfies the boolean query
        if !self.document_satisfies_boolean_query(chunks, query)? {
            return Ok(None); // Document doesn't satisfy the query
        }
        
        // Combine content from all chunks
        let combined_content = chunks.iter()
            .map(|c| c.content.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        
        // Get the highest score among chunks
        let max_score = chunks.iter()
            .map(|c| c.score)
            .fold(0.0f32, f32::max);
        
        Ok(Some(DocumentResult {
            file_path: file_path.to_string(),
            content: combined_content,
            chunks: chunks.len(),
            score: max_score,
        }))
    }
}
```

### 5. REFACTOR: Add edge case testing
```rust
#[test]
fn test_cross_chunk_edge_cases() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    create_cross_chunk_validation_index(&index_path)?;
    let boolean_engine = BooleanSearchEngine::new(&index_path)?;
    let handler = CrossChunkBooleanHandler::new(boolean_engine);
    
    // Test single chunk document (should still work)
    let single_chunk_index_path = temp_dir.path().join("single_index");
    create_single_chunk_index(&single_chunk_index_path)?;
    let single_engine = BooleanSearchEngine::new(&single_chunk_index_path)?;
    let single_handler = CrossChunkBooleanHandler::new(single_engine);
    
    let results = single_handler.search_across_chunks("pub AND struct")?;
    assert!(!results.is_empty(), "Should handle single chunk documents");
    assert_eq!(results[0].chunks, 1, "Single chunk document should show 1 chunk");
    
    // Test empty query
    let results = handler.search_across_chunks("")?;
    // Should handle gracefully (might return all or none depending on implementation)
    
    Ok(())
}

fn create_single_chunk_index(index_path: &Path) -> Result<()> {
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
        file_path_field => "single.rs",
        content_field => "pub struct SingleChunk { data: String }",
        raw_content_field => "pub struct SingleChunk { data: String }",
        chunk_index_field => 0u64
    ))?;
    
    index_writer.commit()?;
    Ok(())
}
```

## Validation Checklist
- [ ] Cross-chunk AND validation works correctly
- [ ] Cross-chunk OR validation works correctly
- [ ] Cross-chunk NOT validation excludes proper documents
- [ ] Single chunk documents still work correctly
- [ ] Edge cases are handled gracefully

## Context for Next Task
Next task will implement performance testing and optimization to ensure the cross-chunk boolean search meets performance requirements.