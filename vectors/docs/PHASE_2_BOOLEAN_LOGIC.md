# Phase 2: Boolean Logic - Tantivy Query Parser Integration

## Objective
Leverage Tantivy's proven boolean query parser for correct AND/OR/NOT logic, with document-level result aggregation and Windows compatibility.

## Duration
1 Day (8 hours) - Simplified using Tantivy's built-in capabilities

## Why Tantivy Solves the Problem
Tantivy is designed to implement:
- ✅ Correct boolean logic (AND/OR/NOT) designed
- ✅ Query optimization designed
- ✅ Special character handling designed
- ✅ Cross-platform compatibility designed
- ✅ Performance optimization designed

## Technical Approach

### 1. Tantivy Boolean Query Parser
```rust
use tantivy::query::{BooleanQuery, Occur, Query, TermQuery, QueryParser};
use tantivy::schema::Field;
use tantivy::Term;

pub struct BooleanSearchEngine {
    search_engine: SearchEngine,
    query_parser: QueryParser,
}

impl BooleanSearchEngine {
    pub fn new(index_path: &Path) -> anyhow::Result<Self> {
        let search_engine = SearchEngine::new(index_path)?;
        
        // Configure query parser for boolean operations
        let mut query_parser = QueryParser::for_index(
            &search_engine.index,
            vec![
                search_engine.schema.get_field("content")?,
                search_engine.schema.get_field("raw_content")?
            ]
        );
        
        // Enable boolean operations
        query_parser.set_conjunction_by_default(); // AND is default
        
        Ok(Self {
            search_engine,
            query_parser,
        })
    }
    
    pub fn search_boolean(&self, query_str: &str) -> anyhow::Result<Vec<SearchResult>> {
        // Tantivy handles boolean parsing automatically
        let query = self.query_parser.parse_query(query_str)?;
        
        let reader = self.search_engine.index.reader()?;
        let searcher = reader.searcher();
        let top_docs = searcher.search(&query, &TopDocs::with_limit(100))?;
        
        // Aggregate results by document (handle chunks)
        self.aggregate_chunk_results(top_docs, &searcher)
    }
    
    fn aggregate_chunk_results(
        &self, 
        top_docs: Vec<(f32, DocAddress)>, 
        searcher: &Searcher
    ) -> anyhow::Result<Vec<SearchResult>> {
        let mut doc_results: std::collections::HashMap<String, SearchResult> = std::collections::HashMap::new();
        
        for (score, doc_address) in top_docs {
            let doc = searcher.doc(doc_address)?;
            let file_path = self.search_engine.get_field_value(&doc, "file_path")?;
            
            // Aggregate chunks from same document
            match doc_results.get_mut(&file_path) {
                Some(existing) => {
                    // Combine content and take highest score
                    if score > existing.score {
                        existing.score = score;
                        existing.content = self.search_engine.get_field_value(&doc, "content")?;
                    }
                }
                None => {
                    let result = SearchResult {
                        file_path: file_path.clone(),
                        content: self.search_engine.get_field_value(&doc, "content")?,
                        chunk_index: self.search_engine.get_field_u64(&doc, "chunk_index")?,
                        score,
                    };
                    doc_results.insert(file_path, result);
                }
            }
        }
        
        let mut results: Vec<_> = doc_results.into_values().collect();
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        Ok(results)
    }
}
```

### 2. Document-Level Boolean Validation
```rust 
pub struct DocumentLevelValidator {
    search_engine: BooleanSearchEngine,
}

impl DocumentLevelValidator {
    pub fn validate_boolean_results(&self, query: &str, results: &[SearchResult]) -> anyhow::Result<bool> {
        // Parse the query to understand required terms
        let parsed_query = self.parse_boolean_query(query)?;
        
        // Validate each result meets boolean requirements
        for result in results {
            if !self.document_satisfies_query(&result.content, &parsed_query)? {
                return Ok(false);
            }
        }
        
        Ok(true)
    }
    
    fn document_satisfies_query(&self, content: &str, query: &BooleanQueryStructure) -> anyhow::Result<bool> {
        match query {
            BooleanQueryStructure::And(terms) => {
                // ALL terms must be present in the document
                Ok(terms.iter().all(|term| content.contains(term)))
            }
            BooleanQueryStructure::Or(terms) => {
                // ANY term must be present in the document
                Ok(terms.iter().any(|term| content.contains(term)))
            }
            BooleanQueryStructure::Not { include, exclude } => {
                // Include term present but exclude term absent
                Ok(content.contains(include) && !content.contains(exclude))
            }
            BooleanQueryStructure::Complex(sub_queries) => {
                // Handle nested boolean logic recursively
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

#[derive(Debug, Clone)]
enum BooleanQueryStructure {
    And(Vec<String>),
    Or(Vec<String>),
    Not { include: String, exclude: String },
    Complex(Vec<BooleanQueryStructure>),
}
```

### 3. Cross-Chunk Boolean Logic
```rust
pub struct CrossChunkBooleanHandler {
    boolean_engine: BooleanSearchEngine,
}

impl CrossChunkBooleanHandler {
    pub fn search_across_chunks(&self, query: &str) -> anyhow::Result<Vec<DocumentResult>> {
        // Get chunk-level results from Tantivy
        let chunk_results = self.boolean_engine.search_boolean(query)?;
        
        // Group by document and validate boolean logic at document level
        let mut document_groups: std::collections::HashMap<String, Vec<SearchResult>> = std::collections::HashMap::new();
        
        for result in chunk_results {
            document_groups.entry(result.file_path.clone())
                .or_insert_with(Vec::new)
                .push(result);
        }
        
        // Validate boolean logic across all chunks of each document
        let mut document_results = Vec::new();
        for (file_path, chunks) in document_groups {
            if self.document_satisfies_boolean_query(&chunks, query)? {
                let combined_content = chunks.iter()
                    .map(|c| c.content.as_str())
                    .collect::<Vec<_>>()
                    .join(" ");
                
                let max_score = chunks.iter()
                    .map(|c| c.score)
                    .fold(0.0f32, f32::max);
                
                document_results.push(DocumentResult {
                    file_path,
                    content: combined_content,
                    chunks: chunks.len(),
                    score: max_score,
                });
            }
        }
        
        document_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        Ok(document_results)
    }
    
    fn document_satisfies_boolean_query(&self, chunks: &[SearchResult], query: &str) -> anyhow::Result<bool> {
        // Combine all chunk content from the document
        let full_content = chunks.iter()
            .map(|c| c.content.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        
        // Re-validate boolean logic against full document content
        let validator = DocumentLevelValidator { search_engine: self.boolean_engine.clone() };
        validator.validate_boolean_results(query, &[SearchResult {
            file_path: chunks[0].file_path.clone(),
            content: full_content,
            chunk_index: 0,
            score: 0.0,
        }])
    }
}

#[derive(Debug, Clone)]
pub struct DocumentResult {
    pub file_path: String,
    pub content: String,
    pub chunks: usize,
    pub score: f32,
}
```

## Implementation Tasks

### Task 1: Boolean Query Integration (2 hours)
```rust
#[cfg(test)]
mod boolean_tests {
    use super::*;
    
    #[test]
    fn test_boolean_and_logic() -> anyhow::Result<()> {
        let temp_dir = TempDir::new()?;
        let index_path = temp_dir.path().join("index");
        let mut indexer = DocumentIndexer::new(&index_path)?;
        let boolean_engine = BooleanSearchEngine::new(&index_path)?;
        
        // Index test documents
        let test_files = vec![
            ("file1.rs", "pub struct MyStruct { name: String }"),
            ("file2.rs", "fn process() { println!(\"Hello\"); }"),
            ("file3.rs", "pub fn initialize() -> Result<(), Error> { Ok(()) }"),
        ];
        
        for (filename, content) in test_files {
            let file_path = temp_dir.path().join(filename);
            std::fs::write(&file_path, content)?;
            indexer.index_file(&file_path)?;
        }
        
        // Test AND logic - only file3 has both "pub" AND "fn"
        let results = boolean_engine.search_boolean("pub AND fn")?;
        assert_eq!(results.len(), 1, "Should find exactly one document with both pub and fn");
        assert!(results[0].file_path.contains("file3.rs"));
        
        // Test OR logic - file1 and file3 have "pub", file2 and file3 have "fn"
        let results = boolean_engine.search_boolean("struct OR fn")?;
        assert!(results.len() >= 2, "Should find documents with either struct or fn");
        
        // Test NOT logic - exclude documents with "Error"
        let results = boolean_engine.search_boolean("pub NOT Error")?;
        assert_eq!(results.len(), 1, "Should find pub but exclude Error");
        assert!(results[0].file_path.contains("file1.rs"));
        
        Ok(())
    }
    
    #[test]
    fn test_nested_boolean_expressions() -> anyhow::Result<()> {
        let temp_dir = TempDir::new()?;
        let index_path = temp_dir.path().join("index");
        let mut indexer = DocumentIndexer::new(&index_path)?;
        let boolean_engine = BooleanSearchEngine::new(&index_path)?;
        
        // Index documents with various visibility and types
        let test_files = vec![
            ("public_struct.rs", "pub struct Data { value: i32 }"),
            ("private_struct.rs", "struct Internal { secret: String }"),
            ("public_enum.rs", "pub enum Status { Active, Inactive }"),
            ("function.rs", "fn helper() -> bool { true }"),
        ];
        
        for (filename, content) in test_files {
            let file_path = temp_dir.path().join(filename);
            std::fs::write(&file_path, content)?;
            indexer.index_file(&file_path)?;
        }
        
        // Test nested expression: (pub OR private) AND struct
        let results = boolean_engine.search_boolean("(pub OR struct) AND (struct OR enum)")?;
        
        // Should find public_struct.rs, private_struct.rs, public_enum.rs
        assert!(results.len() >= 2, "Should find documents matching nested boolean logic");
        
        // Verify results actually contain the required terms
        for result in &results {
            let has_pub_or_struct = result.content.contains("pub") || result.content.contains("struct");
            let has_struct_or_enum = result.content.contains("struct") || result.content.contains("enum");
            assert!(has_pub_or_struct && has_struct_or_enum, 
                   "Result should satisfy (pub OR struct) AND (struct OR enum)");
        }
        
        Ok(())
    }
}
```

### Task 2: Cross-Chunk Boolean Logic (2 hours)
```rust
#[cfg(test)]
mod cross_chunk_tests {
    use super::*;
    
    #[test]
    fn test_boolean_across_chunks() -> anyhow::Result<()> {
        let temp_dir = TempDir::new()?;
        let index_path = temp_dir.path().join("index");
        let mut indexer = DocumentIndexer::new(&index_path)?;
        let cross_chunk_handler = CrossChunkBooleanHandler {
            boolean_engine: BooleanSearchEngine::new(&index_path)?,
        };
        
        // Create large document that will be split across chunks
        let large_content = format!(
            "{}\\npub struct LargeStruct {{\\n    data: Vec<String>,\\n}}\\n{}\\nimpl Display for LargeStruct {{\\n    fn fmt(&self, f: &mut Formatter) -> Result {{}}\\n}}",
            "x".repeat(1500),  // Force chunking
            "y".repeat(1500),  // "pub" and "Display" in different chunks
        );
        
        let large_file = temp_dir.path().join("large.rs");
        std::fs::write(&large_file, &large_content)?;
        indexer.index_file(&large_file)?;
        
        // Search for terms that span chunks
        let results = cross_chunk_handler.search_across_chunks("pub AND Display")?;
        
        assert_eq!(results.len(), 1, "Should find document with both pub and Display across chunks");
        assert!(results[0].chunks > 1, "Should have multiple chunks");
        assert!(results[0].content.contains("pub") && results[0].content.contains("Display"));
        
        Ok(())
    }
}
```

### Task 3: Performance Optimization (2 hours)
```rust
#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::time::Instant;
    
    #[test]
    fn test_boolean_query_performance() -> anyhow::Result<()> {
        let temp_dir = TempDir::new()?;
        let index_path = temp_dir.path().join("index");
        let mut indexer = DocumentIndexer::new(&index_path)?;
        let boolean_engine = BooleanSearchEngine::new(&index_path)?;
        
        // Index 1000 test documents
        for i in 0..1000 {
            let content = if i % 3 == 0 {
                format!("pub struct Data{} {{ value: i32 }}", i)
            } else if i % 3 == 1 {
                format!("fn process{}() -> Result<(), Error> {{ Ok(()) }}", i)
            } else {
                format!("impl Display for Type{} {{ fn fmt() {{}} }}", i)
            };
            
            let file_path = temp_dir.path().join(format!("file_{}.rs", i));
            std::fs::write(&file_path, content)?;
            indexer.index_file(&file_path)?;
        }
        
        // Test AND query performance
        let start = Instant::now();
        let results = boolean_engine.search_boolean("pub AND struct")?;
        let and_duration = start.elapsed();
        
        assert!(and_duration.as_millis() < 50, "AND query should complete in under 50ms");
        assert!(!results.is_empty(), "Should find results");
        
        // Test complex nested query performance
        let start = Instant::now();
        let results = boolean_engine.search_boolean("(pub AND struct) OR (impl AND Display)")?;
        let complex_duration = start.elapsed();
        
        assert!(complex_duration.as_millis() < 100, "Complex query should complete in under 100ms");
        assert!(!results.is_empty(), "Should find results");
        
        Ok(())
    }
}
```

### Task 4: Document-Level Validation (2 hours)
```rust
#[cfg(test)]
mod validation_tests {
    use super::*;
    
    #[test]
    fn test_document_level_boolean_validation() -> anyhow::Result<()> {
        let temp_dir = TempDir::new()?;
        let index_path = temp_dir.path().join("index");
        let boolean_engine = BooleanSearchEngine::new(&index_path)?;
        let validator = DocumentLevelValidator { search_engine: boolean_engine };
        
        // Test AND validation
        let and_content = "pub fn test() { struct Data; }";
        let and_result = SearchResult {
            file_path: "test.rs".to_string(),
            content: and_content.to_string(),
            chunk_index: 0,
            score: 1.0,
        };
        
        assert!(validator.validate_boolean_results("pub AND struct", &[and_result])?);  // Should pass
        
        let wrong_and_result = SearchResult {
            file_path: "test.rs".to_string(),
            content: "pub fn test() { }".to_string(),  // Missing "struct"
            chunk_index: 0,
            score: 1.0,
        };
        
        assert!(!validator.validate_boolean_results("pub AND struct", &[wrong_and_result])?);  // Should fail
        
        // Test NOT validation
        let not_content = "pub fn test() { println!(\"Hello\"); }";
        let not_result = SearchResult {
            file_path: "test.rs".to_string(),
            content: not_content.to_string(),
            chunk_index: 0,
            score: 1.0,
        };
        
        assert!(validator.validate_boolean_results("pub NOT Error", &[not_result])?);  // Should pass
        
        let wrong_not_result = SearchResult {
            file_path: "test.rs".to_string(),
            content: "pub fn test() -> Result<(), Error> { }".to_string(),  // Contains "Error"
            chunk_index: 0,
            score: 1.0,
        };
        
        assert!(!validator.validate_boolean_results("pub NOT Error", &[wrong_not_result])?);  // Should fail
        
        Ok(())
    }
}
```

## Deliverables

### Rust Source Files
1. `src/boolean.rs` - Boolean search engine
2. `src/cross_chunk.rs` - Cross-chunk boolean logic
3. `src/validator.rs` - Document-level validation
4. `src/query_parser.rs` - Tantivy query parser wrapper

### Test Files
1. `tests/boolean_tests.rs` - Boolean logic tests
2. `tests/cross_chunk_tests.rs` - Cross-chunk tests
3. `tests/validation_tests.rs` - Validation tests
4. `tests/performance_tests.rs` - Performance benchmarks

## Success Metrics

### Functional Requirements ✅ DESIGN COMPLETE
- [x] Correct AND logic (all terms required) designed
- [x] Correct OR logic (any term matches) designed
- [x] Correct NOT logic (exclusion) designed
- [x] Nested boolean expressions designed
- [x] Cross-chunk boolean logic designed
- [x] Document-level validation designed

### Performance Targets ✅ DESIGN TARGETS SET
- [x] Boolean AND/OR queries < 50ms (design target)
- [x] Complex nested queries < 100ms (design target)
- [x] Cross-chunk queries < 150ms (design target)
- [x] Memory efficient aggregation designed

### Quality Gates ✅ DESIGN COMPLETE
- [x] 100% accuracy on boolean logic designed
- [x] Zero false positives/negatives designed
- [x] Proper chunk aggregation designed
- [x] Windows compatibility designed

## Next Phase
With Tantivy's proven boolean logic integrated, proceed to Phase 3: Advanced Search Features (proximity, wildcards, regex).

---

*Phase 2 leverages Tantivy's battle-tested boolean query parser instead of building from scratch.*