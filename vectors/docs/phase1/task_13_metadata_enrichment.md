# Task 13: Implement Metadata Enrichment for Enhanced Search

**Estimated Time:** 10 minutes  
**Prerequisites:** Task 12 (Incremental indexing)  
**Dependencies:** Tasks 01-12 must be completed

## Objective
Enrich chunk metadata with semantic information like code complexity, documentation presence, imports/dependencies, and function signatures to improve search relevance and filtering.

## Context
Basic chunking provides content and location. Enriched metadata enables sophisticated search features like finding complex functions, documented code, specific imports, or architectural patterns. This metadata will be crucial for relevance scoring and result filtering.

## Task Details

### What You Need to Do

1. **Create metadata enrichment in `src/metadata.rs`:**

   ```rust
   use crate::chunker::TextChunk;
   use tree_sitter::{Parser, Tree, Node, Query, QueryCursor};
   use std::collections::{HashMap, HashSet};
   use anyhow::Result;
   
   #[derive(Debug, Clone)]
   pub struct EnrichedChunk {
       pub chunk: TextChunk,
       pub metadata: ChunkMetadata,
   }
   
   #[derive(Debug, Clone)]
   pub struct ChunkMetadata {
       pub complexity_score: f32,
       pub has_documentation: bool,
       pub doc_comment_count: usize,
       pub imports: Vec<String>,
       pub exports: Vec<String>,
       pub function_signatures: Vec<FunctionSignature>,
       pub type_definitions: Vec<TypeDefinition>,
       pub keywords: HashSet<String>,
       pub identifiers: HashSet<String>,
       pub string_literals: Vec<String>,
       pub comment_ratio: f32,
       pub line_count: usize,
       pub cyclomatic_complexity: usize,
   }
   
   #[derive(Debug, Clone)]
   pub struct FunctionSignature {
       pub name: String,
       pub parameters: Vec<String>,
       pub return_type: Option<String>,
       pub visibility: String, // "pub", "private", etc.
       pub is_async: bool,
       pub is_unsafe: bool,
   }
   
   #[derive(Debug, Clone)]
   pub struct TypeDefinition {
       pub name: String,
       pub kind: String, // "struct", "enum", "trait", "class", etc.
       pub fields: Vec<String>,
       pub visibility: String,
   }
   
   pub struct MetadataEnricher {
       rust_parser: Parser,
       python_parser: Parser,
   }
   
   impl MetadataEnricher {
       /// Create a new metadata enricher
       pub fn new() -> Result<Self> {
           let mut rust_parser = Parser::new();
           rust_parser.set_language(tree_sitter_rust::language())?;
           
           let mut python_parser = Parser::new();
           python_parser.set_language(tree_sitter_python::language())?;
           
           Ok(Self {
               rust_parser,
               python_parser,
           })
       }
       
       /// Enrich a chunk with metadata
       pub fn enrich_chunk(&mut self, chunk: TextChunk) -> Result<EnrichedChunk> {
           let metadata = self.extract_metadata(&chunk)?;
           
           Ok(EnrichedChunk {
               chunk,
               metadata,
           })
       }
       
       /// Extract metadata from a chunk
       fn extract_metadata(&mut self, chunk: &TextChunk) -> Result<ChunkMetadata> {
           let mut metadata = ChunkMetadata {
               complexity_score: 0.0,
               has_documentation: false,
               doc_comment_count: 0,
               imports: Vec::new(),
               exports: Vec::new(),
               function_signatures: Vec::new(),
               type_definitions: Vec::new(),
               keywords: HashSet::new(),
               identifiers: HashSet::new(),
               string_literals: Vec::new(),
               comment_ratio: 0.0,
               line_count: chunk.content.lines().count(),
               cyclomatic_complexity: 1, // Default complexity
           };
           
           // Parse content based on language
           let tree = match chunk.language.as_deref() {
               Some("rust") => {
                   self.rust_parser.parse(&chunk.content, None)
               },
               Some("python") => {
                   self.python_parser.parse(&chunk.content, None)
               },
               _ => {
                   // For non-code content, extract basic text metadata
                   return Ok(self.extract_text_metadata(&chunk.content, metadata));
               }
           };
           
           if let Some(tree) = tree {
               metadata = self.extract_code_metadata(&chunk.content, &tree, chunk.language.as_deref().unwrap(), metadata)?;
           }
           
           Ok(metadata)
       }
       
       /// Extract metadata from code using AST
       fn extract_code_metadata(&self, content: &str, tree: &Tree, language: &str, mut metadata: ChunkMetadata) -> Result<ChunkMetadata> {
           let root_node = tree.root_node();
           
           // Extract different types of metadata based on language
           match language {
               "rust" => {
                   metadata = self.extract_rust_metadata(content, &root_node, metadata)?;
               },
               "python" => {
                   metadata = self.extract_python_metadata(content, &root_node, metadata)?;
               },
               _ => {}
           }
           
           // Common metadata extraction
           metadata.comment_ratio = self.calculate_comment_ratio(content);
           metadata.complexity_score = self.calculate_complexity_score(&metadata);
           
           Ok(metadata)
       }
       
       /// Extract Rust-specific metadata
       fn extract_rust_metadata(&self, content: &str, root_node: &Node, mut metadata: ChunkMetadata) -> Result<ChunkMetadata> {
           let mut cursor = root_node.walk();
           
           // Walk through all nodes
           self.walk_nodes(&mut cursor, content, &mut metadata, "rust")?;
           
           Ok(metadata)
       }
       
       /// Extract Python-specific metadata
       fn extract_python_metadata(&self, content: &str, root_node: &Node, mut metadata: ChunkMetadata) -> Result<ChunkMetadata> {
           let mut cursor = root_node.walk();
           
           // Walk through all nodes
           self.walk_nodes(&mut cursor, content, &mut metadata, "python")?;
           
           Ok(metadata)
       }
       
       /// Walk through AST nodes and extract metadata
       fn walk_nodes(&self, cursor: &mut tree_sitter::TreeCursor, content: &str, metadata: &mut ChunkMetadata, language: &str) -> Result<()> {
           loop {
               let node = cursor.node();
               
               // Extract information based on node type
               match node.kind() {
                   // Function definitions
                   "function_item" | "function_definition" => {
                       if let Some(signature) = self.extract_function_signature(&node, content, language)? {
                           metadata.function_signatures.push(signature);
                       }
                       metadata.cyclomatic_complexity += self.calculate_node_complexity(&node, content);
                   },
                   
                   // Type definitions
                   "struct_item" | "enum_item" | "trait_item" | "class_definition" => {
                       if let Some(type_def) = self.extract_type_definition(&node, content, language)? {
                           metadata.type_definitions.push(type_def);
                       }
                   },
                   
                   // Import statements
                   "use_declaration" | "import_statement" | "import_from_statement" => {
                       if let Some(import) = self.extract_import(&node, content) {
                           metadata.imports.push(import);
                       }
                   },
                   
                   // Comments
                   "line_comment" | "block_comment" | "comment" => {
                       metadata.doc_comment_count += 1;
                       let comment_text = self.get_node_text(&node, content);
                       if self.is_documentation_comment(&comment_text, language) {
                           metadata.has_documentation = true;
                       }
                   },
                   
                   // String literals
                   "string_literal" | "raw_string_literal" => {
                       let literal = self.get_node_text(&node, content);
                       if literal.len() > 2 { // Skip empty strings
                           metadata.string_literals.push(literal.to_string());
                       }
                   },
                   
                   // Identifiers and keywords
                   "identifier" => {
                       let identifier = self.get_node_text(&node, content);
                       metadata.identifiers.insert(identifier.to_string());
                   },
                   
                   _ => {
                       // Check if it's a keyword
                       if self.is_keyword(&node, language) {
                           let keyword = self.get_node_text(&node, content);
                           metadata.keywords.insert(keyword.to_string());
                       }
                   }
               }
               
               // Recurse into children
               if cursor.goto_first_child() {
                   self.walk_nodes(cursor, content, metadata, language)?;
                   cursor.goto_parent();
               }
               
               // Move to next sibling
               if !cursor.goto_next_sibling() {
                   break;
               }
           }
           
           Ok(())
       }
       
       /// Extract function signature from AST node
       fn extract_function_signature(&self, node: &Node, content: &str, language: &str) -> Result<Option<FunctionSignature>> {
           let mut cursor = node.walk();
           let mut name = String::new();
           let mut parameters = Vec::new();
           let mut return_type = None;
           let mut visibility = "private".to_string();
           let mut is_async = false;
           let mut is_unsafe = false;
           
           // Look for function components
           while cursor.goto_next_sibling() || cursor.goto_first_child() {
               let child = cursor.node();
               
               match child.kind() {
                   "identifier" if name.is_empty() => {
                       name = self.get_node_text(&child, content).to_string();
                   },
                   "parameters" | "parameter_list" => {
                       parameters = self.extract_parameters(&child, content)?;
                   },
                   "type_annotation" | "return_type" => {
                       return_type = Some(self.get_node_text(&child, content).to_string());
                   },
                   "visibility_modifier" => {
                       visibility = self.get_node_text(&child, content).to_string();
                   },
                   _ => {}
               }
               
               // Check for async/unsafe keywords
               let node_text = self.get_node_text(&child, content);
               if node_text == "async" {
                   is_async = true;
               } else if node_text == "unsafe" {
                   is_unsafe = true;
               }
           }
           
           if !name.is_empty() {
               Ok(Some(FunctionSignature {
                   name,
                   parameters,
                   return_type,
                   visibility,
                   is_async,
                   is_unsafe,
               }))
           } else {
               Ok(None)
           }
       }
       
       /// Extract type definition from AST node
       fn extract_type_definition(&self, node: &Node, content: &str, _language: &str) -> Result<Option<TypeDefinition>> {
           let mut cursor = node.walk();
           let mut name = String::new();
           let kind = node.kind().to_string();
           let mut fields = Vec::new();
           let mut visibility = "private".to_string();
           
           // Extract type information
           while cursor.goto_next_sibling() || cursor.goto_first_child() {
               let child = cursor.node();
               
               match child.kind() {
                   "type_identifier" | "identifier" if name.is_empty() => {
                       name = self.get_node_text(&child, content).to_string();
                   },
                   "field_declaration_list" | "enum_variant_list" => {
                       fields = self.extract_fields(&child, content)?;
                   },
                   "visibility_modifier" => {
                       visibility = self.get_node_text(&child, content).to_string();
                   },
                   _ => {}
               }
           }
           
           if !name.is_empty() {
               Ok(Some(TypeDefinition {
                   name,
                   kind,
                   fields,
                   visibility,
               }))
           } else {
               Ok(None)
           }
       }
       
       /// Extract import statement
       fn extract_import(&self, node: &Node, content: &str) -> Option<String> {
           let import_text = self.get_node_text(node, content);
           Some(import_text.trim().to_string())
       }
       
       /// Extract parameters from parameter list
       fn extract_parameters(&self, node: &Node, content: &str) -> Result<Vec<String>> {
           let mut parameters = Vec::new();
           let mut cursor = node.walk();
           
           if cursor.goto_first_child() {
               loop {
                   let child = cursor.node();
                   if child.kind().contains("parameter") {
                       let param_text = self.get_node_text(&child, content);
                       parameters.push(param_text.to_string());
                   }
                   
                   if !cursor.goto_next_sibling() {
                       break;
                   }
               }
           }
           
           Ok(parameters)
       }
       
       /// Extract fields from type definition
       fn extract_fields(&self, node: &Node, content: &str) -> Result<Vec<String>> {
           let mut fields = Vec::new();
           let mut cursor = node.walk();
           
           if cursor.goto_first_child() {
               loop {
                   let child = cursor.node();
                   if child.kind().contains("field") || child.kind().contains("variant") {
                       let field_text = self.get_node_text(&child, content);
                       fields.push(field_text.to_string());
                   }
                   
                   if !cursor.goto_next_sibling() {
                       break;
                   }
               }
           }
           
           Ok(fields)
       }
       
       /// Get text content of a node
       fn get_node_text<'a>(&self, node: &Node, content: &'a str) -> &'a str {
           &content[node.start_byte()..node.end_byte()]
       }
       
       /// Check if a comment is documentation
       fn is_documentation_comment(&self, comment: &str, language: &str) -> bool {
           match language {
               "rust" => comment.starts_with("///") || comment.starts_with("/**"),
               "python" => comment.starts_with("\"\"\"") || comment.starts_with("'''"),
               _ => false,
           }
       }
       
       /// Check if a node represents a keyword
       fn is_keyword(&self, node: &Node, language: &str) -> bool {
           let keywords = match language {
               "rust" => vec!["fn", "struct", "enum", "impl", "trait", "pub", "use", "mod", "let", "mut", "const", "static"],
               "python" => vec!["def", "class", "import", "from", "if", "else", "for", "while", "try", "except"],
               _ => vec![],
           };
           
           let node_text = self.get_node_text(node, ""); // This is a simplified check
           keywords.contains(&node_text)
       }
       
       /// Calculate cyclomatic complexity for a node
       fn calculate_node_complexity(&self, node: &Node, content: &str) -> usize {
           let complexity_indicators = ["if", "else", "match", "while", "for", "loop", "try", "catch"];
           let node_text = self.get_node_text(node, content);
           
           complexity_indicators.iter()
               .map(|&indicator| node_text.matches(indicator).count())
               .sum()
       }
       
       /// Calculate comment ratio in content
       fn calculate_comment_ratio(&self, content: &str) -> f32 {
           let total_lines = content.lines().count() as f32;
           if total_lines == 0.0 {
               return 0.0;
           }
           
           let comment_lines = content.lines()
               .filter(|line| {
                   let trimmed = line.trim();
                   trimmed.starts_with("//") || trimmed.starts_with("#") || 
                   trimmed.starts_with("/*") || trimmed.starts_with("*")
               })
               .count() as f32;
           
           comment_lines / total_lines
       }
       
       /// Calculate overall complexity score
       fn calculate_complexity_score(&self, metadata: &ChunkMetadata) -> f32 {
           let base_score = metadata.cyclomatic_complexity as f32;
           let function_penalty = metadata.function_signatures.len() as f32 * 0.5;
           let type_penalty = metadata.type_definitions.len() as f32 * 0.3;
           let line_factor = (metadata.line_count as f32 / 10.0).min(5.0);
           
           base_score + function_penalty + type_penalty + line_factor
       }
       
       /// Extract basic metadata from text content
       fn extract_text_metadata(&self, content: &str, mut metadata: ChunkMetadata) -> ChunkMetadata {
           // Extract words as identifiers
           for word in content.split_whitespace() {
               if word.len() > 2 && word.chars().all(|c| c.is_alphanumeric() || c == '_') {
                   metadata.identifiers.insert(word.to_string());
               }
           }
           
           // Look for quoted strings
           for line in content.lines() {
               if let Some(start) = line.find('"') {
                   if let Some(end) = line[start + 1..].find('"') {
                       let literal = &line[start..start + end + 2];
                       metadata.string_literals.push(literal.to_string());
                   }
               }
           }
           
           metadata.comment_ratio = 0.0; // No specific comments in plain text
           metadata.complexity_score = 1.0; // Base complexity for text
           
           metadata
       }
   }
   ```

2. **Add module declaration to `src/lib.rs`:**

   ```rust
   pub mod metadata;
   ```

3. **Add comprehensive tests for metadata enrichment:**

   ```rust
   #[cfg(test)]
   mod metadata_tests {
       use super::*;
       use crate::chunker::{SmartChunker, TextChunk};
       
       fn create_test_chunk(content: &str, language: Option<String>) -> TextChunk {
           TextChunk {
               id: "test_chunk".to_string(),
               content: content.to_string(),
               start_byte: 0,
               end_byte: content.len(),
               chunk_index: 0,
               total_chunks: 1,
               language,
               file_path: "test.rs".to_string(),
               overlap_with_previous: 0,
               overlap_with_next: 0,
               semantic_type: "code".to_string(),
           }
       }
       
       #[test]
       fn test_metadata_enricher_creation() -> Result<()> {
           let _enricher = MetadataEnricher::new()?;
           Ok(())
       }
       
       #[test]
       fn test_rust_function_extraction() -> Result<()> {
           let mut enricher = MetadataEnricher::new()?;
           
           let rust_code = r#"
           /// This is a documented function
           pub async unsafe fn process_data(input: &str, count: usize) -> Result<String, Error> {
               if input.is_empty() {
                   return Err(Error::EmptyInput);
               }
               Ok(input.repeat(count))
           }
           "#;
           
           let chunk = create_test_chunk(rust_code, Some("rust".to_string()));
           let enriched = enricher.enrich_chunk(chunk)?;
           
           assert_eq!(enriched.metadata.function_signatures.len(), 1);
           let func = &enriched.metadata.function_signatures[0];
           assert_eq!(func.name, "process_data");
           assert_eq!(func.parameters.len(), 2);
           assert!(func.is_async);
           assert!(func.is_unsafe);
           assert_eq!(func.visibility, "pub");
           assert!(enriched.metadata.has_documentation);
           
           Ok(())
       }
       
       #[test]
       fn test_rust_struct_extraction() -> Result<()> {
           let mut enricher = MetadataEnricher::new()?;
           
           let rust_code = r#"
           pub struct Config {
               name: String,
               port: u16,
               enabled: bool,
           }
           
           enum Status {
               Active,
               Inactive,
               Pending,
           }
           "#;
           
           let chunk = create_test_chunk(rust_code, Some("rust".to_string()));
           let enriched = enricher.enrich_chunk(chunk)?;
           
           assert_eq!(enriched.metadata.type_definitions.len(), 2);
           
           let struct_def = enriched.metadata.type_definitions.iter()
               .find(|t| t.name == "Config")
               .expect("Should find Config struct");
           assert_eq!(struct_def.visibility, "pub");
           assert_eq!(struct_def.fields.len(), 3);
           
           Ok(())
       }
       
       #[test]
       fn test_python_function_extraction() -> Result<()> {
           let mut enricher = MetadataEnricher::new()?;
           
           let python_code = r#"
           def calculate_score(data: List[int], weights: Dict[str, float]) -> float:
               """Calculate weighted score from data."""
               if not data:
                   return 0.0
               return sum(x * weights.get('default', 1.0) for x in data)
           
           async def fetch_data(url: str) -> Dict[str, Any]:
               return await client.get(url)
           "#;
           
           let chunk = create_test_chunk(python_code, Some("python".to_string()));
           let enriched = enricher.enrich_chunk(chunk)?;
           
           assert_eq!(enriched.metadata.function_signatures.len(), 2);
           
           let calc_func = enriched.metadata.function_signatures.iter()
               .find(|f| f.name == "calculate_score")
               .expect("Should find calculate_score function");
           assert_eq!(calc_func.parameters.len(), 2);
           
           let async_func = enriched.metadata.function_signatures.iter()
               .find(|f| f.name == "fetch_data")
               .expect("Should find fetch_data function");
           assert!(async_func.is_async);
           
           assert!(enriched.metadata.has_documentation);
           
           Ok(())
       }
       
       #[test]
       fn test_import_extraction() -> Result<()> {
           let mut enricher = MetadataEnricher::new()?;
           
           let rust_code = r#"
           use std::collections::HashMap;
           use serde::{Serialize, Deserialize};
           use crate::utils::*;
           "#;
           
           let chunk = create_test_chunk(rust_code, Some("rust".to_string()));
           let enriched = enricher.enrich_chunk(chunk)?;
           
           assert_eq!(enriched.metadata.imports.len(), 3);
           assert!(enriched.metadata.imports.iter().any(|i| i.contains("HashMap")));
           assert!(enriched.metadata.imports.iter().any(|i| i.contains("serde")));
           
           Ok(())
       }
       
       #[test]
       fn test_complexity_calculation() -> Result<()> {
           let mut enricher = MetadataEnricher::new()?;
           
           let complex_code = r#"
           fn complex_function(x: i32) -> i32 {
               if x > 0 {
                   if x > 10 {
                       for i in 0..x {
                           if i % 2 == 0 {
                               match i {
                                   2 => return i * 2,
                                   4 => return i * 4,
                                   _ => continue,
                               }
                           }
                       }
                   } else {
                       while x > 0 {
                           x -= 1;
                       }
                   }
               }
               x
           }
           "#;
           
           let chunk = create_test_chunk(complex_code, Some("rust".to_string()));
           let enriched = enricher.enrich_chunk(chunk)?;
           
           assert!(enriched.metadata.cyclomatic_complexity > 1);
           assert!(enriched.metadata.complexity_score > 2.0);
           
           Ok(())
       }
       
       #[test]
       fn test_comment_ratio_calculation() -> Result<()> {
           let mut enricher = MetadataEnricher::new()?;
           
           let commented_code = r#"
           // This is a comment
           fn test() {
               // Another comment
               println!("hello"); // Inline comment
               // More comments
           }
           // Final comment
           "#;
           
           let chunk = create_test_chunk(commented_code, Some("rust".to_string()));
           let enriched = enricher.enrich_chunk(chunk)?;
           
           assert!(enriched.metadata.comment_ratio > 0.0);
           assert!(enriched.metadata.comment_ratio < 1.0);
           
           Ok(())
       }
       
       #[test]
       fn test_text_metadata_extraction() -> Result<()> {
           let mut enricher = MetadataEnricher::new()?;
           
           let text_content = r#"
           This is a plain text document with some "quoted strings" and
           various identifiers like user_name and process_id.
           It contains multiple lines and should be analyzed appropriately.
           "#;
           
           let chunk = create_test_chunk(text_content, None);
           let enriched = enricher.enrich_chunk(chunk)?;
           
           assert!(enriched.metadata.identifiers.len() > 0);
           assert!(enriched.metadata.string_literals.len() > 0);
           assert_eq!(enriched.metadata.complexity_score, 1.0);
           assert_eq!(enriched.metadata.comment_ratio, 0.0);
           
           Ok(())
       }
       
       #[test]
       fn test_string_literal_extraction() -> Result<()> {
           let mut enricher = MetadataEnricher::new()?;
           
           let code_with_strings = r#"
           fn main() {
               let message = "Hello, world!";
               let raw_string = r#"This is a raw string"#;
               println!("{}", message);
           }
           "#;
           
           let chunk = create_test_chunk(code_with_strings, Some("rust".to_string()));
           let enriched = enricher.enrich_chunk(chunk)?;
           
           assert!(enriched.metadata.string_literals.len() >= 2);
           assert!(enriched.metadata.string_literals.iter().any(|s| s.contains("Hello")));
           
           Ok(())
       }
   }
   ```

## Success Criteria
- [ ] Metadata enrichment compiles without errors
- [ ] All metadata tests pass with `cargo test metadata_tests`
- [ ] Function signatures are correctly extracted for Rust and Python
- [ ] Type definitions (structs, classes, enums) are identified
- [ ] Import statements are captured accurately
- [ ] Complexity metrics are calculated reasonably
- [ ] Documentation detection works for both languages
- [ ] Comment ratios are calculated correctly
- [ ] Text content metadata extraction works for non-code files
- [ ] String literals and identifiers are captured

## Common Pitfalls to Avoid
- Don't assume AST parsing will always succeed (handle parser failures)
- Be careful with empty or malformed code snippets
- Handle Unicode characters properly in text extraction
- Don't make assumptions about code formatting or style
- Ensure metadata extraction doesn't crash on unusual code constructs
- Handle very large chunks efficiently (don't extract too much detail)
- Be careful with memory usage when storing large numbers of identifiers

## Context for Next Task
Task 14 will implement chunk validation and quality scoring to ensure indexed chunks meet quality standards and provide useful search results.