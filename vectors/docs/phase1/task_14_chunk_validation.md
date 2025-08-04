# Task 14: Implement Chunk Validation and Quality Scoring

**Estimated Time:** 10 minutes  
**Prerequisites:** Task 13 (Metadata enrichment)  
**Dependencies:** Tasks 01-13 must be completed

## Objective
Implement validation and quality scoring for chunks to ensure only useful, well-formed chunks are indexed, improving search result quality and reducing index bloat.

## Context
Not all chunks provide equal search value. Empty chunks, chunks with only whitespace, overly complex chunks, or chunks without meaningful content can pollute search results. A quality scoring system helps filter and rank chunks for better search experience.

## Task Details

### What You Need to Do

1. **Create chunk validation in `src/validation.rs`:**

   ```rust
   use crate::chunker::TextChunk;
   use crate::metadata::{EnrichedChunk, ChunkMetadata};
   use std::collections::HashSet;
   use anyhow::Result;
   
   #[derive(Debug, Clone)]
   pub struct ValidationResult {
       pub is_valid: bool,
       pub quality_score: f32,
       pub issues: Vec<ValidationIssue>,
       pub recommendations: Vec<String>,
   }
   
   #[derive(Debug, Clone)]
   pub struct ValidationIssue {
       pub severity: IssueSeverity,
       pub message: String,
       pub category: IssueCategory,
   }
   
   #[derive(Debug, Clone, PartialEq)]
   pub enum IssueSeverity {
       Critical,  // Chunk should not be indexed
       Warning,   // Chunk has issues but can be indexed
       Info,      // Minor issues or suggestions
   }
   
   #[derive(Debug, Clone, PartialEq)]
   pub enum IssueCategory {
       Content,      // Content-related issues
       Structure,    // Code structure issues
       Metadata,     // Metadata quality issues
       Performance,  // Performance-related concerns
   }
   
   #[derive(Debug, Clone)]
   pub struct ValidationConfig {
       pub min_content_length: usize,
       pub max_content_length: usize,
       pub min_meaningful_words: usize,
       pub max_complexity_score: f32,
       pub min_code_ratio: f32,
       pub require_documentation: bool,
       pub block_empty_functions: bool,
       pub block_generated_code: bool,
   }
   
   impl Default for ValidationConfig {
       fn default() -> Self {
           Self {
               min_content_length: 10,
               max_content_length: 50_000,
               min_meaningful_words: 3,
               max_complexity_score: 100.0,
               min_code_ratio: 0.1,
               require_documentation: false,
               block_empty_functions: true,
               block_generated_code: true,
           }
       }
   }
   
   pub struct ChunkValidator {
       config: ValidationConfig,
       stop_words: HashSet<String>,
       generated_code_patterns: Vec<String>,
   }
   
   impl ChunkValidator {
       /// Create a new chunk validator
       pub fn new(config: ValidationConfig) -> Self {
           let stop_words = Self::create_stop_words_set();
           let generated_code_patterns = Self::create_generated_code_patterns();
           
           Self {
               config,
               stop_words,
               generated_code_patterns,
           }
       }
       
       /// Validate an enriched chunk
       pub fn validate_chunk(&self, enriched_chunk: &EnrichedChunk) -> ValidationResult {
           let mut issues = Vec::new();
           let mut quality_score = 100.0f32;
   
           // Basic content validation
           self.validate_content(&enriched_chunk.chunk, &mut issues, &mut quality_score);
           
           // Metadata-based validation
           self.validate_metadata(&enriched_chunk.metadata, &mut issues, &mut quality_score);
           
           // Language-specific validation
           if let Some(ref language) = enriched_chunk.chunk.language {
               self.validate_language_specific(&enriched_chunk.chunk, language, &mut issues, &mut quality_score);
           }
           
           // Structure validation
           self.validate_structure(&enriched_chunk.chunk, &enriched_chunk.metadata, &mut issues, &mut quality_score);
           
           // Generate recommendations
           let recommendations = self.generate_recommendations(&issues);
           
           // Determine if chunk is valid (no critical issues)
           let is_valid = !issues.iter().any(|issue| issue.severity == IssueSeverity::Critical);
           
           // Normalize quality score
           quality_score = quality_score.max(0.0).min(100.0);
           
           ValidationResult {
               is_valid,
               quality_score,
               issues,
               recommendations,
           }
       }
       
       /// Validate basic content properties
       fn validate_content(&self, chunk: &TextChunk, issues: &mut Vec<ValidationIssue>, quality_score: &mut f32) {
           let content = &chunk.content;
           
           // Check content length
           if content.len() < self.config.min_content_length {
               issues.push(ValidationIssue {
                   severity: IssueSeverity::Critical,
                   message: format!("Content too short: {} characters (minimum: {})", 
                                  content.len(), self.config.min_content_length),
                   category: IssueCategory::Content,
               });
               *quality_score -= 50.0;
           }
           
           if content.len() > self.config.max_content_length {
               issues.push(ValidationIssue {
                   severity: IssueSeverity::Warning,
                   message: format!("Content very long: {} characters (maximum recommended: {})", 
                                  content.len(), self.config.max_content_length),
                   category: IssueCategory::Content,
               });
               *quality_score -= 20.0;
           }
           
           // Check for meaningful content
           let meaningful_words = self.count_meaningful_words(content);
           if meaningful_words < self.config.min_meaningful_words {
               issues.push(ValidationIssue {
                   severity: IssueSeverity::Critical,
                   message: format!("Too few meaningful words: {} (minimum: {})", 
                                  meaningful_words, self.config.min_meaningful_words),
                   category: IssueCategory::Content,
               });
               *quality_score -= 40.0;
           }
           
           // Check for whitespace-only content
           if content.trim().is_empty() {
               issues.push(ValidationIssue {
                   severity: IssueSeverity::Critical,
                   message: "Content is empty or whitespace-only".to_string(),
                   category: IssueCategory::Content,
               });
               *quality_score = 0.0;
           }
           
           // Check for generated code patterns
           if self.config.block_generated_code && self.contains_generated_code(content) {
               issues.push(ValidationIssue {
                   severity: IssueSeverity::Warning,
                   message: "Content appears to be generated code".to_string(),
                   category: IssueCategory::Content,
               });
               *quality_score -= 30.0;
           }
       }
       
       /// Validate metadata quality
       fn validate_metadata(&self, metadata: &ChunkMetadata, issues: &mut Vec<ValidationIssue>, quality_score: &mut f32) {
           // Check complexity
           if metadata.complexity_score > self.config.max_complexity_score {
               issues.push(ValidationIssue {
                   severity: IssueSeverity::Warning,
                   message: format!("Very high complexity: {:.1} (maximum recommended: {:.1})", 
                                  metadata.complexity_score, self.config.max_complexity_score),
                   category: IssueCategory::Structure,
               });
               *quality_score -= 15.0;
           }
           
           // Check for documentation requirement
           if self.config.require_documentation && !metadata.has_documentation {
               issues.push(ValidationIssue {
                   severity: IssueSeverity::Info,
                   message: "No documentation found".to_string(),
                   category: IssueCategory::Metadata,
               });
               *quality_score -= 5.0;
           }
           
           // Reward good documentation
           if metadata.has_documentation {
               *quality_score += 10.0;
           }
           
           // Check for reasonable comment ratio
           if metadata.comment_ratio > 0.8 {
               issues.push(ValidationIssue {
                   severity: IssueSeverity::Info,
                   message: format!("Very high comment ratio: {:.1}%", metadata.comment_ratio * 100.0),
                   category: IssueCategory::Content,
               });
           }
           
           // Reward moderate commenting
           if metadata.comment_ratio >= 0.1 && metadata.comment_ratio <= 0.5 {
               *quality_score += 5.0;
           }
       }
       
       /// Validate language-specific properties
       fn validate_language_specific(&self, chunk: &TextChunk, language: &str, issues: &mut Vec<ValidationIssue>, quality_score: &mut f32) {
           match language {
               "rust" => self.validate_rust_specific(chunk, issues, quality_score),
               "python" => self.validate_python_specific(chunk, issues, quality_score),
               _ => {} // No specific validation for other languages
           }
       }
       
       /// Validate Rust-specific code quality
       fn validate_rust_specific(&self, chunk: &TextChunk, issues: &mut Vec<ValidationIssue>, quality_score: &mut f32) {
           let content = &chunk.content;
           
           // Check for common Rust quality indicators
           if content.contains("unwrap()") && !content.contains("// SAFETY:") && !content.contains("expect(") {
               issues.push(ValidationIssue {
                   severity: IssueSeverity::Info,
                   message: "Contains unwrap() without safety comment".to_string(),
                   category: IssueCategory::Structure,
               });
               *quality_score -= 5.0;
           }
           
           // Check for proper error handling
           if content.contains("Result<") && !content.contains("?") && !content.contains("match") {
               issues.push(ValidationIssue {
                   severity: IssueSeverity::Info,
                   message: "Result type without apparent error handling".to_string(),
                   category: IssueCategory::Structure,
               });
           }
           
           // Reward good practices
           if content.contains("?") || content.contains("match") {
               *quality_score += 5.0;
           }
       }
       
       /// Validate Python-specific code quality
       fn validate_python_specific(&self, chunk: &TextChunk, issues: &mut Vec<ValidationIssue>, quality_score: &mut f32) {
           let content = &chunk.content;
           
           // Check for type hints
           if content.contains("def ") && !content.contains("->") && !content.contains(":") {
               issues.push(ValidationIssue {
                   severity: IssueSeverity::Info,
                   message: "Function without type hints".to_string(),
                   category: IssueCategory::Structure,
               });
           }
           
           // Check for docstrings
           if content.contains("def ") && !content.contains("\"\"\"") && !content.contains("'''") {
               issues.push(ValidationIssue {
                   severity: IssueSeverity::Info,
                   message: "Function without docstring".to_string(),
                   category: IssueCategory::Metadata,
               });
           }
           
           // Reward good practices
           if content.contains("\"\"\"") || content.contains("'''") {
               *quality_score += 10.0;
           }
       }
       
       /// Validate code structure
       fn validate_structure(&self, chunk: &TextChunk, metadata: &ChunkMetadata, issues: &mut Vec<ValidationIssue>, quality_score: &mut f32) {
           // Check for empty functions
           if self.config.block_empty_functions && self.is_empty_function(chunk, metadata) {
               issues.push(ValidationIssue {
                   severity: IssueSeverity::Critical,
                   message: "Empty function or method".to_string(),
                   category: IssueCategory::Structure,
               });
               *quality_score -= 30.0;
           }
           
           // Check for reasonable function count
           if metadata.function_signatures.len() > 10 {
               issues.push(ValidationIssue {
                   severity: IssueSeverity::Warning,
                   message: format!("Many functions in single chunk: {}", metadata.function_signatures.len()),
                   category: IssueCategory::Structure,
               });
               *quality_score -= 10.0;
           }
           
           // Check for code-to-comment balance
           let code_lines = chunk.content.lines()
               .filter(|line| {
                   let trimmed = line.trim();
                   !trimmed.is_empty() && !trimmed.starts_with("//") && !trimmed.starts_with("#")
               })
               .count();
               
           let code_ratio = code_lines as f32 / metadata.line_count as f32;
           if code_ratio < self.config.min_code_ratio {
               issues.push(ValidationIssue {
                   severity: IssueSeverity::Warning,
                   message: format!("Low code ratio: {:.1}%", code_ratio * 100.0),
                   category: IssueCategory::Content,
               });
               *quality_score -= 15.0;
           }
       }
       
       /// Count meaningful words (excluding stop words and short words)
       fn count_meaningful_words(&self, content: &str) -> usize {
           content.split_whitespace()
               .filter(|word| {
                   let cleaned = word.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
                   cleaned.len() >= 3 && !self.stop_words.contains(&cleaned)
               })
               .count()
       }
       
       /// Check if content contains generated code patterns
       fn contains_generated_code(&self, content: &str) -> bool {
           self.generated_code_patterns.iter()
               .any(|pattern| content.contains(pattern))
       }
       
       /// Check if function/method is empty
       fn is_empty_function(&self, chunk: &TextChunk, metadata: &ChunkMetadata) -> bool {
           if metadata.function_signatures.is_empty() {
               return false;
           }
           
           let content = &chunk.content;
           
           // Simple heuristic: if function signature is found but no meaningful code
           if metadata.function_signatures.len() == 1 {
               let non_whitespace_lines = content.lines()
                   .filter(|line| {
                       let trimmed = line.trim();
                       !trimmed.is_empty() && 
                       !trimmed.starts_with("//") && 
                       !trimmed.starts_with("#") &&
                       !trimmed.starts_with("def ") &&
                       !trimmed.starts_with("fn ") &&
                       trimmed != "{" &&
                       trimmed != "}" &&
                       trimmed != "pass"
                   })
                   .count();
               
               non_whitespace_lines <= 1
           } else {
               false
           }
       }
       
       /// Generate recommendations based on issues
       fn generate_recommendations(&self, issues: &[ValidationIssue]) -> Vec<String> {
           let mut recommendations = Vec::new();
           
           for issue in issues {
               match (&issue.category, &issue.severity) {
                   (IssueCategory::Content, IssueSeverity::Critical) => {
                       recommendations.push("Consider excluding this chunk from indexing".to_string());
                   },
                   (IssueCategory::Structure, IssueSeverity::Warning) => {
                       recommendations.push("Consider refactoring to reduce complexity".to_string());
                   },
                   (IssueCategory::Metadata, _) => {
                       recommendations.push("Consider adding documentation or comments".to_string());
                   },
                   _ => {}
               }
           }
           
           recommendations.sort();
           recommendations.dedup();
           recommendations
       }
       
       /// Create set of common stop words
       fn create_stop_words_set() -> HashSet<String> {
           let stop_words = vec![
               "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
               "from", "up", "about", "into", "through", "during", "before", "after", "above",
               "below", "between", "among", "this", "that", "these", "those", "is", "are", "was",
               "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will",
               "would", "could", "should", "may", "might", "must", "shall", "can", "let", "var",
               "const", "fn", "def", "class", "struct", "enum", "impl", "trait", "pub", "use",
           ];
           
           stop_words.into_iter().map(String::from).collect()
       }
       
       /// Create patterns that indicate generated code
       fn create_generated_code_patterns() -> Vec<String> {
           vec![
               "// This file was generated".to_string(),
               "# This file was generated".to_string(),
               "// Auto-generated".to_string(),
               "# Auto-generated".to_string(),
               "// DO NOT EDIT".to_string(),
               "# DO NOT EDIT".to_string(),
               "@generated".to_string(),
               "autogenerated".to_string(),
               "code-generation".to_string(),
           ]
       }
   }
   ```

2. **Add module declaration to `src/lib.rs`:**

   ```rust
   pub mod validation;
   ```

3. **Add comprehensive tests for validation:**

   ```rust
   #[cfg(test)]
   mod validation_tests {
       use super::*;
       use crate::chunker::TextChunk;
       use crate::metadata::{EnrichedChunk, ChunkMetadata, FunctionSignature};
       use std::collections::HashSet;
       
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
       
       fn create_test_metadata() -> ChunkMetadata {
           ChunkMetadata {
               complexity_score: 5.0,
               has_documentation: false,
               doc_comment_count: 0,
               imports: Vec::new(),
               exports: Vec::new(),
               function_signatures: Vec::new(),
               type_definitions: Vec::new(),
               keywords: HashSet::new(),
               identifiers: HashSet::new(),
               string_literals: Vec::new(),
               comment_ratio: 0.1,
               line_count: 10,
               cyclomatic_complexity: 2,
           }
       }
       
       #[test]
       fn test_validator_creation() {
           let config = ValidationConfig::default();
           let _validator = ChunkValidator::new(config);
       }
       
       #[test]
       fn test_valid_chunk() {
           let validator = ChunkValidator::new(ValidationConfig::default());
           
           let chunk = create_test_chunk(
               "fn calculate_sum(a: i32, b: i32) -> i32 {\n    a + b\n}",
               Some("rust".to_string())
           );
           let metadata = create_test_metadata();
           let enriched = EnrichedChunk { chunk, metadata };
           
           let result = validator.validate_chunk(&enriched);
           
           assert!(result.is_valid, "Valid chunk should pass validation");
           assert!(result.quality_score > 50.0, "Valid chunk should have decent quality score");
       }
       
       #[test]
       fn test_empty_chunk_validation() {
           let validator = ChunkValidator::new(ValidationConfig::default());
           
           let chunk = create_test_chunk("", Some("rust".to_string()));
           let metadata = create_test_metadata();
           let enriched = EnrichedChunk { chunk, metadata };
           
           let result = validator.validate_chunk(&enriched);
           
           assert!(!result.is_valid, "Empty chunk should not be valid");
           assert_eq!(result.quality_score, 0.0, "Empty chunk should have zero quality");
           assert!(result.issues.iter().any(|i| i.severity == IssueSeverity::Critical));
       }
       
       #[test]
       fn test_short_chunk_validation() {
           let validator = ChunkValidator::new(ValidationConfig::default());
           
           let chunk = create_test_chunk("x", Some("rust".to_string()));
           let metadata = create_test_metadata();
           let enriched = EnrichedChunk { chunk, metadata };
           
           let result = validator.validate_chunk(&enriched);
           
           assert!(!result.is_valid, "Very short chunk should not be valid");
           assert!(result.issues.iter().any(|i| i.category == IssueCategory::Content));
       }
       
       #[test]
       fn test_complex_chunk_validation() {
           let mut config = ValidationConfig::default();
           config.max_complexity_score = 3.0; // Set low threshold
           let validator = ChunkValidator::new(config);
           
           let chunk = create_test_chunk(
               "fn complex() { if a { if b { if c { d } } } }",
               Some("rust".to_string())
           );
           let mut metadata = create_test_metadata();
           metadata.complexity_score = 10.0; // High complexity
           let enriched = EnrichedChunk { chunk, metadata };
           
           let result = validator.validate_chunk(&enriched);
           
           assert!(result.is_valid, "Complex chunk should still be valid");
           assert!(result.issues.iter().any(|i| i.category == IssueCategory::Structure));
           assert!(result.quality_score < 100.0, "Complex chunk should have reduced quality");
       }
       
       #[test]
       fn test_documented_chunk_bonus() {
           let validator = ChunkValidator::new(ValidationConfig::default());
           
           let chunk = create_test_chunk(
               "/// This is a documented function\nfn test() {}",
               Some("rust".to_string())
           );
           let mut metadata = create_test_metadata();
           metadata.has_documentation = true;
           let enriched = EnrichedChunk { chunk, metadata };
           
           let result = validator.validate_chunk(&enriched);
           
           assert!(result.quality_score > 50.0, "Documented chunk should get quality bonus");
       }
       
       #[test]
       fn test_empty_function_detection() {
           let mut config = ValidationConfig::default();
           config.block_empty_functions = true;
           let validator = ChunkValidator::new(config);
           
           let chunk = create_test_chunk(
               "fn empty_function() {\n    // TODO: implement\n}",
               Some("rust".to_string())
           );
           let mut metadata = create_test_metadata();
           metadata.function_signatures.push(FunctionSignature {
               name: "empty_function".to_string(),
               parameters: Vec::new(),
               return_type: None,
               visibility: "private".to_string(),
               is_async: false,
               is_unsafe: false,
           });
           let enriched = EnrichedChunk { chunk, metadata };
           
           let result = validator.validate_chunk(&enriched);
           
           assert!(!result.is_valid, "Empty function should not be valid when blocked");
           assert!(result.issues.iter().any(|i| i.severity == IssueSeverity::Critical));
       }
       
       #[test]
       fn test_generated_code_detection() {
           let mut config = ValidationConfig::default();
           config.block_generated_code = true;
           let validator = ChunkValidator::new(config);
           
           let chunk = create_test_chunk(
               "// This file was generated by code generator\nfn test() {}",
               Some("rust".to_string())
           );
           let metadata = create_test_metadata();
           let enriched = EnrichedChunk { chunk, metadata };
           
           let result = validator.validate_chunk(&enriched);
           
           assert!(result.is_valid, "Generated code should be valid but flagged");
           assert!(result.issues.iter().any(|i| i.message.contains("generated")));
           assert!(result.quality_score < 80.0, "Generated code should have reduced quality");
       }
       
       #[test]
       fn test_rust_specific_validation() {
           let validator = ChunkValidator::new(ValidationConfig::default());
           
           let chunk = create_test_chunk(
               "fn risky() -> String { some_result.unwrap() }",
               Some("rust".to_string())
           );
           let metadata = create_test_metadata();
           let enriched = EnrichedChunk { chunk, metadata };
           
           let result = validator.validate_chunk(&enriched);
           
           assert!(result.is_valid, "Rust code with unwrap should be valid");
           assert!(result.issues.iter().any(|i| i.message.contains("unwrap")));
       }
       
       #[test]
       fn test_python_specific_validation() {
           let validator = ChunkValidator::new(ValidationConfig::default());
           
           let chunk = create_test_chunk(
               "def process_data(data):\n    return data.upper()",
               Some("python".to_string())
           );
           let metadata = create_test_metadata();
           let enriched = EnrichedChunk { chunk, metadata };
           
           let result = validator.validate_chunk(&enriched);
           
           assert!(result.is_valid, "Python code should be valid");
           assert!(result.issues.iter().any(|i| i.message.contains("type hints") || i.message.contains("docstring")));
       }
       
       #[test]
       fn test_meaningful_words_count() {
           let validator = ChunkValidator::new(ValidationConfig::default());
           
           let content = "the and or but function calculate important process data structure";
           let count = validator.count_meaningful_words(content);
           
           // Should filter out stop words but keep meaningful ones
           assert!(count >= 4, "Should count meaningful words, got {}", count);
           assert!(count <= 6, "Should not count stop words, got {}", count);
       }
       
       #[test]
       fn test_recommendation_generation() {
           let validator = ChunkValidator::new(ValidationConfig::default());
           
           let issues = vec![
               ValidationIssue {
                   severity: IssueSeverity::Critical,
                   message: "Too short".to_string(),
                   category: IssueCategory::Content,
               },
               ValidationIssue {
                   severity: IssueSeverity::Info,
                   message: "No docs".to_string(),
                   category: IssueCategory::Metadata,
               },
           ];
           
           let recommendations = validator.generate_recommendations(&issues);
           
           assert!(!recommendations.is_empty(), "Should generate recommendations");
           assert!(recommendations.iter().any(|r| r.contains("excluding")));
           assert!(recommendations.iter().any(|r| r.contains("documentation")));
       }
   }
   ```

## Success Criteria
- [ ] Chunk validation compiles without errors
- [ ] All validation tests pass with `cargo test validation_tests`
- [ ] Content validation catches empty, short, and meaningless chunks
- [ ] Quality scoring provides reasonable scores for different chunk types
- [ ] Language-specific validation works for Rust and Python
- [ ] Generated code detection works correctly
- [ ] Empty function detection prevents indexing of stub functions
- [ ] Documentation bonuses and penalties are applied appropriately
- [ ] Meaningful word counting filters stop words correctly
- [ ] Recommendations are generated based on validation issues

## Common Pitfalls to Avoid
- Don't make validation too strict (allow for different coding styles)
- Handle edge cases like very short functions that are still meaningful
- Be careful with generated code detection (avoid false positives)
- Don't penalize valid code patterns too heavily
- Ensure quality scores are normalized and meaningful
- Handle different comment styles across languages
- Consider context when evaluating complexity (simple functions can be complex in context)

## Context for Next Task
Task 15 will implement the final chunking pipeline integration that combines all components (chunking, enrichment, validation) into a cohesive indexing workflow.