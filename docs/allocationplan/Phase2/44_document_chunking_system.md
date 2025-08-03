# Task 44: Document Chunking System

## Metadata
- **Micro-Phase**: 2.44
- **Duration**: 19 minutes
- **Dependencies**: Task 43 (scoring_system_integration)
- **Output**: `src/document_processing/chunking.rs`

## Description
Implement intelligent document chunking system with semantic-aware segmentation, overlap management, size optimization, and boundary detection. Supports multiple document formats and maintains context coherence with >95% semantic boundary accuracy and 20+ docs/second throughput.

## Test Requirements
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::document_processing::{Document, DocumentFormat, ChunkMetadata};
    use std::collections::HashMap;

    #[test]
    fn test_document_chunker_creation() {
        let chunker = DocumentChunker::new();
        assert_eq!(chunker.get_config().chunk_size, 512);
        assert_eq!(chunker.get_config().overlap_size, 64);
        assert_eq!(chunker.get_config().min_chunk_size, 100);
        assert!(chunker.supports_format(&DocumentFormat::PlainText));
        assert!(chunker.supports_format(&DocumentFormat::Markdown));
    }
    
    #[test]
    fn test_basic_text_chunking() {
        let chunker = DocumentChunker::new();
        
        let document = create_test_document(
            "This is the first paragraph with important content. \
             It contains multiple sentences that should be grouped together. \
             \n\nThis is the second paragraph with different content. \
             It also has multiple sentences but different topics. \
             \n\nThe third paragraph discusses something entirely different. \
             This content should be in a separate chunk for optimal processing.",
            DocumentFormat::PlainText
        );
        
        let chunks = chunker.chunk_document(&document).unwrap();
        
        // Should create multiple chunks
        assert!(chunks.len() >= 2);
        assert!(chunks.len() <= 4);
        
        // Each chunk should have metadata
        for chunk in &chunks {
            assert!(!chunk.content.is_empty());
            assert!(chunk.metadata.chunk_id >= 0);
            assert!(chunk.metadata.start_offset < chunk.metadata.end_offset);
            assert_eq!(chunk.metadata.document_id, document.id);
            assert!(chunk.metadata.semantic_boundaries.len() > 0);
        }
        
        // Chunks should maintain order
        for i in 1..chunks.len() {
            assert!(chunks[i-1].metadata.start_offset < chunks[i].metadata.start_offset);
        }
    }
    
    #[test]
    fn test_semantic_boundary_detection() {
        let chunker = DocumentChunker::new();
        
        let document = create_test_document(
            "Animals are fascinating creatures. Dogs are loyal companions. \
             Cats are independent hunters. Birds can fly through the sky. \
             \n\nTechnology is rapidly advancing. Computers process information quickly. \
             Smartphones connect people globally. Artificial intelligence learns patterns. \
             \n\nCooking is an essential skill. Vegetables provide nutrition. \
             Spices enhance flavors. Techniques vary across cultures.",
            DocumentFormat::PlainText
        );
        
        let chunks = chunker.chunk_document(&document).unwrap();
        
        // Should detect semantic boundaries between topics
        assert!(chunks.len() >= 3); // Animals, Technology, Cooking
        
        // Each chunk should contain related content
        let animal_chunk = chunks.iter().find(|c| c.content.contains("Dogs") || c.content.contains("Animals"));
        let tech_chunk = chunks.iter().find(|c| c.content.contains("Technology") || c.content.contains("Computers"));
        let cooking_chunk = chunks.iter().find(|c| c.content.contains("Cooking") || c.content.contains("Vegetables"));
        
        assert!(animal_chunk.is_some());
        assert!(tech_chunk.is_some());
        assert!(cooking_chunk.is_some());
    }
    
    #[test]
    fn test_overlap_management() {
        let mut config = ChunkingConfig::default();
        config.chunk_size = 200;
        config.overlap_size = 50;
        config.enable_overlap = true;
        
        let chunker = DocumentChunker::with_config(config);
        
        let document = create_test_document(
            "A".repeat(500).as_str(), // 500 character document
            DocumentFormat::PlainText
        );
        
        let chunks = chunker.chunk_document(&document).unwrap();
        
        assert!(chunks.len() >= 2);
        
        // Check overlap between consecutive chunks
        for i in 1..chunks.len() {
            let prev_chunk = &chunks[i-1];
            let curr_chunk = &chunks[i];
            
            // Should have some overlap
            let overlap = calculate_text_overlap(&prev_chunk.content, &curr_chunk.content);
            assert!(overlap >= 30); // At least 30 characters overlap
            assert!(overlap <= 80); // But not too much
        }
    }
    
    #[test]
    fn test_chunk_size_optimization() {
        let chunker = DocumentChunker::new();
        
        // Test with very short document
        let short_doc = create_test_document("Short text.", DocumentFormat::PlainText);
        let short_chunks = chunker.chunk_document(&short_doc).unwrap();
        
        assert_eq!(short_chunks.len(), 1);
        assert_eq!(short_chunks[0].content, "Short text.");
        
        // Test with document exactly at chunk size
        let exact_doc = create_test_document(&"A".repeat(512), DocumentFormat::PlainText);
        let exact_chunks = chunker.chunk_document(&exact_doc).unwrap();
        
        assert_eq!(exact_chunks.len(), 1);
        assert_eq!(exact_chunks[0].content.len(), 512);
        
        // Test with document slightly over chunk size
        let over_doc = create_test_document(&"A".repeat(600), DocumentFormat::PlainText);
        let over_chunks = chunker.chunk_document(&over_doc).unwrap();
        
        assert!(over_chunks.len() >= 2);
        
        // No chunk should be smaller than min_chunk_size (except possibly the last)
        for (i, chunk) in over_chunks.iter().enumerate() {
            if i < over_chunks.len() - 1 { // Not the last chunk
                assert!(chunk.content.len() >= chunker.get_config().min_chunk_size);
            }
        }
    }
    
    #[test]
    fn test_markdown_format_handling() {
        let chunker = DocumentChunker::new();
        
        let markdown_content = r#"
# Chapter 1: Introduction

This is the introduction paragraph with important context.
It sets up the main themes of the document.

## Section 1.1: Background

Here we discuss the background information.
Multiple sentences provide detailed context.

### Subsection 1.1.1: Historical Context

Historical information goes here.
This should be grouped with the subsection.

## Section 1.2: Methodology

Different topic requiring separate chunk.
Methodology details are provided here.

# Chapter 2: Results

Results chapter starts a new major section.
Should definitely be in a separate chunk.
"#;
        
        let document = create_test_document(markdown_content, DocumentFormat::Markdown);
        let chunks = chunker.chunk_document(&document).unwrap();
        
        // Should respect markdown structure
        assert!(chunks.len() >= 3); // Introduction, Methodology, Results at minimum
        
        // Check that headers are preserved with their content
        let intro_chunk = chunks.iter().find(|c| c.content.contains("# Chapter 1: Introduction"));
        assert!(intro_chunk.is_some());
        
        let results_chunk = chunks.iter().find(|c| c.content.contains("# Chapter 2: Results"));
        assert!(results_chunk.is_some());
        
        // Headers should not be separated from their content
        if let Some(chunk) = intro_chunk {
            assert!(chunk.content.contains("This is the introduction paragraph"));
        }
    }
    
    #[test]
    fn test_chunk_metadata_accuracy() {
        let chunker = DocumentChunker::new();
        
        let document = create_test_document(
            "First sentence. Second sentence. Third sentence. Fourth sentence.",
            DocumentFormat::PlainText
        );
        
        let chunks = chunker.chunk_document(&document).unwrap();
        
        for (i, chunk) in chunks.iter().enumerate() {
            let metadata = &chunk.metadata;
            
            // Basic metadata validation
            assert_eq!(metadata.chunk_id, i);
            assert_eq!(metadata.document_id, document.id);
            assert!(metadata.start_offset < metadata.end_offset);
            assert_eq!(metadata.character_count, chunk.content.len());
            
            // Content should match offset positions
            let extracted_content = &document.content[metadata.start_offset..metadata.end_offset];
            assert_eq!(chunk.content.trim(), extracted_content.trim());
            
            // Should have semantic boundary information
            assert!(!metadata.semantic_boundaries.is_empty());
            
            // Should have estimated reading time
            assert!(metadata.estimated_reading_time_ms > 0);
        }
    }
    
    #[test]
    fn test_chunking_performance() {
        let chunker = DocumentChunker::new();
        
        // Create a large document
        let large_content = "This is a test sentence with meaningful content. ".repeat(1000); // ~50KB
        let large_document = create_test_document(&large_content, DocumentFormat::PlainText);
        
        let start = std::time::Instant::now();
        let chunks = chunker.chunk_document(&large_document).unwrap();
        let elapsed = start.elapsed();
        
        // Should process quickly (target: 20+ docs/second means <50ms per doc)
        assert!(elapsed < std::time::Duration::from_millis(50));
        
        // Should create reasonable number of chunks
        assert!(chunks.len() > 10);
        assert!(chunks.len() < 200); // Not too many tiny chunks
        
        // All chunks should be valid
        for chunk in &chunks {
            assert!(!chunk.content.is_empty());
            assert!(chunk.content.len() >= chunker.get_config().min_chunk_size || 
                   chunk.metadata.chunk_id == chunks.len() - 1); // Last chunk can be smaller
        }
    }
    
    #[test]
    fn test_batch_chunking_performance() {
        let chunker = DocumentChunker::new();
        
        // Create multiple documents
        let documents: Vec<_> = (0..20)
            .map(|i| create_test_document(
                &format!("Document {} content with multiple sentences. This creates realistic chunking scenarios. Content varies between documents to test different patterns.", i),
                DocumentFormat::PlainText
            ))
            .collect();
        
        let start = std::time::Instant::now();
        let all_chunks = chunker.batch_chunk_documents(&documents).unwrap();
        let elapsed = start.elapsed();
        
        // Should achieve 20+ docs/second target
        let docs_per_second = documents.len() as f32 / elapsed.as_secs_f32();
        assert!(docs_per_second >= 20.0);
        
        // Should have chunks for all documents
        assert_eq!(all_chunks.len(), documents.len());
        
        // Each document should have at least one chunk
        for chunks in &all_chunks {
            assert!(!chunks.is_empty());
        }
    }
    
    #[test]
    fn test_chunk_boundary_preservation() {
        let chunker = DocumentChunker::new();
        
        let document = create_test_document(
            "Sentence one. Sentence two! Question three? Statement four. \
             \n\nNew paragraph here. Another sentence. Final statement.",
            DocumentFormat::PlainText
        );
        
        let chunks = chunker.chunk_document(&document).unwrap();
        
        // Reconstruct content from chunks
        let mut reconstructed = String::new();
        let mut last_end = 0;
        
        for chunk in &chunks {
            // Check for gaps or overlaps
            if chunk.metadata.start_offset > last_end {
                // There's a gap - should be minimal whitespace only
                let gap = &document.content[last_end..chunk.metadata.start_offset];
                assert!(gap.trim().is_empty() || gap.chars().all(|c| c.is_whitespace()));
            }
            
            reconstructed.push_str(&chunk.content);
            last_end = chunk.metadata.end_offset;
        }
        
        // Reconstructed content should preserve all meaningful text
        let original_meaningful = document.content.chars().filter(|c| !c.is_whitespace()).collect::<String>();
        let reconstructed_meaningful = reconstructed.chars().filter(|c| !c.is_whitespace()).collect::<String>();
        
        // Should preserve at least 95% of meaningful content
        let preservation_ratio = reconstructed_meaningful.len() as f32 / original_meaningful.len() as f32;
        assert!(preservation_ratio >= 0.95);
    }
    
    #[test]
    fn test_chunking_edge_cases() {
        let chunker = DocumentChunker::new();
        
        // Empty document
        let empty_doc = create_test_document("", DocumentFormat::PlainText);
        let empty_chunks = chunker.chunk_document(&empty_doc).unwrap();
        assert!(empty_chunks.is_empty());
        
        // Whitespace only document
        let whitespace_doc = create_test_document("   \n\n   \t  ", DocumentFormat::PlainText);
        let whitespace_chunks = chunker.chunk_document(&whitespace_doc).unwrap();
        assert!(whitespace_chunks.is_empty() || whitespace_chunks[0].content.trim().is_empty());
        
        // Single character document
        let single_char_doc = create_test_document("A", DocumentFormat::PlainText);
        let single_char_chunks = chunker.chunk_document(&single_char_doc).unwrap();
        assert_eq!(single_char_chunks.len(), 1);
        assert_eq!(single_char_chunks[0].content, "A");
        
        // Very long single sentence
        let long_sentence = format!("This is a very long sentence that goes on and on without any punctuation or breaks and should still be handled gracefully by the chunking system even though it exceeds the normal chunk size limits {}", "word ".repeat(100));
        let long_doc = create_test_document(&long_sentence, DocumentFormat::PlainText);
        let long_chunks = chunker.chunk_document(&long_doc).unwrap();
        
        // Should handle gracefully without infinite loops or errors
        assert!(!long_chunks.is_empty());
        for chunk in &long_chunks {
            assert!(!chunk.content.is_empty());
        }
    }
    
    fn create_test_document(content: &str, format: DocumentFormat) -> Document {
        use uuid::Uuid;
        
        Document {
            id: Uuid::new_v4().to_string(),
            content: content.to_string(),
            format,
            metadata: HashMap::new(),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        }
    }
    
    fn calculate_text_overlap(text1: &str, text2: &str) -> usize {
        // Simple overlap calculation - find longest common suffix/prefix
        let bytes1 = text1.as_bytes();
        let bytes2 = text2.as_bytes();
        
        let min_len = bytes1.len().min(bytes2.len());
        let mut overlap = 0;
        
        // Check suffix of text1 with prefix of text2
        for i in 1..=min_len {
            if bytes1[bytes1.len() - i..] == bytes2[..i] {
                overlap = i;
            }
        }
        
        overlap
    }
}
```

## Implementation
```rust
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use rayon::prelude::*;
use regex::Regex;
use uuid::Uuid;

/// Configuration for document chunking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkingConfig {
    /// Target chunk size in characters
    pub chunk_size: usize,
    
    /// Overlap size between chunks in characters
    pub overlap_size: usize,
    
    /// Minimum chunk size (chunks smaller than this will be merged)
    pub min_chunk_size: usize,
    
    /// Maximum chunk size (hard limit)
    pub max_chunk_size: usize,
    
    /// Enable overlap between chunks
    pub enable_overlap: bool,
    
    /// Enable semantic boundary detection
    pub enable_semantic_boundaries: bool,
    
    /// Respect document structure (headers, paragraphs)
    pub respect_structure: bool,
    
    /// Preferred break points (sentence endings, paragraph breaks)
    pub prefer_sentence_breaks: bool,
    
    /// Quality threshold for semantic boundaries (0.0-1.0)
    pub semantic_boundary_threshold: f32,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            overlap_size: 64,
            min_chunk_size: 100,
            max_chunk_size: 1024,
            enable_overlap: true,
            enable_semantic_boundaries: true,
            respect_structure: true,
            prefer_sentence_breaks: true,
            semantic_boundary_threshold: 0.7,
        }
    }
}

/// Document formats supported by the chunker
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DocumentFormat {
    PlainText,
    Markdown,
    Html,
    Json,
    Xml,
    Csv,
}

/// Document structure for processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Unique document identifier
    pub id: String,
    
    /// Document content
    pub content: String,
    
    /// Document format
    pub format: DocumentFormat,
    
    /// Additional metadata
    pub metadata: HashMap<String, String>,
    
    /// Creation timestamp
    pub created_at: u64,
}

/// Chunk of a document with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentChunk {
    /// Chunk content
    pub content: String,
    
    /// Chunk metadata
    pub metadata: ChunkMetadata,
}

/// Metadata for a document chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMetadata {
    /// Chunk identifier within the document
    pub chunk_id: usize,
    
    /// Source document identifier
    pub document_id: String,
    
    /// Start offset in original document
    pub start_offset: usize,
    
    /// End offset in original document
    pub end_offset: usize,
    
    /// Character count in chunk
    pub character_count: usize,
    
    /// Word count in chunk
    pub word_count: usize,
    
    /// Sentence count in chunk
    pub sentence_count: usize,
    
    /// Semantic boundaries within chunk
    pub semantic_boundaries: Vec<SemanticBoundary>,
    
    /// Structural elements (headers, paragraphs)
    pub structural_elements: Vec<StructuralElement>,
    
    /// Overlap information with adjacent chunks
    pub overlap_info: Option<OverlapInfo>,
    
    /// Estimated reading time in milliseconds
    pub estimated_reading_time_ms: u64,
    
    /// Quality score for this chunk (0.0-1.0)
    pub quality_score: f32,
}

/// Semantic boundary information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticBoundary {
    /// Position within chunk
    pub position: usize,
    
    /// Type of boundary
    pub boundary_type: BoundaryType,
    
    /// Confidence in boundary detection
    pub confidence: f32,
    
    /// Topic or theme before boundary
    pub topic_before: Option<String>,
    
    /// Topic or theme after boundary
    pub topic_after: Option<String>,
}

/// Types of semantic boundaries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BoundaryType {
    /// Topic change
    TopicShift,
    
    /// Paragraph break
    ParagraphBreak,
    
    /// Section break
    SectionBreak,
    
    /// Sentence boundary
    SentenceBoundary,
    
    /// Clause boundary
    ClauseBoundary,
}

/// Structural element information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralElement {
    /// Type of structural element
    pub element_type: StructuralElementType,
    
    /// Position within chunk
    pub position: usize,
    
    /// Length of element
    pub length: usize,
    
    /// Text content of element
    pub content: String,
    
    /// Nesting level (for headers)
    pub level: Option<usize>,
}

/// Types of structural elements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StructuralElementType {
    Header,
    Paragraph,
    List,
    ListItem,
    CodeBlock,
    Quote,
    Table,
    Link,
    Image,
}

/// Overlap information between chunks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverlapInfo {
    /// Characters overlapping with previous chunk
    pub previous_overlap: usize,
    
    /// Characters overlapping with next chunk
    pub next_overlap: usize,
    
    /// Quality of overlap (how well it preserves context)
    pub overlap_quality: f32,
}

/// Main document chunking system
pub struct DocumentChunker {
    /// Chunking configuration
    config: ChunkingConfig,
    
    /// Regex patterns for boundary detection
    sentence_pattern: Regex,
    paragraph_pattern: Regex,
    header_pattern: Regex,
    
    /// Format-specific processors
    format_processors: HashMap<DocumentFormat, Box<dyn FormatProcessor + Send + Sync>>,
    
    /// Semantic boundary detector
    semantic_detector: SemanticBoundaryDetector,
}

impl DocumentChunker {
    /// Create a new document chunker with default configuration
    pub fn new() -> Self {
        let config = ChunkingConfig::default();
        Self::with_config(config)
    }
    
    /// Create with custom configuration
    pub fn with_config(config: ChunkingConfig) -> Self {
        let sentence_pattern = Regex::new(r"[.!?]+\s+").unwrap();
        let paragraph_pattern = Regex::new(r"\n\s*\n").unwrap();
        let header_pattern = Regex::new(r"^#+\s+").unwrap();
        
        let mut format_processors: HashMap<DocumentFormat, Box<dyn FormatProcessor + Send + Sync>> = HashMap::new();
        format_processors.insert(DocumentFormat::PlainText, Box::new(PlainTextProcessor::new()));
        format_processors.insert(DocumentFormat::Markdown, Box::new(MarkdownProcessor::new()));
        format_processors.insert(DocumentFormat::Html, Box::new(HtmlProcessor::new()));
        
        Self {
            config,
            sentence_pattern,
            paragraph_pattern,
            header_pattern,
            format_processors,
            semantic_detector: SemanticBoundaryDetector::new(),
        }
    }
    
    /// Chunk a single document
    pub fn chunk_document(&self, document: &Document) -> Result<Vec<DocumentChunk>, ChunkingError> {
        if document.content.trim().is_empty() {
            return Ok(vec![]);
        }
        
        // Get format-specific processor
        let processor = self.format_processors.get(&document.format)
            .ok_or_else(|| ChunkingError::UnsupportedFormat(format!("{:?}", document.format)))?;
        
        // Preprocess document
        let preprocessed = processor.preprocess(&document.content)?;
        
        // Detect structural elements
        let structural_elements = processor.detect_structure(&preprocessed)?;
        
        // Perform chunking
        let chunks = if self.config.respect_structure && !structural_elements.is_empty() {
            self.chunk_with_structure(&preprocessed, &structural_elements, document)?
        } else {
            self.chunk_by_size(&preprocessed, document)?
        };
        
        Ok(chunks)
    }
    
    /// Batch chunk multiple documents
    pub fn batch_chunk_documents(&self, documents: &[Document]) -> Result<Vec<Vec<DocumentChunk>>, ChunkingError> {
        // Process documents in parallel
        let results: Result<Vec<_>, _> = documents.par_iter()
            .map(|doc| self.chunk_document(doc))
            .collect();
        
        results
    }
    
    /// Chunk document respecting structural elements
    fn chunk_with_structure(&self, content: &str, structural_elements: &[StructuralElement], document: &Document) -> Result<Vec<DocumentChunk>, ChunkingError> {
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut current_start = 0;
        let mut chunk_id = 0;
        
        for element in structural_elements {
            let element_content = &content[element.position..element.position + element.length];
            
            // Check if adding this element would exceed chunk size
            if !current_chunk.is_empty() && 
               current_chunk.len() + element_content.len() > self.config.chunk_size {
                
                // Finalize current chunk if it meets minimum size
                if current_chunk.len() >= self.config.min_chunk_size {
                    let chunk = self.create_chunk(
                        current_chunk.trim().to_string(),
                        chunk_id,
                        current_start,
                        current_start + current_chunk.len(),
                        document.id.clone()
                    )?;
                    chunks.push(chunk);
                    chunk_id += 1;
                }
                
                // Start new chunk
                current_chunk.clear();
                current_start = element.position;
            }
            
            if current_chunk.is_empty() {
                current_start = element.position;
            }
            
            current_chunk.push_str(element_content);
            
            // Add separator if needed
            if !element_content.ends_with('\n') && 
               matches!(element.element_type, StructuralElementType::Paragraph | StructuralElementType::Header) {
                current_chunk.push('\n');
            }
        }
        
        // Handle remaining content
        if !current_chunk.trim().is_empty() {
            let chunk = self.create_chunk(
                current_chunk.trim().to_string(),
                chunk_id,
                current_start,
                current_start + current_chunk.len(),
                document.id.clone()
            )?;
            chunks.push(chunk);
        }
        
        // Apply overlap if enabled
        if self.config.enable_overlap && chunks.len() > 1 {
            self.apply_overlap(&mut chunks, content)?;
        }
        
        Ok(chunks)
    }
    
    /// Chunk document by size with smart boundary detection
    fn chunk_by_size(&self, content: &str, document: &Document) -> Result<Vec<DocumentChunk>, ChunkingError> {
        let mut chunks = Vec::new();
        let mut current_pos = 0;
        let mut chunk_id = 0;
        
        while current_pos < content.len() {
            let remaining = content.len() - current_pos;
            let target_size = self.config.chunk_size.min(remaining);
            
            // Find optimal break point
            let end_pos = if remaining <= self.config.chunk_size {
                content.len() // Last chunk
            } else {
                self.find_optimal_break_point(content, current_pos, current_pos + target_size)?
            };
            
            let chunk_content = content[current_pos..end_pos].trim().to_string();
            
            if !chunk_content.is_empty() && chunk_content.len() >= self.config.min_chunk_size {
                let chunk = self.create_chunk(
                    chunk_content,
                    chunk_id,
                    current_pos,
                    end_pos,
                    document.id.clone()
                )?;
                chunks.push(chunk);
                chunk_id += 1;
            }
            
            // Move to next position with overlap consideration
            if self.config.enable_overlap && chunks.len() > 0 && end_pos < content.len() {
                current_pos = end_pos.saturating_sub(self.config.overlap_size);
            } else {
                current_pos = end_pos;
            }
        }
        
        Ok(chunks)
    }
    
    /// Find optimal break point for chunking
    fn find_optimal_break_point(&self, content: &str, start: usize, target_end: usize) -> Result<usize, ChunkingError> {
        let search_window = self.config.overlap_size * 2;
        let search_start = target_end.saturating_sub(search_window);
        let search_end = (target_end + search_window).min(content.len());
        let search_text = &content[search_start..search_end];
        
        // Priority order for break points:
        // 1. Paragraph breaks
        // 2. Sentence endings
        // 3. Word boundaries
        // 4. Character boundary (fallback)
        
        if self.config.prefer_sentence_breaks {
            // Look for paragraph breaks first
            if let Some(para_matches) = self.find_break_candidates(&self.paragraph_pattern, search_text, search_start, target_end) {
                if let Some(best_match) = para_matches.into_iter().min_by_key(|&pos| (pos as i32 - target_end as i32).abs()) {
                    return Ok(best_match);
                }
            }
            
            // Look for sentence endings
            if let Some(sent_matches) = self.find_break_candidates(&self.sentence_pattern, search_text, search_start, target_end) {
                if let Some(best_match) = sent_matches.into_iter().min_by_key(|&pos| (pos as i32 - target_end as i32).abs()) {
                    return Ok(best_match);
                }
            }
        }
        
        // Look for word boundaries
        let word_boundary = self.find_word_boundary(content, target_end)?;
        if word_boundary != target_end {
            return Ok(word_boundary);
        }
        
        // Fallback to target position
        Ok(target_end.min(content.len()))
    }
    
    /// Find break candidates using regex pattern
    fn find_break_candidates(&self, pattern: &Regex, text: &str, text_start_offset: usize, target_pos: usize) -> Option<Vec<usize>> {
        let matches: Vec<_> = pattern.find_iter(text)
            .map(|m| text_start_offset + m.end())
            .filter(|&pos| pos <= target_pos + self.config.overlap_size && pos >= target_pos.saturating_sub(self.config.overlap_size))
            .collect();
        
        if matches.is_empty() {
            None
        } else {
            Some(matches)
        }
    }
    
    /// Find word boundary near target position
    fn find_word_boundary(&self, content: &str, target_pos: usize) -> Result<usize, ChunkingError> {
        if target_pos >= content.len() {
            return Ok(content.len());
        }
        
        // Look backward for word boundary
        for i in (0..=target_pos).rev() {
            if i == 0 || content.chars().nth(i - 1).map_or(false, |c| c.is_whitespace()) {
                return Ok(i);
            }
        }
        
        // Look forward for word boundary
        for i in target_pos..content.len() {
            if content.chars().nth(i).map_or(false, |c| c.is_whitespace()) {
                return Ok(i);
            }
        }
        
        Ok(target_pos)
    }
    
    /// Create a chunk with metadata
    fn create_chunk(&self, content: String, chunk_id: usize, start_offset: usize, end_offset: usize, document_id: String) -> Result<DocumentChunk, ChunkingError> {
        let character_count = content.len();
        let word_count = content.split_whitespace().count();
        let sentence_count = self.sentence_pattern.find_iter(&content).count();
        
        // Detect semantic boundaries if enabled
        let semantic_boundaries = if self.config.enable_semantic_boundaries {
            self.semantic_detector.detect_boundaries(&content, self.config.semantic_boundary_threshold)?
        } else {
            vec![]
        };
        
        // Estimate reading time (average 200 words per minute)
        let estimated_reading_time_ms = if word_count > 0 {
            ((word_count as f32 / 200.0) * 60.0 * 1000.0) as u64
        } else {
            100 // Minimum reading time
        };
        
        // Calculate quality score
        let quality_score = self.calculate_chunk_quality(&content, &semantic_boundaries)?;
        
        let metadata = ChunkMetadata {
            chunk_id,
            document_id,
            start_offset,
            end_offset,
            character_count,
            word_count,
            sentence_count,
            semantic_boundaries,
            structural_elements: vec![], // Could be populated if needed
            overlap_info: None, // Will be set by apply_overlap if needed
            estimated_reading_time_ms,
            quality_score,
        };
        
        Ok(DocumentChunk {
            content,
            metadata,
        })
    }
    
    /// Apply overlap between chunks
    fn apply_overlap(&self, chunks: &mut Vec<DocumentChunk>, original_content: &str) -> Result<(), ChunkingError> {
        for i in 1..chunks.len() {
            let prev_end = chunks[i - 1].metadata.end_offset;
            let curr_start = chunks[i].metadata.start_offset;
            
            if curr_start > prev_end {
                // Calculate overlap
                let overlap_start = prev_end.saturating_sub(self.config.overlap_size);
                let overlap_content = &original_content[overlap_start..curr_start];
                
                // Prepend overlap to current chunk
                chunks[i].content = format!("{}{}", overlap_content, chunks[i].content);
                chunks[i].metadata.start_offset = overlap_start;
                chunks[i].metadata.character_count = chunks[i].content.len();
                
                // Update overlap info
                chunks[i].metadata.overlap_info = Some(OverlapInfo {
                    previous_overlap: curr_start - overlap_start,
                    next_overlap: 0, // Will be set for next chunk
                    overlap_quality: 0.8, // Could be calculated more sophisticatedly
                });
                
                if i > 1 {
                    if let Some(ref mut overlap_info) = chunks[i - 1].metadata.overlap_info {
                        overlap_info.next_overlap = curr_start - overlap_start;
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Calculate quality score for a chunk
    fn calculate_chunk_quality(&self, content: &str, semantic_boundaries: &[SemanticBoundary]) -> Result<f32, ChunkingError> {
        let mut quality = 0.7; // Base quality
        
        // Bonus for good length
        if content.len() >= self.config.min_chunk_size && content.len() <= self.config.chunk_size {
            quality += 0.1;
        }
        
        // Bonus for semantic coherence
        if !semantic_boundaries.is_empty() {
            let avg_confidence = semantic_boundaries.iter().map(|b| b.confidence).sum::<f32>() / semantic_boundaries.len() as f32;
            quality += avg_confidence * 0.2;
        }
        
        // Penalty for very short or very long chunks
        if content.len() < self.config.min_chunk_size / 2 {
            quality -= 0.3;
        } else if content.len() > self.config.max_chunk_size {
            quality -= 0.2;
        }
        
        Ok(quality.max(0.0).min(1.0))
    }
    
    /// Check if format is supported
    pub fn supports_format(&self, format: &DocumentFormat) -> bool {
        self.format_processors.contains_key(format)
    }
    
    /// Get current configuration
    pub fn get_config(&self) -> &ChunkingConfig {
        &self.config
    }
}

/// Semantic boundary detector
struct SemanticBoundaryDetector {
    // Simple implementation - could be enhanced with ML models
}

impl SemanticBoundaryDetector {
    fn new() -> Self {
        Self {}
    }
    
    fn detect_boundaries(&self, content: &str, threshold: f32) -> Result<Vec<SemanticBoundary>, ChunkingError> {
        let mut boundaries = Vec::new();
        
        // Simple heuristic-based detection
        // Look for topic shifts based on keyword changes
        let sentences: Vec<&str> = content.split('.').collect();
        
        for (i, sentence) in sentences.iter().enumerate() {
            if i > 0 && i < sentences.len() - 1 {
                let prev_sentence = sentences[i - 1];
                let next_sentence = sentences[i + 1];
                
                // Simple topic shift detection based on word overlap
                let topic_shift_score = self.calculate_topic_shift_score(prev_sentence, next_sentence);
                
                if topic_shift_score >= threshold {
                    boundaries.push(SemanticBoundary {
                        position: sentence.as_ptr() as usize - content.as_ptr() as usize,
                        boundary_type: BoundaryType::TopicShift,
                        confidence: topic_shift_score,
                        topic_before: Some(self.extract_topic(prev_sentence)),
                        topic_after: Some(self.extract_topic(next_sentence)),
                    });
                }
            }
        }
        
        Ok(boundaries)
    }
    
    fn calculate_topic_shift_score(&self, sentence1: &str, sentence2: &str) -> f32 {
        let words1: std::collections::HashSet<&str> = sentence1.split_whitespace().collect();
        let words2: std::collections::HashSet<&str> = sentence2.split_whitespace().collect();
        
        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();
        
        if union == 0 {
            0.0
        } else {
            1.0 - (intersection as f32 / union as f32)
        }
    }
    
    fn extract_topic(&self, sentence: &str) -> String {
        // Simple topic extraction - take most common meaningful words
        let words: Vec<&str> = sentence.split_whitespace()
            .filter(|word| word.len() > 3) // Filter short words
            .take(3) // Take first few meaningful words
            .collect();
        
        words.join(" ")
    }
}

/// Format-specific processor trait
trait FormatProcessor {
    fn preprocess(&self, content: &str) -> Result<String, ChunkingError>;
    fn detect_structure(&self, content: &str) -> Result<Vec<StructuralElement>, ChunkingError>;
}

/// Plain text processor
struct PlainTextProcessor;

impl PlainTextProcessor {
    fn new() -> Self {
        Self
    }
}

impl FormatProcessor for PlainTextProcessor {
    fn preprocess(&self, content: &str) -> Result<String, ChunkingError> {
        // Minimal preprocessing for plain text
        Ok(content.to_string())
    }
    
    fn detect_structure(&self, content: &str) -> Result<Vec<StructuralElement>, ChunkingError> {
        let mut elements = Vec::new();
        let mut pos = 0;
        
        // Detect paragraphs by double newlines
        for paragraph in content.split("\n\n") {
            if !paragraph.trim().is_empty() {
                elements.push(StructuralElement {
                    element_type: StructuralElementType::Paragraph,
                    position: pos,
                    length: paragraph.len(),
                    content: paragraph.to_string(),
                    level: None,
                });
            }
            pos += paragraph.len() + 2; // +2 for "\n\n"
        }
        
        Ok(elements)
    }
}

/// Markdown processor
struct MarkdownProcessor {
    header_regex: Regex,
}

impl MarkdownProcessor {
    fn new() -> Self {
        Self {
            header_regex: Regex::new(r"^(#{1,6})\s+(.+)$").unwrap(),
        }
    }
}

impl FormatProcessor for MarkdownProcessor {
    fn preprocess(&self, content: &str) -> Result<String, ChunkingError> {
        // Keep markdown structure but normalize whitespace
        Ok(content.to_string())
    }
    
    fn detect_structure(&self, content: &str) -> Result<Vec<StructuralElement>, ChunkingError> {
        let mut elements = Vec::new();
        let mut pos = 0;
        
        for line in content.lines() {
            if let Some(captures) = self.header_regex.captures(line) {
                let level = captures.get(1).unwrap().as_str().len();
                let title = captures.get(2).unwrap().as_str();
                
                elements.push(StructuralElement {
                    element_type: StructuralElementType::Header,
                    position: pos,
                    length: line.len(),
                    content: title.to_string(),
                    level: Some(level),
                });
            } else if !line.trim().is_empty() {
                elements.push(StructuralElement {
                    element_type: StructuralElementType::Paragraph,
                    position: pos,
                    length: line.len(),
                    content: line.to_string(),
                    level: None,
                });
            }
            
            pos += line.len() + 1; // +1 for newline
        }
        
        Ok(elements)
    }
}

/// HTML processor
struct HtmlProcessor;

impl HtmlProcessor {
    fn new() -> Self {
        Self
    }
}

impl FormatProcessor for HtmlProcessor {
    fn preprocess(&self, content: &str) -> Result<String, ChunkingError> {
        // Simple HTML tag removal (in real implementation, use proper HTML parser)
        let tag_regex = Regex::new(r"<[^>]+>").unwrap();
        Ok(tag_regex.replace_all(content, " ").to_string())
    }
    
    fn detect_structure(&self, _content: &str) -> Result<Vec<StructuralElement>, ChunkingError> {
        // Simplified - would parse HTML structure in real implementation
        Ok(vec![])
    }
}

/// Chunking errors
#[derive(Debug, thiserror::Error)]
pub enum ChunkingError {
    #[error("Unsupported document format: {0}")]
    UnsupportedFormat(String),
    
    #[error("Invalid chunk configuration: {0}")]
    InvalidConfiguration(String),
    
    #[error("Content processing error: {0}")]
    ContentProcessingError(String),
    
    #[error("Boundary detection error: {0}")]
    BoundaryDetectionError(String),
}

impl Default for DocumentChunker {
    fn default() -> Self {
        Self::new()
    }
}
```

## Verification Steps
1. Create DocumentChunker with configurable chunking parameters
2. Implement semantic-aware segmentation with boundary detection
3. Add support for multiple document formats (text, markdown, HTML)
4. Implement overlap management and size optimization
5. Add comprehensive metadata tracking for each chunk
6. Ensure performance meets 20+ docs/second throughput target

## Success Criteria
- [ ] DocumentChunker compiles and handles all supported formats
- [ ] Semantic boundary detection achieves >95% accuracy
- [ ] Chunk size optimization respects min/max constraints
- [ ] Overlap management preserves context coherence
- [ ] Metadata tracking provides comprehensive chunk information
- [ ] Performance target of 20+ docs/second throughput achieved
- [ ] All comprehensive tests pass with proper edge case handling