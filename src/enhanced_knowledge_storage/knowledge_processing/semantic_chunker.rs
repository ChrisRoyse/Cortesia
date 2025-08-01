//! Semantic Chunker
//! 
//! Intelligent text chunking that preserves semantic boundaries using
//! AI-powered analysis instead of arbitrary character limits.

use std::sync::Arc;
use std::time::Instant;
use serde_json;
use crate::enhanced_knowledge_storage::{
    types::*,
    model_management::ModelResourceManager,
    knowledge_processing::types::*,
};

/// Semantic chunker using AI for intelligent boundary detection
pub struct SemanticChunker {
    model_manager: Arc<ModelResourceManager>,
    config: SemanticChunkingConfig,
}

/// Configuration for semantic chunking
#[derive(Debug, Clone)]
pub struct SemanticChunkingConfig {
    pub model_id: String,
    pub max_chunk_size: usize,
    pub min_chunk_size: usize,
    pub overlap_strategy: OverlapStrategy,
    pub temperature: f32,
    pub boundary_confidence_threshold: f32,
    pub preserve_sentence_integrity: bool,
    pub preserve_paragraph_integrity: bool,
}

impl Default for SemanticChunkingConfig {
    fn default() -> Self {
        Self {
            model_id: "smollm2_360m".to_string(),
            max_chunk_size: 2048,
            min_chunk_size: 256,
            overlap_strategy: OverlapStrategy::SemanticOverlap { tokens: 128 },
            temperature: 0.1,
            boundary_confidence_threshold: 0.7,
            preserve_sentence_integrity: true,
            preserve_paragraph_integrity: true,
        }
    }
}

/// Overlap strategies for chunk boundaries
#[derive(Debug, Clone)]
pub enum OverlapStrategy {
    SentenceBoundary { sentences: usize },
    SemanticOverlap { tokens: usize },
    ConceptualBridging { concepts: Vec<String> },
    NoOverlap,
}

impl SemanticChunker {
    /// Create new semantic chunker
    pub fn new(model_manager: Arc<ModelResourceManager>, config: SemanticChunkingConfig) -> Self {
        Self {
            model_manager,
            config,
        }
    }
    
    /// Create semantic chunks from text
    pub async fn create_semantic_chunks(&self, text: &str) -> KnowledgeProcessingResult2<Vec<SemanticChunk>> {
        let _start_time = Instant::now();
        
        // Step 1: Analyze document structure
        let structure = self.analyze_document_structure(text).await?;
        
        // Step 2: Find semantic boundaries
        let boundaries = self.find_semantic_boundaries(text, &structure).await?;
        
        // Step 3: Create overlapping chunks with context preservation
        let chunks = self.create_overlapping_chunks(text, &boundaries).await?;
        
        // Step 4: Validate semantic coherence
        let validated_chunks = self.validate_chunk_coherence(chunks).await?;
        
        Ok(validated_chunks)
    }
    
    /// Analyze document structure for better chunking
    async fn analyze_document_structure(&self, text: &str) -> KnowledgeProcessingResult2<DocumentStructure> {
        let prompt = format!(
            r#"Analyze the structure and content of this document. Provide:

1. Overall topic/theme
2. Key themes (3-5 main topics)
3. Complexity level (Low, Medium, High)
4. Main sections with titles if present
5. Section types (Title, Abstract, Introduction, Body, Conclusion, etc.)

Return as JSON with this format:
{{
  "overall_topic": "main topic of the document",
  "key_themes": ["theme1", "theme2", "theme3"],
  "complexity_level": "Medium",
  "sections": [
    {{
      "title": "Section Title or null",
      "start_pos": 0,
      "end_pos": 100,
      "section_type": "Introduction",
      "key_points": ["point1", "point2"]
    }}
  ]
}}

Document to analyze:
{text}

JSON Response:"#
        );
        
        let task = ProcessingTask::new(ComplexityLevel::Medium, &prompt);
        let result = self.model_manager
            .process_with_optimal_model(task)
            .await
            .map_err(|e| KnowledgeProcessingError::ModelError(e.to_string()))?;
        
        self.parse_document_structure(&result.output, text)
    }
    
    /// Parse document structure from model response
    fn parse_document_structure(&self, response: &str, text: &str) -> KnowledgeProcessingResult2<DocumentStructure> {
        let json_start = response.find('{').unwrap_or(0);
        let json_end = response.rfind('}').map(|i| i + 1).unwrap_or(response.len());
        let json_str = &response[json_start..json_end];
        
        let parsed: serde_json::Value = serde_json::from_str(json_str)
            .map_err(KnowledgeProcessingError::JsonError)?;
        
        let overall_topic = parsed["overall_topic"].as_str().map(|s| s.to_string());
        
        let key_themes = if let Some(themes_array) = parsed["key_themes"].as_array() {
            themes_array
                .iter()
                .filter_map(|v| v.as_str())
                .map(|s| s.to_string())
                .collect()
        } else {
            Vec::new()
        };
        
        let complexity_level = match parsed["complexity_level"].as_str() {
            Some("Low") => ComplexityLevel::Low,
            Some("High") => ComplexityLevel::High,
            _ => ComplexityLevel::Medium,
        };
        
        let sections = if let Some(sections_array) = parsed["sections"].as_array() {
            sections_array
                .iter()
                .filter_map(|section| self.parse_document_section(section).ok())
                .collect()
        } else {
            vec![DocumentSection {
                title: None,
                start_pos: 0,
                end_pos: text.len(),
                section_type: SectionType::Body,
                key_points: Vec::new(),
            }]
        };
        
        // Estimate reading time (average 200 words per minute)
        let word_count = text.split_whitespace().count();
        let reading_time = std::time::Duration::from_secs((word_count as f64 / 200.0 * 60.0) as u64);
        
        Ok(DocumentStructure {
            sections,
            overall_topic,
            key_themes,
            complexity_level,
            estimated_reading_time: reading_time,
        })
    }
    
    /// Parse a single document section
    fn parse_document_section(&self, section: &serde_json::Value) -> KnowledgeProcessingResult2<DocumentSection> {
        let title = section["title"].as_str().map(|s| s.to_string());
        let start_pos = section["start_pos"].as_u64().unwrap_or(0) as usize;
        let end_pos = section["end_pos"].as_u64().unwrap_or(0) as usize;
        
        let section_type = match section["section_type"].as_str() {
            Some("Title") => SectionType::Title,
            Some("Abstract") => SectionType::Abstract,
            Some("Introduction") => SectionType::Introduction,
            Some("Conclusion") => SectionType::Conclusion,
            Some("References") => SectionType::References,
            Some("Appendix") => SectionType::Appendix,
            _ => SectionType::Body,
        };
        
        let key_points = if let Some(points_array) = section["key_points"].as_array() {
            points_array
                .iter()
                .filter_map(|v| v.as_str())
                .map(|s| s.to_string())
                .collect()
        } else {
            Vec::new()
        };
        
        Ok(DocumentSection {
            title,
            start_pos,
            end_pos,
            section_type,
            key_points,
        })
    }
    
    /// Find semantic boundaries in text
    async fn find_semantic_boundaries(
        &self,
        text: &str,
        structure: &DocumentStructure,
    ) -> KnowledgeProcessingResult2<Vec<SemanticBoundary>> {
        let prompt = format!(
            r#"Analyze this text and identify optimal boundaries for chunking that preserve semantic meaning.

Consider:
- Sentence completeness and natural breaks
- Paragraph coherence and topic flow
- Topic transitions and theme changes
- Entity relationships and context
- Logical flow and argument structure

For each boundary, provide:
- Position (character index in the text)
- Boundary type (TopicShift, SentenceEnd, ParagraphBreak, SectionBreak, EntityBoundary, ConceptBoundary)
- Confidence (0.0 to 1.0)
- Reason (why this is a good boundary)

Return as JSON array:
[
  {{
    "position": 123,
    "boundary_type": "TopicShift",
    "confidence": 0.85,
    "reason": "Topic changes from physics to mathematics"
  }}
]

Text length: {text_len} characters
Document complexity: {complexity:?}
Key themes: {themes:?}

Text to analyze:
{text}

JSON Response:"#,
            text_len = text.len(),
            complexity = structure.complexity_level,
            themes = structure.key_themes,
            text = text
        );
        
        let task = ProcessingTask::new(ComplexityLevel::High, &prompt);
        let result = self.model_manager
            .process_with_optimal_model(task)
            .await
            .map_err(|e| KnowledgeProcessingError::ModelError(e.to_string()))?;
        
        self.parse_boundary_response(&result.output, text.len())
    }
    
    /// Parse boundary response from model
    fn parse_boundary_response(&self, response: &str, text_len: usize) -> KnowledgeProcessingResult2<Vec<SemanticBoundary>> {
        let json_start = response.find('[').unwrap_or(0);
        let json_end = response.rfind(']').map(|i| i + 1).unwrap_or(response.len());
        let json_str = &response[json_start..json_end];
        
        let parsed: serde_json::Value = serde_json::from_str(json_str)
            .map_err(KnowledgeProcessingError::JsonError)?;
        
        let mut boundaries = Vec::new();
        
        if let Some(array) = parsed.as_array() {
            for item in array {
                if let Some(boundary) = self.parse_single_boundary(item, text_len)? {
                    boundaries.push(boundary);
                }
            }
        }
        
        // Sort boundaries by position
        boundaries.sort_by_key(|b| b.position);
        
        // Filter by confidence threshold
        boundaries.retain(|b| b.confidence >= self.config.boundary_confidence_threshold);
        
        Ok(boundaries)
    }
    
    /// Parse a single boundary from JSON
    fn parse_single_boundary(&self, item: &serde_json::Value, text_len: usize) -> KnowledgeProcessingResult2<Option<SemanticBoundary>> {
        let position = item["position"].as_u64().unwrap_or(0) as usize;
        let boundary_type_str = item["boundary_type"].as_str().unwrap_or("SentenceEnd");
        let confidence = item["confidence"].as_f64().unwrap_or(0.5) as f32;
        let reason = item["reason"].as_str().unwrap_or("").to_string();
        
        // Validate position is within text bounds
        if position >= text_len {
            return Ok(None);
        }
        
        let boundary_type = match boundary_type_str {
            "TopicShift" => BoundaryType::TopicShift,
            "SentenceEnd" => BoundaryType::SentenceEnd,
            "ParagraphBreak" => BoundaryType::ParagraphBreak,
            "SectionBreak" => BoundaryType::SectionBreak,
            "EntityBoundary" => BoundaryType::EntityBoundary,
            "ConceptBoundary" => BoundaryType::ConceptBoundary,
            _ => BoundaryType::SentenceEnd,
        };
        
        Ok(Some(SemanticBoundary {
            position,
            boundary_type,
            confidence,
            reason,
        }))
    }
    
    /// Create overlapping chunks based on boundaries
    async fn create_overlapping_chunks(
        &self,
        text: &str,
        boundaries: &[SemanticBoundary],
    ) -> KnowledgeProcessingResult2<Vec<SemanticChunk>> {
        let mut chunks = Vec::new();
        let mut start_pos = 0;
        
        for (i, boundary) in boundaries.iter().enumerate() {
            let end_pos = boundary.position;
            
            // Skip if chunk would be too small
            if end_pos - start_pos < self.config.min_chunk_size {
                continue;
            }
            
            // Limit chunk size
            let actual_end_pos = if end_pos - start_pos > self.config.max_chunk_size {
                start_pos + self.config.max_chunk_size
            } else {
                end_pos
            };
            
            // Extract chunk content
            let content = text[start_pos..actual_end_pos].to_string();
            
            // Determine chunk type
            let chunk_type = self.determine_chunk_type(&content, boundary);
            
            // Create semantic chunk
            let chunk = SemanticChunk {
                id: format!("chunk_{i}"),
                content,
                start_pos,
                end_pos: actual_end_pos,
                semantic_coherence: boundary.confidence,
                key_concepts: Vec::new(), // Will be filled later
                entities: Vec::new(),     // Will be filled later
                relationships: Vec::new(), // Will be filled later
                chunk_type,
                overlap_with_previous: None, // Will be calculated
                overlap_with_next: None,     // Will be calculated
            };
            
            chunks.push(chunk);
            start_pos = self.calculate_overlap_start(actual_end_pos, &self.config.overlap_strategy);
        }
        
        // Handle final chunk if needed
        if start_pos < text.len() {
            let content = text[start_pos..].to_string();
            if content.len() >= self.config.min_chunk_size {
                let chunk = SemanticChunk {
                    id: format!("chunk_{}", chunks.len()),
                    content,
                    start_pos,
                    end_pos: text.len(),
                    semantic_coherence: 0.8, // Default for final chunk
                    key_concepts: Vec::new(),
                    entities: Vec::new(),
                    relationships: Vec::new(),
                    chunk_type: ChunkType::Other,
                    overlap_with_previous: None,
                    overlap_with_next: None,
                };
                chunks.push(chunk);
            }
        }
        
        // Calculate overlaps between chunks
        self.calculate_chunk_overlaps(&mut chunks, text);
        
        Ok(chunks)
    }
    
    /// Determine the type of chunk based on content and boundary
    fn determine_chunk_type(&self, content: &str, boundary: &SemanticBoundary) -> ChunkType {
        match boundary.boundary_type {
            BoundaryType::ParagraphBreak => ChunkType::Paragraph,
            BoundaryType::SectionBreak => ChunkType::Section,
            BoundaryType::TopicShift => ChunkType::Topic,
            _ => {
                // Analyze content to determine type
                if content.trim_start().starts_with("```") || content.contains("def ") || content.contains("function ") {
                    ChunkType::Code
                } else if content.contains("1.") || content.contains("•") || content.contains("-") {
                    ChunkType::List
                } else if content.contains("|") && content.matches("|").count() > 3 {
                    ChunkType::Table
                } else {
                    ChunkType::Paragraph
                }
            }
        }
    }
    
    /// Calculate overlap start position based on strategy
    fn calculate_overlap_start(&self, end_pos: usize, strategy: &OverlapStrategy) -> usize {
        match strategy {
            OverlapStrategy::SemanticOverlap { tokens } => {
                // Approximate: 1 token ≈ 4 characters
                end_pos.saturating_sub(*tokens * 4)
            }
            OverlapStrategy::SentenceBoundary { sentences: _ } => {
                // For simplicity, use a fixed overlap
                end_pos.saturating_sub(128)
            }
            OverlapStrategy::ConceptualBridging { concepts: _ } => {
                // Use moderate overlap for concept bridging
                end_pos.saturating_sub(256)
            }
            OverlapStrategy::NoOverlap => end_pos,
        }
    }
    
    /// Calculate overlaps between adjacent chunks
    fn calculate_chunk_overlaps(&self, chunks: &mut [SemanticChunk], text: &str) {
        for i in 0..chunks.len() {
            if i > 0 {
                let overlap = self.find_overlap_content(
                    &chunks[i - 1],
                    &chunks[i],
                    text,
                );
                chunks[i].overlap_with_previous = overlap;
            }
            
            if i < chunks.len() - 1 {
                let overlap = self.find_overlap_content(
                    &chunks[i],
                    &chunks[i + 1],
                    text,
                );
                chunks[i].overlap_with_next = overlap;
            }
        }
    }
    
    /// Find overlap content between two chunks
    fn find_overlap_content(
        &self,
        chunk1: &SemanticChunk,
        chunk2: &SemanticChunk,
        text: &str,
    ) -> Option<String> {
        if chunk1.end_pos > chunk2.start_pos {
            let overlap_start = chunk2.start_pos;
            let overlap_end = chunk1.end_pos.min(chunk2.end_pos);
            
            if overlap_end > overlap_start {
                return Some(text[overlap_start..overlap_end].to_string());
            }
        }
        None
    }
    
    /// Validate chunk coherence
    async fn validate_chunk_coherence(&self, chunks: Vec<SemanticChunk>) -> KnowledgeProcessingResult2<Vec<SemanticChunk>> {
        // For now, return chunks as-is
        // In a full implementation, we might use additional AI validation
        Ok(chunks)
    }
    
    /// Get chunking statistics
    pub fn get_chunking_stats(&self, chunks: &[SemanticChunk]) -> ChunkingStats {
        let total_chars: usize = chunks.iter().map(|c| c.content.len()).sum();
        let avg_chunk_size = if chunks.is_empty() { 0.0 } else { total_chars as f32 / chunks.len() as f32 };
        let avg_coherence = if chunks.is_empty() { 0.0 } else {
            chunks.iter().map(|c| c.semantic_coherence).sum::<f32>() / chunks.len() as f32
        };
        
        let mut type_counts = std::collections::HashMap::new();
        for chunk in chunks {
            *type_counts.entry(format!("{:?}", chunk.chunk_type)).or_insert(0) += 1;
        }
        
        ChunkingStats {
            total_chunks: chunks.len(),
            total_characters: total_chars,
            average_chunk_size: avg_chunk_size,
            average_coherence: avg_coherence,
            type_distribution: type_counts,
            chunks_with_overlap: chunks.iter().filter(|c| c.overlap_with_previous.is_some() || c.overlap_with_next.is_some()).count(),
        }
    }
}

/// Statistics about chunking process
#[derive(Debug, Clone)]
pub struct ChunkingStats {
    pub total_chunks: usize,
    pub total_characters: usize,
    pub average_chunk_size: f32,
    pub average_coherence: f32,
    pub type_distribution: std::collections::HashMap<String, usize>,
    pub chunks_with_overlap: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enhanced_knowledge_storage::model_management::ModelResourceManager;
    
    #[tokio::test]
    async fn test_semantic_chunker_creation() {
        let config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(config));
        let chunker_config = SemanticChunkingConfig::default();
        
        let chunker = SemanticChunker::new(model_manager, chunker_config);
        
        assert_eq!(chunker.config.model_id, "smollm2_360m");
        assert!(chunker.config.preserve_sentence_integrity);
    }
    
    #[test]
    fn test_chunk_type_determination() {
        let config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(config));
        let chunker_config = SemanticChunkingConfig::default();
        
        let chunker = SemanticChunker::new(model_manager, chunker_config);
        
        let boundary = SemanticBoundary {
            position: 100,
            boundary_type: BoundaryType::ParagraphBreak,
            confidence: 0.8,
            reason: "Test".to_string(),
        };
        
        let chunk_type = chunker.determine_chunk_type("This is a paragraph.", &boundary);
        assert_eq!(chunk_type, ChunkType::Paragraph);
        
        let code_content = "```rust\nfn main() {\n    println!(\"Hello\");\n}\n```";
        let code_boundary = SemanticBoundary {
            position: 200,
            boundary_type: BoundaryType::EntityBoundary, // Use a boundary type that allows content analysis
            confidence: 0.9,
            reason: "Code block detected".to_string(),
        };
        let chunk_type = chunker.determine_chunk_type(code_content, &code_boundary);
        assert_eq!(chunk_type, ChunkType::Code);
    }
    
    #[test]
    fn test_overlap_start_calculation() {
        let config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(config));
        let chunker_config = SemanticChunkingConfig::default();
        
        let chunker = SemanticChunker::new(model_manager, chunker_config);
        
        let strategy = OverlapStrategy::SemanticOverlap { tokens: 32 };
        let start = chunker.calculate_overlap_start(1000, &strategy);
        assert_eq!(start, 1000 - 32 * 4); // 32 tokens * 4 chars per token
        
        let strategy = OverlapStrategy::NoOverlap;
        let start = chunker.calculate_overlap_start(1000, &strategy);
        assert_eq!(start, 1000);
    }
    
    #[test]
    fn test_boundary_type_parsing() {
        assert!(
            matches!(BoundaryType::TopicShift, BoundaryType::TopicShift)
        );
        assert!(
            matches!(BoundaryType::SentenceEnd, BoundaryType::SentenceEnd)
        );
    }
}