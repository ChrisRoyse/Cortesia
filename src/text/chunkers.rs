//! Text chunking implementations for processing large documents
//!
//! Provides various chunking strategies for breaking down text into manageable pieces.

use crate::error::Result;
use std::collections::HashMap;

/// Represents a text chunk with metadata
#[derive(Debug, Clone)]
pub struct TextChunk {
    pub text: String,
    pub metadata: HashMap<String, String>,
}

/// Trait for text chunking strategies
pub trait Chunker {
    fn chunk(&self, text: &str) -> Result<Vec<TextChunk>>;
}

/// Sliding window chunker that creates overlapping chunks
pub struct SlidingWindowChunker {
    window_size: usize,
    overlap_size: usize,
}

impl SlidingWindowChunker {
    pub fn new(window_size: usize, overlap_size: usize) -> Self {
        Self {
            window_size,
            overlap_size,
        }
    }
}

impl Chunker for SlidingWindowChunker {
    fn chunk(&self, text: &str) -> Result<Vec<TextChunk>> {
        if text.is_empty() {
            return Ok(vec![]);
        }
        
        let mut chunks = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let step_size = self.window_size.saturating_sub(self.overlap_size).max(1);
        
        let mut start = 0;
        while start < chars.len() {
            let end = (start + self.window_size).min(chars.len());
            let chunk_text: String = chars[start..end].iter().collect();
            
            let mut metadata = HashMap::new();
            metadata.insert("start_offset".to_string(), start.to_string());
            metadata.insert("end_offset".to_string(), end.to_string());
            metadata.insert("chunk_size".to_string(), chunk_text.len().to_string());
            
            chunks.push(TextChunk {
                text: chunk_text,
                metadata,
            });
            
            if end >= chars.len() {
                break;
            }
            
            start += step_size;
        }
        
        Ok(chunks)
    }
}

/// Semantic chunker that splits text at natural boundaries
pub struct SemanticChunker {
    max_chunk_size: usize,
    similarity_threshold: f32,
}

impl SemanticChunker {
    pub fn new(max_chunk_size: usize, similarity_threshold: f32) -> Self {
        Self {
            max_chunk_size,
            similarity_threshold,
        }
    }
    
    fn find_sentence_boundaries(&self, text: &str) -> Vec<usize> {
        let mut boundaries = vec![0];
        let chars: Vec<char> = text.chars().collect();
        
        for i in 0..chars.len() {
            if matches!(chars[i], '.' | '!' | '?') {
                // Look for next non-whitespace character
                let mut next_start = i + 1;
                while next_start < chars.len() && chars[next_start].is_whitespace() {
                    next_start += 1;
                }
                if next_start < chars.len() {
                    boundaries.push(next_start);
                }
            }
        }
        
        if boundaries.last() != Some(&chars.len()) {
            boundaries.push(chars.len());
        }
        
        boundaries
    }
}

impl Chunker for SemanticChunker {
    fn chunk(&self, text: &str) -> Result<Vec<TextChunk>> {
        if text.is_empty() {
            return Ok(vec![]);
        }
        
        let boundaries = self.find_sentence_boundaries(text);
        let mut chunks = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        
        let mut current_start = 0;
        let mut current_size = 0;
        
        for i in 1..boundaries.len() {
            let sentence_start = boundaries[i - 1];
            let sentence_end = boundaries[i];
            let sentence_size = sentence_end - sentence_start;
            
            if current_size + sentence_size > self.max_chunk_size && current_size > 0 {
                // Create chunk
                let chunk_text: String = chars[current_start..boundaries[i - 1]].iter().collect();
                
                let mut metadata = HashMap::new();
                metadata.insert("start_offset".to_string(), current_start.to_string());
                metadata.insert("end_offset".to_string(), boundaries[i - 1].to_string());
                metadata.insert("sentence_count".to_string(), (i - 1).to_string());
                
                chunks.push(TextChunk {
                    text: chunk_text.trim().to_string(),
                    metadata,
                });
                
                current_start = sentence_start;
                current_size = sentence_size;
            } else {
                current_size += sentence_size;
            }
        }
        
        // Add final chunk
        if current_size > 0 {
            let chunk_text: String = chars[current_start..].iter().collect();
            
            let mut metadata = HashMap::new();
            metadata.insert("start_offset".to_string(), current_start.to_string());
            metadata.insert("end_offset".to_string(), chars.len().to_string());
            
            chunks.push(TextChunk {
                text: chunk_text.trim().to_string(),
                metadata,
            });
        }
        
        Ok(chunks)
    }
}

/// Adaptive chunker that adjusts chunk size based on content
pub struct AdaptiveChunker {
    min_chunk_size: usize,
    max_chunk_size: usize,
}

impl AdaptiveChunker {
    pub fn new(min_chunk_size: usize, max_chunk_size: usize) -> Self {
        Self {
            min_chunk_size,
            max_chunk_size,
        }
    }
    
    fn find_natural_breaks(&self, text: &str) -> Vec<usize> {
        let mut breaks = vec![0];
        let chars: Vec<char> = text.chars().collect();
        
        // Look for paragraph breaks (double newlines)
        for i in 0..chars.len().saturating_sub(1) {
            if chars[i] == '\n' && i + 1 < chars.len() && chars[i + 1] == '\n' {
                // Skip additional newlines
                let mut j = i + 2;
                while j < chars.len() && chars[j] == '\n' {
                    j += 1;
                }
                if j < chars.len() {
                    breaks.push(j);
                }
            }
        }
        
        // Add chapter/section breaks
        let text_lower = text.to_lowercase();
        for (i, _) in text_lower.match_indices("chapter") {
            if i > 0 && !breaks.contains(&i) {
                breaks.push(i);
            }
        }
        
        breaks.push(chars.len());
        breaks.sort_unstable();
        breaks.dedup();
        
        breaks
    }
}

impl Chunker for AdaptiveChunker {
    fn chunk(&self, text: &str) -> Result<Vec<TextChunk>> {
        if text.is_empty() {
            return Ok(vec![]);
        }
        
        if text.len() <= self.max_chunk_size {
            let mut metadata = HashMap::new();
            metadata.insert("chunk_index".to_string(), "0".to_string());
            metadata.insert("chunk_type".to_string(), "single".to_string());
            
            return Ok(vec![TextChunk {
                text: text.to_string(),
                metadata,
            }]);
        }
        
        let breaks = self.find_natural_breaks(text);
        let mut chunks = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        
        let mut current_start = 0;
        
        for i in 1..breaks.len() {
            let segment_start = breaks[i - 1];
            let segment_end = breaks[i];
            let segment_size = segment_end - segment_start;
            
            if segment_size >= self.min_chunk_size || i == breaks.len() - 1 {
                let chunk_text: String = chars[current_start..segment_end].iter().collect();
                
                if chunk_text.trim().is_empty() {
                    continue;
                }
                
                let mut metadata = HashMap::new();
                metadata.insert("chunk_index".to_string(), chunks.len().to_string());
                metadata.insert("chunk_type".to_string(), 
                    if chunk_text.to_lowercase().contains("chapter") { "chapter" } else { "paragraph" }.to_string()
                );
                metadata.insert("start_offset".to_string(), current_start.to_string());
                metadata.insert("end_offset".to_string(), segment_end.to_string());
                
                chunks.push(TextChunk {
                    text: chunk_text.trim().to_string(),
                    metadata,
                });
                
                current_start = segment_end;
            }
        }
        
        Ok(chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sliding_window_basic() {
        let chunker = SlidingWindowChunker::new(10, 2);
        let chunks = chunker.chunk("Hello world, this is a test").unwrap();
        assert!(chunks.len() > 1);
        
        // Verify overlap
        for i in 1..chunks.len() {
            let prev_end = chunks[i-1].metadata["end_offset"].parse::<usize>().unwrap();
            let curr_start = chunks[i].metadata["start_offset"].parse::<usize>().unwrap();
            assert!(prev_end > curr_start);
        }
    }
    
    #[test]
    fn test_semantic_chunker() {
        let chunker = SemanticChunker::new(50, 0.8);
        let text = "First sentence. Second sentence. Third sentence.";
        let chunks = chunker.chunk(text).unwrap();
        
        for chunk in &chunks {
            assert!(chunk.text.ends_with('.') || chunk.text.ends_with("sentence"));
        }
    }
    
    #[test]
    fn test_adaptive_chunker() {
        let chunker = AdaptiveChunker::new(10, 100);
        let text = "Chapter 1\n\nContent that is long enough to exceed the minimum chunk size.\n\nChapter 2\n\nMore content that is also long enough to ensure we get multiple chunks when processing.";
        let chunks = chunker.chunk(text).unwrap();
        
        assert!(chunks.len() >= 1); // At least one chunk
        // Check if we have chapter type chunks
        let has_chapter_chunk = chunks.iter().any(|c| c.metadata.get("chunk_type").map(|t| t == "chapter").unwrap_or(false));
        assert!(has_chapter_chunk || chunks.len() == 1); // Either we have chapters or it's a single chunk
    }
}