//! Fast text processing for data compression
//! 
//! This module provides ultra-fast text summarization to prevent data bloat
//! in the knowledge graph. All operations are designed to be sub-millisecond.

use crate::error::Result;
use std::collections::HashMap;

const MAX_NODE_WORDS: usize = 400;
const TARGET_SUMMARY_WORDS: usize = 75; // Target 50-100 words

/// Ultra-fast text summarizer for preventing data bloat
pub struct TextCompressor {
    stop_words: Vec<&'static str>,
}

impl TextCompressor {
    pub fn new() -> Self {
        Self {
            // Common English stop words to filter out
            stop_words: vec![
                "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
                "of", "with", "by", "from", "as", "is", "was", "are", "been", "be",
                "have", "has", "had", "do", "does", "did", "will", "would", "could",
                "should", "may", "might", "must", "can", "this", "that", "these",
                "those", "i", "you", "he", "she", "it", "we", "they", "them", "their",
                "what", "which", "who", "when", "where", "why", "how", "all", "each",
                "every", "some", "any", "many", "much", "most", "several", "no", "not",
                "only", "own", "same", "so", "than", "too", "very", "just", "about"
            ],
        }
    }

    /// Compress text to prevent bloat - MUST be sub-millisecond
    pub fn compress(&self, text: &str) -> String {
        let words: Vec<&str> = text.split_whitespace().collect();
        
        // If already within limits, return as-is
        if words.len() <= TARGET_SUMMARY_WORDS {
            return text.to_string();
        }

        // Fast word frequency analysis
        let word_scores = self.calculate_word_scores(&words);
        
        // Extract key sentences based on word scores
        let sentences = self.extract_sentences(text);
        let selected_sentences = self.select_top_sentences(&sentences, &word_scores);
        
        // Join and truncate to target length
        self.build_summary(&selected_sentences, TARGET_SUMMARY_WORDS)
    }

    /// Calculate TF-IDF-like scores for words (simplified for speed)
    fn calculate_word_scores(&self, words: &[&str]) -> HashMap<String, f32> {
        let mut word_freq = HashMap::new();
        let mut scores = HashMap::new();
        
        // Count frequencies
        for word in words {
            let lower = word.to_lowercase();
            if !self.stop_words.contains(&lower.as_str()) && word.len() > 2 {
                *word_freq.entry(lower).or_insert(0) += 1;
            }
        }
        
        // Calculate scores (simplified TF-IDF)
        let total_words = words.len() as f32;
        for (word, freq) in word_freq {
            let tf = freq as f32 / total_words;
            let idf = (total_words / (freq as f32 + 1.0)).ln();
            scores.insert(word, tf * idf);
        }
        
        scores
    }

    /// Fast sentence extraction
    fn extract_sentences<'a>(&self, text: &'a str) -> Vec<&'a str> {
        text.split(['.', '!', '?'])
            .filter(|s| !s.trim().is_empty())
            .collect()
    }

    /// Select top sentences based on word scores
    fn select_top_sentences<'a>(&self, sentences: &[&'a str], word_scores: &HashMap<String, f32>) -> Vec<&'a str> {
        let mut sentence_scores: Vec<(usize, f32)> = sentences.iter()
            .enumerate()
            .map(|(idx, sentence)| {
                let score = sentence.split_whitespace()
                    .map(|word| {
                        word_scores.get(&word.to_lowercase()).unwrap_or(&0.0)
                    })
                    .sum::<f32>();
                (idx, score)
            })
            .collect();
        
        // Sort by score
        sentence_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Take top sentences
        let mut selected = Vec::new();
        let mut word_count = 0;
        
        for (idx, _score) in sentence_scores {
            let sentence = sentences[idx];
            let sentence_words = sentence.split_whitespace().count();
            
            if word_count + sentence_words <= TARGET_SUMMARY_WORDS {
                selected.push((idx, sentence));
                word_count += sentence_words;
            }
            
            if word_count >= TARGET_SUMMARY_WORDS * 3 / 4 {
                break;
            }
        }
        
        // Sort by original order
        selected.sort_by_key(|(idx, _)| *idx);
        selected.into_iter().map(|(_, s)| s).collect()
    }

    /// Build final summary
    fn build_summary(&self, sentences: &[&str], max_words: usize) -> String {
        let mut result = String::new();
        let mut word_count = 0;
        
        for sentence in sentences {
            let sentence_words: Vec<&str> = sentence.split_whitespace().collect();
            
            if word_count + sentence_words.len() <= max_words {
                if !result.is_empty() {
                    result.push_str(". ");
                }
                result.push_str(sentence.trim());
                word_count += sentence_words.len();
            } else if word_count < max_words {
                // Partial sentence to reach target
                let remaining = max_words - word_count;
                if !result.is_empty() {
                    result.push_str(". ");
                }
                result.push_str(&sentence_words[..remaining].join(" "));
                result.push_str("...");
                break;
            }
        }
        
        if !result.ends_with('.') && !result.ends_with("...") {
            result.push('.');
        }
        
        result
    }

    /// Validate text doesn't exceed limits
    pub fn validate_text_size(text: &str) -> Result<()> {
        let word_count = text.split_whitespace().count();
        if word_count > MAX_NODE_WORDS {
            return Err(crate::error::GraphError::InvalidInput(
                format!("Text exceeds maximum word limit of {MAX_NODE_WORDS}. Found {word_count} words. Please summarize before storing.")
            ));
        }
        Ok(())
    }
}

impl Default for TextCompressor {
    fn default() -> Self {
        Self::new()
    }
}

pub mod chunkers;
pub mod normalizer;
pub mod importance;
pub mod structure_predictor;

pub use chunkers::{TextChunk, Chunker, SlidingWindowChunker, SemanticChunker, AdaptiveChunker};
pub use normalizer::StringNormalizer;
pub use importance::{HeuristicImportanceScorer, GraphMetrics};
pub use structure_predictor::{GraphStructurePredictor, GraphOperation};

/// Fast text utilities
pub mod utils {
    /// Count words in text
    pub fn word_count(text: &str) -> usize {
        text.split_whitespace().count()
    }
    
    /// Truncate text to word limit
    pub fn truncate_to_words(text: &str, max_words: usize) -> String {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.len() <= max_words {
            text.to_string()
        } else {
            format!("{}...", words[..max_words].join(" "))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_compression() {
        let compressor = TextCompressor::new();
        
        // Short text should pass through
        let short_text = "This is a short text.";
        assert_eq!(compressor.compress(short_text), short_text);
        
        // Long text should be compressed
        let long_text = "The quick brown fox jumps over the lazy dog. ".repeat(20);
        let compressed = compressor.compress(&long_text);
        let word_count = compressed.split_whitespace().count();
        
        assert!(word_count <= TARGET_SUMMARY_WORDS + 10); // Allow small margin
        assert!(word_count >= 50); // Should be at least 50 words
    }

    #[test]
    fn test_validation() {
        let valid_text = "word ".repeat(300);
        assert!(TextCompressor::validate_text_size(&valid_text).is_ok());
        
        let invalid_text = "word ".repeat(500);
        assert!(TextCompressor::validate_text_size(&invalid_text).is_err());
    }

    #[test]
    fn test_performance() {
        let compressor = TextCompressor::new();
        let text = "The knowledge graph system is designed for ultra-fast performance. ".repeat(50);
        
        let start = std::time::Instant::now();
        for _ in 0..1000 {
            let _ = compressor.compress(&text);
        }
        let elapsed = start.elapsed();
        
        // Should process 1000 compressions in under 5 seconds
        assert!(elapsed.as_millis() < 5000);
        println!("1000 compressions took: {elapsed:?}");
    }
}