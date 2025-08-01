//! Fast string normalization module
//!
//! This module provides efficient text normalization using heuristic-based approaches.
//! All operations are designed to be deterministic and sub-millisecond.

use crate::error::Result;
use std::collections::HashMap;

/// Fast string normalizer using heuristic approaches
#[derive(Clone)]
pub struct StringNormalizer {
    stop_words: Vec<&'static str>,
    synonyms: HashMap<String, String>,
}

impl StringNormalizer {
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
            synonyms: Self::create_basic_synonyms(),
        }
    }


    /// Normalize text to canonical form
    pub fn canonicalize(&self, text: &str) -> Result<String> {
        self.normalize(text)
    }

    /// Normalize text to canonical form
    pub fn normalize(&self, text: &str) -> Result<String> {
        let normalized = self.normalize_sync(text);
        Ok(normalized)
    }

    /// Synchronous normalization for internal use
    pub fn normalize_sync(&self, text: &str) -> String {
        let mut result = text.to_lowercase();
        
        // Remove punctuation and special characters
        result = result.chars()
            .map(|c| if c.is_alphanumeric() || c.is_whitespace() { c } else { ' ' })
            .collect();
        
        // Normalize whitespace
        result = result.split_whitespace()
            .collect::<Vec<&str>>()
            .join(" ");
        
        // Apply basic synonyms
        for (from, to) in &self.synonyms {
            result = result.replace(from, to);
        }
        
        // Remove stop words (optional - can be configured)
        let words: Vec<&str> = result.split_whitespace()
            .filter(|word| !self.stop_words.contains(word) && word.len() > 2)
            .collect();
        
        words.join(" ")
    }

    /// Batch normalization for multiple texts
    pub fn normalize_batch(&self, texts: &[String]) -> Result<Vec<String>> {
        let results = texts.iter()
            .map(|text| self.normalize_sync(text))
            .collect();
        Ok(results)
    }

    /// Check if two texts are canonically equivalent
    pub fn are_equivalent(&self, text1: &str, text2: &str) -> bool {
        let norm1 = self.normalize_sync(text1);
        let norm2 = self.normalize_sync(text2);
        norm1 == norm2
    }

    /// Extract key terms from text
    pub fn extract_key_terms(&self, text: &str) -> Vec<String> {
        let normalized = self.normalize_sync(text);
        normalized.split_whitespace()
            .filter(|word| word.len() > 3)
            .map(|word| word.to_string())
            .collect()
    }

    /// Create basic synonym mappings
    fn create_basic_synonyms() -> HashMap<String, String> {
        let mut synonyms = HashMap::new();
        
        // Common technical synonyms
        synonyms.insert("db".to_string(), "database".to_string());
        synonyms.insert("api".to_string(), "interface".to_string());
        synonyms.insert("ui".to_string(), "interface".to_string());
        synonyms.insert("url".to_string(), "link".to_string());
        synonyms.insert("id".to_string(), "identifier".to_string());
        synonyms.insert("config".to_string(), "configuration".to_string());
        synonyms.insert("auth".to_string(), "authentication".to_string());
        synonyms.insert("repo".to_string(), "repository".to_string());
        synonyms.insert("doc".to_string(), "document".to_string());
        synonyms.insert("info".to_string(), "information".to_string());
        
        // Common contractions and variations
        synonyms.insert("can't".to_string(), "cannot".to_string());
        synonyms.insert("won't".to_string(), "will not".to_string());
        synonyms.insert("don't".to_string(), "do not".to_string());
        synonyms.insert("doesn't".to_string(), "does not".to_string());
        synonyms.insert("isn't".to_string(), "is not".to_string());
        synonyms.insert("aren't".to_string(), "are not".to_string());
        
        synonyms
    }
}

impl Default for StringNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_normalization() {
        let normalizer = StringNormalizer::new();
        
        let text = "The Quick Brown Fox!";
        let normalized = normalizer.normalize_sync(text);
        assert_eq!(normalized, "quick brown fox");
    }

    #[test]
    fn test_canonicalize_compatibility() {
        let normalizer = StringNormalizer::new();
        
        let text = "Database Configuration API";
        let result = normalizer.canonicalize(text).unwrap();
        assert_eq!(result, "database configuration interface");
    }

    #[test]
    fn test_equivalence() {
        let normalizer = StringNormalizer::new();
        
        assert!(normalizer.are_equivalent("Database API", "db interface"));
        assert!(normalizer.are_equivalent("can't do it", "cannot"));
    }

    #[test]
    fn test_key_terms_extraction() {
        let normalizer = StringNormalizer::new();
        
        let text = "Machine learning algorithm optimization";
        let terms = normalizer.extract_key_terms(text);
        assert!(terms.contains(&"machine".to_string()));
        assert!(terms.contains(&"learning".to_string()));
        assert!(terms.contains(&"algorithm".to_string()));
        assert!(terms.contains(&"optimization".to_string()));
    }

    #[test]
    fn test_batch_normalization() {
        let normalizer = StringNormalizer::new();
        
        let texts = vec![
            "Database API".to_string(),
            "User Interface".to_string(),
            "Authentication Service".to_string(),
        ];
        
        let results = normalizer.normalize_batch(&texts).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0], "database interface");
        assert_eq!(results[1], "user interface");
        assert_eq!(results[2], "authentication service");
    }

    #[test]
    fn test_performance() {
        let normalizer = StringNormalizer::new();
        let text = "The quick brown fox jumps over the lazy dog. This is a performance test.";
        
        let start = std::time::Instant::now();
        for _ in 0..1000 {
            let _ = normalizer.normalize_sync(text);
        }
        let elapsed = start.elapsed();
        
        // Should process 1000 normalizations in under 100ms
        assert!(elapsed.as_millis() < 100);
        println!("1000 normalizations took: {:?}", elapsed);
    }
}