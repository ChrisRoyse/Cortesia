use crate::core::triple::Triple;
use crate::error::Result;

pub struct TripleExtractor;

impl Default for TripleExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl TripleExtractor {
    pub fn new() -> Self {
        Self
    }

    pub fn extract_triples_from_text(&self, text: &str) -> Result<Vec<Triple>> {
        // Simplified triple extraction - in production, use NLP models
        let mut triples = Vec::new();
        
        // Look for simple patterns like "X is Y", "X has Y", etc.
        let sentences: Vec<&str> = text.split('.').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
        
        for sentence in sentences {
            if let Some(triple) = self.extract_simple_triple(sentence) {
                triples.push(triple);
            }
        }
        
        Ok(triples)
    }
    
    fn extract_simple_triple(&self, sentence: &str) -> Option<Triple> {
        let words: Vec<&str> = sentence.split_whitespace().collect();
        if words.len() < 3 {
            return None;
        }
        
        // Look for "X is Y" pattern
        if let Some(is_pos) = words.iter().position(|&w| w.to_lowercase() == "is") {
            if is_pos > 0 && is_pos < words.len() - 1 {
                let subject = words[..is_pos].join(" ");
                let object = words[is_pos + 1..].join(" ");
                
                if let Ok(triple) = Triple::new(subject, "is".to_string(), object) {
                    return Some(triple);
                }
            }
        }
        
        // Look for "X has Y" pattern
        if let Some(has_pos) = words.iter().position(|&w| w.to_lowercase() == "has") {
            if has_pos > 0 && has_pos < words.len() - 1 {
                let subject = words[..has_pos].join(" ");
                let object = words[has_pos + 1..].join(" ");
                
                if let Ok(triple) = Triple::new(subject, "has".to_string(), object) {
                    return Some(triple);
                }
            }
        }
        
        None
    }
}