# Task 14d: Add Phrase Search Input Validation

**Estimated Time: 8 minutes**  
**Lines of Code: ~20**
**Prerequisites: Task 14a (ProximitySearchError enum) completed**

## Context
Phrase search operations require specific validation for phrase structure, quotation handling, and term ordering. This task adds defensive validation for phrase search inputs with reuse of the ProximitySearchError enum.

## Your Task
Add input validation specifically for phrase search operations, focusing on phrase structure and quotation validation.

## Required Implementation

Create/update `src/phrase_search/validation.rs`:

```rust
//! Input validation for phrase search operations

use crate::proximity_search::error::{ProximitySearchError, ProximitySearchResult};

/// Validation helper for phrase search inputs
pub struct PhraseSearchValidator {
    max_phrase_length: usize,
    max_words_in_phrase: usize,
}

impl Default for PhraseSearchValidator {
    fn default() -> Self {
        Self {
            max_phrase_length: 200,    // Maximum characters in phrase
            max_words_in_phrase: 10,   // Maximum words in a single phrase
        }
    }
}

impl PhraseSearchValidator {
    pub fn new(max_phrase_length: usize, max_words_in_phrase: usize) -> Self {
        Self {
            max_phrase_length,
            max_words_in_phrase,
        }
    }
    
    /// Validate a phrase string for search
    pub fn validate_phrase(&self, phrase: &str) -> ProximitySearchResult<()> {
        let trimmed_phrase = phrase.trim();
        
        if trimmed_phrase.is_empty() {
            return Err(ProximitySearchError::EmptyQuery);
        }
        
        if trimmed_phrase.len() > self.max_phrase_length {
            return Err(ProximitySearchError::ConfigurationError {
                setting: "max_phrase_length".to_string(),
                value: trimmed_phrase.len().to_string(),
                suggestion: format!("Reduce phrase to {} characters or fewer", self.max_phrase_length),
            });
        }
        
        // Count words in phrase
        let word_count = trimmed_phrase.split_whitespace().count();
        if word_count > self.max_words_in_phrase {
            return Err(ProximitySearchError::ConfigurationError {
                setting: "max_words_in_phrase".to_string(),
                value: word_count.to_string(),
                suggestion: format!("Reduce phrase to {} words or fewer", self.max_words_in_phrase),
            });
        }
        
        if word_count == 0 {
            return Err(ProximitySearchError::EmptyQuery);
        }
        
        Ok(())
    }
    
    /// Validate quoted phrase (handles quote matching)
    pub fn validate_quoted_phrase(&self, quoted_phrase: &str) -> ProximitySearchResult<String> {
        let trimmed = quoted_phrase.trim();
        
        if trimmed.is_empty() {
            return Err(ProximitySearchError::EmptyQuery);
        }
        
        // Handle quoted phrases
        if trimmed.starts_with('"') || trimmed.starts_with('\'') {
            let quote_char = trimmed.chars().next().unwrap();
            if !trimmed.ends_with(quote_char) {
                return Err(ProximitySearchError::query_parsing_failed(
                    quoted_phrase,
                    format!("Unclosed {} quote in phrase", quote_char)
                ));
            }
            
            // Extract content between quotes
            let inner_phrase = &trimmed[1..trimmed.len()-1];
            self.validate_phrase(inner_phrase)?;
            Ok(inner_phrase.to_string())
        } else {
            // Unquoted phrase
            self.validate_phrase(trimmed)?;
            Ok(trimmed.to_string())
        }
    }
    
    /// Validate multiple phrases for batch search
    pub fn validate_phrases(&self, phrases: &[String]) -> ProximitySearchResult<()> {
        if phrases.is_empty() {
            return Err(ProximitySearchError::EmptyQuery);
        }
        
        for phrase in phrases {
            self.validate_phrase(phrase)?;
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_phrase_validation() {
        let validator = PhraseSearchValidator::default();
        
        // Valid phrases
        assert!(validator.validate_phrase("hello world").is_ok());
        assert!(validator.validate_phrase("rust programming language").is_ok());
        
        // Invalid phrases
        assert!(validator.validate_phrase("").is_err());
        assert!(validator.validate_phrase("   ").is_err());
        
        // Too long phrase
        let long_phrase = "word ".repeat(15); // 15 words
        assert!(validator.validate_phrase(&long_phrase).is_err());
    }
    
    #[test]
    fn test_quoted_phrase_validation() {
        let validator = PhraseSearchValidator::default();
        
        // Valid quoted phrases
        assert_eq!(
            validator.validate_quoted_phrase("\"hello world\"").unwrap(),
            "hello world"
        );
        assert_eq!(
            validator.validate_quoted_phrase("'rust code'").unwrap(),
            "rust code"
        );
        
        // Unquoted phrase
        assert_eq!(
            validator.validate_quoted_phrase("plain phrase").unwrap(),
            "plain phrase"
        );
        
        // Invalid quoted phrases
        assert!(validator.validate_quoted_phrase("\"unclosed quote").is_err());
        assert!(validator.validate_quoted_phrase("'mismatched\"").is_err());
    }
    
    #[test]
    fn test_multiple_phrases_validation() {
        let validator = PhraseSearchValidator::default();
        
        let valid_phrases = vec![
            "first phrase".to_string(),
            "second phrase".to_string(),
        ];
        assert!(validator.validate_phrases(&valid_phrases).is_ok());
        
        // Empty phrases array
        assert!(validator.validate_phrases(&[]).is_err());
        
        // Contains invalid phrase
        let invalid_phrases = vec![
            "valid phrase".to_string(),
            "".to_string(), // Invalid empty phrase
        ];
        assert!(validator.validate_phrases(&invalid_phrases).is_err());
    }
}
```

## Success Criteria
- [ ] PhraseSearchValidator struct with configurable limits
- [ ] validate_phrase method for basic phrase validation
- [ ] validate_quoted_phrase method handling quote matching
- [ ] validate_phrases method for batch validation
- [ ] Proper integration with ProximitySearchError enum
- [ ] Unit tests covering phrase validation scenarios
- [ ] File compiles without errors
- [ ] Tests pass

## Validation
Run `cargo test` - all phrase validation tests should pass.

## Next Task
Task 14e will add input validation to NEAR query parsing with distance validation.