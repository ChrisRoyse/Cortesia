# Task 14c: Add Proximity Search Input Validation

**Estimated Time: 10 minutes**  
**Lines of Code: ~30**
**Prerequisites: Task 14a (ProximitySearchError enum) completed**

## Context
The current proximity search methods lack comprehensive input validation. This task adds defensive programming with proper validation of search terms, distances, and configuration parameters using the ProximitySearchError enum.

## Your Task
Add input validation to all proximity search methods with clear error messages and defensive checks.

## Required Implementation

Update `src/proximity_search/mod.rs` to add validation methods:

```rust
//! Input validation for proximity search operations

use crate::proximity_search::error::{ProximitySearchError, ProximitySearchResult};

/// Configuration limits for proximity search validation
pub struct ProximitySearchLimits {
    pub max_distance: u32,
    pub max_term_length: usize,
    pub max_terms_count: usize,
    pub min_term_length: usize,
}

impl Default for ProximitySearchLimits {
    fn default() -> Self {
        Self {
            max_distance: 100,        // Maximum word distance
            max_term_length: 100,     // Maximum characters per term
            max_terms_count: 20,      // Maximum number of search terms
            min_term_length: 1,       // Minimum characters per term
        }
    }
}

/// Validation helper for proximity search inputs
pub struct ProximitySearchValidator {
    limits: ProximitySearchLimits,
}

impl ProximitySearchValidator {
    pub fn new(limits: ProximitySearchLimits) -> Self {
        Self { limits }
    }
    
    pub fn with_default_limits() -> Self {
        Self::new(ProximitySearchLimits::default())
    }
    
    /// Validate proximity distance parameter
    pub fn validate_distance(&self, distance: u32) -> ProximitySearchResult<()> {
        if distance == 0 {
            return Err(ProximitySearchError::InvalidDistance { 
                distance, 
                max_distance: self.limits.max_distance 
            });
        }
        
        if distance > self.limits.max_distance {
            return Err(ProximitySearchError::invalid_distance(
                distance, 
                self.limits.max_distance
            ));
        }
        
        Ok(())
    }
    
    /// Validate search terms array
    pub fn validate_search_terms(&self, terms: &[String]) -> ProximitySearchResult<()> {
        if terms.is_empty() {
            return Err(ProximitySearchError::EmptyQuery);
        }
        
        if terms.len() > self.limits.max_terms_count {
            return Err(ProximitySearchError::ConfigurationError {
                setting: "max_terms_count".to_string(),
                value: terms.len().to_string(),
                suggestion: format!("Reduce to {} or fewer terms", self.limits.max_terms_count),
            });
        }
        
        let mut invalid_terms = Vec::new();
        
        for term in terms {
            if term.trim().is_empty() {
                invalid_terms.push(term.clone());
                continue;
            }
            
            if term.len() < self.limits.min_term_length {
                invalid_terms.push(term.clone());
                continue;
            }
            
            if term.len() > self.limits.max_term_length {
                invalid_terms.push(term.clone());
                continue;
            }
            
            // Check for invalid characters (control characters, etc.)
            if term.chars().any(|c| c.is_control() && !c.is_whitespace()) {
                invalid_terms.push(term.clone());
            }
        }
        
        if !invalid_terms.is_empty() {
            return Err(ProximitySearchError::InvalidSearchTerms { 
                terms: invalid_terms 
            });
        }
        
        Ok(())
    }
    
    /// Validate complete proximity search request
    pub fn validate_proximity_request(
        &self,
        terms: &[String],
        distance: u32,
    ) -> ProximitySearchResult<()> {
        self.validate_search_terms(terms)?;
        self.validate_distance(distance)?;
        Ok(())
    }
    
    /// Validate query string before parsing
    pub fn validate_query_string(&self, query: &str) -> ProximitySearchResult<()> {
        if query.trim().is_empty() {
            return Err(ProximitySearchError::EmptyQuery);
        }
        
        if query.len() > self.limits.max_term_length * self.limits.max_terms_count {
            return Err(ProximitySearchError::ConfigurationError {
                setting: "query_length".to_string(),
                value: query.len().to_string(),
                suggestion: "Reduce query length or split into multiple searches".to_string(),
            });
        }
        
        // Check for obviously malformed queries
        let open_parens = query.chars().filter(|&c| c == '(').count();
        let close_parens = query.chars().filter(|&c| c == ')').count();
        
        if open_parens != close_parens {
            return Err(ProximitySearchError::query_parsing_failed(
                query,
                "Mismatched parentheses in query"
            ));
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_distance_validation() {
        let validator = ProximitySearchValidator::with_default_limits();
        
        // Valid distances
        assert!(validator.validate_distance(1).is_ok());
        assert!(validator.validate_distance(50).is_ok());
        assert!(validator.validate_distance(100).is_ok());
        
        // Invalid distances
        assert!(validator.validate_distance(0).is_err());
        assert!(validator.validate_distance(101).is_err());
    }
    
    #[test]
    fn test_search_terms_validation() {
        let validator = ProximitySearchValidator::with_default_limits();
        
        // Valid terms
        assert!(validator.validate_search_terms(&["hello".to_string(), "world".to_string()]).is_ok());
        
        // Empty terms array
        assert!(validator.validate_search_terms(&[]).is_err());
        
        // Empty term
        assert!(validator.validate_search_terms(&["".to_string()]).is_err());
        
        // Too many terms
        let many_terms: Vec<String> = (0..25).map(|i| format!("term{}", i)).collect();
        assert!(validator.validate_search_terms(&many_terms).is_err());
    }
    
    #[test]
    fn test_query_string_validation() {
        let validator = ProximitySearchValidator::with_default_limits();
        
        // Valid queries
        assert!(validator.validate_query_string("hello world").is_ok());
        assert!(validator.validate_query_string("(term1 AND term2)").is_ok());
        
        // Invalid queries
        assert!(validator.validate_query_string("").is_err());
        assert!(validator.validate_query_string("   ").is_err());
        assert!(validator.validate_query_string("(unclosed paren").is_err());
    }
}
```

## Success Criteria
- [ ] ProximitySearchValidator struct with configurable limits
- [ ] validate_distance method with clear error messages
- [ ] validate_search_terms method with comprehensive checks
- [ ] validate_proximity_request method for complete validation
- [ ] validate_query_string method for query preprocessing
- [ ] Unit tests covering all validation scenarios
- [ ] Integration with ProximitySearchError enum
- [ ] File compiles without errors

## Validation
Run `cargo test` - all validation tests should pass.

## Next Task
Task 14d will add input validation to phrase search methods with proper error handling.