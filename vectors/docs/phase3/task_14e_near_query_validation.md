# Task 14e: Add NEAR Query Input Validation

**Estimated Time: 10 minutes**  
**Lines of Code: ~30**
**Prerequisites: Task 14a (ProximitySearchError enum) completed**

## Context
NEAR query syntax (e.g., "term1 NEAR/5 term2") requires specialized validation for distance parsing, term extraction, and syntax validation. This task adds comprehensive validation for NEAR query operations.

## Your Task
Add input validation for NEAR query syntax with proper parsing and error handling using the ProximitySearchError enum.

## Required Implementation

Create/update `src/near_query/validation.rs`:

```rust
//! Input validation for NEAR query operations

use crate::proximity_search::error::{ProximitySearchError, ProximitySearchResult};
use regex::Regex;
use std::collections::HashMap;

/// NEAR query components after parsing
#[derive(Debug, Clone, PartialEq)]
pub struct NearQueryComponents {
    pub left_term: String,
    pub right_term: String,
    pub distance: u32,
    pub original_query: String,
}

/// Validation helper for NEAR query syntax
pub struct NearQueryValidator {
    max_distance: u32,
    max_term_length: usize,
    near_regex: Regex,
}

impl Default for NearQueryValidator {
    fn default() -> Self {
        Self::new(100, 50) // max_distance=100, max_term_length=50
    }
}

impl NearQueryValidator {
    pub fn new(max_distance: u32, max_term_length: usize) -> Self {
        // Regex pattern for NEAR queries: "term1 NEAR/distance term2"
        let near_pattern = r"^\s*(.+?)\s+NEAR/(\d+)\s+(.+?)\s*$";
        let near_regex = Regex::new(near_pattern)
            .expect("Invalid NEAR query regex pattern");
        
        Self {
            max_distance,
            max_term_length,
            near_regex,
        }
    }
    
    /// Parse and validate a NEAR query string
    pub fn parse_near_query(&self, query: &str) -> ProximitySearchResult<NearQueryComponents> {
        if query.trim().is_empty() {
            return Err(ProximitySearchError::EmptyQuery);
        }
        
        // Check for NEAR keyword presence
        if !query.to_uppercase().contains("NEAR/") {
            return Err(ProximitySearchError::query_parsing_failed(
                query,
                "Query must contain 'NEAR/distance' syntax (e.g., 'term1 NEAR/5 term2')"
            ));
        }
        
        // Parse using regex
        let captures = self.near_regex.captures(query)
            .ok_or_else(|| ProximitySearchError::query_parsing_failed(
                query,
                "Invalid NEAR query syntax. Expected format: 'term1 NEAR/distance term2'"
            ))?;
        
        let left_term = captures.get(1).unwrap().as_str().trim().to_string();
        let distance_str = captures.get(2).unwrap().as_str();
        let right_term = captures.get(3).unwrap().as_str().trim().to_string();
        
        // Validate distance
        let distance = distance_str.parse::<u32>()
            .map_err(|_| ProximitySearchError::query_parsing_failed(
                query,
                format!("Invalid distance value: '{}'. Must be a positive integer", distance_str)
            ))?;
        
        if distance == 0 {
            return Err(ProximitySearchError::InvalidDistance { 
                distance: 0, 
                max_distance: self.max_distance 
            });
        }
        
        if distance > self.max_distance {
            return Err(ProximitySearchError::invalid_distance(distance, self.max_distance));
        }
        
        // Validate terms
        self.validate_near_term(&left_term, "left")?;
        self.validate_near_term(&right_term, "right")?;
        
        Ok(NearQueryComponents {
            left_term,
            right_term,
            distance,
            original_query: query.to_string(),
        })
    }
    
    /// Validate a single term in NEAR query
    pub fn validate_near_term(&self, term: &str, position: &str) -> ProximitySearchResult<()> {
        if term.is_empty() {
            return Err(ProximitySearchError::query_parsing_failed(
                term,
                format!("Empty {} term in NEAR query", position)
            ));
        }
        
        if term.len() > self.max_term_length {
            return Err(ProximitySearchError::ConfigurationError {
                setting: "max_term_length".to_string(),
                value: term.len().to_string(),
                suggestion: format!("Reduce {} term to {} characters or fewer", position, self.max_term_length),
            });
        }
        
        // Check for invalid characters
        if term.chars().any(|c| c.is_control() && !c.is_whitespace()) {
            return Err(ProximitySearchError::InvalidSearchTerms {
                terms: vec![term.to_string()],
            });
        }
        
        // Check for nested NEAR queries (not supported)
        if term.to_uppercase().contains("NEAR/") {
            return Err(ProximitySearchError::query_parsing_failed(
                term,
                "Nested NEAR queries are not supported"
            ));
        }
        
        Ok(())
    }
    
    /// Validate multiple NEAR queries for batch processing
    pub fn validate_near_queries(&self, queries: &[String]) -> ProximitySearchResult<Vec<NearQueryComponents>> {
        if queries.is_empty() {
            return Err(ProximitySearchError::EmptyQuery);
        }
        
        let mut parsed_queries = Vec::new();
        
        for query in queries {
            let parsed = self.parse_near_query(query)?;
            parsed_queries.push(parsed);
        }
        
        Ok(parsed_queries)
    }
    
    /// Extract statistics from parsed NEAR queries
    pub fn analyze_near_queries(&self, components: &[NearQueryComponents]) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        
        stats.insert("total_queries".to_string(), components.len());
        
        let avg_distance = if !components.is_empty() {
            components.iter().map(|c| c.distance as usize).sum::<usize>() / components.len()
        } else {
            0
        };
        stats.insert("average_distance".to_string(), avg_distance);
        
        let max_distance = components.iter()
            .map(|c| c.distance as usize)
            .max()
            .unwrap_or(0);
        stats.insert("max_distance".to_string(), max_distance);
        
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_valid_near_queries() {
        let validator = NearQueryValidator::default();
        
        // Valid NEAR queries
        let result = validator.parse_near_query("hello NEAR/5 world").unwrap();
        assert_eq!(result.left_term, "hello");
        assert_eq!(result.right_term, "world");
        assert_eq!(result.distance, 5);
        
        let result = validator.parse_near_query("  rust   NEAR/10   programming  ").unwrap();
        assert_eq!(result.left_term, "rust");
        assert_eq!(result.right_term, "programming");
        assert_eq!(result.distance, 10);
    }
    
    #[test]
    fn test_invalid_near_queries() {
        let validator = NearQueryValidator::default();
        
        // Missing NEAR keyword
        assert!(validator.parse_near_query("hello world").is_err());
        
        // Invalid distance
        assert!(validator.parse_near_query("hello NEAR/abc world").is_err());
        assert!(validator.parse_near_query("hello NEAR/0 world").is_err());
        assert!(validator.parse_near_query("hello NEAR/999 world").is_err());
        
        // Empty terms
        assert!(validator.parse_near_query("NEAR/5 world").is_err());
        assert!(validator.parse_near_query("hello NEAR/5").is_err());
        
        // Empty query
        assert!(validator.parse_near_query("").is_err());
        assert!(validator.parse_near_query("   ").is_err());
    }
    
    #[test]
    fn test_near_term_validation() {
        let validator = NearQueryValidator::default();
        
        // Valid terms
        assert!(validator.validate_near_term("hello", "left").is_ok());
        assert!(validator.validate_near_term("world test", "right").is_ok());
        
        // Invalid terms
        assert!(validator.validate_near_term("", "left").is_err());
        
        // Too long term
        let long_term = "a".repeat(60);
        assert!(validator.validate_near_term(&long_term, "left").is_err());
        
        // Nested NEAR (not supported)
        assert!(validator.validate_near_term("hello NEAR/3 nested", "left").is_err());
    }
    
    #[test]
    fn test_batch_near_validation() {
        let validator = NearQueryValidator::default();
        
        let valid_queries = vec![
            "hello NEAR/5 world".to_string(),
            "rust NEAR/3 programming".to_string(),
        ];
        
        let results = validator.validate_near_queries(&valid_queries).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].distance, 5);
        assert_eq!(results[1].distance, 3);
        
        // Empty queries
        assert!(validator.validate_near_queries(&[]).is_err());
    }
}
```

## Dependencies to Add
Add to `Cargo.toml` if not already present:
```toml
[dependencies]
regex = "1.0"
```

## Success Criteria
- [ ] NearQueryValidator struct with configurable limits
- [ ] parse_near_query method with comprehensive parsing
- [ ] validate_near_term method for individual term validation
- [ ] validate_near_queries method for batch processing
- [ ] NearQueryComponents struct for parsed query data
- [ ] Regex-based parsing with proper error handling
- [ ] analyze_near_queries method for query statistics
- [ ] Unit tests covering all NEAR query scenarios
- [ ] File compiles without errors

## Validation
Run `cargo test` - all NEAR query validation tests should pass.

## Next Task
Task 14f will add input validation to pattern search methods with comprehensive pattern validation.