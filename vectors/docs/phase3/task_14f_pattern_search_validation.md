# Task 14f: Add Pattern Search Input Validation

**Estimated Time: 10 minutes**  
**Lines of Code: ~30**
**Prerequisites: Task 14b (PatternSearchError enum) completed**

## Context
Pattern search operations involve complex regex, wildcard, and fuzzy search patterns that require comprehensive validation to prevent malicious patterns, performance issues, and syntax errors. This task adds robust validation for all pattern types.

## Your Task
Add comprehensive input validation for pattern search operations with proper pattern complexity analysis and safety checks using the PatternSearchError enum.

## Required Implementation

Create/update `src/pattern_search/validation.rs`:

```rust
//! Input validation for pattern search operations

use crate::pattern_search::error::{PatternSearchError, PatternSearchResult};
use regex::Regex;
use std::collections::HashSet;

/// Pattern complexity metrics
#[derive(Debug, Clone)]
pub struct PatternComplexity {
    pub character_count: usize,
    pub quantifier_count: usize,
    pub group_count: usize,
    pub alternation_count: usize,
    pub complexity_score: usize,
}

/// Validation helper for pattern search inputs
pub struct PatternSearchValidator {
    max_pattern_length: usize,
    max_complexity_score: usize,
    max_wildcard_depth: usize,
    forbidden_patterns: HashSet<String>,
}

impl Default for PatternSearchValidator {
    fn default() -> Self {
        let mut forbidden = HashSet::new();
        // Add potentially dangerous patterns
        forbidden.insert(".*.*.*".to_string());  // Excessive wildcards
        forbidden.insert("(.+)+".to_string());   // Catastrophic backtracking
        
        Self {
            max_pattern_length: 200,
            max_complexity_score: 100,
            max_wildcard_depth: 5,
            forbidden_patterns: forbidden,
        }
    }
}

impl PatternSearchValidator {
    pub fn new(
        max_pattern_length: usize,
        max_complexity_score: usize,
        max_wildcard_depth: usize,
    ) -> Self {
        Self {
            max_pattern_length,
            max_complexity_score,
            max_wildcard_depth,
            forbidden_patterns: HashSet::new(),
        }
    }
    
    /// Validate regex pattern for safety and complexity
    pub fn validate_regex_pattern(&self, pattern: &str) -> PatternSearchResult<()> {
        if pattern.trim().is_empty() {
            return Err(PatternSearchError::EmptyPattern);
        }
        
        if pattern.len() > self.max_pattern_length {
            return Err(PatternSearchError::pattern_too_complex(
                pattern,
                self.max_pattern_length,
                pattern.len(),
            ));
        }
        
        // Check against forbidden patterns
        if self.forbidden_patterns.contains(pattern) {
            return Err(PatternSearchError::invalid_regex(
                pattern,
                "Pattern is in forbidden list due to performance concerns"
            ));
        }
        
        // Test regex compilation
        match Regex::new(pattern) {
            Ok(_) => {
                // Analyze complexity
                let complexity = self.analyze_pattern_complexity(pattern);
                if complexity.complexity_score > self.max_complexity_score {
                    return Err(PatternSearchError::pattern_too_complex(
                        pattern,
                        self.max_complexity_score,
                        complexity.complexity_score,
                    ));
                }
                Ok(())
            }
            Err(e) => Err(PatternSearchError::invalid_regex(
                pattern,
                format!("Regex compilation failed: {}", e)
            )),
        }
    }
    
    /// Validate wildcard pattern (* and ? patterns)
    pub fn validate_wildcard_pattern(&self, pattern: &str) -> PatternSearchResult<()> {
        if pattern.trim().is_empty() {
            return Err(PatternSearchError::EmptyPattern);
        }
        
        if pattern.len() > self.max_pattern_length {
            return Err(PatternSearchError::pattern_too_complex(
                pattern,
                self.max_pattern_length,
                pattern.len(),
            ));
        }
        
        // Check wildcard depth (consecutive wildcards)
        let consecutive_wildcards = self.count_consecutive_wildcards(pattern);
        if consecutive_wildcards > self.max_wildcard_depth {
            return Err(PatternSearchError::invalid_wildcard(
                pattern,
                format!("Too many consecutive wildcards ({}). Maximum allowed: {}", 
                       consecutive_wildcards, self.max_wildcard_depth)
            ));
        }
        
        // Check for patterns that might be too broad
        if pattern == "*" || pattern == "**" || pattern == "***" {
            return Err(PatternSearchError::invalid_wildcard(
                pattern,
                "Overly broad wildcard pattern. Please be more specific"
            ));
        }
        
        // Validate special characters in context
        if pattern.contains("\\") && !self.is_valid_escape_sequence(pattern) {
            return Err(PatternSearchError::invalid_wildcard(
                pattern,
                "Invalid escape sequence in wildcard pattern"
            ));
        }
        
        Ok(())
    }
    
    /// Validate fuzzy search pattern with distance
    pub fn validate_fuzzy_pattern(&self, pattern: &str, max_distance: u32) -> PatternSearchResult<()> {
        if pattern.trim().is_empty() {
            return Err(PatternSearchError::EmptyPattern);
        }
        
        if pattern.len() > self.max_pattern_length {
            return Err(PatternSearchError::pattern_too_complex(
                pattern,
                self.max_pattern_length,
                pattern.len(),
            ));
        }
        
        // Validate fuzzy distance
        if max_distance == 0 {
            return Err(PatternSearchError::invalid_regex(
                pattern,
                "Fuzzy search distance must be greater than 0"
            ));
        }
        
        if max_distance > pattern.len() as u32 {
            return Err(PatternSearchError::invalid_regex(
                pattern,
                format!("Fuzzy distance ({}) cannot exceed pattern length ({})", 
                       max_distance, pattern.len())
            ));
        }
        
        // Check for valid characters (no control characters)
        if pattern.chars().any(|c| c.is_control() && !c.is_whitespace()) {
            return Err(PatternSearchError::invalid_regex(
                pattern,
                "Fuzzy search pattern contains invalid control characters"
            ));
        }
        
        Ok(())
    }
    
    /// Analyze pattern complexity for performance assessment
    pub fn analyze_pattern_complexity(&self, pattern: &str) -> PatternComplexity {
        let character_count = pattern.len();
        let quantifier_count = pattern.matches(&['*', '+', '?', '{'][..]).count();
        let group_count = pattern.matches('(').count();
        let alternation_count = pattern.matches('|').count();
        
        // Calculate complexity score (simple heuristic)
        let complexity_score = character_count
            + (quantifier_count * 5)
            + (group_count * 3)
            + (alternation_count * 4);
        
        PatternComplexity {
            character_count,
            quantifier_count,
            group_count,
            alternation_count,
            complexity_score,
        }
    }
    
    /// Count consecutive wildcard characters
    fn count_consecutive_wildcards(&self, pattern: &str) -> usize {
        let mut max_consecutive = 0;
        let mut current_consecutive = 0;
        
        for ch in pattern.chars() {
            if ch == '*' || ch == '?' {
                current_consecutive += 1;
                max_consecutive = max_consecutive.max(current_consecutive);
            } else {
                current_consecutive = 0;
            }
        }
        
        max_consecutive
    }
    
    /// Validate escape sequences in wildcard patterns
    fn is_valid_escape_sequence(&self, pattern: &str) -> bool {
        let chars: Vec<char> = pattern.chars().collect();
        let mut i = 0;
        
        while i < chars.len() {
            if chars[i] == '\\' {
                if i + 1 >= chars.len() {
                    return false; // Trailing backslash
                }
                
                // Check if next character is valid for escaping
                let escaped_char = chars[i + 1];
                if !matches!(escaped_char, '*' | '?' | '\\' | '[' | ']' | '{' | '}') {
                    return false;
                }
                
                i += 2; // Skip escaped character
            } else {
                i += 1;
            }
        }
        
        true
    }
    
    /// Validate batch of patterns with type detection
    pub fn validate_mixed_patterns(&self, patterns: &[(String, String)]) -> PatternSearchResult<()> {
        if patterns.is_empty() {
            return Err(PatternSearchError::EmptyPattern);
        }
        
        for (pattern, pattern_type) in patterns {
            match pattern_type.as_str() {
                "regex" => self.validate_regex_pattern(pattern)?,
                "wildcard" => self.validate_wildcard_pattern(pattern)?,
                "fuzzy" => self.validate_fuzzy_pattern(pattern, 2)?, // Default distance of 2
                _ => return Err(PatternSearchError::UnsupportedPatternType {
                    pattern_type: pattern_type.clone(),
                }),
            }
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_regex_pattern_validation() {
        let validator = PatternSearchValidator::default();
        
        // Valid regex patterns
        assert!(validator.validate_regex_pattern(r"hello\w+").is_ok());
        assert!(validator.validate_regex_pattern(r"[a-zA-Z]+").is_ok());
        assert!(validator.validate_regex_pattern(r"\d{3}-\d{2}-\d{4}").is_ok());
        
        // Invalid regex patterns
        assert!(validator.validate_regex_pattern("").is_err());
        assert!(validator.validate_regex_pattern("[").is_err()); // Unclosed bracket
        assert!(validator.validate_regex_pattern("(.+)+").is_err()); // Forbidden pattern
    }
    
    #[test]
    fn test_wildcard_pattern_validation() {
        let validator = PatternSearchValidator::default();
        
        // Valid wildcard patterns
        assert!(validator.validate_wildcard_pattern("*.txt").is_ok());
        assert!(validator.validate_wildcard_pattern("file?.log").is_ok());
        assert!(validator.validate_wildcard_pattern("test*file").is_ok());
        
        // Invalid wildcard patterns
        assert!(validator.validate_wildcard_pattern("").is_err());
        assert!(validator.validate_wildcard_pattern("*").is_err()); // Too broad
        assert!(validator.validate_wildcard_pattern("******").is_err()); // Too many consecutive wildcards
    }
    
    #[test]
    fn test_fuzzy_pattern_validation() {
        let validator = PatternSearchValidator::default();
        
        // Valid fuzzy patterns
        assert!(validator.validate_fuzzy_pattern("hello", 1).is_ok());
        assert!(validator.validate_fuzzy_pattern("programming", 3).is_ok());
        
        // Invalid fuzzy patterns
        assert!(validator.validate_fuzzy_pattern("", 1).is_err());
        assert!(validator.validate_fuzzy_pattern("test", 0).is_err()); // Zero distance
        assert!(validator.validate_fuzzy_pattern("hi", 5).is_err()); // Distance > length
    }
    
    #[test]
    fn test_pattern_complexity_analysis() {
        let validator = PatternSearchValidator::default();
        
        let complexity = validator.analyze_pattern_complexity(r"(hello|world)+\d*");
        assert!(complexity.quantifier_count > 0);
        assert!(complexity.group_count > 0);
        assert!(complexity.alternation_count > 0);
        assert!(complexity.complexity_score > 0);
    }
    
    #[test]
    fn test_mixed_patterns_validation() {
        let validator = PatternSearchValidator::default();
        
        let patterns = vec![
            ("*.txt".to_string(), "wildcard".to_string()),
            (r"\d+".to_string(), "regex".to_string()),
            ("hello".to_string(), "fuzzy".to_string()),
        ];
        
        assert!(validator.validate_mixed_patterns(&patterns).is_ok());
        
        // Invalid pattern type
        let invalid_patterns = vec![
            ("test".to_string(), "invalid_type".to_string()),
        ];
        assert!(validator.validate_mixed_patterns(&invalid_patterns).is_err());
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
- [ ] PatternSearchValidator struct with configurable limits
- [ ] validate_regex_pattern method with safety checks
- [ ] validate_wildcard_pattern method with depth analysis
- [ ] validate_fuzzy_pattern method with distance validation
- [ ] analyze_pattern_complexity method for performance assessment
- [ ] validate_mixed_patterns method for batch processing
- [ ] PatternComplexity struct for detailed analysis
- [ ] Forbidden patterns list for security
- [ ] Unit tests covering all pattern validation scenarios
- [ ] File compiles without errors

## Validation
Run `cargo test` - all pattern validation tests should pass.

## Next Task
This completes the error handling improvement micro-tasks series. The system now has comprehensive error handling with:
- ProximitySearchError enum (14a)
- PatternSearchError enum (14b) 
- Proximity search validation (14c)
- Phrase search validation (14d)
- NEAR query validation (14e)
- Pattern search validation (14f)

All error handling is now defensive, descriptive, and properly integrated.