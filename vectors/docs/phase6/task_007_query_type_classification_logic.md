# Task 007: Add Query Type Classification Logic

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This builds on Tasks 001-006. The QueryType enum needs intelligent classification logic to automatically detect query types from query strings, which is critical for automatic test case generation and validation.

## Project Structure
```
src/
  validation/
    ground_truth.rs  <- Extend this file
  lib.rs
Cargo.toml
```

## Task Description
Implement sophisticated query type classification logic that can analyze query strings and automatically determine the appropriate QueryType. This system needs to handle complex queries with multiple features and provide confidence scoring for ambiguous cases.

## Requirements
1. Add to existing `src/validation/ground_truth.rs`
2. Enhance `QueryType::from_query()` with comprehensive classification logic
3. Add confidence scoring for classification decisions
4. Handle multi-feature queries (e.g., boolean with wildcards)
5. Create classification rules for each query type
6. Add fuzzy matching for query pattern recognition
7. Implement priority rules for conflicting classifications

## Expected Code Structure to Add
```rust
use regex::Regex;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct QueryClassification {
    pub query_type: QueryType,
    pub confidence: f64,
    pub detected_features: Vec<QueryFeature>,
    pub reasoning: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum QueryFeature {
    BooleanAndOperator,
    BooleanOrOperator,
    BooleanNotOperator,
    ProximityOperator,
    WildcardChars,
    RegexPattern,
    QuotedPhrase,
    SpecialChars,
    VectorSimilarity,
    HybridSearch,
}

impl QueryType {
    pub fn from_query(query: &str) -> Self {
        Self::classify_query(query).query_type
    }
    
    pub fn classify_query(query: &str) -> QueryClassification {
        let mut classifier = QueryClassifier::new();
        classifier.classify(query)
    }
    
    pub fn classify_with_context(query: &str, context: &QueryContext) -> QueryClassification {
        let mut classifier = QueryClassifier::new();
        classifier.with_context(context);
        classifier.classify(query)
    }
}

#[derive(Debug, Clone)]
pub struct QueryContext {
    pub has_vector_index: bool,
    pub supports_proximity: bool,
    pub supports_regex: bool,
    pub default_search_mode: SearchMode,
}

#[derive(Debug, Clone)]
pub enum SearchMode {
    TextOnly,
    VectorOnly,
    Hybrid,
}

impl Default for QueryContext {
    fn default() -> Self {
        Self {
            has_vector_index: true,
            supports_proximity: true,
            supports_regex: true,
            default_search_mode: SearchMode::Hybrid,
        }
    }
}

pub struct QueryClassifier {
    context: QueryContext,
    patterns: HashMap<QueryFeature, Regex>,
}

impl QueryClassifier {
    pub fn new() -> Self {
        let mut patterns = HashMap::new();
        
        // Compile regex patterns for feature detection
        patterns.insert(
            QueryFeature::BooleanAndOperator,
            Regex::new(r"(?i)\b(AND|&&)\b").unwrap()
        );
        patterns.insert(
            QueryFeature::BooleanOrOperator,
            Regex::new(r"(?i)\b(OR|\|\|)\b").unwrap()
        );
        patterns.insert(
            QueryFeature::BooleanNotOperator,
            Regex::new(r"(?i)(\bNOT\b|^!|[^=]!)").unwrap()
        );
        patterns.insert(
            QueryFeature::ProximityOperator,
            Regex::new(r"(?i)\b(NEAR|WITHIN)\s*\d+\b").unwrap()
        );
        patterns.insert(
            QueryFeature::WildcardChars,
            Regex::new(r"[*?]").unwrap()
        );
        patterns.insert(
            QueryFeature::RegexPattern,
            Regex::new(r"[\[\]{}()|^$\\.]").unwrap()
        );
        patterns.insert(
            QueryFeature::QuotedPhrase,
            Regex::new(r#""[^"]+""#).unwrap()
        );
        patterns.insert(
            QueryFeature::SpecialChars,
            Regex::new(r"[<>@#%&+=]").unwrap()
        );
        patterns.insert(
            QueryFeature::VectorSimilarity,
            Regex::new(r"(?i)\b(similar|semantic|embedding|vector)\b").unwrap()
        );
        patterns.insert(
            QueryFeature::HybridSearch,
            Regex::new(r"(?i)\b(hybrid|combined|text\+vector)\b").unwrap()
        );
        
        Self {
            context: QueryContext::default(),
            patterns,
        }
    }
    
    pub fn with_context(&mut self, context: &QueryContext) {
        self.context = context.clone();
    }
    
    pub fn classify(&self, query: &str) -> QueryClassification {
        let features = self.detect_features(query);
        let (query_type, confidence, reasoning) = self.determine_type(&features, query);
        
        QueryClassification {
            query_type,
            confidence,
            detected_features: features,
            reasoning,
        }
    }
    
    fn detect_features(&self, query: &str) -> Vec<QueryFeature> {
        let mut features = Vec::new();
        
        for (feature, pattern) in &self.patterns {
            if pattern.is_match(query) {
                features.push(feature.clone());
            }
        }
        
        features
    }
    
    fn determine_type(&self, features: &[QueryFeature], query: &str) -> (QueryType, f64, String) {
        let mut scores = HashMap::new();
        let mut reasoning_parts = Vec::new();
        
        // Score each query type based on detected features
        for feature in features {
            match feature {
                QueryFeature::BooleanAndOperator => {
                    *scores.entry(QueryType::BooleanAnd).or_insert(0.0) += 0.9;
                    reasoning_parts.push("Boolean AND operator detected".to_string());
                },
                QueryFeature::BooleanOrOperator => {
                    *scores.entry(QueryType::BooleanOr).or_insert(0.0) += 0.9;
                    reasoning_parts.push("Boolean OR operator detected".to_string());
                },
                QueryFeature::BooleanNotOperator => {
                    *scores.entry(QueryType::BooleanNot).or_insert(0.0) += 0.9;
                    reasoning_parts.push("Boolean NOT operator detected".to_string());
                },
                QueryFeature::ProximityOperator => {
                    *scores.entry(QueryType::Proximity).or_insert(0.0) += 0.95;
                    reasoning_parts.push("Proximity operator detected".to_string());
                },
                QueryFeature::WildcardChars => {
                    *scores.entry(QueryType::Wildcard).or_insert(0.0) += 0.8;
                    reasoning_parts.push("Wildcard characters detected".to_string());
                },
                QueryFeature::RegexPattern => {
                    *scores.entry(QueryType::Regex).or_insert(0.0) += 0.7;
                    reasoning_parts.push("Regex pattern characters detected".to_string());
                },
                QueryFeature::QuotedPhrase => {
                    *scores.entry(QueryType::Phrase).or_insert(0.0) += 0.85;
                    reasoning_parts.push("Quoted phrase detected".to_string());
                },
                QueryFeature::SpecialChars => {
                    *scores.entry(QueryType::SpecialCharacters).or_insert(0.0) += 0.6;
                    reasoning_parts.push("Special characters detected".to_string());
                },
                QueryFeature::VectorSimilarity => {
                    *scores.entry(QueryType::Vector).or_insert(0.0) += 0.9;
                    reasoning_parts.push("Vector similarity keywords detected".to_string());
                },
                QueryFeature::HybridSearch => {
                    *scores.entry(QueryType::Hybrid).or_insert(0.0) += 0.95;
                    reasoning_parts.push("Hybrid search keywords detected".to_string());
                },
            }
        }
        
        // Apply context-based adjustments
        if self.context.has_vector_index {
            if query.len() > 50 && !features.iter().any(|f| matches!(f, QueryFeature::BooleanAndOperator | QueryFeature::BooleanOrOperator)) {
                *scores.entry(QueryType::Vector).or_insert(0.0) += 0.3;
                reasoning_parts.push("Long query without boolean operators suggests vector similarity".to_string());
            }
        }
        
        // Handle multi-feature queries with priority rules
        let (best_type, confidence) = if scores.is_empty() {
            // No specific features detected, classify as special characters if has special chars, otherwise phrase
            if self.has_complex_special_chars(query) {
                (QueryType::SpecialCharacters, 0.7)
            } else {
                (QueryType::Phrase, 0.5)
            }
        } else if scores.len() == 1 {
            // Single clear classification
            let (query_type, score) = scores.into_iter().next().unwrap();
            (query_type, score)
        } else {
            // Multiple features detected, apply priority rules
            self.resolve_multi_feature_conflict(&scores)
        };
        
        let reasoning = if reasoning_parts.is_empty() {
            "No specific features detected, using default classification".to_string()
        } else {
            reasoning_parts.join("; ")
        };
        
        (best_type, confidence, reasoning)
    }
    
    fn resolve_multi_feature_conflict(&self, scores: &HashMap<QueryType, f64>) -> (QueryType, f64) {
        // Priority order for conflicting classifications
        let priority_order = vec![
            QueryType::Hybrid,
            QueryType::Proximity,
            QueryType::BooleanAnd,
            QueryType::BooleanOr,
            QueryType::BooleanNot,
            QueryType::Vector,
            QueryType::Phrase,
            QueryType::Wildcard,
            QueryType::Regex,
            QueryType::SpecialCharacters,
        ];
        
        // Find the highest priority type that has a decent score
        for query_type in priority_order {
            if let Some(&score) = scores.get(&query_type) {
                if score >= 0.6 {
                    return (query_type, score * 0.9); // Slight confidence penalty for conflicts
                }
            }
        }
        
        // Fallback to highest scoring type
        scores.iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(query_type, score)| (query_type.clone(), *score * 0.8))
            .unwrap_or((QueryType::SpecialCharacters, 0.5))
    }
    
    fn has_complex_special_chars(&self, query: &str) -> bool {
        // Check for programming-specific special characters
        let special_patterns = [
            r"\[.*\]",          // Brackets
            r"<.*>",            // Angle brackets
            r"\{.*\}",          // Curly braces
            r"#\w+",            // Hash tags
            r"@\w+",            // At symbols
            r"&\w+",            // Ampersand
            r"%\w+",            // Percent
            r"\+=",             // Compound operators
        ];
        
        special_patterns.iter().any(|pattern| {
            Regex::new(pattern).unwrap().is_match(query)
        })
    }
}

impl QueryClassification {
    pub fn is_confident(&self) -> bool {
        self.confidence >= 0.8
    }
    
    pub fn needs_manual_review(&self) -> bool {
        self.confidence < 0.6
    }
    
    pub fn has_feature(&self, feature: QueryFeature) -> bool {
        self.detected_features.contains(&feature)
    }
    
    pub fn feature_summary(&self) -> String {
        if self.detected_features.is_empty() {
            "No specific features detected".to_string()
        } else {
            let feature_names: Vec<String> = self.detected_features
                .iter()
                .map(|f| format!("{:?}", f))
                .collect();
            feature_names.join(", ")
        }
    }
}

#[cfg(test)]
mod classification_tests {
    use super::*;
    
    #[test]
    fn test_boolean_and_classification() {
        let classification = QueryType::classify_query("rust AND vector");
        assert_eq!(classification.query_type, QueryType::BooleanAnd);
        assert!(classification.is_confident());
        assert!(classification.has_feature(QueryFeature::BooleanAndOperator));
    }
    
    #[test]
    fn test_wildcard_classification() {
        let classification = QueryType::classify_query("*.rs files");
        assert_eq!(classification.query_type, QueryType::Wildcard);
        assert!(classification.has_feature(QueryFeature::WildcardChars));
    }
    
    #[test]
    fn test_special_characters_classification() {
        let classification = QueryType::classify_query("[workspace]");
        assert_eq!(classification.query_type, QueryType::SpecialCharacters);
        assert!(classification.has_feature(QueryFeature::SpecialChars));
    }
    
    #[test]
    fn test_multi_feature_priority() {
        // Query with both wildcard and boolean features
        let classification = QueryType::classify_query("*.rs AND test");
        // Boolean should take priority over wildcard
        assert_eq!(classification.query_type, QueryType::BooleanAnd);
    }
    
    #[test]
    fn test_confidence_scoring() {
        let low_confidence = QueryType::classify_query("simple query");
        let high_confidence = QueryType::classify_query("rust AND vector");
        
        assert!(high_confidence.confidence > low_confidence.confidence);
        assert!(high_confidence.is_confident());
    }
}
```

## Dependencies
Add to Cargo.toml:
```toml
[dependencies]
regex = "1.0"
```

## Success Criteria
- Query classification logic compiles without errors
- All QueryType variants have proper classification rules
- Confidence scoring accurately reflects classification certainty
- Multi-feature queries are handled with appropriate priority rules
- Feature detection patterns work for real-world queries
- Context-aware classification improves accuracy
- Unit tests cover all major query patterns and edge cases
- Classification provides human-readable reasoning

## Time Limit
10 minutes maximum