# Task 15d: Add Wildcard and Regex Pattern Examples

**Estimated Time: 10 minutes**  
**Lines of Code: ~30**
**Prerequisites: Task 15c completion**

## Context
Flexible pattern matching using wildcards and regular expressions enables powerful search capabilities for partial matches, pattern recognition, and dynamic queries. This task creates comprehensive examples for wildcard searches, regex patterns, and fuzzy matching.

## Your Task
Create detailed examples demonstrating wildcard patterns, regular expressions, fuzzy matching, and pattern-based search queries with performance optimization techniques.

## Required Implementation

```rust
//! # Wildcard and Regex Pattern Examples
//! 
//! Demonstrates flexible pattern matching including wildcards, regular expressions,
//! fuzzy matching, and advanced pattern-based searches.

use vector_search::{
    WildcardQuery,
    RegexQuery, 
    FuzzyQuery,
    PatternQuery,
    SearchEngine,
    PatternBuilder
};
use regex::Regex;

/// Wildcard search examples for flexible matching
pub async fn wildcard_search_examples(engine: &SearchEngine) -> Result<(), SearchError> {
    // Basic wildcard patterns
    let prefix_search = WildcardQuery::new("machin*")  // Matches: machine, machines, machining
        .with_field("content");
    
    let results = engine.search_wildcard(&prefix_search).await?;
    println!("Prefix wildcard 'machin*': {} results", results.len());
    
    // Suffix wildcard
    let suffix_search = WildcardQuery::new("*ing")     // Matches: learning, processing, computing
        .with_field("title")
        .with_boost(1.5);
    
    let results = engine.search_wildcard(&suffix_search).await?;
    println!("Suffix wildcard '*ing': {} results", results.len());
    
    // Middle wildcard matching
    let middle_pattern = WildcardQuery::new("neur*network")  // Matches: neural network, neuro network
        .with_case_insensitive(true);
    
    let results = engine.search_wildcard(&middle_pattern).await?;
    
    // Single character wildcard (?)
    let char_wildcard = WildcardQuery::new("dat?base")    // Matches: database, dataBase
        .with_field("keywords");
    
    let results = engine.search_wildcard(&char_wildcard).await?;
    
    // Complex wildcard combinations
    let complex_wildcard = WildcardQuery::new("*learn*algorithm*")  // Multiple wildcards
        .with_max_expansions(100)     // Limit pattern expansions
        .with_min_term_length(3);     // Avoid short meaningless matches
    
    let results = engine.search_wildcard(&complex_wildcard).await?;
    println!("Complex wildcard pattern: {} results", results.len());
    
    Ok(())
}

/// Regular expression search patterns
pub async fn regex_search_examples(engine: &SearchEngine) -> Result<(), SearchError> {
    // Email pattern matching
    let email_regex = RegexQuery::new(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
        .with_field("contact")
        .with_max_determinized_states(10000); // Performance limit
    
    let emails = engine.search_regex(&email_regex).await?;
    println!("Email pattern matches: {}", emails.len());
    
    // Phone number variations
    let phone_regex = RegexQuery::new(r"(\+1[-.\s]?)?(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})")
        .with_field("phone")
        .with_case_insensitive(true);
    
    let phones = engine.search_regex(&phone_regex).await?;
    
    // Version number matching (semantic versioning)
    let version_regex = RegexQuery::new(r"\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?")
        .with_field("version");
    
    let versions = engine.search_regex(&version_regex).await?;
    
    // Code identifier patterns  
    let function_regex = RegexQuery::new(r"(async\s+)?fn\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(")
        .with_field("code")
        .with_language("rust");
    
    let functions = engine.search_regex(&function_regex).await?;
    
    // URL/URI pattern matching
    let url_regex = RegexQuery::new(r"https?://[^\s/$.?#].[^\s]*")
        .with_field("links")
        .with_extract_matches(true); // Extract actual URLs
    
    let urls = engine.search_regex(&url_regex).await?;
    
    // Display extracted matches
    for result in urls.iter().take(5) {
        if let Some(matches) = result.get_regex_matches() {
            println!("Found URLs: {:?}", matches);
        }
    }
    
    Ok(())
}

/// Fuzzy matching for typo tolerance and similarity
pub async fn fuzzy_search_examples(engine: &SearchEngine) -> Result<(), SearchError> {
    // Basic fuzzy search with edit distance
    let fuzzy_query = FuzzyQuery::new("machien")       // Typo for "machine"
        .with_max_edits(1)                             // Allow 1 character difference
        .with_prefix_length(1);                        // First char must match
    
    let results = engine.search_fuzzy(&fuzzy_query).await?;
    println!("Fuzzy search 'machien': {} results", results.len());
    
    // Multi-edit fuzzy matching
    let fuzzy_multi = FuzzyQuery::new("artifical")     // Typo for "artificial"
        .with_max_edits(2)                             // Allow 2 edits
        .with_min_similarity(0.8);                     // 80% similarity threshold
    
    let results = engine.search_fuzzy(&fuzzy_multi).await?;
    
    // Fuzzy phrase matching
    let fuzzy_phrase = FuzzyQuery::new("machien lerning")  // Multiple typos
        .with_max_edits(1)
        .with_phrase_slop(2)                           // Allow word reordering
        .with_boost_exact_matches(2.0);                // Prefer exact matches
    
    let results = engine.search_fuzzy(&fuzzy_phrase).await?;
    
    // Name fuzzy matching (higher tolerance)
    let name_fuzzy = FuzzyQuery::new("Johnathon")      // Variations of "Jonathan"
        .with_max_edits(2)
        .with_field("author")
        .with_phonetic_matching(true);                 // Soundex-like matching
    
    let authors = engine.search_fuzzy(&name_fuzzy).await?;
    
    // Technical term fuzzy search
    let tech_fuzzy = FuzzyQuery::new("PostgreSQL")
        .with_max_edits(1)
        .with_common_misspellings(&["PostgrSQL", "PostgresQL", "PostgeSQL"])
        .with_case_insensitive(true);
    
    let tech_results = engine.search_fuzzy(&tech_fuzzy).await?;
    
    println!("Fuzzy searches completed successfully");
    Ok(())
}

/// Advanced pattern combinations and optimizations
pub async fn advanced_pattern_examples(engine: &SearchEngine) -> Result<(), SearchError> {
    // Combining different pattern types
    let mixed_pattern = PatternBuilder::new()
        .add_wildcard("neural*")                       // Wildcard component
        .add_regex(r"\d{4}")                          // Year pattern
        .add_fuzzy("network", 1)                      // Fuzzy component
        .with_combination_mode(CombineMode::And)       // All must match
        .build();
    
    let results = engine.search_pattern(&mixed_pattern).await?;
    
    // Conditional pattern matching
    let conditional = PatternBuilder::new()
        .add_condition(|doc| doc.get_field("type") == Some("research"))
        .add_wildcard("AI*")
        .add_proximity("deep learning", 5)
        .build();
    
    let filtered_results = engine.search_pattern(&conditional).await?;
    
    // Performance-optimized patterns
    let optimized = PatternBuilder::new()
        .add_wildcard("optim*")
        .with_early_termination(1000)                  // Stop after 1000 matches
        .with_parallel_execution(true)                 // Use multiple threads
        .with_cache_compiled_patterns(true)            // Cache for reuse
        .build();
    
    let fast_results = engine.search_pattern(&optimized).await?;
    
    // Dynamic pattern generation
    let user_input = "machine learning algorithms";
    let dynamic_pattern = generate_flexible_pattern(user_input);
    let dynamic_results = engine.search_pattern(&dynamic_pattern).await?;
    
    println!("Advanced patterns executed: {} total results", 
            results.len() + filtered_results.len() + fast_results.len());
    
    Ok(())
}

/// Generate flexible search patterns from user input
fn generate_flexible_pattern(input: &str) -> PatternQuery {
    let words: Vec<&str> = input.split_whitespace().collect();
    let mut builder = PatternBuilder::new();
    
    // Add exact phrase as highest priority
    builder = builder.add_phrase(input, 0).with_boost(3.0);
    
    // Add proximity search as fallback
    if words.len() > 1 {
        builder = builder.add_proximity(&words.join(" "), 5).with_boost(2.0);
    }
    
    // Add individual wildcards for broad matching
    for word in words {
        if word.len() > 3 {  // Skip short words
            builder = builder.add_wildcard(&format!("{}*", word)).with_boost(1.0);
        }
    }
    
    // Add fuzzy variants for typo tolerance
    builder = builder.add_fuzzy(input, 1).with_boost(1.5);
    
    builder.with_minimum_should_match(1).build()
}

/// Performance considerations for pattern searches
pub fn pattern_performance_guidelines() {
    println!("
    Pattern Search Performance Guidelines:
    
    1. Wildcard Optimization:
       - Avoid leading wildcards (*term) when possible
       - Use min_term_length to filter short matches
       - Set max_expansions to limit memory usage
       
    2. Regex Performance:
       - Keep patterns simple and specific
       - Use anchors (^$) when appropriate
       - Set max_determinized_states limit
       - Avoid backtracking-heavy patterns
       
    3. Fuzzy Search Tuning:
       - Use prefix_length > 0 for performance
       - Limit max_edits (1-2 typically sufficient)
       - Set minimum similarity thresholds
       
    4. Caching Strategies:
       - Cache compiled regex patterns
       - Reuse fuzzy automata for similar queries
       - Cache wildcard expansion results
       
    5. Memory Management:
       - Monitor regex state space usage
       - Limit concurrent pattern searches
       - Use streaming for large result sets
    ");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_wildcard_accuracy() {
        let engine = create_test_engine().await;
        
        let wildcard = WildcardQuery::new("test*")
            .with_field("content");
            
        let results = engine.search_wildcard(&wildcard).await.unwrap();
        
        // Verify all results match pattern
        for result in results {
            let content = result.get_field("content").unwrap();
            assert!(content.contains("test"), "Result should contain 'test' prefix");
        }
    }
    
    #[tokio::test]
    async fn test_regex_compilation() {
        let regex_query = RegexQuery::new(r"\d{3}-\d{2}-\d{4}");
        
        // Should compile without error
        assert!(regex_query.is_valid());
        
        // Test invalid regex
        let invalid_regex = RegexQuery::new(r"[invalid");
        assert!(!invalid_regex.is_valid());
    }
    
    #[tokio::test]
    async fn test_fuzzy_edit_distance() {
        let engine = create_test_engine().await;
        
        let fuzzy = FuzzyQuery::new("tset")  // 2 edits from "test"
            .with_max_edits(2);
            
        let results = engine.search_fuzzy(&fuzzy).await.unwrap();
        
        // Verify edit distance is respected
        for result in results {
            let term = result.get_matched_term();
            let distance = calculate_edit_distance("tset", &term);
            assert!(distance <= 2, "Edit distance {} exceeds limit", distance);
        }
    }
}
```

## Success Criteria
- [ ] Wildcard pattern examples with various combinations implemented
- [ ] Regular expression patterns for common use cases demonstrated
- [ ] Fuzzy matching with edit distance and similarity controls
- [ ] Advanced pattern combinations and optimizations shown
- [ ] Performance guidelines for pattern searches documented
- [ ] Test coverage for pattern accuracy and compilation

## Validation
Test pattern matching functionality:
```bash
cargo test test_wildcard_accuracy
cargo test test_regex_compilation  
cargo test test_fuzzy_edit_distance

# Performance testing
cargo run --example pattern_benchmarks
```

## Next Task
Task 15e will add Rust-specific code search patterns and examples for developer use cases.