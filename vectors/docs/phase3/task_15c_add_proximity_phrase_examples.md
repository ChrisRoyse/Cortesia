# Task 15c: Add Proximity and Phrase Search Examples

**Estimated Time: 10 minutes**  
**Lines of Code: ~30**
**Prerequisites: Task 15b completion**

## Context
Advanced search capabilities like proximity search and phrase matching are critical for semantic accuracy. This task creates comprehensive examples demonstrating how to use proximity operators, phrase queries, and near-distance searches effectively.

## Your Task
Create detailed examples showing proximity search, phrase matching, and near-distance queries with practical use cases and performance considerations.

## Required Implementation

```rust
//! # Proximity and Phrase Search Examples
//! 
//! Demonstrates advanced search patterns including phrase matching,
//! proximity search, and near-distance queries for precise results.

use vector_search::{
    SearchQuery, 
    ProximityQuery, 
    PhraseQuery,
    NearQuery,
    SearchEngine,
    QueryBuilder
};

/// Phrase search examples for exact phrase matching
pub async fn phrase_search_examples(engine: &SearchEngine) -> Result<(), SearchError> {
    // Exact phrase matching
    let exact_phrase = PhraseQuery::new()
        .with_phrase("machine learning")
        .with_slop(0); // No words allowed between terms
    
    let results = engine.search_phrase(&exact_phrase).await?;
    println!("Exact phrase 'machine learning': {} results", results.len());
    
    // Phrase with slop - allows intervening words
    let loose_phrase = PhraseQuery::new()
        .with_phrase("artificial intelligence")
        .with_slop(2); // Up to 2 words between terms
    
    let results = engine.search_phrase(&loose_phrase).await?;
    println!("Phrase with slop=2: {} results", results.len());
    
    // Multi-term phrase search
    let complex_phrase = PhraseQuery::new()
        .with_phrase("deep neural network architecture")
        .with_slop(1)
        .with_field("content"); // Search specific field
    
    let results = engine.search_phrase(&complex_phrase).await?;
    
    // Display results with phrase highlighting
    for result in results.iter().take(5) {
        if let Some(highlight) = result.get_highlight("content") {
            println!("Match: {}", highlight);
        }
    }
    
    Ok(())
}

/// Proximity search examples using NEAR operators
pub async fn proximity_search_examples(engine: &SearchEngine) -> Result<(), SearchError> {
    // Basic proximity search - terms within 5 words
    let near_query = NearQuery::new()
        .with_terms(vec!["neural", "network"])
        .with_distance(5)
        .with_ordered(false); // Order doesn't matter
    
    let results = engine.search_near(&near_query).await?;
    println!("Terms within 5 words: {} results", results.len());
    
    // Ordered proximity - terms must appear in specified order
    let ordered_near = NearQuery::new()
        .with_terms(vec!["machine", "learning", "algorithm"])
        .with_distance(10)
        .with_ordered(true); // Maintain term order
    
    let results = engine.search_near(&ordered_near).await?;
    println!("Ordered proximity search: {} results", results.len());
    
    // Field-specific proximity search
    let field_near = NearQuery::new()
        .with_terms(vec!["performance", "optimization"])
        .with_distance(3)
        .with_field("title") // Search only in title field
        .with_boost(2.0);    // Boost relevance score
    
    let results = engine.search_near(&field_near).await?;
    
    // Multiple proximity conditions
    let complex_near = ProximityQuery::new()
        .add_near_condition("artificial intelligence", 2)
        .add_near_condition("deep learning", 5)
        .with_combine_mode(CombineMode::And); // Both conditions must match
    
    let results = engine.search_proximity(&complex_near).await?;
    println!("Complex proximity: {} results", results.len());
    
    Ok(())
}

/// Advanced proximity patterns for specific use cases
pub async fn advanced_proximity_patterns(engine: &SearchEngine) -> Result<(), SearchError> {
    // Academic paper search - author near institution
    let academic_search = ProximityQuery::new()
        .add_near_condition("John Smith Stanford University", 15)
        .with_field("authors")
        .with_fuzzy_matching(true); // Handle name variations
    
    let papers = engine.search_proximity(&academic_search).await?;
    
    // Technical documentation - concept near implementation
    let tech_docs = NearQuery::new()
        .with_terms(vec!["OAuth", "implementation"])
        .with_distance(20)
        .with_field("content")
        .with_snippet_size(200); // Larger context snippets
    
    let docs = engine.search_near(&tech_docs).await?;
    
    // Code search - function name near usage pattern
    let code_search = ProximityQuery::new()
        .add_near_condition("async fn process_data", 50)
        .add_near_condition("await", 10)
        .with_field("code")
        .with_language_aware(true); // Understand code structure
    
    let code_results = engine.search_proximity(&code_search).await?;
    
    // Geographic proximity - location near event
    let geo_search = NearQuery::new()
        .with_terms(vec!["conference", "San Francisco"])
        .with_distance(8)
        .with_field("location")
        .with_case_sensitive(false);
    
    let events = engine.search_near(&geo_search).await?;
    
    println!("Advanced patterns executed successfully");
    Ok(())
}

/// Combining phrase and proximity searches
pub async fn combined_search_patterns(engine: &SearchEngine) -> Result<(), SearchError> {
    // Query builder for complex combinations
    let complex_query = QueryBuilder::new()
        .add_phrase("machine learning", 0)           // Exact phrase
        .add_near("algorithm performance", 5)        // Terms within 5 words  
        .add_proximity("optimization", "speed", 3)   // Specific proximity
        .with_minimum_should_match(2)                // At least 2 conditions
        .build();
    
    let results = engine.search(&complex_query).await?;
    
    // Filter results by proximity quality
    let high_quality_results: Vec<_> = results
        .into_iter()
        .filter(|r| r.proximity_score() > 0.7)
        .collect();
    
    println!("High-quality proximity results: {}", high_quality_results.len());
    
    // Phrase search with proximity fallback
    let phrase_or_proximity = QueryBuilder::new()
        .add_phrase("artificial intelligence", 0)    // Preferred: exact phrase
        .add_near("artificial intelligence", 3)      // Fallback: proximity
        .with_boost_phrase(2.0)                      // Prefer exact matches
        .build();
    
    let flexible_results = engine.search(&phrase_or_proximity).await?;
    
    // Display results with proximity explanations
    for result in flexible_results.iter().take(3) {
        println!("Score: {:.3}, Type: {}, Snippet: {}", 
                result.score(),
                result.match_type(),
                result.get_snippet().unwrap_or("N/A"));
    }
    
    Ok(())
}

/// Performance considerations for proximity searches
pub fn proximity_performance_tips() {
    println!("
    Proximity Search Performance Tips:
    
    1. Distance Limits:
       - Keep proximity distance < 20 for best performance
       - Use smaller distances for exact matching needs
       
    2. Field Targeting:
       - Search specific fields when possible
       - Avoid searching all fields simultaneously
       
    3. Term Frequency:
       - Common terms in proximity can be slow
       - Consider term filtering for high-frequency words
       
    4. Ordered vs Unordered:
       - Ordered proximity is faster than unordered
       - Use ordered when term sequence matters
       
    5. Caching Strategy:
       - Cache proximity query results
       - Reuse compiled query objects
    ");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_phrase_search_accuracy() {
        let engine = create_test_engine().await;
        
        // Test exact phrase matching
        let phrase = PhraseQuery::new()
            .with_phrase("test phrase")
            .with_slop(0);
            
        let results = engine.search_phrase(&phrase).await.unwrap();
        
        // Verify all results contain exact phrase
        for result in results {
            let content = result.get_field("content").unwrap();
            assert!(content.contains("test phrase"));
        }
    }
    
    #[tokio::test]
    async fn test_proximity_distance_accuracy() {
        let engine = create_test_engine().await;
        
        let near_query = NearQuery::new()
            .with_terms(vec!["word1", "word2"])
            .with_distance(3);
            
        let results = engine.search_near(&near_query).await.unwrap();
        
        // Verify distance constraints are respected
        for result in results {
            let positions = result.get_term_positions();
            let distance = calculate_term_distance(&positions, "word1", "word2");
            assert!(distance <= 3, "Distance {} exceeds limit", distance);
        }
    }
}
```

## Success Criteria
- [ ] Phrase search examples with slop variations implemented
- [ ] Proximity search using NEAR operators demonstrated
- [ ] Advanced proximity patterns for specific use cases shown
- [ ] Complex query combinations with phrase and proximity
- [ ] Performance considerations documented
- [ ] Test coverage for accuracy validation included

## Validation
Test proximity search functionality:
```bash
cargo test test_phrase_search_accuracy
cargo test test_proximity_distance_accuracy

# Manual validation with sample data
cargo run --example proximity_demo
```

## Next Task
Task 15d will add wildcard and regex pattern examples for flexible text matching capabilities.