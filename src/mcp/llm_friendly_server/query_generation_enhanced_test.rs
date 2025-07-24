//! Direct unit tests for enhanced query generation
//! This bypasses the handler to test the core functionality

#[cfg(test)]
mod tests {
    use crate::mcp::llm_friendly_server::query_generation_enhanced::*;

    #[test]
    fn test_extract_entities_about_pattern() {
        let entities = extract_entities_advanced("Find all facts about Einstein");
        assert_eq!(entities, vec!["Einstein"]);
        
        let entities = extract_entities_advanced("Get information about Tesla and Edison");
        assert!(entities.contains(&"Tesla".to_string()));
        assert!(entities.contains(&"Edison".to_string()));
    }

    #[test]
    fn test_extract_entities_between_pattern() {
        let entities = extract_entities_advanced("Show relationships between Einstein and Newton");
        assert_eq!(entities.len(), 2);
        assert!(entities.contains(&"Einstein".to_string()));
        assert!(entities.contains(&"Newton".to_string()));
    }

    #[test]
    fn test_extract_entities_possessive() {
        let entities = extract_entities_advanced("What are Einstein's discoveries?");
        assert!(entities.contains(&"Einstein".to_string()));
    }

    #[test]
    fn test_extract_entities_quoted() {
        let entities = extract_entities_advanced("Find information about \"Theory of Relativity\"");
        assert!(entities.contains(&"Theory of Relativity".to_string()));
    }

    #[test]
    fn test_extract_entities_related_to() {
        let entities = extract_entities_advanced("Find concepts related to quantum mechanics");
        assert!(entities.contains(&"quantum mechanics".to_string()));
    }

    #[test]
    fn test_extract_entities_complex() {
        let entities = extract_entities_advanced("Show connections between Einstein and Tesla related to electricity");
        assert!(entities.contains(&"Einstein".to_string()));
        assert!(entities.contains(&"Tesla".to_string()));
        // Note: "electricity" is lowercase, might not be caught by capitalization rule
    }

    #[test]
    fn test_cypher_generation_simple() {
        let (query, explanation) = generate_cypher_query_enhanced(
            "Find all facts about Einstein", 
            true
        ).unwrap();
        
        assert!(query.contains("Einstein"));
        assert!(query.contains("MATCH"));
        assert!(explanation.is_some());
    }

    #[test]
    fn test_cypher_generation_path() {
        let (query, _) = generate_cypher_query_enhanced(
            "Find the shortest path between Einstein and Tesla",
            false
        ).unwrap();
        
        assert!(query.contains("shortestPath"));
        assert!(query.contains("Einstein"));
        assert!(query.contains("Tesla"));
    }

    #[test]
    fn test_sparql_generation() {
        let (query, _) = generate_sparql_query_enhanced(
            "Find all facts about Einstein",
            false
        ).unwrap();
        
        assert!(query.contains("SELECT"));
        assert!(query.contains("WHERE"));
        assert!(query.contains("Einstein"));
    }

    #[test]
    fn test_gremlin_generation() {
        let (query, _) = generate_gremlin_query_enhanced(
            "Find all facts about Einstein",
            false
        ).unwrap();
        
        assert!(query.contains("g.V()"));
        assert!(query.contains("Einstein"));
    }

    #[test]
    fn test_who_what_questions() {
        let (query, _) = generate_cypher_query_enhanced(
            "Who invented the telephone?",
            false
        ).unwrap();
        
        // Should extract "invented" as predicate
        assert!(query.contains("invented") || query.contains("INVENTED"));
    }

    #[test]
    fn test_count_queries() {
        let (query, _) = generate_cypher_query_enhanced(
            "How many scientists are there?",
            false
        ).unwrap();
        
        assert!(query.contains("COUNT") || query.contains("count"));
    }

    #[test]
    fn test_empty_query_fallback() {
        let entities = extract_entities_advanced("");
        assert_eq!(entities.len(), 0);
        
        let (query, _) = generate_cypher_query_enhanced("", false).unwrap();
        assert!(query.contains("LIMIT")); // Fallback query should have a limit
    }

    #[test]
    fn test_performance() {
        use std::time::Instant;
        
        let start = Instant::now();
        let _ = generate_cypher_query_enhanced(
            "Find all relationships between Einstein, Tesla, and Newton related to physics",
            true
        ).unwrap();
        let duration = start.elapsed();
        
        // Should complete in under 10ms
        assert!(duration.as_millis() < 10, "Query generation took {}ms", duration.as_millis());
    }
}