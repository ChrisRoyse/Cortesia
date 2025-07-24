//! Native LLMKG query generation from natural language
//! Converts natural language to native TripleQuery and other LLMKG query types

use crate::error::Result;
// TripleQuery removed - unused import
use regex::Regex;
use serde_json::{json, Value};

/// Generate a native LLMKG query from natural language
pub fn generate_native_query(natural_query: &str) -> Result<Value> {
    let query_lower = natural_query.to_lowercase();
    let entities = extract_entities(natural_query);
    
    // Determine query type and parameters
    if query_lower.contains("all facts about") || query_lower.contains("information about") {
        // Query all triples about a subject
        Ok(json!({
            "query_type": "triple_query",
            "params": {
                "subject": entities.first().cloned(),
                "predicate": None::<String>,
                "object": None::<String>,
                "limit": 100,
                "min_confidence": 0.0,
                "include_chunks": true
            }
        }))
    } else if query_lower.contains("between") && entities.len() >= 2 {
        // Find relationships between two entities
        Ok(json!({
            "query_type": "path_query",
            "params": {
                "start_entity": entities[0].clone(),
                "end_entity": entities[1].clone(),
                "max_depth": 3,
                "include_chunks": true
            }
        }))
    } else if query_lower.contains("who") || query_lower.contains("what") {
        // Extract predicate from question
        let predicate = extract_predicate_from_question(&query_lower);
        let object = entities.first().cloned();
        
        Ok(json!({
            "query_type": "triple_query",
            "params": {
                "subject": None::<String>,
                "predicate": predicate,
                "object": object,
                "limit": 50,
                "min_confidence": 0.0,
                "include_chunks": false
            }
        }))
    } else if query_lower.contains("related to") || query_lower.contains("connected") {
        // Find related entities
        Ok(json!({
            "query_type": "related_entities",
            "params": {
                "entity": entities.first().cloned().unwrap_or_default(),
                "max_depth": 2,
                "limit": 50
            }
        }))
    } else if query_lower.contains("search") || query_lower.contains("find") {
        // General search - could be in chunks or triples
        let search_terms = entities.join(" ");
        Ok(json!({
            "query_type": "hybrid_search",
            "params": {
                "query": search_terms,
                "search_in": ["triples", "chunks", "entities"],
                "limit": 50
            }
        }))
    } else {
        // Default: search for any mention of the entities
        Ok(json!({
            "query_type": "entity_search",
            "params": {
                "entities": entities,
                "include_chunks": true,
                "limit": 100
            }
        }))
    }
}

/// Extract entities from natural language
fn extract_entities(query: &str) -> Vec<String> {
    let mut entities = Vec::new();
    
    // Pattern 1: "about X"
    if let Ok(about_re) = Regex::new(r"(?i)(?:about|regarding|for)\s+([^,\.\?]+)") {
        for cap in about_re.captures_iter(query) {
            if let Some(entity) = cap.get(1) {
                let text = entity.as_str().trim();
                // Clean up common words
                let cleaned = text.trim_start_matches("the ")
                    .trim_start_matches("a ")
                    .trim_start_matches("an ");
                entities.push(cleaned.to_string());
            }
        }
    }
    
    // Pattern 2: "between X and Y"
    if let Ok(between_re) = Regex::new(r"(?i)between\s+([^\s]+(?:\s+[^\s]+)*?)\s+and\s+([^\s]+(?:\s+[^\s]+)*?)(?:\s|$)") {
        for cap in between_re.captures_iter(query) {
            if let Some(e1) = cap.get(1) {
                entities.push(e1.as_str().trim().to_string());
            }
            if let Some(e2) = cap.get(2) {
                entities.push(e2.as_str().trim().to_string());
            }
        }
    }
    
    // Pattern 3: Quoted strings
    if let Ok(quote_re) = Regex::new(r#"["']([^"']+)["']"#) {
        for cap in quote_re.captures_iter(query) {
            if let Some(quoted) = cap.get(1) {
                entities.push(quoted.as_str().to_string());
            }
        }
    }
    
    // Pattern 4: Capitalized words (proper nouns)
    if entities.is_empty() {
        let words: Vec<&str> = query.split_whitespace().collect();
        for word in words {
            let clean = word.trim_matches(|c: char| !c.is_alphanumeric());
            if !clean.is_empty() && clean.chars().next().unwrap().is_uppercase() {
                // Skip common query words
                let skip = ["Find", "Show", "Get", "What", "Who", "Where", "When", "How", "List", "Search"];
                if !skip.contains(&clean) {
                    entities.push(clean.to_string());
                }
            }
        }
    }
    
    // Remove duplicates
    entities.dedup();
    entities
}

/// Extract predicate from who/what questions
fn extract_predicate_from_question(query: &str) -> Option<String> {
    let patterns = vec![
        (r"who\s+(\w+)", 1),
        (r"what\s+(\w+)", 1),
        (r"(\w+)ed\s+by", 1), // "invented by", "created by"
        (r"(\w+)s\s+", 1),    // "works", "lives"
    ];
    
    for (pattern_str, group) in patterns {
        if let Ok(re) = Regex::new(pattern_str) {
            if let Some(cap) = re.captures(query) {
                if let Some(pred) = cap.get(group) {
                    return Some(pred.as_str().to_lowercase());
                }
            }
        }
    }
    
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_facts_about_query() {
        let result = generate_native_query("Find all facts about Einstein").unwrap();
        assert_eq!(result["query_type"], "triple_query");
        assert_eq!(result["params"]["subject"], "Einstein");
        assert!(result["params"]["predicate"].is_null());
    }
    
    #[test]
    fn test_between_query() {
        let result = generate_native_query("Show relationships between Einstein and Newton").unwrap();
        assert_eq!(result["query_type"], "path_query");
        assert_eq!(result["params"]["start_entity"], "Einstein");
        assert_eq!(result["params"]["end_entity"], "Newton");
    }
    
    #[test]
    fn test_who_question() {
        let result = generate_native_query("Who invented the telephone?").unwrap();
        assert_eq!(result["query_type"], "triple_query");
        assert_eq!(result["params"]["predicate"], "invented");
        assert_eq!(result["params"]["object"], "telephone");
    }
    
    #[test]
    fn test_related_query() {
        let result = generate_native_query("Find concepts related to quantum mechanics").unwrap();
        assert_eq!(result["query_type"], "related_entities");
        assert_eq!(result["params"]["entity"], "quantum mechanics");
    }
    
    #[test]
    fn test_search_query() {
        let result = generate_native_query("Search for information about black holes").unwrap();
        assert_eq!(result["query_type"], "hybrid_search");
        assert_eq!(result["params"]["query"], "black holes");
        assert!(result["params"]["search_in"].as_array().unwrap().contains(&json!("chunks")));
    }
    
    #[test]
    fn test_entity_extraction() {
        let entities = extract_entities("Find information about Einstein and Tesla");
        assert_eq!(entities, vec!["Einstein", "Tesla"]);
        
        let entities = extract_entities("What is the connection between New York and London?");
        assert!(entities.contains(&"New York".to_string()));
        assert!(entities.contains(&"London".to_string()));
        
        let entities = extract_entities("Search for \"Theory of Relativity\"");
        assert_eq!(entities, vec!["Theory of Relativity"]);
    }
}