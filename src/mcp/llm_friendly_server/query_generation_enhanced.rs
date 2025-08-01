//! Enhanced query generation with advanced NLP and pattern matching
//! Advanced NLP entity extraction for query pattern matching

use crate::error::Result;
use regex::Regex;
// HashMap removed - unused import
use lazy_static::lazy_static;

/// Enhanced entity extraction with NLP patterns
pub fn extract_entities_advanced(query: &str) -> Vec<String> {
    let mut entities = Vec::new();
    
    // Pattern 1: "about X", "facts about X", "information about X"
    if let Ok(about_re) = Regex::new(r"(?i)(?:facts?|information|data|details?)\s+(?:about|on|for|regarding)\s+([^,\s]+(?:\s+[^,\s]+)*)") {
        for cap in about_re.captures_iter(query) {
            if let Some(entity) = cap.get(1) {
                entities.push(entity.as_str().trim().to_string());
            }
        }
    }
    
    // Pattern 2: "between X and Y" - improved to stop at keywords
    if let Ok(between_re) = Regex::new(r"(?i)between\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*?)\s+and\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*?)(?:\s+(?:related|connected|about|with|to)|$)") {
        for cap in between_re.captures_iter(query) {
            if let Some(entity1) = cap.get(1) {
                entities.push(entity1.as_str().trim().to_string());
            }
            if let Some(entity2) = cap.get(2) {
                entities.push(entity2.as_str().trim().to_string());
            }
        }
    }
    
    // Also try simpler between pattern for basic cases
    if let Ok(between_simple) = Regex::new(r"(?i)between\s+(\w+)\s+and\s+(\w+)") {
        for cap in between_simple.captures_iter(query) {
            if let Some(entity1) = cap.get(1) {
                let e1 = entity1.as_str();
                if e1.chars().next().unwrap_or('a').is_uppercase() && !entities.contains(&e1.to_string()) {
                    entities.push(e1.to_string());
                }
            }
            if let Some(entity2) = cap.get(2) {
                let e2 = entity2.as_str();
                if e2.chars().next().unwrap_or('a').is_uppercase() && !entities.contains(&e2.to_string()) {
                    entities.push(e2.to_string());
                }
            }
        }
    }
    
    // Pattern 3: "X's Y" (possessive) - improved to handle word boundaries
    if let Ok(poss_re) = Regex::new(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)'s") {
        for cap in poss_re.captures_iter(query) {
            if let Some(entity) = cap.get(1) {
                let entity_str = entity.as_str();
                // Filter out common verbs that might precede the possessive
                let skip_patterns = ["Show", "Find", "Get", "List", "Display"];
                let words: Vec<&str> = entity_str.split_whitespace().collect();
                if words.len() > 1 && skip_patterns.contains(&words[0]) {
                    // Extract only the actual entity name (skip the verb)
                    entities.push(words[1..].join(" "));
                } else {
                    entities.push(entity_str.to_string());
                }
            }
        }
    }
    
    // Pattern 4: "related to X", "connected to X"
    if let Ok(related_re) = Regex::new(r"(?i)(?:related|connected|linked)\s+(?:to|with)\s+([^,\s]+(?:\s+[^,\s]+)*)") {
        for cap in related_re.captures_iter(query) {
            if let Some(entity) = cap.get(1) {
                entities.push(entity.as_str().trim().to_string());
            }
        }
    }
    
    // Pattern 5: Quoted entities
    if let Ok(quote_re) = Regex::new(r#"["']([^"']+)["']"#) {
        for cap in quote_re.captures_iter(query) {
            if let Some(entity) = cap.get(1) {
                entities.push(entity.as_str().to_string());
            }
        }
    }
    
    // Pattern 6: Capitalized words (likely proper nouns) - fallback
    let words: Vec<&str> = query.split_whitespace().collect();
    for word in words {
        let clean_word = word.trim_matches(|c: char| !c.is_alphanumeric());
        if !clean_word.is_empty() && clean_word.chars().next().unwrap().is_uppercase() {
            // Skip common query words
            let skip_words = ["Find", "Show", "Get", "What", "Who", "Where", "When", "How", "List", "Display"];
            if !skip_words.contains(&clean_word) {
                entities.push(clean_word.to_string());
            }
        }
    }
    
    // Remove duplicates while preserving order
    let mut seen = std::collections::HashSet::new();
    entities.retain(|entity| seen.insert(entity.clone()));
    
    entities
}

/// Query template structure
#[derive(Clone)]
struct QueryTemplate {
    name: &'static str,
    pattern: Regex,
    cypher_template: &'static str,
    sparql_template: &'static str,
    gremlin_template: &'static str,
}

lazy_static! {
    static ref QUERY_TEMPLATES: Vec<QueryTemplate> = vec![
        // Template 1: Find all facts about entity
        QueryTemplate {
            name: "find_facts_about_entity",
            pattern: Regex::new(r"(?i)(?:find|show|get|list)\s+(?:all\s+)?(?:facts?|information|data)\s+(?:about|on|for)\s+(.+)").unwrap(),
            cypher_template: "MATCH (n:Entity {name: '$entity1'})-[r]->(m) RETURN n, r, m UNION MATCH (m)-[r]->(n:Entity {name: '$entity1'}) RETURN m, r, n",
            sparql_template: "SELECT ?s ?p ?o WHERE { { ?s ?p ?o . FILTER(STR(?s) = '$entity1') } UNION { ?s ?p ?o . FILTER(STR(?o) = '$entity1') } }",
            gremlin_template: "g.V().has('name', '$entity1').bothE().bothV().path()",
        },
        
        // Template 2: Find relationships between entities
        QueryTemplate {
            name: "find_relationships_between",
            pattern: Regex::new(r"(?i)(?:relationship|connection|link)\s+between\s+([^,\s]+(?:\s+[^,\s]+)*)\s+and\s+([^,\s]+(?:\s+[^,\s]+)*)").unwrap(),
            cypher_template: "MATCH path = (a:Entity {name: '$entity1'})-[*1..3]-(b:Entity {name: '$entity2'}) RETURN path",
            sparql_template: "SELECT ?path WHERE { ?a :name '$entity1' . ?b :name '$entity2' . ?a (:related)* ?b }",
            gremlin_template: "g.V().has('name', '$entity1').repeat(both().simplePath()).until(has('name', '$entity2').or().loops().is(3)).path()",
        },
        
        // Template 3: Who/What questions with specific predicates
        QueryTemplate {
            name: "who_what_questions",
            pattern: Regex::new(r"(?i)(who|what)\s+(invented|created|discovered|wrote|built|designed|founded)\s+(?:the\s+)?(.+?)(?:\?|$)").unwrap(),
            cypher_template: "MATCH (n)-[:$predicate]->(m {name: '$entity1'}) RETURN n",
            sparql_template: "SELECT ?subject WHERE { ?subject :$predicate :$entity1 }",
            gremlin_template: "g.V().has('name', '$entity1').in('$predicate')",
        },
        
        // Template 4: Find entities of type
        QueryTemplate {
            name: "find_entities_of_type",
            pattern: Regex::new(r"(?i)(?:find|show|list)\s+(?:all\s+)?(\w+)s?(?:\s+that\s+are\s+(.+))?").unwrap(),
            cypher_template: "MATCH (n:$entity1) $filter RETURN n",
            sparql_template: "SELECT ?entity WHERE { ?entity a :$entity1 $filter }",
            gremlin_template: "g.V().hasLabel('$entity1')$filter",
        },
        
        // Template 5: Count queries
        QueryTemplate {
            name: "count_queries",
            pattern: Regex::new(r"(?i)(?:count|how\s+many)\s+(.+)").unwrap(),
            cypher_template: "MATCH (n:$entity1) RETURN COUNT(n) as count",
            sparql_template: "SELECT (COUNT(?entity) as ?count) WHERE { ?entity a :$entity1 }",
            gremlin_template: "g.V().hasLabel('$entity1').count()",
        },
        
        // Template 6: Path queries
        QueryTemplate {
            name: "shortest_path",
            pattern: Regex::new(r"(?i)(?:shortest\s+)?path\s+(?:from|between)\s+([^,\s]+(?:\s+[^,\s]+)*)\s+(?:to|and)\s+([^,\s]+(?:\s+[^,\s]+)*)").unwrap(),
            cypher_template: "MATCH path = shortestPath((a:Entity {name: '$entity1'})-[*]-(b:Entity {name: '$entity2'})) RETURN path",
            sparql_template: "SELECT ?path WHERE { ?a :name '$entity1' . ?b :name '$entity2' . ?a (:related)* ?b }",
            gremlin_template: "g.V().has('name', '$entity1').repeat(both().simplePath()).until(has('name', '$entity2')).limit(1).path()",
        },
    ];
}

/// Generate enhanced Cypher query
pub fn generate_cypher_query_enhanced(natural_query: &str, include_explanation: bool) -> Result<(String, Option<String>)> {
    let entities = extract_entities_advanced(natural_query);
    let (query, explanation) = match_query_template(natural_query, &entities, "cypher");
    
    if include_explanation {
        Ok((query, Some(explanation)))
    } else {
        Ok((query, None))
    }
}

/// Generate enhanced SPARQL query
pub fn generate_sparql_query_enhanced(natural_query: &str, include_explanation: bool) -> Result<(String, Option<String>)> {
    let entities = extract_entities_advanced(natural_query);
    let (query, explanation) = match_query_template(natural_query, &entities, "sparql");
    
    if include_explanation {
        Ok((query, Some(explanation)))
    } else {
        Ok((query, None))
    }
}

/// Generate enhanced Gremlin query
pub fn generate_gremlin_query_enhanced(natural_query: &str, include_explanation: bool) -> Result<(String, Option<String>)> {
    let entities = extract_entities_advanced(natural_query);
    let (query, explanation) = match_query_template(natural_query, &entities, "gremlin");
    
    if include_explanation {
        Ok((query, Some(explanation)))
    } else {
        Ok((query, None))
    }
}

/// Match natural query against templates and generate appropriate query
fn match_query_template(natural_query: &str, entities: &[String], query_language: &str) -> (String, String) {
    // Try to match against templates
    for template in QUERY_TEMPLATES.iter() {
        if template.pattern.is_match(natural_query) {
            let mut query = match query_language {
                "cypher" => template.cypher_template.to_string(),
                "sparql" => template.sparql_template.to_string(),
                "gremlin" => template.gremlin_template.to_string(),
                _ => template.cypher_template.to_string(),
            };
            
            // Replace entity placeholders
            for (i, entity) in entities.iter().enumerate() {
                query = query.replace(&format!("$entity{}", i + 1), entity);
            }
            
            // Extract predicate if present
            if let Some(predicate) = extract_predicate(natural_query) {
                query = query.replace("$predicate", &predicate);
            }
            
            // Handle filters
            let filter = extract_filter(natural_query, query_language);
            query = query.replace("$filter", &filter);
            
            let explanation = format!(
                "Using template '{}' to find {} related to: {}",
                template.name,
                if entities.len() > 1 { "relationships" } else { "information" },
                entities.join(", ")
            );
            
            return (query, explanation);
        }
    }
    
    // Fallback: Generate basic query
    generate_fallback_query(entities, query_language)
}

/// Extract predicate from natural language
fn extract_predicate(query: &str) -> Option<String> {
    let predicate_patterns = vec![
        (r"(?i)who\s+(\w+)", 1),
        (r"(?i)what\s+(\w+)", 1),
        (r"(?i)(\w+)ed\s+by", 1), // invented, created, discovered
        (r"(?i)(\w+)s\s+(?:in|at|to)", 1), // works, lives, belongs
    ];
    
    for (pattern_str, group) in predicate_patterns {
        if let Ok(re) = Regex::new(pattern_str) {
            if let Some(cap) = re.captures(query) {
                if let Some(predicate) = cap.get(group) {
                    return Some(predicate.as_str().to_lowercase());
                }
            }
        }
    }
    
    None
}

/// Extract filter conditions
fn extract_filter(query: &str, query_language: &str) -> String {
    let filter_patterns = vec![
        (r"(?i)where\s+(.+)", "WHERE"),
        (r"(?i)that\s+(?:are|is)\s+(.+)", "WHERE"),
        (r"(?i)with\s+(.+)", "WHERE"),
    ];
    
    for (pattern_str, prefix) in filter_patterns {
        if let Ok(re) = Regex::new(pattern_str) {
            if let Some(cap) = re.captures(query) {
                if let Some(condition) = cap.get(1) {
                    return match query_language {
                        "cypher" => format!(" {} {}", prefix, condition.as_str()),
                        "sparql" => format!(" . FILTER({})", condition.as_str()),
                        "gremlin" => format!(".has('{}')", condition.as_str()),
                        _ => String::new(),
                    };
                }
            }
        }
    }
    
    String::new()
}

/// Generate fallback query when no template matches
fn generate_fallback_query(entities: &[String], query_language: &str) -> (String, String) {
    match query_language {
        "cypher" => {
            if entities.is_empty() {
                (
                    "MATCH (n) RETURN n LIMIT 25".to_string(),
                    "Returning first 25 nodes (no specific entities found)".to_string()
                )
            } else if entities.len() == 1 {
                (
                    format!("MATCH (n:Entity {{name: '{}'}})-[r]-(m) RETURN n, r, m", entities[0]),
                    format!("Finding all connections for '{}'", entities[0])
                )
            } else {
                (
                    format!(
                        "MATCH (a:Entity {{name: '{}'}}), (b:Entity {{name: '{}'}}) MATCH path = (a)-[*1..3]-(b) RETURN path",
                        entities[0], entities[1]
                    ),
                    format!("Finding paths between '{}' and '{}'", entities[0], entities[1])
                )
            }
        }
        "sparql" => {
            if entities.is_empty() {
                (
                    "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 25".to_string(),
                    "Returning first 25 triples (no specific entities found)".to_string()
                )
            } else {
                let filters = entities.iter()
                    .map(|e| format!("STR(?s) = '{e}' || STR(?o) = '{e}'"))
                    .collect::<Vec<_>>()
                    .join(" || ");
                (
                    format!("SELECT ?s ?p ?o WHERE {{ ?s ?p ?o . FILTER({filters}) }}"),
                    format!("Finding triples involving: {}", entities.join(", "))
                )
            }
        }
        "gremlin" => {
            if entities.is_empty() {
                (
                    "g.V().limit(25)".to_string(),
                    "Returning first 25 vertices (no specific entities found)".to_string()
                )
            } else if entities.len() == 1 {
                (
                    format!("g.V().has('name', '{}').bothE().bothV().path()", entities[0]),
                    format!("Finding all connections for '{}'", entities[0])
                )
            } else {
                (
                    format!(
                        "g.V().has('name', '{}').repeat(both()).until(has('name', '{}')).path()",
                        entities[0], entities[1]
                    ),
                    format!("Finding paths between '{}' and '{}'", entities[0], entities[1])
                )
            }
        }
        _ => (
            "MATCH (n) RETURN n LIMIT 25".to_string(),
            "Default query (unknown language)".to_string()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_extract_entities_about_pattern() {
        // Test single entity
        let entities = extract_entities_advanced("Find all facts about Einstein");
        assert_eq!(entities, vec!["Einstein"]);
        
        // Test multiple entities in one phrase  
        let entities = extract_entities_advanced("Find all facts about Einstein and Tesla");
        assert_eq!(entities, vec!["Einstein and Tesla"]);
        
        // Test with "information about"
        let entities = extract_entities_advanced("Get information about Tesla");
        assert_eq!(entities, vec!["Tesla"]);
    }
    
    #[test]
    fn test_extract_entities_between_pattern() {
        let entities = extract_entities_advanced("Show relationships between Einstein and Newton");
        assert_eq!(entities.len(), 2);
        assert_eq!(entities[0], "Einstein");
        assert_eq!(entities[1], "Newton");
    }
    
    #[test]
    fn test_extract_entities_possessive() {
        let entities = extract_entities_advanced("What are Einstein's discoveries?");
        assert!(entities.contains(&"Einstein".to_string()));
        
        let entities = extract_entities_advanced("Show Newton's laws");
        assert!(entities.contains(&"Newton".to_string()));
    }
    
    #[test]
    fn test_extract_entities_quoted() {
        let entities = extract_entities_advanced("Find information about \"Theory of Relativity\"");
        assert!(entities.contains(&"Theory of Relativity".to_string()));
        
        let entities = extract_entities_advanced("Search for 'quantum mechanics'");
        assert!(entities.contains(&"quantum mechanics".to_string()));
    }
    
    #[test]
    fn test_extract_entities_related_to() {
        let entities = extract_entities_advanced("Find concepts related to quantum mechanics");
        assert!(entities.contains(&"quantum mechanics".to_string()));
        
        let entities = extract_entities_advanced("Show items connected with Einstein");
        assert!(entities.contains(&"Einstein".to_string()));
    }
    
    #[test]
    fn test_extract_entities_complex() {
        let entities = extract_entities_advanced("Show connections between Einstein and Tesla related to electricity");
        assert!(entities.contains(&"Einstein".to_string()));
        assert!(entities.contains(&"Tesla".to_string()));
        assert!(entities.contains(&"electricity".to_string()));
    }
    
    #[test]
    fn test_extract_entities_capitalized_fallback() {
        // Should extract capitalized words as entities
        let entities = extract_entities_advanced("Find Scientists in Germany");
        assert!(entities.contains(&"Scientists".to_string()));
        assert!(entities.contains(&"Germany".to_string()));
    }
    
    #[test]
    fn test_extract_entities_no_duplicates() {
        let entities = extract_entities_advanced("Einstein studied Einstein's theories about Einstein");
        assert_eq!(entities.iter().filter(|&e| e == "Einstein").count(), 1);
    }
    
    #[test]
    fn test_cypher_query_generation() {
        let (query, _) = generate_cypher_query_enhanced(
            "Find all facts about Einstein",
            false
        ).unwrap();
        assert!(query.contains("Einstein"));
        assert!(query.contains("MATCH"));
        
        let (query, _) = generate_cypher_query_enhanced(
            "Show the shortest path between Einstein and Tesla",
            false
        ).unwrap();
        assert!(query.contains("shortestPath"));
        assert!(query.contains("Einstein"));
        assert!(query.contains("Tesla"));
    }
    
    #[test]
    fn test_sparql_query_generation() {
        let (query, _) = generate_sparql_query_enhanced(
            "Find all facts about Einstein",
            false
        ).unwrap();
        assert!(query.contains("SELECT"));
        assert!(query.contains("WHERE"));
        assert!(query.contains("Einstein"));
    }
    
    #[test]
    fn test_gremlin_query_generation() {
        let (query, _) = generate_gremlin_query_enhanced(
            "Find all facts about Einstein",
            false
        ).unwrap();
        assert!(query.contains("g.V()"));
        assert!(query.contains("Einstein"));
    }
    
    #[test]
    fn test_count_query_generation() {
        let (query, _) = generate_cypher_query_enhanced(
            "How many scientists are there?",
            false
        ).unwrap();
        assert!(query.contains("COUNT") || query.contains("count"));
    }
    
    #[test]
    fn test_who_what_query_generation() {
        let (query, _) = generate_cypher_query_enhanced(
            "Who invented the telephone?",
            false
        ).unwrap();
        assert!(query.contains("invented") || query.contains("INVENTED"));
    }
    
    #[test]
    fn test_template_matching() {
        // Test facts template
        let (query, explanation) = match_query_template(
            "find all facts about Einstein",
            &["Einstein".to_string()],
            "cypher"
        );
        assert!(query.contains("Einstein"));
        assert!(explanation.contains("find_facts_about_entity"));
        
        // Test relationship template
        let (query, _explanation) = match_query_template(
            "show the connection between Einstein and Newton",
            &["Einstein".to_string(), "Newton".to_string()],
            "cypher"
        );
        assert!(query.contains("Einstein"));
        assert!(query.contains("Newton"));
        assert!(query.contains("path"));
    }
    
    #[test]
    fn test_fallback_query_empty_entities() {
        let (query, explanation) = generate_fallback_query(&[], "cypher");
        assert!(query.contains("LIMIT"));
        assert!(explanation.contains("no specific entities"));
    }
    
    #[test]
    fn test_fallback_query_single_entity() {
        let (query, explanation) = generate_fallback_query(&["Einstein".to_string()], "cypher");
        assert!(query.contains("Einstein"));
        assert!(query.contains("MATCH"));
        assert!(explanation.contains("connections"));
    }
    
    #[test]
    fn test_fallback_query_multiple_entities() {
        let (query, explanation) = generate_fallback_query(
            &["Einstein".to_string(), "Tesla".to_string()], 
            "cypher"
        );
        assert!(query.contains("Einstein"));
        assert!(query.contains("Tesla"));
        assert!(query.contains("path"));
        assert!(explanation.contains("paths between"));
    }
    
    #[test] 
    fn test_explanation_included() {
        let (_, explanation) = generate_cypher_query_enhanced(
            "Find all facts about Einstein",
            true
        ).unwrap();
        assert!(explanation.is_some());
        assert!(!explanation.unwrap().is_empty());
    }
    
    #[test]
    fn test_explanation_excluded() {
        let (_, explanation) = generate_cypher_query_enhanced(
            "Find all facts about Einstein",
            false
        ).unwrap();
        assert!(explanation.is_none());
    }
}