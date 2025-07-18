//! Query generation for various graph query languages

use crate::error::Result;

/// Generate a Cypher query from natural language
pub fn generate_cypher_query(natural_query: &str, include_explanation: bool) -> Result<(String, Option<String>)> {
    let (query, explanation) = match extract_query_intent(natural_query) {
        QueryIntent::FindEntity(entity) => {
            (
                format!("MATCH (n:Entity {{name: '{}'}}) RETURN n", entity),
                Some(format!("This finds the entity named '{}'", entity))
            )
        }
        QueryIntent::FindRelationship(subj, pred, obj) => {
            let mut query = "MATCH ".to_string();
            let mut conditions = Vec::new();
            
            if let Some(s) = subj {
                query.push_str("(s:Entity {name: '");
                query.push_str(&s);
                query.push_str("'})");
                conditions.push("s");
            } else {
                query.push_str("(s:Entity)");
            }
            
            query.push_str("-[r");
            if let Some(p) = pred {
                query.push_str(":");
                query.push_str(&p.to_uppercase());
            }
            query.push_str("]->");
            
            if let Some(o) = obj {
                query.push_str("(o:Entity {name: '");
                query.push_str(&o);
                query.push_str("'})");
                conditions.push("o");
            } else {
                query.push_str("(o:Entity)");
            }
            
            query.push_str(" RETURN s, r, o");
            
            (query, Some("This finds relationships matching the pattern".to_string()))
        }
        QueryIntent::FindPath(start, end) => {
            (
                format!(
                    "MATCH path = shortestPath((s:Entity {{name: '{}'}})-[*]-(e:Entity {{name: '{}'}})) RETURN path",
                    start, end
                ),
                Some(format!("This finds the shortest path between '{}' and '{}'", start, end))
            )
        }
        _ => {
            (
                "MATCH (n) RETURN n LIMIT 10".to_string(),
                Some("Generic query returning first 10 nodes".to_string())
            )
        }
    };
    
    if include_explanation {
        Ok((query, explanation))
    } else {
        Ok((query, None))
    }
}

/// Generate a SPARQL query from natural language
pub fn generate_sparql_query(natural_query: &str, include_explanation: bool) -> Result<(String, Option<String>)> {
    let (query, explanation) = match extract_query_intent(natural_query) {
        QueryIntent::FindEntity(entity) => {
            (
                format!(
                    "SELECT ?s ?p ?o WHERE {{ ?s ?p ?o . FILTER(STR(?s) = '{}') }}",
                    entity
                ),
                Some(format!("This finds all triples where '{}' is the subject", entity))
            )
        }
        QueryIntent::FindRelationship(subj, pred, obj) => {
            let mut query = "SELECT ?s ?p ?o WHERE { ".to_string();
            let mut filters = Vec::new();
            
            if let Some(s) = subj {
                filters.push(format!("STR(?s) = '{}'", s));
            }
            if let Some(p) = pred {
                filters.push(format!("STR(?p) = '{}'", p));
            }
            if let Some(o) = obj {
                filters.push(format!("STR(?o) = '{}'", o));
            }
            
            query.push_str("?s ?p ?o . ");
            
            if !filters.is_empty() {
                query.push_str("FILTER(");
                query.push_str(&filters.join(" && "));
                query.push_str(")");
            }
            
            query.push_str(" }");
            
            (query, Some("This finds triples matching the specified pattern".to_string()))
        }
        _ => {
            (
                "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10".to_string(),
                Some("Generic query returning first 10 triples".to_string())
            )
        }
    };
    
    if include_explanation {
        Ok((query, explanation))
    } else {
        Ok((query, None))
    }
}

/// Generate a Gremlin query from natural language
pub fn generate_gremlin_query(natural_query: &str, include_explanation: bool) -> Result<(String, Option<String>)> {
    let (query, explanation) = match extract_query_intent(natural_query) {
        QueryIntent::FindEntity(entity) => {
            (
                format!("g.V().has('name', '{}')", entity),
                Some(format!("This finds vertices with name '{}'", entity))
            )
        }
        QueryIntent::FindRelationship(subj, pred, obj) => {
            let mut query = "g.V()".to_string();
            
            if let Some(s) = subj {
                query.push_str(&format!(".has('name', '{}')", s));
            }
            
            if let Some(p) = pred {
                query.push_str(&format!(".out('{}')", p));
            } else {
                query.push_str(".out()");
            }
            
            if let Some(o) = obj {
                query.push_str(&format!(".has('name', '{}')", o));
            }
            
            query.push_str(".path()");
            
            (query, Some("This traverses the graph following the specified pattern".to_string()))
        }
        QueryIntent::FindPath(start, end) => {
            (
                format!(
                    "g.V().has('name', '{}').repeat(out().simplePath()).until(has('name', '{}')).limit(1).path()",
                    start, end
                ),
                Some(format!("This finds a path from '{}' to '{}'", start, end))
            )
        }
        _ => {
            (
                "g.V().limit(10)".to_string(),
                Some("Generic query returning first 10 vertices".to_string())
            )
        }
    };
    
    if include_explanation {
        Ok((query, explanation))
    } else {
        Ok((query, None))
    }
}

/// Extract entities from natural language query
pub fn extract_entities_from_query(query: &str) -> Vec<String> {
    // Simple entity extraction - in practice would use NER
    let mut entities = Vec::new();
    
    // Look for capitalized words (likely proper nouns)
    for word in query.split_whitespace() {
        if word.chars().next().map_or(false, |c| c.is_uppercase()) {
            entities.push(word.to_string());
        }
    }
    
    // Look for quoted strings
    let mut in_quotes = false;
    let mut current_entity = String::new();
    let mut chars = query.chars();
    
    while let Some(c) = chars.next() {
        if c == '"' || c == '\'' {
            if in_quotes {
                if !current_entity.is_empty() {
                    entities.push(current_entity.clone());
                    current_entity.clear();
                }
            }
            in_quotes = !in_quotes;
        } else if in_quotes {
            current_entity.push(c);
        }
    }
    
    entities
}

/// Estimate query complexity
pub fn estimate_query_complexity(query: &str) -> f32 {
    let mut complexity = 1.0;
    
    // Factors that increase complexity
    if query.contains("path") || query.contains("shortest") {
        complexity *= 2.0;
    }
    
    if query.contains("*") || query.contains("repeat") {
        complexity *= 1.5;
    }
    
    let depth = query.matches("->").count() + query.matches("out(").count();
    complexity *= 1.0 + (depth as f32 * 0.3);
    
    // Factors that limit complexity
    if query.contains("LIMIT") || query.contains("limit") {
        complexity *= 0.8;
    }
    
    complexity.min(10.0)
}

/// Query intent extraction
#[derive(Debug)]
enum QueryIntent {
    FindEntity(String),
    FindRelationship(Option<String>, Option<String>, Option<String>),
    FindPath(String, String),
    AggregateQuery,
    Unknown,
}

/// Extract the intent from a natural language query
fn extract_query_intent(query: &str) -> QueryIntent {
    let lower = query.to_lowercase();
    
    // Path queries
    if (lower.contains("path") || lower.contains("connection") || lower.contains("route")) &&
       lower.contains("between") {
        let entities = extract_entities_from_query(query);
        if entities.len() >= 2 {
            return QueryIntent::FindPath(entities[0].clone(), entities[1].clone());
        }
    }
    
    // Relationship queries
    if lower.contains("relationship") || lower.contains("related") || 
       lower.contains("connected") || lower.contains("who") && 
       (lower.contains("what") || lower.contains("invented") || lower.contains("created")) {
        let entities = extract_entities_from_query(query);
        
        // Try to extract predicate
        let predicate = if lower.contains("invented") {
            Some("invented".to_string())
        } else if lower.contains("created") {
            Some("created".to_string())
        } else if lower.contains("wrote") {
            Some("wrote".to_string())
        } else {
            None
        };
        
        match entities.len() {
            0 => return QueryIntent::FindRelationship(None, predicate, None),
            1 => return QueryIntent::FindRelationship(Some(entities[0].clone()), predicate, None),
            _ => return QueryIntent::FindRelationship(
                Some(entities[0].clone()), 
                predicate, 
                Some(entities[1].clone())
            ),
        }
    }
    
    // Entity queries
    if lower.contains("find") || lower.contains("show") || lower.contains("get") {
        let entities = extract_entities_from_query(query);
        if !entities.is_empty() {
            return QueryIntent::FindEntity(entities[0].clone());
        }
    }
    
    // Aggregate queries
    if lower.contains("count") || lower.contains("average") || 
       lower.contains("sum") || lower.contains("statistics") {
        return QueryIntent::AggregateQuery;
    }
    
    QueryIntent::Unknown
}