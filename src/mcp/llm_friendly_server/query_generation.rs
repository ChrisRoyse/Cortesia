//! Query generation for various graph query languages

use crate::error::Result;
use regex::Regex;
use std::collections::HashMap;

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
        QueryIntent::TemporalQuery { entity, predicate: _, temporal_constraint } => {
            let mut query = "g.V()".to_string();
            
            if let Some(ent) = entity {
                query.push_str(&format!(".has('name', '{}')", ent));
            }
            
            // Add temporal constraint
            match temporal_constraint {
                TemporalConstraint::Before(year) => {
                    query.push_str(&format!(".has('year', lt({}))", year));
                }
                TemporalConstraint::After(year) => {
                    query.push_str(&format!(".has('year', gt({}))", year));
                }
                TemporalConstraint::During(year) => {
                    query.push_str(&format!(".has('year', {})", year));
                }
                TemporalConstraint::Between(start_year, end_year) => {
                    query.push_str(&format!(".has('year', between({}, {}))", start_year, end_year));
                }
            }
            
            let explanation = match temporal_constraint {
                TemporalConstraint::Before(year) => format!("Finds vertices before {}", year),
                TemporalConstraint::After(year) => format!("Finds vertices after {}", year),
                TemporalConstraint::During(year) => format!("Finds vertices during {}", year),
                TemporalConstraint::Between(start_year, end_year) => format!("Finds vertices between {} and {}", start_year, end_year),
            };
            
            (query, Some(explanation))
        }
        QueryIntent::MultiEntityQuery { entities, relationship_type } => {
            let mut query = "g.V()".to_string();
            
            // Create a union query for multiple entities
            let entity_conditions: Vec<String> = entities.iter()
                .map(|e| format!("has('name', '{}')", e))
                .collect();
            
            if !entity_conditions.is_empty() {
                query.push_str(&format!(".or({})", entity_conditions.join(", ")));
            }
            
            if let Some(ref rel_type) = relationship_type {
                query.push_str(&format!(".out('{}')", rel_type));
            } else {
                query.push_str(".out()");
            }
            
            query.push_str(".path()");
            
            let explanation = format!("Finds {} for entities: {}", 
                relationship_type.as_ref().unwrap_or(&"relationships".to_string()),
                entities.join(", "));
            
            (query, Some(explanation))
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

/// Extract entities from natural language query with improved patterns
pub fn extract_entities_from_query(query: &str) -> Vec<String> {
    let mut entities = Vec::new();
    
    // Pattern 1: Proper nouns with correct capitalization
    if let Ok(proper_noun_re) = Regex::new(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b") {
        for mat in proper_noun_re.find_iter(query) {
            let entity = mat.as_str().to_string();
            if !entities.contains(&entity) {
                entities.push(entity);
            }
        }
    }
    
    // Pattern 2: Quoted strings
    if let Ok(quoted_re) = Regex::new(r#"["']([^"']+)["']"#) {
        for cap in quoted_re.captures_iter(query) {
            if let Some(entity) = cap.get(1) {
                let entity_str = entity.as_str().to_string();
                if !entities.contains(&entity_str) {
                    entities.push(entity_str);
                }
            }
        }
    }
    
    // Pattern 3: Common entity patterns like "Einstein's theory", "Newton's laws"
    if let Ok(possessive_re) = Regex::new(r"\b([A-Z][a-z]+)'s\s+([a-z]+(?:\s+[a-z]+)*)\b") {
        for cap in possessive_re.captures_iter(query) {
            if let Some(entity) = cap.get(1) {
                let entity_str = entity.as_str().to_string();
                if !entities.contains(&entity_str) {
                    entities.push(entity_str);
                }
            }
        }
    }
    
    // Pattern 4: Handle "both X and Y" patterns
    if let Ok(both_re) = Regex::new(r"\b([A-Z][a-z]+)\s+and\s+([A-Z][a-z]+)\s+both\b") {
        for cap in both_re.captures_iter(query) {
            if let Some(entity1) = cap.get(1) {
                let entity1_str = entity1.as_str().to_string();
                if !entities.contains(&entity1_str) {
                    entities.push(entity1_str);
                }
            }
            if let Some(entity2) = cap.get(2) {
                let entity2_str = entity2.as_str().to_string();
                if !entities.contains(&entity2_str) {
                    entities.push(entity2_str);
                }
            }
        }
    }
    
    // Pattern 5: Scientific terms and concepts
    if let Ok(concept_re) = Regex::new(r"\b(theory|law|principle|equation|theorem)\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b") {
        for cap in concept_re.captures_iter(query) {
            if let Some(entity) = cap.get(2) {
                let entity_str = entity.as_str().to_string();
                if !entities.contains(&entity_str) {
                    entities.push(entity_str);
                }
            }
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

/// Query intent extraction with temporal and complex pattern support
#[derive(Debug)]
enum QueryIntent {
    FindEntity(String),
    FindRelationship(Option<String>, Option<String>, Option<String>),
    FindPath(String, String),
    TemporalQuery {
        entity: Option<String>,
        predicate: Option<String>,
        temporal_constraint: TemporalConstraint,
    },
    MultiEntityQuery {
        entities: Vec<String>,
        relationship_type: Option<String>,
    },
    AggregateQuery,
    Unknown,
}

#[derive(Debug)]
enum TemporalConstraint {
    Before(i32),
    After(i32),
    During(i32),
    Between(i32, i32),
}

/// Extract the intent from a natural language query with enhanced patterns
fn extract_query_intent(query: &str) -> QueryIntent {
    let lower = query.to_lowercase();
    let entities = extract_entities_from_query(query);
    
    // Temporal queries - handle "before", "after", "during" patterns
    if let Some(temporal_constraint) = extract_temporal_constraint(&lower) {
        let predicate = extract_predicate(&lower);
        let entity = if !entities.is_empty() { Some(entities[0].clone()) } else { None };
        
        return QueryIntent::TemporalQuery {
            entity,
            predicate,
            temporal_constraint,
        };
    }
    
    // Multi-entity queries - "What did Einstein and Newton both discover?"
    if let Ok(both_re) = Regex::new(r"(?i)([A-Z][a-z]+)\s+and\s+([A-Z][a-z]+)\s+both") {
        if both_re.is_match(query) && entities.len() >= 2 {
            let relationship_type = if lower.contains("discover") {
                Some("discovered".to_string())
            } else if lower.contains("invent") {
                Some("invented".to_string())
            } else if lower.contains("create") {
                Some("created".to_string())
            } else {
                None
            };
            
            return QueryIntent::MultiEntityQuery {
                entities: entities.clone(),
                relationship_type,
            };
        }
    }
    
    // Path queries with enhanced patterns
    if (lower.contains("path") || lower.contains("connection") || lower.contains("route") ||
        lower.contains("how are") && lower.contains("connected")) &&
       (lower.contains("between") || lower.contains("from") && lower.contains("to")) {
        if entities.len() >= 2 {
            return QueryIntent::FindPath(entities[0].clone(), entities[1].clone());
        }
    }
    
    // Enhanced relationship queries with better patterns
    if lower.contains("relationship") || lower.contains("related") || 
       lower.contains("connected") || 
       (lower.contains("who") && (lower.contains("what") || lower.contains("invented") || lower.contains("created"))) ||
       (lower.contains("what") && (lower.contains("did") || lower.contains("invented") || lower.contains("discovered"))) {
        
        let predicate = extract_predicate(&lower);
        
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
    
    // Enhanced entity queries with case preservation
    if lower.contains("find") || lower.contains("show") || lower.contains("get") ||
       lower.contains("tell me about") || lower.contains("information about") {
        if !entities.is_empty() {
            return QueryIntent::FindEntity(entities[0].clone());
        }
    }
    
    // Aggregate queries
    if lower.contains("count") || lower.contains("average") || 
       lower.contains("sum") || lower.contains("statistics") ||
       lower.contains("how many") {
        return QueryIntent::AggregateQuery;
    }
    
    QueryIntent::Unknown
}

/// Extract temporal constraints from query
fn extract_temporal_constraint(query: &str) -> Option<TemporalConstraint> {
    // Pattern for "before YYYY", "after YYYY", "during YYYY"
    if let Ok(temporal_re) = Regex::new(r"(?i)(before|after|during)\s+(\d{4})") {
        if let Some(cap) = temporal_re.captures(query) {
            if let (Some(time_word), Some(year_str)) = (cap.get(1), cap.get(2)) {
                if let Ok(year) = year_str.as_str().parse::<i32>() {
                    match time_word.as_str().to_lowercase().as_str() {
                        "before" => return Some(TemporalConstraint::Before(year)),
                        "after" => return Some(TemporalConstraint::After(year)),
                        "during" => return Some(TemporalConstraint::During(year)),
                        _ => {}
                    }
                }
            }
        }
    }
    
    // Pattern for "between YYYY and YYYY"
    if let Ok(between_re) = Regex::new(r"(?i)between\s+(\d{4})\s+and\s+(\d{4})") {
        if let Some(cap) = between_re.captures(query) {
            if let (Some(year1_str), Some(year2_str)) = (cap.get(1), cap.get(2)) {
                if let (Ok(year1), Ok(year2)) = (year1_str.as_str().parse::<i32>(), year2_str.as_str().parse::<i32>()) {
                    return Some(TemporalConstraint::Between(year1.min(year2), year1.max(year2)));
                }
            }
        }
    }
    
    None
}

/// Advanced entity extraction with context awareness
pub fn extract_entities_with_context(query: &str) -> HashMap<String, String> {
    let mut entities_with_context = HashMap::new();
    
    // Pattern for "X's Y" - possessive relationships
    if let Ok(possessive_re) = Regex::new(r"(?i)\b([A-Z][a-z]+)'s\s+([a-z]+(?:\s+[a-z]+)*)\b") {
        for cap in possessive_re.captures_iter(query) {
            if let (Some(entity), Some(context)) = (cap.get(1), cap.get(2)) {
                entities_with_context.insert(
                    entity.as_str().to_string(),
                    format!("possessive: {}", context.as_str())
                );
            }
        }
    }
    
    // Pattern for "X who/that Y" - descriptive clauses
    if let Ok(descriptive_re) = Regex::new(r"(?i)\b([A-Z][a-z]+)\s+(?:who|that)\s+([^.?!]+)") {
        for cap in descriptive_re.captures_iter(query) {
            if let (Some(entity), Some(context)) = (cap.get(1), cap.get(2)) {
                entities_with_context.insert(
                    entity.as_str().to_string(),
                    format!("description: {}", context.as_str().trim())
                );
            }
        }
    }
    
    entities_with_context
}

/// Generate query suggestions based on extracted entities
pub fn suggest_related_queries(entities: &[String]) -> Vec<String> {
    let mut suggestions = Vec::new();
    
    for entity in entities {
        suggestions.push(format!("What did {} discover?", entity));
        suggestions.push(format!("Who influenced {}?", entity));
        suggestions.push(format!("What is {} known for?", entity));
        suggestions.push(format!("When did {} live?", entity));
    }
    
    // Cross-entity suggestions
    if entities.len() >= 2 {
        for i in 0..entities.len() {
            for j in i+1..entities.len() {
                suggestions.push(format!("How are {} and {} related?", entities[i], entities[j]));
                suggestions.push(format!("What did {} and {} both work on?", entities[i], entities[j]));
            }
        }
    }
    
    suggestions.into_iter().take(10).collect() // Limit to 10 suggestions
}

/// Validate and improve query patterns
pub fn validate_and_improve_query(query: &str) -> Result<String> {
    let mut improved_query = query.to_string();
    
    // Fix common capitalization issues
    if let Ok(name_re) = Regex::new(r"\b(einstein|newton|darwin|tesla|galileo|kepler|copernicus)\b") {
        improved_query = name_re.replace_all(&improved_query, |caps: &regex::Captures| {
            let name = &caps[1];
            format!("{}{}", name.chars().next().unwrap().to_uppercase(), &name[1..])
        }).to_string();
    }
    
    // Expand common abbreviations
    let abbreviations = vec![
        ("AI", "Artificial Intelligence"),
        ("ML", "Machine Learning"),
        ("NLP", "Natural Language Processing"),
        ("DNA", "Deoxyribonucleic Acid"),
        ("RNA", "Ribonucleic Acid"),
    ];
    
    for (abbrev, full) in abbreviations {
        if improved_query.contains(abbrev) && !improved_query.contains(full) {
            // Only suggest, don't auto-replace
            improved_query = format!("{} (Note: {} could mean {})", improved_query, abbrev, full);
        }
    }
    
    Ok(improved_query)
}

/// Extract predicate/relationship type from query
fn extract_predicate(query: &str) -> Option<String> {
    let predicates = vec![
        ("invented", "invented"),
        ("discovered", "discovered"),
        ("created", "created"),
        ("wrote", "wrote"),
        ("founded", "founded"),
        ("developed", "developed"),
        ("studied", "studied"),
        ("taught", "taught"),
        ("influenced", "influenced"),
        ("collaborated", "collaborated_with"),
        ("worked", "worked_with"),
        ("born", "born_in"),
        ("died", "died_in"),
        ("lived", "lived_in"),
    ];
    
    for (pattern, predicate) in predicates {
        if query.contains(pattern) {
            return Some(predicate.to_string());
        }
    }
    
    None
}
