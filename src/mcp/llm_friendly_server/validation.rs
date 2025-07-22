//! Validation logic for knowledge graph integrity

use crate::core::triple::Triple;
use crate::mcp::llm_friendly_server::types::ValidationResult;
use crate::error::Result;
use std::collections::HashMap;

/// Validate a triple for consistency and correctness
pub async fn validate_triple(triple: &Triple) -> Result<ValidationResult> {
    let mut conflicts = Vec::new();
    let mut validation_notes = Vec::new();
    let mut confidence = 1.0;
    
    // Check for empty fields
    if triple.subject.is_empty() || triple.predicate.is_empty() || triple.object.is_empty() {
        conflicts.push("Triple contains empty fields".to_string());
        confidence *= 0.5;
    }
    
    // Check field lengths
    if triple.subject.len() > 128 || triple.predicate.len() > 64 || triple.object.len() > 128 {
        validation_notes.push("Field length exceeds recommended limits".to_string());
        confidence *= 0.9;
    }
    
    // Check for problematic characters
    if contains_problematic_chars(&triple.subject) || 
       contains_problematic_chars(&triple.predicate) || 
       contains_problematic_chars(&triple.object) {
        validation_notes.push("Contains special characters that may cause issues".to_string());
        confidence *= 0.8;
    }
    
    // Check predicate format
    if !is_valid_predicate(&triple.predicate) {
        validation_notes.push("Predicate format is non-standard (use lowercase with underscores)".to_string());
        confidence *= 0.9;
    }
    
    Ok(ValidationResult {
        is_valid: conflicts.is_empty(),
        confidence,
        conflicts,
        sources: vec![],
        validation_notes,
    })
}

/// Validate consistency across multiple triples
pub async fn validate_consistency(
    triples: &[Triple],
    existing_triples: &[Triple],
) -> Result<ValidationResult> {
    let mut conflicts = Vec::new();
    let mut validation_notes = Vec::new();
    let mut confidence = 1.0;
    
    // Build index of existing facts
    let mut fact_index: HashMap<(String, String), Vec<String>> = HashMap::new();
    for triple in existing_triples {
        fact_index.entry((triple.subject.clone(), triple.predicate.clone()))
            .or_insert_with(Vec::new)
            .push(triple.object.clone());
    }
    
    // Check for conflicts
    for triple in triples {
        let key = (triple.subject.clone(), triple.predicate.clone());
        
        if let Some(existing_objects) = fact_index.get(&key) {
            if !existing_objects.contains(&triple.object) && !existing_objects.is_empty() {
                // Potential conflict - same subject-predicate with different object
                if is_single_valued_predicate(&triple.predicate) {
                    conflicts.push(format!(
                        "Conflict: {} {} already has value '{}', cannot add '{}'",
                        triple.subject, triple.predicate, existing_objects[0], triple.object
                    ));
                    confidence *= 0.5;
                } else {
                    validation_notes.push(format!(
                        "Adding additional value for {} {}",
                        triple.subject, triple.predicate
                    ));
                }
            }
        }
    }
    
    // Check for circular relationships
    for triple in triples {
        if triple.subject == triple.object && is_problematic_circular(&triple.predicate) {
            conflicts.push(format!(
                "Circular relationship detected: {} {} itself",
                triple.subject, triple.predicate
            ));
            confidence *= 0.7;
        }
    }
    
    Ok(ValidationResult {
        is_valid: conflicts.is_empty(),
        confidence,
        conflicts,
        sources: vec![],
        validation_notes,
    })
}

/// Validate sources and credibility
pub async fn validate_sources(
    triple: &Triple,
    source: Option<&str>,
) -> Result<Vec<String>> {
    let mut sources = Vec::new();
    let mut validation_notes = Vec::new();
    
    if let Some(src) = source {
        // Basic source validation
        let source_lower = src.to_lowercase();
        
        // Check for known reliable sources
        let is_reliable = source_lower.contains("wikipedia") ||
                         source_lower.contains("britannica") ||
                         source_lower.contains("nature.com") ||
                         source_lower.contains("science.org") ||
                         source_lower.contains("arxiv.org") ||
                         source_lower.contains(".edu") ||
                         source_lower.contains(".gov");
        
        // Check for problematic sources
        let is_problematic = source_lower.contains("blog") ||
                            source_lower.contains("forum") ||
                            source_lower.contains("reddit") ||
                            source_lower.contains("twitter") ||
                            source_lower.contains("facebook");
        
        // Validate URL format if it looks like a URL
        let is_valid_url = if src.starts_with("http://") || src.starts_with("https://") {
            src.contains('.') && src.len() > 10
        } else {
            true // Not a URL, could be a book title, paper, etc.
        };
        
        if !is_valid_url {
            validation_notes.push(format!("Source '{}' appears to be malformed", src));
        } else if is_reliable {
            sources.push(format!("{} [reliable]", src));
        } else if is_problematic {
            sources.push(format!("{} [unverified]", src));
            validation_notes.push("Source may require additional verification".to_string());
        } else {
            sources.push(src.to_string());
        }
        
        // Add source metadata
        if source_lower.contains("wikipedia") {
            validation_notes.push("Wikipedia source - check for citations".to_string());
        }
        
        // Check if source matches the content type
        if triple.predicate.contains("scientific") && !source_lower.contains("science") && !source_lower.contains("journal") {
            validation_notes.push("Scientific claim may need peer-reviewed source".to_string());
        }
        
        if triple.predicate.contains("historical") && !source_lower.contains("history") && !source_lower.contains("archive") {
            validation_notes.push("Historical claim may need primary source verification".to_string());
        }
    } else {
        validation_notes.push("No source provided - fact requires citation".to_string());
    }
    
    // Add validation notes as special sources
    for note in validation_notes {
        sources.push(format!("[Note: {}]", note));
    }
    
    Ok(sources)
}

/// Validate with LLM assistance
pub async fn validate_with_llm(
    triple: &Triple,
    context: &str,
) -> Result<ValidationResult> {
    let mut conflicts = Vec::new();
    let mut validation_notes = Vec::new();
    let mut confidence = 0.8; // Base confidence for LLM validation
    
    // Basic semantic validation checks
    
    // 1. Check for logical contradictions
    if triple.predicate == "is" {
        // Check for impossible type combinations
        let impossible_combinations = [
            ("person", "place"),
            ("person", "number"),
            ("place", "person"),
            ("living", "dead"),
            ("true", "false"),
        ];
        
        for (type1, type2) in &impossible_combinations {
            if triple.object.to_lowercase().contains(type1) && 
               context.to_lowercase().contains(&format!("{} is {}", triple.subject, type2)) {
                conflicts.push(format!(
                    "Logical contradiction: {} cannot be both {} and {}",
                    triple.subject, type1, type2
                ));
                confidence *= 0.3;
            }
        }
    }
    
    // 2. Check for temporal inconsistencies
    if triple.predicate.contains("born") || triple.predicate.contains("died") {
        // Extract years if present
        let year_regex = regex::Regex::new(r"\b(1[0-9]{3}|20[0-9]{2})\b").unwrap();
        
        if let Some(captures) = year_regex.captures(&triple.object) {
            if let Ok(year) = captures[1].parse::<i32>() {
                // Check if birth/death year makes sense
                if triple.predicate.contains("born") && year > 2024 {
                    conflicts.push(format!("Future birth year {} is invalid", year));
                    confidence *= 0.1;
                } else if triple.predicate.contains("died") && year > 2024 {
                    validation_notes.push(format!("Death year {} is in the future", year));
                    confidence *= 0.5;
                }
                
                // Check context for conflicting dates
                if context.contains("born") && context.contains("died") {
                    let birth_years: Vec<i32> = year_regex.captures_iter(context)
                        .filter_map(|cap| cap[1].parse().ok())
                        .collect();
                    
                    if birth_years.len() >= 2 {
                        let birth_year = birth_years[0];
                        let death_year = birth_years[1];
                        
                        if death_year < birth_year {
                            conflicts.push("Death year cannot be before birth year".to_string());
                            confidence *= 0.1;
                        } else if death_year - birth_year > 120 {
                            validation_notes.push("Unusually long lifespan detected".to_string());
                            confidence *= 0.7;
                        }
                    }
                }
            }
        }
    }
    
    // 3. Check for common sense violations
    if triple.predicate == "height" || triple.predicate == "weight" {
        // Extract numeric values
        let number_regex = regex::Regex::new(r"(\d+(?:\.\d+)?)\s*(m|meters?|cm|kg|pounds?|lbs?)").unwrap();
        
        if let Some(captures) = number_regex.captures(&triple.object) {
            if let Ok(value) = captures[1].parse::<f64>() {
                let unit = &captures[2];
                
                match (triple.predicate.as_str(), unit) {
                    ("height", "m" | "meters") => {
                        if value > 3.0 {
                            conflicts.push("Height exceeds human maximum".to_string());
                            confidence *= 0.2;
                        } else if value < 0.5 {
                            conflicts.push("Height below human minimum".to_string());
                            confidence *= 0.2;
                        }
                    }
                    ("height", "cm") => {
                        if value > 300.0 || value < 50.0 {
                            validation_notes.push("Height value seems unusual".to_string());
                            confidence *= 0.6;
                        }
                    }
                    ("weight", "kg") => {
                        if value > 650.0 {
                            conflicts.push("Weight exceeds recorded human maximum".to_string());
                            confidence *= 0.3;
                        } else if value < 2.0 {
                            conflicts.push("Weight below human minimum".to_string());
                            confidence *= 0.3;
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    
    // 4. Check for context consistency
    let subject_lower = triple.subject.to_lowercase();
    let context_lower = context.to_lowercase();
    
    // Check if the subject is mentioned in a contradictory way in context
    if context_lower.contains(&subject_lower) {
        // Look for negative statements about the fact
        let negative_patterns = [
            format!("{} is not", subject_lower),
            format!("{} isn't", subject_lower),
            format!("{} was not", subject_lower),
            format!("{} wasn't", subject_lower),
            format!("not {}", subject_lower),
            format!("never {}", subject_lower),
        ];
        
        for pattern in &negative_patterns {
            if context_lower.contains(pattern) {
                validation_notes.push("Context contains negative statements about this fact".to_string());
                confidence *= 0.7;
                break;
            }
        }
    }
    
    // 5. Simple fact plausibility check
    if triple.predicate == "capital_of" {
        // Basic check: capitals are usually cities, not people or abstract concepts
        if context.contains("person") || context.contains("scientist") || 
           context.contains("author") || context.contains("concept") {
            conflicts.push("A person or concept cannot be a capital".to_string());
            confidence *= 0.1;
        }
    }
    
    // Add source information
    let sources = if conflicts.is_empty() {
        vec!["LLM validation passed".to_string()]
    } else {
        vec!["LLM validation with issues".to_string()]
    };
    
    // Add final validation note based on confidence
    if confidence < 0.5 {
        validation_notes.push("Low confidence - fact requires human review".to_string());
    } else if confidence < 0.7 {
        validation_notes.push("Medium confidence - additional sources recommended".to_string());
    }
    
    Ok(ValidationResult {
        is_valid: conflicts.is_empty(),
        confidence,
        conflicts,
        sources,
        validation_notes,
    })
}

/// Check if a string contains problematic characters
fn contains_problematic_chars(s: &str) -> bool {
    s.contains('\n') || s.contains('\r') || s.contains('\t') || 
    s.contains('\0') || s.chars().any(|c| c.is_control())
}

/// Check if a predicate follows naming conventions
fn is_valid_predicate(predicate: &str) -> bool {
    // Should be lowercase with underscores
    predicate.chars().all(|c| c.is_lowercase() || c == '_' || c.is_numeric()) &&
    !predicate.starts_with('_') &&
    !predicate.ends_with('_') &&
    !predicate.contains("__")
}

/// Check if a predicate typically has single values
fn is_single_valued_predicate(predicate: &str) -> bool {
    matches!(
        predicate,
        "is" | "born_on" | "died_on" | "age" | "height" | "weight" | 
        "capital_of" | "president_of" | "ceo_of" | "married_to"
    )
}

/// Check if a circular relationship is problematic
fn is_problematic_circular(predicate: &str) -> bool {
    !matches!(
        predicate,
        "related_to" | "connected_to" | "similar_to" | "depends_on" |
        "influences" | "competes_with"
    )
}

/// Validate completeness of entity information
pub async fn validate_completeness(
    entity: &str,
    triples: &[Triple],
) -> Result<Vec<String>> {
    let mut missing_info = Vec::new();
    
    // Check what information exists about the entity
    let predicates: Vec<_> = triples.iter()
        .filter(|t| t.subject == entity)
        .map(|t| t.predicate.as_str())
        .collect();
    
    // Check for common expected information based on entity type
    if predicates.contains(&"is") {
        let entity_type = triples.iter()
            .find(|t| t.subject == entity && t.predicate == "is")
            .map(|t| t.object.as_str());
        
        if let Some(entity_type) = entity_type {
            let expected = get_expected_predicates(entity_type);
            for exp in expected {
                if !predicates.contains(&exp) {
                    missing_info.push(format!("Missing '{}' for {}", exp, entity));
                }
            }
        }
    }
    
    Ok(missing_info)
}

/// Get expected predicates based on entity type
fn get_expected_predicates(entity_type: &str) -> Vec<&'static str> {
    match entity_type {
        "person" | "scientist" | "author" | "artist" => {
            vec!["born_on", "nationality", "field_of_work"]
        }
        "company" | "organization" => {
            vec!["founded_in", "headquarters", "industry"]
        }
        "place" | "city" | "country" => {
            vec!["located_in", "population", "area"]
        }
        "book" | "movie" | "artwork" => {
            vec!["created_by", "published_in", "genre"]
        }
        _ => vec![]
    }
}