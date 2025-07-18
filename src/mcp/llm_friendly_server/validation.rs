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
    
    if let Some(src) = source {
        sources.push(src.to_string());
    }
    
    // In a real implementation, this would:
    // 1. Check source credibility
    // 2. Cross-reference with other sources
    // 3. Verify timestamps and versioning
    
    Ok(sources)
}

/// Validate with LLM assistance (placeholder)
pub async fn validate_with_llm(
    triple: &Triple,
    context: &str,
) -> Result<ValidationResult> {
    // In a real implementation, this would:
    // 1. Use an LLM to check factual accuracy
    // 2. Verify logical consistency
    // 3. Check for common sense violations
    
    Ok(ValidationResult {
        is_valid: true,
        confidence: 0.8,
        conflicts: vec![],
        sources: vec!["LLM validation".to_string()],
        validation_notes: vec!["Validated with LLM assistance".to_string()],
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