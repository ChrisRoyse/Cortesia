//! Storage-related request handlers

use crate::core::triple::{Triple, NodeType};
use crate::core::knowledge_engine::KnowledgeEngine;
use crate::mcp::llm_friendly_server::utils::{update_usage_stats, StatsOperation};
use crate::mcp::llm_friendly_server::types::UsageStats;
use crate::error::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde_json::{json, Value};

/// Handle store_fact request
pub async fn handle_store_fact(
    knowledge_engine: &Arc<RwLock<KnowledgeEngine>>,
    usage_stats: &Arc<RwLock<UsageStats>>,
    params: Value,
) -> std::result::Result<(Value, String, Vec<String>), String> {
    let subject = params.get("subject").and_then(|v| v.as_str())
        .ok_or("Missing required field: subject")?;
    let predicate = params.get("predicate").and_then(|v| v.as_str())
        .ok_or("Missing required field: predicate")?;
    let object = params.get("object").and_then(|v| v.as_str())
        .ok_or("Missing required field: object")?;
    let confidence = params.get("confidence")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0) as f32;
    
    // Validate inputs
    if subject.is_empty() || predicate.is_empty() || object.is_empty() {
        return Err("Subject, predicate, and object cannot be empty".to_string());
    }
    
    if subject.len() > 128 || object.len() > 128 {
        return Err("Subject and object must be 128 characters or less".to_string());
    }
    
    if predicate.len() > 64 {
        return Err("Predicate must be 64 characters or less".to_string());
    }
    
    // Create and store the triple
    let triple = Triple::new(
        subject.to_string(),
        predicate.to_string(),
        object.to_string(),
    );
    
    let mut engine = knowledge_engine.write().await;
    match engine.add_triple(triple.clone()) {
        Ok(_) => {
            // Update stats
            let _ = update_usage_stats(usage_stats, StatsOperation::StoreTriple, 5).await;
            
            let data = json!({
                "stored": true,
                "triple": {
                    "subject": subject,
                    "predicate": predicate,
                    "object": object,
                    "confidence": confidence
                }
            });
            
            let message = format!(
                "✓ Stored fact: {} {} {} (confidence: {:.2})",
                subject, predicate, object, confidence
            );
            
            let suggestions = vec![
                format!("You can now query this fact with: find_facts(subject=\"{}\")", subject),
                "Consider adding related facts to build a knowledge network".to_string(),
            ];
            
            Ok((data, message, suggestions))
        }
        Err(e) => Err(format!("Failed to store fact: {}", e))
    }
}

/// Handle store_knowledge request
pub async fn handle_store_knowledge(
    knowledge_engine: &Arc<RwLock<KnowledgeEngine>>,
    usage_stats: &Arc<RwLock<UsageStats>>,
    params: Value,
) -> std::result::Result<(Value, String, Vec<String>), String> {
    let content = params.get("content").and_then(|v| v.as_str())
        .ok_or("Missing required field: content")?;
    let title = params.get("title").and_then(|v| v.as_str())
        .ok_or("Missing required field: title")?;
    let category = params.get("category").and_then(|v| v.as_str())
        .unwrap_or("general");
    let source = params.get("source").and_then(|v| v.as_str());
    
    // Validate inputs
    if content.is_empty() || title.is_empty() {
        return Err("Content and title cannot be empty".to_string());
    }
    
    if content.len() > 50000 {
        return Err("Content exceeds maximum length of 50,000 characters".to_string());
    }
    
    // Extract entities and relationships (simplified)
    let extracted_entities = extract_entities_from_text(content);
    let extracted_relationships = extract_relationships_from_text(content, &extracted_entities);
    
    let mut engine = knowledge_engine.write().await;
    
    // Store as knowledge chunk
    let chunk_id = format!("chunk_{}", uuid::Uuid::new_v4());
    let mut chunk_metadata = json!({
        "title": title,
        "category": category,
        "content_length": content.len(),
        "extracted_entities": extracted_entities.len(),
        "extracted_relationships": extracted_relationships.len(),
    });
    
    if let Some(src) = source {
        chunk_metadata["source"] = json!(src);
    }
    
    // Store the chunk (simplified - in real implementation would use proper chunk storage)
    let chunk_triple = Triple::new(
        chunk_id.clone(),
        "is".to_string(),
        "knowledge_chunk".to_string(),
    );
    
    match engine.add_triple(chunk_triple) {
        Ok(_) => {
            // Store extracted entities and relationships
            let mut stored_count = 0;
            
            for entity in &extracted_entities {
                let entity_triple = Triple::new(
                    entity.clone(),
                    "mentioned_in".to_string(),
                    chunk_id.clone(),
                );
                if engine.add_triple(entity_triple).is_ok() {
                    stored_count += 1;
                }
            }
            
            for (subj, pred, obj) in &extracted_relationships {
                let rel_triple = Triple::new(
                    subj.clone(),
                    pred.clone(),
                    obj.clone(),
                );
                if engine.add_triple(rel_triple).is_ok() {
                    stored_count += 1;
                }
            }
            
            // Update stats
            let _ = update_usage_stats(usage_stats, StatsOperation::StoreChunk, 20).await;
            
            let data = json!({
                "stored": true,
                "chunk_id": chunk_id,
                "title": title,
                "category": category,
                "extracted": {
                    "entities": extracted_entities,
                    "relationships": extracted_relationships.len(),
                    "total_stored": stored_count
                }
            });
            
            let message = format!(
                "✓ Stored knowledge chunk '{}' with {} extracted entities and {} relationships",
                title, extracted_entities.len(), extracted_relationships.len()
            );
            
            let suggestions = vec![
                format!("Explore extracted entities with: explore_connections(start_entity=\"{}\")", 
                    extracted_entities.first().unwrap_or(&"entity".to_string())),
                "Use ask_question to query this knowledge".to_string(),
            ];
            
            Ok((data, message, suggestions))
        }
        Err(e) => Err(format!("Failed to store knowledge: {}", e))
    }
}

/// Extract entities from text (simplified)
fn extract_entities_from_text(text: &str) -> Vec<String> {
    let mut entities = Vec::new();
    
    // Simple extraction: capitalized words likely to be entities
    for word in text.split_whitespace() {
        if word.len() > 2 && word.chars().next().map_or(false, |c| c.is_uppercase()) {
            let clean_word = word.trim_matches(|c: char| !c.is_alphanumeric());
            if !clean_word.is_empty() && !is_common_word(clean_word) {
                entities.push(clean_word.to_string());
            }
        }
    }
    
    // Deduplicate
    entities.sort();
    entities.dedup();
    
    entities
}

/// Extract relationships from text (very simplified)
fn extract_relationships_from_text(text: &str, entities: &[String]) -> Vec<(String, String, String)> {
    let mut relationships = Vec::new();
    
    // Very simple pattern matching
    let text_lower = text.to_lowercase();
    
    for i in 0..entities.len() {
        for j in 0..entities.len() {
            if i != j {
                let entity1 = &entities[i];
                let entity2 = &entities[j];
                
                // Check for common relationship patterns
                if text_lower.contains(&format!("{} is", entity1.to_lowercase())) {
                    relationships.push((entity1.clone(), "is".to_string(), entity2.clone()));
                }
                
                if text_lower.contains(&format!("{} created", entity1.to_lowercase())) {
                    relationships.push((entity1.clone(), "created".to_string(), entity2.clone()));
                }
                
                // Add more patterns as needed
            }
        }
    }
    
    relationships
}

/// Check if a word is too common to be an entity
fn is_common_word(word: &str) -> bool {
    matches!(
        word.to_lowercase().as_str(),
        "the" | "and" | "or" | "but" | "in" | "on" | "at" | "to" | "for" |
        "of" | "with" | "by" | "from" | "as" | "is" | "was" | "are" | "were" |
        "been" | "being" | "have" | "has" | "had" | "do" | "does" | "did" |
        "will" | "would" | "could" | "should" | "may" | "might" | "must" |
        "can" | "this" | "that" | "these" | "those" | "a" | "an"
    )
}