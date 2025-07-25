//! Storage-related request handlers with comprehensive error handling

use crate::core::triple::Triple;
use crate::core::knowledge_engine::KnowledgeEngine;
use crate::core::knowledge_types::TripleQuery;
use crate::core::entity_extractor::EntityExtractor;
use crate::core::relationship_extractor::RelationshipExtractor;
use crate::mcp::llm_friendly_server::utils::{update_usage_stats, StatsOperation};
use crate::mcp::llm_friendly_server::types::UsageStats;
use crate::mcp::llm_friendly_server::temporal_tracking::{TEMPORAL_INDEX, TemporalOperation};
use crate::mcp::llm_friendly_server::error_handling::{
    LlmkgError, LlmkgResult, HandlerResult,
    validation::{validate_string_field, validate_numeric_field, sanitize_input},
    graceful::with_fallback
};
use std::sync::Arc;
use tokio::sync::RwLock;
use serde_json::{json, Value};

/// Handle store_fact request with comprehensive error handling and validation
pub async fn handle_store_fact(
    knowledge_engine: &Arc<RwLock<KnowledgeEngine>>,
    usage_stats: &Arc<RwLock<UsageStats>>,
    params: Value,
) -> std::result::Result<(Value, String, Vec<String>), String> {
    match handle_store_fact_internal(knowledge_engine, usage_stats, params).await {
        Ok(result) => Ok(result),
        Err(error) => {
            log::error!("Storage error: {}", error);
            Err(error.to_string())
        }
    }
}

/// Internal implementation with proper error types
async fn handle_store_fact_internal(
    knowledge_engine: &Arc<RwLock<KnowledgeEngine>>,
    usage_stats: &Arc<RwLock<UsageStats>>,
    params: Value,
) -> HandlerResult {
    // Validate and sanitize inputs with comprehensive error handling
    let subject = validate_string_field(
        "subject", 
        params.get("subject").and_then(|v| v.as_str()),
        true, // required
        Some(128), // max_length
        Some(1) // min_length
    ).map(|s| sanitize_input(&s))?;
    
    let predicate = validate_string_field(
        "predicate",
        params.get("predicate").and_then(|v| v.as_str()),
        true, // required
        Some(64), // max_length  
        Some(1) // min_length
    ).map(|s| sanitize_input(&s))?;
    
    let object = validate_string_field(
        "object",
        params.get("object").and_then(|v| v.as_str()),
        true, // required
        Some(128), // max_length
        Some(1) // min_length
    ).map(|s| sanitize_input(&s))?;
    
    let confidence = validate_numeric_field(
        "confidence",
        params.get("confidence").and_then(|v| v.as_f64()).map(|f| f as f32),
        false, // not required
        Some(0.0), // min_value
        Some(1.0) // max_value
    )?.unwrap_or(1.0);
    
    // Create the triple with error handling
    let triple = Triple::with_metadata(
        subject.clone(),
        predicate.clone(),
        object.clone(),
        confidence,
        Some("user_input".to_string()),
    ).map_err(|e| LlmkgError::StorageError {
        operation: "create_triple".to_string(),
        entity_id: Some(format!("{}-{}-{}", subject, predicate, object)),
        cause: format!("Failed to create triple: {}", e),
    })?;
    
    // Check if this triple already exists (for update vs create detection)
    let existing_query = TripleQuery {
        subject: Some(subject.to_string()),
        predicate: Some(predicate.to_string()),
        object: None,
        limit: 1,
        min_confidence: 0.0,
        include_chunks: false,
    };
    
    let existing_result = {
        let engine = knowledge_engine.read().await;
        engine.query_triples(existing_query)
            .map_err(|e| LlmkgError::QueryError {
                query_type: "existence_check".to_string(),
                parameters: json!({
                    "subject": subject,
                    "predicate": predicate
                }),
                cause: format!("Failed to check existing triples: {}", e),
            }).ok()
    };
    
    let (operation, previous_value) = if let Some(result) = existing_result {
        if let Some(existing_triple) = result.triples.first() {
            // This is an update - capture the previous value
            (TemporalOperation::Update, Some(existing_triple.object.clone()))
        } else {
            // This is a new creation
            (TemporalOperation::Create, None)
        }
    } else {
        // Query failed or no results - treat as create
        (TemporalOperation::Create, None)
    };
    
    // Store the triple with graceful error handling
    let node_id = {
        let engine = knowledge_engine.write().await;
        engine.store_triple(triple.clone(), None)
            .map_err(|e| LlmkgError::StorageError {
                operation: "store_triple".to_string(),
                entity_id: Some(format!("{}-{}-{}", subject, predicate, object)),
                cause: format!("Failed to store triple: {}", e),
            })?
    };
    
    // Record in temporal index with error handling
    if let Err(e) = std::panic::catch_unwind(|| {
        TEMPORAL_INDEX.record_operation(triple.clone(), operation, previous_value.clone());
    }) {
        log::warn!("Failed to record temporal operation: {:?}", e);
        // Continue execution - temporal tracking is not critical for core functionality
    }
    
    // Update usage stats (non-critical operation)
    if let Err(e) = update_usage_stats(usage_stats, StatsOperation::StoreTriple, 10).await {
        log::warn!("Failed to update usage stats: {}", e);
        // Continue execution - stats are not critical
    }
    
    let data = json!({
        "success": true,
        "node_id": node_id,
        "subject": subject,
        "predicate": predicate,
        "object": object,
        "confidence": confidence
    });
    
    let message = format!("Stored fact: {} {} {}", subject, predicate, object);
    let suggestions = vec![
        format!("Explore connections with: explore_connections(start_entity=\"{}\")", subject),
        format!("Find related facts with: find_facts(subject=\"{}\")", subject),
    ];
    
    Ok((data, message, suggestions))
}

/// Primary entity and relationship extraction with enhanced methods
async fn extract_entities_and_relationships_primary(content: &str) -> LlmkgResult<(Vec<String>, Vec<(String, String, String)>)> {
    let entity_extractor = EntityExtractor::new();
    let relationship_extractor = RelationshipExtractor::new();
    
    // Extract entities
    let entities = entity_extractor.extract_entities(content);
    let extracted_entities: Vec<String> = entities.iter().map(|e| e.name.clone()).collect();
    
    // Extract relationships
    let relationships = relationship_extractor.extract_relationships(content, &entities);
    let extracted_relationships: Vec<(String, String, String)> = relationships.iter()
        .map(|r| (r.subject.clone(), r.predicate.clone(), r.object.clone()))
        .collect();
    
    if extracted_entities.is_empty() && extracted_relationships.is_empty() {
        return Err(LlmkgError::ExtractionError {
            extraction_type: "primary".to_string(),
            input_sample: content.chars().take(100).collect(),
            cause: "No entities or relationships extracted".to_string(),
        });
    }
    
    Ok((extracted_entities, extracted_relationships))
}

/// Fallback entity and relationship extraction using simple methods
async fn extract_entities_and_relationships_fallback(content: &str) -> LlmkgResult<(Vec<String>, Vec<(String, String, String)>)> {
    log::info!("Using fallback extraction methods for content analysis");
    
    // Simple keyword-based entity extraction
    let words: Vec<&str> = content.split_whitespace().collect();
    let mut entities = Vec::new();
    
    // Look for capitalized words as potential entities (simple heuristic)
    for word in &words {
        if word.len() > 2 {
            if let Some(first_char) = word.chars().next() {
                if first_char.is_uppercase() {
                    let clean_word = word.trim_matches(|c: char| !c.is_alphabetic());
                    if !clean_word.is_empty() && clean_word.len() > 2 {
                        entities.push(clean_word.to_string());
                    }
                }
            }
        }
    }
    
    // Remove duplicates
    entities.sort();
    entities.dedup();
    
    // Simple relationship extraction based on common patterns
    let mut relationships = Vec::new();
    let content_lower = content.to_lowercase();
    
    // Look for "is a", "has", "contains" patterns
    for entity in &entities {
        if content_lower.contains(&format!("{} is a", entity.to_lowercase())) {
            relationships.push((entity.clone(), "is_a".to_string(), "concept".to_string()));
        }
        if content_lower.contains(&format!("{} has", entity.to_lowercase())) {
            relationships.push((entity.clone(), "has".to_string(), "property".to_string()));
        }
    }
    
    // Ensure we have at least something to return
    if entities.is_empty() {
        // Extract any reasonable-length words as entities
        entities = words.iter()
            .filter(|w| w.len() > 3 && w.len() < 20)
            .take(5)
            .map(|w| w.trim_matches(|c: char| !c.is_alphabetic()).to_string())
            .filter(|w| !w.is_empty())
            .collect();
    }
    
    log::info!("Fallback extraction found {} entities and {} relationships", 
        entities.len(), relationships.len());
    
    Ok((entities, relationships))
}

/// Handle store_knowledge request with comprehensive error handling
pub async fn handle_store_knowledge(
    knowledge_engine: &Arc<RwLock<KnowledgeEngine>>,
    usage_stats: &Arc<RwLock<UsageStats>>,
    params: Value,
) -> std::result::Result<(Value, String, Vec<String>), String> {
    match handle_store_knowledge_internal(knowledge_engine, usage_stats, params).await {
        Ok(result) => Ok(result),
        Err(error) => {
            log::error!("Storage error: {}", error);
            Err(error.to_string())
        }
    }
}

/// Internal implementation with proper error types  
async fn handle_store_knowledge_internal(
    knowledge_engine: &Arc<RwLock<KnowledgeEngine>>,
    usage_stats: &Arc<RwLock<UsageStats>>,
    params: Value,
) -> HandlerResult {
    // Validate and sanitize inputs
    let content = validate_string_field(
        "content",
        params.get("content").and_then(|v| v.as_str()),
        true, // required
        Some(50000), // max_length
        Some(1) // min_length
    ).map(|s| sanitize_input(&s))?;
    
    let title = validate_string_field(
        "title",
        params.get("title").and_then(|v| v.as_str()),
        true, // required
        Some(200), // max_length
        Some(1) // min_length
    ).map(|s| sanitize_input(&s))?;
    
    let category = validate_string_field(
        "category",
        params.get("category").and_then(|v| v.as_str()),
        false, // not required
        Some(50), // max_length
        None // no min_length
    ).map(|s| sanitize_input(&s)).unwrap_or_else(|_| "general".to_string());
    
    let source = params.get("source").and_then(|v| v.as_str())
        .map(|s| sanitize_input(s));
    
    // Extract entities and relationships with fallback mechanisms
    let extraction_result = with_fallback(
        extract_entities_and_relationships_primary(&content),
        extract_entities_and_relationships_fallback(&content),
        "entity_relationship_extraction"
    ).await?;
    
    let (extracted_entities, extracted_relationships) = extraction_result;
    
    let engine = knowledge_engine.write().await;
    
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
    
    // Store the chunk with error handling
    let chunk_triple = Triple::new(
        chunk_id.clone(),
        "is".to_string(),
        "knowledge_chunk".to_string(),
    ).map_err(|e| LlmkgError::StorageError {
        operation: "create_chunk_triple".to_string(),
        entity_id: Some(chunk_id.clone()),
        cause: format!("Failed to create chunk triple: {}", e),
    })?;
    
    match engine.store_triple(chunk_triple.clone(), None) {
        Ok(_) => {
            // Record the chunk creation in temporal index
            TEMPORAL_INDEX.record_operation(chunk_triple, TemporalOperation::Create, None);
            
            // Store extracted entities and relationships
            let mut stored_count = 0;
            
            for entity in &extracted_entities {
                if let Ok(entity_triple) = Triple::new(
                    entity.clone(),
                    "mentioned_in".to_string(),
                    chunk_id.clone(),
                ) {
                    // Check if this entity relation already exists
                    let existing_query = TripleQuery {
                        subject: Some(entity.clone()),
                        predicate: Some("mentioned_in".to_string()),
                        object: Some(chunk_id.clone()),
                        limit: 1,
                        min_confidence: 0.0,
                        include_chunks: false,
                    };
                    
                    let exists = engine.query_triples(existing_query)
                        .map(|r| !r.triples.is_empty())
                        .unwrap_or(false);
                    
                    if engine.store_triple(entity_triple.clone(), None).is_ok() {
                        stored_count += 1;
                        // Record temporal operation
                        let operation = if exists { TemporalOperation::Update } else { TemporalOperation::Create };
                        TEMPORAL_INDEX.record_operation(entity_triple, operation, None);
                    }
                }
            }
            
            for (subj, pred, obj) in &extracted_relationships {
                if let Ok(rel_triple) = Triple::new(
                    subj.clone(),
                    pred.clone(),
                    obj.clone(),
                ) {
                    // Check if this relationship already exists
                    let existing_query = TripleQuery {
                        subject: Some(subj.clone()),
                        predicate: Some(pred.clone()),
                        object: Some(obj.clone()),
                        limit: 1,
                        min_confidence: 0.0,
                        include_chunks: false,
                    };
                    
                    let existing_result = engine.query_triples(existing_query).ok();
                    let (operation, previous_value) = if let Some(result) = existing_result {
                        if let Some(existing) = result.triples.first() {
                            // For exact matches, this would be a duplicate, but we might be updating confidence
                            (TemporalOperation::Update, Some(existing.confidence.to_string()))
                        } else {
                            (TemporalOperation::Create, None)
                        }
                    } else {
                        (TemporalOperation::Create, None)
                    };
                    
                    if engine.store_triple(rel_triple.clone(), None).is_ok() {
                        stored_count += 1;
                        // Record temporal operation
                        TEMPORAL_INDEX.record_operation(rel_triple, operation, previous_value);
                    }
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
        Err(e) => Err(LlmkgError::StorageError {
            operation: "store_knowledge_chunk".to_string(),
            entity_id: Some(chunk_id),
            cause: format!("Failed to store knowledge: {}", e),
        })
    }
}

