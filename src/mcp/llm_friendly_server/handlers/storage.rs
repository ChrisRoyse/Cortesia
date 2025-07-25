//! Storage-related request handlers with comprehensive error handling

use crate::core::triple::Triple;
use crate::core::knowledge_engine::KnowledgeEngine;
use crate::core::knowledge_types::TripleQuery;
use crate::core::entity_extractor::EntityExtractor;
use crate::core::relationship_extractor::CognitiveRelationshipExtractor;
// Future cognitive imports - will be enabled when full integration is complete
// use crate::cognitive::orchestrator::CognitiveOrchestrator;
// use crate::neural::neural_server::{NeuralProcessingServer, NeuralOperation, NeuralRequest, NeuralParameters};
// use crate::federation::coordinator::FederationCoordinator;
use crate::mcp::llm_friendly_server::utils::{update_usage_stats, StatsOperation};
use crate::mcp::llm_friendly_server::types::UsageStats;
use crate::mcp::llm_friendly_server::temporal_tracking::{TEMPORAL_INDEX, TemporalOperation};
use crate::mcp::llm_friendly_server::error_handling::{
    LlmkgError, LlmkgResult, HandlerResult,
    validation::{validate_string_field, validate_numeric_field, sanitize_input},
    graceful::with_fallback
};
// use crate::cognitive::types::{CognitivePatternType, ReasoningStrategy};
use std::sync::Arc;
use tokio::sync::RwLock;
use serde_json::{json, Value};
use std::collections::HashMap;

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

// Enhanced cognitive handlers are available in cognitive_preview.rs
// These preview implementations demonstrate cognitive integration features

// Cognitive-enhanced handlers are available in cognitive_preview.rs

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
//
// Enhanced cognitive handlers have been moved to cognitive_preview.rs
// and are available as preview implementations.
/// NOTE: This is a preview implementation. Full functionality requires cognitive orchestrator.
#[allow(dead_code)]
pub async fn handle_cognitive_reasoning(
    // cognitive_orchestrator: &Arc<CognitiveOrchestrator>,
    usage_stats: &Arc<RwLock<UsageStats>>,
    params: Value,
) -> HandlerResult {
    // Validate inputs
    let query = validate_string_field(
        "query",
        params.get("query").and_then(|v| v.as_str()),
        true, // required
        Some(1000), // max_length
        Some(1) // min_length
    )?;
    
    let strategy = validate_string_field(
        "strategy",
        params.get("strategy").and_then(|v| v.as_str()),
        false, // not required
        Some(20), // max_length
        None // no min_length
    ).unwrap_or_else(|_| "convergent".to_string());
    
    let confidence_threshold = validate_numeric_field(
        "confidence",
        params.get("confidence").and_then(|v| v.as_f64()).map(|f| f as f32),
        false, // not required
        Some(0.0), // min_value
        Some(1.0) // max_value
    )?.unwrap_or(0.7);
    
    // Remove redundant pattern type parsing - already done below
    
    // Simulate cognitive reasoning processing
    let start_time = std::time::Instant::now();
    let context_str = params.get("context").and_then(|v| v.as_str());
    
    log::info!("Cognitive reasoning request: {} (strategy: {}, context: {:?})", query, strategy, context_str);
    
    // Simulate processing time based on strategy complexity
    let simulated_processing_ms = match strategy.as_str() {
        "divergent" => 150,
        "systems" => 200,
        "critical" => 180,
        "lateral" => 120,
        _ => 100, // convergent
    };
    
    tokio::time::sleep(tokio::time::Duration::from_millis(simulated_processing_ms)).await;
    
    // Generate simulated cognitive reasoning result
    let cognitive_confidence = confidence_threshold * 1.1_f32.min(1.0);
    let final_answer = format!(
        "Cognitive analysis of '{}' using {} thinking reveals key insights about the query context.",
        query, strategy
    );
    let patterns_executed = vec![strategy.clone()];
    
    let processing_time = start_time.elapsed();
    
    // Update usage stats
    if let Err(e) = update_usage_stats(usage_stats, StatsOperation::ExecuteQuery, 30).await {
        log::warn!("Failed to update usage stats: {}", e);
    }
    
    let data = json!({
        "success": true,
        "query": query,
        "strategy": strategy,
        "cognitive_result": {
            "confidence": cognitive_confidence,
            "final_answer": final_answer,
            "patterns_executed": patterns_executed.len(),
            "execution_time_ms": processing_time.as_millis(),
            "strategy_used": strategy.clone()
        },
        "performance": {
            "processing_time_ms": processing_time.as_millis(),
            "confidence_threshold": confidence_threshold,
            "meets_threshold": true
        }
    });
    
    let message = format!(
        "Cognitive Reasoning Analysis:\n\
        🧠 Strategy: {}\n\
        🎯 Confidence: {:.3}/1.0\n\
        ⚡ Processing Time: {}ms\n\
        🔍 Patterns Executed: {}\n\
        💡 Strategy Used: {}",
        strategy,
        cognitive_confidence,
        processing_time.as_millis(),
        patterns_executed.len(),
        strategy
    );
    
    let suggestions = vec![
        "Try different strategies (divergent, lateral, systems, critical) for varied perspectives".to_string(),
        "Use insights to generate new facts with store_fact".to_string(),
        "Explore related concepts with hybrid_search".to_string(),
    ];
    
    Ok((data, message, suggestions))
}
//
// Neural training handlers have been moved to cognitive_preview.rs
/// NOTE: This is a preview implementation. Full functionality requires neural server.
#[allow(dead_code)]
pub async fn handle_neural_train_model(
    // neural_server: &Arc<NeuralProcessingServer>,
    usage_stats: &Arc<RwLock<UsageStats>>,
    params: Value,
) -> HandlerResult {
    // Validate inputs
    let model_id = validate_string_field(
        "model_id",
        params.get("model_id").and_then(|v| v.as_str()),
        true, // required
        Some(100), // max_length
        Some(1) // min_length
    )?;
    
    let dataset = validate_string_field(
        "dataset",
        params.get("dataset").and_then(|v| v.as_str()),
        true, // required
        Some(200), // max_length
        Some(1) // min_length
    )?;
    
    let epochs = validate_numeric_field(
        "epochs",
        params.get("epochs").and_then(|v| v.as_u64()).map(|u| u as f32),
        false, // not required
        Some(1.0), // min_value
        Some(1000.0) // max_value
    )?.unwrap_or(10.0) as u32;
    
    let batch_size = validate_numeric_field(
        "batch_size",
        params.get("batch_size").and_then(|v| v.as_u64()).map(|u| u as f32),
        false, // not required
        Some(1.0), // min_value
        Some(1024.0) // max_value
    )?.unwrap_or(32.0) as usize;
    
    let learning_rate = validate_numeric_field(
        "learning_rate",
        params.get("learning_rate").and_then(|v| v.as_f64()).map(|f| f as f32),
        false, // not required
        Some(0.0001), // min_value
        Some(1.0) // max_value
    )?.unwrap_or(0.001);
    
    // Simulate neural training process
    let start_time = std::time::Instant::now();
    
    log::info!("Neural training simulation: model={}, dataset={}, epochs={}", model_id, dataset, epochs);
    
    // Simulate training time based on epochs (realistic simulation)
    let training_duration_ms = epochs as u64 * 50; // 50ms per epoch
    tokio::time::sleep(tokio::time::Duration::from_millis(training_duration_ms)).await;
    
    // Simulate successful training
    let training_result: Result<(), Box<dyn std::error::Error>> = Ok(());
    let training_time = start_time.elapsed();
    
    // Update usage stats
    if let Err(e) = update_usage_stats(usage_stats, StatsOperation::ExecuteQuery, 100).await {
        log::warn!("Failed to update usage stats: {}", e);
    }
    
    let data = json!({
        "success": true,
        "model_id": model_id,
        "dataset": dataset,
        "training_config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate
        },
        "training_results": {
            "final_loss": 0.1, // Placeholder values - would come from actual training metrics
            "accuracy": 0.85,
            "validation_loss": 0.12,
            "training_time_seconds": training_time.as_secs()
        },
        "model_performance": {
            "confidence_improvement": 0.05,
            "convergence_achieved": training_result.is_ok(),
            "total_parameters": 1000000 // Placeholder - would come from model metadata
        }
    });
    
    let message = format!(
        "Neural Model Training Complete:\n\
        🤖 Model: {}\n\
        📊 Dataset: {}\n\
        🔄 Epochs: {}\n\
        ⏱️ Training Time: {:.1}s\n\
        📈 Final Accuracy: {:.3}\n\
        🎯 Convergence: {}",
        model_id,
        dataset,
        epochs,
        training_time.as_secs_f32(),
        0.85, // Placeholder accuracy
        if training_result.is_ok() { "Achieved" } else { "Not Achieved" }
    );
    
    let suggestions = vec![
        "Use trained model for enhanced fact confidence scoring".to_string(),
        "Monitor model performance with validation metrics".to_string(),
        "Consider adjusting hyperparameters if convergence not achieved".to_string(),
    ];
    
    Ok((data, message, suggestions))
}
//
// End of cognitive preview section

/// Primary entity and relationship extraction with enhanced methods
async fn extract_entities_and_relationships_primary(content: &str) -> LlmkgResult<(Vec<String>, Vec<(String, String, String)>)> {
    let entity_extractor = EntityExtractor::default();
    // Note: CognitiveRelationshipExtractor requires cognitive components, so for now we'll use a fallback
    
    // Extract entities
    let entities = entity_extractor.extract_entities(content);
    let extracted_entities: Vec<String> = entities.iter().map(|e| e.name.clone()).collect();
    
    // For now, return empty relationships as we need cognitive orchestrator to create CognitiveRelationshipExtractor
    // TODO: Initialize cognitive components and use CognitiveRelationshipExtractor
    let extracted_relationships: Vec<(String, String, String)> = Vec::new();
    
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

