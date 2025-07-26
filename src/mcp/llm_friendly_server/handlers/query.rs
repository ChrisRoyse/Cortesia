//! Query-related request handlers

use crate::core::triple::Triple;
use crate::core::knowledge_engine::KnowledgeEngine;
use crate::core::knowledge_types::TripleQuery;
use crate::core::question_parser::QuestionParser;
use crate::core::answer_generator::AnswerGenerator;
use crate::core::cognitive_question_answering::CognitiveQuestionAnsweringEngine;
use crate::mcp::llm_friendly_server::utils::{update_usage_stats, StatsOperation};
use crate::mcp::llm_friendly_server::types::UsageStats;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde_json::{json, Value};

/// Handle find_facts request
pub async fn handle_find_facts(
    knowledge_engine: &Arc<RwLock<KnowledgeEngine>>,
    usage_stats: &Arc<RwLock<UsageStats>>,
    params: Value,
) -> std::result::Result<(Value, String, Vec<String>), String> {
    let query = params.get("query")
        .ok_or_else(|| "Missing required 'query' parameter".to_string())?;
    
    let subject = query.get("subject").and_then(|v| v.as_str());
    let predicate = query.get("predicate").and_then(|v| v.as_str());
    let object = query.get("object").and_then(|v| v.as_str());
    let limit = params.get("limit")
        .and_then(|v| v.as_u64())
        .unwrap_or(10)
        .min(100) as usize;
    
    // At least one field must be specified (enforced by minProperties in schema)
    if subject.is_none() && predicate.is_none() && object.is_none() {
        return Err("At least one of subject, predicate, or object must be specified in the query".to_string());
    }
    
    // Build query
    let query = TripleQuery {
        subject: subject.map(|s| s.to_string()),
        predicate: predicate.map(|p| p.to_string()),
        object: object.map(|o| o.to_string()),
        limit: 100,
        min_confidence: 0.0,
        include_chunks: false,
    };
    
    let engine = knowledge_engine.read().await;
    match engine.query_triples(query) {
        Ok(triples) => {
            let _ = update_usage_stats(usage_stats, StatsOperation::ExecuteQuery, 10).await;
            
            let facts: Vec<_> = triples.triples.iter().map(|t| {
                json!({
                    "subject": &t.subject,
                    "predicate": &t.predicate,
                    "object": &t.object,
                    "confidence": 1.0 // Would come from metadata
                })
            }).collect();
            
            let data = json!({
                "facts": facts,
                "count": triples.triples.len(),
                "limit": limit,
                "query": {
                    "subject": subject,
                    "predicate": predicate,
                    "object": object
                }
            });
            
            let message = if triples.triples.is_empty() {
                "No facts found matching your query".to_string()
            } else {
                format!("Found {} fact{}:\n{}", 
                    triples.triples.len(),
                    if triples.triples.len() == 1 { "" } else { "s" },
                    format_facts_for_display(&triples.triples, 5)
                )
            };
            
            let suggestions = if triples.triples.is_empty() {
                vec![
                    "Try using fewer constraints in your query".to_string(),
                    "Check spelling and capitalization of entity names".to_string(),
                    "Use 'ask_question' for natural language queries".to_string(),
                ]
            } else {
                vec![
                    "Use 'explore_connections' to find related entities".to_string(),
                    "Try 'ask_question' to understand these facts in context".to_string(),
                ]
            };
            
            Ok((data, message, suggestions))
        }
        Err(e) => Err(format!("Query failed: {}", e))
    }
}

/// Handle ask_question request with cognitive enhancements
pub async fn handle_ask_question_cognitive(
    cognitive_qa_engine: &Arc<CognitiveQuestionAnsweringEngine>,
    usage_stats: &Arc<RwLock<UsageStats>>,
    params: Value,
) -> std::result::Result<(Value, String, Vec<String>), String> {
    let question = params.get("question").and_then(|v| v.as_str())
        .ok_or("Missing required field: question")?;
    let context = params.get("context").and_then(|v| v.as_str());
    let max_results = params.get("max_results")
        .and_then(|v| v.as_u64())
        .unwrap_or(5)
        .min(20) as usize;
    
    if question.is_empty() {
        return Err("Question cannot be empty".to_string());
    }
    
    // Use cognitive question answering engine
    match cognitive_qa_engine.answer_question_cognitive(question, context).await {
        Ok(cognitive_answer) => {
            let _ = update_usage_stats(usage_stats, StatsOperation::ExecuteQuery, 20).await;
            
            // Convert cognitive answer to MCP response format
            let relevant_facts: Vec<_> = cognitive_answer.supporting_facts.iter()
                .take(max_results)
                .map(|f| {
                    json!({
                        "subject": &f.subject,
                        "predicate": &f.predicate,
                        "object": &f.object,
                        "confidence": f.confidence,
                        "cognitive_relevance": f.cognitive_relevance,
                        "neural_salience": f.neural_salience
                    })
                })
                .collect();
            
            let data = json!({
                "question": question,
                "context": context,
                "entities": cognitive_answer.supporting_facts.iter()
                    .flat_map(|f| vec![f.subject.clone(), f.object.clone()])
                    .collect::<std::collections::HashSet<_>>()
                    .into_iter()
                    .collect::<Vec<_>>(),
                "question_type": format!("{:?}", cognitive_answer.answer_type),
                "relevant_facts": relevant_facts,
                "answer": cognitive_answer.text,
                "confidence": cognitive_answer.confidence,
                "relevance_score": cognitive_answer.answer_quality_metrics.relevance_score,
                "cognitive_patterns_used": cognitive_answer.cognitive_patterns_used
                    .iter()
                    .map(|p| format!("{:?}", p))
                    .collect::<Vec<_>>(),
                "neural_models_used": cognitive_answer.neural_models_used,
                "processing_time_ms": cognitive_answer.processing_time_ms,
                "quality_metrics": {
                    "relevance": cognitive_answer.answer_quality_metrics.relevance_score,
                    "completeness": cognitive_answer.answer_quality_metrics.completeness_score,
                    "coherence": cognitive_answer.answer_quality_metrics.coherence_score,
                    "factual_accuracy": cognitive_answer.answer_quality_metrics.factual_accuracy,
                    "neural_confidence": cognitive_answer.answer_quality_metrics.neural_confidence,
                    "cognitive_consistency": cognitive_answer.answer_quality_metrics.cognitive_consistency
                }
            });
            
            let message = format!(
                "{}\n\nConfidence: {:.0}% | Relevance: {:.0}%",
                cognitive_answer.text,
                cognitive_answer.confidence * 100.0,
                cognitive_answer.answer_quality_metrics.relevance_score * 100.0
            );
            
            let suggestions = vec![
                "Try rephrasing your question for different cognitive insights".to_string(),
                "Add more context to improve neural processing".to_string(),
                format!(
                    "Answer generated using {} cognitive patterns in {}ms",
                    cognitive_answer.cognitive_patterns_used.len(),
                    cognitive_answer.processing_time_ms
                ),
            ];
            
            Ok((data, message, suggestions))
        }
        Err(e) => {
            log::error!("Cognitive Q&A failed: {}", e);
            Err(format!("Failed to generate cognitive answer: {}", e))
        }
    }
}

/// Handle ask_question request (legacy version for backward compatibility)
pub async fn handle_ask_question(
    knowledge_engine: &Arc<RwLock<KnowledgeEngine>>,
    usage_stats: &Arc<RwLock<UsageStats>>,
    params: Value,
) -> std::result::Result<(Value, String, Vec<String>), String> {
    let question = params.get("question").and_then(|v| v.as_str())
        .ok_or("Missing required field: question")?;
    let context = params.get("context").and_then(|v| v.as_str());
    let max_results = params.get("max_results")
        .and_then(|v| v.as_u64())
        .unwrap_or(5)
        .min(20) as usize;
    
    if question.is_empty() {
        return Err("Question cannot be empty".to_string());
    }
    
    // Parse the question using enhanced parser
    let intent = QuestionParser::parse_static(question);
    
    if intent.entities.is_empty() {
        return Err("Could not extract meaningful entities from the question".to_string());
    }
    
    let engine = knowledge_engine.read().await;
    let mut all_results = Vec::new();
    
    // Search for facts containing entities from the question
    for entity in &intent.entities {
        // Search as subject
        let subject_query = TripleQuery {
            subject: Some(entity.clone()),
            predicate: None,
            object: None,
            limit: 100,
            min_confidence: 0.0,
            include_chunks: false,
        };
        
        if let Ok(results) = engine.query_triples(subject_query) {
            all_results.extend(results);
        }
        
        // Search as object
        let object_query = TripleQuery {
            subject: None,
            predicate: None,
            object: Some(entity.clone()),
            limit: 100,
            min_confidence: 0.0,
            include_chunks: false,
        };
        
        if let Ok(results) = engine.query_triples(object_query) {
            all_results.extend(results);
        }
    }
    
    // Deduplicate results
    all_results.sort_by(|a, b| a.subject.cmp(&b.subject));
    all_results.dedup();
    
    let _ = update_usage_stats(usage_stats, StatsOperation::ExecuteQuery, 15).await;
    
    // Generate answer using enhanced answer generator
    let answer = AnswerGenerator::generate_answer_static(all_results, intent.clone());
    
    let relevant_facts: Vec<_> = answer.facts.iter().take(max_results).map(|t| {
        json!({
            "subject": &t.subject,
            "predicate": &t.predicate,
            "object": &t.object,
            "confidence": t.confidence
        })
    }).collect();
    
    let data = json!({
        "question": question,
        "context": context,
        "entities": answer.entities,
        "question_type": format!("{:?}", intent.question_type),
        "expected_answer_type": format!("{:?}", intent.expected_answer_type),
        "relevant_facts": relevant_facts,
        "answer": answer.text,
        "confidence": answer.confidence
    });
    
    let message = if answer.facts.is_empty() {
        answer.text.clone()
    } else {
        format!("{}\n\nConfidence: {:.0}%", answer.text, answer.confidence * 100.0)
    };
    
    let suggestions = vec![
        "Try rephrasing your question for different results".to_string(),
        "Use 'find_facts' for more specific searches".to_string(),
        "Add context to disambiguate entities".to_string(),
    ];
    
    Ok((data, message, suggestions))
}

/// Format facts for display
fn format_facts_for_display(triples: &[Triple], max_display: usize) -> String {
    let display_count = triples.len().min(max_display);
    let mut result = String::new();
    
    for (i, triple) in triples.iter().take(display_count).enumerate() {
        result.push_str(&format!("{}. {} {} {}\n", 
            i + 1, triple.subject, triple.predicate, triple.object));
    }
    
    if triples.len() > display_count {
        result.push_str(&format!("... and {} more", triples.len() - display_count));
    }
    
    result
}