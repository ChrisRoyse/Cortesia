//! Query-related request handlers

use crate::core::triple::Triple;
use crate::core::knowledge_engine::KnowledgeEngine;
use crate::core::knowledge_types::TripleQuery;
use crate::mcp::llm_friendly_server::utils::{update_usage_stats, StatsOperation};
use crate::mcp::llm_friendly_server::types::UsageStats;
// TODO: Enhanced storage features temporarily disabled to focus on core warnings cleanup
use std::sync::Arc;
use tokio::sync::RwLock;
use serde_json::{json, Value};

/// Handle find_facts request with enhanced retrieval
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
    
    // TODO: Enhanced retrieval temporarily disabled due to configuration issues
    // if let Ok(enhanced_results) = try_enhanced_find_facts(subject, predicate, object, limit).await {
    //     let _ = update_usage_stats(usage_stats, StatsOperation::ExecuteQuery, 25).await;
    //     return Ok(enhanced_results);
    // }
    
    // Fallback to traditional triple query
    let triple_query = TripleQuery {
        subject: subject.map(|s| s.to_string()),
        predicate: predicate.map(|p| p.to_string()),
        object: object.map(|o| o.to_string()),
        limit: 100,
        min_confidence: 0.0,
        include_chunks: false,
    };
    
    let engine = knowledge_engine.read().await;
    match engine.query_triples(triple_query) {
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
                "fallback_mode": true,
                "query": {
                    "subject": subject,
                    "predicate": predicate,
                    "object": object
                }
            });
            
            let message = if triples.triples.is_empty() {
                "No facts found matching your query (using fallback mode)".to_string()
            } else {
                format!("Found {} fact{} (fallback mode):\n{}", 
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
                    "Enhanced retrieval failed - used basic search".to_string(),
                    "Try 'ask_question' to understand these facts in context".to_string(),
                ]
            };
            
            Ok((data, message, suggestions))
        }
        Err(e) => Err(format!("Query failed: {}", e))
    }
}

/*
/// Try enhanced retrieval for find_facts - TODO: Temporarily disabled
#[allow(dead_code)]
async fn try_enhanced_find_facts(
    subject: Option<&str>,
    predicate: Option<&str>, 
    object: Option<&str>,
    limit: usize,
) -> Result<(Value, String, Vec<String>), String> {
    // Create hierarchical storage engine (would be singleton in production)
    let storage_config = HierarchicalStorageConfig::default();
    let storage_engine = Arc::new(HierarchicalStorageEngine::new(
        MODEL_MANAGER.clone(),
        storage_config,
    ));
    
    // Create retrieval engine
    let retrieval_config = RetrievalConfig {
        embedding_model_id: "minilm_l6_v2".to_string(),
        reasoning_model_id: "smollm2_360m".to_string(),
        max_parallel_searches: 4,
        cache_search_results: true,
        cache_ttl_seconds: 300,
        enable_fuzzy_matching: true,
        fuzzy_threshold: 0.8,
        context_overlap_tokens: 128,
        enable_result_reranking: true,
        reranking_model_id: Some("smollm2_360m".to_string()),
    };
    
    let retrieval_engine = RetrievalEngine::new(
        MODEL_MANAGER.clone(),
        storage_engine,
        retrieval_config,
    );
    
    // Build natural language query from structured parts
    let query_parts = vec![
        subject.map(|s| format!("subject: {}", s)),
        predicate.map(|p| format!("relationship: {}", p)),
        object.map(|o| format!("object: {}", o)),
    ].into_iter().flatten().collect::<Vec<_>>().join(", ");
    
    let natural_query = format!("Find facts where {}", query_parts);
    
    // Create enhanced retrieval query 
    let retrieval_query = RetrievalQuery {
        natural_language_query: natural_query.clone(),
        structured_constraints: Some(StructuredConstraints {
            required_entities: vec![
                subject.unwrap_or("").to_string(),
                object.unwrap_or("").to_string(),
            ].into_iter().filter(|s| !s.is_empty()).collect(),
            required_relationships: predicate.map(|p| vec![p.to_string()]).unwrap_or_default(),
            required_concepts: vec![],
            layer_types: vec![],
            exclude_patterns: vec![],
            metadata_filters: HashMap::new(),
        }),
        retrieval_mode: RetrievalMode::Hybrid,
        max_results: limit,
        min_relevance_score: 0.3,
        enable_multi_hop: true,
        max_reasoning_hops: 2,
        context_window_size: 1000,
        enable_query_expansion: true,
        enable_temporal_filtering: false,
        time_range: None,
    };
    
    // Execute enhanced retrieval
    let result = retrieval_engine.retrieve(retrieval_query).await
        .map_err(|e| format!("Enhanced retrieval failed: {}", e))?;
    
    // Convert results to facts format
    let facts: Vec<_> = result.retrieved_items.iter().map(|item| {
        json!({
            "layer_id": &item.layer_id,
            "content": &item.content,
            "relevance_score": item.relevance_score,
            "match_type": format!("{:?}", item.match_explanation.match_type),
            "matched_keywords": item.match_explanation.matched_keywords,
            "matched_entities": item.match_explanation.matched_entities,
            "reasoning_steps": item.match_explanation.reasoning_steps.len()
        })
    }).collect();
    
    let data = json!({
        "enhanced_results": facts,
        "count": result.retrieved_items.len(),
        "total_matches": result.total_matches,
        "confidence_score": result.confidence_score,
        "retrieval_time_ms": result.retrieval_time_ms,
        "multi_hop_used": result.reasoning_chain.is_some(),
        "query": {
            "natural_language": natural_query,
            "subject": subject,
            "predicate": predicate,
            "object": object
        }
    });
    
    let message = if result.retrieved_items.is_empty() {
        "No relevant information found using enhanced retrieval".to_string()
    } else {
        format!(
            "Enhanced retrieval found {} relevant item{} (confidence: {:.2}, time: {}ms):\n{}",
            result.retrieved_items.len(),
            if result.retrieved_items.len() == 1 { "" } else { "s" },
            result.confidence_score,
            result.retrieval_time_ms,
            result.retrieved_items.iter()
                .take(3)
                .map(|item| format!("• {} (score: {:.2})", 
                    &item.content[..item.content.len().min(100)], 
                    item.relevance_score))
                .collect::<Vec<_>>()
                .join("\n")
        )
    };
    
    let suggestions = if result.retrieved_items.is_empty() {
        vec![
            "Try broadening your search terms".to_string(),
            "Use 'ask_question' for natural language queries".to_string(),
            "Check if the knowledge has been stored recently".to_string(),
        ]
    } else {
        vec![
            "Use 'ask_question' to get natural language explanations".to_string(),
            "Try 'hybrid_search' for more advanced search options".to_string(),
            if result.reasoning_chain.is_some() {
                "Multi-hop reasoning was used to find connections".to_string()
            } else {
                "Try enabling multi-hop reasoning for deeper search".to_string()
            },
        ]
    };
    
    Ok((data, message, suggestions))
}
*/

/// Handle ask_question request with enhanced retrieval
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
    
    // TODO: Enhanced retrieval temporarily disabled due to configuration issues
    // if let Ok(enhanced_results) = try_enhanced_ask_question(question, context, max_results).await {
    //     let _ = update_usage_stats(usage_stats, StatsOperation::ExecuteQuery, 30).await;
    //     return Ok(enhanced_results);
    // }
    
    // Fallback to traditional approach
    let key_terms = extract_key_terms(question);
    
    if key_terms.is_empty() {
        return Err("Could not extract meaningful terms from the question".to_string());
    }
    
    let engine = knowledge_engine.read().await;
    let mut all_results = Vec::new();
    
    // Search for facts containing key terms
    for term in &key_terms {
        // Search as subject
        let subject_query = TripleQuery {
            subject: Some(term.clone()),
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
            object: Some(term.clone()),
            limit: 100,
            min_confidence: 0.0,
            include_chunks: false,
        };
        
        if let Ok(results) = engine.query_triples(object_query) {
            all_results.extend(results);
        }
    }
    
    // Deduplicate and limit results
    all_results.sort_by(|a, b| a.subject.cmp(&b.subject));
    all_results.dedup();
    all_results.truncate(max_results);
    
    let _ = update_usage_stats(usage_stats, StatsOperation::ExecuteQuery, 15).await;
    
    let relevant_facts: Vec<_> = all_results.iter().map(|t| {
        json!({
            "subject": &t.subject,
            "predicate": &t.predicate,
            "object": &t.object,
            "relevance": calculate_relevance(t, question)
        })
    }).collect();
    
    let data = json!({
        "question": question,
        "context": context,
        "key_terms": key_terms,
        "relevant_facts": relevant_facts,
        "answer": generate_answer(&all_results, question),
        "fallback_mode": true
    });
    
    let message = if all_results.is_empty() {
        "No relevant information found for your question (fallback mode)".to_string()
    } else {
        format!("Based on the knowledge graph (fallback mode):\n\n{}\n\nFound {} relevant fact{}",
            generate_answer(&all_results, question),
            all_results.len(),
            if all_results.len() == 1 { "" } else { "s" }
        )
    };
    
    let suggestions = vec![
        "Enhanced retrieval failed - used basic search".to_string(),
        "Try rephrasing your question for different results".to_string(),
        "Use 'find_facts' for more specific searches".to_string(),
    ];
    
    Ok((data, message, suggestions))
}

/*
/// Try enhanced retrieval for ask_question - TODO: Temporarily disabled
#[allow(dead_code)]
async fn try_enhanced_ask_question(
    question: &str,
    context: Option<&str>,
    max_results: usize,
) -> Result<(Value, String, Vec<String>), String> {
    // Create hierarchical storage engine (would be singleton in production)
    let storage_config = HierarchicalStorageConfig::default();
    let storage_engine = Arc::new(HierarchicalStorageEngine::new(
        MODEL_MANAGER.clone(),
        storage_config,
    ));
    
    // Create retrieval engine with question-answering focus
    let retrieval_config = RetrievalConfig {
        embedding_model_id: "minilm_l6_v2".to_string(),
        reasoning_model_id: "smollm2_360m".to_string(),
        max_parallel_searches: 6, // More parallel searches for Q&A
        cache_search_results: true,
        cache_ttl_seconds: 300,
        enable_fuzzy_matching: true,
        fuzzy_threshold: 0.7, // Lower threshold for better Q&A
        context_overlap_tokens: 256, // Larger context for better answers
        enable_result_reranking: true,
        reranking_model_id: Some("smollm2_360m".to_string()),
    };
    
    let retrieval_engine = RetrievalEngine::new(
        MODEL_MANAGER.clone(),
        storage_engine,
        retrieval_config,
    );
    
    // Build comprehensive query with context
    let full_query = if let Some(ctx) = context {
        format!("{}\n\nContext: {}", question, ctx)
    } else {
        question.to_string()
    };
    
    // Create enhanced retrieval query optimized for Q&A
    let retrieval_query = RetrievalQuery {
        natural_language_query: full_query.clone(),
        structured_constraints: None, // Let natural language drive the search
        retrieval_mode: RetrievalMode::Hybrid, // Best for Q&A
        max_results,
        min_relevance_score: 0.2,
        enable_multi_hop: true,
        max_reasoning_hops: 3, // Deep reasoning for complex questions
        context_window_size: 2000,
        enable_query_expansion: true,
        enable_temporal_filtering: false,
        time_range: None,
    };
    
    // Execute enhanced retrieval
    let result = retrieval_engine.retrieve(retrieval_query).await
        .map_err(|e| format!("Enhanced retrieval failed: {}", e))?;
    
    // Generate enhanced answer using retrieved context
    let answer = if result.retrieved_items.is_empty() {
        "I don't have enough information in my knowledge base to answer this question.".to_string()
    } else {
        generate_enhanced_answer(question, &result).await
            .unwrap_or_else(|_| "I found relevant information but couldn't generate a coherent answer.".to_string())
    };
    
    // Collect relevant evidence
    let evidence: Vec<_> = result.retrieved_items.iter().map(|item| {
        json!({
            "layer_id": &item.layer_id,
            "content": &item.content,
            "relevance_score": item.relevance_score,
            "match_explanation": {
                "match_type": format!("{:?}", item.match_explanation.match_type),
                "matched_keywords": item.match_explanation.matched_keywords,
                "matched_entities": item.match_explanation.matched_entities,
                "semantic_similarity": item.match_explanation.semantic_similarity,
                "reasoning_steps": item.match_explanation.reasoning_steps.len()
            },
            "importance_score": item.importance_score
        })
    }).collect();
    
    let data = json!({
        "question": question,
        "context": context,
        "answer": answer,
        "evidence": evidence,
        "enhanced_processing": {
            "total_matches": result.total_matches,
            "confidence_score": result.confidence_score,
            "retrieval_time_ms": result.retrieval_time_ms,
            "multi_hop_used": result.reasoning_chain.is_some(),
            "reasoning_steps": result.reasoning_chain.as_ref()
                .map(|chain| chain.reasoning_steps.len())
                .unwrap_or(0)
        }
    });
    
    let message = if result.retrieved_items.is_empty() {
        "I couldn't find relevant information to answer your question.".to_string()
    } else {
        format!(
            "Based on {} source{} (confidence: {:.2}):\n\n{}\n\n{}",
            result.retrieved_items.len(),
            if result.retrieved_items.len() == 1 { "" } else { "s" },
            result.confidence_score,
            answer,
            if result.reasoning_chain.is_some() {
                format!("\n✓ Used multi-hop reasoning with {} steps", 
                    result.reasoning_chain.as_ref().unwrap().reasoning_steps.len())
            } else {
                String::new()
            }
        )
    };
    
    let suggestions = if result.retrieved_items.is_empty() {
        vec![
            "Try rephrasing your question".to_string(),
            "Check if relevant knowledge has been stored".to_string(),
            "Use 'store_knowledge' to add information first".to_string(),
        ]
    } else {
        vec![
            "Ask follow-up questions for more details".to_string(),
            "Use 'find_facts' to explore specific relationships".to_string(),
            format!("Try asking about: {}", 
                result.retrieved_items.iter()
                    .flat_map(|item| &item.match_explanation.matched_entities)
                    .take(3)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")),
        ]
    };
    
    Ok((data, message, suggestions))
}

/// Generate enhanced answer using retrieved information and reasoning
/// TODO: Temporarily disabled
#[allow(dead_code)]
async fn generate_enhanced_answer(
    question: &str,
    retrieval_result: &crate::enhanced_knowledge_storage::retrieval_system::types::RetrievalResult,
) -> Result<String, String> {
    use crate::enhanced_knowledge_storage::types::{ProcessingTask, ComplexityLevel};
    
    // Collect all relevant content
    let evidence_texts: Vec<String> = retrieval_result.retrieved_items
        .iter()
        .map(|item| format!("• {}", item.content))
        .collect();
    
    if evidence_texts.is_empty() {
        return Ok("No relevant information found.".to_string());
    }
    
    // Create comprehensive prompt for answer generation
    let prompt = format!(
        r#"Based on the following evidence from a knowledge graph, provide a comprehensive answer to the question.

Question: {}

Evidence:
{}

Instructions:
- Synthesize the evidence into a coherent answer
- Be accurate and only use information from the evidence
- If the evidence is insufficient, state what's missing
- Structure your answer clearly
- Cite specific details when relevant

Answer:"#,
        question,
        evidence_texts.join("\n")
    );
    
    // Use model manager to generate answer
    let task = ProcessingTask::new(ComplexityLevel::Medium, &prompt);
    
    match MODEL_MANAGER.process_with_optimal_model(task).await {
        Ok(result) => {
            if result.success && !result.output.trim().is_empty() {
                Ok(result.output.trim().to_string())
            } else {
                Err("Failed to generate answer".to_string())
            }
        }
        Err(e) => Err(format!("Answer generation failed: {}", e))
    }
}
*/

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

/// Extract key terms from a question
fn extract_key_terms(question: &str) -> Vec<String> {
    let mut terms = Vec::new();
    let words: Vec<&str> = question.split_whitespace().collect();
    
    // Extract capitalized words (likely entities)
    for word in &words {
        let clean_word = word.trim_matches(|c: char| !c.is_alphanumeric());
        if clean_word.chars().next().map_or(false, |c| c.is_uppercase()) {
            terms.push(clean_word.to_string());
        }
    }
    
    // Extract quoted phrases
    let mut in_quotes = false;
    let mut current_phrase = String::new();
    
    for char in question.chars() {
        if char == '"' || char == '\'' {
            if in_quotes && !current_phrase.is_empty() {
                terms.push(current_phrase.clone());
                current_phrase.clear();
            }
            in_quotes = !in_quotes;
        } else if in_quotes {
            current_phrase.push(char);
        }
    }
    
    // Look for question keywords and extract following terms
    let _question_lower = question.to_lowercase();
    for (i, word) in words.iter().enumerate() {
        if matches!(word.to_lowercase().as_str(), "who" | "what" | "where" | "when" | "which") {
            if i + 1 < words.len() {
                let next_word = words[i + 1].trim_matches(|c: char| !c.is_alphanumeric());
                if !next_word.is_empty() && !is_stop_word(next_word) {
                    terms.push(next_word.to_string());
                }
            }
        }
    }
    
    // Deduplicate
    terms.sort();
    terms.dedup();
    
    terms
}

/// Check if a word is a stop word
fn is_stop_word(word: &str) -> bool {
    matches!(
        word.to_lowercase().as_str(),
        "is" | "are" | "was" | "were" | "the" | "a" | "an" | "and" | "or" | "but" |
        "in" | "on" | "at" | "to" | "for" | "of" | "with" | "by" | "from" | "about"
    )
}

/// Calculate relevance of a triple to a question
fn calculate_relevance(triple: &Triple, question: &str) -> f32 {
    let question_lower = question.to_lowercase();
    let mut score = 0.0;
    
    // Check if triple components appear in question
    if question_lower.contains(&triple.subject.to_lowercase()) {
        score += 0.4;
    }
    if question_lower.contains(&triple.predicate.to_lowercase()) {
        score += 0.2;
    }
    if question_lower.contains(&triple.object.to_lowercase()) {
        score += 0.4;
    }
    
    f32::min(score, 1.0)
}

/// Generate an answer from relevant facts
fn generate_answer(facts: &[Triple], question: &str) -> String {
    if facts.is_empty() {
        return "I don't have enough information to answer this question.".to_string();
    }
    
    // Simple answer generation based on question type
    let question_lower = question.to_lowercase();
    
    if question_lower.starts_with("what") {
        // Look for "is" relationships
        if let Some(fact) = facts.iter().find(|f| f.predicate == "is") {
            return format!("{} is {}", fact.subject, fact.object);
        }
    } else if question_lower.starts_with("who") {
        // Look for person-related facts
        if let Some(fact) = facts.iter().find(|f| 
            f.predicate == "created" || f.predicate == "invented" || f.predicate == "wrote"
        ) {
            return format!("{} {} {}", fact.subject, fact.predicate, fact.object);
        }
    } else if question_lower.starts_with("where") {
        // Look for location relationships
        if let Some(fact) = facts.iter().find(|f| 
            f.predicate == "located_in" || f.predicate == "from" || f.predicate == "in"
        ) {
            return format!("{} is {} {}", fact.subject, fact.predicate, fact.object);
        }
    }
    
    // Default: return the most relevant facts
    let relevant_facts: Vec<String> = facts.iter()
        .take(3)
        .map(|f| format!("{} {} {}", f.subject, f.predicate, f.object))
        .collect();
    
    relevant_facts.join("; ")
}