//! Query-related request handlers

use crate::core::triple::Triple;
use crate::core::knowledge_engine::{KnowledgeEngine, TripleQuery};
use crate::mcp::llm_friendly_server::utils::{update_usage_stats, StatsOperation};
use crate::mcp::llm_friendly_server::types::UsageStats;
use crate::error::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde_json::{json, Value};

/// Handle find_facts request
pub async fn handle_find_facts(
    knowledge_engine: &Arc<RwLock<KnowledgeEngine>>,
    usage_stats: &Arc<RwLock<UsageStats>>,
    params: Value,
) -> std::result::Result<(Value, String, Vec<String>), String> {
    let subject = params.get("subject").and_then(|v| v.as_str());
    let predicate = params.get("predicate").and_then(|v| v.as_str());
    let object = params.get("object").and_then(|v| v.as_str());
    let limit = params.get("limit")
        .and_then(|v| v.as_u64())
        .unwrap_or(10)
        .min(100) as usize;
    
    // At least one field must be specified
    if subject.is_none() && predicate.is_none() && object.is_none() {
        return Err("At least one of subject, predicate, or object must be specified".to_string());
    }
    
    // Build query
    let query = TripleQuery {
        subject: subject.map(|s| s.to_string()),
        predicate: predicate.map(|p| p.to_string()),
        object: object.map(|o| o.to_string()),
        confidence_threshold: None,
    };
    
    let engine = knowledge_engine.read().await;
    match engine.query_triples(query) {
        Ok(triples) => {
            let _ = update_usage_stats(usage_stats, StatsOperation::ExecuteQuery, 10).await;
            
            let facts: Vec<_> = triples.iter().map(|t| {
                json!({
                    "subject": &t.subject,
                    "predicate": &t.predicate,
                    "object": &t.object,
                    "confidence": 1.0 // Would come from metadata
                })
            }).collect();
            
            let data = json!({
                "facts": facts,
                "count": triples.len(),
                "limit": limit,
                "query": {
                    "subject": subject,
                    "predicate": predicate,
                    "object": object
                }
            });
            
            let message = if triples.is_empty() {
                "No facts found matching your query".to_string()
            } else {
                format!("Found {} fact{}:\n{}", 
                    triples.len(),
                    if triples.len() == 1 { "" } else { "s" },
                    format_facts_for_display(&triples, 5)
                )
            };
            
            let suggestions = if triples.is_empty() {
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

/// Handle ask_question request
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
    
    // Extract key terms from question (simplified)
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
            confidence_threshold: None,
        };
        
        if let Ok(results) = engine.query_triples(subject_query) {
            all_results.extend(results);
        }
        
        // Search as object
        let object_query = TripleQuery {
            subject: None,
            predicate: None,
            object: Some(term.clone()),
            confidence_threshold: None,
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
        "answer": generate_answer(&all_results, question)
    });
    
    let message = if all_results.is_empty() {
        "No relevant information found for your question".to_string()
    } else {
        format!("Based on the knowledge graph:\n\n{}\n\nFound {} relevant fact{}",
            generate_answer(&all_results, question),
            all_results.len(),
            if all_results.len() == 1 { "" } else { "s" }
        )
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
    let question_lower = question.to_lowercase();
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
    
    score.min(1.0)
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