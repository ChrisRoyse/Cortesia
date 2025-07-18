//! Advanced query and validation handlers

use crate::core::knowledge_engine::{KnowledgeEngine, TripleQuery, KnowledgeResult};
use crate::core::triple::Triple;
use crate::mcp::llm_friendly_server::query_generation::{
    generate_cypher_query, generate_sparql_query, generate_gremlin_query,
    extract_entities_from_query, estimate_query_complexity
};
use crate::mcp::llm_friendly_server::search_fusion::{fuse_search_results, get_fusion_weights};
use crate::mcp::llm_friendly_server::validation::{
    validate_consistency, validate_completeness, validate_triple
};
use crate::mcp::llm_friendly_server::utils::{update_usage_stats, StatsOperation};
use crate::mcp::llm_friendly_server::types::UsageStats;
use crate::error::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde_json::{json, Value};
use std::collections::HashMap;

/// Handle generate_graph_query request
pub async fn handle_generate_graph_query(
    knowledge_engine: &Arc<RwLock<KnowledgeEngine>>,
    usage_stats: &Arc<RwLock<UsageStats>>,
    params: Value,
) -> std::result::Result<(Value, String, Vec<String>), String> {
    let natural_query = params.get("natural_query").and_then(|v| v.as_str())
        .ok_or("Missing required field: natural_query")?;
    let query_language = params.get("query_language").and_then(|v| v.as_str())
        .unwrap_or("cypher");
    let include_explanation = params.get("include_explanation")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    
    if natural_query.is_empty() {
        return Err("natural_query cannot be empty".to_string());
    }
    
    // Validate query language
    if !["cypher", "sparql", "gremlin"].contains(&query_language) {
        return Err("Invalid query_language. Must be one of: cypher, sparql, gremlin".to_string());
    }
    
    // Generate the query
    let (query, explanation) = match query_language {
        "cypher" => generate_cypher_query(natural_query, include_explanation)?,
        "sparql" => generate_sparql_query(natural_query, include_explanation)?,
        "gremlin" => generate_gremlin_query(natural_query, include_explanation)?,
        _ => unreachable!(),
    };
    
    let _ = update_usage_stats(usage_stats, StatsOperation::ExecuteQuery, 5).await;
    
    // Extract entities for additional context
    let entities = extract_entities_from_query(natural_query);
    let complexity = estimate_query_complexity(&query);
    
    let data = json!({
        "natural_query": natural_query,
        "query_language": query_language,
        "generated_query": query,
        "explanation": explanation,
        "extracted_entities": entities,
        "complexity_score": complexity,
        "executable": true
    });
    
    let message = format!(
        "{} Query:\n```{}\n{}\n```{}",
        match query_language {
            "cypher" => "Cypher",
            "sparql" => "SPARQL",
            "gremlin" => "Gremlin",
            _ => "Unknown",
        },
        query_language,
        query,
        if let Some(exp) = &explanation {
            format!("\n\nExplanation: {}", exp)
        } else {
            String::new()
        }
    );
    
    let suggestions = vec![
        format!("This query can be executed in {} databases", match query_language {
            "cypher" => "Neo4j",
            "sparql" => "RDF/SPARQL endpoints like Blazegraph",
            "gremlin" => "TinkerPop-compatible databases",
            _ => "compatible",
        }),
        "Start with simple queries and gradually increase complexity".to_string(),
        "Use the generated query as a template for similar queries".to_string(),
    ];
    
    Ok((data, message, suggestions))
}

/// Handle hybrid_search request
pub async fn handle_hybrid_search(
    knowledge_engine: &Arc<RwLock<KnowledgeEngine>>,
    usage_stats: &Arc<RwLock<UsageStats>>,
    params: Value,
) -> std::result::Result<(Value, String, Vec<String>), String> {
    let query = params.get("query").and_then(|v| v.as_str())
        .ok_or("Missing required field: query")?;
    let search_type = params.get("search_type").and_then(|v| v.as_str())
        .unwrap_or("hybrid");
    let filters = params.get("filters");
    let limit = params.get("limit")
        .and_then(|v| v.as_u64())
        .unwrap_or(10)
        .min(50) as usize;
    
    if query.is_empty() {
        return Err("Query cannot be empty".to_string());
    }
    
    // Validate search type
    if !["semantic", "structural", "keyword", "hybrid"].contains(&search_type) {
        return Err("Invalid search_type. Must be one of: semantic, structural, keyword, hybrid".to_string());
    }
    
    let engine = knowledge_engine.read().await;
    
    // Perform different types of searches
    let semantic_results = if ["semantic", "hybrid"].contains(&search_type) {
        perform_semantic_search(&*engine, query, limit).await?
    } else {
        vec![]
    };
    
    let structural_results = if ["structural", "hybrid"].contains(&search_type) {
        perform_structural_search(&*engine, query, limit).await?
    } else {
        vec![]
    };
    
    let keyword_results = if ["keyword", "hybrid"].contains(&search_type) {
        perform_keyword_search(&*engine, query, limit).await?
    } else {
        vec![]
    };
    
    // Fuse results if hybrid
    let final_results = if search_type == "hybrid" {
        let weights = get_fusion_weights(search_type);
        let fused = fuse_search_results(
            semantic_results,
            structural_results,
            keyword_results,
            Some(weights),
        ).await?;
        
        fused.into_iter()
            .take(limit)
            .map(|r| r.result)
            .collect()
    } else {
        // Return the appropriate single search type results
        match search_type {
            "semantic" => semantic_results,
            "structural" => structural_results,
            "keyword" => keyword_results,
            _ => vec![],
        }
    };
    
    let _ = update_usage_stats(usage_stats, StatsOperation::ExecuteQuery, 30).await;
    
    let results_data: Vec<_> = final_results.iter().enumerate().flat_map(|(i, result)| {
        result.triples.iter().enumerate().map(move |(j, triple)| {
            json!({
                "rank": i * 100 + j + 1,
                "type": "triple",
                "subject": triple.subject,
                "predicate": triple.predicate,
                "object": triple.object
            })
        })
    }).collect();
    
    let data = json!({
        "query": query,
        "search_type": search_type,
        "results": results_data,
        "result_count": final_results.len(),
        "filters_applied": filters.is_some()
    });
    
    let message = format!(
        "Found {} results ({} search):\n\n{}",
        final_results.len(),
        search_type,
        format_search_results(&final_results, 5)
    );
    
    let suggestions = vec![
        "Try different search types for different perspectives".to_string(),
        "Use filters to narrow down results".to_string(),
        "Hybrid search combines multiple strategies for best results".to_string(),
    ];
    
    Ok((data, message, suggestions))
}

/// Handle validate_knowledge request
pub async fn handle_validate_knowledge(
    knowledge_engine: &Arc<RwLock<KnowledgeEngine>>,
    usage_stats: &Arc<RwLock<UsageStats>>,
    params: Value,
) -> std::result::Result<(Value, String, Vec<String>), String> {
    let validation_type = params.get("validation_type").and_then(|v| v.as_str())
        .unwrap_or("all");
    let entity = params.get("entity").and_then(|v| v.as_str());
    let fix_issues = params.get("fix_issues")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    
    // Validate validation type
    if !["consistency", "conflicts", "quality", "completeness", "all"].contains(&validation_type) {
        return Err("Invalid validation_type. Must be one of: consistency, conflicts, quality, completeness, all".to_string());
    }
    
    let engine = knowledge_engine.read().await;
    let mut validation_results = HashMap::new();
    
    // Get all triples for validation
    let query = if let Some(e) = entity {
        TripleQuery {
            subject: Some(e.to_string()),
            predicate: None,
            object: None,
            limit: 100,
            min_confidence: 0.0,
            include_chunks: false,
        }
    } else {
        TripleQuery {
            subject: None,
            predicate: None,
            object: None,
            limit: 100,
            min_confidence: 0.0,
            include_chunks: false,
        }
    };
    
    let triples = engine.query_triples(query)
        .map_err(|e| format!("Failed to query triples: {}", e))?;
    
    // Perform validations
    if ["consistency", "all"].contains(&validation_type) {
        let consistency_result = validate_consistency(&triples.triples, &triples.triples).await
            .map_err(|e| format!("Consistency validation failed: {}", e))?;
        validation_results.insert("consistency", json!({
            "passed": consistency_result.is_valid,
            "confidence": consistency_result.confidence,
            "issues": consistency_result.conflicts
        }));
    }
    
    if ["conflicts", "all"].contains(&validation_type) {
        // Check for conflicts (using consistency check)
        let conflicts_result = validate_consistency(&triples.triples, &triples.triples).await
            .map_err(|e| format!("Conflict validation failed: {}", e))?;
        validation_results.insert("conflicts", json!({
            "found": conflicts_result.conflicts.len(),
            "conflicts": conflicts_result.conflicts
        }));
    }
    
    if ["quality", "all"].contains(&validation_type) {
        // Quality checks
        let mut quality_score = 1.0;
        let mut quality_issues = Vec::new();
        
        for triple in &triples.triples {
            let triple_validation = validate_triple(triple).await
                .map_err(|e| format!("Triple validation failed: {}", e))?;
            quality_score *= triple_validation.confidence;
            quality_issues.extend(triple_validation.validation_notes);
        }
        
        validation_results.insert("quality", json!({
            "score": (quality_score * 10.0).min(10.0),
            "issues": quality_issues
        }));
    }
    
    if ["completeness", "all"].contains(&validation_type) && entity.is_some() {
        let missing = validate_completeness(entity.unwrap(), &triples.triples).await
            .map_err(|e| format!("Completeness validation failed: {}", e))?;
        validation_results.insert("completeness", json!({
            "missing_info": missing,
            "is_complete": missing.is_empty()
        }));
    }
    
    let _ = update_usage_stats(usage_stats, StatsOperation::ExecuteQuery, 40).await;
    
    let data = json!({
        "validation_type": validation_type,
        "entity": entity,
        "results": validation_results,
        "fix_issues": fix_issues,
        "timestamp": chrono::Utc::now().to_rfc3339()
    });
    
    let message = format!(
        "Validation Results{}:\n\n{}",
        if let Some(e) = entity { format!(" for {}", e) } else { String::new() },
        format_validation_results(&validation_results)
    );
    
    let suggestions = vec![
        "Run validation periodically to maintain data quality".to_string(),
        "Focus on fixing high-priority issues first".to_string(),
        "Use fix_issues=true with caution, review all changes".to_string(),
    ];
    
    Ok((data, message, suggestions))
}

/// Perform semantic search
async fn perform_semantic_search(
    engine: &KnowledgeEngine,
    query: &str,
    limit: usize,
) -> Result<Vec<KnowledgeResult>> {
    // Simplified semantic search - in practice would use embeddings
    let keywords = extract_entities_from_query(query);
    let mut results = Vec::new();
    
    for keyword in keywords {
        let triple_query = TripleQuery {
            subject: Some(keyword.clone()),
            predicate: None,
            object: None,
            limit: 100,
            min_confidence: 0.0,
            include_chunks: false,
        };
        
        if let Ok(triples) = engine.query_triples(triple_query) {
            results.push(triples);
        }
    }
    
    results.truncate(limit);
    Ok(results)
}

/// Perform structural search
async fn perform_structural_search(
    engine: &KnowledgeEngine,
    query: &str,
    limit: usize,
) -> Result<Vec<KnowledgeResult>> {
    // Simplified structural search - look for graph patterns
    let query_lower = query.to_lowercase();
    let mut results = Vec::new();
    
    // Look for specific structural patterns
    if query_lower.contains("connected") || query_lower.contains("related") {
        // Find highly connected entities
        let all_triples = engine.query_triples(
            TripleQuery {
                subject: None,
                predicate: None,
                object: None,
                limit: 100,
                min_confidence: 0.0,
                include_chunks: false,
            }
        )?;
        
        results.push(all_triples);
    }
    
    results.truncate(limit);
    Ok(results)
}

/// Perform keyword search
async fn perform_keyword_search(
    engine: &KnowledgeEngine,
    query: &str,
    limit: usize,
) -> Result<Vec<KnowledgeResult>> {
    // Simple keyword matching
    let keywords: Vec<&str> = query.split_whitespace().collect();
    let mut results = Vec::new();
    
    let all_triples = engine.query_triples(
        TripleQuery {
            subject: None,
            predicate: None,
            object: None,
            limit: 1000,
            min_confidence: 0.0,
            include_chunks: false,
        }
    )?;
    
    let matching_triples: Vec<Triple> = all_triples.triples.into_iter()
        .filter(|triple| {
            let triple_text = format!("{} {} {}", triple.subject, triple.predicate, triple.object).to_lowercase();
            keywords.iter().any(|k| triple_text.contains(&k.to_lowercase()))
        })
        .take(limit)
        .collect();
    
    if !matching_triples.is_empty() {
        results.push(KnowledgeResult {
            nodes: Vec::new(),
            triples: matching_triples,
            entity_context: std::collections::HashMap::new(),
            query_time_ms: 0,
            total_found: 0,
        });
    }
    
    Ok(results)
}

/// Format search results for display
fn format_search_results(results: &[KnowledgeResult], max_display: usize) -> String {
    let mut output = String::new();
    let mut count = 0;
    
    for result in results.iter() {
        for triple in &result.triples {
            if count >= max_display {
                break;
            }
            output.push_str(&format!("{}. {} {} {}\n", count + 1, triple.subject, triple.predicate, triple.object));
            count += 1;
        }
        if count >= max_display {
            break;
        }
    }
    
    if count > max_display {
        output.push_str(&format!("\n... and {} more results", count - max_display));
    }
    
    output
}

/// Format validation results for display
fn format_validation_results(results: &HashMap<&str, Value>) -> String {
    let mut output = String::new();
    
    if let Some(consistency) = results.get("consistency") {
        output.push_str(&format!("**Consistency**: {}\n",
            if consistency["passed"].as_bool().unwrap_or(false) { "✓ Passed" } else { "✗ Failed" }
        ));
        if let Some(issues) = consistency["issues"].as_array() {
            for issue in issues {
                output.push_str(&format!("  - {}\n", issue.as_str().unwrap_or("")));
            }
        }
    }
    
    if let Some(conflicts) = results.get("conflicts") {
        let count = conflicts["found"].as_u64().unwrap_or(0);
        output.push_str(&format!("\n**Conflicts**: {} issues found\n", count));
        if let Some(conflict_list) = conflicts["conflicts"].as_array() {
            for (i, conflict) in conflict_list.iter().take(3).enumerate() {
                output.push_str(&format!("  {}. {}\n", i + 1, conflict.as_str().unwrap_or("")));
            }
        }
    }
    
    if let Some(quality) = results.get("quality") {
        let score = quality["score"].as_f64().unwrap_or(0.0);
        output.push_str(&format!("\n**Quality**: {} (score: {:.1}/10)\n",
            if score > 8.0 { "✓ Good" } else if score > 6.0 { "⚠ Fair" } else { "✗ Poor" },
            score
        ));
    }
    
    if let Some(completeness) = results.get("completeness") {
        let is_complete = completeness["is_complete"].as_bool().unwrap_or(false);
        output.push_str(&format!("\n**Completeness**: {}\n",
            if is_complete { "✓ Complete" } else { "⚠ Could improve" }
        ));
        if let Some(missing) = completeness["missing_info"].as_array() {
            for info in missing.iter().take(3) {
                output.push_str(&format!("  - {}\n", info.as_str().unwrap_or("")));
            }
        }
    }
    
    output
}