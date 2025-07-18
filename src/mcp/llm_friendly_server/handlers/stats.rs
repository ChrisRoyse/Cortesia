//! Statistics and performance handlers

use crate::core::knowledge_engine::{KnowledgeEngine, TripleQuery, MemoryStats};
use crate::mcp::llm_friendly_server::utils::{update_usage_stats, calculate_efficiency_score, StatsOperation};
use crate::mcp::llm_friendly_server::types::UsageStats;
use crate::error::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde_json::{json, Value};
use std::collections::HashMap;

/// Handle get_stats request
pub async fn handle_get_stats(
    knowledge_engine: &Arc<RwLock<KnowledgeEngine>>,
    usage_stats: &Arc<RwLock<UsageStats>>,
    params: Value,
) -> std::result::Result<(Value, String, Vec<String>), String> {
    let include_details = params.get("include_details")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    
    let engine = knowledge_engine.read().await;
    let usage = usage_stats.read().await;
    
    // Get basic statistics
    let basic_stats = collect_basic_stats(&*engine).await
        .map_err(|e| format!("Failed to collect statistics: {}", e))?;
    
    let memory_stats = get_memory_stats(&*engine).await;
    let efficiency_score = calculate_efficiency_score(&memory_stats);
    
    let mut data = json!({
        "knowledge_graph": {
            "total_facts": basic_stats.total_triples,
            "total_entities": basic_stats.unique_entities,
            "total_relationships": basic_stats.unique_predicates,
            "knowledge_chunks": basic_stats.knowledge_chunks,
            "avg_facts_per_entity": basic_stats.avg_facts_per_entity,
            "density": basic_stats.graph_density
        },
        "memory": {
            "total_bytes": memory_stats.total_bytes,
            "entity_count": memory_stats.entity_count,
            "relationship_count": memory_stats.relationship_count,
            "memory_efficiency": efficiency_score
        },
        "usage": {
            "total_operations": usage.total_operations,
            "triples_stored": usage.triples_stored,
            "chunks_stored": usage.chunks_stored,
            "queries_executed": usage.queries_executed,
            "avg_response_time_ms": usage.avg_response_time_ms,
            "cache_hit_rate": if usage.cache_hits + usage.cache_misses > 0 {
                usage.cache_hits as f64 / (usage.cache_hits + usage.cache_misses) as f64
            } else {
                0.0
            }
        },
        "performance": {
            "memory_efficiency_score": efficiency_score,
            "storage_optimization": calculate_storage_optimization(&memory_stats),
            "query_performance": calculate_query_performance(&usage),
            "overall_health": calculate_overall_health(&basic_stats, &memory_stats, &usage)
        }
    });
    
    // Add detailed breakdown if requested
    if include_details {
        let detailed_stats = collect_detailed_stats(&*engine).await
            .map_err(|e| format!("Failed to collect detailed statistics: {}", e))?;
        
        data["details"] = json!({
            "entity_types": detailed_stats.entity_types,
            "relationship_types": detailed_stats.relationship_types,
            "top_entities": detailed_stats.top_entities,
            "relationship_distribution": detailed_stats.relationship_distribution
        });
    }
    
    let _ = update_usage_stats(usage_stats, StatsOperation::ExecuteQuery, 15).await;
    
    let message = format!(
        "Knowledge Graph Statistics:\n\n\
        📊 **Graph Overview:**\n\
        • Total facts (triples): {}\n\
        • Total entities: {}\n\
        • Total relationship types: {}\n\
        • Knowledge chunks: {}\n\
        • Average facts per entity: {:.1}\n\
        • Graph density: {:.1}%\n\n\
        💾 **Memory Usage:**\n\
        • Total size: {:.1} MB\n\
        • Memory efficiency: {:.1}%\n\
        • Entities per MB: {:.0}\n\n\
        ⚡ **Performance:**\n\
        • Total operations: {}\n\
        • Average response time: {:.1}ms\n\
        • Cache hit rate: {:.1}%\n\
        • Overall health: {:.1}%",
        basic_stats.total_triples,
        basic_stats.unique_entities,
        basic_stats.unique_predicates,
        basic_stats.knowledge_chunks,
        basic_stats.avg_facts_per_entity,
        basic_stats.graph_density * 100.0,
        memory_stats.total_bytes as f64 / 1_048_576.0,
        efficiency_score * 100.0,
        if memory_stats.total_bytes > 0 {
            memory_stats.entity_count as f64 / (memory_stats.total_bytes as f64 / 1_048_576.0)
        } else {
            0.0
        },
        usage.total_operations,
        usage.avg_response_time_ms,
        if usage.cache_hits + usage.cache_misses > 0 {
            usage.cache_hits as f64 / (usage.cache_hits + usage.cache_misses) as f64 * 100.0
        } else {
            0.0
        },
        calculate_overall_health(&basic_stats, &memory_stats, &usage) * 100.0
    );
    
    let suggestions = vec![
        "Monitor memory efficiency regularly".to_string(),
        "Use include_details=true for category breakdowns".to_string(),
        "Cache hit rate shows query optimization effectiveness".to_string(),
    ];
    
    Ok((data, message, suggestions))
}

/// Collect basic statistics about the knowledge graph
async fn collect_basic_stats(engine: &KnowledgeEngine) -> Result<BasicStats> {
    // Get all triples to analyze
    let all_triples = engine.query_triples(
        TripleQuery {
            subject: None,
            predicate: None,
            object: None,
            confidence_threshold: None,
        },
        Some(10000), // Limit for performance
    )?;
    
    let total_triples = all_triples.len();
    
    // Count unique entities (subjects and objects)
    let mut entities = std::collections::HashSet::new();
    let mut predicates = std::collections::HashSet::new();
    
    for triple in &all_triples {
        entities.insert(&triple.subject);
        entities.insert(&triple.object);
        predicates.insert(&triple.predicate);
    }
    
    let unique_entities = entities.len();
    let unique_predicates = predicates.len();
    
    // Count knowledge chunks (simplified - count triples with 'is' -> 'knowledge_chunk')
    let knowledge_chunks = all_triples.iter()
        .filter(|t| t.predicate == "is" && t.object == "knowledge_chunk")
        .count();
    
    // Calculate average facts per entity
    let avg_facts_per_entity = if unique_entities > 0 {
        total_triples as f64 / unique_entities as f64
    } else {
        0.0
    };
    
    // Calculate graph density (simplified)
    let max_possible_edges = unique_entities * (unique_entities - 1);
    let graph_density = if max_possible_edges > 0 {
        total_triples as f64 / max_possible_edges as f64
    } else {
        0.0
    };
    
    Ok(BasicStats {
        total_triples,
        unique_entities,
        unique_predicates,
        knowledge_chunks,
        avg_facts_per_entity,
        graph_density,
    })
}

/// Collect detailed statistics for breakdown
async fn collect_detailed_stats(engine: &KnowledgeEngine) -> Result<DetailedStats> {
    let all_triples = engine.query_triples(
        TripleQuery {
            subject: None,
            predicate: None,
            object: None,
            confidence_threshold: None,
        },
        Some(5000),
    )?;
    
    // Count entity types
    let mut entity_types = HashMap::new();
    for triple in &all_triples {
        if triple.predicate == "is" {
            *entity_types.entry(triple.object.clone()).or_insert(0) += 1;
        }
    }
    
    // Count relationship types
    let mut relationship_types = HashMap::new();
    for triple in &all_triples {
        *relationship_types.entry(triple.predicate.clone()).or_insert(0) += 1;
    }
    
    // Find top entities by number of connections
    let mut entity_connections = HashMap::new();
    for triple in &all_triples {
        *entity_connections.entry(triple.subject.clone()).or_insert(0) += 1;
        *entity_connections.entry(triple.object.clone()).or_insert(0) += 1;
    }
    
    let mut top_entities: Vec<_> = entity_connections.into_iter().collect();
    top_entities.sort_by(|a, b| b.1.cmp(&a.1));
    top_entities.truncate(10);
    
    // Calculate relationship distribution
    let total_relationships = all_triples.len();
    let relationship_distribution: HashMap<String, f64> = relationship_types.iter()
        .map(|(k, &v)| (k.clone(), v as f64 / total_relationships as f64))
        .collect();
    
    Ok(DetailedStats {
        entity_types,
        relationship_types,
        top_entities,
        relationship_distribution,
    })
}

/// Get memory statistics (simplified)
async fn get_memory_stats(engine: &KnowledgeEngine) -> MemoryStats {
    // In a real implementation, this would get actual memory usage
    // For now, we'll estimate based on entity/relationship counts
    
    let all_triples = engine.query_triples(
        TripleQuery {
            subject: None,
            predicate: None,
            object: None,
            confidence_threshold: None,
        },
        Some(10000),
    ).unwrap_or_else(|_| KnowledgeResult {
        nodes: Vec::new(),
        triples: Vec::new(),
        entity_context: std::collections::HashMap::new(),
        query_time_ms: 0,
        total_found: 0,
    });
    
    let estimated_bytes = all_triples.len() * 200; // Rough estimate: 200 bytes per triple
    
    let mut entities = std::collections::HashSet::new();
    for triple in &all_triples {
        entities.insert(&triple.subject);
        entities.insert(&triple.object);
    }
    
    MemoryStats {
        total_bytes: estimated_bytes,
        entity_count: entities.len(),
        relationship_count: all_triples.len(),
    }
}

/// Calculate storage optimization score
fn calculate_storage_optimization(memory_stats: &MemoryStats) -> f64 {
    // Higher entity density = better optimization
    if memory_stats.total_bytes == 0 {
        return 1.0;
    }
    
    let entities_per_byte = memory_stats.entity_count as f64 / memory_stats.total_bytes as f64;
    
    // Normalize to 0-1 scale (assuming 0.01 entities per byte is excellent)
    (entities_per_byte / 0.01).min(1.0)
}

/// Calculate query performance score
fn calculate_query_performance(usage_stats: &UsageStats) -> f64 {
    // Lower response time = better performance
    if usage_stats.avg_response_time_ms <= 10.0 {
        1.0
    } else if usage_stats.avg_response_time_ms <= 100.0 {
        1.0 - (usage_stats.avg_response_time_ms - 10.0) / 90.0 * 0.5
    } else {
        0.5 - (usage_stats.avg_response_time_ms - 100.0) / 400.0 * 0.5
    }.max(0.0)
}

/// Calculate overall health score
fn calculate_overall_health(
    basic_stats: &BasicStats,
    memory_stats: &MemoryStats,
    usage_stats: &UsageStats,
) -> f64 {
    let size_score = if basic_stats.total_triples > 100 { 1.0 } else { 0.5 };
    let efficiency_score = calculate_efficiency_score(memory_stats) as f64;
    let performance_score = calculate_query_performance(usage_stats);
    
    (size_score * 0.3 + efficiency_score * 0.4 + performance_score * 0.3).min(1.0)
}

#[derive(Debug)]
struct BasicStats {
    total_triples: usize,
    unique_entities: usize,
    unique_predicates: usize,
    knowledge_chunks: usize,
    avg_facts_per_entity: f64,
    graph_density: f64,
}

#[derive(Debug)]
struct DetailedStats {
    entity_types: HashMap<String, usize>,
    relationship_types: HashMap<String, usize>,
    top_entities: Vec<(String, usize)>,
    relationship_distribution: HashMap<String, f64>,
}