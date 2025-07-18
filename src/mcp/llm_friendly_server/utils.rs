//! Utility functions for the LLM-friendly MCP server

use crate::core::knowledge_engine::MemoryStats;
use crate::mcp::llm_friendly_server::types::UsageStats;
use crate::error::Result;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Update usage statistics after an operation
pub async fn update_usage_stats(
    stats: &Arc<RwLock<UsageStats>>,
    operation: StatsOperation,
    response_time_ms: u64,
) -> Result<()> {
    let mut usage_stats = stats.write().await;
    
    usage_stats.total_operations += 1;
    
    match operation {
        StatsOperation::StoreTriple => usage_stats.triples_stored += 1,
        StatsOperation::StoreChunk => usage_stats.chunks_stored += 1,
        StatsOperation::ExecuteQuery => usage_stats.queries_executed += 1,
        StatsOperation::CacheHit => usage_stats.cache_hits += 1,
        StatsOperation::CacheMiss => usage_stats.cache_misses += 1,
    }
    
    // Update rolling average response time
    let alpha = 0.1; // Exponential moving average factor
    usage_stats.avg_response_time_ms = 
        alpha * response_time_ms as f64 + (1.0 - alpha) * usage_stats.avg_response_time_ms;
    
    Ok(())
}

/// Calculate memory efficiency score
pub fn calculate_efficiency_score(memory_stats: &MemoryStats) -> f64 {
    if memory_stats.total_bytes == 0 {
        return 1.0;
    }
    
    let entities_per_mb = memory_stats.entity_count as f64 / 
        (memory_stats.total_bytes as f64 / 1_048_576.0);
    
    let relationships_per_mb = memory_stats.relationship_count as f64 / 
        (memory_stats.total_bytes as f64 / 1_048_576.0);
    
    // Normalize scores (assuming good efficiency is 1000+ items per MB)
    let entity_efficiency = (entities_per_mb / 1000.0).min(1.0);
    let relationship_efficiency = (relationships_per_mb / 2000.0).min(1.0);
    
    (entity_efficiency * 0.5 + relationship_efficiency * 0.5).max(0.0).min(1.0)
}

/// Generate helpful information based on the operation
pub fn generate_helpful_info(method: &str) -> String {
    match method {
        "store_fact" => {
            "Tip: Use consistent naming conventions for entities. \
             Consider using 'find_facts' to check for existing similar facts first."
        }
        "store_knowledge" => {
            "Tip: Break long texts into logical chunks. \
             The system will automatically extract entities and relationships."
        }
        "find_facts" => {
            "Tip: Use wildcards by leaving fields empty. \
             Try 'get_suggestions' for ideas on what to search for."
        }
        "ask_question" => {
            "Tip: Be specific in your questions. \
             Add context to disambiguate entities with similar names."
        }
        "explore_connections" => {
            "Tip: Start with max_depth=2 for faster results. \
             Increase depth carefully as it exponentially increases search space."
        }
        "get_suggestions" => {
            "Tip: Use 'knowledge_gaps' to find missing information. \
             'potential_connections' helps discover implicit relationships."
        }
        "generate_graph_query" => {
            "Tip: Start with simple queries to learn the syntax. \
             The generated queries work with standard graph databases."
        }
        "hybrid_search" => {
            "Tip: Use filters to narrow results. \
             'semantic' search is best for concepts, 'structural' for graph patterns."
        }
        "validate_knowledge" => {
            "Tip: Run validation periodically to maintain quality. \
             Use fix_issues=true carefully and review changes."
        }
        _ => "Tip: Check the tool documentation for detailed usage examples."
    }.to_string()
}

/// Generate error help based on the operation
pub fn generate_error_help(method: &str) -> String {
    match method {
        "store_fact" => {
            "Make sure all three fields (subject, predicate, object) are provided. \
             Check field length limits: subject/object max 128 chars, predicate max 64 chars."
        }
        "store_knowledge" => {
            "Ensure 'content' and 'title' are provided. \
             Content limit is 50KB. Break larger texts into multiple chunks."
        }
        "find_facts" => {
            "At least one of subject, predicate, or object must be specified. \
             Check spelling and capitalization of entity names."
        }
        "ask_question" => {
            "Question cannot be empty. \
             Try rephrasing if you're not getting good results."
        }
        "explore_connections" => {
            "start_entity is required. \
             Entity names must match exactly as stored."
        }
        "generate_graph_query" => {
            "natural_query is required. \
             Specify a clear intent (find, connect, path, etc.)."
        }
        _ => "Check that all required parameters are provided and within limits."
    }.to_string()
}

/// Generate error suggestions based on the operation
pub fn generate_error_suggestions(method: &str) -> Vec<String> {
    match method {
        "store_fact" => vec![
            "Use 'find_facts' first to check if the fact already exists".to_string(),
            "Ensure entity names are consistent (e.g., 'Einstein' not 'einstein')".to_string(),
            "Keep predicates simple (e.g., 'born_in' not 'was_born_in_the_city_of')".to_string(),
        ],
        "find_facts" => vec![
            "Try searching with just one field at a time".to_string(),
            "Use 'ask_question' for natural language queries".to_string(),
            "Check 'get_stats' to see what entities exist".to_string(),
        ],
        "ask_question" => vec![
            "Make your question more specific".to_string(),
            "Try 'find_facts' for exact matches".to_string(),
            "Use 'explore_connections' to understand relationships".to_string(),
        ],
        _ => vec![
            "Check the examples in the tool definition".to_string(),
            "Try a simpler query first".to_string(),
            "Use 'get_suggestions' for ideas".to_string(),
        ],
    }
}

/// Generate suggestions based on current state
pub fn generate_suggestions(suggestion_type: &str, focus_area: Option<&str>) -> Vec<String> {
    match suggestion_type {
        "missing_facts" => vec![
            "Consider adding birth/death dates for people".to_string(),
            "Add location information for places and events".to_string(),
            "Include temporal information (when things happened)".to_string(),
            "Add categorical information (type, category, genre)".to_string(),
            "Include quantitative data (size, population, value)".to_string(),
        ],
        "interesting_questions" => vec![
            "What are the connections between X and Y?".to_string(),
            "Who influenced whom in this field?".to_string(),
            "What events happened in the same time period?".to_string(),
            "What are the most connected entities?".to_string(),
            "What patterns emerge from the relationships?".to_string(),
        ],
        "potential_connections" => vec![
            "People who lived in the same era might have met".to_string(),
            "Entities in the same location might be related".to_string(),
            "Similar concepts might have common origins".to_string(),
            "Events in sequence might have causal relationships".to_string(),
            "Entities with similar properties might be grouped".to_string(),
        ],
        "knowledge_gaps" => vec![
            "Missing biographical information for people".to_string(),
            "Incomplete temporal sequences".to_string(),
            "Entities without categorical classification".to_string(),
            "Relationships without confidence scores".to_string(),
            "Facts without source attribution".to_string(),
        ],
        _ => vec!["No specific suggestions available".to_string()],
    }
}

#[derive(Debug, Clone, Copy)]
pub enum StatsOperation {
    StoreTriple,
    StoreChunk,
    ExecuteQuery,
    CacheHit,
    CacheMiss,
}