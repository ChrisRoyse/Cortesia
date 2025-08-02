//! Temporal query handlers for time travel and versioning

use crate::core::knowledge_engine::KnowledgeEngine;
use crate::mcp::llm_friendly_server::temporal_tracking::{
    query_point_in_time, track_entity_evolution, detect_changes,
    TEMPORAL_INDEX
};
use crate::mcp::llm_friendly_server::database_branching::{
    get_branch_manager, MergeStrategy
};
use crate::mcp::llm_friendly_server::utils::{update_usage_stats, StatsOperation};
use crate::mcp::llm_friendly_server::types::UsageStats;
use crate::versioning::MultiDatabaseVersionManager;
use crate::federation::DatabaseId;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde_json::{json, Value};
use chrono::{DateTime, Utc};

/// Handle time_travel_query request
pub async fn handle_time_travel_query(
    _knowledge_engine: &Arc<RwLock<KnowledgeEngine>>,
    usage_stats: &Arc<RwLock<UsageStats>>,
    params: Value,
) -> std::result::Result<(Value, String, Vec<String>), String> {
    log::debug!("handle_time_travel_query: Starting");
    
    let query_type = params.get("query_type")
        .and_then(|v| v.as_str())
        .unwrap_or("point_in_time");
    
    let entity = params.get("entity")
        .and_then(|v| v.as_str());
    
    let timestamp = params.get("timestamp")
        .and_then(|v| v.as_str())
        .map(|ts| DateTime::parse_from_rfc3339(ts)
            .map_err(|e| format!("Invalid timestamp format: {e}"))
            .map(|dt| dt.with_timezone(&Utc)))
        .transpose()?;
    
    let time_range = params.get("time_range")
        .and_then(|v| v.as_object())
        .map(|range| {
            let start = range.get("start")
                .and_then(|v| v.as_str())
                .map(|ts| DateTime::parse_from_rfc3339(ts)
                    .map(|dt| dt.with_timezone(&Utc)));
            let end = range.get("end")
                .and_then(|v| v.as_str())
                .map(|ts| DateTime::parse_from_rfc3339(ts)
                    .map(|dt| dt.with_timezone(&Utc)));
            
            match (start, end) {
                (Some(Ok(s)), Some(Ok(e))) => Ok(Some((s, e))),
                (Some(Err(e)), _) => Err(format!("Invalid start time: {e}")),
                (_, Some(Err(e))) => Err(format!("Invalid end time: {e}")),
                _ => Ok(None)
            }
        })
        .transpose()?
        .flatten();
    
    let start_time = std::time::Instant::now();
    
    // Execute temporal query based on type
    let result = match query_type {
        "point_in_time" => {
            let entity = entity.ok_or("Entity required for point_in_time query")?;
            let timestamp = timestamp.unwrap_or_else(Utc::now);
            query_point_in_time(&TEMPORAL_INDEX, entity, timestamp).await
        }
        "evolution_tracking" => {
            let entity = entity.ok_or("Entity required for evolution_tracking query")?;
            let (start, end) = time_range.unwrap_or((
                DateTime::from_timestamp(0, 0).unwrap(),
                Utc::now()
            ));
            track_entity_evolution(&TEMPORAL_INDEX, entity, Some(start), Some(end)).await
        }
        "temporal_comparison" => {
            if let Some((start, end)) = time_range {
                detect_changes(&TEMPORAL_INDEX, start, end, entity).await
            } else {
                return Err("Time range required for temporal_comparison".to_string());
            }
        }
        "change_detection" => {
            let (start, end) = time_range.unwrap_or_else(|| {
                let end = Utc::now();
                let start = end - chrono::Duration::days(7);
                (start, end)
            });
            detect_changes(&TEMPORAL_INDEX, start, end, entity).await
        }
        _ => return Err(format!("Unknown query type: {query_type}"))
    };
    
    let query_time = start_time.elapsed();
    
    // Format result data
    let data = json!({
        "query_type": result.query_type,
        "results": result.results,
        "time_range": result.time_range.map(|(s, e)| json!({
            "start": s.to_rfc3339(),
            "end": e.to_rfc3339()
        })),
        "total_changes": result.total_changes,
        "insights": result.insights,
        "metadata": {
            "query_time_ms": query_time.as_millis(),
            "data_points": result.results.len(),
            "temporal_span": result.time_range.map(|(s, e)| {
                let duration = e - s;
                json!({
                    "days": duration.num_days(),
                    "hours": duration.num_hours(),
                    "human_readable": format_duration(duration)
                })
            })
        }
    });
    
    let message = format!(
        "Time Travel Query Results:\n\
        ⏰ Query Type: {}\n\
        📊 Data Points: {}\n\
        📈 Changes Detected: {}\n\
        🕰️ Time Span: {}\n\
        🔍 Key Insights: {}",
        result.query_type,
        result.results.len(),
        result.total_changes,
        result.time_range.map(|(s, e)| {
            let duration = e - s;
            format_duration(duration)
        }).unwrap_or_else(|| "N/A".to_string()),
        result.insights.first().unwrap_or(&"No insights available".to_string())
    );
    
    let suggestions = vec![
        "Use 'evolution_tracking' to see how entities change over time".to_string(),
        "Compare different time periods with 'temporal_comparison'".to_string(),
        "Detect anomalies with 'change_detection' queries".to_string(),
    ];
    
    let _ = update_usage_stats(usage_stats, StatsOperation::ExecuteQuery, 80).await;
    
    Ok((data, message, suggestions))
}

/// Handle create_branch request
pub async fn handle_create_branch(
    _knowledge_engine: &Arc<RwLock<KnowledgeEngine>>,
    usage_stats: &Arc<RwLock<UsageStats>>,
    params: Value,
    _version_manager: Arc<MultiDatabaseVersionManager>,
) -> std::result::Result<(Value, String, Vec<String>), String> {
    let source_db_id = params.get("source_db_id")
        .and_then(|v| v.as_str())
        .ok_or("Missing required field: source_db_id")?;
    
    let branch_name = params.get("branch_name")
        .and_then(|v| v.as_str())
        .ok_or("Missing required field: branch_name")?;
    
    let description = params.get("description")
        .and_then(|v| v.as_str())
        .map(String::from);
    
    // Get or initialize branch manager
    let branch_manager_arc = get_branch_manager().await
        .map_err(|e| format!("Failed to get branch manager: {e}"))?;
    
    let branch_manager_guard = branch_manager_arc.read().await;
    let branch_manager = branch_manager_guard.as_ref()
        .ok_or("Branch manager not initialized")?;
    
    // Create the branch
    let new_db_id = branch_manager.create_branch(
        &DatabaseId::new(source_db_id.to_string()),
        branch_name.to_string(),
        description.clone(),
    ).await
        .map_err(|e| format!("Failed to create branch: {e}"))?;
    
    let _ = update_usage_stats(usage_stats, StatsOperation::ExecuteQuery, 100).await;
    
    let data = json!({
        "branch_name": branch_name,
        "database_id": new_db_id.as_str(),
        "source_database_id": source_db_id,
        "description": description,
        "created_at": Utc::now().to_rfc3339(),
        "status": "active"
    });
    
    let message = format!(
        "Branch Created Successfully:\n\
        🌿 Branch Name: {}\n\
        📁 New Database ID: {}\n\
        🔗 Created from: {}\n\
        📝 Description: {}",
        branch_name,
        new_db_id.as_str(),
        source_db_id,
        description.as_deref().unwrap_or("No description")
    );
    
    let suggestions = vec![
        format!("Switch to the new branch with database_id: {}", new_db_id.as_str()),
        "Compare branches to see differences".to_string(),
        "Merge changes back when ready".to_string(),
    ];
    
    Ok((data, message, suggestions))
}

/// Handle list_branches request
pub async fn handle_list_branches(
    _knowledge_engine: &Arc<RwLock<KnowledgeEngine>>,
    usage_stats: &Arc<RwLock<UsageStats>>,
    _params: Value,
) -> std::result::Result<(Value, String, Vec<String>), String> {
    let branch_manager_arc = get_branch_manager().await
        .map_err(|e| format!("Failed to get branch manager: {e}"))?;
    
    let branch_manager_guard = branch_manager_arc.read().await;
    let branch_manager = branch_manager_guard.as_ref()
        .ok_or("Branch manager not initialized")?;
    
    let branches = branch_manager.list_branches().await
        .map_err(|e| format!("Failed to list branches: {e}"))?;
    
    let _ = update_usage_stats(usage_stats, StatsOperation::ExecuteQuery, 10).await;
    
    let branch_data: Vec<Value> = branches.iter().map(|branch| {
        json!({
            "branch_name": branch.branch_name,
            "database_id": branch.database_id.as_str(),
            "created_at": branch.created_at.to_rfc3339(),
            "created_from": branch.created_from.as_str(),
            "description": branch.description,
            "is_active": branch.is_active
        })
    }).collect();
    
    let data = json!({
        "branches": branch_data,
        "total_branches": branches.len()
    });
    
    let message = format!(
        "Found {} branches:\n\n{}",
        branches.len(),
        branches.iter()
            .take(5)
            .map(|b| format!("🌿 {} ({})", b.branch_name, b.database_id.as_str()))
            .collect::<Vec<_>>()
            .join("\n")
    );
    
    let suggestions = vec![
        "Create new branches for experiments".to_string(),
        "Compare branches to track changes".to_string(),
        "Merge completed work back to main".to_string(),
    ];
    
    Ok((data, message, suggestions))
}

/// Handle compare_branches request
pub async fn handle_compare_branches(
    _knowledge_engine: &Arc<RwLock<KnowledgeEngine>>,
    usage_stats: &Arc<RwLock<UsageStats>>,
    params: Value,
) -> std::result::Result<(Value, String, Vec<String>), String> {
    let branch1 = params.get("branch1")
        .and_then(|v| v.as_str())
        .ok_or("Missing required field: branch1")?;
    
    let branch2 = params.get("branch2")
        .and_then(|v| v.as_str())
        .ok_or("Missing required field: branch2")?;
    
    let branch_manager_arc = get_branch_manager().await
        .map_err(|e| format!("Failed to get branch manager: {e}"))?;
    
    let branch_manager_guard = branch_manager_arc.read().await;
    let branch_manager = branch_manager_guard.as_ref()
        .ok_or("Branch manager not initialized")?;
    
    let comparison = branch_manager.compare_branches(branch1, branch2).await
        .map_err(|e| format!("Failed to compare branches: {e}"))?;
    
    let _ = update_usage_stats(usage_stats, StatsOperation::ExecuteQuery, 50).await;
    
    let data = json!({
        "branch1": comparison.branch1,
        "branch2": comparison.branch2,
        "statistics": {
            "branch1": comparison.stats1,
            "branch2": comparison.stats2
        },
        "differences": {
            "only_in_branch1": comparison.differences.only_in_first,
            "only_in_branch2": comparison.differences.only_in_second,
            "common": comparison.differences.common,
            "sample_differences": comparison.differences.sample_differences
        }
    });
    
    let message = format!(
        "Branch Comparison:\n\
        🌿 {} vs {}\n\
        📊 Branch 1: {} triples, {} nodes\n\
        📊 Branch 2: {} triples, {} nodes\n\
        🔍 Unique to {}: {}\n\
        🔍 Unique to {}: {}\n\
        🤝 Common: {}",
        branch1, branch2,
        comparison.stats1.total_triples, comparison.stats1.total_nodes,
        comparison.stats2.total_triples, comparison.stats2.total_nodes,
        branch1, comparison.differences.only_in_first,
        branch2, comparison.differences.only_in_second,
        comparison.differences.common
    );
    
    let suggestions = vec![
        "Review differences before merging".to_string(),
        "Use sample_differences to inspect changes".to_string(),
        "Consider merge strategy based on differences".to_string(),
    ];
    
    Ok((data, message, suggestions))
}

/// Handle merge_branches request
pub async fn handle_merge_branches(
    _knowledge_engine: &Arc<RwLock<KnowledgeEngine>>,
    usage_stats: &Arc<RwLock<UsageStats>>,
    params: Value,
) -> std::result::Result<(Value, String, Vec<String>), String> {
    let source_branch = params.get("source_branch")
        .and_then(|v| v.as_str())
        .ok_or("Missing required field: source_branch")?;
    
    let target_branch = params.get("target_branch")
        .and_then(|v| v.as_str())
        .ok_or("Missing required field: target_branch")?;
    
    let merge_strategy_str = params.get("merge_strategy")
        .and_then(|v| v.as_str())
        .unwrap_or("accept_source");
    
    let merge_strategy = match merge_strategy_str {
        "accept_source" => MergeStrategy::AcceptSource,
        "accept_target" => MergeStrategy::AcceptTarget,
        "manual" => MergeStrategy::Manual,
        _ => return Err(format!("Invalid merge strategy: {merge_strategy_str}"))
    };
    
    let branch_manager_arc = get_branch_manager().await
        .map_err(|e| format!("Failed to get branch manager: {e}"))?;
    
    let branch_manager_guard = branch_manager_arc.read().await;
    let branch_manager = branch_manager_guard.as_ref()
        .ok_or("Branch manager not initialized")?;
    
    let merge_result = branch_manager.merge_branches(
        source_branch,
        target_branch,
        merge_strategy,
    ).await
        .map_err(|e| format!("Failed to merge branches: {e}"))?;
    
    let _ = update_usage_stats(usage_stats, StatsOperation::ExecuteQuery, 100).await;
    
    let data = json!({
        "source_branch": source_branch,
        "target_branch": target_branch,
        "merge_strategy": merge_strategy_str,
        "result": {
            "success": merge_result.success,
            "triples_added": merge_result.triples_added,
            "triples_removed": merge_result.triples_removed,
            "conflicts_resolved": merge_result.conflicts_resolved,
            "message": merge_result.message
        },
        "timestamp": Utc::now().to_rfc3339()
    });
    
    let message = format!(
        "Merge Results:\n\
        ✅ Status: {}\n\
        🔀 {} → {}\n\
        📝 Strategy: {}\n\
        ➕ Triples Added: {}\n\
        ➖ Triples Removed: {}\n\
        🔧 Conflicts Resolved: {}\n\
        💬 {}",
        if merge_result.success { "Success" } else { "Failed" },
        source_branch, target_branch,
        merge_strategy_str,
        merge_result.triples_added,
        merge_result.triples_removed,
        merge_result.conflicts_resolved,
        merge_result.message
    );
    
    let suggestions = vec![
        "Verify the merge results".to_string(),
        "Consider creating a backup branch before major merges".to_string(),
        "Review any unresolved conflicts".to_string(),
    ];
    
    Ok((data, message, suggestions))
}

/// Format duration for human readability
fn format_duration(duration: chrono::Duration) -> String {
    let days = duration.num_days();
    let hours = duration.num_hours() % 24;
    let minutes = duration.num_minutes() % 60;
    
    if days > 0 {
        format!("{days} days, {hours} hours")
    } else if hours > 0 {
        format!("{hours} hours, {minutes} minutes")
    } else {
        format!("{minutes} minutes")
    }
}