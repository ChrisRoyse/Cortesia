//! Exploration and suggestion request handlers

use crate::core::knowledge_engine::{KnowledgeEngine, TripleQuery};
use crate::mcp::llm_friendly_server::utils::{update_usage_stats, generate_suggestions, StatsOperation};
use crate::mcp::llm_friendly_server::types::UsageStats;
use crate::error::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet, VecDeque};

/// Handle explore_connections request
pub async fn handle_explore_connections(
    knowledge_engine: &Arc<RwLock<KnowledgeEngine>>,
    usage_stats: &Arc<RwLock<UsageStats>>,
    params: Value,
) -> std::result::Result<(Value, String, Vec<String>), String> {
    let start_entity = params.get("start_entity").and_then(|v| v.as_str())
        .ok_or("Missing required field: start_entity")?;
    let end_entity = params.get("end_entity").and_then(|v| v.as_str());
    let max_depth = params.get("max_depth")
        .and_then(|v| v.as_u64())
        .unwrap_or(2)
        .min(4) as usize;
    let relationship_types = params.get("relationship_types")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str())
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
        });
    
    if start_entity.is_empty() {
        return Err("start_entity cannot be empty".to_string());
    }
    
    let engine = knowledge_engine.read().await;
    
    if let Some(target) = end_entity {
        // Find paths between two entities
        let paths = find_paths_between(
            &*engine,
            start_entity,
            target,
            max_depth,
            relationship_types.as_ref(),
        ).await?;
        
        let _ = update_usage_stats(usage_stats, StatsOperation::ExecuteQuery, 25).await;
        
        let path_data: Vec<_> = paths.iter().map(|path| {
            json!({
                "length": path.len() - 1,
                "path": format_path(path),
                "entities": path.iter().map(|(e, _)| e).collect::<Vec<_>>(),
                "relationships": path.windows(2).map(|w| &w[0].1).collect::<Vec<_>>()
            })
        }).collect();
        
        let data = json!({
            "start": start_entity,
            "end": target,
            "paths": path_data,
            "paths_found": paths.len(),
            "max_depth": max_depth
        });
        
        let message = if paths.is_empty() {
            format!("No connection found between '{}' and '{}' within {} steps", 
                start_entity, target, max_depth)
        } else {
            format!("Found {} path{} between '{}' and '{}':\n{}",
                paths.len(),
                if paths.len() == 1 { "" } else { "s" },
                start_entity,
                target,
                format_paths_for_display(&paths, 3)
            )
        };
        
        let suggestions = vec![
            "Try increasing max_depth to find longer paths".to_string(),
            "Use relationship_types to filter by specific connections".to_string(),
        ];
        
        Ok((data, message, suggestions))
    } else {
        // Explore all connections from start entity
        let connections = explore_from_entity(
            &*engine,
            start_entity,
            max_depth,
            relationship_types.as_ref(),
        ).await?;
        
        let _ = update_usage_stats(usage_stats, StatsOperation::ExecuteQuery, 20).await;
        
        let connection_data: Vec<_> = connections.iter().map(|(entity, info)| {
            json!({
                "entity": entity,
                "distance": info.distance,
                "relationship": info.relationship,
                "path": info.path
            })
        }).collect();
        
        let data = json!({
            "start": start_entity,
            "connections": connection_data,
            "total_connected": connections.len(),
            "max_depth": max_depth
        });
        
        let message = format!(
            "Found {} entities connected to '{}' within {} steps:\n{}",
            connections.len(),
            start_entity,
            max_depth,
            format_connections_for_display(&connections, 10)
        );
        
        let suggestions = vec![
            "Specify end_entity to find specific paths".to_string(),
            "Use max_depth=1 to see only direct connections".to_string(),
        ];
        
        Ok((data, message, suggestions))
    }
}

/// Handle get_suggestions request
pub async fn handle_get_suggestions(
    knowledge_engine: &Arc<RwLock<KnowledgeEngine>>,
    usage_stats: &Arc<RwLock<UsageStats>>,
    params: Value,
) -> std::result::Result<(Value, String, Vec<String>), String> {
    let suggestion_type = params.get("suggestion_type").and_then(|v| v.as_str())
        .ok_or("Missing required field: suggestion_type")?;
    let focus_area = params.get("focus_area").and_then(|v| v.as_str());
    let limit = params.get("limit")
        .and_then(|v| v.as_u64())
        .unwrap_or(5)
        .min(10) as usize;
    
    // Validate suggestion type
    if !["missing_facts", "interesting_questions", "potential_connections", "knowledge_gaps"]
        .contains(&suggestion_type) {
        return Err("Invalid suggestion_type. Must be one of: missing_facts, interesting_questions, potential_connections, knowledge_gaps".to_string());
    }
    
    let engine = knowledge_engine.read().await;
    let suggestions = match suggestion_type {
        "missing_facts" => generate_missing_facts_suggestions(&*engine, focus_area, limit).await?,
        "interesting_questions" => generate_interesting_questions(&*engine, focus_area, limit).await?,
        "potential_connections" => generate_potential_connections(&*engine, focus_area, limit).await?,
        "knowledge_gaps" => generate_knowledge_gaps(&*engine, focus_area, limit).await?,
        _ => vec![]
    };
    
    let _ = update_usage_stats(usage_stats, StatsOperation::ExecuteQuery, 15).await;
    
    let data = json!({
        "suggestion_type": suggestion_type,
        "focus_area": focus_area,
        "suggestions": suggestions,
        "count": suggestions.len()
    });
    
    let message = format!(
        "Generated {} {} suggestion{}{}:\n\n{}",
        suggestions.len(),
        suggestion_type.replace('_', " "),
        if suggestions.len() == 1 { "" } else { "s" },
        if let Some(area) = focus_area { format!(" for {}", area) } else { "".to_string() },
        suggestions.iter()
            .enumerate()
            .map(|(i, s)| format!("{}. {}", i + 1, s))
            .collect::<Vec<_>>()
            .join("\n")
    );
    
    let next_steps = vec![
        "Use these suggestions to add more knowledge".to_string(),
        "Run 'validate_knowledge' to check data quality".to_string(),
    ];
    
    Ok((data, message, next_steps))
}

/// Find paths between two entities
async fn find_paths_between(
    engine: &KnowledgeEngine,
    start: &str,
    end: &str,
    max_depth: usize,
    relationship_types: Option<&Vec<String>>,
) -> Result<Vec<Vec<(String, String)>>> {
    let mut paths = Vec::new();
    let mut queue = VecDeque::new();
    let mut visited = HashSet::new();
    
    // Start BFS
    queue.push_back((vec![(start.to_string(), String::new())], 0));
    
    while let Some((path, depth)) = queue.pop_front() {
        if depth > max_depth {
            continue;
        }
        
        let current_entity = &path.last().unwrap().0;
        
        if current_entity == end && depth > 0 {
            paths.push(path);
            continue;
        }
        
        if visited.contains(current_entity) {
            continue;
        }
        visited.insert(current_entity.clone());
        
        // Find outgoing relationships
        let query = TripleQuery {
            subject: Some(current_entity.clone()),
            predicate: None,
            object: None,
            limit: 100,
            min_confidence: 0.0,
            include_chunks: false,
        };
        
        if let Ok(triples) = engine.query_triples(query) {
            for triple in triples {
                // Filter by relationship type if specified
                if let Some(types) = relationship_types {
                    if !types.contains(&triple.predicate) {
                        continue;
                    }
                }
                
                if !visited.contains(&triple.object) {
                    let mut new_path = path.clone();
                    new_path.push((triple.object.clone(), triple.predicate.clone()));
                    queue.push_back((new_path, depth + 1));
                }
            }
        }
    }
    
    // Sort paths by length
    paths.sort_by_key(|p| p.len());
    paths.truncate(10); // Limit to 10 paths
    
    Ok(paths)
}

/// Explore connections from an entity
async fn explore_from_entity(
    engine: &KnowledgeEngine,
    start: &str,
    max_depth: usize,
    relationship_types: Option<&Vec<String>>,
) -> Result<HashMap<String, ConnectionInfo>> {
    let mut connections = HashMap::new();
    let mut queue = VecDeque::new();
    let mut visited = HashSet::new();
    
    queue.push_back((start.to_string(), 0, String::new(), vec![start.to_string()]));
    
    while let Some((entity, depth, relationship, path)) = queue.pop_front() {
        if depth > max_depth || visited.contains(&entity) {
            continue;
        }
        visited.insert(entity.clone());
        
        if depth > 0 {
            connections.insert(entity.clone(), ConnectionInfo {
                distance: depth,
                relationship,
                path: path.clone(),
            });
        }
        
        // Find outgoing relationships
        let query = TripleQuery {
            subject: Some(entity.clone()),
            predicate: None,
            object: None,
            limit: 100,
            min_confidence: 0.0,
            include_chunks: false,
        };
        
        if let Ok(triples) = engine.query_triples(query) {
            for triple in triples {
                if let Some(types) = relationship_types {
                    if !types.contains(&triple.predicate) {
                        continue;
                    }
                }
                
                if !visited.contains(&triple.object) {
                    let mut new_path = path.clone();
                    new_path.push(triple.object.clone());
                    queue.push_back((
                        triple.object.clone(),
                        depth + 1,
                        triple.predicate.clone(),
                        new_path,
                    ));
                }
            }
        }
    }
    
    Ok(connections)
}

/// Generate missing facts suggestions
async fn generate_missing_facts_suggestions(
    engine: &KnowledgeEngine,
    focus_area: Option<&str>,
    limit: usize,
) -> Result<Vec<String>> {
    let mut suggestions = Vec::new();
    
    if let Some(entity) = focus_area {
        // Check what's missing for this specific entity
        let query = TripleQuery {
            subject: Some(entity.to_string()),
            predicate: None,
            object: None,
            limit: 100,
            min_confidence: 0.0,
            include_chunks: false,
        };
        
        if let Ok(facts) = engine.query_triples(query) {
            let predicates: HashSet<_> = facts.iter().map(|t| &t.predicate).collect();
            
            // Suggest common missing predicates
            if !predicates.contains(&"born_in".to_string()) && !predicates.contains(&"born_on".to_string()) {
                suggestions.push(format!("Add birth information for {}", entity));
            }
            if !predicates.contains(&"type".to_string()) && !predicates.contains(&"is".to_string()) {
                suggestions.push(format!("Specify what type of entity {} is", entity));
            }
            if !predicates.contains(&"description".to_string()) {
                suggestions.push(format!("Add a description for {}", entity));
            }
        }
    } else {
        // General suggestions
        suggestions.extend(generate_suggestions("missing_facts", None));
    }
    
    suggestions.truncate(limit);
    Ok(suggestions)
}

/// Generate interesting questions
async fn generate_interesting_questions(
    _engine: &KnowledgeEngine,
    focus_area: Option<&str>,
    limit: usize,
) -> Result<Vec<String>> {
    let mut questions = generate_suggestions("interesting_questions", focus_area);
    
    if let Some(entity) = focus_area {
        // Add entity-specific questions
        questions.insert(0, format!("What are all the connections of {}?", entity));
        questions.insert(1, format!("How is {} related to other entities?", entity));
        questions.insert(2, format!("What facts do we know about {}?", entity));
    }
    
    questions.truncate(limit);
    Ok(questions)
}

/// Generate potential connections
async fn generate_potential_connections(
    _engine: &KnowledgeEngine,
    focus_area: Option<&str>,
    limit: usize,
) -> Result<Vec<String>> {
    let suggestions = generate_suggestions("potential_connections", focus_area);
    Ok(suggestions.into_iter().take(limit).collect())
}

/// Generate knowledge gaps
async fn generate_knowledge_gaps(
    _engine: &KnowledgeEngine,
    focus_area: Option<&str>,
    limit: usize,
) -> Result<Vec<String>> {
    let gaps = generate_suggestions("knowledge_gaps", focus_area);
    Ok(gaps.into_iter().take(limit).collect())
}

/// Format a path for display
fn format_path(path: &[(String, String)]) -> String {
    let mut result = String::new();
    
    for (i, (entity, relationship)) in path.iter().enumerate() {
        if i == 0 {
            result.push_str(entity);
        } else {
            result.push_str(&format!(" -> {} -> {}", relationship, entity));
        }
    }
    
    result
}

/// Format multiple paths for display
fn format_paths_for_display(paths: &[Vec<(String, String)>], max_display: usize) -> String {
    let display_count = paths.len().min(max_display);
    let mut result = String::new();
    
    for (i, path) in paths.iter().take(display_count).enumerate() {
        result.push_str(&format!("{}. {} (length: {})\n", 
            i + 1, format_path(path), path.len() - 1));
    }
    
    if paths.len() > display_count {
        result.push_str(&format!("... and {} more paths", paths.len() - display_count));
    }
    
    result
}

/// Format connections for display
fn format_connections_for_display(connections: &HashMap<String, ConnectionInfo>, max_display: usize) -> String {
    let mut sorted: Vec<_> = connections.iter().collect();
    sorted.sort_by_key(|(_, info)| info.distance);
    
    let display_count = sorted.len().min(max_display);
    let mut result = String::new();
    
    for (i, (entity, info)) in sorted.iter().take(display_count).enumerate() {
        result.push_str(&format!("{}. {} (distance: {}, via: {})\n",
            i + 1, entity, info.distance, info.relationship));
    }
    
    if sorted.len() > display_count {
        result.push_str(&format!("... and {} more connections", sorted.len() - display_count));
    }
    
    result
}

#[derive(Debug)]
struct ConnectionInfo {
    distance: usize,
    relationship: String,
    path: Vec<String>,
}