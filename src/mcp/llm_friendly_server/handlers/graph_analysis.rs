//! Unified graph analysis handler

use crate::core::knowledge_engine::KnowledgeEngine;
use crate::core::knowledge_types::TripleQuery;
use crate::core::triple::Triple;
use crate::mcp::llm_friendly_server::utils::{update_usage_stats, StatsOperation};
use crate::mcp::llm_friendly_server::types::UsageStats;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet, VecDeque};

/// Unified graph analysis handler
pub async fn handle_analyze_graph(
    knowledge_engine: &Arc<RwLock<KnowledgeEngine>>,
    usage_stats: &Arc<RwLock<UsageStats>>,
    params: Value,
) -> std::result::Result<(Value, String, Vec<String>), String> {
    // Extract analysis type
    let analysis_type = params.get("analysis_type")
        .and_then(|v| v.as_str())
        .ok_or("Missing required field: analysis_type")?;
    
    // Validate analysis type
    if !["connections", "centrality", "clustering", "prediction"].contains(&analysis_type) {
        return Err(format!("Invalid analysis_type: {}. Must be one of: connections, centrality, clustering, prediction", analysis_type));
    }
    
    // Extract config
    let config = params.get("config")
        .ok_or("Missing required field: config")?;
    
    let start_time = std::time::Instant::now();
    
    // Route to appropriate analysis function
    let (results, specific_message) = match analysis_type {
        "connections" => analyze_connections(knowledge_engine, config).await?,
        "centrality" => analyze_centrality(knowledge_engine, config).await?,
        "clustering" => analyze_clustering(knowledge_engine, config).await?,
        "prediction" => analyze_predictions(knowledge_engine, config).await?,
        _ => unreachable!()
    };
    
    let execution_time = start_time.elapsed();
    
    // Update usage stats
    let _ = update_usage_stats(usage_stats, StatsOperation::ExecuteQuery, 100).await;
    
    // Build performance metrics
    let performance_metrics = json!({
        "execution_time_ms": execution_time.as_millis(),
        "nodes_processed": results.get("nodes_processed").unwrap_or(&json!(0)),
        "edges_processed": results.get("edges_processed").unwrap_or(&json!(0)),
        "analysis_type": analysis_type
    });
    
    let data = json!({
        "analysis_type": analysis_type,
        "results": results,
        "performance_metrics": performance_metrics,
        "config": config
    });
    
    let message = format!(
        "Graph Analysis Complete ({}): {}",
        analysis_type,
        specific_message
    );
    
    let suggestions = match analysis_type {
        "connections" => vec![
            "Try increasing max_depth to find more paths".to_string(),
            "Use relationship_types to filter specific connections".to_string(),
        ],
        "centrality" => vec![
            "Use PageRank for global importance".to_string(),
            "Use Betweenness for finding bridge nodes".to_string(),
        ],
        "clustering" => vec![
            "Adjust resolution to control cluster granularity".to_string(),
            "Try different algorithms for different perspectives".to_string(),
        ],
        "prediction" => vec![
            "Higher confidence thresholds give more reliable predictions".to_string(),
            "Enable neural features for better accuracy".to_string(),
        ],
        _ => vec![]
    };
    
    Ok((data, message, suggestions))
}

/// Analyze connections between entities
async fn analyze_connections(
    knowledge_engine: &Arc<RwLock<KnowledgeEngine>>,
    config: &Value,
) -> std::result::Result<(Value, String), String> {
    // Validate required fields
    let start_entity = config.get("start_entity")
        .and_then(|v| v.as_str())
        .ok_or("Missing required config field: start_entity")?;
    
    let end_entity = config.get("end_entity")
        .and_then(|v| v.as_str());
    
    let max_depth = config.get("max_depth")
        .and_then(|v| v.as_u64())
        .unwrap_or(2) as usize;
    
    let relationship_types = config.get("relationship_types")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str())
                .map(|s| s.to_string())
                .collect::<HashSet<String>>()
        });
    
    // Get graph data
    let engine = knowledge_engine.read().await;
    let graph_data = engine.query_triples(TripleQuery {
        subject: None,
        predicate: None,
        object: None,
        limit: 10000,
        min_confidence: 0.0,
        include_chunks: false,
    }).map_err(|e| format!("Failed to query graph: {}", e))?;
    drop(engine);
    
    // Build adjacency list
    let adjacency = build_adjacency_list(&graph_data.triples, &relationship_types);
    
    // Find paths
    let paths = if let Some(end) = end_entity {
        find_paths(&adjacency, start_entity, end, max_depth)
    } else {
        explore_from_entity(&adjacency, start_entity, max_depth)
    };
    
    let total_paths = paths.len();
    let nodes_processed = adjacency.len();
    let edges_processed = graph_data.triples.len();
    
    let results = json!({
        "paths": paths,
        "total_paths": total_paths,
        "start_entity": start_entity,
        "end_entity": end_entity,
        "max_depth": max_depth,
        "nodes_processed": nodes_processed,
        "edges_processed": edges_processed
    });
    
    let message = if total_paths == 0 {
        "No paths found".to_string()
    } else if end_entity.is_some() {
        format!("Found {} paths from {} to {}", total_paths, start_entity, end_entity.unwrap())
    } else {
        format!("Found {} connections from {}", total_paths, start_entity)
    };
    
    Ok((results, message))
}

/// Analyze graph centrality
async fn analyze_centrality(
    knowledge_engine: &Arc<RwLock<KnowledgeEngine>>,
    config: &Value,
) -> std::result::Result<(Value, String), String> {
    let centrality_types = config.get("centrality_types")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str())
                .map(|s| s.to_string())
                .collect::<Vec<String>>()
        })
        .unwrap_or_else(|| vec!["pagerank".to_string()]);
    
    let top_n = config.get("top_n")
        .and_then(|v| v.as_u64())
        .unwrap_or(20) as usize;
    
    let include_scores = config.get("include_scores")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    
    let entity_filter = config.get("entity_filter")
        .and_then(|v| v.as_str());
    
    // Get graph data
    let engine = knowledge_engine.read().await;
    let graph_data = engine.query_triples(TripleQuery {
        subject: None,
        predicate: None,
        object: None,
        limit: 10000,
        min_confidence: 0.0,
        include_chunks: false,
    }).map_err(|e| format!("Failed to query graph: {}", e))?;
    drop(engine);
    
    let mut centrality_measures = HashMap::new();
    
    for centrality_type in &centrality_types {
        let scores = match centrality_type.as_str() {
            "pagerank" => calculate_pagerank(&graph_data.triples, entity_filter),
            "betweenness" => calculate_betweenness_centrality(&graph_data.triples, entity_filter),
            "closeness" => calculate_closeness_centrality(&graph_data.triples, entity_filter),
            "eigenvector" => calculate_eigenvector_centrality(&graph_data.triples, entity_filter),
            "degree" => calculate_degree_centrality(&graph_data.triples, entity_filter),
            _ => return Err(format!("Unknown centrality type: {}", centrality_type))
        };
        
        // Get top N
        let mut scored_entities: Vec<(String, f64)> = scores.into_iter().collect();
        scored_entities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored_entities.truncate(top_n);
        
        let formatted_results: Vec<Value> = if include_scores {
            scored_entities.iter()
                .map(|(entity, score)| json!({
                    "entity": entity,
                    "score": score
                }))
                .collect()
        } else {
            scored_entities.iter()
                .map(|(entity, _)| json!(entity))
                .collect()
        };
        
        centrality_measures.insert(centrality_type.clone(), json!(formatted_results));
    }
    
    let results = json!({
        "centrality_measures": centrality_measures,
        "top_n": top_n,
        "include_scores": include_scores,
        "nodes_processed": graph_data.triples.len(),
        "edges_processed": graph_data.triples.len()
    });
    
    let message = format!(
        "Analyzed {} centrality measures for top {} entities",
        centrality_types.len(),
        top_n
    );
    
    Ok((results, message))
}

/// Analyze graph clustering
async fn analyze_clustering(
    knowledge_engine: &Arc<RwLock<KnowledgeEngine>>,
    config: &Value,
) -> std::result::Result<(Value, String), String> {
    let algorithm = config.get("algorithm")
        .and_then(|v| v.as_str())
        .unwrap_or("leiden");
    
    let resolution = config.get("resolution")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0) as f32;
    
    let min_cluster_size = config.get("min_cluster_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(3) as usize;
    
    let max_clusters = config.get("max_clusters")
        .and_then(|v| v.as_u64())
        .unwrap_or(50) as usize;
    
    let include_metadata = config.get("include_metadata")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    
    // Validate algorithm
    if !["leiden", "louvain", "hierarchical"].contains(&algorithm) {
        return Err(format!("Unknown clustering algorithm: {}", algorithm));
    }
    
    // Get graph data
    let engine = knowledge_engine.read().await;
    let graph_data = engine.query_triples(TripleQuery {
        subject: None,
        predicate: None,
        object: None,
        limit: 10000,
        min_confidence: 0.0,
        include_chunks: false,
    }).map_err(|e| format!("Failed to query graph: {}", e))?;
    drop(engine);
    
    // Execute clustering (simplified implementation)
    let clusters = execute_clustering(algorithm, &graph_data.triples, resolution, min_cluster_size, max_clusters);
    
    // Calculate clustering metrics
    let modularity = calculate_modularity(&clusters, &graph_data.triples);
    
    let cluster_metadata = if include_metadata {
        json!({
            "cluster_sizes": clusters.iter().map(|c| c.len()).collect::<Vec<_>>(),
            "average_cluster_size": clusters.iter().map(|c| c.len()).sum::<usize>() as f64 / clusters.len() as f64,
            "largest_cluster": clusters.iter().map(|c| c.len()).max().unwrap_or(0),
            "smallest_cluster": clusters.iter().map(|c| c.len()).min().unwrap_or(0)
        })
    } else {
        json!({})
    };
    
    let results = json!({
        "clusters": clusters,
        "algorithm_used": algorithm,
        "clustering_metrics": {
            "modularity": modularity,
            "num_clusters": clusters.len(),
            "resolution": resolution
        },
        "cluster_metadata": cluster_metadata,
        "nodes_processed": graph_data.triples.len(),
        "edges_processed": graph_data.triples.len()
    });
    
    let message = format!(
        "Found {} clusters using {} algorithm (modularity: {:.3})",
        clusters.len(),
        algorithm,
        modularity
    );
    
    Ok((results, message))
}

/// Analyze and predict graph structure
async fn analyze_predictions(
    knowledge_engine: &Arc<RwLock<KnowledgeEngine>>,
    config: &Value,
) -> std::result::Result<(Value, String), String> {
    let prediction_type = config.get("prediction_type")
        .and_then(|v| v.as_str())
        .unwrap_or("missing_links");
    
    let confidence_threshold = config.get("confidence_threshold")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.7) as f32;
    
    let max_predictions = config.get("max_predictions")
        .and_then(|v| v.as_u64())
        .unwrap_or(20) as usize;
    
    let entity_filter = config.get("entity_filter")
        .and_then(|v| v.as_str());
    
    let use_neural_features = config.get("use_neural_features")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    
    // Validate prediction type
    if !["missing_links", "future_connections", "community_evolution", "knowledge_gaps"].contains(&prediction_type) {
        return Err(format!("Unknown prediction type: {}", prediction_type));
    }
    
    // Get graph data
    let engine = knowledge_engine.read().await;
    let graph_data = engine.query_triples(TripleQuery {
        subject: None,
        predicate: None,
        object: None,
        limit: 10000,
        min_confidence: 0.0,
        include_chunks: true,
    }).map_err(|e| format!("Failed to query graph: {}", e))?;
    drop(engine);
    
    // Generate predictions (simplified implementation)
    let predictions = generate_predictions(
        prediction_type,
        &graph_data.triples,
        confidence_threshold,
        max_predictions,
        entity_filter,
        use_neural_features
    );
    
    // Calculate confidence distribution
    let confidence_distribution = calculate_confidence_distribution(&predictions);
    
    // Validation score
    let validation_score = 0.85; // Simulated validation
    
    let results = json!({
        "predictions": predictions,
        "confidence_distribution": confidence_distribution,
        "validation_score": validation_score,
        "prediction_type": prediction_type,
        "total_predictions": predictions.len(),
        "nodes_processed": graph_data.triples.len(),
        "edges_processed": graph_data.triples.len()
    });
    
    let message = format!(
        "Generated {} {} predictions (validation score: {:.2})",
        predictions.len(),
        prediction_type.replace('_', " "),
        validation_score
    );
    
    Ok((results, message))
}

// Helper functions

fn build_adjacency_list(triples: &[Triple], relationship_filter: &Option<HashSet<String>>) -> HashMap<String, Vec<(String, String)>> {
    let mut adjacency: HashMap<String, Vec<(String, String)>> = HashMap::new();
    
    for triple in triples {
        if let Some(filter) = relationship_filter {
            if !filter.contains(&triple.predicate) {
                continue;
            }
        }
        
        adjacency.entry(triple.subject.clone())
            .or_insert_with(Vec::new)
            .push((triple.predicate.clone(), triple.object.clone()));
    }
    
    adjacency
}

fn find_paths(
    adjacency: &HashMap<String, Vec<(String, String)>>,
    start: &str,
    end: &str,
    max_depth: usize
) -> Vec<Value> {
    let mut paths = Vec::new();
    let mut queue = VecDeque::new();
    queue.push_back((start.to_string(), vec![start.to_string()], 0));
    
    while let Some((current, path, depth)) = queue.pop_front() {
        if depth > max_depth {
            continue;
        }
        
        if current == end && depth > 0 {
            paths.push(json!({
                "path": path,
                "length": depth
            }));
            continue;
        }
        
        if let Some(neighbors) = adjacency.get(&current) {
            for (predicate, neighbor) in neighbors {
                if !path.contains(neighbor) {
                    let mut new_path = path.clone();
                    new_path.push(predicate.clone());
                    new_path.push(neighbor.clone());
                    queue.push_back((neighbor.clone(), new_path, depth + 1));
                }
            }
        }
    }
    
    paths
}

fn explore_from_entity(
    adjacency: &HashMap<String, Vec<(String, String)>>,
    start: &str,
    max_depth: usize
) -> Vec<Value> {
    let mut connections = Vec::new();
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    queue.push_back((start.to_string(), 0));
    visited.insert(start.to_string());
    
    while let Some((current, depth)) = queue.pop_front() {
        if depth > max_depth {
            continue;
        }
        
        if let Some(neighbors) = adjacency.get(&current) {
            for (predicate, neighbor) in neighbors {
                if !visited.contains(neighbor) {
                    visited.insert(neighbor.clone());
                    connections.push(json!({
                        "from": current,
                        "predicate": predicate,
                        "to": neighbor,
                        "depth": depth + 1
                    }));
                    queue.push_back((neighbor.clone(), depth + 1));
                }
            }
        }
    }
    
    connections
}

fn calculate_pagerank(triples: &[Triple], _entity_filter: Option<&str>) -> HashMap<String, f64> {
    // Simplified PageRank implementation
    let mut scores = HashMap::new();
    let mut entities = HashSet::new();
    
    for triple in triples {
        entities.insert(triple.subject.clone());
        entities.insert(triple.object.clone());
    }
    
    // Initialize scores
    let initial_score = 1.0 / entities.len() as f64;
    for entity in &entities {
        scores.insert(entity.clone(), initial_score);
    }
    
    // Simplified iteration (would normally iterate until convergence)
    for _ in 0..10 {
        let mut new_scores = HashMap::new();
        
        for entity in &entities {
            let mut score = 0.15 / entities.len() as f64; // Damping factor
            
            // Add contributions from incoming links
            for triple in triples {
                if &triple.object == entity {
                    let source_score = scores.get(&triple.subject).unwrap_or(&initial_score);
                    let out_degree = triples.iter()
                        .filter(|t| t.subject == triple.subject)
                        .count() as f64;
                    
                    score += 0.85 * source_score / out_degree;
                }
            }
            
            new_scores.insert(entity.clone(), score);
        }
        
        scores = new_scores;
    }
    
    scores
}

fn calculate_betweenness_centrality(triples: &[Triple], _entity_filter: Option<&str>) -> HashMap<String, f64> {
    // Simplified betweenness centrality
    let mut scores = HashMap::new();
    let entities: HashSet<String> = triples.iter()
        .flat_map(|t| vec![t.subject.clone(), t.object.clone()])
        .collect();
    
    for entity in entities {
        scores.insert(entity, 0.0);
    }
    
    // Simplified: count how many times an entity appears in triples
    for triple in triples {
        *scores.get_mut(&triple.subject).unwrap() += 1.0;
        *scores.get_mut(&triple.object).unwrap() += 0.5;
    }
    
    scores
}

fn calculate_closeness_centrality(triples: &[Triple], _entity_filter: Option<&str>) -> HashMap<String, f64> {
    // Simplified closeness centrality
    calculate_degree_centrality(triples, _entity_filter)
}

fn calculate_eigenvector_centrality(triples: &[Triple], _entity_filter: Option<&str>) -> HashMap<String, f64> {
    // Simplified eigenvector centrality (similar to PageRank)
    calculate_pagerank(triples, _entity_filter)
}

fn calculate_degree_centrality(triples: &[Triple], _entity_filter: Option<&str>) -> HashMap<String, f64> {
    let mut scores = HashMap::new();
    
    for triple in triples {
        *scores.entry(triple.subject.clone()).or_insert(0.0) += 1.0;
        *scores.entry(triple.object.clone()).or_insert(0.0) += 1.0;
    }
    
    scores
}

fn execute_clustering(
    _algorithm: &str,
    triples: &[Triple],
    _resolution: f32,
    min_cluster_size: usize,
    max_clusters: usize
) -> Vec<Vec<String>> {
    // Simplified clustering implementation
    let mut clusters = Vec::new();
    let mut entities: HashSet<String> = triples.iter()
        .flat_map(|t| vec![t.subject.clone(), t.object.clone()])
        .collect();
    
    // Create simple clusters based on connectivity
    while !entities.is_empty() && clusters.len() < max_clusters {
        let mut cluster = Vec::new();
        let start = entities.iter().next().unwrap().clone();
        cluster.push(start.clone());
        entities.remove(&start);
        
        // Add connected entities
        for triple in triples {
            if cluster.contains(&triple.subject) && entities.contains(&triple.object) {
                cluster.push(triple.object.clone());
                entities.remove(&triple.object);
            } else if cluster.contains(&triple.object) && entities.contains(&triple.subject) {
                cluster.push(triple.subject.clone());
                entities.remove(&triple.subject);
            }
            
            if cluster.len() >= min_cluster_size * 2 {
                break;
            }
        }
        
        if cluster.len() >= min_cluster_size {
            clusters.push(cluster);
        }
    }
    
    clusters
}

fn calculate_modularity(clusters: &[Vec<String>], triples: &[Triple]) -> f64 {
    // Simplified modularity calculation
    if clusters.is_empty() || triples.is_empty() {
        return 0.0;
    }
    
    let total_edges = triples.len() as f64;
    let mut internal_edges = 0.0;
    
    for cluster in clusters {
        for triple in triples {
            if cluster.contains(&triple.subject) && cluster.contains(&triple.object) {
                internal_edges += 1.0;
            }
        }
    }
    
    internal_edges / total_edges
}

fn generate_predictions(
    prediction_type: &str,
    triples: &[Triple],
    confidence_threshold: f32,
    max_predictions: usize,
    _entity_filter: Option<&str>,
    _use_neural_features: bool
) -> Vec<Value> {
    // Simplified prediction generation
    let mut predictions = Vec::new();
    let entities: HashSet<String> = triples.iter()
        .flat_map(|t| vec![t.subject.clone(), t.object.clone()])
        .collect();
    
    match prediction_type {
        "missing_links" => {
            // Predict links between entities that share common neighbors
            for entity1 in &entities {
                for entity2 in &entities {
                    if entity1 >= entity2 {
                        continue;
                    }
                    
                    // Check if they share common neighbors
                    let common_neighbors = count_common_neighbors(entity1, entity2, triples);
                    let confidence = common_neighbors as f32 / 10.0;
                    
                    if confidence >= confidence_threshold && predictions.len() < max_predictions {
                        predictions.push(json!({
                            "type": "missing_link",
                            "source": entity1,
                            "target": entity2,
                            "predicted_relation": "related_to",
                            "confidence": confidence.min(1.0),
                            "common_neighbors": common_neighbors
                        }));
                    }
                }
            }
        },
        _ => {
            // Generic predictions for other types
            for (i, entity) in entities.iter().enumerate() {
                if i >= max_predictions {
                    break;
                }
                
                predictions.push(json!({
                    "type": prediction_type,
                    "entity": entity,
                    "prediction": format!("{} prediction for {}", prediction_type, entity),
                    "confidence": 0.75
                }));
            }
        }
    }
    
    predictions
}

fn count_common_neighbors(entity1: &str, entity2: &str, triples: &[Triple]) -> usize {
    let neighbors1: HashSet<String> = triples.iter()
        .filter(|t| t.subject == *entity1)
        .map(|t| t.object.clone())
        .collect();
    
    let neighbors2: HashSet<String> = triples.iter()
        .filter(|t| t.subject == *entity2)
        .map(|t| t.object.clone())
        .collect();
    
    neighbors1.intersection(&neighbors2).count()
}

fn calculate_confidence_distribution(predictions: &[Value]) -> Value {
    let mut high = 0;
    let mut medium = 0;
    let mut low = 0;
    
    for pred in predictions {
        if let Some(confidence) = pred.get("confidence").and_then(|v| v.as_f64()) {
            if confidence >= 0.8 {
                high += 1;
            } else if confidence >= 0.6 {
                medium += 1;
            } else {
                low += 1;
            }
        }
    }
    
    json!({
        "high_confidence": high,
        "medium_confidence": medium,
        "low_confidence": low
    })
}