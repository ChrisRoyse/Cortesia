# analyze_graph - Comprehensive Graph Analysis Suite

## Overview

The `analyze_graph` tool provides a comprehensive suite of graph analysis capabilities for the LLMKG system. It supports four distinct analysis types: connections exploration, centrality analysis, clustering detection, and structure prediction. This unified tool consolidates multiple graph analysis algorithms into a single interface, enabling sophisticated network analysis, community detection, and predictive modeling of knowledge graph structures.

## Implementation Details

### Handler Location
- **File**: `src/mcp/llm_friendly_server/handlers/graph_analysis.rs`
- **Function**: `handle_analyze_graph`
- **Lines**: 14-91

### Core Functionality

The tool implements four major analysis categories:

1. **Connections Analysis**: Path finding and exploration between entities
2. **Centrality Analysis**: Importance scoring using various centrality measures
3. **Clustering Analysis**: Community detection and graph partitioning
4. **Prediction Analysis**: Missing link prediction and structure forecasting

### Input Schema

```json
{
  "type": "object",
  "properties": {
    "analysis_type": {
      "type": "string",
      "description": "Type of graph analysis to perform",
      "enum": ["connections", "centrality", "clustering", "prediction"]
    },
    "config": {
      "type": "object",
      "description": "Analysis-specific configuration",
      "properties": {
        // Dynamic based on analysis_type
      }
    }
  },
  "required": ["analysis_type", "config"]
}
```

### Key Variables and Functions

#### Primary Handler Function
```rust
pub async fn handle_analyze_graph(
    knowledge_engine: &Arc<RwLock<KnowledgeEngine>>,
    usage_stats: &Arc<RwLock<UsageStats>>,
    params: Value,
) -> std::result::Result<(Value, String, Vec<String>), String>
```

#### Analysis Type Validation
```rust
let analysis_type = params.get("analysis_type")
    .and_then(|v| v.as_str())
    .ok_or("Missing required field: analysis_type")?;

// Validate analysis type
if !["connections", "centrality", "clustering", "prediction"].contains(&analysis_type) {
    return Err(format!("Invalid analysis_type: {}. Must be one of: connections, centrality, clustering, prediction", analysis_type));
}
```

#### Analysis Routing
```rust
let (results, specific_message) = match analysis_type {
    "connections" => analyze_connections(knowledge_engine, config).await?,
    "centrality" => analyze_centrality(knowledge_engine, config).await?,
    "clustering" => analyze_clustering(knowledge_engine, config).await?,
    "prediction" => analyze_predictions(knowledge_engine, config).await?,
    _ => unreachable!()
};
```

## Analysis Types

### 1. Connections Analysis

#### Function Implementation
```rust
async fn analyze_connections(
    knowledge_engine: &Arc<RwLock<KnowledgeEngine>>,
    config: &Value,
) -> std::result::Result<(Value, String), String>
```

#### Configuration Parameters
```rust
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
```

**Path Finding Algorithm:**
```rust
let paths = if let Some(end) = end_entity {
    find_paths(&adjacency, start_entity, end, max_depth)
} else {
    explore_from_entity(&adjacency, start_entity, max_depth)
};
```

**Use Cases:**
- Find shortest paths between specific entities
- Explore connections emanating from a starting entity
- Filter connections by relationship types
- Analyze connectivity patterns within depth limits

#### Example Configuration
```json
{
  "analysis_type": "connections",
  "config": {
    "start_entity": "Einstein",
    "end_entity": "Nobel_Prize",
    "max_depth": 3,
    "relationship_types": ["won", "received", "awarded"]
  }
}
```

### 2. Centrality Analysis

#### Function Implementation
```rust
async fn analyze_centrality(
    knowledge_engine: &Arc<RwLock<KnowledgeEngine>>,
    config: &Value,
) -> std::result::Result<(Value, String), String>
```

#### Supported Centrality Measures
```rust
let scores = match centrality_type.as_str() {
    "pagerank" => calculate_pagerank(&graph_data.triples, entity_filter),
    "betweenness" => calculate_betweenness_centrality(&graph_data.triples, entity_filter),
    "closeness" => calculate_closeness_centrality(&graph_data.triples, entity_filter),
    "eigenvector" => calculate_eigenvector_centrality(&graph_data.triples, entity_filter),
    "degree" => calculate_degree_centrality(&graph_data.triples, entity_filter),
    _ => return Err(format!("Unknown centrality type: {}", centrality_type))
};
```

**PageRank Implementation:**
```rust
fn calculate_pagerank(triples: &[Triple], _entity_filter: Option<&str>) -> HashMap<String, f64> {
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
    
    // Iterative refinement (10 iterations)
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
```

**Centrality Types:**
- **PageRank**: Global importance based on link structure and authority
- **Betweenness**: Importance as bridges between other nodes
- **Closeness**: Importance based on proximity to all other nodes
- **Eigenvector**: Importance based on connections to important nodes
- **Degree**: Simple connectivity-based importance

#### Example Configuration
```json
{
  "analysis_type": "centrality",
  "config": {
    "centrality_types": ["pagerank", "betweenness"],
    "top_n": 10,
    "include_scores": true,
    "entity_filter": "scientist"
  }
}
```

### 3. Clustering Analysis

#### Function Implementation
```rust
async fn analyze_clustering(
    knowledge_engine: &Arc<RwLock<KnowledgeEngine>>,
    config: &Value,
) -> std::result::Result<(Value, String), String>
```

#### Clustering Algorithms
```rust
let algorithm = config.get("algorithm")
    .and_then(|v| v.as_str())
    .unwrap_or("leiden");

// Validate algorithm
if !["leiden", "louvain", "hierarchical"].contains(&algorithm) {
    return Err(format!("Unknown clustering algorithm: {}", algorithm));
}
```

**Clustering Execution:**
```rust
let clusters = execute_clustering(algorithm, &graph_data.triples, resolution, min_cluster_size, max_clusters);
let modularity = calculate_modularity(&clusters, &graph_data.triples);
```

**Algorithm Features:**
- **Leiden**: High-quality community detection with resolution parameter
- **Louvain**: Fast modularity-based clustering
- **Hierarchical**: Multi-level clustering with dendrogram support

**Quality Metrics:**
```rust
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
```

#### Example Configuration
```json
{
  "analysis_type": "clustering",
  "config": {
    "algorithm": "leiden",
    "resolution": 1.2,
    "min_cluster_size": 5,
    "max_clusters": 20,
    "include_metadata": true
  }
}
```

### 4. Prediction Analysis

#### Function Implementation
```rust
async fn analyze_predictions(
    knowledge_engine: &Arc<RwLock<KnowledgeEngine>>,
    config: &Value,
) -> std::result::Result<(Value, String), String>
```

#### Prediction Types
```rust
// Validate prediction type
if !["missing_links", "future_connections", "community_evolution", "knowledge_gaps"].contains(&prediction_type) {
    return Err(format!("Unknown prediction type: {}", prediction_type));
}
```

**Missing Link Prediction:**
```rust
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
}
```

**Common Neighbors Algorithm:**
```rust
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
```

**Prediction Types:**
- **Missing Links**: Predict likely but unrecorded relationships
- **Future Connections**: Forecast probable future relationships
- **Community Evolution**: Predict cluster membership changes
- **Knowledge Gaps**: Identify areas needing more information

#### Example Configuration
```json
{
  "analysis_type": "prediction",
  "config": {
    "prediction_type": "missing_links",
    "confidence_threshold": 0.8,
    "max_predictions": 10,
    "entity_filter": "scientist"
  }
}
```

## Output Format

### Comprehensive Analysis Response
```json
{
  "analysis_type": "connections",
  "results": {
    "paths": [
      {
        "path": ["Einstein", "developed", "relativity", "revolutionized", "physics"],
        "length": 2
      }
    ],
    "total_paths": 1,
    "start_entity": "Einstein",
    "end_entity": "physics",
    "max_depth": 3,
    "nodes_processed": 523,
    "edges_processed": 1247
  },
  "performance_metrics": {
    "execution_time_ms": 156,
    "nodes_processed": 523,
    "edges_processed": 1247,
    "analysis_type": "connections"
  },
  "config": {
    "start_entity": "Einstein",
    "end_entity": "physics",
    "max_depth": 3
  }
}
```

### Centrality Analysis Results
```json
{
  "analysis_type": "centrality",
  "results": {
    "centrality_measures": {
      "pagerank": [
        {"entity": "Einstein", "score": 0.043},
        {"entity": "physics", "score": 0.039},
        {"entity": "relativity", "score": 0.035}
      ],
      "betweenness": [
        {"entity": "Einstein", "score": 15.7},
        {"entity": "science", "score": 12.3},
        {"entity": "theory", "score": 8.9}
      ]
    },
    "top_n": 10,
    "include_scores": true,
    "nodes_processed": 523,
    "edges_processed": 1247
  }
}
```

### Clustering Analysis Results
```json
{
  "analysis_type": "clustering",
  "results": {
    "clusters": [
      ["Einstein", "relativity", "physics", "quantum"],
      ["programming", "Python", "software", "development"],
      ["history", "world_war", "politics", "events"]
    ],
    "algorithm_used": "leiden",
    "clustering_metrics": {
      "modularity": 0.847,
      "num_clusters": 12,
      "resolution": 1.2
    },
    "cluster_metadata": {
      "cluster_sizes": [4, 4, 4],
      "average_cluster_size": 4.0,
      "largest_cluster": 8,
      "smallest_cluster": 3
    },
    "nodes_processed": 523,
    "edges_processed": 1247
  }
}
```

### Prediction Analysis Results
```json
{
  "analysis_type": "prediction",
  "results": {
    "predictions": [
      {
        "type": "missing_link",
        "source": "Einstein",
        "target": "quantum_mechanics",
        "predicted_relation": "contributed_to",
        "confidence": 0.89,
        "common_neighbors": 7
      }
    ],
    "confidence_distribution": {
      "high_confidence": 3,
      "medium_confidence": 4,
      "low_confidence": 1
    },
    "validation_score": 0.85,
    "prediction_type": "missing_links",
    "total_predictions": 8,
    "nodes_processed": 523,
    "edges_processed": 1247
  }
}
```

## Performance Characteristics

### Complexity Analysis
- **Connections**: O(V + E) for BFS traversal where V = vertices, E = edges
- **PageRank**: O(I × E) where I = iterations (typically 10)
- **Clustering**: O(E log V) for most community detection algorithms
- **Predictions**: O(V²) for missing link prediction using common neighbors

### Memory Usage
- **Graph Storage**: Adjacency lists and entity sets
- **Algorithm State**: Temporary structures for scores and clusters
- **Results**: JSON structures with analysis results and metadata

### Usage Statistics Impact
- **Weight**: 100 points per operation (high complexity graph analysis)
- **Operation Type**: `StatsOperation::ExecuteQuery`

## Error Handling

### Analysis Type Validation
```rust
if !["connections", "centrality", "clustering", "prediction"].contains(&analysis_type) {
    return Err(format!("Invalid analysis_type: {}. Must be one of: connections, centrality, clustering, prediction", analysis_type));
}
```

### Configuration Validation
```rust
let start_entity = config.get("start_entity")
    .and_then(|v| v.as_str())
    .ok_or("Missing required config field: start_entity")?;
```

### Algorithm Validation
```rust
if !["leiden", "louvain", "hierarchical"].contains(&algorithm) {
    return Err(format!("Unknown clustering algorithm: {}", algorithm));
}
```

## Integration Points

### With Knowledge Engine
```rust
let graph_data = engine.query_triples(TripleQuery {
    subject: None,
    predicate: None,
    object: None,
    limit: 10000,
    min_confidence: 0.0,
    include_chunks: false,
}).map_err(|e| format!("Failed to query graph: {}", e))?;
```

### With Graph Algorithms
Direct implementation of standard graph analysis algorithms:
- Path finding using breadth-first search
- Centrality calculations using iterative methods
- Community detection using modularity optimization
- Link prediction using structural similarity

## Best Practices for Developers

1. **Analysis Selection**: Choose appropriate analysis type for the research question
2. **Parameter Tuning**: Adjust algorithm parameters for optimal results
3. **Performance Monitoring**: Watch execution times for large graphs
4. **Result Interpretation**: Understand the meaning of different centrality measures
5. **Validation**: Use prediction validation scores to assess model quality

## Usage Examples

### Connection Exploration
```json
{
  "analysis_type": "connections",
  "config": {
    "start_entity": "artificial_intelligence",
    "max_depth": 2,
    "relationship_types": ["includes", "enables", "requires"]
  }
}
```

### Scientific Impact Analysis
```json
{
  "analysis_type": "centrality",
  "config": {
    "centrality_types": ["pagerank", "betweenness"],
    "top_n": 20,
    "entity_filter": "scientist"
  }
}
```

### Knowledge Domain Discovery
```json
{
  "analysis_type": "clustering",
  "config": {
    "algorithm": "leiden",
    "resolution": 0.8,
    "min_cluster_size": 10
  }
}
```

### Research Gap Identification
```json
{
  "analysis_type": "prediction",
  "config": {
    "prediction_type": "knowledge_gaps",
    "confidence_threshold": 0.7,
    "max_predictions": 15
  }
}
```

## Suggestions System
```rust
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
        "Enable advanced features for better accuracy".to_string(),
    ],
    _ => vec![]
};
```

## Research Applications

### Network Analysis
- **Social Networks**: Identify influential entities and communities
- **Citation Networks**: Analyze research impact and collaboration patterns
- **Knowledge Networks**: Discover conceptual relationships and domains

### Predictive Modeling
- **Link Prediction**: Forecast likely but unrecorded relationships
- **Community Evolution**: Predict how knowledge domains will develop
- **Research Trends**: Identify emerging research areas and connections

### Knowledge Discovery
- **Domain Mapping**: Understand the structure of knowledge areas
- **Cross-Domain Connections**: Find unexpected relationships between fields
- **Research Gaps**: Identify areas needing further investigation

## Tool Integration Workflow

1. **Input Processing**: Validate analysis type and configuration parameters
2. **Graph Data Retrieval**: Query comprehensive triple data from knowledge engine
3. **Algorithm Selection**: Route to appropriate analysis function based on type
4. **Analysis Execution**: Run selected algorithm with performance monitoring
5. **Result Synthesis**: Compile analysis results with metadata and metrics
6. **Performance Reporting**: Include execution times and processing statistics
7. **Suggestion Generation**: Provide context-appropriate recommendations
8. **Usage Tracking**: Update system analytics for graph analysis effectiveness

This tool provides sophisticated graph analysis capabilities for the LLMKG system, enabling comprehensive network analysis, community detection, and predictive modeling through a unified interface that consolidates multiple specialized algorithms into a single, powerful analysis suite.