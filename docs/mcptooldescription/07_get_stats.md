# get_stats - Knowledge Graph Statistics and Analytics Tool

## Overview

The `get_stats` tool provides comprehensive statistical analysis and performance metrics for the LLMKG knowledge graph. It offers detailed insights into graph structure, memory usage, operational performance, and system health, with optional detailed breakdowns for category-specific analysis. This tool is essential for monitoring system performance, optimizing resource usage, and understanding knowledge graph growth patterns.

## Implementation Details

### Handler Location
- **File**: `src/mcp/llm_friendly_server/handlers/stats.rs`
- **Function**: `handle_get_stats`
- **Lines**: 14-177

### Core Functionality

The tool implements comprehensive analytics and performance monitoring:

1. **Basic Statistics Collection**: Core metrics about knowledge graph size and structure
2. **Memory Analysis**: Detailed memory usage and efficiency metrics
3. **Performance Tracking**: Operation timing and throughput analysis
4. **Health Assessment**: Overall system health scoring
5. **Detailed Breakdowns**: Optional category-specific analysis
6. **Efficiency Scoring**: Memory and query performance optimization metrics

### Input Schema

```json
{
  "type": "object",
  "properties": {
    "include_details": {
      "type": "boolean",
      "description": "Include detailed breakdown by category/type",
      "default": false
    }
  },
  "additionalProperties": false
}
```

### Key Variables and Functions

#### Primary Handler Function
```rust
pub async fn handle_get_stats(
    knowledge_engine: &Arc<RwLock<KnowledgeEngine>>,
    usage_stats: &Arc<RwLock<UsageStats>>,
    params: Value,
) -> std::result::Result<(Value, String, Vec<String>), String>
```

#### Input Processing
```rust
let include_details = params.get("include_details")
    .and_then(|v| v.as_bool())
    .unwrap_or(false);
```

### Statistics Collection System

#### Basic Statistics Structure
```rust
#[derive(Debug)]
struct BasicStats {
    total_triples: usize,
    unique_entities: usize,
    unique_predicates: usize,
    knowledge_chunks: usize,
    avg_facts_per_entity: f64,
    graph_density: f64,
}
```

#### Basic Statistics Collection Function
```rust
fn collect_basic_stats(engine: &KnowledgeEngine) -> Result<BasicStats>
```

**Optimized Collection Strategy:**
The system uses cached memory statistics for performance rather than expensive real-time queries:

```rust
// Use cached memory stats instead of expensive query
let memory_stats = engine.get_memory_stats();
let total_triples = memory_stats.total_triples;
let total_nodes = memory_stats.total_nodes;

// Estimate entity and predicate counts based on typical ratios
let unique_entities = (total_nodes as f64 * 0.7) as usize; // Assume 70% of nodes are entities
let unique_predicates = (total_triples as f64 * 0.1).max(1.0) as usize; // Assume ~10% unique predicates

// Estimate knowledge chunks
let knowledge_chunks = (total_nodes as f64 * 0.1) as usize; // Assume 10% are knowledge chunks
```

**Calculated Metrics:**
```rust
// Calculate average facts per entity
let avg_facts_per_entity = if unique_entities > 0 {
    total_triples as f64 / unique_entities as f64
} else {
    0.0
};

// Calculate graph density (simplified)
let max_possible_edges = if unique_entities > 1 {
    unique_entities * (unique_entities - 1)
} else {
    0
};
let graph_density = if max_possible_edges > 0 {
    total_triples as f64 / max_possible_edges as f64
} else {
    0.0
};
```

#### Detailed Statistics Structure
```rust
#[derive(Debug)]
struct DetailedStats {
    entity_types: HashMap<String, usize>,
    relationship_types: HashMap<String, usize>,
    top_entities: Vec<(String, usize)>,
    relationship_distribution: HashMap<String, f64>,
}
```

#### Detailed Statistics Collection Function
```rust
fn collect_detailed_stats(engine: &KnowledgeEngine) -> Result<DetailedStats>
```

**Real-Time Analysis Implementation:**
```rust
let all_triples = engine.query_triples(
    TripleQuery {
        subject: None,
        predicate: None,
        object: None,
        limit: 1000, // Increased limit for better statistics
        min_confidence: 0.0,
        include_chunks: false,
    }
)?;

// Count entity types
let mut entity_types = HashMap::new();
for triple in &all_triples.triples {
    if triple.predicate == "is" || triple.predicate == "type" {
        *entity_types.entry(triple.object.clone()).or_insert(0) += 1;
    }
}

// Count relationship types  
let mut relationship_types = HashMap::new();
for triple in &all_triples.triples {
    *relationship_types.entry(triple.predicate.clone()).or_insert(0) += 1;
}

// Find top entities by number of connections
let mut entity_connections = HashMap::new();
for triple in &all_triples.triples {
    *entity_connections.entry(triple.subject.clone()).or_insert(0) += 1;
    *entity_connections.entry(triple.object.clone()).or_insert(0) += 1;
}

let mut top_entities: Vec<_> = entity_connections.into_iter().collect();
top_entities.sort_by(|a, b| b.1.cmp(&a.1));
top_entities.truncate(10);
```

### Memory Analysis System

#### Memory Statistics Function
```rust
fn get_memory_stats(engine: &KnowledgeEngine) -> MemoryStats
```

**Cached Memory Statistics:**
```rust
// Simply return the cached memory stats instead of expensive computation
let result = engine.get_memory_stats();
result
```

The `MemoryStats` structure includes:
- `total_bytes`: Total memory usage in bytes
- `total_nodes`: Number of nodes in the graph
- `total_triples`: Number of stored triples
- Additional memory efficiency metrics

#### Storage Optimization Scoring
```rust
fn calculate_storage_optimization(memory_stats: &MemoryStats) -> f64 {
    // Higher entity density = better optimization
    if memory_stats.total_bytes == 0 {
        return 1.0;
    }
    
    let entities_per_byte = memory_stats.total_nodes as f64 / memory_stats.total_bytes as f64;
    
    // Normalize to 0-1 scale (assuming 0.01 entities per byte is excellent)
    (entities_per_byte / 0.01).min(1.0)
}
```

### Performance Analysis System

#### Query Performance Scoring
```rust
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
```

**Performance Scoring Bands:**
- **≤10ms**: Perfect score (1.0)
- **10-100ms**: Linear degradation to 0.5
- **100-500ms**: Further degradation to 0.0
- **>500ms**: Minimum score (0.0)

#### Overall Health Assessment
```rust
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
```

**Health Score Components:**
- **Size Score**: 30% weight (knowledge graph completeness)
- **Efficiency Score**: 40% weight (memory optimization)
- **Performance Score**: 30% weight (query responsiveness)

### Output Format

#### Comprehensive Statistics Response
```json
{
  "knowledge_graph": {
    "total_facts": 1247,
    "total_entities": 523,
    "total_relationships": 42,
    "knowledge_chunks": 89,
    "avg_facts_per_entity": 2.4,
    "density": 0.943
  },
  "memory": {
    "total_bytes": 2458624,
    "entity_count": 523,
    "relationship_count": 1247,
    "memory_efficiency": 0.847
  },
  "usage": {
    "total_operations": 1532,
    "triples_stored": 1247,
    "chunks_stored": 89,
    "queries_executed": 346,
    "avg_response_time_ms": 23.4,
    "cache_hit_rate": 0.73
  },
  "performance": {
    "memory_efficiency_score": 0.847,
    "storage_optimization": 0.892,
    "query_performance": 0.756,
    "overall_health": 0.831
  }
}
```

#### Detailed Breakdown (Optional)
```json
{
  "details": {
    "entity_types": {
      "person": 145,
      "organization": 67,
      "location": 89,
      "concept": 222
    },
    "relationship_types": {
      "is": 234,
      "created": 56,
      "located_in": 78,
      "related_to": 123
    },
    "top_entities": [
      ["Einstein", 23],
      ["Physics", 19],
      ["Germany", 15]
    ],
    "relationship_distribution": {
      "is": 0.234,
      "created": 0.056,
      "located_in": 0.078
    }
  }
}
```

#### Human-Readable Message Format
```rust
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
        memory_stats.total_nodes as f64 / (memory_stats.total_bytes as f64 / 1_048_576.0)
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
```

### Performance Characteristics

#### Execution Strategy
The tool uses a multi-phase approach to balance performance and accuracy:

1. **Basic Statistics**: Uses cached data for performance
2. **Memory Statistics**: Direct cache access
3. **Usage Statistics**: Lock acquisition with cloning for quick release
4. **Detailed Statistics**: Real-time queries only when requested

#### Complexity Analysis
- **Basic Collection**: O(1) using cached data
- **Detailed Collection**: O(n) where n is number of triples (when requested)
- **Memory Analysis**: O(1) using cached memory stats
- **Performance Calculation**: O(1) mathematical operations

#### Memory Usage
- **Temporary Collections**: HashMaps for detailed analysis
- **Result Structures**: JSON serialization memory
- **Cache Access**: Minimal memory overhead

#### Usage Statistics Impact
- **Weight**: 15 points per operation
- **Operation Type**: `StatsOperation::ExecuteQuery`

### Advanced Features

#### Efficiency Scoring Integration
```rust
use crate::mcp::llm_friendly_server::utils::calculate_efficiency_score;

let efficiency_score = calculate_efficiency_score(&memory_stats);
```

#### Cache Hit Rate Calculation
```rust
"cache_hit_rate": if usage.cache_hits + usage.cache_misses > 0 {
    usage.cache_hits as f64 / (usage.cache_hits + usage.cache_misses) as f64
} else {
    0.0
}
```

#### Storage Optimization Metrics
- **Entities per byte**: Measures data density
- **Memory efficiency**: Overall storage effectiveness
- **Storage optimization score**: Normalized efficiency rating

### Error Handling

#### Statistics Collection Errors
```rust
let basic_stats = {
    let engine = knowledge_engine.read().await;
    let stats = collect_basic_stats(&*engine)
        .map_err(|e| format!("Failed to collect statistics: {}", e))?;
    stats
};
```

#### Detailed Statistics Errors (Optional)
```rust
if include_details {
    let detailed_stats = {
        let engine = knowledge_engine.read().await;
        collect_detailed_stats(&*engine)
            .map_err(|e| format!("Failed to collect detailed statistics: {}", e))?
    };
}
```

### Integration Points

#### With Knowledge Engine
Direct integration for memory statistics:
```rust
let memory_stats = engine.get_memory_stats();
```

#### With Usage Analytics System
```rust
use crate::mcp::llm_friendly_server::types::UsageStats;

let usage = {
    let usage = usage_stats.read().await;
    usage.clone()  // Clone to release the lock
};
```

#### With Utility Functions
```rust
use crate::mcp::llm_friendly_server::utils::{update_usage_stats, calculate_efficiency_score, StatsOperation};
```

### Best Practices for Developers

1. **Regular Monitoring**: Check stats periodically for performance trends
2. **Detailed Analysis**: Use `include_details=true` for troubleshooting
3. **Performance Tracking**: Monitor response times and cache hit rates
4. **Memory Management**: Watch memory efficiency scores for optimization opportunities
5. **Health Assessment**: Use overall health score for system status monitoring

### Usage Examples

#### Basic Statistics
```json
{
  "include_details": false
}
```

#### Comprehensive Analysis
```json
{
  "include_details": true
}
```

### Suggestions System
```rust
let suggestions = vec![
    "Monitor memory efficiency regularly".to_string(),
    "Use include_details=true for category breakdowns".to_string(),
    "Cache hit rate shows query optimization effectiveness".to_string(),
];
```

### Tool Integration Workflow

1. **Input Processing**: Validate parameters and determine analysis scope
2. **Basic Statistics**: Collect core metrics using cached data for performance
3. **Memory Analysis**: Retrieve memory usage and efficiency metrics
4. **Usage Statistics**: Access operational performance data with lock management
5. **Performance Calculation**: Compute efficiency and health scores
6. **Detailed Analysis**: Optionally perform real-time detailed breakdown
7. **Response Generation**: Format comprehensive statistics with human-readable summaries
8. **Usage Tracking**: Update system analytics for tool usage monitoring

This tool provides essential insights into knowledge graph health, performance, and optimization opportunities, serving as the primary dashboard for system monitoring and maintenance in the LLMKG system.