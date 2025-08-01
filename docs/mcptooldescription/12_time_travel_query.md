# time_travel_query - Temporal Knowledge Analysis Tool

## Overview

The `time_travel_query` tool provides advanced temporal database capabilities for the LLMKG system, enabling users to query knowledge at any point in time, track entity evolution, detect changes over time periods, and analyze trends. This tool leverages the system's temporal tracking infrastructure to provide comprehensive time-based analysis of knowledge graph evolution and historical data exploration.

## Implementation Details

### Handler Location
- **Primary Handler**: `src/mcp/llm_friendly_server/handlers/temporal.rs`
- **Secondary Handler**: `src/mcp/llm_friendly_server/handlers/cognitive.rs` (legacy implementation)
- **Function**: `handle_time_travel_query`
- **Lines**: 21-151 (temporal.rs), 162-238 (cognitive.rs)

### Core Functionality

The tool implements sophisticated temporal analysis capabilities:

1. **Point-in-Time Queries**: Retrieve knowledge state at specific timestamps
2. **Evolution Tracking**: Monitor how entities change over time periods
3. **Temporal Comparison**: Compare knowledge states across different time points
4. **Change Detection**: Identify modifications, additions, and deletions
5. **Trend Analysis**: Discover patterns in knowledge evolution
6. **Historical Reconstruction**: Rebuild past states of the knowledge graph

### Input Schema

```json
{
  "type": "object",
  "properties": {
    "query_type": {
      "type": "string",
      "description": "Type of temporal query",
      "enum": ["point_in_time", "evolution_tracking", "temporal_comparison", "change_detection"],
      "default": "point_in_time"
    },
    "timestamp": {
      "type": "string",
      "description": "ISO timestamp for point-in-time queries",
      "format": "date-time"
    },
    "entity": {
      "type": "string",
      "description": "Entity to track over time",
      "maxLength": 200
    },
    "time_range": {
      "type": "object",
      "description": "Time range for comparison/evolution queries",
      "properties": {
        "start": {"type": "string", "format": "date-time"},
        "end": {"type": "string", "format": "date-time"}
      }
    }
  },
  "required": []
}
```

### Key Variables and Functions

#### Primary Handler Function
```rust
pub async fn handle_time_travel_query(
    _knowledge_engine: &Arc<RwLock<KnowledgeEngine>>,
    usage_stats: &Arc<RwLock<UsageStats>>,
    params: Value,
) -> std::result::Result<(Value, String, Vec<String>), String>
```

#### Input Processing and Validation
```rust
let query_type = params.get("query_type")
    .and_then(|v| v.as_str())
    .unwrap_or("point_in_time");

let entity = params.get("entity")
    .and_then(|v| v.as_str());

let timestamp = params.get("timestamp")
    .and_then(|v| v.as_str())
    .map(|ts| DateTime::parse_from_rfc3339(ts)
        .map_err(|e| format!("Invalid timestamp format: {}", e))
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
            (Some(Err(e)), _) => Err(format!("Invalid start time: {}", e)),
            (_, Some(Err(e))) => Err(format!("Invalid end time: {}", e)),
            _ => Ok(None)
        }
    })
    .transpose()?
    .flatten();
```

### Temporal Index Integration

#### Temporal Index Access
```rust
use crate::mcp::llm_friendly_server::temporal_tracking::{
    query_point_in_time, track_entity_evolution, detect_changes,
    TEMPORAL_INDEX
};
```

The tool integrates directly with the temporal tracking system that records all knowledge graph operations with timestamps.

#### Temporal Operations
The system records three types of temporal operations:
- **Create**: New knowledge added to the graph
- **Update**: Existing knowledge modified
- **Delete**: Knowledge removed from the graph

### Query Type Implementations

#### 1. Point-in-Time Queries
```rust
"point_in_time" => {
    let entity = entity.ok_or("Entity required for point_in_time query")?;
    let timestamp = timestamp.unwrap_or_else(Utc::now);
    query_point_in_time(&*TEMPORAL_INDEX, entity, timestamp).await
}
```

**Functionality:**
- Retrieves the state of an entity at a specific timestamp
- Shows what information was known about the entity at that time
- Reconstructs historical knowledge graph states
- Default timestamp: current time if not specified

**Use Cases:**
- "What did we know about Einstein on January 1, 2023?"
- "Show the state of our AI knowledge base last month"
- "What information was available about quantum physics in 2020?"

#### 2. Evolution Tracking
```rust
"evolution_tracking" => {
    let entity = entity.ok_or("Entity required for evolution_tracking query")?;
    let (start, end) = time_range.unwrap_or((
        DateTime::from_timestamp(0, 0).unwrap(),
        Utc::now()
    ));
    track_entity_evolution(&*TEMPORAL_INDEX, entity, Some(start), Some(end)).await
}
```

**Functionality:**
- Tracks how an entity's information changed over a time period
- Shows the sequence of modifications and updates
- Identifies key evolution milestones
- Default range: from beginning of time to now

**Use Cases:**
- "How has our understanding of AI evolved over the past year?"
- "Track the changes to Einstein's biography since 2022"
- "Show the evolution of quantum physics knowledge"

#### 3. Temporal Comparison
```rust
"temporal_comparison" => {
    if let Some((start, end)) = time_range {
        detect_changes(&*TEMPORAL_INDEX, start, end, entity).await
    } else {
        return Err("Time range required for temporal_comparison".to_string());
    }
}
```

**Functionality:**
- Compares knowledge states between two specific time points
- Identifies differences, additions, and deletions
- Provides detailed change analysis
- Requires explicit time range specification

**Use Cases:**
- "Compare our knowledge of AI between 2022 and 2024"
- "What changed in Einstein's information this month vs last month?"
- "Show differences in quantum physics knowledge before and after recent updates"

#### 4. Change Detection
```rust
"change_detection" => {
    let (start, end) = time_range.unwrap_or_else(|| {
        let end = Utc::now();
        let start = end - chrono::Duration::days(7);
        (start, end)
    });
    detect_changes(&*TEMPORAL_INDEX, start, end, entity).await
}
```

**Functionality:**
- Detects all changes within a specified time range
- Default range: last 7 days if not specified
- Can focus on specific entity or analyze all changes
- Provides change categorization and impact analysis

**Use Cases:**
- "What knowledge was added or modified this week?"
- "Detect any changes to Einstein's information in the last month"
- "Show all modifications made to the quantum physics domain"

### Temporal Result Structure

#### Temporal Query Result
```rust
// Inferred from usage patterns
struct TemporalQueryResult {
    query_type: String,
    results: Vec<serde_json::Value>,
    time_range: Option<(DateTime<Utc>, DateTime<Utc>)>,
    total_changes: usize,
    insights: Vec<String>,
    // Additional metadata
}
```

#### Result Components
**Query Results**: Specific temporal data matching the query
**Time Range**: Actual time period analyzed
**Change Count**: Number of modifications detected
**Insights**: Human-readable analysis of temporal patterns
**Metadata**: Query execution details and statistics

### Output Format

#### Comprehensive Temporal Response
```json
{
  "query_type": "evolution_tracking",
  "results": [
    {
      "timestamp": "2024-01-15T10:30:00Z",
      "operation": "create",
      "entity": "Einstein",
      "change": "Added birth_date: 1879-03-14",
      "confidence": 1.0
    },
    {
      "timestamp": "2024-02-01T14:20:00Z", 
      "operation": "update",
      "entity": "Einstein",
      "change": "Updated description: Added Nobel Prize information",
      "previous_value": "German physicist",
      "new_value": "German physicist, Nobel Prize winner 1921"
    }
  ],
  "time_range": {
    "start": "2024-01-01T00:00:00Z",
    "end": "2024-03-01T00:00:00Z"
  },
  "total_changes": 2,
  "insights": [
    "Entity information expanded significantly in February",
    "Most changes were additions rather than modifications",
    "High confidence scores indicate reliable information sources"
  ],
  "metadata": {
    "query_time_ms": 234,
    "data_points": 2,
    "temporal_span": {
      "days": 59,
      "hours": 1416,
      "human_readable": "59 days, 0 hours"
    }
  }
}
```

#### Human-Readable Message Format
```rust
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
```

**Example Human-Readable Output:**
```
Time Travel Query Results:
⏰ Query Type: evolution_tracking
📊 Data Points: 47
📈 Changes Detected: 12
🕰️ Time Span: 55 years
🔍 Key Insights: Nobel Prize 1921, relativity acceptance grew gradually, peak recognition 1919 eclipse
```

### Duration Formatting Utility

#### Human-Readable Duration Function
```rust
fn format_duration(duration: chrono::Duration) -> String {
    let days = duration.num_days();
    let hours = duration.num_hours() % 24;
    let minutes = duration.num_minutes() % 60;
    
    if days > 0 {
        format!("{} days, {} hours", days, hours)
    } else if hours > 0 {
        format!("{} hours, {} minutes", hours, minutes)
    } else {
        format!("{} minutes", minutes)
    }
}
```

### Error Handling

#### Query Type Validation
```rust
match query_type {
    "point_in_time" => { /* implementation */ }
    "evolution_tracking" => { /* implementation */ }
    "temporal_comparison" => { /* implementation */ }
    "change_detection" => { /* implementation */ }
    _ => return Err(format!("Unknown query type: {}", query_type))
}
```

#### Timestamp Validation
```rust
let timestamp = params.get("timestamp")
    .and_then(|v| v.as_str())
    .map(|ts| DateTime::parse_from_rfc3339(ts)
        .map_err(|e| format!("Invalid timestamp format: {}", e))
        .map(|dt| dt.with_timezone(&Utc)))
    .transpose()?;
```

#### Required Parameter Validation
```rust
let entity = entity.ok_or("Entity required for point_in_time query")?;
```

### Performance Characteristics

#### Complexity Analysis
- **Point-in-Time**: O(log n) where n is temporal index size
- **Evolution Tracking**: O(k) where k is changes for the entity
- **Temporal Comparison**: O(m) where m is changes in time range
- **Change Detection**: O(t) where t is total changes in range

#### Memory Usage
- **Result Storage**: Vectors for temporal data points
- **Time Range Calculation**: DateTime objects and duration calculations
- **Insight Generation**: String allocation for analysis results

#### Usage Statistics Impact
- **Weight**: 80 points per operation (complex temporal processing)
- **Operation Type**: `StatsOperation::ExecuteQuery`

### Integration Points

#### With Temporal Tracking System
```rust
use crate::mcp::llm_friendly_server::temporal_tracking::{
    query_point_in_time, track_entity_evolution, detect_changes,
    TEMPORAL_INDEX
};
```

#### With Chrono Date/Time Library
```rust
use chrono::{DateTime, Utc};
```

Comprehensive timestamp parsing and manipulation capabilities.

### Advanced Features

#### Automatic Time Range Defaults
```rust
let (start, end) = time_range.unwrap_or_else(|| {
    let end = Utc::now();
    let start = end - chrono::Duration::days(7);
    (start, end)
});
```

#### Multi-Granularity Analysis
The system provides insights at multiple time scales:
- **Minutes/Hours**: Real-time operation tracking
- **Days/Weeks**: Short-term evolution analysis
- **Months/Years**: Long-term trend identification
- **Historical**: Complete knowledge graph evolution

### Best Practices for Developers

1. **Time Range Specification**: Use appropriate time ranges for analysis scope
2. **Entity Focus**: Specify entities for targeted temporal analysis
3. **Query Type Selection**: Choose appropriate query type for desired insights
4. **Timestamp Format**: Use ISO 8601 format for timestamp parameters
5. **Performance Consideration**: Monitor query complexity for large time ranges

### Usage Examples

#### Point-in-Time Analysis
```json
{
  "query_type": "point_in_time",
  "entity": "Einstein",
  "timestamp": "2023-06-15T12:00:00Z"
}
```

#### Entity Evolution Tracking
```json
{
  "query_type": "evolution_tracking",
  "entity": "artificial_intelligence",
  "time_range": {
    "start": "2020-01-01T00:00:00Z",
    "end": "2024-01-01T00:00:00Z"
  }
}
```

#### Recent Change Detection
```json
{
  "query_type": "change_detection",
  "time_range": {
    "start": "2024-03-01T00:00:00Z",
    "end": "2024-03-31T23:59:59Z"
  }
}
```

#### Historical Comparison
```json
{
  "query_type": "temporal_comparison",
  "entity": "quantum_physics",
  "time_range": {
    "start": "2022-01-01T00:00:00Z",
    "end": "2024-01-01T00:00:00Z"
  }
}
```

### Suggestions System
```rust
let suggestions = vec![
    "Use 'evolution_tracking' to see how entities change over time".to_string(),
    "Compare different time periods with 'temporal_comparison'".to_string(),
    "Detect anomalies with 'change_detection' queries".to_string(),
];
```

### Research and Analysis Applications

#### Academic Research
- **Historical Analysis**: Tracking knowledge evolution in specific domains
- **Citation Analysis**: Understanding how concepts develop over time
- **Research Impact**: Measuring knowledge growth and change patterns

#### Business Intelligence
- **Knowledge Audit**: Understanding organizational knowledge evolution
- **Change Impact**: Analyzing effects of knowledge updates
- **Trend Analysis**: Identifying patterns in information development

#### Quality Assurance
- **Change Tracking**: Monitoring knowledge graph modifications
- **Version Control**: Understanding what changed when and why
- **Rollback Support**: Identifying specific changes for potential reversal

### Tool Integration Workflow

1. **Input Processing**: Parse and validate temporal query parameters
2. **Timestamp Handling**: Convert and validate ISO 8601 timestamps
3. **Query Routing**: Dispatch to appropriate temporal analysis function
4. **Temporal Index Access**: Query the temporal tracking system
5. **Result Processing**: Format and analyze temporal data points
6. **Insight Generation**: Create human-readable analysis of temporal patterns
7. **Duration Calculation**: Compute and format time spans for display
8. **Response Formatting**: Structure comprehensive temporal analysis results
9. **Usage Tracking**: Update system analytics for temporal query effectiveness

This tool provides essential temporal analysis capabilities for the LLMKG system, enabling comprehensive time-based exploration of knowledge graph evolution and historical data analysis through sophisticated temporal tracking integration.