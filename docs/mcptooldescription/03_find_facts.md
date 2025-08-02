# find_facts - Pattern-Based Triple Retrieval Tool

## Overview

The `find_facts` tool provides precise pattern-based querying capabilities for retrieving stored triples from the LLMKG knowledge graph. It allows users to search for facts by specifying any combination of subject, predicate, or object patterns, making it the primary tool for structured knowledge retrieval and exploration.

## Implementation Details

### Handler Location
- **File**: `src/mcp/llm_friendly_server/handlers/query.rs`
- **Function**: `handle_find_facts`
- **Lines**: 12-96

### Core Functionality

The tool implements flexible pattern matching for triple retrieval:

1. **Pattern Validation**: Ensures at least one search parameter is provided
2. **Query Construction**: Builds structured queries for the knowledge engine
3. **Result Processing**: Formats and limits retrieved triples
4. **Response Generation**: Provides human-readable fact displays
5. **Usage Tracking**: Records query operations for analytics

### Input Schema

```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "object",
      "description": "Query parameters - must provide at least one",
      "properties": {
        "subject": {
          "type": "string",
          "description": "Find facts about this subject"
        },
        "predicate": {
          "type": "string",
          "description": "Find facts with this relationship type"
        },
        "object": {
          "type": "string",
          "description": "Find facts pointing to this object"
        }
      },
      "minProperties": 1,
      "additionalProperties": false
    },
    "limit": {
      "type": "integer",
      "description": "Maximum number of facts to return",
      "minimum": 1,
      "maximum": 100,
      "default": 10
    }
  },
  "required": ["query"],
  "additionalProperties": false
}
```

### Key Variables and Functions

#### Primary Handler Function
```rust
pub async fn handle_find_facts(
    knowledge_engine: &Arc<RwLock<KnowledgeEngine>>,
    usage_stats: &Arc<RwLock<UsageStats>>,
    params: Value,
) -> std::result::Result<(Value, String, Vec<String>), String>
```

#### Input Processing Variables
```rust
let query = params.get("query")
    .ok_or_else(|| "Missing required 'query' parameter".to_string())?;

let subject = query.get("subject").and_then(|v| v.as_str());
let predicate = query.get("predicate").and_then(|v| v.as_str());
let object = query.get("object").and_then(|v| v.as_str());
let limit = params.get("limit")
    .and_then(|v| v.as_u64())
    .unwrap_or(10)
    .min(100) as usize;
```

#### Query Validation Logic
```rust
// At least one field must be specified (enforced by minProperties in schema)
if subject.is_none() && predicate.is_none() && object.is_none() {
    return Err("At least one of subject, predicate, or object must be specified in the query".to_string());
}
```

#### Triple Query Construction
```rust
let query = TripleQuery {
    subject: subject.map(|s| s.to_string()),
    predicate: predicate.map(|p| p.to_string()),
    object: object.map(|o| o.to_string()),
    limit: 100,  // Internal limit for performance
    min_confidence: 0.0,  // Accept all confidence levels
    include_chunks: false,  // Only return triples, not chunks
};
```

#### Result Processing
```rust
let facts: Vec<_> = triples.triples.iter().map(|t| {
    json!({
        "subject": &t.subject,
        "predicate": &t.predicate,
        "object": &t.object,
        "confidence": 1.0 // Would come from metadata in full implementation
    })
}).collect();
```

### Query Execution Process

#### 1. Parameter Extraction and Validation
The system extracts search parameters from the input and validates that at least one parameter is specified:

```rust
let subject = query.get("subject").and_then(|v| v.as_str());
let predicate = query.get("predicate").and_then(|v| v.as_str());
let object = query.get("object").and_then(|v| v.as_str());
```

#### 2. Knowledge Engine Query
The query is executed against the knowledge engine using the `TripleQuery` structure:

```rust
let engine = knowledge_engine.read().await;
match engine.query_triples(query) {
    Ok(triples) => {
        // Process results
    }
    Err(e) => Err(format!("Query failed: {}", e))
}
```

#### 3. Result Formatting
Retrieved triples are formatted into a structured JSON response:

```rust
let facts: Vec<_> = triples.triples.iter().map(|t| {
    json!({
        "subject": &t.subject,
        "predicate": &t.predicate,
        "object": &t.object,
        "confidence": 1.0
    })
}).collect();
```

### Display Formatting

#### Fact Display Function
```rust
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
```

#### Message Generation Logic
```rust
let message = if triples.triples.is_empty() {
    "No facts found matching your query".to_string()
} else {
    format!("Found {} fact{}:\n{}", 
        triples.triples.len(),
        if triples.triples.len() == 1 { "" } else { "s" },
        format_facts_for_display(&triples.triples, 5)
    )
};
```

### Output Format

#### Success Response Data
```json
{
  "facts": [
    {
      "subject": "Einstein",
      "predicate": "is",
      "object": "scientist",
      "confidence": 1.0
    },
    {
      "subject": "Einstein",
      "predicate": "invented",
      "object": "relativity",
      "confidence": 1.0
    }
  ],
  "count": 2,
  "limit": 10,
  "query": {
    "subject": "Einstein",
    "predicate": null,
    "object": null
  }
}
```

#### Human-Readable Message
```
Found 2 facts:
1. Einstein is scientist
2. Einstein invented relativity
```

#### Contextual Suggestions

**When Results Found:**
```rust
let suggestions = vec![
    "Use 'explore_connections' to find related entities".to_string(),
    "Try 'ask_question' to understand these facts in context".to_string(),
];
```

**When No Results Found:**
```rust
let suggestions = vec![
    "Try using fewer constraints in your query".to_string(),
    "Check spelling and capitalization of entity names".to_string(),
    "Use 'ask_question' for natural language queries".to_string(),
];
```

### Query Pattern Examples

#### Subject-Based Queries
Find all facts about a specific entity:
```json
{
  "query": {
    "subject": "Einstein"
  },
  "limit": 5
}
```

#### Predicate-Based Queries
Find all facts with a specific relationship:
```json
{
  "query": {
    "predicate": "invented"
  },
  "limit": 10
}
```

#### Object-Based Queries
Find all facts pointing to a specific value:
```json
{
  "query": {
    "object": "scientist"
  }
}
```

#### Combined Queries
More specific searches using multiple parameters:
```json
{
  "query": {
    "subject": "Einstein",
    "predicate": "won"
  }
}
```

### Error Handling

The tool implements robust error handling for various scenarios:

#### 1. Missing Query Parameter
```rust
let query = params.get("query")
    .ok_or_else(|| "Missing required 'query' parameter".to_string())?;
```

#### 2. Empty Query Validation
```rust
if subject.is_none() && predicate.is_none() && object.is_none() {
    return Err("At least one of subject, predicate, or object must be specified in the query".to_string());
}
```

#### 3. Query Execution Errors
```rust
match engine.query_triples(query) {
    Ok(triples) => {
        // Process success case
    }
    Err(e) => Err(format!("Query failed: {}", e))
}
```

### Performance Characteristics

#### Query Complexity
- **Index Lookups**: O(log n) for indexed fields (subject, predicate, object)
- **Result Processing**: O(k) where k is the number of matching results
- **Limit Enforcement**: Results are limited to 100 internally for performance

#### Memory Usage
- **Result Set**: Temporary storage for matching triples
- **JSON Serialization**: Memory for response formatting
- **Display Formatting**: String allocation for human-readable output

#### Usage Statistics Impact
- **Weight**: 10 points per query operation
- **Operation Type**: `StatsOperation::ExecuteQuery`
- **Tracking**: Query count, response times, result sizes

### Integration Points

#### With Knowledge Engine
Direct integration with the core storage system:
```rust
let engine = knowledge_engine.read().await;
match engine.query_triples(query) {
    // Query execution
}
```

#### With Triple Query System
Uses the structured `TripleQuery` for precise pattern matching:
```rust
use crate::core::knowledge_types::TripleQuery;
```

#### With Usage Analytics
Each query updates system metrics:
```rust
let _ = update_usage_stats(usage_stats, StatsOperation::ExecuteQuery, 10).await;
```

### Advanced Query Features

#### Wildcard Patterns
Omitting parameters creates wildcard behavior:
- No subject: Find facts with any subject
- No predicate: Find facts with any relationship
- No object: Find facts with any object

#### Confidence Filtering
The system supports confidence-based filtering (currently set to 0.0 to accept all):
```rust
min_confidence: 0.0,
```

#### Result Limiting
Flexible result limiting for performance control:
- User-specified limit (default: 10, max: 100)
- Internal engine limit for safety
- Display truncation for readability

### Best Practices for Developers

1. **Pattern Specificity**: Use multiple parameters for more precise queries
2. **Result Limits**: Set appropriate limits based on use case requirements
3. **Error Handling**: Always handle empty result cases gracefully
4. **Performance**: Consider query complexity when building compound searches
5. **User Experience**: Provide helpful suggestions when no results are found

### Usage Examples

#### Find All Facts About Einstein
```json
{
  "query": {
    "subject": "Einstein"
  },
  "limit": 5
}
```

**Expected Output:**
```
Found 5 facts:
1. Einstein is scientist
2. Einstein invented relativity
3. Einstein born_in Germany
4. Einstein won Nobel_Prize
5. Einstein died_in 1955
```

#### Find All Invention Relationships
```json
{
  "query": {
    "predicate": "invented"
  },
  "limit": 3
}
```

**Expected Output:**
```
Found 3 facts:
1. Einstein invented relativity
2. Edison invented light_bulb
3. Tesla invented AC_motor
```

#### Find What Makes Someone a Scientist
```json
{
  "query": {
    "predicate": "is",
    "object": "scientist"
  }
}
```

### Tool Integration Workflow

1. **Input Validation**: Verify query structure and parameter requirements
2. **Pattern Construction**: Build TripleQuery from user parameters
3. **Engine Query**: Execute search against knowledge graph
4. **Result Processing**: Format and limit retrieved triples
5. **Display Generation**: Create human-readable fact lists
6. **Suggestion Generation**: Provide contextual next steps
7. **Usage Tracking**: Update system analytics and performance metrics

This tool provides the foundation for structured knowledge retrieval in the LLMKG system, enabling precise pattern-based queries with flexible wildcard support and comprehensive result formatting.