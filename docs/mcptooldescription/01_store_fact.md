# store_fact - Simple Triple Storage Tool

## Overview

The `store_fact` tool is the most fundamental MCP tool in the LLMKG system, designed for storing simple knowledge as Subject-Predicate-Object triples. This tool forms the foundation of the knowledge graph by enabling users to add individual facts in a structured format.

## Implementation Details

### Handler Location
- **File**: `src/mcp/llm_friendly_server/handlers/storage.rs`
- **Function**: `handle_store_fact`
- **Lines**: 14-107

### Core Functionality

The tool processes incoming fact storage requests by:

1. **Input Validation**: Validates all required fields and enforces length constraints
2. **Triple Creation**: Creates a `Triple` object with metadata
3. **Duplicate Detection**: Checks for existing triples to determine create vs update operations
4. **Storage**: Persists the triple in the knowledge engine
5. **Temporal Tracking**: Records the operation in the temporal index for time-travel queries
6. **Usage Statistics**: Updates system usage metrics

### Input Schema

```json
{
  "type": "object",
  "properties": {
    "subject": {
      "type": "string",
      "description": "The entity the fact is about",
      "maxLength": 128
    },
    "predicate": {
      "type": "string", 
      "description": "The relationship or property",
      "maxLength": 64
    },
    "object": {
      "type": "string",
      "description": "What the subject is related to",
      "maxLength": 128
    },
    "confidence": {
      "type": "number",
      "description": "Confidence score (0.0 to 1.0)",
      "minimum": 0.0,
      "maximum": 1.0,
      "default": 1.0
    }
  },
  "required": ["subject", "predicate", "object"]
}
```

### Key Variables and Functions

#### Primary Handler Function
```rust
pub async fn handle_store_fact(
    knowledge_engine: &Arc<RwLock<KnowledgeEngine>>,
    usage_stats: &Arc<RwLock<UsageStats>>,
    params: Value,
) -> std::result::Result<(Value, String, Vec<String>), String>
```

#### Input Processing Variables
- `subject`: String extracted from params, limited to 128 characters
- `predicate`: String extracted from params, limited to 64 characters  
- `object`: String extracted from params, limited to 128 characters
- `confidence`: Float value (0.0-1.0), defaults to 1.0

#### Validation Logic
```rust
// Validate inputs
if subject.is_empty() || predicate.is_empty() || object.is_empty() {
    return Err("Subject, predicate, and object cannot be empty".to_string());
}

if subject.len() > 128 || object.len() > 128 {
    return Err("Subject and object must be 128 characters or less".to_string());
}

if predicate.len() > 64 {
    return Err("Predicate must be 64 characters or less".to_string());
}
```

#### Triple Creation
```rust
let triple = Triple::with_metadata(
    subject.to_string(),
    predicate.to_string(),
    object.to_string(),
    confidence,
    Some("user_input".to_string()),
).map_err(|e| format!("Failed to create triple: {}", e))?;
```

#### Duplicate Detection Logic
The system performs sophisticated duplicate detection:

```rust
let existing_query = TripleQuery {
    subject: Some(subject.to_string()),
    predicate: Some(predicate.to_string()),
    object: None,
    limit: 1,
    min_confidence: 0.0,
    include_chunks: false,
};

let (operation, previous_value) = if let Some(result) = existing_result {
    if let Some(existing_triple) = result.triples.first() {
        (TemporalOperation::Update, Some(existing_triple.object.clone()))
    } else {
        (TemporalOperation::Create, None)
    }
} else {
    (TemporalOperation::Create, None)
};
```

#### Temporal Tracking Integration
```rust
TEMPORAL_INDEX.record_operation(triple, operation, previous_value);
```

### Output Format

#### Success Response Data
```json
{
  "success": true,
  "node_id": "generated_node_id",
  "subject": "Einstein",
  "predicate": "is",
  "object": "scientist",
  "confidence": 1.0
}
```

#### Response Message Format
```rust
let message = format!("Stored fact: {} {} {}", subject, predicate, object);
```

#### Generated Suggestions
```rust
let suggestions = vec![
    format!("Explore connections with: explore_connections(start_entity=\"{}\")", subject),
    format!("Find related facts with: find_facts(subject=\"{}\")", subject),
];
```

### Dependencies and Imports

#### Core Dependencies
- `crate::core::triple::Triple` - Triple data structure
- `crate::core::knowledge_engine::KnowledgeEngine` - Main knowledge storage engine
- `crate::core::knowledge_types::TripleQuery` - Query structure for existing triples

#### Temporal System Integration
- `crate::mcp::llm_friendly_server::temporal_tracking::{TEMPORAL_INDEX, TemporalOperation}` - Time-travel functionality

#### Utility Functions
- `crate::mcp::llm_friendly_server::utils::{update_usage_stats, StatsOperation}` - Usage tracking

### Error Handling

The tool implements comprehensive error handling:

1. **Missing Fields**: Returns descriptive error for missing required fields
2. **Empty Values**: Prevents storage of empty strings
3. **Length Validation**: Enforces character limits for each field
4. **Triple Creation Errors**: Handles errors from the Triple constructor
5. **Storage Errors**: Manages database storage failures

### Performance Characteristics

#### Complexity
- **Time Complexity**: O(log n) for duplicate detection query + O(1) for storage
- **Space Complexity**: O(1) per triple stored

#### Usage Statistics Impact
- **Weight**: 10 points per operation (defined in `StatsOperation::StoreTriple`)
- **Tracked Metrics**: Total operations, response times, storage counts

### Integration Points

#### With Temporal System
Every fact storage operation is recorded in the temporal index, enabling:
- Time-travel queries to see facts at specific points in time
- Evolution tracking for entities
- Change detection across time ranges

#### With Knowledge Engine
Direct integration with the core `KnowledgeEngine`:
```rust
let node_id = engine.store_triple(triple.clone(), None)
    .map_err(|e| format!("Failed to store triple: {}", e))?;
```

#### With Usage Statistics
Each operation updates system-wide usage statistics:
```rust
let _ = update_usage_stats(usage_stats, StatsOperation::StoreTriple, 10).await;
```

### Best Practices for Developers

1. **Predicate Consistency**: Keep predicates short (1-3 words) and reusable
2. **Entity Naming**: Use consistent naming conventions (e.g., "New_York" not "new york")
3. **Confidence Levels**: Use 0.8 instead of 1.0 when uncertain
4. **Batch Operations**: For multiple facts, consider using `store_knowledge` instead

### Usage Examples

#### Basic Fact Storage
```json
{
  "subject": "Einstein",
  "predicate": "is",
  "object": "scientist"
}
```

#### Fact with Confidence
```json
{
  "subject": "Python",
  "predicate": "created_by",
  "object": "Guido van Rossum",
  "confidence": 1.0
}
```

### Tool Integration Workflow

1. **Pre-Storage**: Input validation and sanitization
2. **Duplicate Check**: Query existing triples to determine operation type
3. **Storage**: Persist triple in knowledge engine
4. **Post-Storage**: Update temporal index and usage statistics
5. **Response**: Return success confirmation with suggestions

This tool serves as the foundation for all knowledge graph operations in the LLMKG system, providing reliable, validated triple storage with comprehensive tracking and integration capabilities.