# validate_knowledge - Knowledge Quality Assurance Tool

## Overview

The `validate_knowledge` tool provides comprehensive validation and quality assurance for stored knowledge in the LLMKG system. It performs consistency checks, conflict detection, quality assessment, and completeness analysis to ensure the integrity and reliability of the knowledge graph. The tool supports both entity-specific validation and system-wide quality assessment with optional automatic issue resolution.

## Implementation Details

### Handler Location
- **File**: `src/mcp/llm_friendly_server/handlers/advanced.rs`
- **Function**: `handle_validate_knowledge`
- **Lines**: 618-760

### Core Functionality

The tool implements multi-dimensional knowledge validation:

1. **Consistency Validation**: Ensures logical consistency across stored facts
2. **Conflict Detection**: Identifies contradictory information
3. **Quality Assessment**: Evaluates data quality and confidence levels
4. **Completeness Analysis**: Finds missing information gaps
5. **Comprehensive Metrics**: Provides detailed quality breakdowns
6. **Automatic Remediation**: Optional issue fixing capabilities

### Input Schema

```json
{
  "type": "object",
  "properties": {
    "validation_type": {
      "type": "string",
      "description": "What to validate",
      "enum": ["consistency", "conflicts", "quality", "completeness", "all"],
      "default": "all"
    },
    "entity": {
      "type": "string",
      "description": "Specific entity to validate (optional)",
      "maxLength": 128
    },
    "fix_issues": {
      "type": "boolean",
      "description": "Attempt to automatically fix found issues",
      "default": false
    },
    "scope": {
      "type": "string",
      "description": "Validation scope (standard or comprehensive)",
      "enum": ["standard", "comprehensive"],
      "default": "standard"
    },
    "include_metrics": {
      "type": "boolean",
      "description": "Include detailed quality metrics",
      "default": false
    },
    "quality_threshold": {
      "type": "number",
      "description": "Minimum quality threshold",
      "minimum": 0.0,
      "maximum": 1.0,
      "default": 0.7
    },
    "importance_threshold": {
      "type": "number", 
      "description": "Minimum importance threshold",
      "minimum": 0.0,
      "maximum": 1.0,
      "default": 0.6
    }
  },
  "additionalProperties": false
}
```

### Key Variables and Functions

#### Primary Handler Function
```rust
pub async fn handle_validate_knowledge(
    knowledge_engine: &Arc<RwLock<KnowledgeEngine>>,
    usage_stats: &Arc<RwLock<UsageStats>>,
    params: Value,
) -> std::result::Result<(Value, String, Vec<String>), String>
```

#### Input Processing Variables
```rust
let validation_type = params.get("validation_type").and_then(|v| v.as_str())
    .unwrap_or("all");
let entity = params.get("entity").and_then(|v| v.as_str());
let fix_issues = params.get("fix_issues")
    .and_then(|v| v.as_bool())
    .unwrap_or(false);
let scope = params.get("scope").and_then(|v| v.as_str()).unwrap_or("standard");
let include_metrics = params.get("include_metrics").and_then(|v| v.as_bool()).unwrap_or(false);
let quality_threshold = params.get("quality_threshold").and_then(|v| v.as_f64()).unwrap_or(0.7) as f32;
let importance_threshold = params.get("importance_threshold").and_then(|v| v.as_f64()).unwrap_or(0.6) as f32;
```

### Validation Type System

#### Validation Type Validation
```rust
if !["consistency", "conflicts", "quality", "completeness", "all"].contains(&validation_type) {
    return Err("Invalid validation_type. Must be one of: consistency, conflicts, quality, completeness, all".to_string());
}
```

### Triple Query Construction

#### Entity-Specific Validation
```rust
let query = if let Some(e) = entity {
    TripleQuery {
        subject: Some(e.to_string()),
        predicate: None,
        object: None,
        limit: 100,
        min_confidence: 0.0,
        include_chunks: false,
    }
} else {
    TripleQuery {
        subject: None,
        predicate: None,
        object: None,
        limit: 100,
        min_confidence: 0.0,
        include_chunks: false,
    }
};
```

### Validation Methods

#### 1. Consistency Validation
```rust
use crate::mcp::llm_friendly_server::validation::validate_consistency;

if ["consistency", "all"].contains(&validation_type) {
    let consistency_result = validate_consistency(&triples.triples, &triples.triples).await
        .map_err(|e| format!("Consistency validation failed: {}", e))?;
    validation_results.insert("consistency".to_string(), json!({
        "passed": consistency_result.is_valid,
        "confidence": consistency_result.confidence,
        "issues": consistency_result.conflicts
    }));
}
```

**Consistency Checks Include:**
- Referential integrity validation
- Logical coherence assessment
- Cross-reference verification
- Temporal consistency analysis

#### 2. Conflict Detection
```rust
if ["conflicts", "all"].contains(&validation_type) {
    let conflicts_result = validate_consistency(&triples.triples, &triples.triples).await
        .map_err(|e| format!("Conflict validation failed: {}", e))?;
    validation_results.insert("conflicts".to_string(), json!({
        "found": conflicts_result.conflicts.len(),
        "conflicts": conflicts_result.conflicts
    }));
}
```

**Conflict Types Detected:**
- Contradictory facts about the same entity
- Inconsistent relationship assertions
- Temporal impossibilities
- Logical contradictions

#### 3. Quality Assessment
```rust
use crate::mcp::llm_friendly_server::validation::validate_triple;

if ["quality", "all"].contains(&validation_type) {
    let mut quality_score = 1.0;
    let mut quality_issues = Vec::new();
    
    for triple in &triples.triples {
        let triple_validation = validate_triple(triple).await
            .map_err(|e| format!("Triple validation failed: {}", e))?;
        quality_score *= triple_validation.confidence;
        quality_issues.extend(triple_validation.validation_notes);
    }
    
    validation_results.insert("quality".to_string(), json!({
        "score": (quality_score * 10.0).min(10.0),
        "issues": quality_issues
    }));
}
```

**Quality Metrics:**
- **Confidence Scores**: Individual triple confidence assessment
- **Data Completeness**: Missing field detection
- **Format Validation**: Structure and syntax checking
- **Source Verification**: Attribution and provenance validation

#### 4. Completeness Analysis
```rust
use crate::mcp::llm_friendly_server::validation::validate_completeness;

if ["completeness", "all"].contains(&validation_type) && entity.is_some() {
    let missing = validate_completeness(entity.unwrap(), &triples.triples).await
        .map_err(|e| format!("Completeness validation failed: {}", e))?;
    validation_results.insert("completeness".to_string(), json!({
        "missing_info": missing,
        "is_complete": missing.is_empty()
    }));
}
```

**Completeness Checks:**
- Essential attribute presence
- Standard relationship completeness
- Domain-specific requirement fulfillment
- Information density analysis

### Comprehensive Quality Metrics

#### Advanced Quality Assessment
```rust
let mut quality_metrics = None;
if scope == "comprehensive" || include_metrics {
    quality_metrics = Some(generate_quality_metrics(
        &triples.triples,
        quality_threshold,
        importance_threshold,
        &engine
    ).await?);
}
```

#### Quality Metrics Generation Function
```rust
async fn generate_quality_metrics(
    triples: &[Triple],
    quality_threshold: f32,
    _importance_threshold: f32,
    _engine: &KnowledgeEngine,
) -> std::result::Result<Value, String>
```

**Comprehensive Metrics Include:**

**1. Importance Scoring**
```rust
// Calculate importance scores for entities
let mut entity_connections: HashMap<String, usize> = HashMap::new();
let mut entity_confidence: HashMap<String, Vec<f32>> = HashMap::new();

for triple in triples {
    *entity_connections.entry(triple.subject.clone()).or_insert(0) += 1;
    *entity_connections.entry(triple.object.clone()).or_insert(0) += 1;
    
    entity_confidence.entry(triple.subject.clone())
        .or_insert_with(Vec::new)
        .push(triple.confidence);
    entity_confidence.entry(triple.object.clone())
        .or_insert_with(Vec::new)
        .push(triple.confidence);
}
```

**2. Quality Level Classification**
```rust
let importance = (connections.clone() as f32 / 10.0).min(1.0) * avg_confidence;

importance_scores.push(json!({
    "entity": entity,
    "importance": importance,
    "connections": connections,
    "quality_level": if importance > 0.8 { "Excellent" } 
                  else if importance > 0.6 { "Good" }
                  else if importance > 0.4 { "Fair" }
                  else { "Poor" }
}));
```

**3. Content Quality Analysis**
```rust
let total_triples = triples.len();
let high_confidence_triples = triples.iter().filter(|t| t.confidence > 0.8).count();
let avg_confidence = if total_triples > 0 {
    triples.iter().map(|t| t.confidence).sum::<f32>() / total_triples as f32
} else {
    0.0
};

content_quality = json!({
    "total_facts": total_triples,
    "high_quality_facts": high_confidence_triples,
    "average_confidence": avg_confidence,
    "quality_ratio": if total_triples > 0 { 
        high_confidence_triples as f32 / total_triples as f32 
    } else { 0.0 }
});
```

**4. Knowledge Density Assessment**
```rust
let avg_connections = if !entity_connections.is_empty() {
    entity_connections.values().sum::<usize>() as f32 / entity_connections.len() as f32
} else {
    0.0
};

let highly_connected = entity_connections.iter()
    .filter(|(_, &count)| count > 5)
    .map(|(entity, count)| json!({
        "entity": entity,
        "connections": count
    }))
    .collect::<Vec<_>>();

let isolated = entity_connections.iter()
    .filter(|(_, &count)| count == 1)
    .map(|(entity, count)| json!({
        "entity": entity,
        "connections": count
    }))
    .collect::<Vec<_>>();
```

### Output Format

#### Standard Validation Response
```json
{
  "validation_type": "all",
  "entity": null,
  "fix_issues": false,
  "timestamp": "2024-01-15T10:30:00Z",
  "consistency": {
    "passed": true,
    "confidence": 0.95,
    "issues": []
  },
  "conflicts": {
    "found": 2,
    "conflicts": [
      "Einstein birth date: '1879' vs '1878' (different sources)",
      "Python creation date: '1989' vs '1991'"
    ]
  },
  "quality": {
    "score": 8.7,
    "issues": [
      "Missing confidence scores for 23 relationships",
      "15 entities lack descriptions"
    ]
  },
  "completeness": {
    "missing_info": [
      "Einstein's educational background",
      "Einstein's family information"
    ],
    "is_complete": false
  }
}
```

#### Comprehensive Quality Metrics
```json
{
  "quality_metrics": {
    "importance_scores": [
      {
        "entity": "Einstein",
        "importance": 0.92,
        "connections": 15,
        "quality_level": "Excellent"
      }
    ],
    "content_quality": {
      "total_facts": 1247,
      "high_quality_facts": 892,
      "average_confidence": 0.87,
      "quality_ratio": 0.715
    },
    "knowledge_density": {
      "average_connections": 4.2,
      "total_entities": 523,
      "density_score": 0.42,
      "density_distribution": {
        "highly_connected": 67,
        "moderately_connected": 234,
        "isolated": 89
      }
    },
    "below_threshold_entities": [
      {
        "entity": "UnknownEntity",
        "confidence": 0.45,
        "below_by": 0.25
      }
    ]
  }
}
```

### Human-Readable Output

#### Validation Results Formatting
```rust
fn format_validation_results(results: &HashMap<String, Value>) -> String {
    let mut output = String::new();
    
    if let Some(consistency) = results.get("consistency") {
        output.push_str(&format!("**Consistency**: {}\n",
            if consistency["passed"].as_bool().unwrap_or(false) { "✓ Passed" } else { "✗ Failed" }
        ));
        if let Some(issues) = consistency["issues"].as_array() {
            for issue in issues {
                output.push_str(&format!("  - {}\n", issue.as_str().unwrap_or("")));
            }
        }
    }
    
    if let Some(conflicts) = results.get("conflicts") {
        let count = conflicts["found"].as_u64().unwrap_or(0);
        output.push_str(&format!("\n**Conflicts**: {} issues found\n", count));
        if let Some(conflict_list) = conflicts["conflicts"].as_array() {
            for (i, conflict) in conflict_list.iter().take(3).enumerate() {
                output.push_str(&format!("  {}. {}\n", i + 1, conflict.as_str().unwrap_or("")));
            }
        }
    }
    
    output
}
```

**Example Formatted Output:**
```
Validation Results:

**Consistency**: ✓ Passed
- All entity references valid
- No orphaned relationships

**Conflicts**: ⚠ 2 issues found
1. Einstein birth date: '1879' vs '1878' (different sources)
2. Python creation date: '1989' vs '1991'

**Quality**: ✓ Good (score: 8.7/10)
- Average confidence: 0.87
- Most facts have sources

**Completeness**: ⚠ Could improve
- 15 entities missing descriptions
- 23 relationships lack confidence scores
```

### Error Handling

#### Validation Process Errors
```rust
let consistency_result = validate_consistency(&triples.triples, &triples.triples).await
    .map_err(|e| format!("Consistency validation failed: {}", e))?;
```

#### Quality Metrics Generation Errors
```rust
if let Some(metrics) = quality_metrics {
    data["quality_metrics"] = metrics;
} else {
    // Handle case where comprehensive metrics failed
}
```

### Performance Characteristics

#### Complexity Analysis
- **Consistency Check**: O(n²) for pairwise conflict detection
- **Quality Assessment**: O(n) for individual triple validation  
- **Completeness Check**: O(n) for entity analysis
- **Comprehensive Metrics**: O(n log n) for connection analysis

#### Memory Usage
- **Validation Results**: HashMap storage for results
- **Quality Metrics**: Complex nested JSON structures
- **Temporary Collections**: Entity and relationship maps

#### Usage Statistics Impact
- **Weight**: 40 points per operation (high complexity)
- **Operation Type**: `StatsOperation::ExecuteQuery`

### Integration Points

#### With Validation Engine
```rust
use crate::mcp::llm_friendly_server::validation::{
    validate_consistency, validate_completeness, validate_triple
};
```

#### With Knowledge Engine
Direct access for comprehensive analysis:
```rust
let triples = engine.query_triples(query)
    .map_err(|e| format!("Failed to query triples: {}", e))?;
```

### Best Practices for Developers

1. **Regular Validation**: Run validation periodically to maintain data quality
2. **Entity-Specific Focus**: Use entity parameter for targeted validation
3. **Comprehensive Analysis**: Enable metrics for detailed quality assessment
4. **Threshold Tuning**: Adjust quality thresholds based on domain requirements
5. **Issue Resolution**: Use fix_issues carefully and review changes

### Usage Examples

#### Complete System Validation
```json
{
  "validation_type": "all",
  "scope": "comprehensive",
  "include_metrics": true
}
```

#### Entity-Specific Quality Check
```json
{
  "validation_type": "all",
  "entity": "Einstein",
  "quality_threshold": 0.8
}
```

#### Conflict Detection Only
```json
{
  "validation_type": "conflicts",
  "fix_issues": false
}
```

### Suggestions System
```rust
let suggestions = vec![
    "Run validation periodically to maintain data quality".to_string(),
    "Focus on fixing high-priority issues first".to_string(),
    "Use fix_issues=true with caution, review all changes".to_string(),
];
```

### Tool Integration Workflow

1. **Input Processing**: Validate parameters and determine validation scope
2. **Data Retrieval**: Query relevant triples based on entity specificity
3. **Validation Execution**: Run selected validation types against retrieved data
4. **Quality Metrics**: Generate comprehensive metrics if requested
5. **Issue Identification**: Catalog all detected problems and inconsistencies
6. **Automatic Remediation**: Apply fixes if enabled and safe to do so
7. **Result Formatting**: Structure validation results for both API and human consumption
8. **Usage Tracking**: Update system analytics for validation system effectiveness

This tool ensures the integrity and quality of knowledge stored in the LLMKG system, providing comprehensive validation capabilities essential for maintaining reliable and trustworthy knowledge graphs.