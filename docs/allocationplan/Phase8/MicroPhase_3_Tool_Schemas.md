# MicroPhase 3: Tool Schemas and Validation

**Duration**: 4-6 hours (25 micro-tasks)  
**Priority**: High - Required for MCP compliance  
**Prerequisites**: MicroPhase 1 (Foundation)

## Overview

Define comprehensive MCP tool schemas with Zod validation, input/output types, and proper JSON-RPC 2.0 compliance for all 7 neuromorphic memory tools. Each schema broken into atomic micro-tasks for rapid AI completion.

## AI-Actionable Micro-Tasks

### Micro-Task 3.1.1: Create Core MCP Types
**Estimated Time**: 15 minutes  
**Expected Deliverable**: `src/mcp/schemas/base_types.rs` (Core structs only)

**Task Prompt for AI**: Create the foundational MCP input/output types for tool communication. Focus only on the core data structures without validation logic.

```rust
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPToolInput {
    pub tool_name: String,
    pub parameters: serde_json::Value,
    pub request_id: String,
    pub session_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPToolOutput {
    pub result: serde_json::Value,
    pub request_id: String,
    pub processing_metadata: ProcessingMetadata,
    pub neuromorphic_trace: NeuromorphicTrace,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingMetadata {
    pub processing_time_ms: f32,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub server_version: String,
    pub success: bool,
    pub error_message: Option<String>,
}
```

**Verification**: Compiles successfully + serde serialization works

### Micro-Task 3.1.2: Create Neuromorphic Trace Types
**Estimated Time**: 18 minutes  
**Expected Deliverable**: Add neuromorphic structures to `base_types.rs`

**Task Prompt for AI**: Add the neuromorphic-specific data structures that capture brain-like processing traces. Build on existing base_types.rs file.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuromorphicTrace {
    pub cortical_consensus: CorticalConsensusInfo,
    pub neural_pathway: Vec<String>,
    pub activation_strengths: HashMap<String, f32>,
    pub ttfs_encoding_time_ms: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorticalConsensusInfo {
    pub winning_column: String,
    pub consensus_strength: f32,
    pub inhibition_applied: bool,
    pub column_activations: Vec<ColumnActivation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnActivation {
    pub column_type: String,
    pub activation_strength: f32,
    pub confidence: f32,
    pub processing_time_ms: f32,
}
```

**Verification**: Compiles + neuromorphic fields accessible

### Micro-Task 3.1.3: Add Validation Framework
**Estimated Time**: 15 minutes  
**Expected Deliverable**: Complete `base_types.rs` with validation traits

**Task Prompt for AI**: Add validation traits and error types to enable schema validation across all MCP tools.

```rust
// Validation traits for schema compliance
pub trait ValidatedInput {
    fn validate(&self) -> Result<(), ValidationError>;
}

pub trait ValidatedOutput {
    fn validate(&self) -> Result<(), ValidationError>;
}

#[derive(Debug, thiserror::Error)]
pub enum ValidationError {
    #[error("Missing required field: {0}")]
    MissingField(String),
    
    #[error("Invalid field value: {field} - {reason}")]
    InvalidValue { field: String, reason: String },
    
    #[error("Schema validation failed: {0}")]
    SchemaError(String),
}
```

**Verification**: Traits compile + error types work with thiserror

### Micro-Task 3.2.1: Create StoreMemory Input Schema
**Estimated Time**: 15 minutes  
**Expected Deliverable**: `src/mcp/schemas/store_memory_schema.rs` (Input types only)

**Task Prompt for AI**: Create input schema for storing memories in the neuromorphic system. Focus only on input struct and basic validation.

```rust
use super::base_types::{ValidatedInput, ValidationError};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreMemoryInput {
    #[serde(description = "Content to store in neuromorphic memory")]
    pub content: String,
    
    #[serde(description = "Contextual information for better allocation")]
    pub context: Option<String>,
    
    #[serde(description = "Source attribution for provenance tracking")]
    pub source: Option<String>,
    
    #[serde(description = "Confidence score between 0.0 and 1.0")]
    pub confidence: Option<f32>,
    
    #[serde(description = "Tags for categorization")]
    pub tags: Option<Vec<String>>,
    
    #[serde(description = "Importance weight for allocation priority")]
    pub importance: Option<f32>,
}
```

**Verification**: Compiles + serde descriptions work

### Micro-Task 3.2.2: Add StoreMemory Input Validation
**Estimated Time**: 18 minutes  
**Expected Deliverable**: Complete input validation for StoreMemoryInput

**Task Prompt for AI**: Implement comprehensive validation logic for StoreMemoryInput. Add to existing store_memory_schema.rs file.

```rust
impl ValidatedInput for StoreMemoryInput {
    fn validate(&self) -> Result<(), ValidationError> {
        // Content validation
        if self.content.trim().is_empty() {
            return Err(ValidationError::InvalidValue {
                field: "content".to_string(),
                reason: "Content cannot be empty".to_string(),
            });
        }
        
        if self.content.len() > 10000 {
            return Err(ValidationError::InvalidValue {
                field: "content".to_string(),
                reason: "Content exceeds maximum length of 10000 characters".to_string(),
            });
        }
        
        // Confidence validation
        if let Some(confidence) = self.confidence {
            if !(0.0..=1.0).contains(&confidence) {
                return Err(ValidationError::InvalidValue {
                    field: "confidence".to_string(),
                    reason: "Confidence must be between 0.0 and 1.0".to_string(),
                });
            }
        }
        
        // Importance validation
        if let Some(importance) = self.importance {
            if !(0.0..=1.0).contains(&importance) {
                return Err(ValidationError::InvalidValue {
                    field: "importance".to_string(),
                    reason: "Importance must be between 0.0 and 1.0".to_string(),
                });
            }
        }
        
        // Tags validation
        if let Some(ref tags) = self.tags {
            if tags.len() > 20 {
                return Err(ValidationError::InvalidValue {
                    field: "tags".to_string(),
                    reason: "Maximum 20 tags allowed".to_string(),
                });
            }
            
            for tag in tags {
                if tag.len() > 50 {
                    return Err(ValidationError::InvalidValue {
                        field: "tags".to_string(),
                        reason: "Individual tag cannot exceed 50 characters".to_string(),
                    });
                }
            }
        }
        
        Ok(())
    }
}
```

**Verification**: Input validation passes all test cases

### Micro-Task 3.2.3: Create StoreMemory Output Schema
**Estimated Time**: 16 minutes  
**Expected Deliverable**: Add output types to `store_memory_schema.rs`

**Task Prompt for AI**: Create output schema structures for store memory operation. Add to existing store_memory_schema.rs file.

```rust
use super::base_types::{ValidatedOutput, ValidationError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreMemoryOutput {
    pub memory_id: String,
    pub allocation_path: Vec<String>,
    pub cortical_decision: CorticalDecision,
    pub synaptic_changes: SynapticChanges,
    pub storage_metadata: StorageMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorticalDecision {
    pub primary_column: String,
    pub decision_confidence: f32,
    pub alternative_paths: Vec<AlternativePath>,
    pub inhibition_strength: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlternativePath {
    pub column_type: String,
    pub activation_strength: f32,
    pub path_description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynapticChanges {
    pub weights_updated: usize,
    pub new_connections: usize,
    pub strengthened_pathways: Vec<String>,
    pub stdp_applications: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageMetadata {
    pub storage_location: String,
    pub allocation_algorithm: String,
    pub compression_ratio: f32,
    pub predicted_retrieval_time_ms: f32,
}
```

**Verification**: Output types compile + serialize correctly

### Micro-Task 3.2.4: Add StoreMemory Output Validation
**Estimated Time**: 15 minutes  
**Expected Deliverable**: Complete output validation for StoreMemoryOutput

**Task Prompt for AI**: Implement validation logic for StoreMemoryOutput. Add to existing store_memory_schema.rs file.

```rust
impl ValidatedOutput for StoreMemoryOutput {
    fn validate(&self) -> Result<(), ValidationError> {
        // Memory ID validation
        if self.memory_id.is_empty() {
            return Err(ValidationError::MissingField("memory_id".to_string()));
        }
        
        // Decision confidence validation
        if !(0.0..=1.0).contains(&self.cortical_decision.decision_confidence) {
            return Err(ValidationError::InvalidValue {
                field: "cortical_decision.decision_confidence".to_string(),
                reason: "Decision confidence must be between 0.0 and 1.0".to_string(),
            });
        }
        
        Ok(())
    }
}
```

**Verification**: Output validation works correctly

### Micro-Task 3.2.5: Create StoreMemory JSON Schema
**Estimated Time**: 17 minutes  
**Expected Deliverable**: Complete JSON schema generation for store_memory_schema.rs

**Task Prompt for AI**: Create JSON schema generation function for MCP compliance. Add to existing store_memory_schema.rs file.

```rust
pub fn generate_store_memory_json_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "Content to store in neuromorphic memory",
                "maxLength": 10000,
                "minLength": 1
            },
            "context": {
                "type": "string",
                "description": "Contextual information for better allocation",
                "maxLength": 1000
            },
            "source": {
                "type": "string", 
                "description": "Source attribution for provenance tracking",
                "maxLength": 200
            },
            "confidence": {
                "type": "number",
                "description": "Confidence score between 0.0 and 1.0",
                "minimum": 0.0,
                "maximum": 1.0
            },
            "tags": {
                "type": "array",
                "description": "Tags for categorization",
                "items": {
                    "type": "string",
                    "maxLength": 50
                },
                "maxItems": 20
            },
            "importance": {
                "type": "number",
                "description": "Importance weight for allocation priority",
                "minimum": 0.0,
                "maximum": 1.0
            }
        },
        "required": ["content"],
        "additionalProperties": false
    })
}
```

**Verification**: JSON schema validates correctly against sample inputs

### Micro-Task 3.3.1: Create RetrieveMemory Input Schema
**Estimated Time**: 16 minutes  
**Expected Deliverable**: `src/mcp/schemas/retrieve_memory_schema.rs` (Input types only)

**Task Prompt for AI**: Create input schema for retrieving memories with advanced filtering. Focus only on input struct definition.

```rust
use super::base_types::{ValidatedInput, ValidationError};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrieveMemoryInput {
    #[serde(description = "Query for memory retrieval using natural language")]
    pub query: String,
    
    #[serde(description = "Maximum number of results to return")]
    pub limit: Option<usize>,
    
    #[serde(description = "Minimum similarity threshold (0.0-1.0)")]
    pub threshold: Option<f32>,
    
    #[serde(description = "Include detailed reasoning path in response")]
    pub include_reasoning: Option<bool>,
    
    #[serde(description = "Filter by specific tags")]
    pub tag_filter: Option<Vec<String>>,
    
    #[serde(description = "Temporal range for time-based filtering")]
    pub time_range: Option<TimeRange>,
    
    #[serde(description = "Retrieval strategy to use")]
    pub strategy: Option<RetrievalStrategy>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    pub start: Option<chrono::DateTime<chrono::Utc>>,
    pub end: Option<chrono::DateTime<chrono::Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RetrievalStrategy {
    SemanticSimilarity,
    StructuralMatching,
    TemporalProximity,
    HybridApproach,
    SpreadingActivation,
}
```

**Verification**: Compiles + enum serialization works

### Micro-Task 3.3.2: Add RetrieveMemory Input Validation
**Estimated Time**: 19 minutes  
**Expected Deliverable**: Complete input validation for RetrieveMemoryInput

**Task Prompt for AI**: Implement comprehensive validation logic for RetrieveMemoryInput including query limits and time range validation.

```rust
impl ValidatedInput for RetrieveMemoryInput {
    fn validate(&self) -> Result<(), ValidationError> {
        // Query validation
        if self.query.trim().is_empty() {
            return Err(ValidationError::InvalidValue {
                field: "query".to_string(),
                reason: "Query cannot be empty".to_string(),
            });
        }
        
        if self.query.len() > 1000 {
            return Err(ValidationError::InvalidValue {
                field: "query".to_string(),
                reason: "Query exceeds maximum length of 1000 characters".to_string(),
            });
        }
        
        // Limit validation
        if let Some(limit) = self.limit {
            if limit == 0 || limit > 100 {
                return Err(ValidationError::InvalidValue {
                    field: "limit".to_string(),
                    reason: "Limit must be between 1 and 100".to_string(),
                });
            }
        }
        
        // Threshold validation
        if let Some(threshold) = self.threshold {
            if !(0.0..=1.0).contains(&threshold) {
                return Err(ValidationError::InvalidValue {
                    field: "threshold".to_string(),
                    reason: "Threshold must be between 0.0 and 1.0".to_string(),
                });
            }
        }
        
        // Time range validation
        if let Some(ref time_range) = self.time_range {
            if let (Some(start), Some(end)) = (time_range.start, time_range.end) {
                if start >= end {
                    return Err(ValidationError::InvalidValue {
                        field: "time_range".to_string(),
                        reason: "Start time must be before end time".to_string(),
                    });
                }
            }
        }
        
        Ok(())
    }
}
```

**Verification**: Input validation handles all edge cases

### Micro-Task 3.3.3: Create RetrieveMemory Output Schema
**Estimated Time**: 18 minutes  
**Expected Deliverable**: Add output types to `retrieve_memory_schema.rs`

**Task Prompt for AI**: Create comprehensive output schema structures for retrieve memory operation. Add to existing retrieve_memory_schema.rs file.

```rust
use super::base_types::{ValidatedOutput, ValidationError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrieveMemoryOutput {
    pub memories: Vec<MemoryResult>,
    pub retrieval_metadata: RetrievalMetadata,
    pub neural_activations: Vec<NeuralActivation>,
    pub similarity_scores: Vec<f32>,
    pub total_found: usize,
    pub query_understanding: QueryUnderstanding,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryResult {
    pub memory_id: String,
    pub content: String,
    pub similarity_score: f32,
    pub context: Option<String>,
    pub source: Option<String>,
    pub tags: Vec<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub retrieval_path: Vec<String>,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalMetadata {
    pub strategy_used: String,
    pub search_depth: usize,
    pub nodes_traversed: usize,
    pub cortical_columns_activated: Vec<String>,
    pub spreading_activation_hops: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralActivation {
    pub column_type: String,
    pub activation_pattern: Vec<f32>,
    pub peak_activation: f32,
    pub activation_duration_ms: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryUnderstanding {
    pub semantic_concepts: Vec<String>,
    pub temporal_indicators: Vec<String>,
    pub structural_hints: Vec<String>,
    pub confidence_score: f32,
}
```

**Verification**: Output types compile + all fields accessible

### Micro-Task 3.3.4: Add RetrieveMemory Output Validation
**Estimated Time**: 16 minutes  
**Expected Deliverable**: Complete output validation for RetrieveMemoryOutput

**Task Prompt for AI**: Implement validation logic for RetrieveMemoryOutput including similarity score consistency. Add to existing retrieve_memory_schema.rs file.

```rust
impl ValidatedOutput for RetrieveMemoryOutput {
    fn validate(&self) -> Result<(), ValidationError> {
        // Validate similarity scores match memory count
        if self.similarity_scores.len() != self.memories.len() {
            return Err(ValidationError::InvalidValue {
                field: "similarity_scores".to_string(),
                reason: "Number of similarity scores must match number of memories".to_string(),
            });
        }
        
        // Validate all similarity scores are in valid range
        for (i, score) in self.similarity_scores.iter().enumerate() {
            if !(0.0..=1.0).contains(score) {
                return Err(ValidationError::InvalidValue {
                    field: format!("similarity_scores[{}]", i),
                    reason: "Similarity score must be between 0.0 and 1.0".to_string(),
                });
            }
        }
        
        // Validate memory results
        for (i, memory) in self.memories.iter().enumerate() {
            if memory.memory_id.is_empty() {
                return Err(ValidationError::InvalidValue {
                    field: format!("memories[{}].memory_id", i),
                    reason: "Memory ID cannot be empty".to_string(),
                });
            }
            
            if !(0.0..=1.0).contains(&memory.similarity_score) {
                return Err(ValidationError::InvalidValue {
                    field: format!("memories[{}].similarity_score", i),
                    reason: "Memory similarity score must be between 0.0 and 1.0".to_string(),
                });
            }
        }
        
        Ok(())
    }
}
```

**Verification**: Output validation catches all inconsistencies

### Micro-Task 3.3.5: Create RetrieveMemory JSON Schema
**Estimated Time**: 15 minutes  
**Expected Deliverable**: Complete JSON schema generation for retrieve_memory_schema.rs

**Task Prompt for AI**: Create JSON schema generation function for retrieve memory tool. Add to existing retrieve_memory_schema.rs file.

```rust
pub fn generate_retrieve_memory_json_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Query for memory retrieval using natural language",
                "maxLength": 1000,
                "minLength": 1
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return",
                "minimum": 1,
                "maximum": 100,
                "default": 10
            },
            "threshold": {
                "type": "number",
                "description": "Minimum similarity threshold (0.0-1.0)",
                "minimum": 0.0,
                "maximum": 1.0,
                "default": 0.3
            },
            "include_reasoning": {
                "type": "boolean",
                "description": "Include detailed reasoning path in response",
                "default": false
            },
            "tag_filter": {
                "type": "array",
                "description": "Filter by specific tags",
                "items": {
                    "type": "string",
                    "maxLength": 50
                },
                "maxItems": 10
            },
            "time_range": {
                "type": "object",
                "description": "Temporal range for time-based filtering",
                "properties": {
                    "start": {
                        "type": "string",
                        "format": "date-time"
                    },
                    "end": {
                        "type": "string", 
                        "format": "date-time"
                    }
                }
            },
            "strategy": {
                "type": "string",
                "description": "Retrieval strategy to use",
                "enum": ["SemanticSimilarity", "StructuralMatching", "TemporalProximity", "HybridApproach", "SpreadingActivation"],
                "default": "HybridApproach"
            }
        },
        "required": ["query"],
        "additionalProperties": false
    })
}
```

**Verification**: JSON schema validates retrieve requests correctly

### Micro-Task 3.4.1: Create UpdateMemory Input Schema
**Estimated Time**: 17 minutes  
**Expected Deliverable**: `src/mcp/schemas/update_memory_schema.rs` (Input types only)

**Task Prompt for AI**: Create input schema for updating memories with selective field updates and plasticity modes.

```rust
use super::base_types::{ValidatedInput, ValidationError};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateMemoryInput {
    #[serde(description = "ID of the memory to update")]
    pub memory_id: String,
    
    #[serde(description = "New content for the memory")]
    pub new_content: Option<String>,
    
    #[serde(description = "Updated context information")]
    pub new_context: Option<String>,
    
    #[serde(description = "Updated confidence score")]
    pub new_confidence: Option<f32>,
    
    #[serde(description = "Tags to add")]
    pub add_tags: Option<Vec<String>>,
    
    #[serde(description = "Tags to remove")]
    pub remove_tags: Option<Vec<String>>,
    
    #[serde(description = "Update importance weight")]
    pub new_importance: Option<f32>,
    
    #[serde(description = "Synaptic plasticity mode for updates")]
    pub plasticity_mode: Option<PlasticityMode>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlasticityMode {
    Conservative,  // Minimal changes to existing pathways
    Adaptive,      // Moderate synaptic updates
    Aggressive,    // Significant pathway reorganization
}
```

**Verification**: Compiles + plasticity enum works

### Micro-Task 3.4.2: Add UpdateMemory Input Validation
**Estimated Time**: 18 minutes  
**Expected Deliverable**: Complete input validation for UpdateMemoryInput

**Task Prompt for AI**: Implement comprehensive validation logic for UpdateMemoryInput including at-least-one-field requirement.

```rust
impl ValidatedInput for UpdateMemoryInput {
    fn validate(&self) -> Result<(), ValidationError> {
        // Memory ID validation
        if self.memory_id.trim().is_empty() {
            return Err(ValidationError::MissingField("memory_id".to_string()));
        }
        
        // At least one update field must be provided
        if self.new_content.is_none() 
            && self.new_context.is_none()
            && self.new_confidence.is_none()
            && self.add_tags.is_none()
            && self.remove_tags.is_none()
            && self.new_importance.is_none() {
            return Err(ValidationError::InvalidValue {
                field: "update_fields".to_string(),
                reason: "At least one field must be provided for update".to_string(),
            });
        }
        
        // Content validation
        if let Some(ref content) = self.new_content {
            if content.trim().is_empty() {
                return Err(ValidationError::InvalidValue {
                    field: "new_content".to_string(),
                    reason: "New content cannot be empty".to_string(),
                });
            }
            if content.len() > 10000 {
                return Err(ValidationError::InvalidValue {
                    field: "new_content".to_string(),
                    reason: "Content exceeds maximum length of 10000 characters".to_string(),
                });
            }
        }
        
        // Confidence validation
        if let Some(confidence) = self.new_confidence {
            if !(0.0..=1.0).contains(&confidence) {
                return Err(ValidationError::InvalidValue {
                    field: "new_confidence".to_string(),
                    reason: "Confidence must be between 0.0 and 1.0".to_string(),
                });
            }
        }
        
        // Importance validation
        if let Some(importance) = self.new_importance {
            if !(0.0..=1.0).contains(&importance) {
                return Err(ValidationError::InvalidValue {
                    field: "new_importance".to_string(),
                    reason: "Importance must be between 0.0 and 1.0".to_string(),
                });
            }
        }
        
        Ok(())
    }
}
```

**Verification**: Input validation enforces all constraints

### Micro-Task 3.4.3: Create UpdateMemory Output Schema
**Estimated Time**: 16 minutes  
**Expected Deliverable**: Add output types to `update_memory_schema.rs`

**Task Prompt for AI**: Create output schema structures for update memory operation. Add to existing update_memory_schema.rs file.

```rust
use super::base_types::{ValidatedOutput, ValidationError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateMemoryOutput {
    pub memory_id: String,
    pub update_successful: bool,
    pub changes_applied: Vec<ChangeRecord>,
    pub synaptic_modifications: SynapticModifications,
    pub new_neural_pathways: Vec<String>,
    pub affected_connections: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeRecord {
    pub field_name: String,
    pub old_value: Option<String>,
    pub new_value: String,
    pub change_timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynapticModifications {
    pub strengthened_connections: usize,
    pub weakened_connections: usize,
    pub new_connections_formed: usize,
    pub pruned_connections: usize,
    pub plasticity_applied: String,
}
```

**Verification**: Output types compile + change tracking works

### Micro-Task 3.4.4: Add UpdateMemory Output Validation + JSON Schema
**Estimated Time**: 17 minutes  
**Expected Deliverable**: Complete validation and JSON schema for update_memory_schema.rs

**Task Prompt for AI**: Implement output validation and JSON schema generation for UpdateMemory tool.

```rust
impl ValidatedOutput for UpdateMemoryOutput {
    fn validate(&self) -> Result<(), ValidationError> {
        if self.memory_id.is_empty() {
            return Err(ValidationError::MissingField("memory_id".to_string()));
        }
        
        if self.changes_applied.is_empty() && self.update_successful {
            return Err(ValidationError::InvalidValue {
                field: "changes_applied".to_string(),
                reason: "Successful update must have at least one change record".to_string(),
            });
        }
        
        Ok(())
    }
}

pub fn generate_update_memory_json_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "string",
                "description": "ID of the memory to update",
                "minLength": 1
            },
            "new_content": {
                "type": "string",
                "description": "New content for the memory",
                "maxLength": 10000
            },
            "new_context": {
                "type": "string",
                "description": "Updated context information",
                "maxLength": 1000
            },
            "new_confidence": {
                "type": "number",
                "description": "Updated confidence score",
                "minimum": 0.0,
                "maximum": 1.0
            },
            "add_tags": {
                "type": "array",
                "description": "Tags to add",
                "items": {
                    "type": "string",
                    "maxLength": 50
                },
                "maxItems": 20
            },
            "remove_tags": {
                "type": "array",
                "description": "Tags to remove", 
                "items": {
                    "type": "string",
                    "maxLength": 50
                },
                "maxItems": 20
            },
            "new_importance": {
                "type": "number",
                "description": "Update importance weight",
                "minimum": 0.0,
                "maximum": 1.0
            },
            "plasticity_mode": {
                "type": "string",
                "description": "Synaptic plasticity mode for updates",
                "enum": ["Conservative", "Adaptive", "Aggressive"],
                "default": "Adaptive"
            }
        },
        "required": ["memory_id"],
        "additionalProperties": false
    })
}
```

**Verification**: Validation + JSON schema work correctly

### Micro-Task 3.5.1: Create DeleteMemory Schema
**Estimated Time**: 15 minutes  
**Expected Deliverable**: `src/mcp/schemas/delete_memory_schema.rs`

**Task Prompt for AI**: Create complete schema for delete memory tool with cleanup modes.

```rust
use super::base_types::{ValidatedInput, ValidatedOutput, ValidationError};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteMemoryInput {
    #[serde(description = "ID of the memory to delete")]
    pub memory_id: String,
    
    #[serde(description = "Force deletion even if memory has strong connections")]
    pub force_delete: Option<bool>,
    
    #[serde(description = "Cleanup mode for neural pathways")]
    pub cleanup_mode: Option<CleanupMode>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CleanupMode {
    Minimal,     // Keep pathways, just mark as deleted
    Standard,    // Remove direct connections
    Aggressive,  // Full pathway cleanup and reorganization
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteMemoryOutput {
    pub memory_id: String,
    pub deletion_successful: bool,
    pub connections_removed: usize,
    pub pathways_reorganized: usize,
    pub cleanup_performed: String,
}

impl ValidatedInput for DeleteMemoryInput {
    fn validate(&self) -> Result<(), ValidationError> {
        if self.memory_id.trim().is_empty() {
            return Err(ValidationError::MissingField("memory_id".to_string()));
        }
        Ok(())
    }
}

impl ValidatedOutput for DeleteMemoryOutput {
    fn validate(&self) -> Result<(), ValidationError> {
        if self.memory_id.is_empty() {
            return Err(ValidationError::MissingField("memory_id".to_string()));
        }
        Ok(())
    }
}
```

**Verification**: Delete schema compiles + validation works

### Micro-Task 3.5.2: Create AnalyzeGraph Schema
**Estimated Time**: 18 minutes  
**Expected Deliverable**: `src/mcp/schemas/analyze_graph_schema.rs`

**Task Prompt for AI**: Create complete schema for analyze graph tool with multiple analysis types.

```rust
use super::base_types::{ValidatedInput, ValidatedOutput, ValidationError};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyzeGraphInput {
    #[serde(description = "Type of analysis to perform")]
    pub analysis_type: AnalysisType,
    
    #[serde(description = "Focus on specific memory IDs")]
    pub memory_ids: Option<Vec<String>>,
    
    #[serde(description = "Include detailed neural pathway analysis")]
    pub include_pathways: Option<bool>,
    
    #[serde(description = "Maximum depth for graph traversal")]
    pub max_depth: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnalysisType {
    Connectivity,
    Centrality,
    Clustering,
    PathwayEfficiency,
    SynapticStrength,
    MemoryIntegrity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyzeGraphOutput {
    pub analysis_type: String,
    pub graph_metrics: GraphMetrics,
    pub neural_insights: Vec<NeuralInsight>,
    pub recommendations: Vec<String>,
    pub performance_indicators: PerformanceIndicators,
}
```

**Verification**: Analysis schema compiles + enums work

### Micro-Task 3.5.3: Add AnalyzeGraph Support Types
**Estimated Time**: 16 minutes  
**Expected Deliverable**: Complete support types for analyze_graph_schema.rs

**Task Prompt for AI**: Add all support structures and validation for analyze graph schema.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMetrics {
    pub total_nodes: usize,
    pub total_connections: usize,
    pub average_connectivity: f32,
    pub clustering_coefficient: f32,
    pub path_efficiency: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralInsight {
    pub insight_type: String,
    pub description: String,
    pub confidence: f32,
    pub affected_memories: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceIndicators {
    pub retrieval_efficiency: f32,
    pub allocation_success_rate: f32,
    pub pathway_utilization: f32,
    pub memory_density: f32,
}

impl ValidatedInput for AnalyzeGraphInput {
    fn validate(&self) -> Result<(), ValidationError> {
        if let Some(max_depth) = self.max_depth {
            if max_depth == 0 || max_depth > 50 {
                return Err(ValidationError::InvalidValue {
                    field: "max_depth".to_string(),
                    reason: "Max depth must be between 1 and 50".to_string(),
                });
            }
        }
        Ok(())
    }
}

impl ValidatedOutput for AnalyzeGraphOutput {
    fn validate(&self) -> Result<(), ValidationError> {
        if self.analysis_type.is_empty() {
            return Err(ValidationError::MissingField("analysis_type".to_string()));
        }
        Ok(())
    }
}
```

**Verification**: Analysis graph schema fully functional

### Micro-Task 3.5.4: Create MemoryStats Schema
**Estimated Time**: 17 minutes  
**Expected Deliverable**: `src/mcp/schemas/memory_stats_schema.rs`

**Task Prompt for AI**: Create complete schema for memory stats tool with comprehensive metrics.

```rust
use super::base_types::{ValidatedInput, ValidatedOutput, ValidationError};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStatsInput {
    #[serde(description = "Include detailed breakdown by column type")]
    pub detailed_breakdown: Option<bool>,
    
    #[serde(description = "Time range for statistics")]
    pub time_range: Option<StatsTimeRange>,
    
    #[serde(description = "Include performance metrics")]
    pub include_performance: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatsTimeRange {
    pub start: chrono::DateTime<chrono::Utc>,
    pub end: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStatsOutput {
    pub system_overview: SystemOverview,
    pub column_statistics: Vec<ColumnStats>,
    pub performance_metrics: PerformanceMetrics,
    pub health_indicators: HealthIndicators,
}
```

**Verification**: Stats schema compiles correctly

### Micro-Task 3.5.5: Add MemoryStats Support Types
**Estimated Time**: 16 minutes  
**Expected Deliverable**: Complete support types for memory_stats_schema.rs

**Task Prompt for AI**: Add all support structures and validation for memory stats schema.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemOverview {
    pub total_memories: usize,
    pub total_connections: usize,
    pub memory_usage_mb: f32,
    pub uptime_hours: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnStats {
    pub column_type: String,
    pub activations_count: usize,
    pub average_activation_strength: f32,
    pub successful_allocations: usize,
    pub processing_time_ms_avg: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub avg_response_time_ms: f32,
    pub throughput_ops_per_minute: f32,
    pub error_rate_percent: f32,
    pub memory_efficiency: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthIndicators {
    pub overall_health_score: f32,
    pub neural_integrity: f32,
    pub synaptic_health: f32,
    pub allocation_efficiency: f32,
}

impl ValidatedInput for MemoryStatsInput {
    fn validate(&self) -> Result<(), ValidationError> {
        if let Some(ref time_range) = self.time_range {
            if time_range.start >= time_range.end {
                return Err(ValidationError::InvalidValue {
                    field: "time_range".to_string(),
                    reason: "Start time must be before end time".to_string(),
                });
            }
        }
        Ok(())
    }
}

impl ValidatedOutput for MemoryStatsOutput {
    fn validate(&self) -> Result<(), ValidationError> {
        if self.system_overview.total_memories == 0 && !self.column_statistics.is_empty() {
            return Err(ValidationError::InvalidValue {
                field: "system_overview".to_string(),
                reason: "Column stats exist but total memories is zero".to_string(),
            });
        }
        Ok(())
    }
}
```

**Verification**: Memory stats schema fully functional

### Micro-Task 3.5.6: Create ConfigureLearning Schema
**Estimated Time**: 18 minutes  
**Expected Deliverable**: `src/mcp/schemas/configure_learning_schema.rs`

**Task Prompt for AI**: Create complete schema for configure learning tool with learning modes and parameter validation.

```rust
use super::base_types::{ValidatedInput, ValidatedOutput, ValidationError};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigureLearningInput {
    #[serde(description = "STDP learning rate")]
    pub learning_rate: Option<f32>,
    
    #[serde(description = "Synaptic decay rate")]
    pub decay_rate: Option<f32>,
    
    #[serde(description = "Activation threshold for learning")]
    pub activation_threshold: Option<f32>,
    
    #[serde(description = "Learning mode configuration")]
    pub learning_mode: Option<LearningMode>,
    
    #[serde(description = "Plasticity window duration in milliseconds")]
    pub plasticity_window_ms: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningMode {
    Hebbian,
    AntiHebbian,
    STDP,
    Homeostatic,
    MetaLearning,
}
```

**Verification**: Configure learning schema compiles

### Micro-Task 3.5.7: Add ConfigureLearning Support Types
**Estimated Time**: 15 minutes  
**Expected Deliverable**: Complete support types for configure_learning_schema.rs

**Task Prompt for AI**: Add all support structures and validation for configure learning schema.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigureLearningOutput {
    pub configuration_applied: bool,
    pub updated_parameters: Vec<ParameterUpdate>,
    pub expected_behavior_changes: Vec<String>,
    pub learning_system_status: LearningSystemStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterUpdate {
    pub parameter_name: String,
    pub old_value: f32,
    pub new_value: f32,
    pub update_timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningSystemStatus {
    pub is_learning_enabled: bool,
    pub active_learning_modes: Vec<String>,
    pub synaptic_update_rate: f32,
    pub memory_consolidation_active: bool,
}

impl ValidatedInput for ConfigureLearningInput {
    fn validate(&self) -> Result<(), ValidationError> {
        if let Some(learning_rate) = self.learning_rate {
            if !(0.0..=1.0).contains(&learning_rate) {
                return Err(ValidationError::InvalidValue {
                    field: "learning_rate".to_string(),
                    reason: "Learning rate must be between 0.0 and 1.0".to_string(),
                });
            }
        }
        
        if let Some(decay_rate) = self.decay_rate {
            if !(0.0..=1.0).contains(&decay_rate) {
                return Err(ValidationError::InvalidValue {
                    field: "decay_rate".to_string(),
                    reason: "Decay rate must be between 0.0 and 1.0".to_string(),
                });
            }
        }
        
        if let Some(window) = self.plasticity_window_ms {
            if !(0.1..=1000.0).contains(&window) {
                return Err(ValidationError::InvalidValue {
                    field: "plasticity_window_ms".to_string(),
                    reason: "Plasticity window must be between 0.1 and 1000.0 ms".to_string(),
                });
            }
        }
        
        Ok(())
    }
}

impl ValidatedOutput for ConfigureLearningOutput {
    fn validate(&self) -> Result<(), ValidationError> {
        if self.updated_parameters.is_empty() && self.configuration_applied {
            return Err(ValidationError::InvalidValue {
                field: "updated_parameters".to_string(),
                reason: "Applied configuration must have parameter updates".to_string(),
            });
        }
        Ok(())
    }
}
```

**Verification**: Configure learning schema fully functional

### Micro-Task 3.6.1: Create Schema Registry Core
**Estimated Time**: 15 minutes  
**Expected Deliverable**: `src/mcp/schemas/mod.rs` (Core registry only)

**Task Prompt for AI**: Create the central schema registry with tool schema management.

```rust
pub mod base_types;
pub mod store_memory_schema;
pub mod retrieve_memory_schema;
pub mod update_memory_schema;
pub mod delete_memory_schema;
pub mod analyze_graph_schema;
pub mod memory_stats_schema;
pub mod configure_learning_schema;

use base_types::{MCPToolInput, MCPToolOutput, ValidatedInput, ValidatedOutput, ValidationError};
use serde_json::Value;
use std::collections::HashMap;

pub struct SchemaRegistry {
    tool_schemas: HashMap<String, Value>,
}

impl SchemaRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            tool_schemas: HashMap::new(),
        };
        
        registry.register_all_schemas();
        registry
    }
    
    pub fn get_schema(&self, tool_name: &str) -> Option<&Value> {
        self.tool_schemas.get(tool_name)
    }
    
    pub fn list_available_tools(&self) -> Vec<String> {
        self.tool_schemas.keys().cloned().collect()
    }
}
```

**Verification**: Registry compiles + basic functionality works

### Micro-Task 3.6.2: Add Schema Registration
**Estimated Time**: 18 minutes  
**Expected Deliverable**: Complete schema registration for all tools

**Task Prompt for AI**: Implement schema registration for all 7 MCP tools. Add to existing mod.rs file.

```rust
fn register_all_schemas(&mut self) {
    // Register all tool schemas
    self.tool_schemas.insert(
        "store_memory".to_string(),
        store_memory_schema::generate_store_memory_json_schema(),
    );
    
    self.tool_schemas.insert(
        "retrieve_memory".to_string(),
        retrieve_memory_schema::generate_retrieve_memory_json_schema(),
    );
    
    self.tool_schemas.insert(
        "update_memory".to_string(),
        update_memory_schema::generate_update_memory_json_schema(),
    );
    
    self.tool_schemas.insert(
        "delete_memory".to_string(),
        serde_json::json!({
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "minLength": 1},
                "force_delete": {"type": "boolean", "default": false},
                "cleanup_mode": {
                    "type": "string",
                    "enum": ["Minimal", "Standard", "Aggressive"],
                    "default": "Standard"
                }
            },
            "required": ["memory_id"]
        }),
    );
    
    self.tool_schemas.insert(
        "analyze_memory_graph".to_string(),
        serde_json::json!({
            "type": "object",
            "properties": {
                "analysis_type": {
                    "type": "string",
                    "enum": ["Connectivity", "Centrality", "Clustering", "PathwayEfficiency", "SynapticStrength", "MemoryIntegrity"]
                },
                "memory_ids": {"type": "array", "items": {"type": "string"}},
                "include_pathways": {"type": "boolean", "default": false},
                "max_depth": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10}
            },
            "required": ["analysis_type"]
        }),
    );
    
    self.tool_schemas.insert(
        "get_memory_stats".to_string(),
        serde_json::json!({
            "type": "object",
            "properties": {
                "detailed_breakdown": {"type": "boolean", "default": false},
                "time_range": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "string", "format": "date-time"},
                        "end": {"type": "string", "format": "date-time"}
                    }
                },
                "include_performance": {"type": "boolean", "default": true}
            }
        }),
    );
    
    self.tool_schemas.insert(
        "configure_learning".to_string(),
        serde_json::json!({
            "type": "object",
            "properties": {
                "learning_rate": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "decay_rate": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "activation_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "learning_mode": {
                    "type": "string",
                    "enum": ["Hebbian", "AntiHebbian", "STDP", "Homeostatic", "MetaLearning"]
                },
                "plasticity_window_ms": {"type": "number", "minimum": 0.1, "maximum": 1000.0}
            }
        }),
    );
}
```

**Verification**: All 7 tool schemas registered correctly

### Micro-Task 3.6.3: Add Schema Validation Engine
**Estimated Time**: 16 minutes  
**Expected Deliverable**: Complete validation engine for schema registry

**Task Prompt for AI**: Implement schema validation framework and utility functions. Add to existing mod.rs file.
    
```rust
pub fn validate_tool_input(&self, tool_name: &str, input: &Value) -> Result<(), ValidationError> {
    let schema = self.get_schema(tool_name)
        .ok_or_else(|| ValidationError::SchemaError(format!("Unknown tool: {}", tool_name)))?;
    
    // Perform JSON schema validation
    self.validate_against_schema(input, schema)
}

fn validate_against_schema(&self, input: &Value, schema: &Value) -> Result<(), ValidationError> {
    // Basic validation implementation
    // In production, use a proper JSON schema validator like jsonschema crate
    
    if let Some(required) = schema.get("required") {
        if let Some(required_array) = required.as_array() {
            for required_field in required_array {
                if let Some(field_name) = required_field.as_str() {
                    if !input.get(field_name).is_some() {
                        return Err(ValidationError::MissingField(field_name.to_string()));
                    }
                }
            }
        }
    }
    
    Ok(())
}
}

// Validation utilities
pub fn validate_mcp_input<T: ValidatedInput>(input: &T) -> Result<(), ValidationError> {
    input.validate()
}

pub fn validate_mcp_output<T: ValidatedOutput>(output: &T) -> Result<(), ValidationError> {
    output.validate()
}
```

**Verification**: Validation engine works correctly

### Micro-Task 3.6.4: Add Schema Registry Tests
**Estimated Time**: 14 minutes  
**Expected Deliverable**: Complete test suite for schema registry

**Task Prompt for AI**: Create comprehensive tests for schema registry functionality. Add to existing mod.rs file.

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_schema_registry_creation() {
        let registry = SchemaRegistry::new();
        assert_eq!(registry.list_available_tools().len(), 7);
    }
    
    #[test]
    fn test_schema_retrieval() {
        let registry = SchemaRegistry::new();
        let schema = registry.get_schema("store_memory");
        assert!(schema.is_some());
    }
    
    #[test]
    fn test_all_tools_registered() {
        let registry = SchemaRegistry::new();
        let tools = registry.list_available_tools();
        
        assert!(tools.contains(&"store_memory".to_string()));
        assert!(tools.contains(&"retrieve_memory".to_string()));
        assert!(tools.contains(&"update_memory".to_string()));
        assert!(tools.contains(&"delete_memory".to_string()));
        assert!(tools.contains(&"analyze_memory_graph".to_string()));
        assert!(tools.contains(&"get_memory_stats".to_string()));
        assert!(tools.contains(&"configure_learning".to_string()));
    }
    
    #[test]
    fn test_validation_utilities() {
        use store_memory_schema::StoreMemoryInput;
        
        let valid_input = StoreMemoryInput {
            content: "test content".to_string(),
            context: None,
            source: None,
            confidence: Some(0.8),
            tags: None,
            importance: None,
        };
        
        assert!(validate_mcp_input(&valid_input).is_ok());
    }
}
```

**Verification**: All tests pass + coverage complete

## Validation Checklist

**After completing all 25 micro-tasks:**
- [ ] All 7 MCP tool schemas defined with proper types
- [ ] Input validation logic comprehensive and tested
- [ ] Output validation ensures data integrity
- [ ] JSON schema generation for MCP compliance
- [ ] Schema registry provides centralized management
- [ ] Neuromorphic-specific fields included in all schemas
- [ ] Error handling covers all validation scenarios
- [ ] Unit tests verify schema correctness
- [ ] Each schema file compiles independently
- [ ] All validation functions handle edge cases
- [ ] JSON schemas validate against sample data
- [ ] Registry correctly manages all 7 tools

## Next Phase Dependencies

This phase provides schema foundation for:
- MicroPhase 4: Tool implementation with validated I/O
- MicroPhase 5: Authentication with request validation
- MicroPhase 7: Testing with schema compliance verification

## Micro-Task Summary

**Total: 25 micro-tasks (15-19 minutes each)**
- Base Types: 3 tasks (Core types, neuromorphic structures, validation framework)
- Store Memory: 5 tasks (Input schema, input validation, output schema, output validation, JSON schema)
- Retrieve Memory: 5 tasks (Input schema, input validation, output schema, output validation, JSON schema)
- Update Memory: 4 tasks (Input schema, input validation, output schema, validation + JSON schema)
- Delete Memory: 1 task (Complete delete schema)
- Analyze Graph: 2 tasks (Input + output schema, support types + validation)
- Memory Stats: 2 tasks (Input + output schema, support types + validation)
- Configure Learning: 2 tasks (Input + output schema, support types + validation)
- Schema Registry: 4 tasks (Core registry, registration, validation engine, tests)

**Pattern Consistency**: Each micro-task creates one deliverable, follows 15-20 minute constraint, includes specific verification criteria.