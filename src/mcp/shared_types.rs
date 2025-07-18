use serde::{Deserialize, Serialize};

/// Shared MCP tool definition used across all MCP servers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPTool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// Enhanced MCP tool with examples and tips for LLM-friendly servers
#[derive(Debug, Serialize, Deserialize)]
pub struct LLMMCPTool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
    pub examples: Vec<LLMExample>,
    pub tips: Vec<String>,
}

/// Example for LLM tools
#[derive(Debug, Serialize, Deserialize)]
pub struct LLMExample {
    pub description: String,
    pub input: serde_json::Value,
    pub expected_output: String,
}

/// Shared MCP request structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPRequest {
    pub tool: String,
    pub arguments: serde_json::Value,
}

/// Enhanced MCP request for LLM-friendly servers
#[derive(Debug, Serialize, Deserialize)]
pub struct LLMMCPRequest {
    pub method: String,
    pub params: serde_json::Value,
}

/// Shared MCP response structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPResponse {
    pub content: Vec<MCPContent>,
    #[serde(default)]
    pub is_error: bool,
}

/// Enhanced MCP response for LLM-friendly servers with structured content
#[derive(Debug, Serialize, Deserialize)]
pub struct LLMMCPResponse {
    pub success: bool,
    pub data: serde_json::Value,
    pub message: String,
    pub helpful_info: Option<String>,
    pub suggestions: Vec<String>,
    pub performance: PerformanceInfo,
}

/// MCP content item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPContent {
    #[serde(rename = "type")]
    pub type_: String,
    pub text: String,
}

/// Response metadata for enhanced tracking
#[derive(Debug, Serialize, Deserialize)]
pub struct ResponseMetadata {
    pub execution_time_ms: f64,
    pub operation_type: String,
    pub data_source: String,
}

/// Performance information for enhanced responses
#[derive(Debug, Serialize, Deserialize)]
pub struct PerformanceInfo {
    pub execution_time_ms: f64,
    pub memory_used_bytes: u64,
    pub cache_hit: bool,
    pub complexity_score: f32,
}