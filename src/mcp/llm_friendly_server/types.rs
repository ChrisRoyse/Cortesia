//! Type definitions for the LLM-friendly MCP server

use serde::{Deserialize, Serialize};

// Re-export shared types
pub use crate::mcp::shared_types::{LLMMCPTool, LLMExample, LLMMCPRequest, LLMMCPResponse, PerformanceInfo};

#[derive(Debug, Clone, Default)]
pub struct UsageStats {
    pub total_operations: u64,
    pub triples_stored: u64,
    pub chunks_stored: u64,
    pub queries_executed: u64,
    pub avg_response_time_ms: f64,
    pub memory_efficiency: f64,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub confidence: f64,
    pub conflicts: Vec<String>,
    pub sources: Vec<String>,
    pub validation_notes: Vec<String>,
}