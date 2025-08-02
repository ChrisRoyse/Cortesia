//! Type definitions for the LLM-friendly MCP server

use serde::{Deserialize, Serialize};

// Re-export shared types
pub use crate::mcp::shared_types::{LLMMCPTool, LLMExample, LLMMCPRequest, LLMMCPResponse, PerformanceInfo};

#[derive(Debug, Clone)]
pub struct UsageStats {
    pub total_operations: u64,
    pub triples_stored: u64,
    pub chunks_stored: u64,
    pub queries_executed: u64,
    pub avg_response_time_ms: f64,
    pub memory_efficiency: f64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub uptime: std::time::Instant,
}

impl Default for UsageStats {
    fn default() -> Self {
        Self {
            total_operations: 0,
            triples_stored: 0,
            chunks_stored: 0,
            queries_executed: 0,
            avg_response_time_ms: 0.0,
            memory_efficiency: 0.0,
            cache_hits: 0,
            cache_misses: 0,
            uptime: std::time::Instant::now(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub confidence: f64,
    pub conflicts: Vec<String>,
    pub sources: Vec<String>,
    pub validation_notes: Vec<String>,
}