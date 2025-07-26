//! Request handlers for LLM-friendly MCP server

pub mod storage;
pub mod query;
pub mod cognitive_query;
pub mod exploration;
pub mod advanced;
pub mod cognitive;
pub mod stats;
pub mod enhanced_search;
pub mod graph_analysis;
pub mod temporal;
pub mod cognitive_preview;

#[cfg(test)]
pub mod tests;

pub use storage::{handle_store_fact, handle_store_knowledge};
pub use cognitive_preview::{handle_store_fact_cognitive_preview, handle_cognitive_reasoning_preview, handle_neural_train_model_preview};
pub use query::*;
pub use exploration::*;
pub use advanced::*;
pub use cognitive::{handle_neural_importance_scoring, handle_divergent_thinking_engine, handle_simd_ultra_fast_search, handle_analyze_graph_centrality};
pub use stats::*;
pub use enhanced_search::*;
pub use graph_analysis::*;
pub use temporal::*;