// LLMKG Library - Neuromorphic Knowledge Graph with Scalable Allocation Architecture

//! # LLMKG: Large Language Model Knowledge Graph
//! 
//! A neuromorphic knowledge graph system implementing allocation-first paradigm
//! with scalable architecture supporting billion-node graphs.
//! 
//! ## Core Features
//! 
//! - **Neuromorphic Allocation**: Time-to-First-Spike (TTFS) encoding with lateral inhibition
//! - **Scalable Architecture**: HNSW indexing, multi-tier caching, distributed processing
//! - **Adaptive Quantization**: 4-32x memory reduction while maintaining accuracy
//! - **Production Ready**: Enterprise deployment with monitoring and observability
//! 
//! ## Architecture Overview
//! 
//! ```text
//! ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
//! │   Core Engine   │────│ Scalable Engine  │────│ Production Ops  │
//! │                 │    │                  │    │                 │
//! │ • TTFS Encoding │    │ • HNSW Indexing  │    │ • Monitoring    │
//! │ • Lateral Inhib │    │ • Multi-tier     │    │ • Deployment    │
//! │ • SNN Processing│    │   Caching        │    │ • Load Testing  │
//! │ • 29 Networks   │    │ • Distributed    │    │ • Health Checks │
//! └─────────────────┘    └──────────────────┘    └─────────────────┘
//! ```

pub mod core;
pub mod scalable;

#[cfg(feature = "test-support")]
pub mod test_support;

pub use core::{AllocationEngine, Fact, AllocationResult, NodeId};
pub use scalable::{ScalableAllocationEngine, ScalabilityConfig};

/// Re-exports for common usage
pub mod prelude {
    pub use crate::core::{AllocationEngine, Fact, AllocationResult, NodeId};
    pub use crate::scalable::{ScalableAllocationEngine, ScalabilityConfig};
}