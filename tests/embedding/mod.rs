// Embedding module tests - comprehensive test suite for high-performance vector operations
//
// This test module provides complete coverage for:
// - Product quantization: 8-32x compression with <5% accuracy loss
// - SIMD similarity: Numerical equivalence across scalar/SSE/AVX2 implementations  
// - Batch processing: High-throughput embedding operations
// - Hardware compatibility: Graceful fallback across CPU architectures
// - Search latency: <1ms for 1M embedding similarity search
// - Compression validation: Round-trip accuracy testing

pub mod test_quantizer;
pub mod test_simd_search;
pub mod test_embedding_store;
pub mod test_similarity;
pub mod performance_tests;
pub mod compression_tests;
pub mod hardware_compatibility_tests;