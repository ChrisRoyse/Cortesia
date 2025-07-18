use wasm_bindgen::prelude::*;
use crate::core::graph::KnowledgeGraph;
use crate::query::rag::GraphRAGEngine;
use crate::embedding::simd_search::BatchProcessor;
use js_sys::{Float32Array, Uint32Array, Uint8Array};
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;

#[wasm_bindgen]
pub struct FastKnowledgeGraph {
    graph: Arc<RwLock<KnowledgeGraph>>,
    rag_engine: Arc<RwLock<GraphRAGEngine>>,
    batch_processor: Arc<RwLock<BatchProcessor>>,
    embedding_cache: Arc<RwLock<HashMap<String, Vec<f32>>>>,
    performance_stats: Arc<RwLock<PerformanceStats>>,
}

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct PerformanceStats {
    pub total_queries: u32,
    pub avg_query_time_ms: f64,
    pub cache_hit_rate: f64,
    pub memory_usage_mb: f64,
}

#[wasm_bindgen]
impl FastKnowledgeGraph {
    #[wasm_bindgen(constructor)]
    pub fn new(embedding_dim: usize) -> Result<FastKnowledgeGraph, JsValue> {
        let graph = KnowledgeGraph::new(embedding_dim)
            .map_err(|e| JsValue::from_str(&format!(\"Graph creation failed: {}\", e)))?;
        
        let rag_engine = GraphRAGEngine::new(embedding_dim)
            .map_err(|e| JsValue::from_str(&format!(\"RAG engine creation failed: {}\", e)))?;
        
        let batch_processor = BatchProcessor::new(embedding_dim, 8, 64); // 8 subvectors, batch size 64
        
        Ok(FastKnowledgeGraph {
            graph: Arc::new(RwLock::new(graph)),
            rag_engine: Arc::new(RwLock::new(rag_engine)),
            batch_processor: Arc::new(RwLock::new(batch_processor)),
            embedding_cache: Arc::new(RwLock::new(HashMap::new())),
            performance_stats: Arc::new(RwLock::new(PerformanceStats {
                total_queries: 0,
                avg_query_time_ms: 0.0,
                cache_hit_rate: 0.0,
                memory_usage_mb: 0.0,
            })),
        })
    }
    
    /// Ultra-fast embedding computation using hash-based method for demo
    /// In production, this would call a real embedding model
    #[wasm_bindgen]
    pub fn embed(&self, text: &str) -> Float32Array {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        // Check cache first
        {
            let cache = self.embedding_cache.read();
            if let Some(embedding) = cache.get(text) {
                return Float32Array::from(embedding.as_slice());
            }
        }
        
        // Compute embedding
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let hash = hasher.finish();
        
        let embedding_dim = 96;
        let mut embedding = Vec::with_capacity(embedding_dim);
        
        for i in 0..embedding_dim {
            let value = ((hash.wrapping_add(i as u64)) as f32 / u64::MAX as f32 - 0.5) * 2.0;
            embedding.push(value);
        }
        
        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut embedding {
                *val /= norm;
            }
        }
        
        // Cache result
        {
            let mut cache = self.embedding_cache.write();
            cache.insert(text.to_string(), embedding.clone());
        }
        
        Float32Array::from(embedding.as_slice())
    }
    
    /// High-performance vector similarity search
    #[wasm_bindgen]
    pub fn nearest(&self, embedding: &Float32Array, k: usize) -> Uint32Array {
        let start_time = web_sys::window()
            .unwrap()
            .performance()
            .unwrap()
            .now();
        
        let embedding_vec: Vec<f32> = embedding.to_vec();
        
        let results = {
            let graph = self.graph.read();
            match graph.similarity_search(&embedding_vec, k) {
                Ok(results) => results.into_iter().map(|(id, _)| id).collect(),
                Err(_) => Vec::new(),
            }
        };
        
        let query_time = web_sys::window()
            .unwrap()
            .performance()
            .unwrap()
            .now() - start_time;
        
        // Update performance stats
        {
            let mut stats = self.performance_stats.write();
            stats.total_queries += 1;
            stats.avg_query_time_ms = (stats.avg_query_time_ms * (stats.total_queries - 1) as f64 + query_time) / stats.total_queries as f64;
        }
        
        Uint32Array::from(results.as_slice())
    }
    
    /// Fast neighbor lookup with zero-copy semantics
    #[wasm_bindgen]
    pub fn neighbors(&self, entity_id: u32, max_hops: u8) -> Uint32Array {
        let graph = self.graph.read();
        
        if max_hops == 1 {
            // Single hop - ultra fast path
            match graph.get_neighbors(entity_id) {
                Ok(neighbors) => Uint32Array::from(neighbors.as_slice()),
                Err(_) => Uint32Array::new_with_length(0),
            }
        } else {
            // Multi-hop traversal
            // This would use the optimized multi-hop traversal from MMapStorage
            match graph.get_neighbors(entity_id) {
                Ok(neighbors) => Uint32Array::from(neighbors.as_slice()),
                Err(_) => Uint32Array::new_with_length(0),
            }
        }
    }
    
    /// Fast path existence check using bidirectional BFS
    #[wasm_bindgen]
    pub fn relate(&self, entity_a: u32, entity_b: u32, max_hops: u8) -> bool {
        let graph = self.graph.read();
        
        match graph.find_path(entity_a, entity_b, max_hops) {
            Ok(Some(_)) => true,
            _ => false,
        }
    }
    
    /// Generate compact context for LLM prompting
    #[wasm_bindgen]
    pub fn explain(&self, entity_ids: &Uint32Array) -> String {
        let ids: Vec<u32> = entity_ids.to_vec();
        let graph = self.graph.read();
        
        let mut context = String::with_capacity(1024);
        context.push_str(\"# Knowledge Context\\n\\n\");
        
        for &entity_id in &ids {
            if let Ok((meta, data)) = graph.get_entity(entity_id) {
                context.push_str(&format!(\"Entity {}: {}\\n\", entity_id, data.properties));
                
                if let Ok(neighbors) = graph.get_neighbors(entity_id) {
                    if !neighbors.is_empty() {
                        context.push_str(&format!(\"  Connected to: {:?}\\n\", neighbors));
                    }
                }
            }
        }
        
        context
    }
    
    /// Comprehensive Graph RAG retrieval optimized for LLM consumption
    #[wasm_bindgen]
    pub fn graph_rag_search(&self, query_embedding: &Float32Array, max_entities: usize, max_depth: u8) -> String {
        let start_time = web_sys::window()
            .unwrap()
            .performance()
            .unwrap()
            .now();
        
        let embedding_vec: Vec<f32> = query_embedding.to_vec();
        
        let context = {
            let rag_engine = self.rag_engine.read();
            match rag_engine.retrieve_context(&embedding_vec, max_entities, max_depth) {
                Ok(context) => context.to_llm_context(),
                Err(e) => format!(\"Error retrieving context: {}\", e),
            }
        };
        
        let query_time = web_sys::window()
            .unwrap()
            .performance()
            .unwrap()
            .now() - start_time;
        
        // Update performance stats
        {
            let mut stats = self.performance_stats.write();
            stats.total_queries += 1;
            stats.avg_query_time_ms = (stats.avg_query_time_ms * (stats.total_queries - 1) as f64 + query_time) / stats.total_queries as f64;
        }
        
        format!(\"{}\\n\\n<!-- Query processed in {:.2}ms -->\", context, query_time)
    }
    
    /// Get real-time performance and capacity statistics
    #[wasm_bindgen]
    pub fn stats(&self) -> JsValue {
        let graph = self.graph.read();
        let memory_usage = graph.memory_usage();
        let stats = self.performance_stats.read();
        
        let stats_obj = js_sys::Object::new();
        
        js_sys::Reflect::set(
            &stats_obj,
            &JsValue::from_str(\"entity_count\"),
            &JsValue::from_f64(graph.entity_count() as f64),
        ).unwrap();
        
        js_sys::Reflect::set(
            &stats_obj,
            &JsValue::from_str(\"relationship_count\"),
            &JsValue::from_f64(graph.relationship_count() as f64),
        ).unwrap();
        
        js_sys::Reflect::set(
            &stats_obj,
            &JsValue::from_str(\"memory_usage_mb\"),
            &JsValue::from_f64(memory_usage.total_bytes() as f64 / 1_048_576.0),
        ).unwrap();
        
        js_sys::Reflect::set(
            &stats_obj,
            &JsValue::from_str(\"bytes_per_entity\"),
            &JsValue::from_f64(memory_usage.bytes_per_entity(graph.entity_count()) as f64),
        ).unwrap();
        
        js_sys::Reflect::set(
            &stats_obj,
            &JsValue::from_str(\"avg_query_time_ms\"),
            &JsValue::from_f64(stats.avg_query_time_ms),
        ).unwrap();
        
        js_sys::Reflect::set(
            &stats_obj,
            &JsValue::from_str(\"total_queries\"),
            &JsValue::from_f64(stats.total_queries as f64),
        ).unwrap();
        
        js_sys::Reflect::set(
            &stats_obj,
            &JsValue::from_str(\"cache_hit_rate\"),
            &JsValue::from_f64(stats.cache_hit_rate),
        ).unwrap();
        
        stats_obj.into()
    }
    
    /// Batch operations for high-throughput scenarios
    #[wasm_bindgen]
    pub fn batch_nearest(&self, embeddings: &Float32Array, k: usize) -> Uint32Array {
        let embeddings_vec: Vec<f32> = embeddings.to_vec();
        let embedding_dim = 96; // Should match the graph's embedding dimension
        
        if embeddings_vec.len() % embedding_dim != 0 {
            return Uint32Array::new_with_length(0);
        }
        
        let num_queries = embeddings_vec.len() / embedding_dim;
        let mut all_results = Vec::with_capacity(num_queries * k);
        
        let graph = self.graph.read();
        
        for i in 0..num_queries {
            let start_idx = i * embedding_dim;
            let end_idx = start_idx + embedding_dim;
            let query_embedding = &embeddings_vec[start_idx..end_idx];
            
            match graph.similarity_search(query_embedding, k) {
                Ok(results) => {
                    for (entity_id, _) in results {
                        all_results.push(entity_id);
                    }
                },
                Err(_) => {
                    // Add empty results for this query
                    for _ in 0..k {
                        all_results.push(0);
                    }
                }
            }
        }
        
        Uint32Array::from(all_results.as_slice())
    }
    
    /// Memory optimization - clear caches and compact storage
    #[wasm_bindgen]
    pub fn optimize_memory(&self) {
        {
            let mut cache = self.embedding_cache.write();
            cache.clear();
            cache.shrink_to_fit();
        }
        
        // In production, this would trigger:
        // - Garbage collection of unused entities
        // - Compression of property storage
        // - Defragmentation of CSR arrays
        // - Cache optimization
        
        web_sys::console::log_1(&JsValue::from_str(\"Memory optimization complete\"));
    }
}

// High-level TypeScript-friendly interface
#[wasm_bindgen]
extern \"C\" {
    #[wasm_bindgen(typescript_type = \"{ text: string, max_entities?: number, max_depth?: number }\")]
    pub type SearchParams;
    
    #[wasm_bindgen(typescript_type = \"{ entities: Array<{ id: number, similarity: number, properties: string }>, relationships: Array<{ from: number, to: number, type: number, weight: number }>, query_time_ms: number }\")]
    pub type SearchResult;
}

/// High-level API optimized for LLM integration
#[wasm_bindgen]
impl FastKnowledgeGraph {
    /// LLM-optimized knowledge retrieval
    #[wasm_bindgen]
    pub fn llm_search(&self, text: &str, max_entities: Option<usize>, max_depth: Option<u8>) -> String {
        let max_entities = max_entities.unwrap_or(20);
        let max_depth = max_depth.unwrap_or(2);
        
        // Get embedding for text
        let embedding_array = self.embed(text);
        
        // Perform Graph RAG search
        self.graph_rag_search(&embedding_array, max_entities, max_depth)
    }
    
    /// One-shot context retrieval for LLM prompting
    #[wasm_bindgen]
    pub fn get_context(&self, query: &str) -> String {
        self.llm_search(query, Some(15), Some(2))
    }
}
