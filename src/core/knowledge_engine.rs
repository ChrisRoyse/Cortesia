use crate::core::triple::{Triple, KnowledgeNode, NodeType, PredicateVocabulary, MAX_CHUNK_SIZE_BYTES};
use crate::core::knowledge_embedding::EmbeddingGenerator;
use crate::core::knowledge_extraction::TripleExtractor;
use crate::embedding::simd_search::BatchProcessor;
use crate::error::{GraphError, Result};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use parking_lot::RwLock;

// Re-export types for external use
pub use crate::core::knowledge_types::{MemoryStats, TripleQuery, KnowledgeResult, EntityContext};

/// Ultra-fast knowledge engine optimized for LLM SPO triple storage and retrieval
/// Maintains <60 bytes per entity while supporting dynamic chunk sizes
pub struct KnowledgeEngine {
    // Core storage optimized for triples
    nodes: Arc<RwLock<HashMap<String, KnowledgeNode>>>,
    
    // Triple indexing for fast SPO queries
    subject_index: Arc<RwLock<HashMap<String, HashSet<String>>>>,  // subject -> node_ids
    predicate_index: Arc<RwLock<HashMap<String, HashSet<String>>>>, // predicate -> node_ids
    object_index: Arc<RwLock<HashMap<String, HashSet<String>>>>,   // object -> node_ids
    
    // Entity type tracking for LLM context
    entity_types: Arc<RwLock<HashMap<String, String>>>, // entity_name -> type
    
    // Predicate vocabulary for consistency
    predicate_vocab: PredicateVocabulary,
    
    // Embedding system
    batch_processor: Arc<RwLock<BatchProcessor>>,
    embedding_generator: EmbeddingGenerator,
    triple_extractor: TripleExtractor,
    
    // Memory management
    memory_stats: Arc<RwLock<MemoryStats>>,
    max_nodes: usize,
    
    // LLM optimization
    frequent_patterns: Arc<RwLock<HashMap<String, u32>>>, // pattern -> frequency
}


impl KnowledgeEngine {
    pub fn new(embedding_dim: usize, max_nodes: usize) -> Result<Self> {
        Ok(Self {
            nodes: Arc::new(RwLock::new(HashMap::new())),
            subject_index: Arc::new(RwLock::new(HashMap::new())),
            predicate_index: Arc::new(RwLock::new(HashMap::new())),
            object_index: Arc::new(RwLock::new(HashMap::new())),
            entity_types: Arc::new(RwLock::new(HashMap::new())),
            predicate_vocab: PredicateVocabulary::new(),
            batch_processor: Arc::new(RwLock::new(BatchProcessor::new(embedding_dim, 8, 64))),
            embedding_generator: EmbeddingGenerator::new(embedding_dim),
            triple_extractor: TripleExtractor::new(),
            memory_stats: Arc::new(RwLock::new(MemoryStats::default())),
            max_nodes,
            frequent_patterns: Arc::new(RwLock::new(HashMap::new())),
        })
    }
    
    /// Store a triple with automatic embedding generation
    pub fn store_triple(&self, mut triple: Triple, embedding: Option<Vec<f32>>) -> Result<String> {
        // Normalize predicate using vocabulary
        triple.predicate = self.predicate_vocab.normalize(&triple.predicate);
        
        // Generate embedding if not provided
        let embedding = match embedding {
            Some(emb) => emb,
            None => self.embedding_generator.generate_embedding_for_triple(&triple)?,
        };
        
        // Create knowledge node
        let node = KnowledgeNode::new_triple(triple.clone(), embedding);
        let node_id = node.id.clone();
        let node_size = node.metadata.size_bytes;
        
        // Store node
        {
            let mut nodes = self.nodes.write();
            
            // Check memory limits and evict if necessary
            if nodes.len() >= self.max_nodes {
                self.evict_least_valuable_node(&mut nodes)?;
            }
            
            nodes.insert(node_id.clone(), node);
        }
        
        // Update indexes
        self.update_indexes(&node_id, &triple)?;
        
        // Update memory stats incrementally
        self.update_memory_stats_incremental(1, node_size)?;
        
        // Track frequent patterns for LLM optimization
        self.track_pattern(&triple);
        
        Ok(node_id)
    }
    
    /// Store a knowledge chunk with automatic triple extraction
    pub fn store_chunk(&self, text: String, embedding: Option<Vec<f32>>) -> Result<String> {
        // Validate chunk size
        if text.len() > MAX_CHUNK_SIZE_BYTES {
            return Err(GraphError::SerializationError(
                format!("Chunk too large: {} bytes > {} bytes", text.len(), MAX_CHUNK_SIZE_BYTES)
            ));
        }
        
        // Extract triples from chunk (simplified extraction)
        let extracted_triples = self.triple_extractor.extract_triples_from_text(&text)?;
        
        // Generate embedding if not provided
        let embedding = match embedding {
            Some(emb) => emb,
            None => self.embedding_generator.generate_embedding_for_text(&text)?,
        };
        
        // Create chunk node
        let node = KnowledgeNode::new_chunk(text, embedding, extracted_triples.clone())?;
        let node_id = node.id.clone();
        
        // Store node
        {
            let mut nodes = self.nodes.write();
            
            if nodes.len() >= self.max_nodes {
                self.evict_least_valuable_node(&mut nodes)?;
            }
            
            nodes.insert(node_id.clone(), node);
        }
        
        // Index extracted triples
        for triple in &extracted_triples {
            self.update_indexes(&node_id, triple)?;
        }
        
        self.update_memory_stats()?;
        
        Ok(node_id)
    }
    
    /// Store an entity definition for better LLM context
    pub fn store_entity(&self, name: String, entity_type: String, description: String, properties: HashMap<String, String>) -> Result<String> {
        // Generate embedding for entity
        let entity_text = format!("{name} is a {entity_type} - {description}");
        let embedding = self.embedding_generator.generate_embedding_for_text(&entity_text)?;
        
        // Create entity node
        let node = KnowledgeNode::new_entity(name.clone(), description, entity_type.clone(), properties, embedding)?;
        let node_id = node.id.clone();
        
        // Store node
        {
            let mut nodes = self.nodes.write();
            nodes.insert(node_id.clone(), node);
        }
        
        // Update entity type tracking
        {
            let mut entity_types = self.entity_types.write();
            entity_types.insert(name, entity_type);
        }
        
        self.update_memory_stats()?;
        
        Ok(node_id)
    }
    
    /// Get total number of entities/nodes
    pub fn get_entity_count(&self) -> usize {
        self.nodes.read().len()
    }
    
    /// Query triples with SPO pattern matching
    pub fn query_triples(&self, query: TripleQuery) -> Result<KnowledgeResult> {
        let start_time = std::time::Instant::now();
        
        let mut candidate_node_ids = HashSet::new();
        let mut has_filters = false;
        
        // Subject filter
        if let Some(subject) = &query.subject {
            has_filters = true;
            let subject_index = self.subject_index.read();
            if let Some(node_ids) = subject_index.get(subject) {
                if candidate_node_ids.is_empty() {
                    candidate_node_ids = node_ids.clone();
                } else {
                    candidate_node_ids = candidate_node_ids.intersection(node_ids).cloned().collect();
                }
            } else {
                // No matches for subject
                return Ok(KnowledgeResult {
                    nodes: Vec::new(),
                    triples: Vec::new(),
                    entity_context: HashMap::new(),
                    query_time_ms: start_time.elapsed().as_millis() as u64,
                    total_found: 0,
                });
            }
        }
        
        // Predicate filter
        if let Some(predicate) = &query.predicate {
            has_filters = true;
            let normalized_predicate = self.predicate_vocab.normalize(predicate);
            let predicate_index = self.predicate_index.read();
            if let Some(node_ids) = predicate_index.get(&normalized_predicate) {
                if candidate_node_ids.is_empty() {
                    candidate_node_ids = node_ids.clone();
                } else {
                    candidate_node_ids = candidate_node_ids.intersection(node_ids).cloned().collect();
                }
            } else {
                return Ok(KnowledgeResult {
                    nodes: Vec::new(),
                    triples: Vec::new(),
                    entity_context: HashMap::new(),
                    query_time_ms: start_time.elapsed().as_millis() as u64,
                    total_found: 0,
                });
            }
        }
        
        // Object filter
        if let Some(object) = &query.object {
            has_filters = true;
            let object_index = self.object_index.read();
            if let Some(node_ids) = object_index.get(object) {
                if candidate_node_ids.is_empty() {
                    candidate_node_ids = node_ids.clone();
                } else {
                    candidate_node_ids = candidate_node_ids.intersection(node_ids).cloned().collect();
                }
            } else {
                return Ok(KnowledgeResult {
                    nodes: Vec::new(),
                    triples: Vec::new(),
                    entity_context: HashMap::new(),
                    query_time_ms: start_time.elapsed().as_millis() as u64,
                    total_found: 0,
                });
            }
        }
        
        // If no filters, get all nodes (limited)
        if !has_filters {
            let nodes = self.nodes.read();
            candidate_node_ids = nodes.keys().take(query.limit).cloned().collect();
        }
        
        // Retrieve and filter nodes
        let nodes_guard = self.nodes.read();
        let mut matching_nodes = Vec::new();
        let mut all_triples = Vec::new();
        let mut entity_context = HashMap::new();
        
        for node_id in &candidate_node_ids {
            if let Some(node) = nodes_guard.get(node_id) {
                // Apply confidence filter
                let node_triples = node.get_triples();
                let valid_triples: Vec<&Triple> = node_triples
                    .into_iter()
                    .filter(|t| t.confidence >= query.min_confidence)
                    .collect();
                
                if !valid_triples.is_empty() {
                    // Include chunk nodes only if requested
                    if query.include_chunks || !matches!(node.node_type, NodeType::Chunk) {
                        matching_nodes.push(node.clone());
                    }
                    
                    // Collect triples
                    for triple in valid_triples {
                        all_triples.push(triple.clone());
                        
                        // Build entity context
                        self.add_entity_context(&mut entity_context, &triple.subject);
                        self.add_entity_context(&mut entity_context, &triple.object);
                    }
                }
                
                if matching_nodes.len() >= query.limit {
                    break;
                }
            }
        }
        
        // Sort by quality score
        matching_nodes.sort_by(|a, b| {
            b.metadata.quality_score.partial_cmp(&a.metadata.quality_score).unwrap()
        });
        
        let query_time_ms = start_time.elapsed().as_millis() as u64;
        
        Ok(KnowledgeResult {
            nodes: matching_nodes,
            triples: all_triples,
            entity_context,
            query_time_ms,
            total_found: candidate_node_ids.len(),
        })
    }
    
    /// Semantic search using embeddings
    pub fn semantic_search(&self, query_text: &str, limit: usize) -> Result<KnowledgeResult> {
        let start_time = std::time::Instant::now();
        
        // Generate query embedding
        let query_embedding = self.embedding_generator.generate_embedding_for_text(query_text)?;
        
        // Get all node embeddings for comparison
        let nodes = self.nodes.read();
        let mut similarities = Vec::new();
        
        for (node_id, node) in nodes.iter() {
            let similarity = self.embedding_generator.cosine_similarity(&query_embedding, &node.embedding);
            similarities.push((node_id.clone(), similarity, node.clone()));
        }
        
        // Sort by similarity
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Take top results
        let top_results = similarities.into_iter().take(limit).collect::<Vec<_>>();
        
        let mut matching_nodes = Vec::new();
        let mut all_triples = Vec::new();
        let mut entity_context = HashMap::new();
        
        for (_, _similarity, node) in top_results {
            matching_nodes.push(node.clone());
            
            // Collect triples
            for triple in node.get_triples() {
                all_triples.push(triple.clone());
                self.add_entity_context(&mut entity_context, &triple.subject);
                self.add_entity_context(&mut entity_context, &triple.object);
            }
        }
        
        let query_time_ms = start_time.elapsed().as_millis() as u64;
        
        Ok(KnowledgeResult {
            nodes: matching_nodes,
            triples: all_triples,
            entity_context,
            query_time_ms,
            total_found: nodes.len(),
        })
    }
    
    /// Get entity relationships for LLM context building
    pub fn get_entity_relationships(&self, entity_name: &str, max_hops: u8) -> Result<Vec<Triple>> {
        let mut all_triples = Vec::new();
        let mut visited_entities = HashSet::new();
        let mut current_entities = vec![entity_name.to_string()];
        
        for _hop in 0..max_hops {
            let mut next_entities = Vec::new();
            
            for entity in &current_entities {
                if visited_entities.contains(entity) {
                    continue;
                }
                visited_entities.insert(entity.clone());
                
                // Find triples where entity is subject
                let subject_index = self.subject_index.read();
                if let Some(node_ids) = subject_index.get(entity) {
                    let nodes = self.nodes.read();
                    for node_id in node_ids {
                        if let Some(node) = nodes.get(node_id) {
                            for triple in node.get_triples() {
                                all_triples.push(triple.clone());
                                if !visited_entities.contains(&triple.object) {
                                    next_entities.push(triple.object.clone());
                                }
                            }
                        }
                    }
                }
                
                // Find triples where entity is object
                let object_index = self.object_index.read();
                if let Some(node_ids) = object_index.get(entity) {
                    let nodes = self.nodes.read();
                    for node_id in node_ids {
                        if let Some(node) = nodes.get(node_id) {
                            for triple in node.get_triples() {
                                all_triples.push(triple.clone());
                                if !visited_entities.contains(&triple.subject) {
                                    next_entities.push(triple.subject.clone());
                                }
                            }
                        }
                    }
                }
            }
            
            if next_entities.is_empty() {
                break;
            }
            current_entities = next_entities;
        }
        
        // Remove duplicates
        let mut unique_triples = Vec::new();
        let mut seen = HashSet::new();
        for triple in all_triples {
            let key = format!("{}:{}:{}", triple.subject, triple.predicate, triple.object);
            if !seen.contains(&key) {
                seen.insert(key);
                unique_triples.push(triple);
            }
        }
        
        Ok(unique_triples)
    }
    
    /// Get memory statistics for monitoring
    pub fn get_memory_stats(&self) -> MemoryStats {
        self.memory_stats.read().clone()
    }
    
    /// Get predicate suggestions for LLM
    pub fn suggest_predicates(&self, context: &str) -> Vec<String> {
        self.predicate_vocab.suggest_predicates(context)
    }
    
    /// Get entity types for LLM context
    pub fn get_entity_types(&self) -> HashMap<String, String> {
        self.entity_types.read().clone()
    }
    
    // Helper methods
    
    fn update_indexes(&self, node_id: &str, triple: &Triple) -> Result<()> {
        // Update subject index
        {
            let mut subject_index = self.subject_index.write();
            subject_index.entry(triple.subject.clone())
                .or_default()
                .insert(node_id.to_string());
        }
        
        // Update predicate index
        {
            let mut predicate_index = self.predicate_index.write();
            predicate_index.entry(triple.predicate.clone())
                .or_default()
                .insert(node_id.to_string());
        }
        
        // Update object index
        {
            let mut object_index = self.object_index.write();
            object_index.entry(triple.object.clone())
                .or_default()
                .insert(node_id.to_string());
        }
        
        Ok(())
    }
    
    
    
    fn evict_least_valuable_node(&self, nodes: &mut HashMap<String, KnowledgeNode>) -> Result<()> {
        // Find node with lowest quality score
        let (worst_id, _) = nodes.iter()
            .min_by(|(_, a), (_, b)| a.metadata.quality_score.partial_cmp(&b.metadata.quality_score).unwrap())
            .map(|(id, node)| (id.clone(), node.clone()))
            .ok_or(GraphError::IndexCorruption)?;
        
        // Remove from main storage
        if let Some(node) = nodes.remove(&worst_id) {
            // Remove from indexes
            for triple in node.get_triples() {
                self.remove_from_indexes(&worst_id, triple);
            }
        }
        
        Ok(())
    }
    
    fn remove_from_indexes(&self, node_id: &str, triple: &Triple) {
        // Remove from subject index
        if let Some(mut subject_index) = self.subject_index.try_write() {
            if let Some(node_ids) = subject_index.get_mut(&triple.subject) {
                node_ids.remove(node_id);
                if node_ids.is_empty() {
                    subject_index.remove(&triple.subject);
                }
            }
        }
        
        // Remove from predicate index
        if let Some(mut predicate_index) = self.predicate_index.try_write() {
            if let Some(node_ids) = predicate_index.get_mut(&triple.predicate) {
                node_ids.remove(node_id);
                if node_ids.is_empty() {
                    predicate_index.remove(&triple.predicate);
                }
            }
        }
        
        // Remove from object index
        if let Some(mut object_index) = self.object_index.try_write() {
            if let Some(node_ids) = object_index.get_mut(&triple.object) {
                node_ids.remove(node_id);
                if node_ids.is_empty() {
                    object_index.remove(&triple.object);
                }
            }
        }
    }
    
    fn update_memory_stats(&self) -> Result<()> {
        let nodes = self.nodes.read();
        let total_nodes = nodes.len();
        let total_bytes: usize = nodes.values().map(|n| n.metadata.size_bytes).sum();
        let total_triples: usize = nodes.values().map(|n| n.get_triples().len()).sum();
        
        let mut stats = self.memory_stats.write();
        stats.total_nodes = total_nodes;
        stats.total_triples = total_triples;
        stats.total_bytes = total_bytes;
        stats.bytes_per_node = if total_nodes > 0 {
            total_bytes as f64 / total_nodes as f64
        } else {
            0.0
        };
        
        Ok(())
    }
    
    /// Update memory stats incrementally when adding nodes (much faster)
    fn update_memory_stats_incremental(&self, added_triples: usize, added_bytes: usize) -> Result<()> {
        let mut stats = self.memory_stats.write();
        stats.total_nodes += 1;
        stats.total_triples += added_triples;
        stats.total_bytes += added_bytes;
        stats.bytes_per_node = if stats.total_nodes > 0 {
            stats.total_bytes as f64 / stats.total_nodes as f64
        } else {
            0.0
        };
        Ok(())
    }
    
    fn track_pattern(&self, triple: &Triple) {
        let pattern = format!("{}:{}", triple.predicate, "pattern");
        let mut patterns = self.frequent_patterns.write();
        *patterns.entry(pattern).or_insert(0) += 1;
    }
    
    fn add_entity_context(&self, context_map: &mut HashMap<String, EntityContext>, entity_name: &str) {
        if context_map.contains_key(entity_name) {
            return;
        }
        
        // Use try_read to avoid blocking and nested lock contention
        let entity_type = if let Some(entity_types) = self.entity_types.try_read() {
            entity_types.get(entity_name).cloned().unwrap_or_else(|| "Unknown".to_string())
        } else {
            "Unknown".to_string() // Fallback if lock is busy
        };
        
        // Skip expensive relationship lookup to avoid nested locking
        // This prevents the hanging issue while maintaining performance
        let related_triples = Vec::new(); // We can populate this differently if needed
        
        context_map.insert(entity_name.to_string(), EntityContext {
            entity_name: entity_name.to_string(),
            entity_type,
            description: format!("Entity: {entity_name}"),
            related_triples,
            confidence_score: 1.0,
        });
    }
    
    /// Convenience method for tests - add a triple with basic parameters
    pub fn add_triple(&self, subject: &str, predicate: &str, object: &str, confidence: f32) -> Result<String> {
        let triple = Triple {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
            confidence,
            source: None,
        };
        self.store_triple(triple, None)
    }
    
    /// Convenience method for tests - add a knowledge chunk with basic parameters
    pub fn add_knowledge_chunk(&self, title: &str, content: &str, category: Option<&str>, source: Option<&str>) -> Result<String> {
        // Build text with metadata
        let mut text_parts = vec![format!("{}: {}", title, content)];
        
        if let Some(cat) = category {
            text_parts.push(format!("Category: {cat}"));
        }
        
        if let Some(src) = source {
            text_parts.push(format!("Source: {src}"));
        }
        
        let text = text_parts.join("\n");
        self.store_chunk(text, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::triple::{Triple, NodeType};
    use std::collections::HashMap;

    fn create_test_engine() -> KnowledgeEngine {
        KnowledgeEngine::new(128, 1000).expect("Failed to create test engine")
    }

    fn create_test_triple() -> Triple {
        Triple {
            subject: "Alice".to_string(),
            predicate: "likes".to_string(),
            object: "chocolate".to_string(),
            confidence: 0.9,
            source: Some("test".to_string()),
        }
    }

    #[test]
    fn test_store_triple_basic() {
        let engine = create_test_engine();
        let triple = create_test_triple();
        
        let result = engine.store_triple(triple.clone(), None);
        assert!(result.is_ok());
        
        let node_id = result.unwrap();
        assert!(!node_id.is_empty());
        
        // Verify node was stored
        let nodes = engine.nodes.read();
        assert!(nodes.contains_key(&node_id));
        
        // Verify the stored node
        let stored_node = &nodes[&node_id];
        assert_eq!(stored_node.node_type, NodeType::Triple);
        let stored_triples = stored_node.get_triples();
        assert_eq!(stored_triples.len(), 1);
        assert_eq!(stored_triples[0].subject, triple.subject);
        assert_eq!(stored_triples[0].predicate, "likes"); // Should be normalized
        assert_eq!(stored_triples[0].object, triple.object);
    }

    #[test]
    fn test_store_triple_with_embedding() {
        let engine = create_test_engine();
        let triple = create_test_triple();
        let embedding = vec![0.1; 128];
        
        let result = engine.store_triple(triple, Some(embedding.clone()));
        assert!(result.is_ok());
        
        let node_id = result.unwrap();
        let nodes = engine.nodes.read();
        let stored_node = &nodes[&node_id];
        assert_eq!(stored_node.embedding, embedding);
    }

    #[test]
    fn test_store_triple_predicate_normalization() {
        let engine = create_test_engine();
        let mut triple = create_test_triple();
        triple.predicate = "LIKES".to_string(); // Test normalization
        
        let result = engine.store_triple(triple, None);
        assert!(result.is_ok());
        
        let node_id = result.unwrap();
        let nodes = engine.nodes.read();
        let stored_node = &nodes[&node_id];
        let stored_triples = stored_node.get_triples();
        // Predicate should be normalized by vocabulary
        assert_eq!(stored_triples[0].predicate, "likes");
    }

    #[test]
    fn test_store_triple_memory_limit() {
        // Create engine with very small limit
        let engine = KnowledgeEngine::new(128, 2).expect("Failed to create test engine");
        
        // Store enough triples to trigger eviction
        for i in 0..5 {
            let mut triple = create_test_triple();
            triple.subject = format!("subject_{i}");
            let result = engine.store_triple(triple, None);
            assert!(result.is_ok());
        }
        
        // Should not exceed max_nodes
        let nodes = engine.nodes.read();
        assert!(nodes.len() <= 2);
    }

    #[test]
    fn test_query_triples_empty_query() {
        let engine = create_test_engine();
        
        let query = TripleQuery {
            subject: None,
            predicate: None,
            object: None,
            min_confidence: 0.0,
            limit: 10,
            include_chunks: false,
        };
        
        let result = engine.query_triples(query);
        assert!(result.is_ok());
        
        let knowledge_result = result.unwrap();
        assert_eq!(knowledge_result.nodes.len(), 0);
        assert_eq!(knowledge_result.triples.len(), 0);
        assert!(knowledge_result.query_time_ms < 10000); // Query should complete in reasonable time
    }

    #[test]
    fn test_query_triples_subject_filter() {
        let engine = create_test_engine();
        let triple = create_test_triple();
        
        // Store a triple
        engine.store_triple(triple.clone(), None).unwrap();
        
        // Query by subject
        let query = TripleQuery {
            subject: Some("Alice".to_string()),
            predicate: None,
            object: None,
            min_confidence: 0.0,
            limit: 10,
            include_chunks: false,
        };
        
        let result = engine.query_triples(query).unwrap();
        assert_eq!(result.nodes.len(), 1);
        assert_eq!(result.triples.len(), 1);
        assert_eq!(result.triples[0].subject, "Alice");
    }

    #[test]
    fn test_query_triples_predicate_filter() {
        let engine = create_test_engine();
        let triple = create_test_triple();
        
        // Store a triple
        engine.store_triple(triple.clone(), None).unwrap();
        
        // Query by predicate
        let query = TripleQuery {
            subject: None,
            predicate: Some("likes".to_string()),
            object: None,
            min_confidence: 0.0,
            limit: 10,
            include_chunks: false,
        };
        
        let result = engine.query_triples(query).unwrap();
        assert_eq!(result.nodes.len(), 1);
        assert_eq!(result.triples.len(), 1);
        assert_eq!(result.triples[0].predicate, "likes");
    }

    #[test]
    fn test_query_triples_object_filter() {
        let engine = create_test_engine();
        let triple = create_test_triple();
        
        // Store a triple
        engine.store_triple(triple.clone(), None).unwrap();
        
        // Query by object
        let query = TripleQuery {
            subject: None,
            predicate: None,
            object: Some("chocolate".to_string()),
            min_confidence: 0.0,
            limit: 10,
            include_chunks: false,
        };
        
        let result = engine.query_triples(query).unwrap();
        assert_eq!(result.nodes.len(), 1);
        assert_eq!(result.triples.len(), 1);
        assert_eq!(result.triples[0].object, "chocolate");
    }

    #[test]
    fn test_query_triples_no_matches() {
        let engine = create_test_engine();
        let triple = create_test_triple();
        
        // Store a triple
        engine.store_triple(triple, None).unwrap();
        
        // Query for non-existent subject
        let query = TripleQuery {
            subject: Some("NonExistent".to_string()),
            predicate: None,
            object: None,
            min_confidence: 0.0,
            limit: 10,
            include_chunks: false,
        };
        
        let result = engine.query_triples(query).unwrap();
        assert_eq!(result.nodes.len(), 0);
        assert_eq!(result.triples.len(), 0);
        assert_eq!(result.total_found, 0);
    }

    #[test]
    fn test_query_triples_confidence_filter() {
        let engine = create_test_engine();
        let mut triple = create_test_triple();
        triple.confidence = 0.3; // Low confidence
        
        // Store a triple
        engine.store_triple(triple, None).unwrap();
        
        // Query with high confidence threshold
        let query = TripleQuery {
            subject: Some("Alice".to_string()),
            predicate: None,
            object: None,
            min_confidence: 0.5, // Higher than stored triple
            limit: 10,
            include_chunks: false,
        };
        
        let result = engine.query_triples(query).unwrap();
        assert_eq!(result.nodes.len(), 0); // Should be filtered out
        assert_eq!(result.triples.len(), 0);
    }

    #[test]
    fn test_query_triples_limit() {
        let engine = create_test_engine();
        
        // Store multiple triples
        for i in 0..5 {
            let mut triple = create_test_triple();
            triple.subject = format!("Alice_{i}");
            engine.store_triple(triple, None).unwrap();
        }
        
        // Query with small limit
        let query = TripleQuery {
            subject: None,
            predicate: Some("likes".to_string()),
            object: None,
            min_confidence: 0.0,
            limit: 2,
            include_chunks: false,
        };
        
        let result = engine.query_triples(query).unwrap();
        assert!(result.nodes.len() <= 2);
    }

    #[test]
    fn test_query_triples_combined_filters() {
        let engine = create_test_engine();
        let triple = create_test_triple();
        
        // Store a triple
        engine.store_triple(triple, None).unwrap();
        
        // Query with multiple filters
        let query = TripleQuery {
            subject: Some("Alice".to_string()),
            predicate: Some("likes".to_string()),
            object: Some("chocolate".to_string()),
            min_confidence: 0.0,
            limit: 10,
            include_chunks: false,
        };
        
        let result = engine.query_triples(query).unwrap();
        assert_eq!(result.nodes.len(), 1);
        assert_eq!(result.triples.len(), 1);
    }

    #[test]
    fn test_semantic_search_empty_query() {
        let engine = create_test_engine();
        
        let result = engine.semantic_search("", 5);
        assert!(result.is_ok());
        
        let knowledge_result = result.unwrap();
        assert_eq!(knowledge_result.nodes.len(), 0);
        assert!(knowledge_result.query_time_ms < 10000); // Query should complete in reasonable time
    }

    #[test]
    fn test_semantic_search_with_stored_data() {
        let engine = create_test_engine();
        let triple = create_test_triple();
        
        // Store a triple
        engine.store_triple(triple, None).unwrap();
        
        // Perform semantic search
        let result = engine.semantic_search("Alice likes chocolate", 5);
        assert!(result.is_ok());
        
        let knowledge_result = result.unwrap();
        assert_eq!(knowledge_result.nodes.len(), 1);
        assert_eq!(knowledge_result.triples.len(), 1);
        assert!(knowledge_result.query_time_ms < 10000); // Query should complete in reasonable time
    }

    #[test]
    fn test_semantic_search_limit() {
        let engine = create_test_engine();
        
        // Store multiple triples
        for i in 0..5 {
            let mut triple = create_test_triple();
            triple.subject = format!("Person_{i}");
            engine.store_triple(triple, None).unwrap();
        }
        
        // Search with limit
        let result = engine.semantic_search("person likes", 2);
        assert!(result.is_ok());
        
        let knowledge_result = result.unwrap();
        assert!(knowledge_result.nodes.len() <= 2);
    }

    #[test]
    fn test_get_entity_relationships_empty() {
        let engine = create_test_engine();
        
        let result = engine.get_entity_relationships("NonExistent", 2);
        assert!(result.is_ok());
        
        let triples = result.unwrap();
        assert_eq!(triples.len(), 0);
    }

    #[test]
    fn test_get_entity_relationships_single_hop() {
        let engine = create_test_engine();
        let triple = create_test_triple();
        
        // Store a triple
        engine.store_triple(triple.clone(), None).unwrap();
        
        // Get relationships for Alice (subject)
        let result = engine.get_entity_relationships("Alice", 1);
        assert!(result.is_ok());
        
        let triples = result.unwrap();
        assert_eq!(triples.len(), 1);
        assert_eq!(triples[0].subject, "Alice");
        assert_eq!(triples[0].object, "chocolate");
    }

    #[test]
    fn test_get_entity_relationships_multi_hop() {
        let engine = create_test_engine();
        
        // Create a chain: Alice -> likes -> chocolate -> hasProperty -> sweet
        let triple1 = Triple {
            subject: "Alice".to_string(),
            predicate: "likes".to_string(),
            object: "chocolate".to_string(),
            confidence: 0.9,
            source: Some("test".to_string()),
        };
        
        let triple2 = Triple {
            subject: "chocolate".to_string(),
            predicate: "hasProperty".to_string(),
            object: "sweet".to_string(),
            confidence: 0.8,
            source: Some("test".to_string()),
        };
        
        engine.store_triple(triple1, None).unwrap();
        engine.store_triple(triple2, None).unwrap();
        
        // Get 2-hop relationships from Alice
        let result = engine.get_entity_relationships("Alice", 2);
        assert!(result.is_ok());
        
        let triples = result.unwrap();
        assert_eq!(triples.len(), 2);
        
        // Should contain both triples in the chain
        let subjects: Vec<&String> = triples.iter().map(|t| &t.subject).collect();
        assert!(subjects.contains(&&"Alice".to_string()));
        assert!(subjects.contains(&&"chocolate".to_string()));
    }

    #[test]
    fn test_get_entity_relationships_max_hops_zero() {
        let engine = create_test_engine();
        let triple = create_test_triple();
        
        // Store a triple
        engine.store_triple(triple, None).unwrap();
        
        // Get relationships with 0 hops
        let result = engine.get_entity_relationships("Alice", 0);
        assert!(result.is_ok());
        
        let triples = result.unwrap();
        assert_eq!(triples.len(), 0); // No hops means no relationships
    }

    #[test]
    fn test_get_entity_relationships_as_object() {
        let engine = create_test_engine();
        let triple = create_test_triple();
        
        // Store a triple
        engine.store_triple(triple.clone(), None).unwrap();
        
        // Get relationships for chocolate (object)
        let result = engine.get_entity_relationships("chocolate", 1);
        assert!(result.is_ok());
        
        let triples = result.unwrap();
        assert_eq!(triples.len(), 1);
        assert_eq!(triples[0].object, "chocolate");
        assert_eq!(triples[0].subject, "Alice");
    }

    #[test]
    fn test_get_entity_relationships_duplicate_removal() {
        let engine = create_test_engine();
        
        // Store the same triple twice (different node IDs)
        let triple = create_test_triple();
        engine.store_triple(triple.clone(), None).unwrap();
        engine.store_triple(triple, None).unwrap();
        
        let result = engine.get_entity_relationships("Alice", 1);
        assert!(result.is_ok());
        
        let triples = result.unwrap();
        // Should deduplicate identical triples
        assert_eq!(triples.len(), 1);
    }

    #[test]
    fn test_update_indexes() {
        let engine = create_test_engine();
        let triple = create_test_triple();
        let node_id = "test_node_123";
        
        // Test private method through public interface
        let result = engine.update_indexes(node_id, &triple);
        assert!(result.is_ok());
        
        // Verify subject index
        {
            let subject_index = engine.subject_index.read();
            assert!(subject_index.contains_key("Alice"));
            assert!(subject_index["Alice"].contains(node_id));
        }
        
        // Verify predicate index
        {
            let predicate_index = engine.predicate_index.read();
            assert!(predicate_index.contains_key("likes"));
            assert!(predicate_index["likes"].contains(node_id));
        }
        
        // Verify object index
        {
            let object_index = engine.object_index.read();
            assert!(object_index.contains_key("chocolate"));
            assert!(object_index["chocolate"].contains(node_id));
        }
    }

    #[test]
    fn test_remove_from_indexes() {
        let engine = create_test_engine();
        let triple = create_test_triple();
        let node_id = "test_node_123";
        
        // First add to indexes
        engine.update_indexes(node_id, &triple).unwrap();
        
        // Then remove
        engine.remove_from_indexes(node_id, &triple);
        
        // Verify removal from subject index
        {
            let subject_index = engine.subject_index.read();
            assert!(!subject_index.contains_key("Alice") || 
                   !subject_index["Alice"].contains(node_id));
        }
        
        // Verify removal from predicate index
        {
            let predicate_index = engine.predicate_index.read();
            assert!(!predicate_index.contains_key("likes") || 
                   !predicate_index["likes"].contains(node_id));
        }
        
        // Verify removal from object index
        {
            let object_index = engine.object_index.read();
            assert!(!object_index.contains_key("chocolate") || 
                   !object_index["chocolate"].contains(node_id));
        }
    }

    #[test]
    fn test_update_memory_stats() {
        let engine = create_test_engine();
        
        // Initial stats should be zero
        {
            let stats = engine.memory_stats.read();
            assert_eq!(stats.total_nodes, 0);
            assert_eq!(stats.total_triples, 0);
            assert_eq!(stats.total_bytes, 0);
        }
        
        // Store a triple and update stats
        let triple = create_test_triple();
        engine.store_triple(triple, None).unwrap();
        
        // Stats should be updated automatically
        {
            let stats = engine.memory_stats.read();
            assert_eq!(stats.total_nodes, 1);
            assert_eq!(stats.total_triples, 1);
            assert!(stats.total_bytes > 0);
            assert!(stats.bytes_per_node > 0.0);
        }
    }

    #[test]
    fn test_track_pattern() {
        let engine = create_test_engine();
        let triple = create_test_triple();
        
        // Track pattern
        engine.track_pattern(&triple);
        
        // Verify pattern was tracked
        let patterns = engine.frequent_patterns.read();
        let expected_pattern = "likes:pattern";
        assert!(patterns.contains_key(expected_pattern));
        assert_eq!(patterns[expected_pattern], 1);
        
        // Track same pattern again
        engine.track_pattern(&triple);
        
        // Should increment counter
        let patterns = engine.frequent_patterns.read();
        assert_eq!(patterns[expected_pattern], 2);
    }

    #[test]
    fn test_add_entity_context() {
        let engine = create_test_engine();
        let mut context_map = HashMap::new();
        
        // Add entity context
        engine.add_entity_context(&mut context_map, "Alice");
        
        // Verify context was added
        assert!(context_map.contains_key("Alice"));
        let context = &context_map["Alice"];
        assert_eq!(context.entity_name, "Alice");
        assert_eq!(context.entity_type, "Unknown"); // Default type
        assert!(context.description.contains("Alice"));
        assert_eq!(context.confidence_score, 1.0);
    }

    #[test]
    fn test_add_entity_context_with_type() {
        let engine = create_test_engine();
        
        // First store an entity with type
        engine.store_entity(
            "Alice".to_string(),
            "Person".to_string(),
            "A human person".to_string(),
            HashMap::new()
        ).unwrap();
        
        let mut context_map = HashMap::new();
        engine.add_entity_context(&mut context_map, "Alice");
        
        // Verify context uses stored type
        assert!(context_map.contains_key("Alice"));
        let context = &context_map["Alice"];
        assert_eq!(context.entity_type, "Person");
    }

    #[test]
    fn test_add_entity_context_duplicate() {
        let engine = create_test_engine();
        let mut context_map = HashMap::new();
        
        // Add context twice
        engine.add_entity_context(&mut context_map, "Alice");
        engine.add_entity_context(&mut context_map, "Alice");
        
        // Should only have one entry
        assert_eq!(context_map.len(), 1);
        assert!(context_map.contains_key("Alice"));
    }

    #[test]
    fn test_evict_least_valuable_node() {
        // Create engine with small limit
        let engine = KnowledgeEngine::new(128, 2).unwrap();
        
        // Store multiple nodes with different quality scores
        let mut triple1 = create_test_triple();
        triple1.subject = "Alice".to_string();
        let node_id1 = engine.store_triple(triple1, None).unwrap();
        
        let mut triple2 = create_test_triple();
        triple2.subject = "Bob".to_string();
        let node_id2 = engine.store_triple(triple2, None).unwrap();
        
        // Manually set different quality scores
        {
            let mut nodes = engine.nodes.write();
            if let Some(node) = nodes.get_mut(&node_id1) {
                node.metadata.quality_score = 0.1; // Low quality
            }
            if let Some(node) = nodes.get_mut(&node_id2) {
                node.metadata.quality_score = 0.9; // High quality
            }
        }
        
        // Store another triple to trigger eviction
        let mut triple3 = create_test_triple();
        triple3.subject = "Charlie".to_string();
        engine.store_triple(triple3, None).unwrap();
        
        // Should not exceed limit
        let nodes = engine.nodes.read();
        assert!(nodes.len() <= 2);
        
        // Low quality node should be evicted
        assert!(!nodes.contains_key(&node_id1));
    }

    #[test]
    fn test_store_triple_large_batch() {
        let engine = create_test_engine();
        let mut stored_ids = Vec::new();
        
        // Store many triples
        for i in 0..100 {
            let mut triple = create_test_triple();
            triple.subject = format!("Subject_{i}");
            triple.object = format!("Object_{i}");
            
            let result = engine.store_triple(triple, None);
            assert!(result.is_ok());
            stored_ids.push(result.unwrap());
        }
        
        // Verify all stored
        let nodes = engine.nodes.read();
        for id in &stored_ids {
            if nodes.len() < engine.max_nodes {
                assert!(nodes.contains_key(id));
            }
        }
        
        // Verify indexes are consistent
        let subject_index = engine.subject_index.read();
        let predicate_index = engine.predicate_index.read();
        let object_index = engine.object_index.read();
        
        assert!(!subject_index.is_empty());
        assert!(!predicate_index.is_empty());
        assert!(!object_index.is_empty());
    }

    #[test]
    fn test_query_triples_performance() {
        let engine = create_test_engine();
        
        // Store many triples for performance testing
        for i in 0..1000 {
            let mut triple = create_test_triple();
            triple.subject = format!("Subject_{i}");
            engine.store_triple(triple, None).unwrap();
        }
        
        let query = TripleQuery {
            subject: None,
            predicate: Some("likes".to_string()),
            object: None,
            min_confidence: 0.0,
            limit: 10,
            include_chunks: false,
        };
        
        let start = std::time::Instant::now();
        let result = engine.query_triples(query);
        let duration = start.elapsed();
        
        assert!(result.is_ok());
        assert!(duration.as_millis() < 1000); // Should be fast
        
        let knowledge_result = result.unwrap();
        assert!(knowledge_result.query_time_ms > 0);
        assert!(knowledge_result.nodes.len() <= 10);
    }

    #[test]
    fn test_semantic_search_performance() {
        let engine = create_test_engine();
        
        // Store many triples
        for i in 0..100 {
            let mut triple = create_test_triple();
            triple.subject = format!("Person_{i}");
            engine.store_triple(triple, None).unwrap();
        }
        
        let start = std::time::Instant::now();
        let result = engine.semantic_search("person likes things", 5);
        let duration = start.elapsed();
        
        assert!(result.is_ok());
        assert!(duration.as_millis() < 5000); // Should be reasonably fast
        
        let knowledge_result = result.unwrap();
        assert!(knowledge_result.query_time_ms > 0);
        assert!(knowledge_result.nodes.len() <= 5);
    }

    #[test]
    fn test_memory_management_under_pressure() {
        // Create engine with very small limit
        let engine = KnowledgeEngine::new(128, 5).unwrap();
        
        // Store many triples to test memory management
        for i in 0..20 {
            let mut triple = create_test_triple();
            triple.subject = format!("Subject_{i}");
            triple.object = format!("Object_{i}");
            
            let result = engine.store_triple(triple, None);
            assert!(result.is_ok());
        }
        
        // Should not exceed memory limit
        let nodes = engine.nodes.read();
        assert!(nodes.len() <= 5);
        
        // Memory stats should be consistent
        let stats = engine.get_memory_stats();
        assert_eq!(stats.total_nodes, nodes.len());
        assert!(stats.bytes_per_node > 0.0);
    }
}
