use crate::core::triple::{Triple, KnowledgeNode, NodeType, PredicateVocabulary, MAX_CHUNK_SIZE_BYTES};
use crate::core::knowledge_types::{MemoryStats, TripleQuery, KnowledgeResult, EntityContext};
use crate::core::knowledge_embedding::EmbeddingGenerator;
use crate::core::knowledge_extraction::TripleExtractor;
use crate::embedding::simd_search::BatchProcessor;
use crate::error::{GraphError, Result};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use parking_lot::RwLock;

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
        
        // Update memory stats
        self.update_memory_stats()?;
        
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
        let entity_text = format!("{} is a {} - {}", name, entity_type, description);
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
                .or_insert_with(HashSet::new)
                .insert(node_id.to_string());
        }
        
        // Update predicate index
        {
            let mut predicate_index = self.predicate_index.write();
            predicate_index.entry(triple.predicate.clone())
                .or_insert_with(HashSet::new)
                .insert(node_id.to_string());
        }
        
        // Update object index
        {
            let mut object_index = self.object_index.write();
            object_index.entry(triple.object.clone())
                .or_insert_with(HashSet::new)
                .insert(node_id.to_string());
        }
        
        Ok(())
    }
    
    
    
    fn evict_least_valuable_node(&self, nodes: &mut HashMap<String, KnowledgeNode>) -> Result<()> {
        // Find node with lowest quality score
        let (worst_id, _) = nodes.iter()
            .min_by(|(_, a), (_, b)| a.metadata.quality_score.partial_cmp(&b.metadata.quality_score).unwrap())
            .map(|(id, node)| (id.clone(), node.clone()))
            .ok_or_else(|| GraphError::IndexCorruption)?;
        
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
    
    fn track_pattern(&self, triple: &Triple) {
        let pattern = format!("{}:{}", triple.predicate, "pattern");
        let mut patterns = self.frequent_patterns.write();
        *patterns.entry(pattern).or_insert(0) += 1;
    }
    
    fn add_entity_context(&self, context_map: &mut HashMap<String, EntityContext>, entity_name: &str) {
        if context_map.contains_key(entity_name) {
            return;
        }
        
        let entity_types = self.entity_types.read();
        let entity_type = entity_types.get(entity_name).cloned().unwrap_or_else(|| "Unknown".to_string());
        
        // Get related triples for this entity
        let related_triples = self.get_entity_relationships(entity_name, 1).unwrap_or_default();
        
        context_map.insert(entity_name.to_string(), EntityContext {
            entity_name: entity_name.to_string(),
            entity_type,
            description: format!("Entity: {}", entity_name),
            related_triples,
            confidence_score: 1.0,
        });
    }
}
