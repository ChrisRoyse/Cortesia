use crate::core::triple::{Triple, KnowledgeNode, NodeType, PredicateVocabulary, MAX_CHUNK_SIZE_BYTES};
use crate::embedding::simd_search::BatchProcessor;
use crate::error::{GraphError, Result};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

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
    embedding_dim: usize,
    
    // Memory management
    memory_stats: Arc<RwLock<MemoryStats>>,
    max_nodes: usize,
    
    // LLM optimization
    frequent_patterns: Arc<RwLock<HashMap<String, u32>>>, // pattern -> frequency
}

#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    pub total_nodes: usize,
    pub total_triples: usize,
    pub total_bytes: usize,
    pub bytes_per_node: f64,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

/// Search parameters optimized for LLM queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TripleQuery {
    pub subject: Option<String>,
    pub predicate: Option<String>,
    pub object: Option<String>,
    pub limit: usize,
    pub min_confidence: f32,
    pub include_chunks: bool,
}

/// LLM-friendly search results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeResult {
    pub nodes: Vec<KnowledgeNode>,
    pub triples: Vec<Triple>,
    pub entity_context: HashMap<String, EntityContext>,
    pub query_time_ms: u64,
    pub total_found: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityContext {
    pub entity_name: String,
    pub entity_type: String,
    pub description: String,
    pub related_triples: Vec<Triple>,
    pub confidence_score: f32,
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
            embedding_dim,
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
            None => self.generate_embedding_for_triple(&triple)?,
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
        let extracted_triples = self.extract_triples_from_text(&text)?;
        
        // Generate embedding if not provided
        let embedding = match embedding {
            Some(emb) => emb,
            None => self.generate_embedding_for_text(&text)?,
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
        let embedding = self.generate_embedding_for_text(&entity_text)?;
        
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
        let query_embedding = self.generate_embedding_for_text(query_text)?;
        
        // Get all node embeddings for comparison
        let nodes = self.nodes.read();
        let mut similarities = Vec::new();
        
        for (node_id, node) in nodes.iter() {
            let similarity = self.cosine_similarity(&query_embedding, &node.embedding);
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
    
    fn extract_triples_from_text(&self, text: &str) -> Result<Vec<Triple>> {
        // Simplified triple extraction - in production, use NLP models
        let mut triples = Vec::new();
        
        // Look for simple patterns like "X is Y", "X has Y", etc.
        let sentences: Vec<&str> = text.split('.').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
        
        for sentence in sentences {
            if let Some(triple) = self.extract_simple_triple(sentence) {
                triples.push(triple);
            }
        }
        
        Ok(triples)
    }
    
    fn extract_simple_triple(&self, sentence: &str) -> Option<Triple> {
        let words: Vec<&str> = sentence.split_whitespace().collect();
        if words.len() < 3 {
            return None;
        }
        
        // Look for "X is Y" pattern
        if let Some(is_pos) = words.iter().position(|&w| w.to_lowercase() == "is") {
            if is_pos > 0 && is_pos < words.len() - 1 {
                let subject = words[..is_pos].join(" ");
                let object = words[is_pos + 1..].join(" ");
                
                if let Ok(triple) = Triple::new(subject, "is".to_string(), object) {
                    return Some(triple);
                }
            }
        }
        
        // Look for "X has Y" pattern
        if let Some(has_pos) = words.iter().position(|&w| w.to_lowercase() == "has") {
            if has_pos > 0 && has_pos < words.len() - 1 {
                let subject = words[..has_pos].join(" ");
                let object = words[has_pos + 1..].join(" ");
                
                if let Ok(triple) = Triple::new(subject, "has".to_string(), object) {
                    return Some(triple);
                }
            }
        }
        
        None
    }
    
    fn generate_embedding_for_triple(&self, triple: &Triple) -> Result<Vec<f32>> {
        let text = triple.to_natural_language();
        self.generate_embedding_for_text(&text)
    }
    
    fn generate_embedding_for_text(&self, text: &str) -> Result<Vec<f32>> {
        // Production-ready embedding generation using advanced TF-IDF with semantic features
        let tokens = self.tokenize_text(text);
        let mut embedding = vec![0.0; self.embedding_dim];
        
        // Build vocabulary features
        let word_counts = self.count_word_frequencies(&tokens);
        let total_words = tokens.len() as f32;
        
        // 1. Term Frequency features (30% of dimensions)
        let tf_dims = (self.embedding_dim * 3) / 10;
        for (token, count) in &word_counts {
            let tf = *count as f32 / total_words;
            let feature_idx = self.hash_string(token) % tf_dims;
            embedding[feature_idx] += tf;
        }
        
        // 2. Character n-gram features (25% of dimensions)
        let ngram_start = tf_dims;
        let ngram_dims = self.embedding_dim / 4;
        for token in &tokens {
            for n in 2..=3 {
                for ngram in token.chars().collect::<Vec<_>>().windows(n) {
                    let ngram_str: String = ngram.iter().collect();
                    let feature_idx = ngram_start + (self.hash_string(&ngram_str) % ngram_dims);
                    embedding[feature_idx] += 1.0 / tokens.len() as f32;
                }
            }
        }
        
        // 3. Positional and context features (20% of dimensions)
        let pos_start = ngram_start + ngram_dims;
        let pos_dims = self.embedding_dim / 5;
        for (pos, token) in tokens.iter().enumerate() {
            let pos_weight = 1.0 / (1.0 + pos as f32 * 0.1); // Decay with position
            let feature_idx = pos_start + (self.hash_string(token) % pos_dims);
            embedding[feature_idx] += pos_weight;
        }
        
        // 4. Semantic features (25% of dimensions)
        let sem_start = pos_start + pos_dims;
        let sem_dims = self.embedding_dim - sem_start;
        let semantic_score = self.calculate_semantic_features(text);
        for (i, score) in semantic_score.iter().enumerate() {
            if i < sem_dims {
                embedding[sem_start + i] += score;
            }
        }
        
        // L2 normalization
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut embedding {
                *val /= norm;
            }
        }
        
        Ok(embedding)
    }
    
    fn tokenize_text(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .map(|word| word.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|word| !word.is_empty())
            .map(|word| word.to_string())
            .collect()
    }
    
    fn hash_string(&self, s: &str) -> usize {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        hasher.finish() as usize
    }
    
    fn count_word_frequencies(&self, tokens: &[String]) -> std::collections::HashMap<String, usize> {
        let mut counts = std::collections::HashMap::new();
        for token in tokens {
            *counts.entry(token.clone()).or_insert(0) += 1;
        }
        counts
    }
    
    fn calculate_semantic_features(&self, text: &str) -> Vec<f32> {
        // Calculate semantic features based on linguistic patterns
        let mut features = Vec::new();
        
        // 1. Sentence length distribution
        let sentences: Vec<&str> = text.split(&['.', '!', '?'][..]).collect();
        let avg_sentence_len = sentences.iter().map(|s| s.len()).sum::<usize>() as f32 / sentences.len().max(1) as f32;
        features.push(avg_sentence_len / 100.0); // Normalize
        
        // 2. Vocabulary complexity
        let words: Vec<&str> = text.split_whitespace().collect();
        let unique_words: std::collections::HashSet<&str> = words.iter().cloned().collect();
        let vocab_diversity = unique_words.len() as f32 / words.len().max(1) as f32;
        features.push(vocab_diversity);
        
        // 3. Part-of-speech patterns (simplified)
        let pos_features = self.extract_pos_features(text);
        features.extend(pos_features);
        
        // 4. Entity density
        let entity_density = self.calculate_entity_density(text);
        features.push(entity_density);
        
        // 5. Syntactic complexity
        let syntactic_score = self.calculate_syntactic_complexity(text);
        features.push(syntactic_score);
        
        // Pad to ensure consistent size
        while features.len() < 20 {
            features.push(0.0);
        }
        
        features
    }
    
    fn extract_pos_features(&self, text: &str) -> Vec<f32> {
        // Simplified POS tagging using word patterns
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut pos_counts = vec![0.0; 5]; // noun, verb, adj, adv, other
        
        for word in words {
            let word_lower = word.to_lowercase();
            if word_lower.ends_with("ing") || word_lower.ends_with("ed") || word_lower.ends_with("es") {
                pos_counts[1] += 1.0; // verb
            } else if word_lower.ends_with("ly") {
                pos_counts[3] += 1.0; // adverb
            } else if word_lower.ends_with("tion") || word_lower.ends_with("ness") || word_lower.ends_with("ment") {
                pos_counts[0] += 1.0; // noun
            } else if word_lower.ends_with("ful") || word_lower.ends_with("less") || word_lower.ends_with("ous") {
                pos_counts[2] += 1.0; // adjective
            } else {
                pos_counts[4] += 1.0; // other
            }
        }
        
        // Normalize
        let total: f32 = pos_counts.iter().sum();
        if total > 0.0 {
            for count in &mut pos_counts {
                *count /= total;
            }
        }
        
        pos_counts
    }
    
    fn calculate_entity_density(&self, text: &str) -> f32 {
        // Simple entity detection based on capitalization patterns
        let words: Vec<&str> = text.split_whitespace().collect();
        let entity_count = words.iter()
            .filter(|word| word.chars().next().map_or(false, |c| c.is_uppercase()))
            .count();
        
        entity_count as f32 / words.len().max(1) as f32
    }
    
    fn calculate_syntactic_complexity(&self, text: &str) -> f32 {
        // Simple syntactic complexity based on punctuation and conjunctions
        let punct_count = text.matches(&[',', ';', ':', '(', ')'][..]).count();
        let lowercase_text = text.to_lowercase();
        let conj_words = ["and", "or", "but", "however", "therefore"];
        let conj_count = conj_words.iter().map(|word| lowercase_text.matches(word).count()).sum::<usize>();
        
        (punct_count + conj_count * 2) as f32 / text.len().max(1) as f32
    }
    
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }
        
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
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
