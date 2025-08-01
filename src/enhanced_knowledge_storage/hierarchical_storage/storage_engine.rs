//! Hierarchical Storage Engine
//! 
//! Main storage engine that coordinates all hierarchical storage components
//! and provides the primary interface for storing and retrieving knowledge.

use std::sync::Arc;
use std::collections::HashMap;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::{info, error, debug, instrument};
use crate::enhanced_knowledge_storage::{
    model_management::ModelResourceManager,
    knowledge_processing::types::KnowledgeProcessingResult,
    hierarchical_storage::{
        types::{HierarchicalStorageConfig, HierarchicalKnowledge, KnowledgeLayer, GlobalContext, HierarchicalStorageResult, HierarchicalStorageError, HierarchicalStorageStats, SemanticLinkType},
        KnowledgeLayerManager,
        SemanticLinkManager,
        HierarchicalIndexManager,
        IndexQuery,
        SearchResult,
    },
    logging::LogContext,
};

/// Main hierarchical storage engine
pub struct HierarchicalStorageEngine {
    model_manager: Arc<ModelResourceManager>,
    layer_manager: Arc<KnowledgeLayerManager>,
    link_manager: Arc<SemanticLinkManager>,
    index_manager: Arc<HierarchicalIndexManager>,
    storage: Arc<RwLock<InMemoryStorage>>,
    config: HierarchicalStorageConfig,
}

/// In-memory storage for hierarchical knowledge
#[derive(Debug, Default)]
struct InMemoryStorage {
    documents: HashMap<String, HierarchicalKnowledge>,
    layer_cache: HashMap<String, KnowledgeLayer>,
    access_log: Vec<AccessLogEntry>,
}

/// Access log entry for tracking usage
#[derive(Debug, Clone)]
struct AccessLogEntry {
    timestamp: u64,
    document_id: String,
    layer_id: Option<String>,
    operation: StorageOperation,
}

/// Types of storage operations
#[derive(Debug, Clone)]
enum StorageOperation {
    Store,
    Retrieve,
    Update,
    Delete,
    Search,
}

impl HierarchicalStorageEngine {
    /// Create new hierarchical storage engine
    pub fn new(
        model_manager: Arc<ModelResourceManager>,
        config: HierarchicalStorageConfig,
    ) -> Self {
        let layer_manager = Arc::new(KnowledgeLayerManager::new(
            model_manager.clone(),
            config.clone(),
        ));
        
        let link_manager = Arc::new(SemanticLinkManager::new(
            model_manager.clone(),
            config.clone(),
        ));
        
        let index_manager = Arc::new(HierarchicalIndexManager::new(
            model_manager.clone(),
            config.clone(),
        ));
        
        Self {
            model_manager,
            layer_manager,
            link_manager,
            index_manager,
            storage: Arc::new(RwLock::new(InMemoryStorage::default())),
            config,
        }
    }
    
    /// Store processed knowledge in hierarchical structure
    #[instrument(
        skip(self, processing_result, global_context),
        fields(
            document_id = %processing_result.document_id,
            chunks_count = processing_result.chunks.len(),
            entities_count = processing_result.global_entities.len(),
            relationships_count = processing_result.global_relationships.len()
        )
    )]
    pub async fn store_knowledge(
        &self,
        processing_result: KnowledgeProcessingResult,
        global_context: GlobalContext,
    ) -> HierarchicalStorageResult<String> {
        let start_time = Instant::now();
        let document_id = processing_result.document_id.clone();
        
        let log_context = LogContext::new("store_knowledge", "hierarchical_storage_engine")
            .with_request_id(document_id.clone());
        
        info!(
            context = ?log_context,
            document_id = %document_id,
            chunks_count = processing_result.chunks.len(),
            entities_count = processing_result.global_entities.len(),
            relationships_count = processing_result.global_relationships.len(),
            "Starting hierarchical knowledge storage"
        );
        
        // Step 1: Create hierarchical layers
        info!("Step 1/5: Creating hierarchical layers");
        let layer_start = Instant::now();
        let layers = self.layer_manager
            .create_hierarchical_layers(&processing_result)
            .await
            .map_err(|e| {
                error!(
                    context = ?log_context,
                    error = %e,
                    "Failed to create hierarchical layers"
                );
                e
            })?;
        debug!(
            layer_creation_time_ms = layer_start.elapsed().as_millis(),
            layers_created = layers.len(),
            "Hierarchical layer creation completed"
        );
        
        // Validate layer count
        if layers.len() > self.config.max_layers_per_document {
            let error_msg = format!("Too many layers: {} > {}", layers.len(), self.config.max_layers_per_document);
            error!(
                context = ?log_context,
                layers_created = layers.len(),
                max_allowed = self.config.max_layers_per_document,
                "Layer count exceeds maximum allowed"
            );
            return Err(HierarchicalStorageError::InvalidLayerStructure(error_msg));
        }
        
        // Step 2: Build semantic link graph
        info!("Step 2/5: Building semantic link graph");
        let link_start = Instant::now();
        let semantic_links = self.link_manager
            .build_semantic_link_graph(&layers)
            .await
            .map_err(|e| {
                error!(
                    context = ?log_context,
                    error = %e,
                    "Failed to build semantic link graph"
                );
                e
            })?;
        debug!(
            link_building_time_ms = link_start.elapsed().as_millis(),
            semantic_links_created = semantic_links.len(),
            "Semantic link graph building completed"
        );
        
        // Step 3: Build hierarchical index
        info!("Step 3/5: Building hierarchical index");
        let index_start = Instant::now();
        let retrieval_index = self.index_manager
            .build_hierarchical_index(&layers)
            .await
            .map_err(|e| {
                error!(
                    context = ?log_context,
                    error = %e,
                    "Failed to build hierarchical index"
                );
                e
            })?;
        debug!(
            index_building_time_ms = index_start.elapsed().as_millis(),
            index_entries = retrieval_index.layer_index.len(),
            "Hierarchical index building completed"
        );
        
        // Step 4: Create hierarchical knowledge structure
        info!("Step 4/5: Creating hierarchical knowledge structure");
        let struct_start = Instant::now();
        let mut hierarchical_knowledge = HierarchicalKnowledge::new(
            document_id.clone(),
            global_context,
        );
        
        hierarchical_knowledge.knowledge_layers = layers.clone();
        hierarchical_knowledge.semantic_links = semantic_links.clone();
        hierarchical_knowledge.retrieval_index = retrieval_index;
        debug!(
            structure_creation_time_ms = struct_start.elapsed().as_millis(),
            "Hierarchical knowledge structure creation completed"
        );
        
        // Step 5: Store in memory
        info!("Step 5/5: Storing in memory and updating cache");
        let storage_start = Instant::now();
        let mut storage = self.storage.write().await;
        
        let initial_layer_cache_size = storage.layer_cache.len();
        let initial_document_count = storage.documents.len();
        
        // Cache individual layers
        for layer in &layers {
            storage.layer_cache.insert(layer.layer_id.clone(), layer.clone());
        }
        
        // Store complete document
        storage.documents.insert(document_id.clone(), hierarchical_knowledge);
        
        // Log operation
        storage.access_log.push(AccessLogEntry {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            document_id: document_id.clone(),
            layer_id: None,
            operation: StorageOperation::Store,
        });
        
        debug!(
            storage_time_ms = storage_start.elapsed().as_millis(),
            layer_cache_before = initial_layer_cache_size,
            layer_cache_after = storage.layer_cache.len(),
            document_count_before = initial_document_count,
            document_count_after = storage.documents.len(),
            "In-memory storage completed"
        );
        
        // Manage cache size
        let cache_mgmt_start = Instant::now();
        self.manage_cache_size(&mut storage).await
            .map_err(|e| {
                error!(
                    context = ?log_context,
                    error = %e,
                    "Failed to manage cache size"
                );
                e
            })?;
        debug!(
            cache_management_time_ms = cache_mgmt_start.elapsed().as_millis(),
            final_cache_size = storage.layer_cache.len(),
            "Cache management completed"
        );
        
        let total_time = start_time.elapsed();
        info!(
            context = ?log_context,
            document_id = %document_id,
            total_storage_time_ms = total_time.as_millis(),
            layers_stored = layers.len(),
            semantic_links_stored = semantic_links.len(),
            final_document_count = storage.documents.len(),
            final_cache_size = storage.layer_cache.len(),
            "Hierarchical knowledge storage completed successfully"
        );
        
        Ok(document_id)
    }
    
    /// Retrieve complete hierarchical knowledge for a document
    #[instrument(skip(self), fields(document_id = %document_id))]
    pub async fn retrieve_document(
        &self,
        document_id: &str,
    ) -> HierarchicalStorageResult<HierarchicalKnowledge> {
        let start_time = Instant::now();
        
        debug!(document_id = %document_id, "Retrieving hierarchical document");
        let mut storage = self.storage.write().await;
        
        // Log access
        storage.access_log.push(AccessLogEntry {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            document_id: document_id.to_string(),
            layer_id: None,
            operation: StorageOperation::Retrieve,
        });
        
        // Update access time in index
        if let Some(document) = storage.documents.get_mut(document_id) {
            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
            
            let mut updated_entries = 0;
            for entry in document.retrieval_index.layer_index.values_mut() {
                entry.last_accessed = timestamp;
                entry.access_count += 1;
                updated_entries += 1;
            }
            
            let retrieval_time = start_time.elapsed();
            info!(
                document_id = %document_id,
                retrieval_time_ms = retrieval_time.as_millis(),
                layers_count = document.knowledge_layers.len(),
                semantic_links_count = document.semantic_links.len(),
                index_entries_updated = updated_entries,
                "Document retrieved successfully"
            );
            
            Ok(document.clone())
        } else {
            let error_msg = format!("Document not found: {document_id}");
            error!(
                document_id = %document_id,
                retrieval_time_ms = start_time.elapsed().as_millis(),
                "Document not found in storage"
            );
            Err(HierarchicalStorageError::LayerNotFound(error_msg))
        }
    }
    
    /// Retrieve specific layer by ID
    pub async fn retrieve_layer(
        &self,
        layer_id: &str,
    ) -> HierarchicalStorageResult<KnowledgeLayer> {
        let storage = self.storage.read().await;
        
        // Check cache first
        if let Some(layer) = storage.layer_cache.get(layer_id) {
            return Ok(layer.clone());
        }
        
        // Search in documents
        for document in storage.documents.values() {
            if let Some(layer) = document.get_layer(layer_id) {
                return Ok(layer.clone());
            }
        }
        
        Err(HierarchicalStorageError::LayerNotFound(
            format!("Layer not found: {layer_id}")
        ))
    }
    
    /// Search across all documents with query
    #[instrument(
        skip(self, query),
        fields(
            max_results = query.max_results,
            has_keywords = query.keywords.as_ref().map(|k| !k.is_empty()).unwrap_or(false),
            has_entities = query.entities.as_ref().map(|e| !e.is_empty()).unwrap_or(false),
            has_concepts = query.concepts.as_ref().map(|c| !c.is_empty()).unwrap_or(false)
        )
    )]
    pub async fn search(
        &self,
        query: IndexQuery,
    ) -> HierarchicalStorageResult<Vec<SearchResult>> {
        let start_time = Instant::now();
        
        debug!(
            max_results = query.max_results,
            min_score = query.min_score,
            keywords_count = query.keywords.as_ref().map(|k| k.len()).unwrap_or(0),
            entities_count = query.entities.as_ref().map(|e| e.len()).unwrap_or(0),
            concepts_count = query.concepts.as_ref().map(|c| c.len()).unwrap_or(0),
            "Starting hierarchical search"
        );
        
        let storage = self.storage.read().await;
        let document_count = storage.documents.len();
        let mut all_results = Vec::new();
        let mut documents_searched = 0;
        
        // Search each document's index
        for document in storage.documents.values() {
            let doc_search_start = Instant::now();
            let results = self.index_manager.search_index(&document.retrieval_index, &query);
            let doc_results_count = results.len();
            all_results.extend(results);
            documents_searched += 1;
            
            debug!(
                document_search_time_ms = doc_search_start.elapsed().as_millis(),
                results_from_document = doc_results_count,
                total_results_so_far = all_results.len(),
                "Document search completed"
            );
        }
        
        let search_time = start_time.elapsed();
        let pre_sort_count = all_results.len();
        
        // Sort by score globally
        let sort_start = Instant::now();
        all_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        let sort_time = sort_start.elapsed();
        
        // Limit to requested number
        let pre_limit_count = all_results.len();
        all_results.truncate(query.max_results);
        let final_count = all_results.len();
        
        let total_time = start_time.elapsed();
        
        // Calculate average relevance score
        let avg_score = if all_results.is_empty() { 0.0 } else {
            all_results.iter().map(|r| r.score).sum::<f32>() / all_results.len() as f32
        };
        
        info!(
            total_search_time_ms = total_time.as_millis(),
            documents_searched = documents_searched,
            total_document_count = document_count,
            search_phase_time_ms = search_time.as_millis(),
            sort_time_ms = sort_time.as_millis(),
            results_before_sort = pre_sort_count,
            results_before_limit = pre_limit_count,
            final_results_count = final_count,
            requested_max = query.max_results,
            average_relevance_score = avg_score,
            "Hierarchical search completed"
        );
        
        Ok(all_results)
    }
    
    /// Find semantically similar content
    #[instrument(
        skip(self, target_embedding),
        fields(
            embedding_dim = target_embedding.len(),
            max_results = max_results
        )
    )]
    pub async fn find_similar(
        &self,
        target_embedding: &[f32],
        max_results: usize,
    ) -> HierarchicalStorageResult<Vec<(String, f32)>> {
        let start_time = Instant::now();
        
        debug!(
            embedding_dimension = target_embedding.len(),
            max_results = max_results,
            "Starting semantic similarity search"
        );
        
        let storage = self.storage.read().await;
        let document_count = storage.documents.len();
        let mut all_similarities = Vec::new();
        let mut documents_searched = 0;
        
        // Search across all documents
        for document in storage.documents.values() {
            let doc_search_start = Instant::now();
            let similarities = self.index_manager.find_similar_layers(
                &document.retrieval_index,
                target_embedding,
                max_results,
            );
            let doc_results_count = similarities.len();
            all_similarities.extend(similarities);
            documents_searched += 1;
            
            debug!(
                document_search_time_ms = doc_search_start.elapsed().as_millis(),
                similarities_found = doc_results_count,
                total_similarities_so_far = all_similarities.len(),
                "Document similarity search completed"
            );
        }
        
        let search_time = start_time.elapsed();
        let pre_sort_count = all_similarities.len();
        
        // Sort by similarity globally
        let sort_start = Instant::now();
        all_similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let sort_time = sort_start.elapsed();
        
        // Limit results
        let pre_limit_count = all_similarities.len();
        all_similarities.truncate(max_results);
        let final_count = all_similarities.len();
        
        let total_time = start_time.elapsed();
        
        // Calculate average similarity score
        let avg_similarity = if all_similarities.is_empty() { 0.0 } else {
            all_similarities.iter().map(|(_, sim)| sim).sum::<f32>() / all_similarities.len() as f32
        };
        
        info!(
            total_similarity_search_time_ms = total_time.as_millis(),
            documents_searched = documents_searched,
            total_document_count = document_count,
            search_phase_time_ms = search_time.as_millis(),
            sort_time_ms = sort_time.as_millis(),
            results_before_sort = pre_sort_count,
            results_before_limit = pre_limit_count,
            final_results_count = final_count,
            requested_max = max_results,
            average_similarity_score = avg_similarity,
            "Semantic similarity search completed"
        );
        
        Ok(all_similarities)
    }
    
    /// Get connected layers through semantic links
    pub async fn get_connected_layers(
        &self,
        layer_id: &str,
        link_types: Option<Vec<SemanticLinkType>>,
        max_hops: u32,
    ) -> HierarchicalStorageResult<Vec<(String, Vec<SemanticLinkType>)>> {
        let storage = self.storage.read().await;
        
        // Find document containing this layer
        let document = storage.documents.values()
            .find(|doc| doc.knowledge_layers.iter().any(|l| l.layer_id == layer_id))
            .ok_or_else(|| HierarchicalStorageError::LayerNotFound(
                format!("Layer not found: {layer_id}")
            ))?;
        
        // Perform breadth-first search through semantic links
        let mut visited = HashMap::new();
        let mut queue = vec![(layer_id.to_string(), Vec::new(), 0)];
        let mut results = Vec::new();
        
        while let Some((current_id, path, depth)) = queue.pop() {
            if depth >= max_hops {
                continue;
            }
            
            // Mark as visited
            visited.insert(current_id.clone(), path.clone());
            
            // Find connected nodes
            let node_id = format!("node_{current_id}");
            for edge in &document.semantic_links.edges {
                let (connected_id, _is_source) = if edge.source_node_id == node_id {
                    (edge.target_node_id.clone(), true)
                } else if edge.target_node_id == node_id {
                    (edge.source_node_id.clone(), false)
                } else {
                    continue;
                };
                
                // Check if link type matches filter
                if let Some(ref types) = link_types {
                    if !types.contains(&edge.link_type) {
                        continue;
                    }
                }
                
                // Extract layer ID from node ID
                let connected_layer_id = connected_id.strip_prefix("node_")
                    .unwrap_or(&connected_id)
                    .to_string();
                
                // Skip if already visited
                if visited.contains_key(&connected_layer_id) {
                    continue;
                }
                
                // Add to results
                let mut new_path = path.clone();
                new_path.push(edge.link_type.clone());
                
                results.push((connected_layer_id.clone(), new_path.clone()));
                
                // Add to queue for further exploration
                queue.push((connected_layer_id, new_path, depth + 1));
            }
        }
        
        Ok(results)
    }
    
    /// Get storage statistics
    pub async fn get_storage_stats(&self) -> HierarchicalStorageStats {
        let storage = self.storage.read().await;
        
        let mut total_layers = 0;
        let mut total_nodes = 0;
        let mut total_edges = 0;
        let mut layer_type_distribution = HashMap::new();
        let mut link_type_distribution = HashMap::new();
        let mut depth_sum = 0;
        
        for document in storage.documents.values() {
            total_layers += document.knowledge_layers.len();
            total_nodes += document.semantic_links.nodes.len();
            total_edges += document.semantic_links.edges.len();
            depth_sum += document.max_depth() as usize;
            
            // Count layer types
            for layer in &document.knowledge_layers {
                *layer_type_distribution.entry(layer.layer_type.clone()).or_insert(0) += 1;
            }
            
            // Count link types
            for edge in &document.semantic_links.edges {
                *link_type_distribution.entry(edge.link_type.clone()).or_insert(0) += 1;
            }
        }
        
        let average_depth = if storage.documents.is_empty() {
            0.0
        } else {
            depth_sum as f32 / storage.documents.len() as f32
        };
        
        let cache_hit_rate = self.calculate_cache_hit_rate(&storage.access_log);
        
        HierarchicalStorageStats {
            total_documents: storage.documents.len(),
            total_layers,
            total_nodes,
            total_edges,
            average_depth,
            layer_type_distribution,
            link_type_distribution,
            storage_efficiency: self.calculate_storage_efficiency(&storage),
            index_coverage: self.calculate_index_coverage(&storage),
            cache_hit_rate,
        }
    }
    
    /// Update an existing document with new knowledge
    pub async fn update_document(
        &self,
        document_id: &str,
        new_layers: Vec<KnowledgeLayer>,
    ) -> HierarchicalStorageResult<()> {
        let mut storage = self.storage.write().await;
        
        let document = storage.documents.get_mut(document_id)
            .ok_or_else(|| HierarchicalStorageError::LayerNotFound(
                format!("Document not found: {document_id}")
            ))?;
        
        // Add new layers
        document.knowledge_layers.extend(new_layers.clone());
        
        // Update timestamp
        document.last_updated = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        // Rebuild semantic links
        let all_layers = document.knowledge_layers.clone();
        document.semantic_links = self.link_manager
            .build_semantic_link_graph(&all_layers)
            .await?;
        
        // Update index incrementally
        self.index_manager.update_index_incremental(
            &mut document.retrieval_index,
            &new_layers,
        ).await?;
        
        // Update cache
        for layer in new_layers {
            storage.layer_cache.insert(layer.layer_id.clone(), layer);
        }
        
        // Log operation
        storage.access_log.push(AccessLogEntry {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            document_id: document_id.to_string(),
            layer_id: None,
            operation: StorageOperation::Update,
        });
        
        Ok(())
    }
    
    /// Delete a document from storage
    pub async fn delete_document(
        &self,
        document_id: &str,
    ) -> HierarchicalStorageResult<()> {
        let mut storage = self.storage.write().await;
        
        // Remove from documents
        let document = storage.documents.remove(document_id)
            .ok_or_else(|| HierarchicalStorageError::LayerNotFound(
                format!("Document not found: {document_id}")
            ))?;
        
        // Remove layers from cache
        for layer in document.knowledge_layers {
            storage.layer_cache.remove(&layer.layer_id);
        }
        
        // Log operation
        storage.access_log.push(AccessLogEntry {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            document_id: document_id.to_string(),
            layer_id: None,
            operation: StorageOperation::Delete,
        });
        
        Ok(())
    }
    
    // Helper methods
    
    /// Manage cache size by evicting least recently used items
    async fn manage_cache_size(
        &self,
        storage: &mut InMemoryStorage,
    ) -> HierarchicalStorageResult<()> {
        if storage.layer_cache.len() <= self.config.cache_size_limit {
            return Ok(());
        }
        
        // Collect cache entries with access times
        let mut cache_entries: Vec<(String, u64)> = Vec::new();
        
        for layer_id in storage.layer_cache.keys() {
            // Find access time from any document's index
            let mut last_accessed = 0;
            for document in storage.documents.values() {
                if let Some(entry) = document.retrieval_index.layer_index.get(layer_id) {
                    last_accessed = entry.last_accessed;
                    break;
                }
            }
            cache_entries.push((layer_id.clone(), last_accessed));
        }
        
        // Sort by access time (oldest first)
        cache_entries.sort_by_key(|entry| entry.1);
        
        // Remove oldest entries
        let num_to_remove = storage.layer_cache.len() - self.config.cache_size_limit;
        for (layer_id, _) in cache_entries.into_iter().take(num_to_remove) {
            storage.layer_cache.remove(&layer_id);
        }
        
        Ok(())
    }
    
    /// Calculate cache hit rate from access log
    fn calculate_cache_hit_rate(&self, access_log: &[AccessLogEntry]) -> f32 {
        if access_log.is_empty() {
            return 0.0;
        }
        
        // Simple approximation - in real system would track actual cache hits
        let recent_accesses = access_log.len().min(100);
        let unique_items = access_log[access_log.len() - recent_accesses..]
            .iter()
            .filter_map(|entry| entry.layer_id.as_ref())
            .collect::<std::collections::HashSet<_>>()
            .len();
        
        if unique_items == 0 {
            return 0.0;
        }
        
        // Assume cache hit if accessing fewer unique items than cache size
        let hit_ratio = 1.0 - (unique_items as f32 / self.config.cache_size_limit as f32);
        hit_ratio.max(0.0).min(1.0)
    }
    
    /// Calculate storage efficiency
    fn calculate_storage_efficiency(&self, storage: &InMemoryStorage) -> f32 {
        if storage.documents.is_empty() {
            return 0.0;
        }
        
        let mut total_content_size = 0;
        let mut total_overhead_size = 0;
        
        for document in storage.documents.values() {
            for layer in &document.knowledge_layers {
                total_content_size += layer.content.raw_text.len();
                total_overhead_size += layer.layer_id.len() + 
                    layer.content.key_phrases.iter().map(|p| p.len()).sum::<usize>();
            }
        }
        
        if total_content_size + total_overhead_size == 0 {
            return 0.0;
        }
        
        total_content_size as f32 / (total_content_size + total_overhead_size) as f32
    }
    
    /// Calculate index coverage
    fn calculate_index_coverage(&self, storage: &InMemoryStorage) -> f32 {
        if storage.documents.is_empty() {
            return 0.0;
        }
        
        let mut indexed_layers = 0;
        let mut total_layers = 0;
        
        for document in storage.documents.values() {
            total_layers += document.knowledge_layers.len();
            indexed_layers += document.retrieval_index.layer_index.len();
        }
        
        if total_layers == 0 {
            return 0.0;
        }
        
        indexed_layers as f32 / total_layers as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enhanced_knowledge_storage::model_management::ModelResourceManager;
    use crate::enhanced_knowledge_storage::types::{ModelResourceConfig, ComplexityLevel};
    use crate::enhanced_knowledge_storage::knowledge_processing::types::{DocumentStructure, QualityMetrics};
    use crate::enhanced_knowledge_storage::hierarchical_storage::types::ComplexityIndicators;
    
    #[tokio::test]
    async fn test_storage_engine_creation() {
        let model_config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(model_config).await.unwrap());
        let storage_config = HierarchicalStorageConfig::default();
        
        let engine = HierarchicalStorageEngine::new(model_manager, storage_config);
        
        let stats = engine.get_storage_stats().await;
        assert_eq!(stats.total_documents, 0);
        assert_eq!(stats.total_layers, 0);
    }
    
    #[tokio::test]
    async fn test_document_storage_and_retrieval() {
        let model_config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(model_config).await.unwrap());
        let storage_config = HierarchicalStorageConfig::default();
        
        let engine = HierarchicalStorageEngine::new(model_manager, storage_config);
        
        // Create test processing result
        let processing_result = KnowledgeProcessingResult {
            document_id: "test_doc".to_string(),
            chunks: vec![],
            global_entities: vec![],
            global_relationships: vec![],
            document_structure: DocumentStructure {
                sections: vec![],
                overall_topic: Some("Test Document".to_string()),
                key_themes: vec!["Testing".to_string()],
                complexity_level: ComplexityLevel::Low,
                estimated_reading_time: std::time::Duration::from_secs(60),
            },
            processing_metadata: crate::enhanced_knowledge_storage::knowledge_processing::types::ProcessingMetadata {
                processing_time: std::time::Duration::from_millis(100),
                models_used: vec!["test_model".to_string()],
                total_tokens_processed: 100,
                chunks_created: 0,
                entities_extracted: 0,
                relationships_extracted: 0,
                memory_usage_peak: 1000000,
            },
            quality_metrics: QualityMetrics {
                entity_extraction_quality: 0.8,
                relationship_extraction_quality: 0.7,
                semantic_coherence: 0.9,
                context_preservation: 0.8,
                overall_quality: 0.8,
            },
        };
        
        let global_context = GlobalContext {
            document_theme: "Test Document".to_string(),
            key_entities: vec![],
            main_relationships: vec![],
            conceptual_framework: vec![],
            context_preservation_score: 0.8,
            domain_classification: vec!["Testing".to_string()],
            complexity_indicators: ComplexityIndicators {
                vocabulary_complexity: 0.5,
                syntactic_complexity: 0.5,
                conceptual_density: 0.5,
                relationship_complexity: 0.5,
                overall_complexity: ComplexityLevel::Low,
            },
        };
        
        // Store document
        let doc_id = engine.store_knowledge(processing_result, global_context).await.unwrap();
        assert_eq!(doc_id, "test_doc");
        
        // Retrieve document
        let retrieved = engine.retrieve_document(&doc_id).await.unwrap();
        assert_eq!(retrieved.document_id, "test_doc");
        
        // Check stats
        let stats = engine.get_storage_stats().await;
        assert_eq!(stats.total_documents, 1);
    }
    
    #[tokio::test]
    async fn test_search_functionality() {
        let model_config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(model_config).await.unwrap());
        let storage_config = HierarchicalStorageConfig::default();
        
        let engine = HierarchicalStorageEngine::new(model_manager, storage_config);
        
        // Create empty query
        let query = IndexQuery {
            keywords: Some(vec!["test".to_string()]),
            ..Default::default()
        };
        
        let results = engine.search(query).await.unwrap();
        assert_eq!(results.len(), 0); // No documents stored yet
    }
}