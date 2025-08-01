//! Hierarchical Storage Engine
//! 
//! Main storage engine that coordinates all hierarchical storage components
//! and provides the primary interface for storing and retrieving knowledge.

use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::RwLock;
use crate::enhanced_knowledge_storage::{
    types::{ModelResourceConfig},
    model_management::ModelResourceManager,
    knowledge_processing::types::{KnowledgeProcessingResult, DocumentStructure, ComplexityLevel, DocumentSection, SectionType, ProcessingMetadata, QualityMetrics, ComplexityIndicators},
    hierarchical_storage::{
        types::{HierarchicalStorageConfig, HierarchicalKnowledge, KnowledgeLayer, GlobalContext, HierarchicalStorageResult, HierarchicalStorageError, HierarchicalStorageStats, SemanticLinkType},
        KnowledgeLayerManager,
        SemanticLinkManager,
        HierarchicalIndexManager,
        IndexQuery,
        SearchResult,
    },
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
    pub async fn store_knowledge(
        &self,
        processing_result: KnowledgeProcessingResult,
        global_context: GlobalContext,
    ) -> HierarchicalStorageResult<String> {
        let document_id = processing_result.document_id.clone();
        
        // Step 1: Create hierarchical layers
        let layers = self.layer_manager
            .create_hierarchical_layers(&processing_result)
            .await?;
        
        // Validate layer count
        if layers.len() > self.config.max_layers_per_document {
            return Err(HierarchicalStorageError::InvalidLayerStructure(
                format!("Too many layers: {} > {}", layers.len(), self.config.max_layers_per_document)
            ));
        }
        
        // Step 2: Build semantic link graph
        let semantic_links = self.link_manager
            .build_semantic_link_graph(&layers)
            .await?;
        
        // Step 3: Build hierarchical index
        let retrieval_index = self.index_manager
            .build_hierarchical_index(&layers)
            .await?;
        
        // Step 4: Create hierarchical knowledge structure
        let mut hierarchical_knowledge = HierarchicalKnowledge::new(
            document_id.clone(),
            global_context,
        );
        
        hierarchical_knowledge.knowledge_layers = layers.clone();
        hierarchical_knowledge.semantic_links = semantic_links;
        hierarchical_knowledge.retrieval_index = retrieval_index;
        
        // Step 5: Store in memory
        let mut storage = self.storage.write().await;
        
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
        
        // Manage cache size
        self.manage_cache_size(&mut storage).await?;
        
        Ok(document_id)
    }
    
    /// Retrieve complete hierarchical knowledge for a document
    pub async fn retrieve_document(
        &self,
        document_id: &str,
    ) -> HierarchicalStorageResult<HierarchicalKnowledge> {
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
            
            for entry in document.retrieval_index.layer_index.values_mut() {
                entry.last_accessed = timestamp;
                entry.access_count += 1;
            }
            
            Ok(document.clone())
        } else {
            Err(HierarchicalStorageError::LayerNotFound(
                format!("Document not found: {}", document_id)
            ))
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
            format!("Layer not found: {}", layer_id)
        ))
    }
    
    /// Search across all documents with query
    pub async fn search(
        &self,
        query: IndexQuery,
    ) -> HierarchicalStorageResult<Vec<SearchResult>> {
        let storage = self.storage.read().await;
        let mut all_results = Vec::new();
        
        // Search each document's index
        for (_doc_id, document) in &storage.documents {
            let results = self.index_manager.search_index(&document.retrieval_index, &query);
            all_results.extend(results);
        }
        
        // Sort by score globally
        all_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        
        // Limit to requested number
        all_results.truncate(query.max_results);
        
        Ok(all_results)
    }
    
    /// Find semantically similar content
    pub async fn find_similar(
        &self,
        target_embedding: &[f32],
        max_results: usize,
    ) -> HierarchicalStorageResult<Vec<(String, f32)>> {
        let storage = self.storage.read().await;
        let mut all_similarities = Vec::new();
        
        // Search across all documents
        for document in storage.documents.values() {
            let similarities = self.index_manager.find_similar_layers(
                &document.retrieval_index,
                target_embedding,
                max_results,
            );
            all_similarities.extend(similarities);
        }
        
        // Sort by similarity globally
        all_similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Limit results
        all_similarities.truncate(max_results);
        
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
                format!("Layer not found: {}", layer_id)
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
            let node_id = format!("node_{}", current_id);
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
                format!("Document not found: {}", document_id)
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
                format!("Document not found: {}", document_id)
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
        
        for (layer_id, _) in &storage.layer_cache {
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
    
    #[tokio::test]
    async fn test_storage_engine_creation() {
        let model_config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(model_config));
        let storage_config = HierarchicalStorageConfig::default();
        
        let engine = HierarchicalStorageEngine::new(model_manager, storage_config);
        
        let stats = engine.get_storage_stats().await;
        assert_eq!(stats.total_documents, 0);
        assert_eq!(stats.total_layers, 0);
    }
    
    #[tokio::test]
    async fn test_document_storage_and_retrieval() {
        let model_config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(model_config));
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
        let model_manager = Arc::new(ModelResourceManager::new(model_config));
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