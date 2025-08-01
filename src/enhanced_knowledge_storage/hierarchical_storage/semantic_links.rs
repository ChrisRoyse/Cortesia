//! Semantic Links Management
//! 
//! Creates and manages semantic links between knowledge layers to preserve
//! context and enable intelligent traversal of the knowledge graph.

use std::sync::Arc;
use std::collections::{HashMap, HashSet};
use crate::enhanced_knowledge_storage::{
    model_management::ModelResourceManager,
    hierarchical_storage::types::{KnowledgeLayer, SemanticLinkGraph, SemanticLinkType, SemanticNode, SemanticEdge, HierarchicalStorageConfig, HierarchicalStorageResult, HierarchicalStorageError, LayerType, LinkTypeInfo, SemanticNodeType, CentralityScores},
};

/// Manager for creating and analyzing semantic links
pub struct SemanticLinkManager {
    model_manager: Arc<ModelResourceManager>,
    config: HierarchicalStorageConfig,
}

impl SemanticLinkManager {
    /// Create new semantic link manager
    pub fn new(
        model_manager: Arc<ModelResourceManager>,
        config: HierarchicalStorageConfig,
    ) -> Self {
        Self {
            model_manager,
            config,
        }
    }
    
    /// Build complete semantic link graph from knowledge layers
    pub async fn build_semantic_link_graph(
        &self,
        layers: &[KnowledgeLayer],
    ) -> HierarchicalStorageResult<SemanticLinkGraph> {
        let mut graph = SemanticLinkGraph {
            nodes: HashMap::new(),
            edges: Vec::new(),
            link_types: self.initialize_link_types(),
        };
        
        // Step 1: Create semantic nodes for each layer
        self.create_semantic_nodes(&mut graph, layers).await?;
        
        // Step 2: Create hierarchical links (parent-child relationships)
        self.create_hierarchical_links(&mut graph, layers)?;
        
        // Step 3: Create sequential links (document flow)
        self.create_sequential_links(&mut graph, layers)?;
        
        // Step 4: Create referential links (entity mentions)
        self.create_referential_links(&mut graph, layers).await?;
        
        // Step 5: Create semantic similarity links
        self.create_semantic_similarity_links(&mut graph, layers).await?;
        
        // Step 6: Create causal and temporal links
        self.create_causal_temporal_links(&mut graph, layers).await?;
        
        // Step 7: Create categorical links
        self.create_categorical_links(&mut graph, layers)?;
        
        // Step 8: Calculate centrality scores
        self.calculate_centrality_scores(&mut graph)?;
        
        // Step 9: Prune weak links and optimize graph
        self.optimize_graph(&mut graph)?;
        
        Ok(graph)
    }
    
    /// Initialize standard link types with their properties
    fn initialize_link_types(&self) -> HashMap<String, LinkTypeInfo> {
        let mut link_types = HashMap::new();
        
        let types = vec![
            (SemanticLinkType::Hierarchical, "Parent-child structural relationship", true, (0.8, 1.0), "#FF6B6B"),
            (SemanticLinkType::Sequential, "Sequential document flow", true, (0.7, 0.9), "#4ECDC4"),
            (SemanticLinkType::Referential, "Cross-references and mentions", false, (0.3, 0.7), "#45B7D1"),
            (SemanticLinkType::Semantic, "Semantic similarity", false, (0.4, 0.8), "#96CEB4"),
            (SemanticLinkType::Causal, "Cause-effect relationships", true, (0.6, 0.9), "#FFEAA7"),
            (SemanticLinkType::Temporal, "Time-based relationships", true, (0.5, 0.8), "#DDA0DD"),
            (SemanticLinkType::Categorical, "Category membership", false, (0.5, 0.8), "#98D8C8"),
            (SemanticLinkType::Comparative, "Comparison relationships", false, (0.4, 0.7), "#F7DC6F"),
            (SemanticLinkType::Definitional, "Definition relationships", true, (0.7, 0.9), "#BB8FCE"),
            (SemanticLinkType::Explanatory, "Explanation relationships", true, (0.6, 0.8), "#85C1E9"),
        ];
        
        for (link_type, description, is_directional, strength_range, color) in types {
            link_types.insert(
                format!("{link_type:?}"),
                LinkTypeInfo {
                    link_type,
                    description: description.to_string(),
                    is_directional,
                    typical_strength_range: strength_range,
                    color_code: color.to_string(),
                }
            );
        }
        
        link_types
    }
    
    /// Create semantic nodes for each knowledge layer
    async fn create_semantic_nodes(
        &self,
        graph: &mut SemanticLinkGraph,
        layers: &[KnowledgeLayer],
    ) -> HierarchicalStorageResult<()> {
        for layer in layers {
            let node_type = match layer.layer_type {
                LayerType::Entity => SemanticNodeType::EntityNode,
                LayerType::Relationship => SemanticNodeType::RelationshipNode,
                LayerType::Concept => SemanticNodeType::ConceptNode,
                _ => SemanticNodeType::LayerNode,
            };
            
            let node = SemanticNode {
                node_id: format!("node_{}", layer.layer_id),
                layer_id: layer.layer_id.clone(),
                node_type,
                importance_weight: layer.importance_score,
                centrality_scores: CentralityScores {
                    degree_centrality: 0.0,
                    betweenness_centrality: 0.0,
                    closeness_centrality: 0.0,
                    eigenvector_centrality: 0.0,
                },
                connected_nodes: Vec::new(),
            };
            
            graph.nodes.insert(node.node_id.clone(), node);
        }
        
        Ok(())
    }
    
    /// Create hierarchical links between parent and child layers
    fn create_hierarchical_links(
        &self,
        graph: &mut SemanticLinkGraph,
        layers: &[KnowledgeLayer],
    ) -> HierarchicalStorageResult<()> {
        for layer in layers {
            if let Some(parent_id) = &layer.parent_layer_id {
                let source_node_id = format!("node_{parent_id}");
                let target_node_id = format!("node_{}", layer.layer_id);
                
                if graph.nodes.contains_key(&source_node_id) && graph.nodes.contains_key(&target_node_id) {
                    let edge = SemanticEdge {
                        edge_id: format!("hierarchical_{}_{}", parent_id, layer.layer_id),
                        source_node_id: source_node_id.clone(),
                        target_node_id: target_node_id.clone(),
                        link_type: SemanticLinkType::Hierarchical,
                        weight: 0.9, // High weight for structural relationships
                        confidence: 1.0, // Perfect confidence for explicit hierarchy
                        supporting_evidence: vec!["Parent-child relationship in document structure".to_string()],
                        created_at: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
                    };
                    
                    // Update connected nodes
                    if let Some(source_node) = graph.nodes.get_mut(&source_node_id) {
                        source_node.connected_nodes.push(target_node_id.clone());
                    }
                    if let Some(target_node) = graph.nodes.get_mut(&target_node_id) {
                        target_node.connected_nodes.push(source_node_id);
                    }
                    
                    graph.edges.push(edge);
                }
            }
        }
        
        Ok(())
    }
    
    /// Create sequential links based on document flow
    fn create_sequential_links(
        &self,
        graph: &mut SemanticLinkGraph,
        layers: &[KnowledgeLayer],
    ) -> HierarchicalStorageResult<()> {
        // Group layers by type and parent for sequential linking
        let mut layer_groups: HashMap<(LayerType, Option<String>), Vec<&KnowledgeLayer>> = HashMap::new();
        
        for layer in layers {
            let key = (layer.layer_type.clone(), layer.parent_layer_id.clone());
            layer_groups.entry(key).or_default().push(layer);
        }
        
        // Create sequential links within each group
        for ((layer_type, _), mut group_layers) in layer_groups {
            if group_layers.len() < 2 {
                continue;
            }
            
            // Sort by sequence number
            group_layers.sort_by_key(|layer| layer.position.sequence_number);
            
            for i in 0..group_layers.len() - 1 {
                let current_layer = group_layers[i];
                let next_layer = group_layers[i + 1];
                
                let source_node_id = format!("node_{}", current_layer.layer_id);
                let target_node_id = format!("node_{}", next_layer.layer_id);
                
                if graph.nodes.contains_key(&source_node_id) && graph.nodes.contains_key(&target_node_id) {
                    let weight = self.calculate_sequential_weight(current_layer, next_layer);
                    
                    let edge = SemanticEdge {
                        edge_id: format!("sequential_{}_{}", current_layer.layer_id, next_layer.layer_id),
                        source_node_id: source_node_id.clone(),
                        target_node_id: target_node_id.clone(),
                        link_type: SemanticLinkType::Sequential,
                        weight,
                        confidence: 0.8,
                        supporting_evidence: vec![format!("Sequential {:?} order", layer_type)],
                        created_at: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
                    };
                    
                    // Update connected nodes
                    if let Some(source_node) = graph.nodes.get_mut(&source_node_id) {
                        source_node.connected_nodes.push(target_node_id.clone());
                    }
                    if let Some(target_node) = graph.nodes.get_mut(&target_node_id) {
                        target_node.connected_nodes.push(source_node_id);
                    }
                    
                    graph.edges.push(edge);
                }
            }
        }
        
        Ok(())
    }
    
    /// Create referential links based on entity mentions
    async fn create_referential_links(
        &self,
        graph: &mut SemanticLinkGraph,
        layers: &[KnowledgeLayer],
    ) -> HierarchicalStorageResult<()> {
        // Create entity index
        let mut entity_layers: HashMap<String, Vec<&KnowledgeLayer>> = HashMap::new();
        
        for layer in layers {
            for entity in &layer.entities {
                entity_layers.entry(entity.name.clone()).or_default().push(layer);
            }
        }
        
        // Create referential links for shared entities
        for (entity_name, entity_layer_list) in entity_layers {
            if entity_layer_list.len() < 2 {
                continue;
            }
            
            // Create links between all pairs of layers that mention this entity
            for i in 0..entity_layer_list.len() {
                for j in i + 1..entity_layer_list.len() {
                    let layer1 = entity_layer_list[i];
                    let layer2 = entity_layer_list[j];
                    
                    let source_node_id = format!("node_{}", layer1.layer_id);
                    let target_node_id = format!("node_{}", layer2.layer_id);
                    
                    if graph.nodes.contains_key(&source_node_id) && graph.nodes.contains_key(&target_node_id) {
                        let weight = self.calculate_referential_weight(layer1, layer2, &entity_name);
                        
                        let edge = SemanticEdge {
                            edge_id: format!("referential_{}_{}_{}", layer1.layer_id, layer2.layer_id, entity_name.replace(" ", "_")),
                            source_node_id: source_node_id.clone(),
                            target_node_id: target_node_id.clone(),
                            link_type: SemanticLinkType::Referential,
                            weight,
                            confidence: self.calculate_entity_confidence(layer1, layer2, &entity_name),
                            supporting_evidence: vec![format!("Shared entity: {}", entity_name)],
                            created_at: std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_secs(),
                        };
                        
                        // Update connected nodes
                        if let Some(source_node) = graph.nodes.get_mut(&source_node_id) {
                            if !source_node.connected_nodes.contains(&target_node_id) {
                                source_node.connected_nodes.push(target_node_id.clone());
                            }
                        }
                        if let Some(target_node) = graph.nodes.get_mut(&target_node_id) {
                            if !target_node.connected_nodes.contains(&source_node_id) {
                                target_node.connected_nodes.push(source_node_id);
                            }
                        }
                        
                        graph.edges.push(edge);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Create semantic similarity links using embeddings
    async fn create_semantic_similarity_links(
        &self,
        graph: &mut SemanticLinkGraph,
        layers: &[KnowledgeLayer],
    ) -> HierarchicalStorageResult<()> {
        // Only create semantic links for layers with embeddings
        let layers_with_embeddings: Vec<&KnowledgeLayer> = layers
            .iter()
            .filter(|layer| layer.semantic_embedding.is_some())
            .collect();
        
        for i in 0..layers_with_embeddings.len() {
            for j in i + 1..layers_with_embeddings.len() {
                let layer1 = layers_with_embeddings[i];
                let layer2 = layers_with_embeddings[j];
                
                // Skip if layers are already connected via hierarchy
                if self.are_hierarchically_related(layer1, layer2) {
                    continue;
                }
                
                let similarity = self.calculate_embedding_similarity(
                    layer1.semantic_embedding.as_ref().unwrap(),
                    layer2.semantic_embedding.as_ref().unwrap(),
                )?;
                
                if similarity >= self.config.semantic_similarity_threshold {
                    let source_node_id = format!("node_{}", layer1.layer_id);
                    let target_node_id = format!("node_{}", layer2.layer_id);
                    
                    if graph.nodes.contains_key(&source_node_id) && graph.nodes.contains_key(&target_node_id) {
                        let edge = SemanticEdge {
                            edge_id: format!("semantic_{}_{}", layer1.layer_id, layer2.layer_id),
                            source_node_id: source_node_id.clone(),
                            target_node_id: target_node_id.clone(),
                            link_type: SemanticLinkType::Semantic,
                            weight: similarity,
                            confidence: similarity,
                            supporting_evidence: vec![format!("Semantic similarity: {:.3}", similarity)],
                            created_at: std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_secs(),
                        };
                        
                        // Update connected nodes
                        if let Some(source_node) = graph.nodes.get_mut(&source_node_id) {
                            if !source_node.connected_nodes.contains(&target_node_id) {
                                source_node.connected_nodes.push(target_node_id.clone());
                            }
                        }
                        if let Some(target_node) = graph.nodes.get_mut(&target_node_id) {
                            if !target_node.connected_nodes.contains(&source_node_id) {
                                target_node.connected_nodes.push(source_node_id);
                            }
                        }
                        
                        graph.edges.push(edge);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Create causal and temporal links using AI analysis
    async fn create_causal_temporal_links(
        &self,
        graph: &mut SemanticLinkGraph,
        layers: &[KnowledgeLayer],
    ) -> HierarchicalStorageResult<()> {
        // Focus on layers with relationships that have temporal information
        let relevant_layers: Vec<&KnowledgeLayer> = layers
            .iter()
            .filter(|layer| {
                !layer.relationships.is_empty() && 
                layer.relationships.iter().any(|rel| rel.temporal_info.is_some())
            })
            .collect();
        
        for i in 0..relevant_layers.len() {
            for j in i + 1..relevant_layers.len() {
                let layer1 = relevant_layers[i];
                let layer2 = relevant_layers[j];
                
                if let Some((link_type, weight, confidence, evidence)) = 
                    self.analyze_causal_temporal_relationship(layer1, layer2).await? {
                    
                    let source_node_id = format!("node_{}", layer1.layer_id);
                    let target_node_id = format!("node_{}", layer2.layer_id);
                    
                    if graph.nodes.contains_key(&source_node_id) && graph.nodes.contains_key(&target_node_id) {
                        let edge = SemanticEdge {
                            edge_id: format!("{}_{}_{}_{}", 
                                format!("{link_type:?}").to_lowercase(),
                                layer1.layer_id, 
                                layer2.layer_id,
                                std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap()
                                    .as_secs()
                            ),
                            source_node_id: source_node_id.clone(),
                            target_node_id: target_node_id.clone(),
                            link_type,
                            weight,
                            confidence,
                            supporting_evidence: evidence,
                            created_at: std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_secs(),
                        };
                        
                        // Update connected nodes
                        if let Some(source_node) = graph.nodes.get_mut(&source_node_id) {
                            if !source_node.connected_nodes.contains(&target_node_id) {
                                source_node.connected_nodes.push(target_node_id.clone());
                            }
                        }
                        if let Some(target_node) = graph.nodes.get_mut(&target_node_id) {
                            if !target_node.connected_nodes.contains(&source_node_id) {
                                target_node.connected_nodes.push(source_node_id);
                            }
                        }
                        
                        graph.edges.push(edge);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Create categorical links for similar layer types and concepts
    fn create_categorical_links(
        &self,
        graph: &mut SemanticLinkGraph,
        layers: &[KnowledgeLayer],
    ) -> HierarchicalStorageResult<()> {
        // Group layers by type
        let mut type_groups: HashMap<LayerType, Vec<&KnowledgeLayer>> = HashMap::new();
        
        for layer in layers {
            type_groups.entry(layer.layer_type.clone()).or_default().push(layer);
        }
        
        // Create categorical links within each type group
        for (layer_type, group_layers) in type_groups {
            if group_layers.len() < 2 || layer_type == LayerType::Document {
                continue; // Skip single layers and document level
            }
            
            // Create links between layers of the same type with similar tags
            for i in 0..group_layers.len() {
                for j in i + 1..group_layers.len() {
                    let layer1 = group_layers[i];
                    let layer2 = group_layers[j];
                    
                    let similarity = self.calculate_categorical_similarity(layer1, layer2);
                    
                    if similarity > 0.5 { // Threshold for categorical similarity
                        let source_node_id = format!("node_{}", layer1.layer_id);
                        let target_node_id = format!("node_{}", layer2.layer_id);
                        
                        if graph.nodes.contains_key(&source_node_id) && graph.nodes.contains_key(&target_node_id) {
                            let edge = SemanticEdge {
                                edge_id: format!("categorical_{}_{}", layer1.layer_id, layer2.layer_id),
                                source_node_id: source_node_id.clone(),
                                target_node_id: target_node_id.clone(),
                                link_type: SemanticLinkType::Categorical,
                                weight: similarity,
                                confidence: similarity,
                                supporting_evidence: vec![format!("Same type: {:?}", layer_type)],
                                created_at: std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap()
                                    .as_secs(),
                            };
                            
                            // Update connected nodes
                            if let Some(source_node) = graph.nodes.get_mut(&source_node_id) {
                                if !source_node.connected_nodes.contains(&target_node_id) {
                                    source_node.connected_nodes.push(target_node_id.clone());
                                }
                            }
                            if let Some(target_node) = graph.nodes.get_mut(&target_node_id) {
                                if !target_node.connected_nodes.contains(&source_node_id) {
                                    target_node.connected_nodes.push(source_node_id);
                                }
                            }
                            
                            graph.edges.push(edge);
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Calculate centrality scores for all nodes
    fn calculate_centrality_scores(
        &self,
        graph: &mut SemanticLinkGraph,
    ) -> HierarchicalStorageResult<()> {
        let node_count = graph.nodes.len() as f32;
        
        if node_count == 0.0 {
            return Ok(());
        }
        
        // Calculate degree centrality
        for (_node_id, node) in graph.nodes.iter_mut() {
            let degree = node.connected_nodes.len() as f32;
            node.centrality_scores.degree_centrality = degree / (node_count - 1.0);
        }
        
        // Simple betweenness centrality approximation
        // Collect node IDs first to avoid borrow checker issues
        let node_ids: Vec<String> = graph.nodes.keys().cloned().collect();
        
        for node_id in node_ids {
            let mut betweenness = 0.0;
            
            // Get connected nodes for this node
            let connected_nodes: Vec<String> = if let Some(node) = graph.nodes.get(&node_id) {
                node.connected_nodes.clone()
            } else {
                continue;
            };
            
            // For each connected node, check if this node is a bridge
            for connected_id in &connected_nodes {
                if let Some(connected_node) = graph.nodes.get(connected_id) {
                    let node = graph.nodes.get(&node_id).unwrap();
                    let mutual_connections = node.connected_nodes
                        .iter()
                        .filter(|&id| connected_node.connected_nodes.contains(id))
                        .count() as f32;
                    
                    if mutual_connections < connected_node.connected_nodes.len() as f32 {
                        betweenness += 1.0;
                    }
                }
            }
            
            // Update the betweenness centrality
            if let Some(node) = graph.nodes.get_mut(&node_id) {
                node.centrality_scores.betweenness_centrality = betweenness / (node_count * (node_count - 1.0) / 2.0);
            }
        }
        
        // Simple closeness centrality (inverse of average distance)
        for (_node_id, node) in graph.nodes.iter_mut() {
            if node.connected_nodes.is_empty() {
                node.centrality_scores.closeness_centrality = 0.0;
            } else {
                // Approximate as inverse of degree (full calculation would require BFS)
                node.centrality_scores.closeness_centrality = node.connected_nodes.len() as f32 / node_count;
            }
        }
        
        // Simple eigenvector centrality approximation
        // Collect node IDs again to avoid borrow checker issues
        let node_ids: Vec<String> = graph.nodes.keys().cloned().collect();
        
        for node_id in node_ids {
            let mut eigenvector_score = 0.0;
            
            // Get connected nodes for this node
            let connected_nodes: Vec<String> = if let Some(node) = graph.nodes.get(&node_id) {
                node.connected_nodes.clone()
            } else {
                continue;
            };
            
            for connected_id in &connected_nodes {
                if let Some(connected_node) = graph.nodes.get(connected_id) {
                    eigenvector_score += connected_node.importance_weight;
                }
            }
            
            // Update the eigenvector centrality
            if let Some(node) = graph.nodes.get_mut(&node_id) {
                node.centrality_scores.eigenvector_centrality = eigenvector_score / node_count;
            }
        }
        
        Ok(())
    }
    
    /// Optimize graph by pruning weak links and consolidating similar edges
    fn optimize_graph(
        &self,
        graph: &mut SemanticLinkGraph,
    ) -> HierarchicalStorageResult<()> {
        // Remove edges with very low weight or confidence
        let min_weight = 0.3;
        let min_confidence = 0.4;
        
        graph.edges.retain(|edge| {
            edge.weight >= min_weight && edge.confidence >= min_confidence
        });
        
        // Update connected_nodes lists based on remaining edges
        for node in graph.nodes.values_mut() {
            node.connected_nodes.clear();
        }
        
        for edge in &graph.edges {
            if let Some(source_node) = graph.nodes.get_mut(&edge.source_node_id) {
                if !source_node.connected_nodes.contains(&edge.target_node_id) {
                    source_node.connected_nodes.push(edge.target_node_id.clone());
                }
            }
            if let Some(target_node) = graph.nodes.get_mut(&edge.target_node_id) {
                if !target_node.connected_nodes.contains(&edge.source_node_id) {
                    target_node.connected_nodes.push(edge.source_node_id.clone());
                }
            }
        }
        
        Ok(())
    }
    
    // Helper methods
    
    /// Calculate weight for sequential links
    fn calculate_sequential_weight(&self, layer1: &KnowledgeLayer, layer2: &KnowledgeLayer) -> f32 {
        let base_weight = 0.7;
        
        // Increase weight if layers are consecutive
        let sequence_bonus = if layer2.position.sequence_number == layer1.position.sequence_number + 1 {
            0.2
        } else {
            0.0
        };
        
        // Adjust based on coherence scores
        let coherence_factor = (layer1.coherence_score + layer2.coherence_score) / 2.0 * 0.1;
        
        (base_weight + sequence_bonus + coherence_factor).min(1.0)
    }
    
    /// Calculate weight for referential links
    fn calculate_referential_weight(&self, layer1: &KnowledgeLayer, layer2: &KnowledgeLayer, entity_name: &str) -> f32 {
        let base_weight = 0.5;
        
        // Find the entity in both layers and use confidence
        let entity1_conf = layer1.entities
            .iter()
            .find(|e| e.name == entity_name)
            .map(|e| e.confidence)
            .unwrap_or(0.5);
        
        let entity2_conf = layer2.entities
            .iter()
            .find(|e| e.name == entity_name)
            .map(|e| e.confidence)
            .unwrap_or(0.5);
        
        let confidence_factor = (entity1_conf + entity2_conf) / 2.0 * 0.3;
        
        (base_weight + confidence_factor).min(1.0)
    }
    
    /// Calculate entity confidence for referential links
    fn calculate_entity_confidence(&self, layer1: &KnowledgeLayer, layer2: &KnowledgeLayer, entity_name: &str) -> f32 {
        let entity1_conf = layer1.entities
            .iter()
            .find(|e| e.name == entity_name)
            .map(|e| e.confidence)
            .unwrap_or(0.5);
        
        let entity2_conf = layer2.entities
            .iter()
            .find(|e| e.name == entity_name)
            .map(|e| e.confidence)
            .unwrap_or(0.5);
        
        (entity1_conf + entity2_conf) / 2.0
    }
    
    /// Check if two layers are hierarchically related
    fn are_hierarchically_related(&self, layer1: &KnowledgeLayer, layer2: &KnowledgeLayer) -> bool {
        layer1.parent_layer_id == Some(layer2.layer_id.clone()) ||
        layer2.parent_layer_id == Some(layer1.layer_id.clone()) ||
        layer1.child_layer_ids.contains(&layer2.layer_id) ||
        layer2.child_layer_ids.contains(&layer1.layer_id)
    }
    
    /// Calculate embedding similarity using cosine similarity
    fn calculate_embedding_similarity(&self, embedding1: &[f32], embedding2: &[f32]) -> HierarchicalStorageResult<f32> {
        if embedding1.len() != embedding2.len() {
            return Err(HierarchicalStorageError::SemanticAnalysisError(
                "Embedding dimensions don't match".to_string()
            ));
        }
        
        let dot_product: f32 = embedding1
            .iter()
            .zip(embedding2.iter())
            .map(|(a, b)| a * b)
            .sum();
        
        let norm1: f32 = embedding1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = embedding2.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm1 == 0.0 || norm2 == 0.0 {
            return Ok(0.0);
        }
        
        Ok(dot_product / (norm1 * norm2))
    }
    
    /// Analyze causal/temporal relationships between layers
    async fn analyze_causal_temporal_relationship(
        &self,
        layer1: &KnowledgeLayer,
        layer2: &KnowledgeLayer,
    ) -> HierarchicalStorageResult<Option<(SemanticLinkType, f32, f32, Vec<String>)>> {
        // Simple heuristic-based approach (could be enhanced with AI analysis)
        let text1 = &layer1.content.processed_text;
        let text2 = &layer2.content.processed_text;
        
        // Check for causal indicators
        let causal_words = ["causes", "results in", "leads to", "because", "due to", "therefore"];
        let temporal_words = ["before", "after", "during", "then", "next", "previously"];
        
        let causal_score = causal_words
            .iter()
            .map(|&word| {
                (text1.to_lowercase().contains(word) as u32 + text2.to_lowercase().contains(word) as u32) as f32
            })
            .sum::<f32>() / causal_words.len() as f32;
        
        let temporal_score = temporal_words
            .iter()
            .map(|&word| {
                (text1.to_lowercase().contains(word) as u32 + text2.to_lowercase().contains(word) as u32) as f32
            })
            .sum::<f32>() / temporal_words.len() as f32;
        
        if causal_score > 0.0 && causal_score >= temporal_score {
            Ok(Some((
                SemanticLinkType::Causal,
                causal_score * 0.7,
                causal_score,
                vec!["Causal relationship detected via linguistic analysis".to_string()],
            )))
        } else if temporal_score > 0.0 {
            Ok(Some((
                SemanticLinkType::Temporal,
                temporal_score * 0.6,
                temporal_score,
                vec!["Temporal relationship detected via linguistic analysis".to_string()],
            )))
        } else {
            Ok(None)
        }
    }
    
    /// Calculate categorical similarity between layers
    fn calculate_categorical_similarity(&self, layer1: &KnowledgeLayer, layer2: &KnowledgeLayer) -> f32 {
        let tags1: HashSet<_> = layer1.content.metadata.tags.iter().collect();
        let tags2: HashSet<_> = layer2.content.metadata.tags.iter().collect();
        
        if tags1.is_empty() && tags2.is_empty() {
            return 0.5; // Neutral similarity
        }
        
        let intersection_size = tags1.intersection(&tags2).count() as f32;
        let union_size = tags1.union(&tags2).count() as f32;
        
        if union_size == 0.0 {
            0.5
        } else {
            intersection_size / union_size
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enhanced_knowledge_storage::model_management::ModelResourceManager;
    use crate::enhanced_knowledge_storage::types::{ModelResourceConfig, ComplexityLevel};
    use crate::enhanced_knowledge_storage::hierarchical_storage::types::{LayerContent, LayerMetadata, LayerPosition};
    use std::time::Duration;
    
    #[tokio::test]
    async fn test_semantic_link_manager_creation() {
        let model_config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(model_config));
        let storage_config = HierarchicalStorageConfig::default();
        
        let manager = SemanticLinkManager::new(model_manager, storage_config);
        
        assert_eq!(manager.config.semantic_similarity_threshold, 0.7);
    }
    
    #[test]
    fn test_embedding_similarity_calculation() {
        let model_config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(model_config));
        let storage_config = HierarchicalStorageConfig::default();
        
        let manager = SemanticLinkManager::new(model_manager, storage_config);
        
        let embedding1 = vec![1.0, 0.0, 0.0];
        let embedding2 = vec![0.0, 1.0, 0.0];
        let embedding3 = vec![1.0, 0.0, 0.0];
        
        let similarity1 = manager.calculate_embedding_similarity(&embedding1, &embedding2).unwrap();
        let similarity2 = manager.calculate_embedding_similarity(&embedding1, &embedding3).unwrap();
        
        assert!((similarity1 - 0.0).abs() < 0.001); // Orthogonal vectors
        assert!((similarity2 - 1.0).abs() < 0.001); // Identical vectors
    }
    
    #[test]
    fn test_categorical_similarity() {
        let model_config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(model_config));
        let storage_config = HierarchicalStorageConfig::default();
        
        let manager = SemanticLinkManager::new(model_manager, storage_config);
        
        let layer1 = KnowledgeLayer {
            layer_id: "layer1".to_string(),
            layer_type: LayerType::Paragraph,
            parent_layer_id: None,
            child_layer_ids: Vec::new(),
            content: LayerContent {
                raw_text: "content".to_string(),
                processed_text: "content".to_string(),
                key_phrases: Vec::new(),
                summary: None,
                metadata: LayerMetadata {
                    word_count: 1,
                    character_count: 7,
                    complexity_level: ComplexityLevel::Low,
                    reading_time: Duration::from_secs(1),
                    tags: vec!["science".to_string(), "physics".to_string()],
                    custom_attributes: HashMap::new(),
                },
            },
            entities: Vec::new(),
            relationships: Vec::new(),
            semantic_embedding: None,
            importance_score: 0.5,
            coherence_score: 0.7,
            position: LayerPosition {
                start_offset: 0,
                end_offset: 7,
                depth_level: 1,
                sequence_number: 0,
            },
        };
        
        let layer2 = KnowledgeLayer {
            layer_id: "layer2".to_string(),
            layer_type: LayerType::Paragraph,
            parent_layer_id: None,
            child_layer_ids: Vec::new(),
            content: LayerContent {
                raw_text: "content2".to_string(),
                processed_text: "content2".to_string(),
                key_phrases: Vec::new(),
                summary: None,
                metadata: LayerMetadata {
                    word_count: 1,
                    character_count: 8,
                    complexity_level: ComplexityLevel::Low,
                    reading_time: Duration::from_secs(1),
                    tags: vec!["science".to_string(), "mathematics".to_string()],
                    custom_attributes: HashMap::new(),
                },
            },
            entities: Vec::new(),
            relationships: Vec::new(),
            semantic_embedding: None,
            importance_score: 0.5,
            coherence_score: 0.7,
            position: LayerPosition {
                start_offset: 0,
                end_offset: 8,
                depth_level: 1,
                sequence_number: 1,
            },
        };
        
        let similarity = manager.calculate_categorical_similarity(&layer1, &layer2);
        
        // Should be 1/3 (intersection "science" / union "science", "physics", "mathematics")
        assert!((similarity - 0.333).abs() < 0.01);
    }
}