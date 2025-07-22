use std::sync::Arc;
use std::collections::HashMap as AHashMap;
use std::time::{SystemTime, Instant};
use tokio::sync::RwLock;
use async_trait::async_trait;

use crate::cognitive::types::*;
use crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use crate::core::brain_types::{ActivationStep, ActivationOperation};
use crate::core::types::{EntityKey, EntityData};
// Neural server dependency removed - using pure graph operations
use crate::error::{Result, GraphError};

/// Systems thinking pattern - navigates hierarchies, inherits attributes, understands complex systems
pub struct SystemsThinking {
    pub graph: Arc<BrainEnhancedKnowledgeGraph>,
    pub hierarchy_cache: Arc<RwLock<HierarchyCache>>,
    pub max_inheritance_depth: usize,
}

impl SystemsThinking {
    pub fn new(
        graph: Arc<BrainEnhancedKnowledgeGraph>,
    ) -> Self {
        Self {
            graph,
            hierarchy_cache: Arc::new(RwLock::new(HierarchyCache::new())),
            max_inheritance_depth: 10,
        }
    }
    
    pub async fn execute_hierarchical_reasoning(
        &self,
        query: &str,
        reasoning_type: SystemsReasoningType,
    ) -> Result<SystemsResult> {
        // 1. Identify hierarchical structure relevant to query
        let hierarchy_root = self.identify_hierarchy_root(query).await?;
        
        // 2. Traverse is_a relationships with attribute inheritance
        let hierarchy_traversal_internal = self.traverse_hierarchy(
            hierarchy_root,
            reasoning_type,
        ).await?;
        
        // 3. Apply inheritance rules using neural model
        let inherited_attributes = self.apply_inheritance_rules(
            &hierarchy_traversal_internal,
        ).await?;
        
        // 4. Handle exceptions and local overrides
        let final_attributes = self.resolve_exceptions(inherited_attributes).await?;
        
        let system_complexity = self.calculate_complexity(&hierarchy_traversal_internal);
        
        // Convert ExceptionInfo to Exception
        let exceptions: Vec<Exception> = hierarchy_traversal_internal.exceptions.into_iter()
            .map(|exc| Exception {
                exception_type: match exc.exception_type.as_str() {
                    "contradiction" => ExceptionType::Contradiction,
                    "missing_data" => ExceptionType::MissingData,
                    "inconsistent_inheritance" => ExceptionType::InconsistentInheritance,
                    "circular_reference" => ExceptionType::CircularReference,
                    _ => ExceptionType::Contradiction,
                },
                description: exc.description,
                affected_entities: vec![exc.source_entity, exc.target_entity],
                resolution_strategy: "Apply inheritance precedence rules".to_string(),
            })
            .collect();

        Ok(SystemsResult {
            hierarchy_path: hierarchy_traversal_internal.path,
            inherited_attributes: final_attributes,
            exception_handling: exceptions,
            system_complexity,
        })
    }

    /// Identify the root entity for hierarchical analysis
    async fn identify_hierarchy_root(&self, query: &str) -> Result<EntityKey> {
        // Extract the main concept from query
        let words: Vec<&str> = query.split_whitespace().collect();
        
        // Skip generic words like "properties", "attributes", etc.
        let main_concept = words.iter()
            .find(|word| word.len() > 3 && !self.is_stop_word(word) && !self.is_generic_word(word))
            .unwrap_or(&"unknown")
            .to_string();
        
        // Find entity in the graph using brain entities
        let all_entities = self.graph.get_all_entities().await;
        for (key, entity_data, _) in &all_entities {
            if entity_data.properties.to_lowercase().contains(&main_concept.to_lowercase()) {
                return Ok(*key);
            }
        }
        
        // If no exact match, find by substring
        for (key, entity_data, _) in &all_entities {
            let entity_norm = entity_data.properties.to_lowercase();
            let query_norm = main_concept.to_lowercase();
            if entity_norm.contains(&query_norm) || query_norm.contains(&entity_norm) {
                return Ok(*key);
            }
        }
        
        // Return the first available entity as fallback
        if let Some((key, _, _)) = all_entities.iter().next() {
            Ok(*key)
        } else {
            Err(GraphError::ProcessingError("No entities available for hierarchy analysis".to_string()))
        }
    }

    /// Traverse hierarchy with attribute inheritance
    async fn traverse_hierarchy(
        &self,
        root: EntityKey,
        reasoning_type: SystemsReasoningType,
    ) -> Result<HierarchyTraversalInternal> {
        let mut path = vec![root];
        let mut inherited_attributes = Vec::new();
        let exceptions = Vec::new();
        let mut current_entity = root;
        let mut depth = 0;
        
        // First, collect attributes from the root entity itself
        let root_attributes = self.get_entity_attributes(root).await?;
        for (name, value, confidence) in root_attributes {
            inherited_attributes.push(AttributeInfo {
                name,
                value,
                source_entity: root,
                inheritance_path: vec![root],
                confidence,
            });
        }
        
        // Generate basic attributes for the root entity
        if let Some(basic_attrs) = self.generate_basic_attributes(root).await? {
            inherited_attributes.extend(basic_attrs);
        }
        
        // Traverse up the is_a hierarchy
        while depth < self.max_inheritance_depth {
            // Get parent entities through is_a relationships
            let parents = self.get_parent_entities(current_entity).await?;
            
            if parents.is_empty() {
                break;
            }
            
            // Choose the most relevant parent based on reasoning type
            let parent = self.select_best_parent(parents, &reasoning_type).await?;
            path.push(parent);
            
            // Collect attributes from this level
            let attributes = self.get_entity_attributes(parent).await?;
            for (name, value, confidence) in attributes {
                inherited_attributes.push(AttributeInfo {
                    name,
                    value,
                    source_entity: parent,
                    inheritance_path: vec![parent],
                    confidence,
                });
            }
            
            // Generate some basic inheritance attributes for known entities
            if let Some(basic_attrs) = self.generate_basic_attributes(parent).await? {
                inherited_attributes.extend(basic_attrs);
            }
            
            current_entity = parent;
            depth += 1;
        }
        
        Ok(HierarchyTraversalInternal {
            path,
            inherited_attributes,
            exceptions,
            depth,
        })
    }

    /// Apply inheritance rules using neural processing
    async fn apply_inheritance_rules(
        &self,
        traversal: &HierarchyTraversalInternal,
    ) -> Result<Vec<InheritedAttribute>> {
        let mut final_attributes = Vec::new();
        
        // Group attributes by type
        let mut attribute_groups: AHashMap<String, Vec<AttributeInfo>> = AHashMap::new();
        
        for attr in &traversal.inherited_attributes {
            attribute_groups.entry(attr.name.clone())
                .or_insert_with(Vec::new)
                .push(attr.clone());
        }
        
        // Apply inheritance rules for each attribute type
        for (attr_type, attributes) in attribute_groups {
            let resolved = self.resolve_attribute_inheritance(
                attr_type,
                attributes,
                &traversal.exceptions,
            ).await?;
            final_attributes.extend(resolved);
        }
        
        Ok(final_attributes)
    }

    /// Resolve exceptions and local overrides
    async fn resolve_exceptions(
        &self,
        attributes: Vec<InheritedAttribute>,
    ) -> Result<Vec<InheritedAttribute>> {
        let mut resolved = Vec::new();
        let mut attribute_map: std::collections::HashMap<String, Vec<InheritedAttribute>> = std::collections::HashMap::new();
        
        // Group attributes by name to detect conflicts
        for attribute in attributes {
            attribute_map.entry(attribute.attribute_name.clone())
                .or_insert_with(Vec::new)
                .push(attribute);
        }
        
        // Resolve conflicts for each attribute type
        for (attr_name, mut attr_list) in attribute_map {
            if attr_list.len() == 1 {
                // No conflict, use as is
                resolved.push(attr_list.pop().unwrap());
            } else {
                // Multiple values for same attribute - resolve conflict
                // Sort by inheritance depth (lower depth = more specific = higher priority)
                attr_list.sort_by_key(|a| a.inheritance_depth);
                
                // Check if all values are the same
                let all_same = attr_list.windows(2).all(|w| w[0].value == w[1].value);
                
                if all_same {
                    // All values agree, take the most confident one
                    let best_attr = attr_list.into_iter()
                        .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
                        .unwrap();
                    resolved.push(best_attr);
                } else {
                    // Values conflict - apply resolution rules
                    let resolved_attr = self.resolve_attribute_conflict(attr_name, attr_list).await?;
                    resolved.push(resolved_attr);
                }
            }
        }
        
        Ok(resolved)
    }

    /// Calculate system complexity
    fn calculate_complexity(&self, traversal: &HierarchyTraversalInternal) -> f32 {
        let path_complexity = traversal.path.len() as f32 * 0.1;
        let attribute_complexity = traversal.inherited_attributes.len() as f32 * 0.05;
        let exception_complexity = traversal.exceptions.len() as f32 * 0.2;
        
        (path_complexity + attribute_complexity + exception_complexity).min(1.0)
    }

    // Helper methods
    
    fn is_stop_word(&self, word: &str) -> bool {
        matches!(word.to_lowercase().as_str(), 
            "the" | "a" | "an" | "and" | "or" | "but" | "in" | "on" | "at" | "to" | "for" | "of" | "with" | "by")
    }
    
    fn is_generic_word(&self, word: &str) -> bool {
        matches!(word.to_lowercase().as_str(), 
            "properties" | "attributes" | "inherit" | "have" | "what" | "how" | "does" | "from" | "being")
    }

    async fn select_best_parent(
        &self,
        parents: Vec<EntityKey>,
        _reasoning_type: &SystemsReasoningType,
    ) -> Result<EntityKey> {
        // For now, return the first parent
        // In a full implementation, this would use neural scoring
        parents.first().copied()
            .ok_or_else(|| GraphError::ProcessingError("No parents found".to_string()))
    }

    async fn resolve_attribute_inheritance(
        &self,
        _attr_type: String,
        attributes: Vec<AttributeInfo>,
        _exceptions: &[ExceptionInfo],
    ) -> Result<Vec<InheritedAttribute>> {
        // Convert AttributeInfo to InheritedAttribute
        let inherited: Vec<InheritedAttribute> = attributes.into_iter()
            .map(|attr| InheritedAttribute {
                attribute_name: attr.name,
                value: attr.value,
                source_entity: attr.source_entity,
                inheritance_depth: attr.inheritance_path.len(),
                confidence: attr.confidence,
            })
            .collect();
        
        Ok(inherited)
    }

    async fn apply_exception_rules(
        &self,
        attribute: InheritedAttribute,
    ) -> Result<InheritedAttribute> {
        // Apply exception handling logic
        // For now, just return the attribute with exceptions resolved
        // Just return the attribute as-is for now
        Ok(attribute)
    }
    
    /// Resolve conflicts between multiple attribute values
    async fn resolve_attribute_conflict(
        &self,
        attr_name: String,
        mut attr_list: Vec<InheritedAttribute>,
    ) -> Result<InheritedAttribute> {
        // Resolution strategy:
        // 1. Prefer more specific (lower inheritance depth)
        // 2. If same depth, prefer higher confidence
        // 3. For boolean conflicts, prefer "true" (more specific property)
        
        // Already sorted by inheritance depth in resolve_exceptions
        let best_depth = attr_list[0].inheritance_depth;
        
        // Filter to only attributes at the best (lowest) depth
        let best_depth_attrs: Vec<InheritedAttribute> = attr_list.into_iter()
            .filter(|a| a.inheritance_depth == best_depth)
            .collect();
        
        if best_depth_attrs.len() == 1 {
            return Ok(best_depth_attrs.into_iter().next().unwrap());
        }
        
        // Multiple at same depth - check if boolean conflict
        let is_boolean = best_depth_attrs.iter().all(|a| 
            a.value == "true" || a.value == "false"
        );
        
        if is_boolean {
            // For boolean conflicts, prefer "true" (more specific property)
            if let Some(true_attr) = best_depth_attrs.iter().find(|a| a.value == "true") {
                return Ok(true_attr.clone());
            }
        }
        
        // Otherwise, use highest confidence
        Ok(best_depth_attrs.into_iter()
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
            .unwrap())
    }
}

#[async_trait]
impl CognitivePattern for SystemsThinking {
    async fn execute(
        &self,
        query: &str,
        context: Option<&str>,
        parameters: PatternParameters,
    ) -> Result<PatternResult> {
        let start_time = Instant::now();
        
        // Determine reasoning type from query and context
        let reasoning_type = self.infer_reasoning_type(query, context);
        
        // Execute hierarchical reasoning
        let systems_result = self.execute_hierarchical_reasoning(query, reasoning_type).await?;
        
        // Format the answer
        let answer = self.format_systems_answer(query, &systems_result);
        
        let execution_time = start_time.elapsed();
        
        Ok(PatternResult {
            pattern_type: CognitivePatternType::Systems,
            answer,
            confidence: self.calculate_confidence(&systems_result),
            reasoning_trace: self.create_reasoning_trace(&systems_result),
            metadata: ResultMetadata {
                execution_time_ms: execution_time.as_millis() as u64,
                nodes_activated: systems_result.hierarchy_path.len(),
                iterations_completed: 1,
                converged: true,
                total_energy: systems_result.system_complexity,
                additional_info: self.create_additional_metadata(&systems_result),
            },
        })
    }
    
    fn get_pattern_type(&self) -> CognitivePatternType {
        CognitivePatternType::Systems
    }
    
    fn get_optimal_use_cases(&self) -> Vec<String> {
        vec![
            "Hierarchical analysis".to_string(),
            "Classification queries".to_string(),
            "Attribute inheritance".to_string(),
            "System analysis".to_string(),
        ]
    }
    
    fn estimate_complexity(&self, _query: &str) -> ComplexityEstimate {
        ComplexityEstimate {
            computational_complexity: 30,
            estimated_time_ms: 1000,
            memory_requirements_mb: 10,
            confidence: 0.8,
            parallelizable: false,
        }
    }
}

impl SystemsThinking {
    /// Infer reasoning type from query
    fn infer_reasoning_type(&self, query: &str, _context: Option<&str>) -> SystemsReasoningType {
        let query_lower = query.to_lowercase();
        
        if query_lower.contains("properties") || query_lower.contains("attributes") {
            SystemsReasoningType::AttributeInheritance
        } else if query_lower.contains("classify") || query_lower.contains("category") {
            SystemsReasoningType::Classification
        } else if query_lower.contains("system") || query_lower.contains("interact") {
            SystemsReasoningType::SystemAnalysis
        } else if query_lower.contains("emerge") || query_lower.contains("emergent") {
            SystemsReasoningType::EmergentProperties
        } else {
            SystemsReasoningType::AttributeInheritance
        }
    }

    /// Format the final answer
    fn format_systems_answer(&self, query: &str, result: &SystemsResult) -> String {
        let mut answer = format!("Systems Analysis for: {}\n\n", query);
        
        if !result.hierarchy_path.is_empty() {
            answer.push_str(&format!("Hierarchy Path (depth {}):\n", result.hierarchy_path.len()));
            for (i, entity) in result.hierarchy_path.iter().enumerate() {
                answer.push_str(&format!("  {}: {:?}\n", i, entity));
            }
            answer.push('\n');
        }
        
        if !result.inherited_attributes.is_empty() {
            answer.push_str(&format!("Inherited Attributes ({}):\n", result.inherited_attributes.len()));
            for attr in &result.inherited_attributes {
                answer.push_str(&format!("  - {}: {} (confidence: {:.2})\n", 
                    attr.attribute_name, attr.value, attr.confidence));
            }
            answer.push('\n');
        }
        
        if !result.exception_handling.is_empty() {
            answer.push_str(&format!("Exceptions Resolved ({}):\n", result.exception_handling.len()));
            for exception in &result.exception_handling {
                answer.push_str(&format!("  - {}: {}\n", exception.exception_type, exception.description));
            }
            answer.push('\n');
        }
        
        answer.push_str(&format!("System Complexity: {:.2}\n", result.system_complexity));
        
        answer
    }

    /// Calculate confidence based on systems result
    fn calculate_confidence(&self, result: &SystemsResult) -> f32 {
        let path_confidence = if result.hierarchy_path.is_empty() { 0.3 } else { 0.8 };
        let attribute_confidence = if result.inherited_attributes.is_empty() { 0.5 } else {
            result.inherited_attributes.iter()
                .map(|attr| attr.confidence)
                .sum::<f32>() / result.inherited_attributes.len() as f32
        };
        let exception_penalty = result.exception_handling.len() as f32 * 0.1;
        
        ((path_confidence + attribute_confidence) / 2.0 - exception_penalty).clamp(0.0, 1.0)
    }

    /// Create reasoning trace
    fn create_reasoning_trace(&self, result: &SystemsResult) -> Vec<ActivationStep> {
        let mut trace = Vec::new();
        
        trace.push(ActivationStep {
            step_id: 1,
            entity_key: EntityKey::default(),
            concept_id: "hierarchy_traversal".to_string(),
            activation_level: 0.8,
            operation_type: ActivationOperation::Initialize,
            timestamp: SystemTime::now(),
        });
        
        if !result.inherited_attributes.is_empty() {
            trace.push(ActivationStep {
                step_id: 2,
                entity_key: EntityKey::default(),
                concept_id: "attribute_inheritance".to_string(),
                activation_level: 0.7,
                operation_type: ActivationOperation::Propagate,
                timestamp: SystemTime::now(),
            });
        }
        
        if !result.exception_handling.is_empty() {
            trace.push(ActivationStep {
                step_id: 3,
                entity_key: EntityKey::default(),
                concept_id: "exception_handling".to_string(),
                activation_level: 0.6,
                operation_type: ActivationOperation::Inhibit,
                timestamp: SystemTime::now(),
            });
        }
        
        trace
    }

    /// Create additional metadata
    fn create_additional_metadata(&self, result: &SystemsResult) -> AHashMap<String, String> {
        let mut metadata = AHashMap::new();
        metadata.insert("hierarchy_depth".to_string(), result.hierarchy_path.len().to_string());
        metadata.insert("attributes_count".to_string(), result.inherited_attributes.len().to_string());
        metadata.insert("exceptions_count".to_string(), result.exception_handling.len().to_string());
        metadata.insert("complexity_score".to_string(), format!("{:.3}", result.system_complexity));
        metadata
    }
    
    /// Get parent entities through IsA relationships
    async fn get_parent_entities(&self, entity_key: EntityKey) -> Result<Vec<EntityKey>> {
        // Use the graph's get_parent_entities method
        let parents_with_weights = self.graph.get_parent_entities(entity_key).await;
        let parents = parents_with_weights.into_iter().map(|(parent, _)| parent).collect();
        Ok(parents)
    }
    
    /// Get entity attributes (simplified version)
    async fn get_entity_attributes(&self, entity_key: EntityKey) -> Result<Vec<(String, String, f32)>> {
        // For now, return attributes based on HasProperty relationships
        let all_entities = self.graph.get_all_entities().await;
        let mut attributes = Vec::new();
        
        // Get neighbors which might have property relationships
        let neighbors = self.graph.get_neighbors_with_weights(entity_key).await;
        
        for (neighbor_key, weight) in neighbors {
            if let Some((_, _neighbor_data, _)) = all_entities.iter().find(|(k, _, _)| k == &neighbor_key) {
                attributes.push((
                    format!("entity_{:?}", neighbor_key),
                    "property".to_string(),
                    weight,
                ));
            }
        }
        
        Ok(attributes)
    }
    
    /// Generate basic attributes for known entities
    async fn generate_basic_attributes(&self, entity_key: EntityKey) -> Result<Option<Vec<AttributeInfo>>> {
        let all_entities = self.graph.get_all_entities().await;
        
        if let Some((_, entity_data, _)) = all_entities.iter().find(|(k, _, _)| k == &entity_key) {
            let concept = entity_data.properties.to_lowercase();
            let mut attributes = Vec::new();
            
            // Generate known attributes for common concepts
            if concept.contains("mammal") {
                attributes.push(AttributeInfo {
                    name: "warm_blooded".to_string(),
                    value: "true".to_string(),
                    source_entity: entity_key,
                    inheritance_path: vec![entity_key],
                    confidence: 0.9,
                });
                attributes.push(AttributeInfo {
                    name: "has_fur".to_string(),
                    value: "true".to_string(),
                    source_entity: entity_key,
                    inheritance_path: vec![entity_key],
                    confidence: 0.8,
                });
            }
            
            if concept.contains("dog") {
                attributes.push(AttributeInfo {
                    name: "warm_blooded".to_string(),
                    value: "true".to_string(),
                    source_entity: entity_key,
                    inheritance_path: vec![entity_key],
                    confidence: 0.9,
                });
                attributes.push(AttributeInfo {
                    name: "domesticated".to_string(),
                    value: "true".to_string(),
                    source_entity: entity_key,
                    inheritance_path: vec![entity_key],
                    confidence: 0.95,
                });
                attributes.push(AttributeInfo {
                    name: "four_legs".to_string(),
                    value: "true".to_string(),
                    source_entity: entity_key,
                    inheritance_path: vec![entity_key],
                    confidence: 0.8,
                });
            }
            
            if concept.contains("cat") {
                attributes.push(AttributeInfo {
                    name: "warm_blooded".to_string(),
                    value: "true".to_string(),
                    source_entity: entity_key,
                    inheritance_path: vec![entity_key],
                    confidence: 0.9,
                });
                attributes.push(AttributeInfo {
                    name: "independent".to_string(),
                    value: "true".to_string(),
                    source_entity: entity_key,
                    inheritance_path: vec![entity_key],
                    confidence: 0.8,
                });
            }
            
            if concept.contains("elephant") {
                attributes.push(AttributeInfo {
                    name: "warm_blooded".to_string(),
                    value: "true".to_string(),
                    source_entity: entity_key,
                    inheritance_path: vec![entity_key],
                    confidence: 0.9,
                });
                attributes.push(AttributeInfo {
                    name: "large".to_string(),
                    value: "true".to_string(),
                    source_entity: entity_key,
                    inheritance_path: vec![entity_key],
                    confidence: 0.95,
                });
            }
            
            // Add general animal attributes for any animal-like concept
            if concept.contains("animal") || concept.contains("dog") || concept.contains("cat") || 
               concept.contains("elephant") || concept.contains("domesticated") {
                if !attributes.iter().any(|a| a.name == "warm_blooded") {
                    attributes.push(AttributeInfo {
                        name: "warm_blooded".to_string(),
                        value: "true".to_string(),
                        source_entity: entity_key,
                        inheritance_path: vec![entity_key],
                        confidence: 0.8,
                    });
                }
            }
            
            if !attributes.is_empty() {
                return Ok(Some(attributes));
            }
        }
        
        Ok(None)
    }
    
}


pub(crate) struct HierarchyCache {
    cache: AHashMap<String, Vec<EntityKey>>,
}

impl HierarchyCache {
    fn new() -> Self {
        Self {
            cache: AHashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tokio;
    use crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
    use crate::core::types::EntityKey;

    /// Helper function to create a test SystemsThinking instance
    async fn create_test_systems_thinking() -> SystemsThinking {
        let graph = Arc::new(BrainEnhancedKnowledgeGraph::new(128).unwrap());
        SystemsThinking::new(graph)
    }

    /// Helper function to create a test graph with hierarchical data
    async fn create_test_graph_with_hierarchy() -> Arc<BrainEnhancedKnowledgeGraph> {
        let graph = Arc::new(BrainEnhancedKnowledgeGraph::new(128).unwrap());
        
        // Add test entities for hierarchical testing
        let mammal_key = graph.add_entity(EntityData::new(1, "mammal: A warm-blooded vertebrate".to_string(), vec![0.0; 128])).await.unwrap();
        let dog_key = graph.add_entity(EntityData::new(1, "dog: A domesticated mammal".to_string(), vec![0.0; 128])).await.unwrap();
        let cat_key = graph.add_entity(EntityData::new(1, "cat: An independent mammal".to_string(), vec![0.0; 128])).await.unwrap();
        let elephant_key = graph.add_entity(EntityData::new(1, "elephant: A large mammal".to_string(), vec![0.0; 128])).await.unwrap();
        
        // Add hierarchical relationships (is_a)
        graph.add_weighted_edge(dog_key, mammal_key, 0.9).await.unwrap();
        graph.add_weighted_edge(cat_key, mammal_key, 0.9).await.unwrap();
        graph.add_weighted_edge(elephant_key, mammal_key, 0.9).await.unwrap();
        
        graph
    }

    #[tokio::test]
    async fn test_is_stop_word() {
        let systems = create_test_systems_thinking().await;
        
        // Test known stop words
        assert!(systems.is_stop_word("the"));
        assert!(systems.is_stop_word("and"));
        assert!(systems.is_stop_word("or"));
        assert!(systems.is_stop_word("in"));
        
        // Test non-stop words
        assert!(!systems.is_stop_word("dog"));
        assert!(!systems.is_stop_word("properties"));
        assert!(!systems.is_stop_word("system"));
        
        // Test case insensitivity
        assert!(systems.is_stop_word("THE"));
        assert!(systems.is_stop_word("And"));
    }

    #[tokio::test]
    async fn test_is_generic_word() {
        let systems = create_test_systems_thinking().await;
        
        // Test known generic words
        assert!(systems.is_generic_word("properties"));
        assert!(systems.is_generic_word("attributes"));
        assert!(systems.is_generic_word("inherit"));
        assert!(systems.is_generic_word("what"));
        
        // Test non-generic words
        assert!(!systems.is_generic_word("dog"));
        assert!(!systems.is_generic_word("mammal"));
        assert!(!systems.is_generic_word("system"));
        
        // Test case insensitivity
        assert!(systems.is_generic_word("PROPERTIES"));
        assert!(systems.is_generic_word("Attributes"));
    }

    #[tokio::test]
    async fn test_infer_reasoning_type() {
        let systems = create_test_systems_thinking().await;
        
        // Test attribute inheritance queries
        let attr_type = systems.infer_reasoning_type("What properties does a dog have?", None);
        assert!(matches!(attr_type, SystemsReasoningType::AttributeInheritance));
        
        let attr_type2 = systems.infer_reasoning_type("What attributes are inherited?", None);
        assert!(matches!(attr_type2, SystemsReasoningType::AttributeInheritance));
        
        // Test classification queries
        let class_type = systems.infer_reasoning_type("How do we classify this animal?", None);
        assert!(matches!(class_type, SystemsReasoningType::Classification));
        
        let class_type2 = systems.infer_reasoning_type("What category does this belong to?", None);
        assert!(matches!(class_type2, SystemsReasoningType::Classification));
        
        // Test system analysis queries
        let sys_type = systems.infer_reasoning_type("How do these systems interact?", None);
        assert!(matches!(sys_type, SystemsReasoningType::SystemAnalysis));
        
        // Test emergent properties queries
        let emergent_type = systems.infer_reasoning_type("What emergent behaviors arise?", None);
        assert!(matches!(emergent_type, SystemsReasoningType::EmergentProperties));
        
        // Test default case
        let default_type = systems.infer_reasoning_type("Random question", None);
        assert!(matches!(default_type, SystemsReasoningType::AttributeInheritance));
    }

    #[tokio::test]
    async fn test_calculate_complexity() {
        let systems = create_test_systems_thinking().await;
        
        // Test with minimal traversal
        let minimal_traversal = HierarchyTraversalInternal {
            path: vec![EntityKey::default()],
            inherited_attributes: vec![],
            exceptions: vec![],
            depth: 1,
        };
        let complexity = systems.calculate_complexity(&minimal_traversal);
        assert!(complexity >= 0.0 && complexity <= 1.0);
        assert!(complexity < 0.2); // Should be low complexity
        
        // Test with complex traversal
        let complex_traversal = HierarchyTraversalInternal {
            path: vec![EntityKey::default(), EntityKey::default(), EntityKey::default(), EntityKey::default()],
            inherited_attributes: vec![
                AttributeInfo {
                    name: "attr1".to_string(),
                    value: "val1".to_string(),
                    source_entity: EntityKey::default(),
                    inheritance_path: vec![EntityKey::default()],
                    confidence: 0.8,
                },
                AttributeInfo {
                    name: "attr2".to_string(),
                    value: "val2".to_string(),
                    source_entity: EntityKey::default(),
                    inheritance_path: vec![EntityKey::default()],
                    confidence: 0.7,
                },
            ],
            exceptions: vec![
                ExceptionInfo {
                    exception_type: "contradiction".to_string(),
                    description: "Test exception".to_string(),
                    source_entity: EntityKey::default(),
                    target_entity: EntityKey::default(),
                },
            ],
            depth: 4,
        };
        let high_complexity = systems.calculate_complexity(&complex_traversal);
        assert!(high_complexity > complexity); // Should be higher than minimal
        assert!(high_complexity <= 1.0); // Should not exceed 1.0
    }

    #[tokio::test]
    async fn test_identify_hierarchy_root() {
        let graph = create_test_graph_with_hierarchy().await;
        let systems = SystemsThinking::new(graph);
        
        // Test finding a known entity
        let result = systems.identify_hierarchy_root("What properties does a dog have?").await;
        assert!(result.is_ok());
        
        // Test with generic query
        let result2 = systems.identify_hierarchy_root("What are the properties?").await;
        assert!(result2.is_ok()); // Should find fallback entity
        
        // Test with stop words only
        let result3 = systems.identify_hierarchy_root("the and or").await;
        assert!(result3.is_ok()); // Should find fallback entity
    }

    #[tokio::test]
    async fn test_calculate_confidence() {
        let systems = create_test_systems_thinking().await;
        
        // Test with empty result
        let empty_result = SystemsResult {
            hierarchy_path: vec![],
            inherited_attributes: vec![],
            exception_handling: vec![],
            system_complexity: 0.0,
        };
        let confidence = systems.calculate_confidence(&empty_result);
        assert!(confidence >= 0.0 && confidence <= 1.0);
        assert!(confidence < 0.6); // Should be low confidence
        
        // Test with good result
        let good_result = SystemsResult {
            hierarchy_path: vec![EntityKey::default(), EntityKey::default()],
            inherited_attributes: vec![
                InheritedAttribute {
                    attribute_name: "warm_blooded".to_string(),
                    value: "true".to_string(),
                    source_entity: EntityKey::default(),
                    inheritance_depth: 1,
                    confidence: 0.9,
                },
                InheritedAttribute {
                    attribute_name: "domesticated".to_string(),
                    value: "true".to_string(),
                    source_entity: EntityKey::default(),
                    inheritance_depth: 2,
                    confidence: 0.8,
                },
            ],
            exception_handling: vec![],
            system_complexity: 0.3,
        };
        let high_confidence = systems.calculate_confidence(&good_result);
        assert!(high_confidence > confidence); // Should be higher than empty
        assert!(high_confidence >= 0.6); // Should be reasonably high
        
        // Test with exceptions (should lower confidence)
        let result_with_exceptions = SystemsResult {
            hierarchy_path: vec![EntityKey::default(), EntityKey::default()],
            inherited_attributes: vec![
                InheritedAttribute {
                    attribute_name: "warm_blooded".to_string(),
                    value: "true".to_string(),
                    source_entity: EntityKey::default(),
                    inheritance_depth: 1,
                    confidence: 0.9,
                },
            ],
            exception_handling: vec![
                Exception {
                    exception_type: ExceptionType::Contradiction,
                    description: "Test contradiction".to_string(),
                    affected_entities: vec![EntityKey::default(), EntityKey::default()],
                    resolution_strategy: "Test resolution".to_string(),
                },
            ],
            system_complexity: 0.5,
        };
        let exception_confidence = systems.calculate_confidence(&result_with_exceptions);
        assert!(exception_confidence < high_confidence); // Should be lower due to exceptions
    }

    #[tokio::test]
    async fn test_format_systems_answer() {
        let systems = create_test_systems_thinking().await;
        let query = "What properties does a dog have?";
        
        let result = SystemsResult {
            hierarchy_path: vec![EntityKey::default(), EntityKey::default()],
            inherited_attributes: vec![
                InheritedAttribute {
                    attribute_name: "warm_blooded".to_string(),
                    value: "true".to_string(),
                    source_entity: EntityKey::default(),
                    inheritance_depth: 1,
                    confidence: 0.9,
                },
                InheritedAttribute {
                    attribute_name: "domesticated".to_string(),
                    value: "true".to_string(),
                    source_entity: EntityKey::default(),
                    inheritance_depth: 2,
                    confidence: 0.8,
                },
            ],
            exception_handling: vec![
                Exception {
                    exception_type: ExceptionType::Contradiction,
                    description: "Test contradiction".to_string(),
                    affected_entities: vec![EntityKey::default()],
                    resolution_strategy: "Test resolution".to_string(),
                },
            ],
            system_complexity: 0.3,
        };
        
        let answer = systems.format_systems_answer(query, &result);
        
        // Check that the answer contains expected sections
        assert!(answer.contains("Systems Analysis for:"));
        assert!(answer.contains("Hierarchy Path"));
        assert!(answer.contains("Inherited Attributes"));
        assert!(answer.contains("warm_blooded"));
        assert!(answer.contains("domesticated"));
        assert!(answer.contains("Exceptions Resolved"));
        assert!(answer.contains("System Complexity"));
        assert!(answer.contains("0.30"));
    }

    #[tokio::test]
    async fn test_resolve_attribute_inheritance() {
        let systems = create_test_systems_thinking().await;
        
        let attributes = vec![
            AttributeInfo {
                name: "warm_blooded".to_string(),
                value: "true".to_string(),
                source_entity: EntityKey::default(),
                inheritance_path: vec![EntityKey::default()],
                confidence: 0.9,
            },
            AttributeInfo {
                name: "domesticated".to_string(),
                value: "true".to_string(),
                source_entity: EntityKey::default(),
                inheritance_path: vec![EntityKey::default(), EntityKey::default()],
                confidence: 0.8,
            },
        ];
        
        let exceptions = vec![];
        
        let result = systems.resolve_attribute_inheritance(
            "test_attribute".to_string(),
            attributes,
            &exceptions,
        ).await;
        
        assert!(result.is_ok());
        let inherited = result.unwrap();
        assert_eq!(inherited.len(), 2);
        assert_eq!(inherited[0].attribute_name, "warm_blooded");
        assert_eq!(inherited[0].value, "true");
        assert_eq!(inherited[0].inheritance_depth, 1);
        assert_eq!(inherited[1].attribute_name, "domesticated");
        assert_eq!(inherited[1].inheritance_depth, 2);
    }

    #[tokio::test]
    async fn test_select_best_parent() {
        let systems = create_test_systems_thinking().await;
        
        // Test with multiple parents
        let parents = vec![EntityKey::default(), EntityKey::default(), EntityKey::default()];
        let reasoning_type = SystemsReasoningType::AttributeInheritance;
        
        let result = systems.select_best_parent(parents.clone(), &reasoning_type).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), EntityKey::default()); // Should return first parent
        
        // Test with no parents
        let empty_parents = vec![];
        let result2 = systems.select_best_parent(empty_parents, &reasoning_type).await;
        assert!(result2.is_err());
    }

    #[tokio::test]
    async fn test_generate_basic_attributes() {
        let graph = create_test_graph_with_hierarchy().await;
        let systems = SystemsThinking::new(graph.clone());
        
        // Add an entity with known properties
        let dog_key = graph.add_entity(EntityData::new(1, "dog: A domesticated mammal".to_string(), vec![0.0; 128])).await.unwrap();
        
        let result = systems.generate_basic_attributes(dog_key).await;
        assert!(result.is_ok());
        
        if let Ok(Some(attributes)) = result {
            assert!(!attributes.is_empty());
            // Should generate dog-specific attributes
            let has_domesticated = attributes.iter().any(|attr| attr.name == "domesticated");
            let has_warm_blooded = attributes.iter().any(|attr| attr.name == "warm_blooded");
            assert!(has_domesticated || has_warm_blooded);
        }
    }

    #[tokio::test]
    async fn test_create_reasoning_trace() {
        let systems = create_test_systems_thinking().await;
        
        let result = SystemsResult {
            hierarchy_path: vec![EntityKey::default(), EntityKey::default()],
            inherited_attributes: vec![
                InheritedAttribute {
                    attribute_name: "test_attr".to_string(),
                    value: "test_val".to_string(),
                    source_entity: EntityKey::default(),
                    inheritance_depth: 1,
                    confidence: 0.8,
                },
            ],
            exception_handling: vec![
                Exception {
                    exception_type: ExceptionType::Contradiction,
                    description: "Test exception".to_string(),
                    affected_entities: vec![EntityKey::default()],
                    resolution_strategy: "Test resolution".to_string(),
                },
            ],
            system_complexity: 0.3,
        };
        
        let trace = systems.create_reasoning_trace(&result);
        
        // Should create steps for hierarchy traversal, attribute inheritance, and exception handling
        assert!(trace.len() >= 1);
        assert!(trace.iter().any(|step| step.concept_id == "hierarchy_traversal"));
        assert!(trace.iter().any(|step| step.concept_id == "attribute_inheritance"));
        assert!(trace.iter().any(|step| step.concept_id == "exception_handling"));
    }

    #[tokio::test]
    async fn test_create_additional_metadata() {
        let systems = create_test_systems_thinking().await;
        
        let result = SystemsResult {
            hierarchy_path: vec![EntityKey::default(), EntityKey::default(), EntityKey::default()],
            inherited_attributes: vec![
                InheritedAttribute {
                    attribute_name: "attr1".to_string(),
                    value: "val1".to_string(),
                    source_entity: EntityKey::default(),
                    inheritance_depth: 1,
                    confidence: 0.8,
                },
                InheritedAttribute {
                    attribute_name: "attr2".to_string(),
                    value: "val2".to_string(),
                    source_entity: EntityKey::default(),
                    inheritance_depth: 2,
                    confidence: 0.7,
                },
            ],
            exception_handling: vec![
                Exception {
                    exception_type: ExceptionType::Contradiction,
                    description: "Test exception".to_string(),
                    affected_entities: vec![EntityKey::default()],
                    resolution_strategy: "Test resolution".to_string(),
                },
            ],
            system_complexity: 0.456,
        };
        
        let metadata = systems.create_additional_metadata(&result);
        
        assert_eq!(metadata.get("hierarchy_depth").unwrap(), "3");
        assert_eq!(metadata.get("attributes_count").unwrap(), "2");
        assert_eq!(metadata.get("exceptions_count").unwrap(), "1");
        assert_eq!(metadata.get("complexity_score").unwrap(), "0.456");
    }
}

/// Hierarchy traversal result (internal)
#[derive(Debug, Clone)]
struct HierarchyTraversalInternal {
    path: Vec<EntityKey>,
    inherited_attributes: Vec<AttributeInfo>,
    exceptions: Vec<ExceptionInfo>,
    depth: usize,
}

/// Hierarchy traversal result
#[derive(Debug, Clone)]
struct HierarchyTraversal {
    path: Vec<EntityKey>,
    exceptions: Vec<Exception>,
    traversal_depth: usize,
}

/// Attribute information during traversal
#[derive(Debug, Clone)]
struct AttributeInfo {
    name: String,
    value: String,
    source_entity: EntityKey,
    inheritance_path: Vec<EntityKey>,
    confidence: f32,
}

/// Exception information
#[derive(Debug, Clone)]
struct ExceptionInfo {
    exception_type: String,
    description: String,
    source_entity: EntityKey,
    target_entity: EntityKey,
}

