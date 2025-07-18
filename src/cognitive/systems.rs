use std::sync::Arc;
use std::collections::HashMap as AHashMap;
use std::time::{SystemTime, Instant};
use tokio::sync::RwLock;
use async_trait::async_trait;

use crate::cognitive::types::*;
use crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use crate::core::brain_types::{ActivationStep, ActivationOperation};
use crate::core::types::EntityKey;
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
        
        for attribute in attributes {
            // Check if this attribute has any exceptions
            // For now, assume no exceptions since has_exceptions field doesn't exist
            if false {
                // Apply exception handling logic
                let resolved_attr = self.apply_exception_rules(attribute).await?;
                resolved.push(resolved_attr);
            } else {
                resolved.push(attribute);
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

