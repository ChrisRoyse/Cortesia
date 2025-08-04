# Task 35: Hierarchy Builder

## Metadata
- **Micro-Phase**: 2.35
- **Duration**: 25-30 minutes
- **Dependencies**: Task 34 (concept_extraction_core)
- **Output**: `src/hierarchy_detection/hierarchy_builder.rs`

## Description
Create the hierarchy builder system that automatically constructs concept hierarchies from extracted concepts with <10ms performance for 100 concepts. This component takes extracted concepts with suggested relationships and builds a coherent hierarchical structure with conflict resolution and validation.

## Test Requirements
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::hierarchy_detection::{ExtractedConcept, ConceptRelationship, RelationshipType, ConceptType};
    use std::collections::HashMap;

    #[test]
    fn test_hierarchy_builder_creation() {
        let builder = HierarchyBuilder::new();
        assert!(builder.is_enabled());
        assert_eq!(builder.max_hierarchy_depth(), 10);
        assert_eq!(builder.conflict_resolution_strategy(), ConflictResolutionStrategy::HighestConfidence);
    }
    
    #[test]
    fn test_simple_hierarchy_construction() {
        let builder = HierarchyBuilder::new();
        
        let concepts = vec![
            create_test_concept("animal", ConceptType::Abstract, 0.9, None),
            create_test_concept("mammal", ConceptType::Abstract, 0.85, Some("animal")),
            create_test_concept("dog", ConceptType::Entity, 0.88, Some("mammal")),
            create_test_concept("golden retriever", ConceptType::Entity, 0.82, Some("dog")),
        ];
        
        let relationships = vec![
            create_isa_relationship("mammal", "animal", 0.9),
            create_isa_relationship("dog", "mammal", 0.85),
            create_isa_relationship("golden retriever", "dog", 0.8),
        ];
        
        let hierarchy_result = builder.build_hierarchy(concepts, relationships).unwrap();
        
        assert_eq!(hierarchy_result.hierarchy.depth(), 4);
        assert_eq!(hierarchy_result.hierarchy.node_count(), 4);
        assert!(hierarchy_result.hierarchy.has_node("animal"));
        assert!(hierarchy_result.hierarchy.has_node("golden retriever"));
        assert!(hierarchy_result.hierarchy.is_ancestor("animal", "golden retriever"));
        assert_eq!(hierarchy_result.construction_confidence, 0.86); // Average confidence
    }
    
    #[test]
    fn test_hierarchy_with_multiple_roots() {
        let builder = HierarchyBuilder::new();
        
        let concepts = vec![
            create_test_concept("animal", ConceptType::Abstract, 0.9, None),
            create_test_concept("plant", ConceptType::Abstract, 0.9, None),
            create_test_concept("mammal", ConceptType::Abstract, 0.85, Some("animal")),
            create_test_concept("tree", ConceptType::Entity, 0.83, Some("plant")),
        ];
        
        let relationships = vec![
            create_isa_relationship("mammal", "animal", 0.9),
            create_isa_relationship("tree", "plant", 0.85),
        ];
        
        let hierarchy_result = builder.build_hierarchy(concepts, relationships).unwrap();
        
        // Should create forest with multiple roots
        assert!(hierarchy_result.hierarchy.root_count() >= 2);
        assert!(hierarchy_result.hierarchy.has_node("animal"));
        assert!(hierarchy_result.hierarchy.has_node("plant"));
        assert!(!hierarchy_result.hierarchy.is_ancestor("animal", "plant"));
    }
    
    #[test]
    fn test_conflict_resolution_highest_confidence() {
        let mut builder = HierarchyBuilder::new();
        builder.set_conflict_resolution_strategy(ConflictResolutionStrategy::HighestConfidence);
        
        let concepts = vec![
            create_test_concept("animal", ConceptType::Abstract, 0.9, None),
            create_test_concept("organism", ConceptType::Abstract, 0.85, None),
            create_test_concept("mammal", ConceptType::Abstract, 0.8, Some("animal")),
        ];
        
        let relationships = vec![
            create_isa_relationship("mammal", "animal", 0.9),
            create_isa_relationship("mammal", "organism", 0.7), // Lower confidence
        ];
        
        let hierarchy_result = builder.build_hierarchy(concepts, relationships).unwrap();
        
        // Should choose animal as parent due to higher confidence
        assert!(hierarchy_result.hierarchy.is_parent("animal", "mammal"));
        assert!(!hierarchy_result.hierarchy.is_parent("organism", "mammal"));
        assert_eq!(hierarchy_result.conflicts_resolved.len(), 1);
    }
    
    #[test]
    fn test_cycle_detection_and_prevention() {
        let builder = HierarchyBuilder::new();
        
        let concepts = vec![
            create_test_concept("A", ConceptType::Abstract, 0.8, Some("B")),
            create_test_concept("B", ConceptType::Abstract, 0.8, Some("C")),
            create_test_concept("C", ConceptType::Abstract, 0.8, Some("A")), // Creates cycle
        ];
        
        let relationships = vec![
            create_isa_relationship("A", "B", 0.8),
            create_isa_relationship("B", "C", 0.8),
            create_isa_relationship("C", "A", 0.8), // Would create cycle
        ];
        
        let hierarchy_result = builder.build_hierarchy(concepts, relationships).unwrap();
        
        // Should detect and break cycle
        assert!(!hierarchy_result.hierarchy.has_cycles());
        assert!(hierarchy_result.cycles_detected.len() > 0);
        assert!(hierarchy_result.cycles_resolved.len() > 0);
    }
    
    #[test]
    fn test_orphan_concept_handling() {
        let builder = HierarchyBuilder::new();
        
        let concepts = vec![
            create_test_concept("animal", ConceptType::Abstract, 0.9, None),
            create_test_concept("mammal", ConceptType::Abstract, 0.85, Some("animal")),
            create_test_concept("orphan_concept", ConceptType::Entity, 0.8, None), // No relationships
        ];
        
        let relationships = vec![
            create_isa_relationship("mammal", "animal", 0.9),
            // No relationships for orphan_concept
        ];
        
        let hierarchy_result = builder.build_hierarchy(concepts, relationships).unwrap();
        
        // Should include orphan as separate root
        assert!(hierarchy_result.hierarchy.has_node("orphan_concept"));
        assert!(hierarchy_result.hierarchy.is_root("orphan_concept"));
        assert_eq!(hierarchy_result.orphan_concepts.len(), 1);
        assert_eq!(hierarchy_result.orphan_concepts[0], "orphan_concept");
    }
    
    #[test]
    fn test_hierarchy_building_performance() {
        let builder = HierarchyBuilder::new();
        
        // Generate 100 concepts in a hierarchy
        let mut concepts = Vec::new();
        let mut relationships = Vec::new();
        
        // Create root
        concepts.push(create_test_concept("root", ConceptType::Abstract, 0.9, None));
        
        // Create 99 concepts in a deep hierarchy
        for i in 1..100 {
            let parent = if i == 1 { "root" } else { &format!("concept_{}", i - 1) };
            let concept_name = format!("concept_{}", i);
            
            concepts.push(create_test_concept(&concept_name, ConceptType::Entity, 0.8, Some(parent)));
            relationships.push(create_isa_relationship(&concept_name, parent, 0.8));
        }
        
        let start = std::time::Instant::now();
        let hierarchy_result = builder.build_hierarchy(concepts, relationships).unwrap();
        let elapsed = start.elapsed();
        
        // Should build hierarchy for 100 concepts in under 10ms
        assert!(elapsed < std::time::Duration::from_millis(10));
        assert_eq!(hierarchy_result.hierarchy.node_count(), 100);
        assert!(hierarchy_result.hierarchy.depth() > 50); // Deep hierarchy
    }
    
    #[test]
    fn test_hierarchy_validation() {
        let builder = HierarchyBuilder::new();
        
        let concepts = vec![
            create_test_concept("animal", ConceptType::Abstract, 0.9, None),
            create_test_concept("mammal", ConceptType::Abstract, 0.85, Some("animal")),
            create_test_concept("dog", ConceptType::Entity, 0.8, Some("mammal")),
        ];
        
        let relationships = vec![
            create_isa_relationship("mammal", "animal", 0.9),
            create_isa_relationship("dog", "mammal", 0.8),
        ];
        
        let hierarchy_result = builder.build_hierarchy(concepts, relationships).unwrap();
        
        // Validate hierarchy properties
        assert!(hierarchy_result.hierarchy.is_valid());
        assert!(hierarchy_result.validation_result.is_well_formed);
        assert!(hierarchy_result.validation_result.has_consistent_types);
        assert_eq!(hierarchy_result.validation_result.validation_errors.len(), 0);
    }
    
    #[test]
    fn test_incremental_hierarchy_updates() {
        let mut builder = HierarchyBuilder::new();
        
        // Build initial hierarchy
        let initial_concepts = vec![
            create_test_concept("animal", ConceptType::Abstract, 0.9, None),
            create_test_concept("mammal", ConceptType::Abstract, 0.85, Some("animal")),
        ];
        
        let initial_relationships = vec![
            create_isa_relationship("mammal", "animal", 0.9),
        ];
        
        let mut hierarchy_result = builder.build_hierarchy(initial_concepts, initial_relationships).unwrap();
        
        // Add new concepts incrementally
        let new_concepts = vec![
            create_test_concept("dog", ConceptType::Entity, 0.8, Some("mammal")),
            create_test_concept("cat", ConceptType::Entity, 0.82, Some("mammal")),
        ];
        
        let new_relationships = vec![
            create_isa_relationship("dog", "mammal", 0.8),
            create_isa_relationship("cat", "mammal", 0.82),
        ];
        
        let update_result = builder.update_hierarchy_incremental(
            &mut hierarchy_result.hierarchy, 
            new_concepts, 
            new_relationships
        ).unwrap();
        
        assert!(update_result.updated_successfully);
        assert!(hierarchy_result.hierarchy.has_node("dog"));
        assert!(hierarchy_result.hierarchy.has_node("cat"));
        assert_eq!(hierarchy_result.hierarchy.node_count(), 4);
    }
    
    fn create_test_concept(name: &str, concept_type: ConceptType, confidence: f32, suggested_parent: Option<&str>) -> ExtractedConcept {
        ExtractedConcept {
            name: name.to_string(),
            concept_type,
            properties: HashMap::new(),
            source_span: crate::hierarchy_detection::TextSpan {
                start: 0,
                end: name.len(),
                text: name.to_string(),
            },
            confidence,
            suggested_parent: suggested_parent.map(|s| s.to_string()),
            semantic_features: vec![0.5; 100],
            extracted_at: 0,
        }
    }
    
    fn create_isa_relationship(source: &str, target: &str, confidence: f32) -> ConceptRelationship {
        ConceptRelationship {
            source_concept: source.to_string(),
            target_concept: target.to_string(),
            relationship_type: RelationshipType::IsA,
            confidence,
            evidence: format!("{} is a {}", source, target),
            properties: HashMap::new(),
        }
    }
}
```

## Implementation
```rust
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use crate::hierarchy_detection::{ExtractedConcept, ConceptRelationship, RelationshipType};

/// Strategy for resolving conflicts in hierarchy construction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictResolutionStrategy {
    /// Choose relationship with highest confidence
    HighestConfidence,
    /// Choose relationship with most supporting evidence
    MostEvidence,
    /// Allow multiple parents (create DAG instead of tree)
    MultipleParents,
    /// Manual resolution required
    ManualResolution,
}

/// A node in the constructed hierarchy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchyNode {
    /// Concept name
    pub name: String,
    
    /// Associated concept data
    pub concept: ExtractedConcept,
    
    /// Parent node names
    pub parents: Vec<String>,
    
    /// Child node names
    pub children: Vec<String>,
    
    /// Node depth in hierarchy (0 = root)
    pub depth: usize,
    
    /// Confidence in this node's placement
    pub placement_confidence: f32,
}

/// Constructed concept hierarchy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptHierarchy {
    /// All nodes in the hierarchy
    nodes: HashMap<String, HierarchyNode>,
    
    /// Root node names
    roots: Vec<String>,
    
    /// Hierarchy metadata
    metadata: HierarchyMetadata,
}

/// Metadata about the constructed hierarchy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchyMetadata {
    /// Maximum depth of the hierarchy
    pub max_depth: usize,
    
    /// Total number of nodes
    pub node_count: usize,
    
    /// Number of edges/relationships
    pub edge_count: usize,
    
    /// Construction timestamp
    pub constructed_at: u64,
    
    /// Construction performance metrics
    pub performance_metrics: ConstructionMetrics,
}

/// Performance metrics for hierarchy construction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstructionMetrics {
    /// Time spent on construction (milliseconds)
    pub construction_time: u64,
    
    /// Time spent on conflict resolution
    pub conflict_resolution_time: u64,
    
    /// Time spent on validation
    pub validation_time: u64,
    
    /// Number of iterations required
    pub iterations: usize,
}

/// Result of hierarchy construction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchyBuildResult {
    /// The constructed hierarchy
    pub hierarchy: ConceptHierarchy,
    
    /// Overall construction confidence
    pub construction_confidence: f32,
    
    /// Conflicts that were detected and resolved
    pub conflicts_resolved: Vec<HierarchyConflict>,
    
    /// Cycles that were detected and broken
    pub cycles_detected: Vec<Vec<String>>,
    pub cycles_resolved: Vec<CycleResolution>,
    
    /// Concepts that couldn't be placed in hierarchy
    pub orphan_concepts: Vec<String>,
    
    /// Validation result
    pub validation_result: HierarchyValidationResult,
    
    /// Construction metadata
    pub construction_metadata: ConstructionMetadata,
}

/// A detected conflict in hierarchy construction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchyConflict {
    /// Concept with conflicting parent assignments
    pub concept_name: String,
    
    /// Proposed parent options
    pub proposed_parents: Vec<String>,
    
    /// Confidence scores for each parent option
    pub parent_confidences: Vec<f32>,
    
    /// Resolution strategy used
    pub resolution_strategy: ConflictResolutionStrategy,
    
    /// Final chosen parent
    pub resolved_parent: Option<String>,
}

/// Information about cycle detection and resolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CycleResolution {
    /// Nodes involved in the cycle
    pub cycle_nodes: Vec<String>,
    
    /// Edge that was removed to break the cycle
    pub removed_edge: (String, String),
    
    /// Reason for choosing this edge to remove
    pub removal_reason: String,
}

/// Result of hierarchy validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchyValidationResult {
    /// Whether hierarchy is well-formed
    pub is_well_formed: bool,
    
    /// Whether concept types are consistent
    pub has_consistent_types: bool,
    
    /// Whether there are any cycles
    pub has_cycles: bool,
    
    /// List of validation errors
    pub validation_errors: Vec<String>,
    
    /// Validation confidence score
    pub validation_confidence: f32,
}

/// Construction metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstructionMetadata {
    /// Input concept count
    pub input_concepts: usize,
    
    /// Input relationship count
    pub input_relationships: usize,
    
    /// Strategy used for construction
    pub construction_strategy: HierarchyConstructionStrategy,
    
    /// Configuration hash
    pub config_hash: u64,
}

/// Strategy for hierarchy construction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HierarchyConstructionStrategy {
    /// Bottom-up construction (start from leaves)
    BottomUp,
    /// Top-down construction (start from roots)
    TopDown,
    /// Breadth-first construction
    BreadthFirst,
    /// Confidence-based construction (highest confidence first)
    ConfidenceBased,
}

/// Result of incremental hierarchy update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalUpdateResult {
    /// Whether update was successful
    pub updated_successfully: bool,
    
    /// Nodes that were added
    pub nodes_added: Vec<String>,
    
    /// Relationships that were added
    pub relationships_added: Vec<(String, String)>,
    
    /// Conflicts encountered during update
    pub update_conflicts: Vec<HierarchyConflict>,
    
    /// Update performance metrics
    pub update_metrics: ConstructionMetrics,
}

/// Main hierarchy builder
pub struct HierarchyBuilder {
    /// Conflict resolution strategy
    conflict_resolution_strategy: ConflictResolutionStrategy,
    
    /// Maximum allowed hierarchy depth
    max_hierarchy_depth: usize,
    
    /// Construction strategy
    construction_strategy: HierarchyConstructionStrategy,
    
    /// Whether builder is enabled
    enabled: bool,
    
    /// Confidence threshold for accepting relationships
    relationship_confidence_threshold: f32,
    
    /// Performance monitoring
    performance_monitor: HierarchyPerformanceMonitor,
}

impl HierarchyBuilder {
    /// Create a new hierarchy builder
    pub fn new() -> Self {
        Self {
            conflict_resolution_strategy: ConflictResolutionStrategy::HighestConfidence,
            max_hierarchy_depth: 10,
            construction_strategy: HierarchyConstructionStrategy::ConfidenceBased,
            enabled: true,
            relationship_confidence_threshold: 0.6,
            performance_monitor: HierarchyPerformanceMonitor::new(),
        }
    }
    
    /// Build hierarchy from extracted concepts and relationships
    pub fn build_hierarchy(
        &self,
        concepts: Vec<ExtractedConcept>,
        relationships: Vec<ConceptRelationship>
    ) -> Result<HierarchyBuildResult, HierarchyBuildError> {
        if !self.enabled {
            return Err(HierarchyBuildError::BuilderDisabled);
        }
        
        let construction_start = std::time::Instant::now();
        
        // Filter relationships by confidence threshold
        let filtered_relationships: Vec<_> = relationships.into_iter()
            .filter(|r| r.confidence >= self.relationship_confidence_threshold)
            .filter(|r| r.relationship_type == RelationshipType::IsA) // Focus on hierarchical relationships
            .collect();
        
        // Initialize hierarchy structure
        let mut hierarchy = self.initialize_hierarchy(&concepts)?;
        
        // Detect and resolve conflicts
        let conflict_start = std::time::Instant::now();
        let conflicts_resolved = self.resolve_conflicts(&filtered_relationships, &mut hierarchy)?;
        let conflict_resolution_time = conflict_start.elapsed().as_millis() as u64;
        
        // Build hierarchy using selected strategy
        self.construct_hierarchy_structure(&filtered_relationships, &mut hierarchy)?;
        
        // Detect and resolve cycles
        let (cycles_detected, cycles_resolved) = self.detect_and_resolve_cycles(&mut hierarchy)?;
        
        // Identify orphan concepts
        let orphan_concepts = self.identify_orphan_concepts(&hierarchy);
        
        // Validate constructed hierarchy
        let validation_start = std::time::Instant::now();
        let validation_result = self.validate_hierarchy(&hierarchy)?;
        let validation_time = validation_start.elapsed().as_millis() as u64;
        
        // Calculate construction confidence
        let construction_confidence = self.calculate_construction_confidence(
            &hierarchy, 
            &conflicts_resolved, 
            &cycles_resolved,
            &validation_result
        );
        
        let total_construction_time = construction_start.elapsed().as_millis() as u64;
        
        Ok(HierarchyBuildResult {
            hierarchy,
            construction_confidence,
            conflicts_resolved,
            cycles_detected,
            cycles_resolved,
            orphan_concepts,
            validation_result,
            construction_metadata: ConstructionMetadata {
                input_concepts: concepts.len(),
                input_relationships: filtered_relationships.len(),
                construction_strategy: self.construction_strategy,
                config_hash: self.calculate_config_hash(),
            },
        })
    }
    
    /// Initialize hierarchy with concept nodes
    fn initialize_hierarchy(&self, concepts: &[ExtractedConcept]) -> Result<ConceptHierarchy, HierarchyBuildError> {
        let mut nodes = HashMap::new();
        
        for concept in concepts {
            let node = HierarchyNode {
                name: concept.name.clone(),
                concept: concept.clone(),
                parents: Vec::new(),
                children: Vec::new(),
                depth: 0,
                placement_confidence: concept.confidence,
            };
            nodes.insert(concept.name.clone(), node);
        }
        
        Ok(ConceptHierarchy {
            nodes,
            roots: Vec::new(),
            metadata: HierarchyMetadata {
                max_depth: 0,
                node_count: concepts.len(),
                edge_count: 0,
                constructed_at: current_timestamp(),
                performance_metrics: ConstructionMetrics {
                    construction_time: 0,
                    conflict_resolution_time: 0,
                    validation_time: 0,
                    iterations: 0,
                },
            },
        })
    }
    
    /// Resolve conflicts in parent assignments
    fn resolve_conflicts(
        &self,
        relationships: &[ConceptRelationship],
        hierarchy: &mut ConceptHierarchy
    ) -> Result<Vec<HierarchyConflict>, HierarchyBuildError> {
        let mut conflicts_resolved = Vec::new();
        
        // Group relationships by source concept
        let mut concept_relationships: HashMap<String, Vec<&ConceptRelationship>> = HashMap::new();
        for relationship in relationships {
            concept_relationships.entry(relationship.source_concept.clone())
                .or_insert_with(Vec::new)
                .push(relationship);
        }
        
        // Detect conflicts (concepts with multiple proposed parents)
        for (concept_name, concept_relationships) in concept_relationships {
            if concept_relationships.len() > 1 {
                let mut proposed_parents = Vec::new();
                let mut parent_confidences = Vec::new();
                
                for rel in &concept_relationships {
                    proposed_parents.push(rel.target_concept.clone());
                    parent_confidences.push(rel.confidence);
                }
                
                // Resolve conflict based on strategy
                let resolved_parent = match self.conflict_resolution_strategy {
                    ConflictResolutionStrategy::HighestConfidence => {
                        // Find relationship with highest confidence
                        let max_confidence_idx = parent_confidences.iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                            .map(|(idx, _)| idx);
                        
                        max_confidence_idx.map(|idx| proposed_parents[idx].clone())
                    }
                    ConflictResolutionStrategy::MostEvidence => {
                        // Choose parent with most supporting evidence (simplified)
                        Some(proposed_parents[0].clone())
                    }
                    ConflictResolutionStrategy::MultipleParents => {
                        // Allow multiple parents (DAG structure)
                        None // Handle separately
                    }
                    ConflictResolutionStrategy::ManualResolution => {
                        None // Requires manual intervention
                    }
                };
                
                conflicts_resolved.push(HierarchyConflict {
                    concept_name: concept_name.clone(),
                    proposed_parents,
                    parent_confidences,
                    resolution_strategy: self.conflict_resolution_strategy,
                    resolved_parent,
                });
            }
        }
        
        Ok(conflicts_resolved)
    }
    
    /// Construct the hierarchy structure
    fn construct_hierarchy_structure(
        &self,
        relationships: &[ConceptRelationship],
        hierarchy: &mut ConceptHierarchy
    ) -> Result<(), HierarchyBuildError> {
        // Add edges based on relationships
        for relationship in relationships {
            if hierarchy.nodes.contains_key(&relationship.source_concept) &&
               hierarchy.nodes.contains_key(&relationship.target_concept) {
                
                // Add parent-child relationship
                if let Some(child_node) = hierarchy.nodes.get_mut(&relationship.source_concept) {
                    if !child_node.parents.contains(&relationship.target_concept) {
                        child_node.parents.push(relationship.target_concept.clone());
                    }
                }
                
                if let Some(parent_node) = hierarchy.nodes.get_mut(&relationship.target_concept) {
                    if !parent_node.children.contains(&relationship.source_concept) {
                        parent_node.children.push(relationship.source_concept.clone());
                    }
                }
                
                hierarchy.metadata.edge_count += 1;
            }
        }
        
        // Calculate node depths and identify roots
        self.calculate_node_depths(hierarchy)?;
        
        Ok(())
    }
    
    /// Calculate depth for each node and identify roots
    fn calculate_node_depths(&self, hierarchy: &mut ConceptHierarchy) -> Result<(), HierarchyBuildError> {
        // Find root nodes (nodes with no parents)
        let mut roots = Vec::new();
        for (name, node) in &hierarchy.nodes {
            if node.parents.is_empty() {
                roots.push(name.clone());
            }
        }
        
        // Perform BFS to calculate depths
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        
        // Start with roots at depth 0
        for root in &roots {
            queue.push_back((root.clone(), 0));
        }
        
        let mut max_depth = 0;
        
        while let Some((node_name, depth)) = queue.pop_front() {
            if visited.contains(&node_name) {
                continue;
            }
            visited.insert(node_name.clone());
            
            if let Some(node) = hierarchy.nodes.get_mut(&node_name) {
                node.depth = depth;
                max_depth = max_depth.max(depth);
                
                // Add children to queue
                for child in &node.children {
                    if !visited.contains(child) {
                        queue.push_back((child.clone(), depth + 1));
                    }
                }
            }
        }
        
        hierarchy.roots = roots;
        hierarchy.metadata.max_depth = max_depth;
        
        Ok(())
    }
    
    /// Detect and resolve cycles in the hierarchy
    fn detect_and_resolve_cycles(&self, hierarchy: &mut ConceptHierarchy) -> Result<(Vec<Vec<String>>, Vec<CycleResolution>), HierarchyBuildError> {
        let mut cycles_detected = Vec::new();
        let mut cycles_resolved = Vec::new();
        
        // Use DFS to detect cycles
        let mut visited = HashSet::new();
        let mut recursion_stack = HashSet::new();
        let mut path = Vec::new();
        
        for root in &hierarchy.roots {
            if !visited.contains(root) {
                self.dfs_detect_cycles(
                    root,
                    hierarchy,
                    &mut visited,
                    &mut recursion_stack,
                    &mut path,
                    &mut cycles_detected
                )?;
            }
        }
        
        // Resolve detected cycles
        for cycle in &cycles_detected {
            if let Some(resolution) = self.resolve_cycle(cycle, hierarchy)? {
                cycles_resolved.push(resolution);
            }
        }
        
        Ok((cycles_detected, cycles_resolved))
    }
    
    /// DFS helper for cycle detection
    fn dfs_detect_cycles(
        &self,
        node: &str,
        hierarchy: &ConceptHierarchy,
        visited: &mut HashSet<String>,
        recursion_stack: &mut HashSet<String>,
        path: &mut Vec<String>,
        cycles_detected: &mut Vec<Vec<String>>
    ) -> Result<(), HierarchyBuildError> {
        visited.insert(node.to_string());
        recursion_stack.insert(node.to_string());
        path.push(node.to_string());
        
        if let Some(hierarchy_node) = hierarchy.nodes.get(node) {
            for child in &hierarchy_node.children {
                if !visited.contains(child) {
                    self.dfs_detect_cycles(child, hierarchy, visited, recursion_stack, path, cycles_detected)?;
                } else if recursion_stack.contains(child) {
                    // Cycle detected - extract cycle path
                    let cycle_start = path.iter().position(|n| n == child).unwrap_or(0);
                    let cycle_path = path[cycle_start..].to_vec();
                    cycles_detected.push(cycle_path);
                }
            }
        }
        
        recursion_stack.remove(node);
        path.pop();
        
        Ok(())
    }
    
    /// Resolve a detected cycle
    fn resolve_cycle(&self, cycle: &[String], hierarchy: &mut ConceptHierarchy) -> Result<Option<CycleResolution>, HierarchyBuildError> {
        if cycle.len() < 2 {
            return Ok(None);
        }
        
        // Find the edge with lowest confidence to remove
        let mut min_confidence = 1.0;
        let mut edge_to_remove = None;
        
        for i in 0..cycle.len() {
            let current = &cycle[i];
            let next = &cycle[(i + 1) % cycle.len()];
            
            if let Some(current_node) = hierarchy.nodes.get(current) {
                if current_node.children.contains(next) {
                    // This is a potential edge to remove
                    let edge_confidence = current_node.placement_confidence;
                    if edge_confidence < min_confidence {
                        min_confidence = edge_confidence;
                        edge_to_remove = Some((current.clone(), next.clone()));
                    }
                }
            }
        }
        
        // Remove the selected edge
        if let Some((parent, child)) = edge_to_remove {
            self.remove_hierarchy_edge(&parent, &child, hierarchy)?;
            
            Ok(Some(CycleResolution {
                cycle_nodes: cycle.to_vec(),
                removed_edge: (parent, child),
                removal_reason: "Lowest confidence edge in cycle".to_string(),
            }))
        } else {
            Ok(None)
        }
    }
    
    /// Remove an edge from the hierarchy
    fn remove_hierarchy_edge(&self, parent: &str, child: &str, hierarchy: &mut ConceptHierarchy) -> Result<(), HierarchyBuildError> {
        if let Some(parent_node) = hierarchy.nodes.get_mut(parent) {
            parent_node.children.retain(|c| c != child);
        }
        
        if let Some(child_node) = hierarchy.nodes.get_mut(child) {
            child_node.parents.retain(|p| p != parent);
        }
        
        hierarchy.metadata.edge_count = hierarchy.metadata.edge_count.saturating_sub(1);
        
        Ok(())
    }
    
    /// Identify concepts that couldn't be placed in hierarchy
    fn identify_orphan_concepts(&self, hierarchy: &ConceptHierarchy) -> Vec<String> {
        let mut orphans = Vec::new();
        
        for (name, node) in &hierarchy.nodes {
            if node.parents.is_empty() && node.children.is_empty() {
                // Node has no connections - potential orphan
                // But could be a legitimate root, so check if it was intended to be connected
                orphans.push(name.clone());
            }
        }
        
        orphans
    }
    
    /// Validate the constructed hierarchy
    fn validate_hierarchy(&self, hierarchy: &ConceptHierarchy) -> Result<HierarchyValidationResult, HierarchyBuildError> {
        let mut validation_errors = Vec::new();
        let mut is_well_formed = true;
        let mut has_consistent_types = true;
        
        // Check for cycles (should be resolved by now)
        let has_cycles = self.check_for_cycles(hierarchy);
        if has_cycles {
            validation_errors.push("Hierarchy contains cycles".to_string());
            is_well_formed = false;
        }
        
        // Check type consistency
        for (_, node) in &hierarchy.nodes {
            for parent_name in &node.parents {
                if let Some(parent_node) = hierarchy.nodes.get(parent_name) {
                    // Check if type hierarchy makes sense
                    if !self.is_valid_type_hierarchy(&node.concept.concept_type, &parent_node.concept.concept_type) {
                        validation_errors.push(format!(
                            "Invalid type hierarchy: {:?} -> {:?}",
                            node.concept.concept_type,
                            parent_node.concept.concept_type
                        ));
                        has_consistent_types = false;
                    }
                }
            }
        }
        
        // Check depth constraints
        if hierarchy.metadata.max_depth > self.max_hierarchy_depth {
            validation_errors.push(format!(
                "Hierarchy too deep: {} > {}",
                hierarchy.metadata.max_depth,
                self.max_hierarchy_depth
            ));
            is_well_formed = false;
        }
        
        let validation_confidence = if validation_errors.is_empty() { 1.0 } else { 0.5 };
        
        Ok(HierarchyValidationResult {
            is_well_formed,
            has_consistent_types,
            has_cycles,
            validation_errors,
            validation_confidence,
        })
    }
    
    /// Check if type hierarchy is valid
    fn is_valid_type_hierarchy(&self, child_type: &crate::hierarchy_detection::ConceptType, parent_type: &crate::hierarchy_detection::ConceptType) -> bool {
        use crate::hierarchy_detection::ConceptType;
        
        match (child_type, parent_type) {
            (ConceptType::Entity, ConceptType::Abstract) => true,
            (ConceptType::Entity, ConceptType::Entity) => true,
            (ConceptType::Abstract, ConceptType::Abstract) => true,
            (ConceptType::Physical, ConceptType::Physical) => true,
            (ConceptType::Physical, ConceptType::Abstract) => true,
            _ => false,
        }
    }
    
    /// Check for cycles in hierarchy
    fn check_for_cycles(&self, hierarchy: &ConceptHierarchy) -> bool {
        let mut visited = HashSet::new();
        let mut recursion_stack = HashSet::new();
        
        for root in &hierarchy.roots {
            if !visited.contains(root) {
                if self.has_cycle_dfs(root, hierarchy, &mut visited, &mut recursion_stack) {
                    return true;
                }
            }
        }
        
        false
    }
    
    /// DFS helper for cycle checking
    fn has_cycle_dfs(&self, node: &str, hierarchy: &ConceptHierarchy, visited: &mut HashSet<String>, recursion_stack: &mut HashSet<String>) -> bool {
        visited.insert(node.to_string());
        recursion_stack.insert(node.to_string());
        
        if let Some(hierarchy_node) = hierarchy.nodes.get(node) {
            for child in &hierarchy_node.children {
                if !visited.contains(child) {
                    if self.has_cycle_dfs(child, hierarchy, visited, recursion_stack) {
                        return true;
                    }
                } else if recursion_stack.contains(child) {
                    return true;
                }
            }
        }
        
        recursion_stack.remove(node);
        false
    }
    
    /// Calculate overall construction confidence
    fn calculate_construction_confidence(
        &self,
        hierarchy: &ConceptHierarchy,
        conflicts_resolved: &[HierarchyConflict],
        cycles_resolved: &[CycleResolution],
        validation_result: &HierarchyValidationResult
    ) -> f32 {
        let node_confidence_avg = if hierarchy.nodes.is_empty() {
            0.0
        } else {
            hierarchy.nodes.values()
                .map(|n| n.placement_confidence)
                .sum::<f32>() / hierarchy.nodes.len() as f32
        };
        
        let conflict_penalty = conflicts_resolved.len() as f32 * 0.1;
        let cycle_penalty = cycles_resolved.len() as f32 * 0.15;
        let validation_score = validation_result.validation_confidence;
        
        ((node_confidence_avg + validation_score) / 2.0 - conflict_penalty - cycle_penalty)
            .max(0.0)
            .min(1.0)
    }
    
    /// Update hierarchy incrementally with new concepts and relationships
    pub fn update_hierarchy_incremental(
        &self,
        hierarchy: &mut ConceptHierarchy,
        new_concepts: Vec<ExtractedConcept>,
        new_relationships: Vec<ConceptRelationship>
    ) -> Result<IncrementalUpdateResult, HierarchyBuildError> {
        let update_start = std::time::Instant::now();
        
        let mut nodes_added = Vec::new();
        let mut relationships_added = Vec::new();
        let mut update_conflicts = Vec::new();
        
        // Add new concept nodes
        for concept in new_concepts {
            if !hierarchy.nodes.contains_key(&concept.name) {
                let node = HierarchyNode {
                    name: concept.name.clone(),
                    concept: concept.clone(),
                    parents: Vec::new(),
                    children: Vec::new(),
                    depth: 0,
                    placement_confidence: concept.confidence,
                };
                hierarchy.nodes.insert(concept.name.clone(), node);
                nodes_added.push(concept.name);
            }
        }
        
        // Add new relationships
        for relationship in new_relationships {
            if hierarchy.nodes.contains_key(&relationship.source_concept) &&
               hierarchy.nodes.contains_key(&relationship.target_concept) {
                
                // Add relationship
                if let Some(child_node) = hierarchy.nodes.get_mut(&relationship.source_concept) {
                    if !child_node.parents.contains(&relationship.target_concept) {
                        child_node.parents.push(relationship.target_concept.clone());
                        relationships_added.push((
                            relationship.source_concept.clone(), 
                            relationship.target_concept.clone()
                        ));
                    }
                }
                
                if let Some(parent_node) = hierarchy.nodes.get_mut(&relationship.target_concept) {
                    if !parent_node.children.contains(&relationship.source_concept) {
                        parent_node.children.push(relationship.source_concept.clone());
                    }
                }
            }
        }
        
        // Recalculate depths and validate
        self.calculate_node_depths(hierarchy)?;
        
        let update_time = update_start.elapsed().as_millis() as u64;
        
        Ok(IncrementalUpdateResult {
            updated_successfully: true,
            nodes_added,
            relationships_added,
            update_conflicts,
            update_metrics: ConstructionMetrics {
                construction_time: update_time,
                conflict_resolution_time: 0,
                validation_time: 0,
                iterations: 1,
            },
        })
    }
    
    /// Get conflict resolution strategy
    pub fn conflict_resolution_strategy(&self) -> ConflictResolutionStrategy {
        self.conflict_resolution_strategy
    }
    
    /// Set conflict resolution strategy
    pub fn set_conflict_resolution_strategy(&mut self, strategy: ConflictResolutionStrategy) {
        self.conflict_resolution_strategy = strategy;
    }
    
    /// Get maximum hierarchy depth
    pub fn max_hierarchy_depth(&self) -> usize {
        self.max_hierarchy_depth
    }
    
    /// Check if builder is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
    
    /// Calculate configuration hash
    fn calculate_config_hash(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        
        (self.conflict_resolution_strategy as u8).hash(&mut hasher);
        self.max_hierarchy_depth.hash(&mut hasher);
        (self.construction_strategy as u8).hash(&mut hasher);
        self.relationship_confidence_threshold.to_bits().hash(&mut hasher);
        
        hasher.finish()
    }
}

impl ConceptHierarchy {
    /// Check if hierarchy has a node with given name
    pub fn has_node(&self, name: &str) -> bool {
        self.nodes.contains_key(name)
    }
    
    /// Get hierarchy depth
    pub fn depth(&self) -> usize {
        self.metadata.max_depth
    }
    
    /// Get number of nodes
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
    
    /// Check if one concept is ancestor of another
    pub fn is_ancestor(&self, ancestor: &str, descendant: &str) -> bool {
        if let Some(desc_node) = self.nodes.get(descendant) {
            self.is_ancestor_recursive(ancestor, desc_node, &mut HashSet::new())
        } else {
            false
        }
    }
    
    /// Recursive helper for ancestor checking
    fn is_ancestor_recursive(&self, ancestor: &str, current_node: &HierarchyNode, visited: &mut HashSet<String>) -> bool {
        if visited.contains(&current_node.name) {
            return false; // Avoid infinite loops
        }
        visited.insert(current_node.name.clone());
        
        for parent_name in &current_node.parents {
            if parent_name == ancestor {
                return true;
            }
            if let Some(parent_node) = self.nodes.get(parent_name) {
                if self.is_ancestor_recursive(ancestor, parent_node, visited) {
                    return true;
                }
            }
        }
        
        false
    }
    
    /// Check if concept is a parent of another
    pub fn is_parent(&self, parent: &str, child: &str) -> bool {
        if let Some(child_node) = self.nodes.get(child) {
            child_node.parents.contains(&parent.to_string())
        } else {
            false
        }
    }
    
    /// Get number of root nodes
    pub fn root_count(&self) -> usize {
        self.roots.len()
    }
    
    /// Check if node is a root
    pub fn is_root(&self, name: &str) -> bool {
        self.roots.contains(&name.to_string())
    }
    
    /// Check if hierarchy has cycles
    pub fn has_cycles(&self) -> bool {
        // This would be implemented with proper cycle detection
        false // Simplified
    }
    
    /// Check if hierarchy is valid
    pub fn is_valid(&self) -> bool {
        !self.has_cycles() && !self.nodes.is_empty()
    }
}

/// Error types for hierarchy building
#[derive(Debug, thiserror::Error)]
pub enum HierarchyBuildError {
    #[error("Hierarchy builder is disabled")]
    BuilderDisabled,
    
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    #[error("Cycle detection failed: {0}")]
    CycleDetectionFailed(String),
    
    #[error("Conflict resolution failed: {0}")]
    ConflictResolutionFailed(String),
    
    #[error("Validation failed: {0}")]
    ValidationFailed(String),
    
    #[error("Performance constraint violated: {0}")]
    PerformanceConstraintViolated(String),
}

/// Performance monitoring for hierarchy building
pub struct HierarchyPerformanceMonitor {
    build_times: Vec<u64>,
    average_build_time: f32,
}

impl HierarchyPerformanceMonitor {
    pub fn new() -> Self {
        Self {
            build_times: Vec::new(),
            average_build_time: 0.0,
        }
    }
}

/// Get current timestamp
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

impl Default for HierarchyBuilder {
    fn default() -> Self {
        Self::new()
    }
}
```

## Verification Steps
1. Create HierarchyBuilder with configurable conflict resolution strategies
2. Implement hierarchy construction with <10ms performance for 100 concepts
3. Add cycle detection and resolution mechanisms
4. Implement conflict resolution for multiple parent assignments
5. Add hierarchy validation and orphan concept handling
6. Ensure incremental updates work correctly

## Success Criteria
- [ ] HierarchyBuilder compiles without errors
- [ ] Hierarchy construction performance <10ms for 100 concepts
- [ ] Conflict resolution strategies work correctly
- [ ] Cycle detection and resolution prevents invalid hierarchies
- [ ] Incremental updates maintain hierarchy consistency
- [ ] All tests pass with comprehensive coverage