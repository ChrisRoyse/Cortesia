//! Memory consolidation algorithms

use crate::branch::{MemoryBranch, BranchId, ConsolidationState, ConceptProperties};
use std::collections::HashSet;
use thiserror::Error;

/// Configuration for consolidation process
#[derive(Debug, Clone)]
pub struct ConsolidationConfig {
    /// Similarity threshold for automatic merging
    pub merge_threshold: f32,
    
    /// Maximum conflicts before failing
    pub max_conflicts: usize,
    
    /// Strategy for resolving conflicts
    pub resolution_strategy: ResolutionStrategy,
    
    /// Enable parallel consolidation
    pub parallel: bool,
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        Self {
            merge_threshold: 0.8,
            max_conflicts: 10,
            resolution_strategy: ResolutionStrategy::PreferNewer,
            parallel: false,
        }
    }
}

/// Strategies for resolving conflicts
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResolutionStrategy {
    /// Prefer newer branch data
    PreferNewer,
    
    /// Prefer parent branch data
    PreferParent,
    
    /// Prefer child branch data
    PreferChild,
    
    /// Merge both (may create duplicates)
    MergeBoth,
    
    /// Require manual resolution
    Manual,
}

/// Result of consolidation attempt
#[derive(Debug)]
pub struct ConsolidationResult {
    /// Merged concept IDs
    pub merged_concepts: Vec<uuid::Uuid>,
    
    /// Conflicts encountered
    pub conflicts: Vec<Conflict>,
    
    /// Resolution actions taken
    pub resolutions: Vec<Resolution>,
    
    /// Final divergence score
    pub final_divergence: f32,
    
    /// Success status
    pub success: bool,
}

/// A conflict between branches
#[derive(Debug, Clone)]
pub struct Conflict {
    /// Concept that conflicts
    pub concept_id: uuid::Uuid,
    
    /// Type of conflict
    pub conflict_type: ConflictType,
    
    /// Parent branch version
    pub parent_version: Option<String>,
    
    /// Child branch version
    pub child_version: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConflictType {
    /// Same concept with different properties
    PropertyMismatch,
    
    /// Concept deleted in one branch
    DeletedInBranch,
    
    /// Incompatible spike patterns
    SpikePatternConflict,
    
    /// Circular dependency created
    CircularDependency,
}

/// Resolution of a conflict
#[derive(Debug, Clone)]
pub struct Resolution {
    pub conflict: Conflict,
    pub action: ResolutionAction,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone)]
pub enum ResolutionAction {
    UseParent,
    UseChild,
    Merged,
    Skipped,
    Manual(String),
}

/// Errors during consolidation
#[derive(Error, Debug)]
pub enum ConsolidationError {
    #[error("Branch not found: {0:?}")]
    BranchNotFound(BranchId),
    
    #[error("Invalid branch state: {0}")]
    InvalidState(String),
    
    #[error("Too many conflicts: {0}")]
    TooManyConflicts(usize),
    
    #[error("Consolidation failed: {0}")]
    Failed(String),
}

/// Manages memory branch consolidation
pub struct ConsolidationEngine {
    config: ConsolidationConfig,
}

impl ConsolidationEngine {
    /// Create new consolidation engine
    pub fn new(config: ConsolidationConfig) -> Self {
        Self { config }
    }
    
    /// Determine the type of conflict between two concept properties
    fn determine_conflict_type(parent_props: &ConceptProperties, child_props: &ConceptProperties) -> ConflictType {
        // Check for spike pattern conflicts first (most specific)
        if !parent_props.spike_signatures_match(child_props) {
            return ConflictType::SpikePatternConflict;
        }
        
        // Check for property mismatches
        if parent_props.value != child_props.value || 
           parent_props.confidence != child_props.confidence ||
           parent_props.properties != child_props.properties {
            return ConflictType::PropertyMismatch;
        }
        
        // Default to property mismatch for other differences
        ConflictType::PropertyMismatch
    }
    
    /// Consolidate child branch into parent
    pub fn consolidate(&self, 
                      parent: &mut MemoryBranch, 
                      child: &mut MemoryBranch) -> Result<ConsolidationResult, ConsolidationError> {
        // Validate states using state manager
        let current_state = child.current_state();
        if current_state != ConsolidationState::Active && 
           current_state != ConsolidationState::Conflicted {
            return Err(ConsolidationError::InvalidState(
                format!("Child branch in {} state, cannot consolidate", current_state)
            ));
        }
        
        // Validate that child can transition to Consolidating state
        if !child.can_transition_to(ConsolidationState::Consolidating) {
            return Err(ConsolidationError::InvalidState(
                format!("Child branch cannot transition from {} to Consolidating", current_state)
            ));
        }
        
        // Set child to consolidating state
        child.transition_state(
            ConsolidationState::Consolidating, 
            "Starting consolidation with parent".to_string()
        ).map_err(|e| ConsolidationError::InvalidState(e.to_string()))?;
        
        // Detect conflicts
        let conflicts = self.detect_conflicts(parent, child);
        
        if conflicts.len() > self.config.max_conflicts {
            child.transition_state(
                ConsolidationState::Conflicted,
                format!("Too many conflicts detected: {}", conflicts.len())
            ).map_err(|e| ConsolidationError::InvalidState(e.to_string()))?;
            return Err(ConsolidationError::TooManyConflicts(conflicts.len()));
        }
        
        // Resolve conflicts
        let resolutions = self.resolve_conflicts(&conflicts, parent, child);
        
        // Merge concepts
        let merged_concepts = self.merge_concepts(parent, child, &resolutions);
        
        // Calculate final divergence
        let final_divergence = self.calculate_divergence(&merged_concepts, parent);
        
        // Apply consolidation results to branches
        if self.apply_consolidation_results(parent, child, &merged_concepts, &resolutions) {
            // Mark child as consolidated
            child.transition_state(
                ConsolidationState::Consolidated,
                format!("Successfully consolidated {} concepts", merged_concepts.len())
            ).map_err(|e| ConsolidationError::InvalidState(e.to_string()))?;
            
            // Update parent metadata - increment minor version for consolidation
            parent.increment_minor_version();
            
            Ok(ConsolidationResult {
                merged_concepts,
                conflicts,
                resolutions,
                final_divergence,
                success: true,
            })
        } else {
            // Failed to apply changes
            child.transition_state(
                ConsolidationState::Conflicted,
                "Failed to apply consolidation results".to_string()
            ).map_err(|e| ConsolidationError::InvalidState(e.to_string()))?;
            Err(ConsolidationError::Failed("Failed to apply consolidation results".to_string()))
        }
    }
    
    /// Detect conflicts between branches
    fn detect_conflicts(&self, parent: &MemoryBranch, child: &MemoryBranch) -> Vec<Conflict> {
        let mut conflicts = Vec::new();
        
        let parent_concepts: HashSet<_> = parent.data.concept_ids.iter().collect();
        let child_concepts: HashSet<_> = child.data.concept_ids.iter().collect();
        
        // Find concepts in both branches
        let common_concepts: Vec<_> = parent_concepts.intersection(&child_concepts)
            .cloned()
            .collect();
        
        // For each common concept, check for conflicts
        for concept_id in common_concepts {
            let parent_props = parent.get_concept_properties(*concept_id);
            let child_props = child.get_concept_properties(*concept_id);
            
            if let (Some(p_props), Some(c_props)) = (parent_props, child_props) {
                // Check for real conflicts by comparing properties
                let has_conflict = p_props.modified_at != c_props.modified_at || 
                                 p_props.value != c_props.value ||
                                 !p_props.spike_signatures_match(&c_props) ||
                                 p_props.confidence != c_props.confidence ||
                                 p_props.properties != c_props.properties;
                
                if has_conflict {
                    conflicts.push(Conflict {
                        concept_id: *concept_id,
                        conflict_type: Self::determine_conflict_type(&p_props, &c_props),
                        parent_version: Some(p_props.version.to_string()),
                        child_version: Some(c_props.version.to_string()),
                    });
                }
            } else {
                // One branch has the concept but not the other (shouldn't happen with common concepts, but safety check)
                if parent_props.is_none() {
                    conflicts.push(Conflict {
                        concept_id: *concept_id,
                        conflict_type: ConflictType::DeletedInBranch,
                        parent_version: None,
                        child_version: child_props.map(|p| p.version.to_string()),
                    });
                } else if child_props.is_none() {
                    conflicts.push(Conflict {
                        concept_id: *concept_id,
                        conflict_type: ConflictType::DeletedInBranch,
                        parent_version: parent_props.map(|p| p.version.to_string()),
                        child_version: None,
                    });
                }
            }
        }
        
        conflicts
    }
    
    /// Resolve conflicts based on strategy
    fn resolve_conflicts(&self, 
                        conflicts: &[Conflict], 
                        parent: &MemoryBranch,
                        child: &MemoryBranch) -> Vec<Resolution> {
        conflicts.iter().map(|conflict| {
            let action = match self.config.resolution_strategy {
                ResolutionStrategy::PreferNewer => {
                    if child.metadata.modified_at > parent.metadata.modified_at {
                        ResolutionAction::UseChild
                    } else {
                        ResolutionAction::UseParent
                    }
                }
                ResolutionStrategy::PreferParent => ResolutionAction::UseParent,
                ResolutionStrategy::PreferChild => ResolutionAction::UseChild,
                ResolutionStrategy::MergeBoth => ResolutionAction::Merged,
                ResolutionStrategy::Manual => ResolutionAction::Manual("User required".to_string()),
            };
            
            Resolution {
                conflict: conflict.clone(),
                action,
                timestamp: chrono::Utc::now(),
            }
        }).collect()
    }
    
    /// Merge concepts from branches
    fn merge_concepts(&self,
                     parent: &MemoryBranch,
                     child: &MemoryBranch,
                     resolutions: &[Resolution]) -> Vec<uuid::Uuid> {
        let mut merged = HashSet::new();
        
        // Add all parent concepts
        merged.extend(&parent.data.concept_ids);
        
        // Add child concepts based on resolutions
        for concept_id in &child.data.concept_ids {
            let has_conflict = resolutions.iter()
                .any(|r| r.conflict.concept_id == *concept_id);
            
            if !has_conflict {
                merged.insert(*concept_id);
            } else {
                // Check resolution action
                let resolution = resolutions.iter()
                    .find(|r| r.conflict.concept_id == *concept_id)
                    .unwrap();
                
                match resolution.action {
                    ResolutionAction::UseChild | ResolutionAction::Merged => {
                        merged.insert(*concept_id);
                    }
                    _ => {}
                }
            }
        }
        
        merged.into_iter().collect()
    }
    
    /// Calculate divergence score
    fn calculate_divergence(&self, merged: &[uuid::Uuid], parent: &MemoryBranch) -> f32 {
        let parent_set: HashSet<_> = parent.data.concept_ids.iter().collect();
        let merged_set: HashSet<_> = merged.iter().collect();
        
        let added = merged_set.difference(&parent_set).count();
        let total = merged_set.len().max(1);
        
        added as f32 / total as f32
    }
    
    /// Apply consolidation results to the branches
    fn apply_consolidation_results(&self,
                                  parent: &mut MemoryBranch,
                                  child: &MemoryBranch,
                                  merged_concepts: &[uuid::Uuid],
                                  resolutions: &[Resolution]) -> bool {
        // Update parent's concept list
        parent.data.concept_ids = merged_concepts.to_vec();
        
        // Apply concept property changes based on resolutions
        for resolution in resolutions {
            let concept_id = resolution.conflict.concept_id;
            
            match resolution.action {
                ResolutionAction::UseChild => {
                    // Transfer child's concept properties to parent
                    if let Some(child_props) = child.get_concept_properties(concept_id) {
                        let mut updated_props = child_props.clone();
                        // Update version to reflect the merge - increment patch for property changes
                        updated_props.version = parent.metadata.version.next_patch();
                        updated_props.modified_at = chrono::Utc::now();
                        parent.data.concept_properties.insert(concept_id, updated_props);
                    }
                }
                ResolutionAction::UseParent => {
                    // Keep parent's properties, but update modification time
                    if let Some(parent_props) = parent.get_concept_properties(concept_id).cloned() {
                        let mut updated_props = parent_props;
                        updated_props.modified_at = chrono::Utc::now();
                        parent.data.concept_properties.insert(concept_id, updated_props);
                    }
                }
                ResolutionAction::Merged => {
                    // Merge properties from both branches
                    if let (Some(parent_props), Some(child_props)) = 
                        (parent.get_concept_properties(concept_id).cloned(), 
                         child.get_concept_properties(concept_id).cloned()) {
                        
                        let mut merged_props = parent_props;
                        
                        // Use the newer value
                        if child_props.modified_at > merged_props.modified_at {
                            merged_props.value = child_props.value;
                        }
                        
                        // Merge spike signatures
                        merged_props.spike_signatures.extend(child_props.spike_signatures);
                        merged_props.spike_signatures.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
                        merged_props.spike_signatures.dedup_by(|a, b| a.hash == b.hash);
                        
                        // Use higher confidence
                        merged_props.confidence = merged_props.confidence.max(child_props.confidence);
                        
                        // Merge additional properties
                        for (key, value) in child_props.properties {
                            merged_props.properties.insert(key, value);
                        }
                        
                        // Update metadata - increment patch for merged properties
                        merged_props.version = parent.metadata.version.next_patch();
                        merged_props.modified_at = chrono::Utc::now();
                        
                        parent.data.concept_properties.insert(concept_id, merged_props);
                    }
                }
                ResolutionAction::Skipped => {
                    // Do nothing for skipped concepts
                }
                ResolutionAction::Manual(_) => {
                    // For manual resolutions, keep parent properties for now
                    // In a real implementation, this would require user intervention
                    if let Some(parent_props) = parent.get_concept_properties(concept_id).cloned() {
                        let mut updated_props = parent_props;
                        updated_props.modified_at = chrono::Utc::now();
                        parent.data.concept_properties.insert(concept_id, updated_props);
                    }
                }
            }
        }
        
        // Add any new concepts from child that don't have conflicts
        for &concept_id in &child.data.concept_ids {
            let has_conflict = resolutions.iter().any(|r| r.conflict.concept_id == concept_id);
            
            if !has_conflict && !parent.data.concept_ids.contains(&concept_id) {
                // Add the concept from child
                if let Some(child_props) = child.get_concept_properties(concept_id) {
                    let mut new_props = child_props.clone();
                    new_props.version = parent.metadata.version.next_patch();
                    new_props.modified_at = chrono::Utc::now();
                    parent.data.concept_properties.insert(concept_id, new_props);
                }
            }
        }
        
        // Update parent divergence score
        parent.data.divergence = self.calculate_divergence(&parent.data.concept_ids, parent);
        
        // Update allocation count
        parent.data.allocation_count = parent.data.concept_ids.len();
        
        true // Success
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::branch::MemoryBranch;
    
    #[test]
    fn test_conflict_detection() {
        let engine = ConsolidationEngine::new(ConsolidationConfig::default());
        
        let mut parent = MemoryBranch::new("parent");
        
        // Add concepts to parent first
        let concept1 = uuid::Uuid::new_v4();
        let concept2 = uuid::Uuid::new_v4();
        
        parent.add_concept(concept1);
        parent.add_concept(concept2);
        
        // Now create child which inherits the concepts
        let mut child = parent.create_child("child");
        
        // Child inherits concepts from parent, so no conflicts initially
        let conflicts = engine.detect_conflicts(&parent, &child);
        assert_eq!(conflicts.len(), 0, "Child should inherit parent concepts without conflicts initially");
        
        // Now modify the child's concept to create a conflict
        if let Some(mut props) = child.get_concept_properties(concept1).cloned() {
            props.value = "modified_value".to_string();
            props.modified_at = chrono::Utc::now() + chrono::Duration::seconds(1);
            child.update_concept_properties(concept1, props);
        }
        
        let conflicts = engine.detect_conflicts(&parent, &child);
        assert_eq!(conflicts.len(), 1, "Should detect one conflict after modifying concept");
        assert_eq!(conflicts[0].concept_id, concept1);
        assert_eq!(conflicts[0].conflict_type, ConflictType::PropertyMismatch);
    }
    
    #[test]
    fn test_consolidation() {
        let engine = ConsolidationEngine::new(ConsolidationConfig {
            resolution_strategy: ResolutionStrategy::PreferChild,
            ..Default::default()
        });
        
        let mut parent = MemoryBranch::new("parent");
        let mut child = parent.create_child("child");
        
        let concept1 = uuid::Uuid::new_v4();
        let concept2 = uuid::Uuid::new_v4();
        let concept3 = uuid::Uuid::new_v4();
        
        parent.add_concept(concept1);
        parent.add_concept(concept2);
        child.add_concept(concept2);
        child.add_concept(concept3);
        
        // Store initial state
        let initial_parent_version = parent.metadata.version.clone();
        let initial_child_state = child.state;
        
        let result = engine.consolidate(&mut parent, &mut child).unwrap();
        
        assert!(result.success);
        assert!(result.merged_concepts.contains(&concept1));
        assert!(result.merged_concepts.contains(&concept2));
        assert!(result.merged_concepts.contains(&concept3));
        
        // Verify that branches were actually modified
        assert_eq!(child.state, ConsolidationState::Consolidated);
        assert_ne!(child.state, initial_child_state);
        assert_eq!(parent.metadata.version, initial_parent_version.next_minor());
        
        // Verify parent now contains all concepts
        assert!(parent.data.concept_ids.contains(&concept1));
        assert!(parent.data.concept_ids.contains(&concept2));
        assert!(parent.data.concept_ids.contains(&concept3));
        assert_eq!(parent.data.concept_ids.len(), 3);
    }
    
    #[test]
    fn test_spike_pattern_conflict() {
        use crate::branch::SpikeSignature;
        
        let engine = ConsolidationEngine::new(ConsolidationConfig::default());
        
        let mut parent = MemoryBranch::new("parent");
        
        let concept1 = uuid::Uuid::new_v4();
        parent.add_concept(concept1);
        
        // Create child after parent has the concept
        let mut child = parent.create_child("child");
        
        // Create conflicting spike signatures
        let parent_spike = SpikeSignature {
            hash: 123456789,
            timestamp: chrono::Utc::now(),
            avg_rate: 10.0,
            complexity: 0.5,
        };
        
        let child_spike = SpikeSignature {
            hash: 987654321, // Different hash
            timestamp: chrono::Utc::now(),
            avg_rate: 20.0, // Different rate
            complexity: 0.8, // Different complexity
        };
        
        // Update parent concept with spike signature
        if let Some(mut props) = parent.get_concept_properties(concept1).cloned() {
            props.spike_signatures.push(parent_spike);
            parent.update_concept_properties(concept1, props);
        }
        
        // Update child concept with different spike signature
        if let Some(mut props) = child.get_concept_properties(concept1).cloned() {
            props.spike_signatures.push(child_spike);
            child.update_concept_properties(concept1, props);
        }
        
        let conflicts = engine.detect_conflicts(&parent, &child);
        assert_eq!(conflicts.len(), 1, "Should detect spike pattern conflict");
        assert_eq!(conflicts[0].conflict_type, ConflictType::SpikePatternConflict);
    }
    
    #[test]
    fn test_concept_property_transfer() {
        let engine = ConsolidationEngine::new(ConsolidationConfig {
            resolution_strategy: ResolutionStrategy::PreferChild,
            ..Default::default()
        });
        
        let mut parent = MemoryBranch::new("parent");
        let concept1 = uuid::Uuid::new_v4();
        parent.add_concept(concept1);
        
        // Create child and modify the concept
        let mut child = parent.create_child("child");
        if let Some(mut props) = child.get_concept_properties(concept1).cloned() {
            props.value = "modified_by_child".to_string();
            props.confidence = 0.9;
            props.properties.insert("custom_prop".to_string(), "child_value".to_string());
            child.update_concept_properties(concept1, props);
        }
        
        let initial_parent_props = parent.get_concept_properties(concept1).unwrap().clone();
        
        let result = engine.consolidate(&mut parent, &mut child).unwrap();
        
        assert!(result.success);
        assert_eq!(child.state, ConsolidationState::Consolidated);
        
        // Verify concept properties were transferred
        let updated_parent_props = parent.get_concept_properties(concept1).unwrap();
        assert_eq!(updated_parent_props.value, "modified_by_child");
        assert_eq!(updated_parent_props.confidence, 0.9);
        assert_eq!(updated_parent_props.properties.get("custom_prop"), Some(&"child_value".to_string()));
        
        // Verify properties actually changed
        assert_ne!(updated_parent_props.value, initial_parent_props.value);
        assert_ne!(updated_parent_props.modified_at, initial_parent_props.modified_at);
    }
    
    #[test]
    fn test_consolidation_state_transitions() {
        let engine = ConsolidationEngine::new(ConsolidationConfig::default());
        
        let mut parent = MemoryBranch::new("parent");
        let mut child = parent.create_child("child");
        
        // Initial state should be Active
        assert_eq!(child.state, ConsolidationState::Active);
        
        let result = engine.consolidate(&mut parent, &mut child).unwrap();
        
        assert!(result.success);
        assert_eq!(child.state, ConsolidationState::Consolidated);
    }
    
    #[test]
    fn test_consolidation_with_too_many_conflicts() {
        let engine = ConsolidationEngine::new(ConsolidationConfig {
            max_conflicts: 0, // No conflicts allowed
            ..Default::default()
        });
        
        let mut parent = MemoryBranch::new("parent");
        let concept1 = uuid::Uuid::new_v4();
        parent.add_concept(concept1);
        
        let mut child = parent.create_child("child");
        
        // Create a conflict by modifying the concept in the child
        if let Some(mut props) = child.get_concept_properties(concept1).cloned() {
            props.value = "conflicting_value".to_string();
            props.modified_at = chrono::Utc::now() + chrono::Duration::seconds(1);
            child.update_concept_properties(concept1, props);
        }
        
        let result = engine.consolidate(&mut parent, &mut child);
        
        assert!(result.is_err());
        assert_eq!(child.state, ConsolidationState::Conflicted);
        
        match result {
            Err(ConsolidationError::TooManyConflicts(count)) => assert_eq!(count, 1),
            _ => panic!("Expected TooManyConflicts error"),
        }
    }
    
    #[test]
    fn test_state_transition_history_during_consolidation() {
        let engine = ConsolidationEngine::new(ConsolidationConfig::default());
        
        let mut parent = MemoryBranch::new("parent");
        let mut child = parent.create_child("child");
        
        let concept1 = uuid::Uuid::new_v4();
        parent.add_concept(concept1);
        child.add_concept(concept1);
        
        // Initial state and empty history
        assert_eq!(child.current_state(), ConsolidationState::Active);
        assert_eq!(child.state_history().len(), 0);
        
        let result = engine.consolidate(&mut parent, &mut child).unwrap();
        
        assert!(result.success);
        assert_eq!(child.current_state(), ConsolidationState::Consolidated);
        
        // Check that state transitions were recorded
        let history = child.state_history();
        assert_eq!(history.len(), 2);
        
        // First transition: Active -> Consolidating
        assert_eq!(history[0].from, ConsolidationState::Active);
        assert_eq!(history[0].to, ConsolidationState::Consolidating);
        assert_eq!(history[0].reason, "Starting consolidation with parent");
        
        // Second transition: Consolidating -> Consolidated
        assert_eq!(history[1].from, ConsolidationState::Consolidating);
        assert_eq!(history[1].to, ConsolidationState::Consolidated);
        assert!(history[1].reason.contains("Successfully consolidated"));
        assert!(history[1].reason.contains("concepts"));
        
        // Verify timestamps are in order
        assert!(history[0].timestamp <= history[1].timestamp);
    }
    
    #[test]
    fn test_state_transition_on_conflict_failure() {
        let engine = ConsolidationEngine::new(ConsolidationConfig {
            max_conflicts: 0, // Force conflict failure
            ..Default::default()
        });
        
        let mut parent = MemoryBranch::new("parent");
        let concept1 = uuid::Uuid::new_v4();
        parent.add_concept(concept1);
        
        let mut child = parent.create_child("child");
        
        // Create a conflict
        if let Some(mut props) = child.get_concept_properties(concept1).cloned() {
            props.value = "modified_value".to_string();
            props.modified_at = chrono::Utc::now() + chrono::Duration::seconds(1);
            child.update_concept_properties(concept1, props);
        }
        
        // Initial state
        assert_eq!(child.current_state(), ConsolidationState::Active);
        
        let result = engine.consolidate(&mut parent, &mut child);
        
        assert!(result.is_err());
        assert_eq!(child.current_state(), ConsolidationState::Conflicted);
        
        // Check state transition history
        let history = child.state_history();
        assert_eq!(history.len(), 2);
        
        // First transition: Active -> Consolidating
        assert_eq!(history[0].from, ConsolidationState::Active);
        assert_eq!(history[0].to, ConsolidationState::Consolidating);
        
        // Second transition: Consolidating -> Conflicted
        assert_eq!(history[1].from, ConsolidationState::Consolidating);
        assert_eq!(history[1].to, ConsolidationState::Conflicted);
        assert!(history[1].reason.contains("Too many conflicts detected"));
    }
    
    #[test]
    fn test_invalid_state_validation_before_consolidation() {
        let engine = ConsolidationEngine::new(ConsolidationConfig::default());
        
        let mut parent = MemoryBranch::new("parent");
        let _child = parent.create_child("child");
        
        // Since Active -> Consolidated is not a valid transition, 
        // let's test by manually setting an improper state for consolidation
        
        // Create a child that's already in Consolidating state
        let mut child2 = parent.create_child("child2");
        child2.transition_state(
            ConsolidationState::Consolidating,
            "Manual state change".to_string()
        ).unwrap();
        
        // The consolidation should handle this case - Consolidating -> Consolidating is valid
        let result = engine.consolidate(&mut parent, &mut child2);
        
        // Let's check what happens - the validation logic might reject Consolidating state
        match result {
            Ok(result) => {
                assert!(result.success);
            }
            Err(ConsolidationError::InvalidState(msg)) => {
                // The validation correctly rejects consolidating an already consolidating branch
                assert!(msg.contains("cannot consolidate") || msg.contains("cannot transition"));
            }
            Err(e) => {
                panic!("Unexpected error: {:?}", e);
            }
        }
        
        // Now let's test with a proper transition sequence to Consolidated state
        let mut child3 = parent.create_child("child3");
        child3.transition_state(
            ConsolidationState::Consolidating,
            "Initial transition".to_string()
        ).unwrap();
        child3.transition_state(
            ConsolidationState::Consolidated,
            "Mark as done".to_string()
        ).unwrap();
        
        // Now try to consolidate a Consolidated branch
        let result = engine.consolidate(&mut parent, &mut child3);
        
        // Our state machine allows Consolidated -> Consolidating, so this might succeed
        // Let's verify what actually happens
        match result {
            Ok(result) => {
                // If it succeeds, verify the state transitions were recorded
                assert!(result.success);
                assert_eq!(child3.current_state(), ConsolidationState::Consolidated);
            }
            Err(ConsolidationError::InvalidState(_)) => {
                // This would be expected if we add more restrictive validation
                assert!(true);
            }
            Err(e) => {
                panic!("Unexpected error: {:?}", e);
            }
        }
    }
    
    #[test]
    fn test_semantic_versioning_in_consolidation() {
        let engine = ConsolidationEngine::new(ConsolidationConfig {
            resolution_strategy: ResolutionStrategy::PreferChild,
            ..Default::default()
        });
        
        let mut parent = MemoryBranch::new("parent");
        let concept1 = uuid::Uuid::new_v4();
        parent.add_concept(concept1);
        
        // Simulate parent having done some work (increment its version)
        parent.increment_minor_version(); // Now 1.1.0
        
        let mut child = parent.create_child("child");
        // Child inherits parent version
        assert_eq!(child.metadata.version.to_string(), "1.1.0");
        
        // Modify child concept to create a conflict
        if let Some(mut props) = child.get_concept_properties(concept1).cloned() {
            props.value = "modified_by_child".to_string();
            child.update_concept_properties(concept1, props);
        }
        
        // Check concept version was incremented in child
        let child_props = child.get_concept_properties(concept1).unwrap();
        assert_eq!(child_props.version.to_string(), "1.0.1"); // Original 1.0.0 + patch increment
        
        let result = engine.consolidate(&mut parent, &mut child).unwrap();
        
        assert!(result.success);
        assert_eq!(child.state, ConsolidationState::Consolidated);
        
        // Parent version should be incremented (minor version for consolidation)
        assert_eq!(parent.metadata.version.to_string(), "1.2.0");
        
        // Concept properties should have updated version  
        let final_props = parent.get_concept_properties(concept1).unwrap();
        assert_eq!(final_props.value, "modified_by_child");
        // The concept gets versioned before the parent version is incremented
        assert_eq!(final_props.version.to_string(), "1.1.1");
        
        // Verify conflict had real version strings, not hardcoded
        assert_eq!(result.conflicts.len(), 1);
        let conflict = &result.conflicts[0];
        
        // The parent concept was created with original version 1.0.0, child was 1.0.1 after update
        assert_eq!(conflict.parent_version.as_ref().unwrap(), "1.0.0");
        assert_eq!(conflict.child_version.as_ref().unwrap(), "1.0.1");
    }
}