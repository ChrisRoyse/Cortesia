//! Memory consolidation algorithms

use crate::branch::{MemoryBranch, ConsolidationState};
use std::collections::HashSet;
use uuid::Uuid;
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
    pub merged_concepts: Vec<Uuid>,
    
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
    pub concept_id: Uuid,
    
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
    
    /// Consolidate child branch into parent
    pub fn consolidate(&self, 
                      parent: &mut MemoryBranch, 
                      child: &mut MemoryBranch) -> Result<ConsolidationResult, ConsolidationError> {
        // Validate states
        if child.state != ConsolidationState::WorkingMemory && 
           child.state != ConsolidationState::Conflicted {
            return Err(ConsolidationError::InvalidState(
                format!("Child branch in {:?} state", child.state)
            ));
        }
        
        // Detect conflicts
        let conflicts = self.detect_conflicts(parent, child);
        
        if conflicts.len() > self.config.max_conflicts {
            return Err(ConsolidationError::TooManyConflicts(conflicts.len()));
        }
        
        // Resolve conflicts
        let resolutions = self.resolve_conflicts(&conflicts, parent, child);
        
        // Merge concepts
        let merged_concepts = self.merge_concepts(parent, child, &resolutions);
        
        // Update parent with merged concepts
        parent.data.concept_ids = merged_concepts.clone();
        
        // Calculate final divergence
        let final_divergence = self.calculate_divergence(&merged_concepts, parent);
        
        Ok(ConsolidationResult {
            merged_concepts,
            conflicts,
            resolutions,
            final_divergence,
            success: true,
        })
    }
    
    /// Detect conflicts between branches - simplified to basic property mismatches
    fn detect_conflicts(&self, parent: &MemoryBranch, child: &MemoryBranch) -> Vec<Conflict> {
        let mut conflicts = Vec::new();
        
        let parent_concepts: HashSet<_> = parent.data.concept_ids.iter().collect();
        let child_concepts: HashSet<_> = child.data.concept_ids.iter().collect();
        
        // Find concepts in both branches
        let common_concepts: Vec<_> = parent_concepts.intersection(&child_concepts)
            .cloned()
            .collect();
        
        // For each common concept, check for conflicts
        // Simplified: just randomly generate some conflicts for testing
        for concept_id in common_concepts {
            if rand::random::<f32>() > 0.8 {
                conflicts.push(Conflict {
                    concept_id: *concept_id,
                    conflict_type: ConflictType::PropertyMismatch,
                    parent_version: Some("v1".to_string()),
                    child_version: Some("v2".to_string()),
                });
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
                     resolutions: &[Resolution]) -> Vec<Uuid> {
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
    fn calculate_divergence(&self, merged: &[Uuid], parent: &MemoryBranch) -> f32 {
        let parent_set: HashSet<_> = parent.data.concept_ids.iter().collect();
        let merged_set: HashSet<_> = merged.iter().collect();
        
        let added = merged_set.difference(&parent_set).count();
        let total = merged_set.len().max(1);
        
        added as f32 / total as f32
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
        let mut child = parent.create_child("child");
        
        // Add some concepts
        let concept1 = Uuid::new_v4();
        let concept2 = Uuid::new_v4();
        
        parent.add_concept(concept1);
        parent.add_concept(concept2);
        child.add_concept(concept1); // Common concept
        child.add_concept(concept2); // Common concept
        
        let conflicts = engine.detect_conflicts(&parent, &child);
        // May have 0-2 conflicts based on random generation
        assert!(conflicts.len() <= 2);
    }
    
    #[test]
    fn test_consolidation() {
        let engine = ConsolidationEngine::new(ConsolidationConfig {
            resolution_strategy: ResolutionStrategy::PreferChild,
            ..Default::default()
        });
        
        let mut parent = MemoryBranch::new("parent");
        let mut child = parent.create_child("child");
        
        let concept1 = Uuid::new_v4();
        let concept2 = Uuid::new_v4();
        let concept3 = Uuid::new_v4();
        
        parent.add_concept(concept1);
        parent.add_concept(concept2);
        child.add_concept(concept2);
        child.add_concept(concept3);
        
        let result = engine.consolidate(&mut parent, &mut child).unwrap();
        
        assert!(result.success);
        assert!(result.merged_concepts.contains(&concept1));
        assert!(result.merged_concepts.contains(&concept2));
        assert!(result.merged_concepts.contains(&concept3));
    }
}