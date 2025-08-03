//! Branch lifecycle management
//!
//! This module implements the Branch Manager for temporal memory, handling
//! the complete lifecycle of memory branches including creation, traversal,
//! consolidation orchestration, and event management.

use super::{MemoryBranch, BranchConfig, ConsolidationState, BranchId};
use crate::consolidation::{ConsolidationEngine, ConsolidationConfig, ConsolidationResult};
use dashmap::DashMap;
use parking_lot::RwLock;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::broadcast;

/// Events emitted by the branch manager
#[derive(Debug, Clone)]
pub enum BranchEvent {
    /// Branch was created
    Created { 
        branch_id: BranchId, 
        parent_id: Option<BranchId> 
    },
    
    /// Branch state changed
    StateChanged { 
        branch_id: BranchId, 
        from: ConsolidationState, 
        to: ConsolidationState 
    },
    
    /// Branch was consolidated
    Consolidated { 
        parent_id: BranchId, 
        child_id: BranchId 
    },
    
    /// Branch was pruned
    Pruned { 
        branch_id: BranchId 
    },
    
    /// Concept allocated to branch
    ConceptAllocated { 
        branch_id: BranchId, 
        concept_id: uuid::Uuid 
    },
}

/// Errors that can occur in branch management
#[derive(Error, Debug)]
pub enum BranchError {
    #[error("Branch not found: {0}")]
    NotFound(BranchId),
    
    #[error("Invalid parent branch: {0}")]
    InvalidParent(BranchId),
    
    #[error("Circular dependency detected")]
    CircularDependency,
    
    #[error("Branch is not active: {0}")]
    NotActive(BranchId),
    
    #[error("Consolidation failed: {0}")]
    ConsolidationFailed(String),
    
    #[error("Operation not allowed in current state")]
    InvalidOperation,
}

/// Manages the lifecycle of memory branches
pub struct BranchManager {
    /// Configuration
    config: BranchConfig,
    
    /// Branch storage
    branches: DashMap<BranchId, Arc<RwLock<MemoryBranch>>>,
    
    /// Root branch ID
    root_id: BranchId,
    
    /// Consolidation engine
    consolidation_engine: ConsolidationEngine,
    
    /// Event channel
    event_sender: broadcast::Sender<BranchEvent>,
    
    /// Statistics
    stats: RwLock<BranchStats>,
}

/// Statistics about branch operations
#[derive(Debug, Default, Clone)]
pub struct BranchStats {
    pub total_branches: usize,
    pub active_branches: usize,
    pub consolidated_branches: usize,
    pub total_concepts: usize,
    pub consolidation_attempts: usize,
    pub successful_consolidations: usize,
}

impl BranchManager {
    /// Create new branch manager
    pub fn new(config: BranchConfig) -> Self {
        let root = MemoryBranch::new("root");
        let root_id = root.id().to_string();
        
        let branches = DashMap::new();
        branches.insert(root_id.clone(), Arc::new(RwLock::new(root)));
        
        let (event_sender, _) = broadcast::channel(1000);
        
        // Create consolidation config based on branch config
        let consolidation_config = ConsolidationConfig {
            merge_threshold: config.auto_consolidate_threshold,
            max_conflicts: 10,
            resolution_strategy: crate::consolidation::ResolutionStrategy::PreferNewer,
            parallel: true, // Enable parallel consolidation by default
        };
        
        Self {
            config,
            branches,
            root_id: root_id.clone(),
            consolidation_engine: ConsolidationEngine::new(consolidation_config),
            event_sender,
            stats: RwLock::new(BranchStats {
                total_branches: 1,
                active_branches: 1,
                ..Default::default()
            }),
        }
    }
    
    /// Get event receiver
    pub fn subscribe(&self) -> broadcast::Receiver<BranchEvent> {
        self.event_sender.subscribe()
    }
    
    /// Get root branch ID
    pub fn root_id(&self) -> &BranchId {
        &self.root_id
    }
    
    /// Get the branch configuration
    pub fn config(&self) -> &BranchConfig {
        &self.config
    }
    
    /// Create a new branch
    pub fn create_branch(&self, 
                        name: impl Into<String>, 
                        parent_id: Option<BranchId>) -> Result<BranchId, BranchError> {
        let parent_id = parent_id.unwrap_or_else(|| self.root_id.clone());
        
        // Get parent branch
        let parent_arc = self.branches.get(&parent_id)
            .ok_or_else(|| BranchError::InvalidParent(parent_id.clone()))?;
        
        let mut parent = parent_arc.write();
        
        // Create child branch
        let child = parent.create_child(name);
        let child_id = child.id().to_string();
        
        // Store child
        self.branches.insert(child_id.clone(), Arc::new(RwLock::new(child)));
        
        // Update stats
        {
            let mut stats = self.stats.write();
            stats.total_branches += 1;
            stats.active_branches += 1;
        }
        
        // Emit event
        let _ = self.event_sender.send(BranchEvent::Created {
            branch_id: child_id.clone(),
            parent_id: Some(parent_id.clone()),
        });
        
        Ok(child_id)
    }
    
    /// Get a branch by ID
    pub fn get_branch(&self, id: &BranchId) -> Option<Arc<RwLock<MemoryBranch>>> {
        self.branches.get(id).map(|b| Arc::clone(&b))
    }
    
    /// List all branches
    pub fn list_branches(&self) -> Vec<BranchId> {
        self.branches.iter()
            .map(|entry| entry.key().clone())
            .collect()
    }
    
    /// Find branches by state
    pub fn find_by_state(&self, state: ConsolidationState) -> Vec<BranchId> {
        self.branches.iter()
            .filter(|entry| entry.value().read().state == state)
            .map(|entry| entry.key().clone())
            .collect()
    }
    
    /// Get branch ancestry (path from root to branch)
    pub fn get_ancestry(&self, branch_id: &BranchId) -> Result<Vec<BranchId>, BranchError> {
        let mut ancestry = Vec::new();
        let mut current_id = Some(branch_id.clone());
        let mut visited = std::collections::HashSet::new();
        
        while let Some(id) = current_id {
            // Detect cycles
            if !visited.insert(id.clone()) {
                return Err(BranchError::CircularDependency);
            }
            
            ancestry.push(id.clone());
            
            let branch = self.get_branch(&id)
                .ok_or_else(|| BranchError::NotFound(id.clone()))?;
            
            current_id = branch.read().parent_id().map(|s| s.to_string());
        }
        
        ancestry.reverse(); // Root first
        Ok(ancestry)
    }
    
    /// Get all descendants of a branch
    pub fn get_descendants(&self, branch_id: &BranchId) -> Result<Vec<BranchId>, BranchError> {
        let mut descendants = Vec::new();
        let mut to_visit = vec![branch_id.clone()];
        
        while let Some(current_id) = to_visit.pop() {
            let branch = self.get_branch(&current_id)
                .ok_or_else(|| BranchError::NotFound(current_id.clone()))?;
            
            let children = branch.read().children.clone();
            
            for child_id in children {
                descendants.push(child_id.clone());
                to_visit.push(child_id);
            }
        }
        
        Ok(descendants)
    }
    
    /// Allocate a concept to a branch
    pub fn allocate_concept(&self, 
                           branch_id: &BranchId, 
                           concept_id: uuid::Uuid) -> Result<(), BranchError> {
        let branch = self.get_branch(branch_id)
            .ok_or_else(|| BranchError::NotFound(branch_id.clone()))?;
        
        let mut branch_write = branch.write();
        
        // Check if branch can accept allocations
        if branch_write.state != ConsolidationState::WorkingMemory {
            return Err(BranchError::NotActive(branch_id.clone()));
        }
        
        branch_write.add_concept(concept_id);
        
        // Update stats
        self.stats.write().total_concepts += 1;
        
        // Emit event
        let _ = self.event_sender.send(BranchEvent::ConceptAllocated {
            branch_id: branch_id.clone(),
            concept_id,
        });
        
        Ok(())
    }
    
    /// Start consolidation of a branch into its parent
    /// This is now truly asynchronous, using tokio::spawn for parallel processing
    pub async fn consolidate_branch(&self, branch_id: &BranchId) -> Result<ConsolidationResult, BranchError> {
        let branch = self.get_branch(branch_id)
            .ok_or_else(|| BranchError::NotFound(branch_id.clone()))?;
        
        let parent_id = {
            let b = branch.read();
            b.parent_id().ok_or(BranchError::InvalidOperation)?.to_string()
        };
        
        let parent = self.get_branch(&parent_id)
            .ok_or_else(|| BranchError::NotFound(parent_id.clone()))?;
        
        // Update state to consolidating
        self.update_branch_state(branch_id, ConsolidationState::Consolidating)?;
        
        // Clone necessary data for async processing
        let branch_id_clone = branch_id.clone();
        let parent_id_clone = parent_id.clone();
        let parent_arc = Arc::clone(&parent);
        let child_arc = Arc::clone(&branch);
        let consolidation_engine = self.consolidation_engine.clone();
        let event_sender = self.event_sender.clone();
        
        // Perform consolidation with proper parallel/sequential handling
        let result = if self.consolidation_engine.config().parallel {
            // Parallel: spawn task for concurrent execution
            let task = tokio::spawn(async move {
                Self::perform_async_consolidation(
                    consolidation_engine,
                    parent_arc,
                    child_arc,
                ).await
            });
            task.await
                .map_err(|e| BranchError::ConsolidationFailed(format!("Async task failed: {}", e)))?
                .map_err(|e| BranchError::ConsolidationFailed(e.to_string()))?
        } else {
            // Sequential: direct execution without spawning
            Self::perform_async_consolidation(
                consolidation_engine,
                parent_arc,
                child_arc,
            ).await
                .map_err(|e| BranchError::ConsolidationFailed(e.to_string()))?
        };
        
        // Update state based on result
        if result.success {
            self.update_branch_state(&branch_id_clone, ConsolidationState::LongTerm)?;
            
            // Update stats
            let mut stats_write = self.stats.write();
            stats_write.consolidation_attempts += 1;
            stats_write.successful_consolidations += 1;
            stats_write.active_branches -= 1;
            stats_write.consolidated_branches += 1;
            
            // Emit event
            let _ = event_sender.send(BranchEvent::Consolidated {
                parent_id: parent_id_clone,
                child_id: branch_id_clone,
            });
        } else {
            self.update_branch_state(&branch_id_clone, ConsolidationState::Conflicted)?;
            
            self.stats.write().consolidation_attempts += 1;
        }
        
        Ok(result)
    }
    
    /// Perform the actual consolidation work asynchronously
    /// This allows for proper async processing and yielding to the tokio runtime
    async fn perform_async_consolidation(
        consolidation_engine: ConsolidationEngine,
        parent: Arc<RwLock<MemoryBranch>>,
        child: Arc<RwLock<MemoryBranch>>,
    ) -> Result<ConsolidationResult, crate::consolidation::ConsolidationError> {
        // Use spawn_blocking for CPU-intensive synchronous work
        tokio::task::spawn_blocking(move || {
            let mut parent_write = parent.write();
            let mut child_write = child.write();
            
            consolidation_engine.consolidate(&mut parent_write, &mut child_write)
        })
        .await
        .map_err(|e| crate::consolidation::ConsolidationError::Failed(format!("Spawn blocking failed: {}", e)))?
    }
    
    /// Update branch state
    fn update_branch_state(&self, 
                          branch_id: &BranchId, 
                          new_state: ConsolidationState) -> Result<(), BranchError> {
        let branch = self.get_branch(branch_id)
            .ok_or_else(|| BranchError::NotFound(branch_id.clone()))?;
        
        let mut branch_write = branch.write();
        let old_state = branch_write.state;
        
        // Use transition_state method which handles validation
        branch_write.transition_state(new_state, "Manager update".to_string())
            .map_err(|_| BranchError::InvalidOperation)?;
        
        // Emit event
        let _ = self.event_sender.send(BranchEvent::StateChanged {
            branch_id: branch_id.clone(),
            from: old_state,
            to: new_state,
        });
        
        Ok(())
    }
    
    /// Prune old consolidated branches
    pub fn prune_consolidated(&self, max_age: chrono::Duration) -> Vec<BranchId> {
        let mut pruned = Vec::new();
        
        let to_prune: Vec<_> = self.branches.iter()
            .filter(|entry| {
                let branch = entry.value().read();
                branch.state == ConsolidationState::LongTerm &&
                branch.age() > max_age
            })
            .map(|entry| entry.key().clone())
            .collect();
        
        for branch_id in to_prune {
            if let Some((_, branch)) = self.branches.remove(&branch_id) {
                // Remove from parent's children
                if let Some(parent_id) = branch.read().parent_id() {
                    let parent_id_owned = parent_id.to_string();
                    if let Some(parent) = self.get_branch(&parent_id_owned) {
                        let mut parent_write = parent.write();
                        parent_write.children.retain(|id| id != &branch_id);
                    }
                }
                
                pruned.push(branch_id.clone());
                
                // Emit event
                let _ = self.event_sender.send(BranchEvent::Pruned { 
                    branch_id: branch_id.clone() 
                });
            }
        }
        
        // Update stats
        self.stats.write().total_branches -= pruned.len();
        
        pruned
    }
    
    /// Split a branch at a specific point
    /// Creates a new branch with the concepts added after the split point
    pub fn split_branch(&self, 
                       branch_id: &BranchId, 
                       split_point: usize,
                       new_branch_name: impl Into<String>) -> Result<BranchId, BranchError> {
        let branch = self.get_branch(branch_id)
            .ok_or_else(|| BranchError::NotFound(branch_id.clone()))?;
        
        let mut branch_write = branch.write();
        
        // Can only split active branches
        if branch_write.state != ConsolidationState::WorkingMemory {
            return Err(BranchError::InvalidOperation);
        }
        
        // Get concepts to move to new branch
        let concepts_to_move: Vec<_> = branch_write.data.concept_ids
            .drain(split_point..)
            .collect();
        
        drop(branch_write); // Release lock before creating child
        
        // Create new branch as child
        let new_branch_id = self.create_branch(new_branch_name, Some(branch_id.clone()))?;
        
        // Add concepts to new branch
        if let Some(new_branch) = self.get_branch(&new_branch_id) {
            let mut new_branch_write = new_branch.write();
            new_branch_write.data.concept_ids = concepts_to_move;
            new_branch_write.data.allocation_count = new_branch_write.data.concept_ids.len();
        }
        
        Ok(new_branch_id)
    }
    
    /// Merge two sibling branches
    /// Combines concepts from both branches into a new branch
    pub fn merge_branches(&self,
                         branch1_id: &BranchId,
                         branch2_id: &BranchId,
                         merged_name: impl Into<String>) -> Result<BranchId, BranchError> {
        let branch1 = self.get_branch(branch1_id)
            .ok_or_else(|| BranchError::NotFound(branch1_id.clone()))?;
        let branch2 = self.get_branch(branch2_id)
            .ok_or_else(|| BranchError::NotFound(branch2_id.clone()))?;
        
        let (parent_id, concepts1, concepts2) = {
            let b1 = branch1.read();
            let b2 = branch2.read();
            
            // Verify they are siblings
            if b1.parent_id() != b2.parent_id() {
                return Err(BranchError::InvalidOperation);
            }
            
            (
                b1.parent_id().map(|s| s.to_string()),
                b1.data.concept_ids.clone(),
                b2.data.concept_ids.clone()
            )
        };
        
        // Create new merged branch
        let merged_id = self.create_branch(merged_name, parent_id)?;
        
        // Add all concepts to merged branch
        if let Some(merged_branch) = self.get_branch(&merged_id) {
            let mut merged_write = merged_branch.write();
            
            // Combine concepts, removing duplicates
            let mut all_concepts = concepts1;
            for concept in concepts2 {
                if !all_concepts.contains(&concept) {
                    all_concepts.push(concept);
                }
            }
            
            merged_write.data.concept_ids = all_concepts;
            merged_write.data.allocation_count = merged_write.data.concept_ids.len();
        }
        
        Ok(merged_id)
    }
    
    /// Get branch statistics
    pub fn stats(&self) -> BranchStats {
        self.stats.read().clone()
    }
    
    /// Find common ancestor of two branches
    pub fn find_common_ancestor(&self, branch1_id: &BranchId, branch2_id: &BranchId) -> Result<Option<BranchId>, BranchError> {
        let ancestry1 = self.get_ancestry(branch1_id)?;
        let ancestry2 = self.get_ancestry(branch2_id)?;
        
        // Find the deepest common ancestor by comparing ordered lists
        let mut common_ancestor = None;
        
        // Compare the ancestry lists element by element
        for (a1, a2) in ancestry1.iter().zip(ancestry2.iter()) {
            if a1 == a2 {
                common_ancestor = Some(a1.clone());
            } else {
                break; // Stop at first divergence
            }
        }
        
        Ok(common_ancestor)
    }
    
    /// Check if a branch should be auto-consolidated based on config
    pub fn should_auto_consolidate(&self, branch_id: &BranchId) -> bool {
        if !self.config.detect_conflicts {
            return false;
        }
        
        if let Some(branch) = self.get_branch(branch_id) {
            let branch_read = branch.read();
            
            // Check divergence threshold
            if branch_read.data.divergence > self.config.max_divergence {
                return true;
            }
            
            // Check age
            let age = branch_read.age();
            let max_age_duration = chrono::Duration::from_std(self.config.max_age).unwrap_or(chrono::Duration::days(1));
            if age > max_age_duration {
                return true;
            }
            
            // Check confidence threshold
            if branch_read.data.confidence > self.config.auto_consolidate_threshold {
                return true;
            }
        }
        
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_branch_creation() {
        let manager = BranchManager::new(BranchConfig::default());
        
        let branch1 = manager.create_branch("feature1", None).unwrap();
        let branch2 = manager.create_branch("feature2", Some(branch1.clone())).unwrap();
        
        assert!(manager.get_branch(&branch1).is_some());
        assert!(manager.get_branch(&branch2).is_some());
        
        let stats = manager.stats();
        assert_eq!(stats.total_branches, 3); // root + 2 branches
        assert_eq!(stats.active_branches, 3);
    }
    
    #[test]
    fn test_ancestry_traversal() {
        let manager = BranchManager::new(BranchConfig::default());
        
        let branch1 = manager.create_branch("level1", None).unwrap();
        let branch2 = manager.create_branch("level2", Some(branch1.clone())).unwrap();
        let branch3 = manager.create_branch("level3", Some(branch2.clone())).unwrap();
        
        let ancestry = manager.get_ancestry(&branch3).unwrap();
        
        assert_eq!(ancestry.len(), 4); // root -> branch1 -> branch2 -> branch3
        assert_eq!(ancestry[0], *manager.root_id());
        assert_eq!(ancestry[3], branch3);
    }
    
    #[test]
    fn test_descendants_traversal() {
        let manager = BranchManager::new(BranchConfig::default());
        
        let parent = manager.create_branch("parent", None).unwrap();
        let child1 = manager.create_branch("child1", Some(parent.clone())).unwrap();
        let child2 = manager.create_branch("child2", Some(parent.clone())).unwrap();
        let grandchild = manager.create_branch("grandchild", Some(child1.clone())).unwrap();
        
        let descendants = manager.get_descendants(&parent).unwrap();
        
        assert_eq!(descendants.len(), 3);
        assert!(descendants.contains(&child1));
        assert!(descendants.contains(&child2));
        assert!(descendants.contains(&grandchild));
    }
    
    #[test]
    fn test_concept_allocation() {
        let manager = BranchManager::new(BranchConfig::default());
        let branch = manager.create_branch("test", None).unwrap();
        let concept_id = uuid::Uuid::new_v4();
        
        assert!(manager.allocate_concept(&branch, concept_id).is_ok());
        
        let branch_ref = manager.get_branch(&branch).unwrap();
        assert!(branch_ref.read().data.concept_ids.contains(&concept_id));
        
        assert_eq!(manager.stats().total_concepts, 1);
    }
    
    #[test]
    fn test_split_branch() {
        let manager = BranchManager::new(BranchConfig::default());
        let branch = manager.create_branch("original", None).unwrap();
        
        // Add some concepts
        let concepts: Vec<_> = (0..5).map(|_| uuid::Uuid::new_v4()).collect();
        for concept in &concepts {
            manager.allocate_concept(&branch, *concept).unwrap();
        }
        
        // Split at position 3
        let new_branch = manager.split_branch(&branch, 3, "split").unwrap();
        
        // Check concepts distribution
        let original = manager.get_branch(&branch).unwrap();
        let split = manager.get_branch(&new_branch).unwrap();
        
        assert_eq!(original.read().data.concept_ids.len(), 3);
        assert_eq!(split.read().data.concept_ids.len(), 2);
    }
    
    #[test]
    fn test_merge_branches() {
        let manager = BranchManager::new(BranchConfig::default());
        let parent = manager.create_branch("parent", None).unwrap();
        let branch1 = manager.create_branch("branch1", Some(parent.clone())).unwrap();
        let branch2 = manager.create_branch("branch2", Some(parent.clone())).unwrap();
        
        // Add concepts to both branches
        let concept1 = uuid::Uuid::new_v4();
        let concept2 = uuid::Uuid::new_v4();
        let concept3 = uuid::Uuid::new_v4();
        
        manager.allocate_concept(&branch1, concept1).unwrap();
        manager.allocate_concept(&branch1, concept2).unwrap();
        manager.allocate_concept(&branch2, concept2).unwrap(); // Duplicate
        manager.allocate_concept(&branch2, concept3).unwrap();
        
        // Merge branches
        let merged = manager.merge_branches(&branch1, &branch2, "merged").unwrap();
        
        let merged_branch = manager.get_branch(&merged).unwrap();
        let concepts = &merged_branch.read().data.concept_ids;
        
        assert_eq!(concepts.len(), 3); // No duplicates
        assert!(concepts.contains(&concept1));
        assert!(concepts.contains(&concept2));
        assert!(concepts.contains(&concept3));
    }
    
    #[test]
    fn test_find_common_ancestor() {
        let manager = BranchManager::new(BranchConfig::default());
        
        let trunk = manager.create_branch("trunk", None).unwrap();
        let branch1 = manager.create_branch("branch1", Some(trunk.clone())).unwrap();
        let branch2 = manager.create_branch("branch2", Some(trunk.clone())).unwrap();
        let leaf1 = manager.create_branch("leaf1", Some(branch1.clone())).unwrap();
        let leaf2 = manager.create_branch("leaf2", Some(branch2.clone())).unwrap();
        
        // Find common ancestor
        let common = manager.find_common_ancestor(&leaf1, &leaf2).unwrap();
        
        // The common ancestor of leaf1 and leaf2 should be trunk
        // But we need to check the ancestry correctly
        let ancestry1 = manager.get_ancestry(&leaf1).unwrap();
        let ancestry2 = manager.get_ancestry(&leaf2).unwrap();
        
        // Both should contain trunk
        assert!(ancestry1.contains(&trunk));
        assert!(ancestry2.contains(&trunk));
        
        // The common ancestor should be trunk (last common element)
        assert_eq!(common, Some(trunk));
    }
    
    #[test]
    fn test_find_by_state() {
        let manager = BranchManager::new(BranchConfig::default());
        
        let branch1 = manager.create_branch("branch1", None).unwrap();
        let _branch2 = manager.create_branch("branch2", None).unwrap();
        
        // Change state of branch1
        manager.update_branch_state(&branch1, ConsolidationState::Consolidating).unwrap();
        
        let consolidating = manager.find_by_state(ConsolidationState::Consolidating);
        assert_eq!(consolidating.len(), 1);
        assert_eq!(consolidating[0], branch1);
        
        let active = manager.find_by_state(ConsolidationState::WorkingMemory);
        assert_eq!(active.len(), 2); // root and branch2
    }
    
    #[test]
    fn test_event_subscription() {
        let manager = BranchManager::new(BranchConfig::default());
        let mut receiver = manager.subscribe();
        
        let branch_id = manager.create_branch("test", None).unwrap();
        
        // Should receive creation event
        let event = receiver.try_recv().unwrap();
        match event {
            BranchEvent::Created { branch_id: id, .. } => {
                assert_eq!(id, branch_id);
            }
            _ => panic!("Expected Created event"),
        }
    }
    
    #[tokio::test]
    async fn test_consolidation_async() {
        let manager = BranchManager::new(BranchConfig::default());
        
        let parent = manager.create_branch("parent", None).unwrap();
        let child = manager.create_branch("child", Some(parent.clone())).unwrap();
        
        // Add concepts
        let concept = uuid::Uuid::new_v4();
        manager.allocate_concept(&child, concept).unwrap();
        
        // Attempt consolidation
        let result = manager.consolidate_branch(&child).await;
        assert!(result.is_ok());
        
        let stats = manager.stats();
        assert_eq!(stats.consolidation_attempts, 1);
    }
}