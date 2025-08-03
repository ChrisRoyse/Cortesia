//! Core types for memory branches

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;
use chrono::{DateTime, Utc};
use super::state::{StateError, StateChange};

/// Semantic version with major.minor.patch format
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct SemanticVersion {
    pub major: u16,
    pub minor: u16,
    pub patch: u16,
}

impl SemanticVersion {
    /// Create a new semantic version
    pub fn new(major: u16, minor: u16, patch: u16) -> Self {
        Self { major, minor, patch }
    }
    
    /// Create initial version (1.0.0)
    pub fn initial() -> Self {
        Self::new(1, 0, 0)
    }
    
    /// Increment patch version (1.0.0 -> 1.0.1)
    pub fn increment_patch(&mut self) {
        self.patch += 1;
    }
    
    /// Increment minor version and reset patch (1.0.1 -> 1.1.0)
    pub fn increment_minor(&mut self) {
        self.minor += 1;
        self.patch = 0;
    }
    
    /// Increment major version and reset minor/patch (1.1.1 -> 2.0.0)
    pub fn increment_major(&mut self) {
        self.major += 1;
        self.minor = 0;
        self.patch = 0;
    }
    
    /// Create a new version with incremented patch
    pub fn next_patch(&self) -> Self {
        let mut new_version = self.clone();
        new_version.increment_patch();
        new_version
    }
    
    /// Create a new version with incremented minor
    pub fn next_minor(&self) -> Self {
        let mut new_version = self.clone();
        new_version.increment_minor();
        new_version
    }
    
    /// Create a new version with incremented major
    pub fn next_major(&self) -> Self {
        let mut new_version = self.clone();
        new_version.increment_major();
        new_version
    }
}

impl fmt::Display for SemanticVersion {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl Default for SemanticVersion {
    fn default() -> Self {
        Self::initial()
    }
}

/// Unique identifier for a memory branch
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BranchId(pub uuid::Uuid);

impl BranchId {
    /// Create a new branch ID
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4())
    }
}

impl Default for BranchId {
    fn default() -> Self {
        Self::new()
    }
}

/// A memory branch representing a version of knowledge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryBranch {
    /// Unique identifier
    pub id: BranchId,
    
    /// Human-readable name
    pub name: String,
    
    /// Branch metadata
    pub metadata: BranchMetadata,
    
    /// Parent branch ID (if any)
    pub parent_id: Option<BranchId>,
    
    /// Child branch IDs
    pub children: Vec<BranchId>,
    
    /// Consolidation state (legacy - use state_manager instead)
    pub state: super::ConsolidationState,
    
    /// State transition manager
    pub state_manager: super::StateTransition,
    
    /// Branch-specific data
    pub data: BranchData,
}

/// Metadata about a memory branch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchMetadata {
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    
    /// Last modification timestamp
    pub modified_at: DateTime<Utc>,
    
    /// Version number
    pub version: SemanticVersion,
    
    /// Creator/source of the branch
    pub source: String,
    
    /// Description of branch purpose
    pub description: String,
    
    /// Tags for categorization
    pub tags: Vec<String>,
    
    /// Custom properties
    pub properties: HashMap<String, String>,
}

/// Data stored in a memory branch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchData {
    /// Concept IDs in this branch
    pub concept_ids: Vec<uuid::Uuid>,
    
    /// Concept properties indexed by concept ID
    pub concept_properties: HashMap<uuid::Uuid, ConceptProperties>,
    
    /// Number of allocations
    pub allocation_count: usize,
    
    /// Divergence score from parent
    pub divergence: f32,
    
    /// Confidence score
    pub confidence: f32,
    
    /// Spike pattern signatures
    pub spike_signatures: Vec<SpikeSignature>,
}

/// Compressed spike pattern signature
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SpikeSignature {
    /// Hash of the spike pattern
    pub hash: u64,
    
    /// Timestamp when recorded
    pub timestamp: DateTime<Utc>,
    
    /// Average spike rate
    pub avg_rate: f32,
    
    /// Pattern complexity
    pub complexity: f32,
}

/// Properties of a concept including metadata and spike patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptProperties {
    /// Concept unique identifier
    pub concept_id: uuid::Uuid,
    
    /// When the concept was last modified
    pub modified_at: DateTime<Utc>,
    
    /// Version identifier
    pub version: SemanticVersion,
    
    /// Concept value/data (simplified as string for now)
    pub value: String,
    
    /// Associated spike signatures
    pub spike_signatures: Vec<SpikeSignature>,
    
    /// Confidence score for this concept
    pub confidence: f32,
    
    /// Additional properties
    pub properties: HashMap<String, String>,
}

impl ConceptProperties {
    /// Check if spike signatures match with another concept's signatures
    pub fn spike_signatures_match(&self, other: &ConceptProperties) -> bool {
        if self.spike_signatures.len() != other.spike_signatures.len() {
            return false;
        }
        
        // Compare signature hashes and patterns
        for (sig1, sig2) in self.spike_signatures.iter().zip(other.spike_signatures.iter()) {
            if sig1.hash != sig2.hash || 
               (sig1.avg_rate - sig2.avg_rate).abs() > 0.1 ||
               (sig1.complexity - sig2.complexity).abs() > 0.1 {
                return false;
            }
        }
        
        true
    }
}

/// Relationship between branches
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BranchRelationship {
    /// Direct parent-child relationship
    Parent,
    
    /// Direct child of parent
    Child,
    
    /// Share same parent
    Sibling,
    
    /// Ancestor relationship
    Ancestor(u32), // generations removed
    
    /// Descendant relationship
    Descendant(u32), // generations removed
    
    /// No direct relationship
    Unrelated,
}

impl MemoryBranch {
    /// Create a new memory branch
    pub fn new(name: impl Into<String>) -> Self {
        let now = Utc::now();
        let initial_state = super::ConsolidationState::Active;
        
        Self {
            id: BranchId::new(),
            name: name.into(),
            metadata: BranchMetadata {
                created_at: now,
                modified_at: now,
                version: SemanticVersion::initial(),
                source: "manual".to_string(),
                description: String::new(),
                tags: Vec::new(),
                properties: HashMap::new(),
            },
            parent_id: None,
            children: Vec::new(),
            state: initial_state,
            state_manager: super::StateTransition::new(initial_state),
            data: BranchData {
                concept_ids: Vec::new(),
                concept_properties: HashMap::new(),
                allocation_count: 0,
                divergence: 0.0,
                confidence: 1.0,
                spike_signatures: Vec::new(),
            },
        }
    }
    
    /// Create a child branch
    pub fn create_child(&mut self, name: impl Into<String>) -> MemoryBranch {
        let mut child = Self::new(name);
        child.parent_id = Some(self.id);
        child.metadata.source = format!("branched from {}", self.name);
        
        // Inherit parent's version as starting point
        child.metadata.version = self.metadata.version.clone();
        
        // Copy some parent data
        child.data.concept_ids = self.data.concept_ids.clone();
        child.data.concept_properties = self.data.concept_properties.clone();
        child.metadata.tags = self.metadata.tags.clone();
        
        // Add to parent's children
        self.children.push(child.id);
        
        child
    }
    
    /// Calculate relationship to another branch
    /// 
    /// Note: This method only handles direct relationships. For ancestor/descendant
    /// relationships, use `relationship_to_with_lookup` which can traverse the branch tree.
    pub fn relationship_to(&self, other: &MemoryBranch) -> BranchRelationship {
        // Direct relationships
        if self.parent_id == Some(other.id) {
            return BranchRelationship::Child;
        }
        
        if other.parent_id == Some(self.id) {
            return BranchRelationship::Parent;
        }
        
        // Sibling check
        if let (Some(p1), Some(p2)) = (self.parent_id, other.parent_id) {
            if p1 == p2 {
                return BranchRelationship::Sibling;
            }
        }
        
        BranchRelationship::Unrelated
    }
    
    /// Calculate relationship to another branch with ancestor/descendant traversal
    /// 
    /// This method can detect indirect relationships by traversing the branch tree.
    /// The lookup function should return the branch for a given BranchId, or None if not found.
    /// 
    /// # Arguments
    /// * `other` - The branch to compare relationships with
    /// * `branch_lookup` - A function that returns a branch by ID for tree traversal
    /// 
    /// # Examples
    /// ```rust
    /// use temporal_memory::{MemoryBranch, BranchId, BranchRelationship};
    /// use std::collections::HashMap;
    /// 
    /// // Create a three-generation hierarchy
    /// let mut grandparent = MemoryBranch::new("grandparent");
    /// let mut parent = grandparent.create_child("parent");
    /// let child = parent.create_child("child");
    /// 
    /// // Create a branch lookup function
    /// let mut branches = HashMap::new();
    /// branches.insert(grandparent.id, &grandparent);
    /// branches.insert(parent.id, &parent);
    /// branches.insert(child.id, &child);
    /// 
    /// let lookup = |id: BranchId| branches.get(&id).copied();
    /// let relationship = child.relationship_to_with_lookup(&grandparent, &lookup);
    /// assert_eq!(relationship, BranchRelationship::Ancestor(2));
    /// ```
    pub fn relationship_to_with_lookup<'a, F>(&self, other: &MemoryBranch, branch_lookup: F) -> BranchRelationship 
    where
        F: Fn(BranchId) -> Option<&'a MemoryBranch>,
    {
        // First check direct relationships
        if self.parent_id == Some(other.id) {
            return BranchRelationship::Child;
        }
        
        if other.parent_id == Some(self.id) {
            return BranchRelationship::Parent;
        }
        
        // Sibling check
        if let (Some(p1), Some(p2)) = (self.parent_id, other.parent_id) {
            if p1 == p2 {
                return BranchRelationship::Sibling;
            }
        }
        
        // Check if other is an ancestor of self
        if let Some(generations) = self.find_ancestor_distance(other.id, &branch_lookup) {
            return BranchRelationship::Ancestor(generations);
        }
        
        // Check if other is a descendant of self
        if let Some(generations) = other.find_ancestor_distance(self.id, &branch_lookup) {
            return BranchRelationship::Descendant(generations);
        }
        
        BranchRelationship::Unrelated
    }
    
    /// Find the distance to an ancestor branch
    /// 
    /// Returns the number of generations to the ancestor, or None if not an ancestor.
    /// Includes cycle detection to prevent infinite loops.
    fn find_ancestor_distance<'a, F>(&self, ancestor_id: BranchId, branch_lookup: &F) -> Option<u32>
    where
        F: Fn(BranchId) -> Option<&'a MemoryBranch>,
    {
        let mut current_id = self.parent_id?; // Start with immediate parent
        let mut generations = 1u32;
        let mut visited = HashSet::new();
        
        loop {
            // Cycle detection
            if !visited.insert(current_id) {
                // We've seen this branch before - cycle detected
                break;
            }
            
            // Check if we found the ancestor
            if current_id == ancestor_id {
                return Some(generations);
            }
            
            // Get the current branch and move to its parent
            let current_branch = branch_lookup(current_id)?;
            current_id = current_branch.parent_id?;
            generations += 1;
            
            // Safety check to prevent excessively deep traversal
            if generations > 1000 {
                break;
            }
        }
        
        None
    }
    
    /// Update modification timestamp
    pub fn touch(&mut self) {
        self.metadata.modified_at = Utc::now();
    }
    
    /// Increment the branch minor version (for consolidation operations)
    pub fn increment_minor_version(&mut self) {
        self.metadata.version.increment_minor();
        self.touch();
    }
    
    /// Increment the branch patch version (for minor updates)
    pub fn increment_patch_version(&mut self) {
        self.metadata.version.increment_patch();
        self.touch();
    }
    
    /// Increment the branch major version (for breaking changes)
    pub fn increment_major_version(&mut self) {
        self.metadata.version.increment_major();
        self.touch();
    }
    
    /// Add a concept to this branch
    pub fn add_concept(&mut self, concept_id: uuid::Uuid) {
        if !self.data.concept_ids.contains(&concept_id) {
            self.data.concept_ids.push(concept_id);
            
            // Create basic concept properties
            let properties = ConceptProperties {
                concept_id,
                modified_at: Utc::now(),
                version: self.metadata.version.clone(),
                value: format!("concept_{}", concept_id),
                spike_signatures: Vec::new(),
                confidence: 1.0,
                properties: HashMap::new(),
            };
            
            self.data.concept_properties.insert(concept_id, properties);
            self.data.allocation_count += 1;
            self.touch();
        }
    }
    
    /// Add a concept with specific properties
    pub fn add_concept_with_properties(&mut self, concept_id: uuid::Uuid, properties: ConceptProperties) {
        if !self.data.concept_ids.contains(&concept_id) {
            self.data.concept_ids.push(concept_id);
            self.data.concept_properties.insert(concept_id, properties);
            self.data.allocation_count += 1;
            self.touch();
        }
    }
    
    /// Get concept properties
    pub fn get_concept_properties(&self, concept_id: uuid::Uuid) -> Option<&ConceptProperties> {
        self.data.concept_properties.get(&concept_id)
    }
    
    /// Update concept properties
    pub fn update_concept_properties(&mut self, concept_id: uuid::Uuid, mut properties: ConceptProperties) {
        if self.data.concept_ids.contains(&concept_id) {
            // Increment the concept's version when updating
            properties.version.increment_patch();
            properties.modified_at = Utc::now();
            self.data.concept_properties.insert(concept_id, properties);
            self.touch();
        }
    }
    
    /// Calculate age of the branch
    pub fn age(&self) -> chrono::Duration {
        Utc::now() - self.metadata.created_at
    }
    
    /// Transition to a new state with validation and history tracking
    pub fn transition_state(&mut self, new_state: super::ConsolidationState, reason: String) -> Result<(), StateError> {
        self.state_manager.transition_to(new_state, reason)?;
        self.state = self.state_manager.current();
        self.touch();
        Ok(())
    }
    
    /// Get current state (prefer this over direct state field access)
    pub fn current_state(&self) -> super::ConsolidationState {
        self.state_manager.current()
    }
    
    /// Get state transition history
    pub fn state_history(&self) -> &[StateChange] {
        self.state_manager.history()
    }
    
    /// Check if branch can accept new allocations
    pub fn can_allocate(&self) -> bool {
        self.state_manager.can_allocate()
    }
    
    /// Check if branch is in a terminal state
    pub fn is_terminal(&self) -> bool {
        self.state_manager.is_terminal()
    }
    
    /// Validate if the branch can transition to a specific state
    pub fn can_transition_to(&self, new_state: super::ConsolidationState) -> bool {
        self.state_manager.current().can_transition_to(new_state)
    }
}

impl Default for BranchMetadata {
    fn default() -> Self {
        let now = Utc::now();
        Self {
            created_at: now,
            modified_at: now,
            version: SemanticVersion::initial(),
            source: "unknown".to_string(),
            description: String::new(),
            tags: Vec::new(),
            properties: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::ConsolidationState;
    
    #[test]
    fn test_branch_creation() {
        let branch = MemoryBranch::new("test_branch");
        
        assert_eq!(branch.name, "test_branch");
        assert!(branch.parent_id.is_none());
        assert!(branch.children.is_empty());
        assert_eq!(branch.metadata.version, SemanticVersion::initial());
    }
    
    #[test]
    fn test_child_branch() {
        let mut parent = MemoryBranch::new("parent");
        let child = parent.create_child("child");
        
        assert_eq!(child.parent_id, Some(parent.id));
        assert!(parent.children.contains(&child.id));
        assert_eq!(child.relationship_to(&parent), BranchRelationship::Child);
        assert_eq!(parent.relationship_to(&child), BranchRelationship::Parent);
    }
    
    #[test]
    fn test_sibling_relationship() {
        let mut parent = MemoryBranch::new("parent");
        let child1 = parent.create_child("child1");
        let child2 = parent.create_child("child2");
        
        assert_eq!(child1.relationship_to(&child2), BranchRelationship::Sibling);
        assert_eq!(child2.relationship_to(&child1), BranchRelationship::Sibling);
    }
    
    #[test]
    fn test_concept_addition() {
        let mut branch = MemoryBranch::new("test");
        let concept_id = uuid::Uuid::new_v4();
        
        branch.add_concept(concept_id);
        assert!(branch.data.concept_ids.contains(&concept_id));
        assert_eq!(branch.data.allocation_count, 1);
        
        // Adding same concept again shouldn't duplicate
        branch.add_concept(concept_id);
        assert_eq!(branch.data.concept_ids.len(), 1);
        assert_eq!(branch.data.allocation_count, 1);
    }
    
    #[test]
    fn test_state_transition_integration() {
        let mut branch = MemoryBranch::new("test_state");
        
        // Initial state should be Active
        assert_eq!(branch.current_state(), ConsolidationState::Active);
        assert_eq!(branch.state, ConsolidationState::Active);
        assert!(branch.can_allocate());
        assert!(!branch.is_terminal());
        
        // Test transition to Consolidating
        assert!(branch.transition_state(
            ConsolidationState::Consolidating,
            "Testing state transition".to_string()
        ).is_ok());
        
        assert_eq!(branch.current_state(), ConsolidationState::Consolidating);
        assert_eq!(branch.state, ConsolidationState::Consolidating);
        assert!(!branch.can_allocate());
        assert!(!branch.is_terminal());
        
        // Check history
        let history = branch.state_history();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].from, ConsolidationState::Active);
        assert_eq!(history[0].to, ConsolidationState::Consolidating);
        assert_eq!(history[0].reason, "Testing state transition");
        
        // Test transition to Consolidated
        assert!(branch.transition_state(
            ConsolidationState::Consolidated,
            "Testing completion".to_string()
        ).is_ok());
        
        assert_eq!(branch.current_state(), ConsolidationState::Consolidated);
        assert_eq!(branch.state, ConsolidationState::Consolidated);
        assert!(!branch.can_allocate());
        assert!(branch.is_terminal());
        
        // Check history again
        let history = branch.state_history();
        assert_eq!(history.len(), 2);
        assert_eq!(history[1].from, ConsolidationState::Consolidating);
        assert_eq!(history[1].to, ConsolidationState::Consolidated);
    }
    
    #[test]
    fn test_invalid_state_transition() {
        let mut branch = MemoryBranch::new("test_invalid");
        
        // Try invalid transition (Active -> Consolidated directly)
        let result = branch.transition_state(
            ConsolidationState::Consolidated,
            "Invalid transition".to_string()
        );
        
        assert!(result.is_err());
        assert_eq!(branch.current_state(), ConsolidationState::Active);
        assert_eq!(branch.state, ConsolidationState::Active);
        
        // History should be empty after failed transition
        assert_eq!(branch.state_history().len(), 0);
    }
    
    #[test]
    fn test_can_transition_validation() {
        let branch = MemoryBranch::new("test_validation");
        
        // From Active state
        assert!(branch.can_transition_to(ConsolidationState::Consolidating));
        assert!(branch.can_transition_to(ConsolidationState::Conflicted));
        assert!(branch.can_transition_to(ConsolidationState::Active));
        assert!(!branch.can_transition_to(ConsolidationState::Consolidated));
    }
    
    #[test]
    fn test_ancestor_descendant_relationships() {
        // Create a three-generation hierarchy
        let mut grandparent = MemoryBranch::new("grandparent");
        let mut parent = grandparent.create_child("parent");
        let child = parent.create_child("child");
        
        // Create a branch lookup function
        let mut branches = std::collections::HashMap::new();
        branches.insert(grandparent.id, &grandparent);
        branches.insert(parent.id, &parent);
        branches.insert(child.id, &child);
        
        let lookup = |id: BranchId| branches.get(&id).copied();
        
        // Test ancestor relationships
        assert_eq!(
            child.relationship_to_with_lookup(&grandparent, &lookup),
            BranchRelationship::Ancestor(2)
        );
        // Note: Direct parent-child relationship takes precedence over ancestor
        assert_eq!(
            child.relationship_to_with_lookup(&parent, &lookup),
            BranchRelationship::Child
        );
        
        // Test descendant relationships
        assert_eq!(
            grandparent.relationship_to_with_lookup(&child, &lookup),
            BranchRelationship::Descendant(2)
        );
        // Note: Direct parent-child relationship takes precedence over descendant
        assert_eq!(
            parent.relationship_to_with_lookup(&child, &lookup),
            BranchRelationship::Parent
        );
        
        // Test direct relationships still work
        assert_eq!(
            child.relationship_to_with_lookup(&parent, &lookup),
            BranchRelationship::Child
        );
        assert_eq!(
            parent.relationship_to_with_lookup(&child, &lookup),
            BranchRelationship::Parent
        );
    }
    
    #[test]
    fn test_sibling_relationships_with_lookup() {
        let mut parent = MemoryBranch::new("parent");
        let child1 = parent.create_child("child1");
        let child2 = parent.create_child("child2");
        
        let mut branches = std::collections::HashMap::new();
        branches.insert(parent.id, &parent);
        branches.insert(child1.id, &child1);
        branches.insert(child2.id, &child2);
        
        let lookup = |id: BranchId| branches.get(&id).copied();
        
        // Siblings should be detected correctly
        assert_eq!(
            child1.relationship_to_with_lookup(&child2, &lookup),
            BranchRelationship::Sibling
        );
        assert_eq!(
            child2.relationship_to_with_lookup(&child1, &lookup),
            BranchRelationship::Sibling
        );
    }
    
    #[test]
    fn test_unrelated_branches() {
        let branch1 = MemoryBranch::new("branch1");
        let mut branch2_parent = MemoryBranch::new("branch2_parent");
        let branch2 = branch2_parent.create_child("branch2");
        
        let mut branches = std::collections::HashMap::new();
        branches.insert(branch1.id, &branch1);
        branches.insert(branch2_parent.id, &branch2_parent);
        branches.insert(branch2.id, &branch2);
        
        let lookup = |id: BranchId| branches.get(&id).copied();
        
        // Branches with no common ancestry should be unrelated
        assert_eq!(
            branch1.relationship_to_with_lookup(&branch2, &lookup),
            BranchRelationship::Unrelated
        );
        assert_eq!(
            branch2.relationship_to_with_lookup(&branch1, &lookup),
            BranchRelationship::Unrelated
        );
    }
    
    #[test]
    fn test_cycle_detection() {
        let mut branch1 = MemoryBranch::new("branch1");
        let mut branch2 = MemoryBranch::new("branch2");
        let mut branch3 = MemoryBranch::new("branch3");
        
        // Create a cycle: branch1 -> branch2 -> branch3 -> branch1
        branch1.parent_id = Some(branch2.id);
        branch2.parent_id = Some(branch3.id);
        branch3.parent_id = Some(branch1.id);
        
        let mut branches = std::collections::HashMap::new();
        branches.insert(branch1.id, &branch1);
        branches.insert(branch2.id, &branch2);
        branches.insert(branch3.id, &branch3);
        
        let lookup = |id: BranchId| branches.get(&id).copied();
        
        // When trying to find an ancestor, should detect cycle and return unrelated
        // Since branch1's immediate parent is branch2, it's a direct Child relationship
        assert_eq!(
            branch1.relationship_to_with_lookup(&branch2, &lookup),
            BranchRelationship::Child
        );
        
        // But when looking for an ancestor that would require cycle traversal,
        // it should return unrelated due to cycle detection
        let unrelated_branch = MemoryBranch::new("unrelated");
        assert_eq!(
            branch1.relationship_to_with_lookup(&unrelated_branch, &lookup),
            BranchRelationship::Unrelated
        );
    }
    
    #[test]
    fn test_deep_hierarchy() {
        // Create a 5-generation hierarchy to test depth limits
        let mut current = MemoryBranch::new("gen0");
        let mut all_branches = std::collections::HashMap::new();
        all_branches.insert(current.id, current.clone());
        
        let mut generations = vec![current.clone()];
        
        for i in 1..=5 {
            let child = current.create_child(&format!("gen{}", i));
            all_branches.insert(child.id, child.clone());
            generations.push(child.clone());
            current = child;
        }
        
        // Update the map with modified branches (after children were added)
        for gen in &generations {
            all_branches.insert(gen.id, gen.clone());
        }
        
        let lookup = |id: BranchId| all_branches.get(&id);
        
        // Test that the deepest child recognizes the root as ancestor
        let deepest = &generations[5];
        let root = &generations[0];
        
        assert_eq!(
            deepest.relationship_to_with_lookup(root, &lookup),
            BranchRelationship::Ancestor(5)
        );
        
        // Test that root recognizes deepest as descendant
        assert_eq!(
            root.relationship_to_with_lookup(deepest, &lookup),
            BranchRelationship::Descendant(5)
        );
    }
    
    #[test]
    fn test_branch_version_operations() {
        let mut branch = MemoryBranch::new("test");
        
        // Test different version increment operations
        let initial_version = branch.metadata.version.clone();
        
        branch.increment_patch_version();
        assert_eq!(branch.metadata.version, initial_version.next_patch());
        
        branch.increment_minor_version();
        assert_eq!(branch.metadata.version.to_string(), "1.1.0");
        
        branch.increment_major_version();
        assert_eq!(branch.metadata.version.to_string(), "2.0.0");
    }
    
    #[test]
    fn test_missing_branch_lookup() {
        let mut parent = MemoryBranch::new("parent");
        let child = parent.create_child("child");
        
        // Create a lookup that doesn't contain the parent
        let mut branches = std::collections::HashMap::new();
        branches.insert(child.id, &child);
        // Intentionally omit parent
        
        let lookup = |id: BranchId| branches.get(&id).copied();
        
        // Should return unrelated when parent lookup fails
        let unrelated = MemoryBranch::new("unrelated");
        assert_eq!(
            child.relationship_to_with_lookup(&unrelated, &lookup),
            BranchRelationship::Unrelated
        );
    }
    
    #[test]
    fn test_backwards_compatibility() {
        // Ensure original relationship_to method still works
        let mut parent = MemoryBranch::new("parent");
        let child = parent.create_child("child");
        
        // Direct relationships should work without lookup
        assert_eq!(child.relationship_to(&parent), BranchRelationship::Child);
        assert_eq!(parent.relationship_to(&child), BranchRelationship::Parent);
        
        // Ancestor/descendant should return unrelated without lookup
        let mut grandparent = MemoryBranch::new("grandparent");
        let mut intermediate_parent = grandparent.create_child("intermediate");
        let grandchild = intermediate_parent.create_child("grandchild");
        
        // Without lookup, should be unrelated
        assert_eq!(
            grandchild.relationship_to(&grandparent), 
            BranchRelationship::Unrelated
        );
    }
    
    #[test]
    fn test_semantic_versioning() {
        // Test basic semantic version operations
        let mut version = SemanticVersion::new(1, 2, 3);
        assert_eq!(version.to_string(), "1.2.3");
        
        // Test increments
        version.increment_patch();
        assert_eq!(version.to_string(), "1.2.4");
        
        version.increment_minor();
        assert_eq!(version.to_string(), "1.3.0");
        
        version.increment_major();
        assert_eq!(version.to_string(), "2.0.0");
        
        // Test next_* methods
        let v1 = SemanticVersion::new(1, 2, 3);
        assert_eq!(v1.next_patch().to_string(), "1.2.4");
        assert_eq!(v1.next_minor().to_string(), "1.3.0");
        assert_eq!(v1.next_major().to_string(), "2.0.0");
        assert_eq!(v1.to_string(), "1.2.3"); // Original unchanged
    }
    
    #[test]
    fn test_branch_version_inheritance() {
        let mut parent = MemoryBranch::new("parent");
        
        // Parent starts with initial version
        assert_eq!(parent.metadata.version, SemanticVersion::initial());
        
        // Increment parent version
        parent.increment_minor_version();
        assert_eq!(parent.metadata.version.to_string(), "1.1.0");
        
        // Child should inherit parent's current version
        let child = parent.create_child("child");
        assert_eq!(child.metadata.version.to_string(), "1.1.0");
        
        // Parent and child versions should be independent after creation
        let mut child_mut = child;
        child_mut.increment_patch_version();
        assert_eq!(child_mut.metadata.version.to_string(), "1.1.1");
        assert_eq!(parent.metadata.version.to_string(), "1.1.0");
    }
    
    #[test]
    fn test_concept_version_tracking() {
        let mut branch = MemoryBranch::new("test");
        let concept_id = uuid::Uuid::new_v4();
        
        // Add concept
        branch.add_concept(concept_id);
        let initial_props = branch.get_concept_properties(concept_id).unwrap().clone();
        assert_eq!(initial_props.version, SemanticVersion::initial());
        
        // Update concept properties - should increment version
        let mut updated_props = initial_props.clone();
        updated_props.value = "updated_value".to_string();
        branch.update_concept_properties(concept_id, updated_props);
        
        let final_props = branch.get_concept_properties(concept_id).unwrap();
        assert_eq!(final_props.version.to_string(), "1.0.1");
        assert_eq!(final_props.value, "updated_value");
        assert_ne!(final_props.modified_at, initial_props.modified_at);
    }
}