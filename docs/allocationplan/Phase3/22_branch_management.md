# Task 22: Branch Management and Version Merging
**Estimated Time**: 15-20 minutes
**Dependencies**: 21_temporal_versioning.md
**Stage**: Advanced Features

## Objective
Implement comprehensive branch management system with version branching, merging, and conflict resolution capabilities that enables parallel development workflows and experimental feature testing within the knowledge graph.

## Specific Requirements

### 1. Branch Creation and Management
- Create isolated branches from any version point
- Branch-aware property inheritance and resolution
- Branch metadata tracking and lineage preservation
- Efficient branch switching and isolation

### 2. Version Merging Operations
- Three-way merge algorithms for version reconciliation
- Automatic conflict detection and resolution strategies
- Manual conflict resolution interface and workflows
- Merge validation and consistency checking

### 3. Branch Operations and Workflows
- Cherry-picking specific changes between branches
- Branch comparison and diff visualization
- Branch cleanup and garbage collection
- Branch access control and permissions

## Implementation Steps

### 1. Create Branch Management Core System
```rust
// src/inheritance/temporal/branch_manager.rs
use std::collections::{HashMap, HashSet, BTreeMap};
use chrono::{DateTime, Utc};
use uuid::Uuid;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone)]
pub struct BranchManager {
    branches: Arc<RwLock<HashMap<String, Branch>>>,
    branch_index: Arc<RwLock<BranchIndex>>,
    merge_engine: Arc<MergeEngine>,
    conflict_resolver: Arc<ConflictResolver>,
    access_control: Arc<BranchAccessControl>,
    versioning_system: Arc<TemporalVersioningSystem>,
    config: BranchConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Branch {
    pub branch_id: String,
    pub branch_name: String,
    pub parent_branch: Option<String>,
    pub branch_point: BranchPoint,
    pub head_version: DateTime<Utc>,
    pub branch_metadata: BranchMetadata,
    pub isolation_level: IsolationLevel,
    pub permissions: BranchPermissions,
    pub merge_history: Vec<MergeRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchPoint {
    pub timestamp: DateTime<Utc>,
    pub source_branch: String,
    pub source_version: Uuid,
    pub reason: String,
    pub creator: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchMetadata {
    pub created_at: DateTime<Utc>,
    pub created_by: Option<String>,
    pub description: Option<String>,
    pub tags: Vec<String>,
    pub status: BranchStatus,
    pub last_activity: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BranchStatus {
    Active,
    Merged,
    Abandoned,
    Protected,
    ReadOnly,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IsolationLevel {
    Full,      // Complete isolation, no inheritance from parent
    Partial,   // Selective inheritance with overrides
    Shared,    // Shared baseline with branch-specific changes
}

impl BranchManager {
    pub async fn new(
        versioning_system: Arc<TemporalVersioningSystem>,
        config: BranchConfig,
    ) -> Result<Self, BranchError> {
        let branches = Arc::new(RwLock::new(HashMap::new()));
        let branch_index = Arc::new(RwLock::new(BranchIndex::new()));
        let merge_engine = Arc::new(MergeEngine::new(config.merge_config.clone()));
        let conflict_resolver = Arc::new(ConflictResolver::new(config.conflict_config.clone()));
        let access_control = Arc::new(BranchAccessControl::new(config.access_config.clone()));
        
        // Create main branch
        let main_branch = Branch {
            branch_id: "main".to_string(),
            branch_name: "main".to_string(),
            parent_branch: None,
            branch_point: BranchPoint {
                timestamp: Utc::now(),
                source_branch: "root".to_string(),
                source_version: Uuid::new_v4(),
                reason: "Initial branch".to_string(),
                creator: None,
            },
            head_version: Utc::now(),
            branch_metadata: BranchMetadata {
                created_at: Utc::now(),
                created_by: None,
                description: Some("Main development branch".to_string()),
                tags: vec!["main".to_string()],
                status: BranchStatus::Active,
                last_activity: Utc::now(),
            },
            isolation_level: IsolationLevel::Full,
            permissions: BranchPermissions::default(),
            merge_history: Vec::new(),
        };
        
        let mut branches_map = HashMap::new();
        branches_map.insert("main".to_string(), main_branch);
        *branches.write().await = branches_map;
        
        Ok(Self {
            branches,
            branch_index,
            merge_engine,
            conflict_resolver,
            access_control,
            versioning_system,
            config,
        })
    }
    
    pub async fn create_branch(
        &self,
        branch_name: String,
        source_branch: &str,
        branch_point: Option<DateTime<Utc>>,
        isolation_level: IsolationLevel,
        metadata: BranchMetadata,
    ) -> Result<Branch, BranchCreationError> {
        let branch_id = Uuid::new_v4().to_string();
        let creation_time = Utc::now();
        
        // Validate source branch exists
        let branches = self.branches.read().await;
        let source = branches.get(source_branch)
            .ok_or_else(|| BranchCreationError::SourceBranchNotFound(source_branch.to_string()))?;
        
        // Determine branch point
        let effective_branch_point = branch_point.unwrap_or(source.head_version);
        
        // Validate branch point exists in source branch
        let source_version = self.versioning_system
            .get_node_at_time("branch_metadata", effective_branch_point)
            .await?
            .ok_or(BranchCreationError::InvalidBranchPoint)?;
        
        let new_branch = Branch {
            branch_id: branch_id.clone(),
            branch_name: branch_name.clone(),
            parent_branch: Some(source_branch.to_string()),
            branch_point: BranchPoint {
                timestamp: effective_branch_point,
                source_branch: source_branch.to_string(),
                source_version: source_version.version_id,
                reason: metadata.description.clone().unwrap_or_default(),
                creator: metadata.created_by.clone(),
            },
            head_version: effective_branch_point,
            branch_metadata: metadata,
            isolation_level,
            permissions: BranchPermissions::default(),
            merge_history: Vec::new(),
        };
        
        drop(branches); // Release read lock
        
        // Add branch to collection
        let mut branches = self.branches.write().await;
        branches.insert(branch_id.clone(), new_branch.clone());
        
        // Update branch index
        let mut index = self.branch_index.write().await;
        index.add_branch(&new_branch).await?;
        
        info!("Created branch '{}' from '{}' at {}", branch_name, source_branch, effective_branch_point);
        
        Ok(new_branch)
    }
    
    pub async fn merge_branches(
        &self,
        source_branch: &str,
        target_branch: &str,
        merge_strategy: MergeStrategy,
        merge_metadata: MergeMetadata,
    ) -> Result<MergeResult, MergeError> {
        let merge_start = Instant::now();
        let merge_id = Uuid::new_v4().to_string();
        
        // Validate branches exist and are compatible
        let branches = self.branches.read().await;
        let source = branches.get(source_branch)
            .ok_or_else(|| MergeError::BranchNotFound(source_branch.to_string()))?;
        let target = branches.get(target_branch)
            .ok_or_else(|| MergeError::BranchNotFound(target_branch.to_string()))?;
        
        // Check merge permissions
        self.access_control.check_merge_permission(source, target, &merge_metadata.user).await?;
        
        // Find common ancestor
        let common_ancestor = self.find_common_ancestor(source, target).await?;
        
        // Collect changes since common ancestor
        let source_changes = self.collect_branch_changes(source, &common_ancestor).await?;
        let target_changes = self.collect_branch_changes(target, &common_ancestor).await?;
        
        drop(branches); // Release read lock
        
        // Perform three-way merge
        let merge_result = self.merge_engine.perform_merge(
            &common_ancestor,
            &source_changes,
            &target_changes,
            merge_strategy,
        ).await?;
        
        // Handle conflicts if any
        let resolved_conflicts = if !merge_result.conflicts.is_empty() {
            self.conflict_resolver.resolve_conflicts(
                &merge_result.conflicts,
                merge_metadata.conflict_resolution_strategy,
            ).await?
        } else {
            Vec::new()
        };
        
        // Apply merge to target branch
        let final_merge_result = self.apply_merge_result(
            target_branch,
            &merge_result,
            &resolved_conflicts,
            &merge_metadata,
        ).await?;
        
        // Record merge in history
        let merge_record = MergeRecord {
            merge_id: merge_id.clone(),
            timestamp: Utc::now(),
            source_branch: source_branch.to_string(),
            target_branch: target_branch.to_string(),
            merge_strategy,
            conflicts_resolved: resolved_conflicts.len(),
            merge_metadata: merge_metadata.clone(),
            merge_duration: merge_start.elapsed(),
        };
        
        // Update branch merge history
        let mut branches = self.branches.write().await;
        if let Some(target_branch_obj) = branches.get_mut(target_branch) {
            target_branch_obj.merge_history.push(merge_record);
            target_branch_obj.branch_metadata.last_activity = Utc::now();
        }
        
        info!(
            "Completed merge from '{}' to '{}' with {} conflicts resolved in {:?}",
            source_branch, target_branch, resolved_conflicts.len(), merge_start.elapsed()
        );
        
        Ok(final_merge_result)
    }
    
    async fn find_common_ancestor(
        &self,
        branch1: &Branch,
        branch2: &Branch,
    ) -> Result<CommonAncestor, AncestorError> {
        // Build lineage for both branches
        let lineage1 = self.build_branch_lineage(&branch1.branch_id).await?;
        let lineage2 = self.build_branch_lineage(&branch2.branch_id).await?;
        
        // Find intersection point
        for ancestor1 in &lineage1 {
            for ancestor2 in &lineage2 {
                if ancestor1.branch_id == ancestor2.branch_id && 
                   ancestor1.timestamp == ancestor2.timestamp {
                    return Ok(CommonAncestor {
                        branch_id: ancestor1.branch_id.clone(),
                        timestamp: ancestor1.timestamp,
                        version_id: ancestor1.version_id,
                    });
                }
            }
        }
        
        Err(AncestorError::NoCommonAncestor)
    }
    
    async fn collect_branch_changes(
        &self,
        branch: &Branch,
        since_ancestor: &CommonAncestor,
    ) -> Result<Vec<ChangeSet>, ChangeCollectionError> {
        let mut changes = Vec::new();
        
        // Get all versions in branch since ancestor
        let versions = self.versioning_system
            .get_versions_in_range(
                &branch.branch_id,
                since_ancestor.timestamp,
                branch.head_version,
            )
            .await?;
        
        for version in versions {
            let change_set = ChangeSet {
                version_id: version.version_id,
                timestamp: version.timestamp,
                node_changes: self.extract_node_changes(&version).await?,
                property_changes: self.extract_property_changes(&version).await?,
                relationship_changes: self.extract_relationship_changes(&version).await?,
                metadata: version.metadata.clone(),
            };
            
            changes.push(change_set);
        }
        
        Ok(changes)
    }
    
    pub async fn cherry_pick(
        &self,
        source_branch: &str,
        target_branch: &str,
        version_id: Uuid,
        cherry_pick_metadata: CherryPickMetadata,
    ) -> Result<CherryPickResult, CherryPickError> {
        let cherry_pick_start = Instant::now();
        
        // Get the specific version to cherry-pick
        let source_version = self.versioning_system
            .get_version_by_id(version_id)
            .await?
            .ok_or(CherryPickError::VersionNotFound(version_id))?;
        
        // Extract changes from the version
        let changes = self.extract_version_changes(&source_version).await?;
        
        // Apply changes to target branch
        let application_result = self.apply_changes_to_branch(
            target_branch,
            &changes,
            &cherry_pick_metadata,
        ).await?;
        
        // Handle any conflicts
        let resolved_conflicts = if !application_result.conflicts.is_empty() {
            self.conflict_resolver.resolve_conflicts(
                &application_result.conflicts,
                cherry_pick_metadata.conflict_resolution_strategy,
            ).await?
        } else {
            Vec::new()
        };
        
        let cherry_pick_result = CherryPickResult {
            cherry_pick_id: Uuid::new_v4().to_string(),
            source_version_id: version_id,
            target_branch: target_branch.to_string(),
            changes_applied: changes.len(),
            conflicts_resolved: resolved_conflicts.len(),
            cherry_pick_duration: cherry_pick_start.elapsed(),
        };
        
        info!(
            "Cherry-picked version {} to branch '{}' in {:?}",
            version_id, target_branch, cherry_pick_start.elapsed()
        );
        
        Ok(cherry_pick_result)
    }
    
    pub async fn get_branch_diff(
        &self,
        branch1: &str,
        branch2: &str,
        diff_options: DiffOptions,
    ) -> Result<BranchDiff, DiffError> {
        let branches = self.branches.read().await;
        let b1 = branches.get(branch1)
            .ok_or_else(|| DiffError::BranchNotFound(branch1.to_string()))?;
        let b2 = branches.get(branch2)
            .ok_or_else(|| DiffError::BranchNotFound(branch2.to_string()))?;
        
        // Find common ancestor for meaningful diff
        let common_ancestor = self.find_common_ancestor(b1, b2).await?;
        
        // Collect changes in both branches since ancestor
        let changes1 = self.collect_branch_changes(b1, &common_ancestor).await?;
        let changes2 = self.collect_branch_changes(b2, &common_ancestor).await?;
        
        // Compute diff
        let diff = BranchDiff {
            branch1: branch1.to_string(),
            branch2: branch2.to_string(),
            common_ancestor: common_ancestor.clone(),
            unique_to_branch1: self.find_unique_changes(&changes1, &changes2),
            unique_to_branch2: self.find_unique_changes(&changes2, &changes1),
            conflicting_changes: self.find_conflicting_changes(&changes1, &changes2),
            diff_statistics: self.compute_diff_statistics(&changes1, &changes2),
        };
        
        Ok(diff)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MergeStrategy {
    FastForward,
    ThreeWay,
    Squash,
    NoFastForward,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeResult {
    pub merge_id: String,
    pub merged_version: NodeVersion,
    pub conflicts_resolved: usize,
    pub merge_duration: Duration,
    pub merge_strategy_used: MergeStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CherryPickResult {
    pub cherry_pick_id: String,
    pub source_version_id: Uuid,
    pub target_branch: String,
    pub changes_applied: usize,
    pub conflicts_resolved: usize,
    pub cherry_pick_duration: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchDiff {
    pub branch1: String,
    pub branch2: String,
    pub common_ancestor: CommonAncestor,
    pub unique_to_branch1: Vec<ChangeSet>,
    pub unique_to_branch2: Vec<ChangeSet>,
    pub conflicting_changes: Vec<ConflictingChange>,
    pub diff_statistics: DiffStatistics,
}
```

### 2. Implement Merge Engine and Conflict Resolution
```rust
// src/inheritance/temporal/merge_engine.rs
#[derive(Debug)]
pub struct MergeEngine {
    merge_strategies: HashMap<MergeStrategy, Box<dyn MergeAlgorithm>>,
    conflict_detector: Arc<ConflictDetector>,
    validation_engine: Arc<MergeValidationEngine>,
    config: MergeConfig,
}

pub trait MergeAlgorithm: Send + Sync {
    fn merge(
        &self,
        base: &CommonAncestor,
        source_changes: &[ChangeSet],
        target_changes: &[ChangeSet],
    ) -> Result<MergeResult, MergeAlgorithmError>;
}

pub struct ThreeWayMergeAlgorithm {
    conflict_threshold: f64,
}

impl MergeAlgorithm for ThreeWayMergeAlgorithm {
    fn merge(
        &self,
        base: &CommonAncestor,
        source_changes: &[ChangeSet],
        target_changes: &[ChangeSet],
    ) -> Result<MergeResult, MergeAlgorithmError> {
        let merge_start = Instant::now();
        let mut merged_properties = HashMap::new();
        let mut conflicts = Vec::new();
        
        // Merge property changes
        for source_change in source_changes {
            for property_change in &source_change.property_changes {
                let property_key = &property_change.property_name;
                
                // Check if target also modified this property
                if let Some(target_change) = self.find_conflicting_property_change(
                    property_key,
                    target_changes,
                ) {
                    // Detect conflict
                    if self.is_conflicting_change(property_change, target_change) {
                        conflicts.push(PropertyConflict {
                            property_name: property_key.clone(),
                            source_value: property_change.new_value.clone(),
                            target_value: target_change.new_value.clone(),
                            base_value: property_change.old_value.clone(),
                            conflict_type: ConflictType::PropertyModification,
                        });
                    } else {
                        // Compatible changes, merge automatically
                        merged_properties.insert(
                            property_key.clone(),
                            self.merge_compatible_changes(property_change, target_change)?,
                        );
                    }
                } else {
                    // No conflict, apply source change
                    merged_properties.insert(
                        property_key.clone(),
                        property_change.new_value.clone(),
                    );
                }
            }
        }
        
        // Apply non-conflicting target changes
        for target_change in target_changes {
            for property_change in &target_change.property_changes {
                let property_key = &property_change.property_name;
                
                if !merged_properties.contains_key(property_key) {
                    merged_properties.insert(
                        property_key.clone(),
                        property_change.new_value.clone(),
                    );
                }
            }
        }
        
        Ok(MergeResult {
            merge_id: Uuid::new_v4().to_string(),
            merged_properties,
            conflicts,
            merge_duration: merge_start.elapsed(),
            merge_strategy_used: MergeStrategy::ThreeWay,
        })
    }
}
```

## Acceptance Criteria

### Functional Requirements
- [ ] Branch creation from any version point with isolation levels
- [ ] Three-way merge algorithm with automatic conflict detection
- [ ] Cherry-pick operations for selective change application
- [ ] Branch comparison and diff visualization
- [ ] Conflict resolution strategies and manual intervention

### Performance Requirements
- [ ] Branch creation time < 15ms
- [ ] Merge operation completion < 100ms for branches with <1000 changes
- [ ] Cherry-pick operation < 50ms per version
- [ ] Branch diff computation < 200ms for moderate-sized branches
- [ ] Conflict detection accuracy > 95%

### Testing Requirements
- [ ] Unit tests for branch creation and management
- [ ] Integration tests for merge operations
- [ ] Conflict resolution scenario tests
- [ ] Performance benchmarks for branch operations

## Validation Steps

1. **Test branch creation and isolation**:
   ```rust
   let manager = BranchManager::new(versioning_system, config).await?;
   let branch = manager.create_branch("feature", "main", None, IsolationLevel::Full, metadata).await?;
   assert_eq!(branch.branch_name, "feature");
   ```

2. **Test merge operations**:
   ```rust
   let merge_result = manager.merge_branches("feature", "main", MergeStrategy::ThreeWay, metadata).await?;
   assert!(merge_result.conflicts_resolved >= 0);
   ```

3. **Run branch management tests**:
   ```bash
   cargo test branch_management_tests --release
   ```

## Files to Create/Modify
- `src/inheritance/temporal/branch_manager.rs` - Core branch management
- `src/inheritance/temporal/merge_engine.rs` - Merge algorithms and strategies
- `src/inheritance/temporal/conflict_resolver.rs` - Conflict detection and resolution
- `src/inheritance/temporal/branch_index.rs` - Branch indexing and search
- `tests/inheritance/branch_tests.rs` - Branch management test suite

## Success Metrics
- Branch creation latency: <15ms average
- Merge completion time: <100ms for moderate changes
- Conflict detection accuracy: >95%
- Cherry-pick performance: <50ms per operation

## Next Task
Upon completion, proceed to **23_spreading_activation.md** to implement spreading activation search algorithms.