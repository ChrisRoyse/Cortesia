# MicroPhase1 Branch Management: Detailed Execution Sequence

## Overview
This document provides explicit task sequencing for MicroPhase1 Branch Management implementation. It defines exact execution order, dependencies, parallel opportunities, and validation checkpoints for AI-driven development.

**Total Estimated Duration**: 12 hours (1.5 work days)  
**Prerequisites**: Phase 2 (Neuromorphic Allocation Engine), Phase 3 (Knowledge Graph Schema) completed

## Atomic Task Breakdown and Dependencies

### Task Identification System
- **1.1.X**: Core data structures  
- **1.2.X**: Copy-on-Write implementation
- **1.3.X**: Branch manager operations
- **1.4.X**: Consolidation state machine
- **1.5.X**: Testing and validation

## Phase 1A: Foundation Layer (Parallel Execution - 90 minutes)

### Critical Path Analysis
**Primary Path**: 1.1.1 → 1.1.2 → 1.2.1 → 1.3.1 (Serial, blocking)  
**Secondary Paths**: 1.1.3, 1.1.4, 1.4.1 (Parallel, non-blocking)

### Execution Block 1A.1 (0-30 minutes) - Parallel Launch

#### Task 1.1.1: BranchId Type Implementation
**Duration**: 15 minutes  
**Dependencies**: None  
**Parallel Opportunity**: ✅ Can run with 1.1.3, 1.1.4  

**AI Execution Prompt**:
```rust
// Create src/temporal/branch/types.rs - Start with BranchId only
use uuid::Uuid;
use std::fmt;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BranchId(Uuid);

impl BranchId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
    
    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }
    
    pub fn as_uuid(&self) -> Uuid {
        self.0
    }
}

impl fmt::Display for BranchId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Default for BranchId {
    fn default() -> Self {
        Self::new()
    }
}
```

**Acceptance Criteria**:
- [ ] BranchId compiles without warnings
- [ ] UUID generation works correctly  
- [ ] Serialization/deserialization functional
- [ ] Display trait implemented correctly

#### Task 1.1.3: ConsolidationState Enum (Parallel)
**Duration**: 15 minutes  
**Dependencies**: None  
**Parallel Opportunity**: ✅ Can run with 1.1.1, 1.1.4

**AI Execution Prompt**:
```rust
// Add to src/temporal/branch/types.rs
use std::time::{Duration, SystemTime};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsolidationState {
    WorkingMemory,  // 0-30 seconds
    ShortTerm,      // 30 seconds - 1 hour  
    Consolidating,  // 1-24 hours
    LongTerm,       // >24 hours
}

impl ConsolidationState {
    pub fn duration_threshold(&self) -> Duration {
        match self {
            Self::WorkingMemory => Duration::from_secs(30),
            Self::ShortTerm => Duration::from_secs(3600),      // 1 hour
            Self::Consolidating => Duration::from_secs(86400), // 24 hours
            Self::LongTerm => Duration::MAX,
        }
    }
    
    pub fn next_state(&self) -> Option<Self> {
        match self {
            Self::WorkingMemory => Some(Self::ShortTerm),
            Self::ShortTerm => Some(Self::Consolidating),
            Self::Consolidating => Some(Self::LongTerm),
            Self::LongTerm => None,
        }
    }
    
    pub fn should_transition(&self, age: Duration) -> bool {
        age >= self.duration_threshold()
    }
}
```

**Acceptance Criteria**:
- [ ] All enum variants defined correctly
- [ ] Duration thresholds match biological timing
- [ ] State transition logic is correct
- [ ] Serialization support implemented

#### Task 1.1.4: BranchMetadata Struct (Parallel)
**Duration**: 20 minutes  
**Dependencies**: None  
**Parallel Opportunity**: ✅ Can run with 1.1.1, 1.1.3

**AI Execution Prompt**:
```rust
// Add to src/temporal/branch/types.rs  
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Serialize, Deserialize)]
pub struct BranchMetadata {
    pub creation_time: SystemTime,
    pub last_modified: SystemTime,
    #[serde(skip)] // Skip atomic for serialization
    access_count: AtomicU64,
    pub neural_pathway_id: Option<String>,
    pub memory_estimation_bytes: u64,
}

impl BranchMetadata {
    pub fn new() -> Self {
        let now = SystemTime::now();
        Self {
            creation_time: now,
            last_modified: now,
            access_count: AtomicU64::new(0),
            neural_pathway_id: None,
            memory_estimation_bytes: 0,
        }
    }
    
    pub fn increment_access(&self) {
        self.access_count.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn get_access_count(&self) -> u64 {
        self.access_count.load(Ordering::Relaxed)
    }
    
    pub fn update_last_modified(&mut self) {
        self.last_modified = SystemTime::now();
    }
    
    pub fn age(&self) -> Duration {
        SystemTime::now().duration_since(self.creation_time)
            .unwrap_or(Duration::ZERO)
    }
}

impl Clone for BranchMetadata {
    fn clone(&self) -> Self {
        Self {
            creation_time: self.creation_time,
            last_modified: self.last_modified,
            access_count: AtomicU64::new(self.access_count.load(Ordering::Relaxed)),
            neural_pathway_id: self.neural_pathway_id.clone(),
            memory_estimation_bytes: self.memory_estimation_bytes,
        }
    }
}
```

**Acceptance Criteria**:
- [ ] Atomic access count operations work correctly
- [ ] Serialization handles atomic fields properly
- [ ] Time tracking functions correctly
- [ ] Memory estimation placeholder implemented

### CHECKPOINT 1A.1 (30 minutes): Basic Types Validation
**Validation Commands**:
```bash
cd C:/code/LLMKG
cargo check --lib
cargo test types:: --lib
```

**Success Criteria**:
- [ ] All basic types compile without warnings
- [ ] UUID generation produces unique IDs
- [ ] Atomic operations work correctly
- [ ] Serialization round-trips successfully

### Execution Block 1A.2 (30-60 minutes) - Sequential Critical Path

#### Task 1.1.2: Branch Struct Implementation  
**Duration**: 30 minutes  
**Dependencies**: 1.1.1 (BranchId), 1.1.3 (ConsolidationState), 1.1.4 (BranchMetadata)  
**Parallel Opportunity**: ❌ Must wait for dependencies

**AI Execution Prompt**:
```rust
// Add to src/temporal/branch/types.rs
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Branch {
    pub id: BranchId,
    pub name: String,
    pub parent: Option<BranchId>,
    pub consolidation_state: ConsolidationState,
    pub head_version: u64,
    pub metadata: BranchMetadata,
}

impl Branch {
    pub fn new(name: String, parent: Option<BranchId>) -> Result<Self, BranchError> {
        // Validate branch name
        if name.is_empty() || name.len() > 50 {
            return Err(BranchError::InvalidName("Name must be 1-50 characters".to_string()));
        }
        
        // Check for special characters
        if !name.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '-') {
            return Err(BranchError::InvalidName("Name can only contain alphanumeric, underscore, and dash".to_string()));
        }
        
        Ok(Self {
            id: BranchId::new(),
            name,
            parent,
            consolidation_state: ConsolidationState::WorkingMemory,
            head_version: 0,
            metadata: BranchMetadata::new(),
        })
    }
    
    pub fn should_consolidate(&self) -> bool {
        self.consolidation_state.should_transition(self.metadata.age())
    }
    
    pub fn advance_consolidation_state(&mut self) {
        if let Some(next_state) = self.consolidation_state.next_state() {
            if self.should_consolidate() {
                self.consolidation_state = next_state;
                self.metadata.update_last_modified();
            }
        }
    }
    
    pub fn estimate_memory_usage(&self) -> u64 {
        // Base branch overhead: ~200 bytes
        let base_size = 200u64;
        base_size + self.metadata.memory_estimation_bytes
    }
}

#[derive(Debug, thiserror::Error)]
pub enum BranchError {
    #[error("Invalid branch name: {0}")]
    InvalidName(String),
    #[error("Branch not found: {0}")]
    NotFound(BranchId),
    #[error("Branch operation failed: {0}")]
    OperationFailed(String),
}
```

**Acceptance Criteria**:
- [ ] Branch creation validates name correctly
- [ ] Consolidation state transitions work
- [ ] Memory estimation provides reasonable values
- [ ] Error handling covers edge cases

### CHECKPOINT 1A.2 (60 minutes): Core Types Complete
**Validation Commands**:
```bash
cargo test branch::types:: --lib
cargo clippy --lib
cargo doc --lib --no-deps
```

**Success Criteria**:
- [ ] All core types functional
- [ ] No clippy warnings
- [ ] Documentation generates correctly
- [ ] Memory estimation within reasonable bounds

### Execution Block 1A.3 (60-90 minutes) - Parallel Preparation

#### Task 1.4.1: Consolidation Timer Foundation (Parallel)
**Duration**: 30 minutes  
**Dependencies**: 1.1.3 (ConsolidationState)  
**Parallel Opportunity**: ✅ Can run with 1.2.1

**AI Execution Prompt**:
```rust
// Create src/temporal/branch/consolidation.rs
use std::time::{Duration, SystemTime};
use tokio::time::{interval, Interval};
use crate::temporal::branch::types::{BranchId, ConsolidationState, Branch};

#[derive(Debug)]
pub struct ConsolidationTimer {
    branch_id: BranchId,
    current_state: ConsolidationState,
    created_at: SystemTime,
    last_check: SystemTime,
}

impl ConsolidationTimer {
    pub fn new(branch_id: BranchId) -> Self {
        let now = SystemTime::now();
        Self {
            branch_id,
            current_state: ConsolidationState::WorkingMemory,
            created_at: now,
            last_check: now,
        }
    }
    
    pub fn age(&self) -> Duration {
        SystemTime::now().duration_since(self.created_at)
            .unwrap_or(Duration::ZERO)
    }
    
    pub fn should_transition(&self) -> bool {
        self.current_state.should_transition(self.age())
    }
    
    pub fn next_state(&self) -> Option<ConsolidationState> {
        self.current_state.next_state()
    }
    
    pub fn update_state(&mut self, new_state: ConsolidationState) {
        self.current_state = new_state;
        self.last_check = SystemTime::now();
    }
}

#[derive(Debug, Clone)]
pub struct StateTransitionEvent {
    pub branch_id: BranchId,
    pub from_state: ConsolidationState,
    pub to_state: ConsolidationState,
    pub timestamp: SystemTime,
}
```

**Acceptance Criteria**:
- [ ] Timer tracks branch state correctly
- [ ] Age calculation is accurate
- [ ] Transition detection works
- [ ] Event structure is complete

## Phase 1B: Core Implementation (Sequential - 180 minutes)

### Execution Block 1B.1 (90-180 minutes) - Copy-on-Write Foundation

#### Task 1.2.1: Page-Based Storage System
**Duration**: 90 minutes  
**Dependencies**: 1.1.1 (BranchId), 1.1.2 (Branch struct)  
**Parallel Opportunity**: ❌ Critical path component

**AI Execution Prompt**:
```rust
// Create src/temporal/branch/cow.rs
use std::collections::HashMap;
use std::sync::Arc;
use dashmap::DashMap;
use serde::{Serialize, Deserialize};
use crate::temporal::branch::types::{BranchId, BranchError};

pub const PAGE_SIZE: usize = 4096;
pub const NODES_PER_PAGE: usize = PAGE_SIZE / 64; // Assuming 64 bytes per node

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PageId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u64);

impl NodeId {
    pub fn to_page_id(&self) -> PageId {
        PageId(self.0 / NODES_PER_PAGE as u64)
    }
    
    pub fn page_offset(&self) -> usize {
        (self.0 % NODES_PER_PAGE as u64) as usize
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Page {
    pub id: PageId,
    pub data: Vec<u8>,
    pub dirty: bool,
    pub node_count: usize,
}

impl Page {
    pub fn new(id: PageId) -> Self {
        Self {
            id,
            data: vec![0u8; PAGE_SIZE],
            dirty: false,
            node_count: 0,
        }
    }
    
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }
    
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }
    
    pub fn memory_usage(&self) -> usize {
        PAGE_SIZE + std::mem::size_of::<Self>()
    }
}

#[derive(Debug)]
pub struct ChangeLog {
    pub branch_id: BranchId,
    pub modified_pages: HashMap<PageId, Page>,
    pub created_nodes: Vec<NodeId>,
    pub deleted_nodes: Vec<NodeId>,
}

impl ChangeLog {
    pub fn new(branch_id: BranchId) -> Self {
        Self {
            branch_id,
            modified_pages: HashMap::new(),
            created_nodes: Vec::new(),
            deleted_nodes: Vec::new(),
        }
    }
    
    pub fn add_modified_page(&mut self, page: Page) {
        self.modified_pages.insert(page.id, page);
    }
    
    pub fn add_created_node(&mut self, node_id: NodeId) {
        self.created_nodes.push(node_id);
    }
    
    pub fn add_deleted_node(&mut self, node_id: NodeId) {
        self.deleted_nodes.push(node_id);
    }
    
    pub fn memory_usage(&self) -> usize {
        let page_memory: usize = self.modified_pages.values()
            .map(|p| p.memory_usage())
            .sum();
        let node_memory = (self.created_nodes.len() + self.deleted_nodes.len()) * 8;
        page_memory + node_memory
    }
}
```

**Acceptance Criteria**:
- [ ] Page size calculations correct (4KB pages)
- [ ] NodeId to PageId mapping works
- [ ] Change tracking functional
- [ ] Memory usage calculation accurate

### CHECKPOINT 1B.1 (180 minutes): Storage Foundation Ready
**Validation Commands**:
```bash
cargo test cow::page:: --lib
cargo test cow::change_log:: --lib
```

**Success Criteria**:
- [ ] Page-based storage working
- [ ] Node ID mapping correct
- [ ] Change tracking functional
- [ ] Memory calculations accurate

### Execution Block 1B.2 (180-270 minutes) - COW Implementation

#### Task 1.2.2: Copy-on-Write Graph Implementation
**Duration**: 90 minutes  
**Dependencies**: 1.2.1 (Page system)  
**Parallel Opportunity**: ❌ Sequential dependency

**AI Execution Prompt**:
```rust
// Continue in src/temporal/branch/cow.rs
use std::sync::RwLock;

#[derive(Debug)]
pub struct CopyOnWriteGraph {
    pub branch_id: BranchId,
    pub base_snapshot: Option<Arc<BaseSnapshot>>,
    pub local_pages: DashMap<PageId, Arc<Page>>,
    pub change_log: RwLock<ChangeLog>,
    pub memory_usage: Arc<std::sync::atomic::AtomicU64>,
}

#[derive(Debug)]
pub struct BaseSnapshot {
    pub pages: HashMap<PageId, Arc<Page>>,
    pub total_nodes: u64,
    pub snapshot_id: u64,
}

impl CopyOnWriteGraph {
    pub fn new(branch_id: BranchId, base_snapshot: Option<Arc<BaseSnapshot>>) -> Self {
        Self {
            branch_id,
            base_snapshot,
            local_pages: DashMap::new(),
            change_log: RwLock::new(ChangeLog::new(branch_id)),
            memory_usage: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }
    
    pub fn read_page(&self, page_id: PageId) -> Result<Arc<Page>, BranchError> {
        // First check local pages (COW writes)
        if let Some(page) = self.local_pages.get(&page_id) {
            return Ok(page.clone());
        }
        
        // Then check base snapshot (COW reads)
        if let Some(base) = &self.base_snapshot {
            if let Some(page) = base.pages.get(&page_id) {
                return Ok(page.clone());
            }
        }
        
        Err(BranchError::OperationFailed(format!("Page {} not found", page_id.0)))
    }
    
    pub fn write_page(&self, mut page: Page) -> Result<(), BranchError> {
        page.mark_dirty();
        
        // Update memory usage
        let page_memory = page.memory_usage() as u64;
        self.memory_usage.fetch_add(page_memory, std::sync::atomic::Ordering::Relaxed);
        
        // Store in local pages (COW write)
        self.local_pages.insert(page.id, Arc::new(page.clone()));
        
        // Update change log
        if let Ok(mut change_log) = self.change_log.write() {
            change_log.add_modified_page(page);
        }
        
        Ok(())
    }
    
    pub fn get_memory_usage(&self) -> u64 {
        self.memory_usage.load(std::sync::atomic::Ordering::Relaxed)
    }
    
    pub fn create_node(&self, node_id: NodeId, data: Vec<u8>) -> Result<(), BranchError> {
        let page_id = node_id.to_page_id();
        
        // Get or create page
        let mut page = match self.read_page(page_id) {
            Ok(existing_page) => (*existing_page).clone(),
            Err(_) => Page::new(page_id),
        };
        
        // Add node data to page
        let offset = node_id.page_offset();
        if offset * 64 + data.len() <= PAGE_SIZE {
            page.data[offset * 64..offset * 64 + data.len()].copy_from_slice(&data);
            page.node_count += 1;
            
            // Write modified page
            self.write_page(page)?;
            
            // Track node creation
            if let Ok(mut change_log) = self.change_log.write() {
                change_log.add_created_node(node_id);
            }
        }
        
        Ok(())
    }
    
    pub fn delete_node(&self, node_id: NodeId) -> Result<(), BranchError> {
        let page_id = node_id.to_page_id();
        
        // Get existing page
        let mut page = self.read_page(page_id)?.as_ref().clone();
        
        // Zero out node data
        let offset = node_id.page_offset();
        page.data[offset * 64..(offset + 1) * 64].fill(0);
        page.node_count = page.node_count.saturating_sub(1);
        
        // Write modified page
        self.write_page(page)?;
        
        // Track node deletion
        if let Ok(mut change_log) = self.change_log.write() {
            change_log.add_deleted_node(node_id);
        }
        
        Ok(())
    }
    
    pub fn get_dirty_pages(&self) -> Vec<PageId> {
        self.local_pages.iter()
            .filter_map(|entry| {
                if entry.is_dirty() { Some(entry.id) } else { None }
            })
            .collect()
    }
}
```

**Acceptance Criteria**:
- [ ] COW semantics working correctly
- [ ] Reads from base, writes to local
- [ ] Memory tracking accurate
- [ ] Node creation/deletion functional

### CHECKPOINT 1B.2 (270 minutes): COW Implementation Complete
**Validation Commands**:
```bash
cargo test cow::graph:: --lib
cargo test cow::memory:: --lib
```

**Success Criteria**:
- [ ] COW reads working correctly
- [ ] COW writes isolated to branch
- [ ] Memory usage starts at 0 for new branches
- [ ] Change tracking complete

## Phase 1C: Branch Operations (Sequential - 240 minutes)

### Execution Block 1C.1 (270-360 minutes) - Branch Manager Core

#### Task 1.3.1: BranchManager Implementation
**Duration**: 90 minutes  
**Dependencies**: 1.1.2 (Branch), 1.2.2 (CopyOnWriteGraph)  
**Parallel Opportunity**: ❌ Critical path component

**AI Execution Prompt**:
```rust
// Create src/temporal/branch/manager.rs
use std::sync::Arc;
use dashmap::DashMap;
use tokio::time::Instant;
use crate::temporal::branch::types::{Branch, BranchId, BranchError};
use crate::temporal::branch::cow::{CopyOnWriteGraph, BaseSnapshot};

pub struct BranchManager {
    branches: DashMap<BranchId, Arc<Branch>>,
    graphs: DashMap<BranchId, Arc<CopyOnWriteGraph>>,
    current_branch: Arc<std::sync::RwLock<Option<BranchId>>>,
    base_snapshots: DashMap<u64, Arc<BaseSnapshot>>,
}

impl BranchManager {
    pub fn new() -> Self {
        Self {
            branches: DashMap::new(),
            graphs: DashMap::new(),
            current_branch: Arc::new(std::sync::RwLock::new(None)),
            base_snapshots: DashMap::new(),
        }
    }
    
    pub async fn create_branch(&self, name: String, parent: Option<BranchId>) -> Result<BranchId, BranchError> {
        let start_time = Instant::now();
        
        // Create branch
        let branch = Branch::new(name, parent)?;
        let branch_id = branch.id;
        
        // Get base snapshot for COW
        let base_snapshot = if let Some(parent_id) = parent {
            // Use parent's current state as base
            self.create_snapshot_from_parent(parent_id).await?
        } else {
            None
        };
        
        // Create COW graph
        let cow_graph = CopyOnWriteGraph::new(branch_id, base_snapshot);
        
        // Store branch and graph
        self.branches.insert(branch_id, Arc::new(branch));
        self.graphs.insert(branch_id, Arc::new(cow_graph));
        
        // Performance validation
        let elapsed = start_time.elapsed();
        if elapsed.as_millis() > 10 {
            log::warn!("Branch creation took {}ms, target is <10ms", elapsed.as_millis());
        }
        
        Ok(branch_id)
    }
    
    pub async fn switch_branch(&self, branch_id: BranchId) -> Result<(), BranchError> {
        let start_time = Instant::now();
        
        // Verify branch exists
        if !self.branches.contains_key(&branch_id) {
            return Err(BranchError::NotFound(branch_id));
        }
        
        // Switch current branch (zero-copy)
        if let Ok(mut current) = self.current_branch.write() {
            *current = Some(branch_id);
        }
        
        // Update access count
        if let Some(branch_ref) = self.branches.get(&branch_id) {
            branch_ref.metadata.increment_access();
        }
        
        // Performance validation
        let elapsed = start_time.elapsed();
        if elapsed.as_millis() > 1 {
            log::warn!("Branch switch took {}ms, target is <1ms", elapsed.as_millis());
        }
        
        Ok(())
    }
    
    pub async fn delete_branch(&self, branch_id: BranchId) -> Result<(), BranchError> {
        // Verify branch exists
        let branch = self.branches.get(&branch_id)
            .ok_or(BranchError::NotFound(branch_id))?;
        
        // Prevent deleting current branch
        if let Ok(current) = self.current_branch.read() {
            if *current == Some(branch_id) {
                return Err(BranchError::OperationFailed("Cannot delete current branch".to_string()));
            }
        }
        
        // Remove branch and graph
        self.branches.remove(&branch_id);
        self.graphs.remove(&branch_id);
        
        Ok(())
    }
    
    pub fn get_current_branch(&self) -> Option<BranchId> {
        self.current_branch.read().ok()?.clone()
    }
    
    pub fn get_branch(&self, branch_id: BranchId) -> Option<Arc<Branch>> {
        self.branches.get(&branch_id).map(|b| b.clone())
    }
    
    pub fn get_graph(&self, branch_id: BranchId) -> Option<Arc<CopyOnWriteGraph>> {
        self.graphs.get(&branch_id).map(|g| g.clone())
    }
    
    pub fn list_branches(&self) -> Vec<BranchId> {
        self.branches.iter().map(|entry| entry.key().clone()).collect()
    }
    
    async fn create_snapshot_from_parent(&self, parent_id: BranchId) -> Result<Option<Arc<BaseSnapshot>>, BranchError> {
        // For now, return None - full implementation requires graph serialization
        // This will be expanded in Phase 1C.2
        Ok(None)
    }
}

impl Default for BranchManager {
    fn default() -> Self {
        Self::new()
    }
}
```

**Acceptance Criteria**:
- [ ] Branch creation <10ms performance target
- [ ] Branch switching <1ms performance target
- [ ] Proper error handling and validation
- [ ] Thread-safe concurrent operations
- [ ] Memory management working

### CHECKPOINT 1C.1 (360 minutes): Branch Manager Functional
**Validation Commands**:
```bash
cargo test manager::create_branch --lib
cargo test manager::switch_branch --lib  
cargo test manager::delete_branch --lib
```

**Success Criteria**:
- [ ] All basic operations working
- [ ] Performance targets met
- [ ] No memory leaks detected
- [ ] Concurrent access safe

### Execution Block 1C.2 (360-450 minutes) - Neural Integration

#### Task 1.3.2: Neural Pathway Integration
**Duration**: 90 minutes  
**Dependencies**: 1.3.1 (BranchManager), Phase 2 (Neuromorphic Allocation Engine)  
**Parallel Opportunity**: ❌ Requires manager foundation

**AI Execution Prompt**:
```rust
// Extend src/temporal/branch/manager.rs with neural integration
use crate::core::allocation::AllocationEngine; // From Phase 2
use crate::core::neural_pathway::NeuralPathway; // From Phase 2

impl BranchManager {
    pub async fn create_branch_with_neural_guidance(
        &self, 
        name: String, 
        parent: Option<BranchId>,
        allocation_engine: &AllocationEngine
    ) -> Result<BranchId, BranchError> {
        let start_time = Instant::now();
        
        // Use cortical column voting for branch placement
        let neural_pathway = allocation_engine.vote_for_branch_placement(&name).await
            .map_err(|e| BranchError::OperationFailed(format!("Neural guidance failed: {}", e)))?;
        
        // Create branch with neural pathway recording
        let mut branch = Branch::new(name, parent)?;
        branch.metadata.neural_pathway_id = Some(neural_pathway.id());
        let branch_id = branch.id;
        
        // Record TTFS timings for branch access patterns
        allocation_engine.record_branch_creation_timing(&neural_pathway, start_time.elapsed()).await;
        
        // Get base snapshot with neural optimization
        let base_snapshot = if let Some(parent_id) = parent {
            self.create_optimized_snapshot(parent_id, &neural_pathway).await?
        } else {
            None
        };
        
        // Create COW graph with neural placement hints
        let cow_graph = CopyOnWriteGraph::new(branch_id, base_snapshot);
        
        // Store with neural pathway metadata
        self.branches.insert(branch_id, Arc::new(branch));
        self.graphs.insert(branch_id, Arc::new(cow_graph));
        
        // Apply lateral inhibition for branch conflict resolution
        allocation_engine.resolve_branch_conflicts(branch_id).await;
        
        let elapsed = start_time.elapsed();
        if elapsed.as_millis() > 10 {
            log::warn!("Neural branch creation took {}ms, target is <10ms", elapsed.as_millis());
        }
        
        Ok(branch_id)
    }
    
    pub async fn suggest_branch_name(&self, context: &str, allocation_engine: &AllocationEngine) -> Vec<String> {
        // Use cortical column consensus for naming suggestions
        allocation_engine.generate_branch_name_suggestions(context).await
            .unwrap_or_else(|_| vec!["new_branch".to_string()])
    }
    
    async fn create_optimized_snapshot(
        &self, 
        parent_id: BranchId, 
        neural_pathway: &NeuralPathway
    ) -> Result<Option<Arc<BaseSnapshot>>, BranchError> {
        // Neural-guided snapshot optimization
        // This will be fully implemented when graph serialization is complete
        Ok(None)
    }
}
```

**Acceptance Criteria**:
- [ ] Neural pathway integration working
- [ ] TTFS timing recording functional
- [ ] Lateral inhibition applied correctly
- [ ] Branch naming suggestions generated
- [ ] Performance targets maintained

### CHECKPOINT 1C.2 (450 minutes): Neural Integration Complete  
**Validation Commands**:
```bash
cargo test manager::neural:: --lib
cargo test integration::phase2:: --lib
```

**Success Criteria**:
- [ ] Neural pathways recorded correctly
- [ ] Integration with Phase 2 working
- [ ] Branch naming suggestions functional
- [ ] No regression in performance

## Phase 1D: Consolidation & Testing (Parallel - 180 minutes)

### Execution Block 1D.1 (450-540 minutes) - State Machine Completion

#### Task 1.4.2: Background Task Scheduler (Parallel)
**Duration**: 60 minutes  
**Dependencies**: 1.4.1 (Consolidation Timer), 1.3.1 (BranchManager)  
**Parallel Opportunity**: ✅ Can run with 1.5.1

**AI Execution Prompt**:
```rust
// Extend src/temporal/branch/consolidation.rs
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use std::collections::HashMap;

pub struct ConsolidationScheduler {
    timers: DashMap<BranchId, ConsolidationTimer>,
    event_sender: mpsc::UnboundedSender<StateTransitionEvent>,
    event_receiver: Option<mpsc::UnboundedReceiver<StateTransitionEvent>>,
    background_task: Option<JoinHandle<()>>,
}

impl ConsolidationScheduler {
    pub fn new() -> Self {
        let (sender, receiver) = mpsc::unbounded_channel();
        Self {
            timers: DashMap::new(),
            event_sender: sender,
            event_receiver: Some(receiver),
            background_task: None,
        }
    }
    
    pub async fn start(&mut self, branch_manager: Arc<BranchManager>) {
        let event_receiver = self.event_receiver.take()
            .expect("Scheduler already started");
        let timers = self.timers.clone();
        
        self.background_task = Some(tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(1));
            let mut event_receiver = event_receiver;
            
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        Self::check_all_timers(&timers, &branch_manager).await;
                    }
                    event = event_receiver.recv() => {
                        if let Some(event) = event {
                            Self::process_transition_event(event, &branch_manager).await;
                        }
                    }
                }
            }
        }));
    }
    
    pub fn add_branch(&self, branch_id: BranchId) {
        self.timers.insert(branch_id, ConsolidationTimer::new(branch_id));
    }
    
    pub fn remove_branch(&self, branch_id: BranchId) {
        self.timers.remove(&branch_id);
    }
    
    async fn check_all_timers(timers: &DashMap<BranchId, ConsolidationTimer>, branch_manager: &BranchManager) {
        for mut timer_ref in timers.iter_mut() {
            let timer = timer_ref.value_mut();
            
            if timer.should_transition() {
                if let Some(next_state) = timer.next_state() {
                    let event = StateTransitionEvent {
                        branch_id: timer.branch_id,
                        from_state: timer.current_state,
                        to_state: next_state,
                        timestamp: SystemTime::now(),
                    };
                    
                    // Update timer state
                    timer.update_state(next_state);
                    
                    // Trigger consolidation process
                    Self::trigger_consolidation(event, branch_manager).await;
                }
            }
        }
    }
    
    async fn process_transition_event(event: StateTransitionEvent, branch_manager: &BranchManager) {
        log::info!("Processing state transition for branch {}: {:?} -> {:?}", 
                   event.branch_id, event.from_state, event.to_state);
        
        // Update branch state
        if let Some(mut branch) = branch_manager.get_branch(event.branch_id) {
            // This requires making Branch fields mutable - will need Arc<RwLock<Branch>>
            // For now, log the transition
            log::info!("Branch {} transitioned to {:?}", event.branch_id, event.to_state);
        }
    }
    
    async fn trigger_consolidation(event: StateTransitionEvent, branch_manager: &BranchManager) {
        match event.to_state {
            ConsolidationState::ShortTerm => {
                // Trigger short-term consolidation
                log::info!("Triggering short-term consolidation for branch {}", event.branch_id);
            }
            ConsolidationState::Consolidating => {
                // Trigger compression phase
                log::info!("Triggering compression for branch {}", event.branch_id);
            }
            ConsolidationState::LongTerm => {
                // Trigger long-term storage optimization
                log::info!("Triggering long-term storage for branch {}", event.branch_id);
            }
            _ => {}
        }
    }
}
```

**Acceptance Criteria**:
- [ ] Background task runs without impacting performance
- [ ] State transitions occur at exact time boundaries
- [ ] Events trigger appropriate consolidation processes
- [ ] Scheduler is deterministic and testable

#### Task 1.5.1: Core Testing Suite (Parallel)
**Duration**: 90 minutes  
**Dependencies**: 1.1.2 (Branch), 1.2.2 (COW), 1.3.1 (Manager)  
**Parallel Opportunity**: ✅ Can run with 1.4.2

**AI Execution Prompt**:
```rust
// Create tests/temporal/branch_management_tests.rs
use std::time::Duration;
use tokio::time::sleep;
use llmkg::temporal::branch::{BranchManager, Branch, BranchId, ConsolidationState};

#[tokio::test]
async fn test_branch_creation() {
    let manager = BranchManager::new();
    
    let branch_id = manager.create_branch("test_branch".to_string(), None)
        .await
        .expect("Branch creation failed");
    
    assert!(manager.get_branch(branch_id).is_some());
    assert!(manager.get_graph(branch_id).is_some());
}

#[tokio::test]
async fn test_branch_creation_performance() {
    let manager = BranchManager::new();
    
    let start = std::time::Instant::now();
    let _branch_id = manager.create_branch("perf_test".to_string(), None)
        .await
        .expect("Branch creation failed");
    let elapsed = start.elapsed();
    
    assert!(elapsed.as_millis() < 10, "Branch creation took {}ms, should be <10ms", elapsed.as_millis());
}

#[tokio::test]
async fn test_branch_switching_performance() {
    let manager = BranchManager::new();
    
    let branch_id = manager.create_branch("switch_test".to_string(), None)
        .await
        .expect("Branch creation failed");
    
    let start = std::time::Instant::now();
    manager.switch_branch(branch_id)
        .await
        .expect("Branch switch failed");
    let elapsed = start.elapsed();
    
    assert!(elapsed.as_millis() < 1, "Branch switch took {}ms, should be <1ms", elapsed.as_millis());
    assert_eq!(manager.get_current_branch(), Some(branch_id));
}

#[tokio::test]
async fn test_cow_semantics() {
    let manager = BranchManager::new();
    
    // Create parent branch
    let parent_id = manager.create_branch("parent".to_string(), None)
        .await
        .expect("Parent creation failed");
    
    // Create child branch
    let child_id = manager.create_branch("child".to_string(), Some(parent_id))
        .await
        .expect("Child creation failed");
    
    // Verify COW: child should start with 0 memory usage
    let child_graph = manager.get_graph(child_id).unwrap();
    assert_eq!(child_graph.get_memory_usage(), 0, "New branch should have 0 memory usage");
}

#[tokio::test]
async fn test_branch_name_validation() {
    let manager = BranchManager::new();
    
    // Test invalid names
    assert!(manager.create_branch("".to_string(), None).await.is_err());
    assert!(manager.create_branch("a".repeat(51), None).await.is_err());
    assert!(manager.create_branch("invalid@name".to_string(), None).await.is_err());
    
    // Test valid names
    assert!(manager.create_branch("valid_name".to_string(), None).await.is_ok());
    assert!(manager.create_branch("valid-name".to_string(), None).await.is_ok());
    assert!(manager.create_branch("valid123".to_string(), None).await.is_ok());
}

#[tokio::test]
async fn test_consolidation_state_transitions() {
    use llmkg::temporal::branch::ConsolidationTimer;
    
    let branch_id = BranchId::new();
    let timer = ConsolidationTimer::new(branch_id);
    
    // Should start in WorkingMemory
    assert_eq!(timer.current_state, ConsolidationState::WorkingMemory);
    
    // Should not transition immediately
    assert!(!timer.should_transition());
    
    // Test state progression
    assert_eq!(timer.next_state(), Some(ConsolidationState::ShortTerm));
}

#[tokio::test]
async fn test_concurrent_branch_operations() {
    let manager = std::sync::Arc::new(BranchManager::new());
    let mut handles = vec![];
    
    // Spawn 10 concurrent branch creation tasks
    for i in 0..10 {
        let manager_clone = manager.clone();
        let handle = tokio::spawn(async move {
            manager_clone.create_branch(format!("concurrent_{}", i), None).await
        });
        handles.push(handle);
    }
    
    // Wait for all to complete
    let results: Vec<_> = futures::future::join_all(handles).await;
    
    // All should succeed
    assert_eq!(results.len(), 10);
    for result in results {
        assert!(result.unwrap().is_ok());
    }
    
    // Verify all branches exist
    assert_eq!(manager.list_branches().len(), 10);
}

#[tokio::test]
async fn test_memory_estimation() {
    let manager = BranchManager::new();
    
    let branch_id = manager.create_branch("memory_test".to_string(), None)
        .await
        .expect("Branch creation failed");
    
    let branch = manager.get_branch(branch_id).unwrap();
    let memory_usage = branch.estimate_memory_usage();
    
    // Should have some reasonable base memory usage
    assert!(memory_usage > 0);
    assert!(memory_usage < 1000); // Should be under 1KB for empty branch
}

mod integration_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_full_branch_lifecycle() {
        let manager = BranchManager::new();
        
        // Create
        let branch_id = manager.create_branch("lifecycle_test".to_string(), None)
            .await
            .expect("Creation failed");
        
        // Switch
        manager.switch_branch(branch_id)
            .await
            .expect("Switch failed");
        
        assert_eq!(manager.get_current_branch(), Some(branch_id));
        
        // Create child
        let child_id = manager.create_branch("child".to_string(), Some(branch_id))
            .await
            .expect("Child creation failed");
        
        // Switch to child
        manager.switch_branch(child_id)
            .await
            .expect("Child switch failed");
        
        // Delete parent (should fail - has children)
        // TODO: Implement child checking
        
        // Delete child
        manager.delete_branch(child_id)
            .await
            .expect("Child deletion failed");
        
        // Switch back to parent
        manager.switch_branch(branch_id)
            .await
            .expect("Parent switch failed");
        
        // Delete parent
        // TODO: Cannot delete current branch - need to switch first
    }
}
```

**Acceptance Criteria**:
- [ ] All tests pass consistently
- [ ] Performance benchmarks meet targets
- [ ] Memory usage validates COW behavior
- [ ] Concurrent tests detect race conditions
- [ ] Integration tests verify full lifecycle

### CHECKPOINT 1D.1 (540 minutes): Core Implementation Complete
**Validation Commands**:
```bash
cargo test temporal::branch:: --lib
cargo test --test branch_management_tests
cargo clippy --all-targets
```

**Success Criteria**:
- [ ] All unit tests passing
- [ ] Integration tests functional
- [ ] No clippy warnings
- [ ] Performance targets met
- [ ] Memory leaks detected and fixed

### Execution Block 1D.2 (540-630 minutes) - Final Integration

#### Task 1.5.2: Performance Validation & Optimization
**Duration**: 90 minutes  
**Dependencies**: All previous tasks completed  
**Parallel Opportunity**: ❌ Requires complete implementation

**AI Execution Prompt**:
```rust
// Create benches/branch_management_benchmarks.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use llmkg::temporal::branch::BranchManager;
use tokio::runtime::Runtime;

fn benchmark_branch_creation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let manager = BranchManager::new();
    
    c.bench_function("branch_creation", |b| {
        let mut counter = 0;
        b.iter(|| {
            rt.block_on(async {
                let branch_id = manager.create_branch(
                    format!("bench_{}", counter),
                    None
                ).await.unwrap();
                counter += 1;
                black_box(branch_id);
            });
        });
    });
}

fn benchmark_branch_switching(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let manager = BranchManager::new();
    
    // Pre-create branches
    let branch_ids: Vec<_> = rt.block_on(async {
        let mut ids = Vec::new();
        for i in 0..100 {
            let id = manager.create_branch(format!("switch_bench_{}", i), None).await.unwrap();
            ids.push(id);
        }
        ids
    });
    
    c.bench_function("branch_switching", |b| {
        let mut index = 0;
        b.iter(|| {
            rt.block_on(async {
                manager.switch_branch(branch_ids[index % branch_ids.len()]).await.unwrap();
                index += 1;
            });
        });
    });
}

fn benchmark_cow_memory_usage(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let manager = BranchManager::new();
    
    c.bench_function("cow_memory_efficiency", |b| {
        b.iter(|| {
            rt.block_on(async {
                // Create parent
                let parent_id = manager.create_branch("cow_parent".to_string(), None).await.unwrap();
                
                // Create multiple children
                let mut memory_usage = 0u64;
                for i in 0..10 {
                    let child_id = manager.create_branch(
                        format!("cow_child_{}", i),
                        Some(parent_id)
                    ).await.unwrap();
                    
                    let graph = manager.get_graph(child_id).unwrap();
                    memory_usage += graph.get_memory_usage();
                }
                
                black_box(memory_usage);
            });
        });
    });
}

criterion_group!(benches, benchmark_branch_creation, benchmark_branch_switching, benchmark_cow_memory_usage);
criterion_main!(benches);
```

**Acceptance Criteria**:
- [ ] Branch creation benchmark <10ms average
- [ ] Branch switching benchmark <1ms average  
- [ ] COW memory efficiency validated
- [ ] No performance regressions detected
- [ ] Memory leak detection passes

### FINAL CHECKPOINT (630 minutes): MicroPhase1 Complete

**Complete System Validation**:
```bash
# Full test suite
cargo test --all

# Performance benchmarks  
cargo bench

# Documentation generation
cargo doc --all --no-deps

# Code quality
cargo clippy --all-targets -- -D warnings

# Memory leak detection (if available)
cargo test --test branch_management_tests -- --test-threads=1

# Integration with Phase 2 (if available)
cargo test integration::phase2:: --features=phase2-integration
```

**Success Criteria**:
- [ ] All performance targets met consistently
- [ ] Zero memory leaks in 24-hour stress test
- [ ] Thread safety verified under load  
- [ ] Neural pathway integration validated
- [ ] Biological timing requirements satisfied
- [ ] Documentation is complete and accurate
- [ ] No clippy warnings or errors
- [ ] Integration tests with Phase 2/3 pass

## Recovery Points & Failure Handling

### Critical Recovery Points

1. **Checkpoint 1A.1 (30 min)**: Basic type compilation failure
   - **Recovery**: Fix compilation errors before proceeding
   - **Alternative**: Use simpler type definitions temporarily

2. **Checkpoint 1B.1 (180 min)**: COW storage foundation failure  
   - **Recovery**: Implement simplified in-memory storage first
   - **Alternative**: Use HashMap-based storage until page system works

3. **Checkpoint 1C.1 (360 min)**: Branch manager core failure
   - **Recovery**: Implement basic operations without neural integration
   - **Alternative**: Mock neural integration interfaces

4. **Final Checkpoint (630 min)**: Performance targets not met
   - **Recovery**: Profile and optimize critical paths
   - **Alternative**: Temporarily relax performance requirements

### Common Failure Scenarios

| Failure Type | Likely Cause | Recovery Action |
|--------------|--------------|-----------------|
| Compilation Error | Missing dependencies | Add required crates to Cargo.toml |
| Performance Regression | Inefficient algorithms | Profile and optimize hot paths |
| Memory Leak | Missing cleanup | Add explicit resource management |
| Race Condition | Insufficient locking | Audit and fix concurrent access |
| Integration Failure | API mismatch | Update integration interfaces |

## Time Management

### Buffer Time Allocation
- **10% buffer** built into each phase (included in estimates)
- **Emergency buffer**: 2 hours reserved for critical fixes
- **Integration buffer**: 1 hour for unexpected Phase 2/3 integration issues

### Parallel Execution Benefits
- **Phase 1A**: 3 tasks running in parallel saves ~30 minutes
- **Phase 1D**: Testing and state machine in parallel saves ~60 minutes  
- **Total time savings**: ~90 minutes from parallelization

### Critical Path Management
- **Primary critical path**: 1.1.1 → 1.1.2 → 1.2.1 → 1.2.2 → 1.3.1 → 1.3.2
- **Total critical path time**: ~450 minutes (7.5 hours)
- **Non-critical tasks**: Can be delayed without impacting completion

This execution sequence provides a comprehensive, step-by-step guide for implementing MicroPhase1 Branch Management with explicit dependencies, parallel opportunities, and built-in quality assurance checkpoints.