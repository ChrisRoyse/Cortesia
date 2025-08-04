# MicroPhase 4: Diff and Merge Algorithms (IMPROVED)

**Duration**: 8 hours (480 minutes)  
**Prerequisites**: MicroPhase 1 (Branch Management), MicroPhase 2 (Version Chain), MicroPhase 3 (Memory Consolidation)  
**Goal**: Implement high-performance diff/merge algorithms with conflict resolution and complete self-containment

## 🚨 CRITICAL IMPROVEMENTS APPLIED

### Environment Validation Commands
```bash
# Pre-execution validation
cargo --version                                   # Must be 1.70+
ls src/temporal/version/types.rs                 # Verify MicroPhase2 complete
ls src/cognitive/memory/consolidation_engine.rs  # Verify MicroPhase3 complete
cargo check --lib                                # All dependencies resolved
```

### Complete Self-Contained Implementation
```bash
# No external diff libraries (git2, similar, etc.)
# No complex graph algorithms from external crates
# All algorithms implemented from scratch with mocks
# Mathematical diff algorithms using native Rust only
```

## ATOMIC TASK BREAKDOWN (15-30 MIN TASKS)

### 🟢 PHASE 4A: Foundation & Core Diff Types (0-120 minutes)

#### Task 4A.1: Module Structure & Diff Types (15 min)
```bash
# Immediate executable commands
mkdir -p src/temporal/diff
mkdir -p src/temporal/merge
touch src/temporal/diff/mod.rs
touch src/temporal/merge/mod.rs
echo "pub mod diff;" >> src/temporal/mod.rs
echo "pub mod merge;" >> src/temporal/mod.rs
cargo check --lib  # MUST PASS
```

**Self-Contained Implementation:**
```rust
// src/temporal/diff/mod.rs
pub mod types;
pub mod algorithms;
pub mod three_way;

pub use types::*;
pub use algorithms::*;
pub use three_way::*;

// Core diff operation result
#[derive(Debug, Clone, PartialEq)]
pub enum DiffOperation {
    Insert { position: usize, content: DiffContent },
    Delete { position: usize, length: usize },
    Modify { position: usize, old_content: DiffContent, new_content: DiffContent },
    Move { from_position: usize, to_position: usize, length: usize },
}

#[derive(Debug, Clone, PartialEq)]
pub enum DiffContent {
    Node { id: u64, properties: Vec<(String, String)> },
    Edge { from: u64, to: u64, edge_type: String },
    Property { key: String, value: String },
    Text { content: String },
}

impl DiffContent {
    pub fn estimate_size(&self) -> usize {
        match self {
            DiffContent::Node { properties, .. } => {
                8 + properties.iter().map(|(k, v)| k.len() + v.len()).sum::<usize>()
            },
            DiffContent::Edge { edge_type, .. } => 16 + edge_type.len(),
            DiffContent::Property { key, value } => key.len() + value.len(),
            DiffContent::Text { content } => content.len(),
        }
    }
    
    pub fn content_hash(&self) -> u64 {
        // Simple hash function for content comparison
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        match self {
            DiffContent::Node { id, properties } => {
                "node".hash(&mut hasher);
                id.hash(&mut hasher);
                for (k, v) in properties {
                    k.hash(&mut hasher);
                    v.hash(&mut hasher);
                }
            },
            DiffContent::Edge { from, to, edge_type } => {
                "edge".hash(&mut hasher);
                from.hash(&mut hasher);
                to.hash(&mut hasher);
                edge_type.hash(&mut hasher);
            },
            DiffContent::Property { key, value } => {
                "property".hash(&mut hasher);
                key.hash(&mut hasher);
                value.hash(&mut hasher);
            },
            DiffContent::Text { content } => {
                "text".hash(&mut hasher);
                content.hash(&mut hasher);
            },
        }
        hasher.finish()
    }
}

#[cfg(test)]
mod foundation_tests {
    use super::*;
    
    #[test]
    fn diff_content_size_estimation() {
        let content = DiffContent::Node {
            id: 1,
            properties: vec![("name".to_string(), "test".to_string())],
        };
        assert!(content.estimate_size() > 0);
    }
    
    #[test]
    fn diff_content_hashing() {
        let content1 = DiffContent::Text { content: "hello".to_string() };
        let content2 = DiffContent::Text { content: "hello".to_string() };
        let content3 = DiffContent::Text { content: "world".to_string() };
        
        assert_eq!(content1.content_hash(), content2.content_hash());
        assert_ne!(content1.content_hash(), content3.content_hash());
    }
}
```

**Immediate Validation:**
```bash
cargo test foundation_tests --lib
```

#### Task 4A.2: Diff Algorithm Types & Configuration (20 min)
```rust
// src/temporal/diff/types.rs
use crate::temporal::version::types::{VersionId, Version};
use crate::temporal::diff::DiffOperation;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct DiffResult {
    pub from_version: VersionId,
    pub to_version: VersionId,
    pub operations: Vec<DiffOperation>,
    pub algorithm_used: DiffAlgorithm,
    pub confidence_score: f32,
    pub computation_time_ms: u64,
    pub memory_usage_bytes: usize,
}

#[derive(Debug, Clone)]
pub enum DiffAlgorithm {
    MyersDiff,          // O(ND) algorithm for text
    StructuralDiff,     // Custom algorithm for graph structures  
    SemanticDiff,       // Meaning-aware diff for properties
    ContextualDiff,     // Context-aware diff considering surroundings
    HybridDiff,         // Combination of multiple algorithms
}

#[derive(Debug, Clone)]
pub struct DiffConfig {
    pub algorithm: DiffAlgorithm,
    pub ignore_whitespace: bool,
    pub ignore_case: bool,
    pub context_lines: usize,
    pub similarity_threshold: f32,     // 0.0 to 1.0
    pub max_operation_size: usize,
    pub enable_move_detection: bool,
    pub parallel_processing: bool,
}

impl Default for DiffConfig {
    fn default() -> Self {
        Self {
            algorithm: DiffAlgorithm::HybridDiff,
            ignore_whitespace: false,
            ignore_case: false,
            context_lines: 3,
            similarity_threshold: 0.8,
            max_operation_size: 1000,
            enable_move_detection: true,
            parallel_processing: true,
        }
    }
}

impl DiffResult {
    pub fn new(from: VersionId, to: VersionId, algorithm: DiffAlgorithm) -> Self {
        Self {
            from_version: from,
            to_version: to,
            operations: Vec::new(),
            algorithm_used: algorithm,
            confidence_score: 0.0,
            computation_time_ms: 0,
            memory_usage_bytes: 0,
        }
    }
    
    pub fn add_operation(&mut self, operation: DiffOperation) {
        self.operations.push(operation);
    }
    
    pub fn calculate_statistics(&mut self) {
        // Calculate confidence based on operation complexity
        let total_ops = self.operations.len();
        if total_ops == 0 {
            self.confidence_score = 1.0;
            return;
        }
        
        let insert_count = self.operations.iter()
            .filter(|op| matches!(op, DiffOperation::Insert { .. }))
            .count();
        let delete_count = self.operations.iter()
            .filter(|op| matches!(op, DiffOperation::Delete { .. }))
            .count();
        let modify_count = self.operations.iter()
            .filter(|op| matches!(op, DiffOperation::Modify { .. }))
            .count();
        let move_count = self.operations.iter()
            .filter(|op| matches!(op, DiffOperation::Move { .. }))
            .count();
        
        // More moves and modifications = higher confidence in semantic understanding
        let semantic_ops = modify_count + move_count;
        let destructive_ops = insert_count + delete_count;
        
        if total_ops > 0 {
            self.confidence_score = (semantic_ops as f32 * 1.2 + destructive_ops as f32 * 0.8) / 
                                    (total_ops as f32 * 1.2);
            self.confidence_score = self.confidence_score.min(1.0);
        }
        
        // Estimate memory usage
        self.memory_usage_bytes = self.operations.iter()
            .map(|op| self.estimate_operation_size(op))
            .sum();
    }
    
    fn estimate_operation_size(&self, operation: &DiffOperation) -> usize {
        match operation {
            DiffOperation::Insert { content, .. } => content.estimate_size() + 16,
            DiffOperation::Delete { .. } => 16,
            DiffOperation::Modify { old_content, new_content, .. } => {
                old_content.estimate_size() + new_content.estimate_size() + 24
            },
            DiffOperation::Move { .. } => 24,
        }
    }
    
    pub fn operation_count(&self) -> usize {
        self.operations.len()
    }
    
    pub fn is_identical(&self) -> bool {
        self.operations.is_empty()
    }
}

#[cfg(test)]
mod types_tests {
    use super::*;
    use crate::temporal::diff::DiffContent;
    
    #[test]
    fn diff_result_creation() {
        let from = VersionId::new();
        let to = VersionId::new();
        let result = DiffResult::new(from, to, DiffAlgorithm::MyersDiff);
        
        assert_eq!(result.from_version, from);
        assert_eq!(result.to_version, to);
        assert!(result.is_identical());
    }
    
    #[test]
    fn diff_result_statistics() {
        let from = VersionId::new();
        let to = VersionId::new();
        let mut result = DiffResult::new(from, to, DiffAlgorithm::StructuralDiff);
        
        result.add_operation(DiffOperation::Insert {
            position: 0,
            content: DiffContent::Text { content: "test".to_string() },
        });
        
        result.calculate_statistics();
        
        assert_eq!(result.operation_count(), 1);
        assert!(!result.is_identical());
        assert!(result.memory_usage_bytes > 0);
    }
}
```

**Immediate Validation:**
```bash
cargo test types_tests --lib
```

#### Task 4A.3: Myers Diff Algorithm Implementation (30 min)
**Self-Contained Myers Algorithm:**
```rust
// src/temporal/diff/algorithms.rs
use crate::temporal::diff::{DiffOperation, DiffContent, DiffResult, DiffAlgorithm};
use std::collections::HashMap;

pub struct MyersDiffAlgorithm {
    max_d: usize,  // Maximum edit distance to consider
}

impl MyersDiffAlgorithm {
    pub fn new() -> Self {
        Self { max_d: 1000 }
    }
    
    /// Implementation of Myers' O(ND) diff algorithm
    /// Finds shortest edit script between two sequences
    pub fn diff_text(&self, from: &str, to: &str) -> DiffResult {
        let start_time = std::time::Instant::now();
        
        let from_lines: Vec<&str> = from.lines().collect();
        let to_lines: Vec<&str> = to.lines().collect();
        
        let mut result = DiffResult::new(
            crate::temporal::version::types::VersionId::new(),
            crate::temporal::version::types::VersionId::new(),
            DiffAlgorithm::MyersDiff
        );
        
        let operations = self.myers_diff(&from_lines, &to_lines);
        for op in operations {
            result.add_operation(op);
        }
        
        result.computation_time_ms = start_time.elapsed().as_millis() as u64;
        result.calculate_statistics();
        
        result
    }
    
    fn myers_diff(&self, from: &[&str], to: &[&str]) -> Vec<DiffOperation> {
        let n = from.len();
        let m = to.len();
        let max = n + m;
        
        // V[k] represents the furthest reaching D-path on diagonal k
        let mut v: HashMap<isize, usize> = HashMap::new();
        v.insert(1, 0);
        
        let mut trace = Vec::new();
        
        for d in 0..=max {
            let mut v_current = v.clone();
            
            let k_start = -(d as isize);
            let k_end = d as isize;
            
            for k in (k_start..=k_end).step_by(2) {
                let mut x = if k == -(d as isize) || (k != d as isize && 
                    v.get(&(k-1)).unwrap_or(&0) < v.get(&(k+1)).unwrap_or(&0)) {
                    *v.get(&(k+1)).unwrap_or(&0)
                } else {
                    v.get(&(k-1)).unwrap_or(&0) + 1
                };
                
                let mut y = (x as isize - k) as usize;
                
                // Follow diagonal matches
                while x < n && y < m && from[x] == to[y] {
                    x += 1;
                    y += 1;
                }
                
                v_current.insert(k, x);
                
                if x >= n && y >= m {
                    trace.push(v_current);
                    return self.backtrack_myers(&trace, from, to, n, m);
                }
            }
            
            trace.push(v_current.clone());
            v = v_current;
        }
        
        // Fallback: create simple replacement if optimal path not found
        vec![
            DiffOperation::Delete { position: 0, length: n },
            DiffOperation::Insert { 
                position: 0, 
                content: DiffContent::Text { content: to.join("\n") }
            },
        ]
    }
    
    fn backtrack_myers(
        &self,
        trace: &[HashMap<isize, usize>],
        from: &[&str],
        to: &[&str],
        n: usize,
        m: usize,
    ) -> Vec<DiffOperation> {
        let mut operations = Vec::new();
        let mut x = n;
        let mut y = m;
        
        for d in (0..trace.len()).rev() {
            let v = &trace[d];
            let k = x as isize - y as isize;
            
            let prev_k = if k == -(d as isize) || 
                (k != d as isize && 
                 v.get(&(k-1)).unwrap_or(&0) < v.get(&(k+1)).unwrap_or(&0)) {
                k + 1
            } else {
                k - 1
            };
            
            let prev_x = *v.get(&prev_k).unwrap_or(&0);
            let prev_y = (prev_x as isize - prev_k) as usize;
            
            // Follow diagonal (no operation needed)
            while x > prev_x && y > prev_y {
                x -= 1;
                y -= 1;
            }
            
            if d > 0 {
                if x > prev_x {
                    // Deletion
                    operations.push(DiffOperation::Delete {
                        position: prev_x,
                        length: 1,
                    });
                    x = prev_x;
                } else {
                    // Insertion
                    operations.push(DiffOperation::Insert {
                        position: prev_x,
                        content: DiffContent::Text { content: to[prev_y].to_string() },
                    });
                    y = prev_y;
                }
            }
        }
        
        operations.reverse();
        operations
    }
    
    /// Specialized diff for graph structures
    pub fn diff_graph_nodes(&self, from_nodes: &[(u64, Vec<(String, String)>)], 
                           to_nodes: &[(u64, Vec<(String, String)>)]) -> Vec<DiffOperation> {
        let mut operations = Vec::new();
        
        // Convert to hash maps for efficient lookup
        let from_map: HashMap<u64, &Vec<(String, String)>> = from_nodes.iter()
            .map(|(id, props)| (*id, props)).collect();
        let to_map: HashMap<u64, &Vec<(String, String)>> = to_nodes.iter()
            .map(|(id, props)| (*id, props)).collect();
        
        // Find deletions (nodes in from but not in to)
        for (&node_id, _) in &from_map {
            if !to_map.contains_key(&node_id) {
                operations.push(DiffOperation::Delete {
                    position: node_id as usize,
                    length: 1,
                });
            }
        }
        
        // Find insertions (nodes in to but not in from)
        for (&node_id, properties) in &to_map {
            if !from_map.contains_key(&node_id) {
                operations.push(DiffOperation::Insert {
                    position: node_id as usize,
                    content: DiffContent::Node {
                        id: node_id,
                        properties: (*properties).clone(),
                    },
                });
            }
        }
        
        // Find modifications (nodes in both but with different properties)
        for (&node_id, from_props) in &from_map {
            if let Some(to_props) = to_map.get(&node_id) {
                if from_props != to_props {
                    operations.push(DiffOperation::Modify {
                        position: node_id as usize,
                        old_content: DiffContent::Node {
                            id: node_id,
                            properties: (*from_props).clone(),
                        },
                        new_content: DiffContent::Node {
                            id: node_id,
                            properties: (*to_props).clone(),
                        },
                    });
                }
            }
        }
        
        operations
    }
}

#[cfg(test)]
mod myers_tests {
    use super::*;
    
    #[test]
    fn myers_diff_identical_text() {
        let myers = MyersDiffAlgorithm::new();
        let result = myers.diff_text("hello\nworld", "hello\nworld");
        assert!(result.is_identical());
    }
    
    #[test]
    fn myers_diff_simple_insertion() {
        let myers = MyersDiffAlgorithm::new();
        let result = myers.diff_text("hello", "hello\nworld");
        
        assert_eq!(result.operation_count(), 1);
        assert!(matches!(result.operations[0], DiffOperation::Insert { .. }));
    }
    
    #[test]
    fn myers_diff_simple_deletion() {
        let myers = MyersDiffAlgorithm::new();
        let result = myers.diff_text("hello\nworld", "hello");
        
        assert_eq!(result.operation_count(), 1);
        assert!(matches!(result.operations[0], DiffOperation::Delete { .. }));
    }
    
    #[test]
    fn graph_node_diff_operations() {
        let myers = MyersDiffAlgorithm::new();
        
        let from_nodes = vec![
            (1, vec![("name".to_string(), "alice".to_string())]),
            (2, vec![("name".to_string(), "bob".to_string())]),
        ];
        
        let to_nodes = vec![
            (1, vec![("name".to_string(), "alice2".to_string())]), // Modified
            (3, vec![("name".to_string(), "charlie".to_string())]), // Added
        ];
        
        let operations = myers.diff_graph_nodes(&from_nodes, &to_nodes);
        
        // Should have: 1 deletion (node 2), 1 insertion (node 3), 1 modification (node 1)
        assert_eq!(operations.len(), 3);
        
        let has_delete = operations.iter().any(|op| matches!(op, DiffOperation::Delete { .. }));
        let has_insert = operations.iter().any(|op| matches!(op, DiffOperation::Insert { .. }));
        let has_modify = operations.iter().any(|op| matches!(op, DiffOperation::Modify { .. }));
        
        assert!(has_delete);
        assert!(has_insert);
        assert!(has_modify);
    }
}
```

**Immediate Validation:**
```bash
cargo test myers_tests --lib
```

### 🟡 PHASE 4B: Three-Way Merge & Conflict Resolution (120-300 minutes)

#### Task 4B.1: Three-Way Merge Foundation (45 min)
```rust
// src/temporal/diff/three_way.rs
use crate::temporal::diff::{DiffOperation, DiffContent, DiffResult, DiffAlgorithm};
use crate::temporal::version::types::VersionId;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct ThreeWayMergeResult {
    pub base_version: VersionId,
    pub left_version: VersionId,
    pub right_version: VersionId,
    pub merged_operations: Vec<DiffOperation>,
    pub conflicts: Vec<MergeConflict>,
    pub resolution_strategy: ConflictResolutionStrategy,
    pub merge_confidence: f32,
    pub computation_time_ms: u64,
}

#[derive(Debug, Clone)]
pub struct MergeConflict {
    pub conflict_id: ConflictId,
    pub conflict_type: ConflictType,
    pub position: usize,
    pub left_content: Option<DiffContent>,
    pub right_content: Option<DiffContent>,
    pub base_content: Option<DiffContent>,
    pub severity: ConflictSeverity,
    pub auto_resolvable: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConflictId(u64);

impl ConflictId {
    pub fn new() -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        Self(SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_micros() as u64)
    }
}

#[derive(Debug, Clone)]
pub enum ConflictType {
    ContentConflict,        // Both sides modified same content differently
    DeleteEditConflict,     // One side deleted, other modified
    PropertyConflict,       // Different property values for same key
    StructuralConflict,     // Graph structure changes that conflict
    SemanticConflict,       // Logically incompatible changes
}

#[derive(Debug, Clone)]
pub enum ConflictSeverity {
    Low,      // Easily auto-resolvable
    Medium,   // Requires simple heuristics
    High,     // Requires human intervention
    Critical, // Could break system consistency
}

#[derive(Debug, Clone)]
pub enum ConflictResolutionStrategy {
    TakeLeft,           // Always prefer left side
    TakeRight,          // Always prefer right side
    TakeNewer,          // Prefer more recent change
    TakeHigherConfidence, // Prefer change with higher confidence score
    ManualResolution,   // Require manual intervention
    Contextual,         // Use context-aware resolution
    Hybrid,             // Combine multiple strategies
}

pub struct ThreeWayMerger {
    resolution_strategy: ConflictResolutionStrategy,
    auto_resolve_low_severity: bool,
    max_conflict_count: usize,
}

impl ThreeWayMerger {
    pub fn new(strategy: ConflictResolutionStrategy) -> Self {
        Self {
            resolution_strategy: strategy,
            auto_resolve_low_severity: true,
            max_conflict_count: 100,
        }
    }
    
    pub fn merge(&self, base: &str, left: &str, right: &str) -> ThreeWayMergeResult {
        let start_time = std::time::Instant::now();
        
        let base_lines: Vec<&str> = base.lines().collect();
        let left_lines: Vec<&str> = left.lines().collect();
        let right_lines: Vec<&str> = right.lines().collect();
        
        // Compute diffs from base to each side
        let left_diff = self.compute_diff(&base_lines, &left_lines);
        let right_diff = self.compute_diff(&base_lines, &right_lines);
        
        // Merge the diffs
        let (merged_ops, conflicts) = self.merge_diffs(&left_diff, &right_diff, &base_lines);
        
        let computation_time = start_time.elapsed().as_millis() as u64;
        
        ThreeWayMergeResult {
            base_version: VersionId::new(),
            left_version: VersionId::new(),
            right_version: VersionId::new(),
            merged_operations: merged_ops,
            conflicts,
            resolution_strategy: self.resolution_strategy.clone(),
            merge_confidence: self.calculate_merge_confidence(&conflicts),
            computation_time_ms: computation_time,
        }
    }
    
    fn compute_diff(&self, from: &[&str], to: &[&str]) -> Vec<DiffOperation> {
        // Use Myers algorithm for basic diff
        let myers = crate::temporal::diff::algorithms::MyersDiffAlgorithm::new();
        let from_text = from.join("\n");
        let to_text = to.join("\n");
        let result = myers.diff_text(&from_text, &to_text);
        result.operations
    }
    
    fn merge_diffs(
        &self,
        left_diff: &[DiffOperation],
        right_diff: &[DiffOperation],
        base_lines: &[&str],
    ) -> (Vec<DiffOperation>, Vec<MergeConflict>) {
        let mut merged_operations = Vec::new();
        let mut conflicts = Vec::new();
        
        // Group operations by position for conflict detection
        let mut left_ops_by_pos = HashMap::new();
        let mut right_ops_by_pos = HashMap::new();
        
        for op in left_diff {
            let pos = self.get_operation_position(op);
            left_ops_by_pos.insert(pos, op);
        }
        
        for op in right_diff {
            let pos = self.get_operation_position(op);
            right_ops_by_pos.insert(pos, op);
        }
        
        // Find all positions that need to be processed
        let mut all_positions: Vec<usize> = left_ops_by_pos.keys()
            .chain(right_ops_by_pos.keys())
            .copied()
            .collect();
        all_positions.sort();
        all_positions.dedup();
        
        for pos in all_positions {
            match (left_ops_by_pos.get(&pos), right_ops_by_pos.get(&pos)) {
                (Some(left_op), Some(right_op)) => {
                    // Both sides have operations at this position - potential conflict
                    if self.operations_compatible(left_op, right_op) {
                        // Compatible operations - merge them
                        if let Some(merged_op) = self.merge_compatible_operations(left_op, right_op) {
                            merged_operations.push(merged_op);
                        }
                    } else {
                        // Conflicting operations
                        let conflict = self.create_conflict(left_op, right_op, pos, base_lines);
                        
                        if conflict.auto_resolvable && self.auto_resolve_low_severity {
                            if let Some(resolved_op) = self.auto_resolve_conflict(&conflict) {
                                merged_operations.push(resolved_op);
                            }
                        } else {
                            conflicts.push(conflict);
                        }
                    }
                },
                (Some(left_op), None) => {
                    // Only left side has operation
                    merged_operations.push((*left_op).clone());
                },
                (None, Some(right_op)) => {
                    // Only right side has operation
                    merged_operations.push((*right_op).clone());
                },
                (None, None) => {
                    // This shouldn't happen
                }
            }
        }
        
        (merged_operations, conflicts)
    }
    
    fn get_operation_position(&self, op: &DiffOperation) -> usize {
        match op {
            DiffOperation::Insert { position, .. } => *position,
            DiffOperation::Delete { position, .. } => *position,
            DiffOperation::Modify { position, .. } => *position,
            DiffOperation::Move { from_position, .. } => *from_position,
        }
    }
    
    fn operations_compatible(&self, left_op: &DiffOperation, right_op: &DiffOperation) -> bool {
        match (left_op, right_op) {
            // Same operation type with same content
            (DiffOperation::Insert { content: left_content, .. }, 
             DiffOperation::Insert { content: right_content, .. }) => {
                left_content.content_hash() == right_content.content_hash()
            },
            (DiffOperation::Delete { length: left_len, .. }, 
             DiffOperation::Delete { length: right_len, .. }) => {
                left_len == right_len
            },
            // Different operation types are generally incompatible
            _ => false,
        }
    }
    
    fn merge_compatible_operations(
        &self,
        left_op: &DiffOperation,
        right_op: &DiffOperation,
    ) -> Option<DiffOperation> {
        match (left_op, right_op) {
            (DiffOperation::Insert { position, content, .. }, 
             DiffOperation::Insert { .. }) => {
                // Compatible inserts - use the first one
                Some(DiffOperation::Insert {
                    position: *position,
                    content: content.clone(),
                })
            },
            (DiffOperation::Delete { position, length, .. }, 
             DiffOperation::Delete { .. }) => {
                // Compatible deletes - use the first one
                Some(DiffOperation::Delete {
                    position: *position,
                    length: *length,
                })
            },
            _ => None,
        }
    }
    
    fn create_conflict(
        &self,
        left_op: &DiffOperation,
        right_op: &DiffOperation,
        position: usize,
        base_lines: &[&str],
    ) -> MergeConflict {
        let conflict_type = self.classify_conflict_type(left_op, right_op);
        let severity = self.assess_conflict_severity(&conflict_type, left_op, right_op);
        
        MergeConflict {
            conflict_id: ConflictId::new(),
            conflict_type,
            position,
            left_content: self.extract_operation_content(left_op),
            right_content: self.extract_operation_content(right_op),
            base_content: if position < base_lines.len() {
                Some(DiffContent::Text { content: base_lines[position].to_string() })
            } else {
                None
            },
            severity,
            auto_resolvable: matches!(severity, ConflictSeverity::Low),
        }
    }
    
    fn classify_conflict_type(&self, left_op: &DiffOperation, right_op: &DiffOperation) -> ConflictType {
        match (left_op, right_op) {
            (DiffOperation::Delete { .. }, DiffOperation::Modify { .. }) |
            (DiffOperation::Modify { .. }, DiffOperation::Delete { .. }) => {
                ConflictType::DeleteEditConflict
            },
            (DiffOperation::Modify { .. }, DiffOperation::Modify { .. }) => {
                ConflictType::ContentConflict
            },
            _ => ConflictType::StructuralConflict,
        }
    }
    
    fn assess_conflict_severity(&self, conflict_type: &ConflictType, _left_op: &DiffOperation, _right_op: &DiffOperation) -> ConflictSeverity {
        match conflict_type {
            ConflictType::ContentConflict => ConflictSeverity::Medium,
            ConflictType::DeleteEditConflict => ConflictSeverity::High,
            ConflictType::PropertyConflict => ConflictSeverity::Low,
            ConflictType::StructuralConflict => ConflictSeverity::High,
            ConflictType::SemanticConflict => ConflictSeverity::Critical,
        }
    }
    
    fn extract_operation_content(&self, op: &DiffOperation) -> Option<DiffContent> {
        match op {
            DiffOperation::Insert { content, .. } => Some(content.clone()),
            DiffOperation::Modify { new_content, .. } => Some(new_content.clone()),
            _ => None,
        }
    }
    
    fn auto_resolve_conflict(&self, conflict: &MergeConflict) -> Option<DiffOperation> {
        if !conflict.auto_resolvable {
            return None;
        }
        
        match &self.resolution_strategy {
            ConflictResolutionStrategy::TakeLeft => {
                if let Some(content) = &conflict.left_content {
                    Some(DiffOperation::Insert {
                        position: conflict.position,
                        content: content.clone(),
                    })
                } else {
                    None
                }
            },
            ConflictResolutionStrategy::TakeRight => {
                if let Some(content) = &conflict.right_content {
                    Some(DiffOperation::Insert {
                        position: conflict.position,
                        content: content.clone(),
                    })
                } else {
                    None
                }
            },
            _ => None, // Other strategies need more complex logic
        }
    }
    
    fn calculate_merge_confidence(&self, conflicts: &[MergeConflict]) -> f32 {
        if conflicts.is_empty() {
            return 1.0;
        }
        
        let total_severity: u32 = conflicts.iter()
            .map(|c| match c.severity {
                ConflictSeverity::Low => 1,
                ConflictSeverity::Medium => 3,
                ConflictSeverity::High => 7,
                ConflictSeverity::Critical => 15,
            })
            .sum();
        
        let max_possible_severity = conflicts.len() as u32 * 15; // All critical
        1.0 - (total_severity as f32 / max_possible_severity as f32)
    }
}

#[cfg(test)]
mod three_way_tests {
    use super::*;
    
    #[test]
    fn three_way_merge_no_conflicts() {
        let merger = ThreeWayMerger::new(ConflictResolutionStrategy::TakeLeft);
        
        let base = "line1\nline2\nline3";
        let left = "line1\nmodified_left\nline3";
        let right = "line1\nline2\nline3\nadded_right";
        
        let result = merger.merge(base, left, right);
        
        // Should merge without conflicts since changes are at different positions
        assert!(result.conflicts.is_empty());
        assert!(!result.merged_operations.is_empty());
        assert!(result.merge_confidence > 0.8);
    }
    
    #[test]
    fn three_way_merge_with_conflicts() {
        let merger = ThreeWayMerger::new(ConflictResolutionStrategy::ManualResolution);
        
        let base = "line1\nline2\nline3";
        let left = "line1\nmodified_left\nline3";
        let right = "line1\nmodified_right\nline3";
        
        let result = merger.merge(base, left, right);
        
        // Should detect conflict since both sides modified line2 differently
        assert!(!result.conflicts.is_empty());
        assert!(result.merge_confidence < 1.0);
    }
    
    #[test]
    fn conflict_creation_and_classification() {
        let merger = ThreeWayMerger::new(ConflictResolutionStrategy::TakeLeft);
        
        let left_op = DiffOperation::Modify {
            position: 1,
            old_content: DiffContent::Text { content: "old".to_string() },
            new_content: DiffContent::Text { content: "left_new".to_string() },
        };
        
        let right_op = DiffOperation::Modify {
            position: 1,
            old_content: DiffContent::Text { content: "old".to_string() },
            new_content: DiffContent::Text { content: "right_new".to_string() },
        };
        
        let base_lines = vec!["line0", "old", "line2"];
        let conflict = merger.create_conflict(&left_op, &right_op, 1, &base_lines);
        
        assert!(matches!(conflict.conflict_type, ConflictType::ContentConflict));
        assert!(matches!(conflict.severity, ConflictSeverity::Medium));
    }
}
```

**Immediate Validation:**
```bash
cargo test three_way_tests --lib
```

### 🔵 PHASE 4C: Advanced Merge Strategies & Integration (300-480 minutes)

#### Task 4C.1: Merge Engine Integration (60 min)
```rust
// src/temporal/merge/mod.rs
pub mod engine;
pub mod strategies;
pub mod conflict_resolution;

pub use engine::*;
pub use strategies::*;
pub use conflict_resolution::*;

// src/temporal/merge/engine.rs
use crate::temporal::diff::three_way::{ThreeWayMerger, ThreeWayMergeResult, ConflictResolutionStrategy};
use crate::temporal::version::types::{VersionId, Version};
use crate::temporal::version::chain::VersionChain;
use crate::temporal::version::delta::Delta;
use crate::temporal::version::snapshot::GraphSnapshot;
use std::collections::HashMap;

#[derive(Debug)]
pub struct MergeEngine {
    merger: ThreeWayMerger,
    conflict_resolver: ConflictResolver,
    merge_cache: HashMap<MergeKey, CachedMergeResult>,
    statistics: MergeStatistics,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct MergeKey {
    base: VersionId,
    left: VersionId,
    right: VersionId,
}

#[derive(Debug, Clone)]
struct CachedMergeResult {
    result: ThreeWayMergeResult,
    cached_at: std::time::SystemTime,
    access_count: u32,
}

#[derive(Debug, Clone)]
pub struct MergeStatistics {
    pub total_merges: u64,
    pub successful_merges: u64,
    pub failed_merges: u64,
    pub conflicts_resolved: u64,
    pub conflicts_unresolved: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub average_merge_time_ms: f32,
}

impl MergeStatistics {
    pub fn new() -> Self {
        Self {
            total_merges: 0,
            successful_merges: 0,
            failed_merges: 0,
            conflicts_resolved: 0,
            conflicts_unresolved: 0,
            cache_hits: 0,
            cache_misses: 0,
            average_merge_time_ms: 0.0,
        }
    }
    
    pub fn success_rate(&self) -> f32 {
        if self.total_merges == 0 { return 0.0; }
        self.successful_merges as f32 / self.total_merges as f32
    }
    
    pub fn conflict_resolution_rate(&self) -> f32 {
        let total_conflicts = self.conflicts_resolved + self.conflicts_unresolved;
        if total_conflicts == 0 { return 1.0; }
        self.conflicts_resolved as f32 / total_conflicts as f32
    }
    
    pub fn cache_hit_rate(&self) -> f32 {
        let total_requests = self.cache_hits + self.cache_misses;
        if total_requests == 0 { return 0.0; }
        self.cache_hits as f32 / total_requests as f32
    }
}

impl MergeEngine {
    pub fn new(strategy: ConflictResolutionStrategy) -> Self {
        Self {
            merger: ThreeWayMerger::new(strategy.clone()),
            conflict_resolver: ConflictResolver::new(strategy),
            merge_cache: HashMap::new(),
            statistics: MergeStatistics::new(),
        }
    }
    
    pub fn merge_versions(
        &mut self,
        base_version: &Version,
        left_version: &Version,
        right_version: &Version,
        version_chain: &VersionChain,
    ) -> Result<MergeResult, MergeError> {
        let start_time = std::time::Instant::now();
        self.statistics.total_merges += 1;
        
        let merge_key = MergeKey {
            base: base_version.id,
            left: left_version.id,
            right: right_version.id,
        };
        
        // Check cache first
        if let Some(cached) = self.merge_cache.get_mut(&merge_key) {
            cached.access_count += 1;
            self.statistics.cache_hits += 1;
            
            return Ok(MergeResult {
                merged_version: self.create_merged_version(&cached.result, left_version, right_version)?,
                conflicts: cached.result.conflicts.clone(),
                merge_confidence: cached.result.merge_confidence,
                from_cache: true,
                computation_time_ms: 0,
            });
        }
        
        self.statistics.cache_misses += 1;
        
        // Get snapshots for each version
        let base_snapshot = self.get_version_snapshot(base_version, version_chain)?;
        let left_snapshot = self.get_version_snapshot(left_version, version_chain)?;
        let right_snapshot = self.get_version_snapshot(right_version, version_chain)?;
        
        // Convert snapshots to text representation for merging
        let base_text = self.snapshot_to_text(&base_snapshot);
        let left_text = self.snapshot_to_text(&left_snapshot);
        let right_text = self.snapshot_to_text(&right_snapshot);
        
        // Perform three-way merge
        let merge_result = self.merger.merge(&base_text, &left_text, &right_text);
        
        // Resolve conflicts if possible
        let resolved_conflicts = self.conflict_resolver.resolve_conflicts(&merge_result.conflicts)?;
        let final_conflicts: Vec<_> = merge_result.conflicts.into_iter()
            .filter(|c| !resolved_conflicts.contains_key(&c.conflict_id))
            .collect();
        
        // Update statistics
        if final_conflicts.is_empty() {
            self.statistics.successful_merges += 1;
        } else {
            self.statistics.failed_merges += 1;
        }
        
        self.statistics.conflicts_resolved += resolved_conflicts.len() as u64;
        self.statistics.conflicts_unresolved += final_conflicts.len() as u64;
        
        // Cache the result
        let cached_result = CachedMergeResult {
            result: merge_result.clone(),
            cached_at: std::time::SystemTime::now(),
            access_count: 1,
        };
        self.merge_cache.insert(merge_key, cached_result);
        
        // Create final merged version
        let merged_version = self.create_merged_version(&merge_result, left_version, right_version)?;
        
        let computation_time = start_time.elapsed().as_millis() as u64;
        self.update_average_time(computation_time);
        
        Ok(MergeResult {
            merged_version,
            conflicts: final_conflicts,
            merge_confidence: merge_result.merge_confidence,
            from_cache: false,
            computation_time_ms: computation_time,
        })
    }
    
    fn get_version_snapshot(&self, version: &Version, version_chain: &VersionChain) -> Result<GraphSnapshot, MergeError> {
        // Mock implementation - in real system would reconstruct from deltas
        Ok(GraphSnapshot::new(version.id))
    }
    
    fn snapshot_to_text(&self, snapshot: &GraphSnapshot) -> String {
        // Convert snapshot to textual representation for merging
        let mut lines = Vec::new();
        
        // Add nodes
        for (node_id, node_data) in &snapshot.nodes {
            lines.push(format!("NODE:{}", node_id));
            if let Some(props) = snapshot.properties.get(node_id) {
                for (key, value) in props {
                    lines.push(format!("  {}={}", key, value));
                }
            }
        }
        
        // Add edges
        for edge in &snapshot.edges {
            lines.push(format!("EDGE:{}->{}:{}", edge.from_node, edge.to_node, edge.edge_type));
        }
        
        lines.join("\n")
    }
    
    fn create_merged_version(
        &self,
        merge_result: &ThreeWayMergeResult,
        left_version: &Version,
        right_version: &Version,
    ) -> Result<Version, MergeError> {
        // Create new version representing the merge
        let mut merged_version = Version::new(
            left_version.branch_id, // Use left branch as primary
            Some(left_version.id),  // Parent is left version
            format!("Merge {} into {}", right_version.id.timestamp(), left_version.id.timestamp()),
        );
        
        // Update metadata
        merged_version.metadata.change_count = merge_result.merged_operations.len();
        merged_version.metadata.size_bytes = merge_result.merged_operations.iter()
            .map(|op| match op {
                crate::temporal::diff::DiffOperation::Insert { content, .. } => content.estimate_size(),
                crate::temporal::diff::DiffOperation::Modify { old_content, new_content, .. } => {
                    old_content.estimate_size() + new_content.estimate_size()
                },
                _ => 16, // Basic operation size
            })
            .sum();
        
        Ok(merged_version)
    }
    
    fn update_average_time(&mut self, new_time: u64) {
        let current_avg = self.statistics.average_merge_time_ms;
        let total_merges = self.statistics.total_merges as f32;
        
        self.statistics.average_merge_time_ms = 
            (current_avg * (total_merges - 1.0) + new_time as f32) / total_merges;
    }
    
    pub fn get_statistics(&self) -> &MergeStatistics {
        &self.statistics
    }
    
    pub fn clear_cache(&mut self) {
        self.merge_cache.clear();
    }
    
    pub fn cache_size(&self) -> usize {
        self.merge_cache.len()
    }
}

#[derive(Debug)]
pub struct MergeResult {
    pub merged_version: Version,
    pub conflicts: Vec<crate::temporal::diff::three_way::MergeConflict>,
    pub merge_confidence: f32,
    pub from_cache: bool,
    pub computation_time_ms: u64,
}

#[derive(Debug)]
pub enum MergeError {
    VersionNotFound(VersionId),
    SnapshotReconstructionFailed(String),
    ConflictResolutionFailed(String),
    InvalidMergeState(String),
}

impl std::fmt::Display for MergeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MergeError::VersionNotFound(id) => write!(f, "Version not found: {:?}", id),
            MergeError::SnapshotReconstructionFailed(msg) => write!(f, "Snapshot reconstruction failed: {}", msg),
            MergeError::ConflictResolutionFailed(msg) => write!(f, "Conflict resolution failed: {}", msg),
            MergeError::InvalidMergeState(msg) => write!(f, "Invalid merge state: {}", msg),
        }
    }
}

impl std::error::Error for MergeError {}

// src/temporal/merge/conflict_resolution.rs
use crate::temporal::diff::three_way::{MergeConflict, ConflictId, ConflictResolutionStrategy, ConflictSeverity};
use crate::temporal::diff::DiffOperation;
use std::collections::HashMap;

pub struct ConflictResolver {
    strategy: ConflictResolutionStrategy,
    resolution_cache: HashMap<u64, DiffOperation>, // Hash of conflict -> resolution
}

impl ConflictResolver {
    pub fn new(strategy: ConflictResolutionStrategy) -> Self {
        Self {
            strategy,
            resolution_cache: HashMap::new(),
        }
    }
    
    pub fn resolve_conflicts(
        &mut self,
        conflicts: &[MergeConflict],
    ) -> Result<HashMap<ConflictId, DiffOperation>, crate::temporal::merge::engine::MergeError> {
        let mut resolutions = HashMap::new();
        
        for conflict in conflicts {
            if let Some(resolution) = self.resolve_single_conflict(conflict)? {
                resolutions.insert(conflict.conflict_id, resolution);
            }
        }
        
        Ok(resolutions)
    }
    
    fn resolve_single_conflict(
        &mut self,
        conflict: &MergeConflict,
    ) -> Result<Option<DiffOperation>, crate::temporal::merge::engine::MergeError> {
        // Check cache first
        let conflict_hash = self.hash_conflict(conflict);
        if let Some(cached_resolution) = self.resolution_cache.get(&conflict_hash) {
            return Ok(Some(cached_resolution.clone()));
        }
        
        let resolution = match &self.strategy {
            ConflictResolutionStrategy::TakeLeft => {
                self.resolve_take_left(conflict)
            },
            ConflictResolutionStrategy::TakeRight => {
                self.resolve_take_right(conflict)
            },
            ConflictResolutionStrategy::TakeNewer => {
                self.resolve_take_newer(conflict)
            },
            ConflictResolutionStrategy::Contextual => {
                self.resolve_contextual(conflict)
            },
            ConflictResolutionStrategy::ManualResolution => {
                None // Requires human intervention
            },
            _ => None,
        };
        
        // Cache the resolution
        if let Some(ref resolution_op) = resolution {
            self.resolution_cache.insert(conflict_hash, resolution_op.clone());
        }
        
        Ok(resolution)
    }
    
    fn resolve_take_left(&self, conflict: &MergeConflict) -> Option<DiffOperation> {
        conflict.left_content.as_ref().map(|content| {
            DiffOperation::Insert {
                position: conflict.position,
                content: content.clone(),
            }
        })
    }
    
    fn resolve_take_right(&self, conflict: &MergeConflict) -> Option<DiffOperation> {
        conflict.right_content.as_ref().map(|content| {
            DiffOperation::Insert {
                position: conflict.position,
                content: content.clone(),
            }
        })
    }
    
    fn resolve_take_newer(&self, conflict: &MergeConflict) -> Option<DiffOperation> {
        // Mock implementation - in real system would check timestamps
        // For now, just prefer right side as "newer"
        self.resolve_take_right(conflict)
    }
    
    fn resolve_contextual(&self, conflict: &MergeConflict) -> Option<DiffOperation> {
        // Context-aware resolution based on severity and type
        match conflict.severity {
            ConflictSeverity::Low => {
                // For low severity, try to merge content if possible
                self.attempt_content_merge(conflict)
            },
            ConflictSeverity::Medium => {
                // For medium severity, prefer the larger change
                self.resolve_prefer_larger_change(conflict)
            },
            _ => None, // High/Critical require manual resolution
        }
    }
    
    fn attempt_content_merge(&self, conflict: &MergeConflict) -> Option<DiffOperation> {
        match (&conflict.left_content, &conflict.right_content) {
            (Some(left), Some(right)) => {
                // Simple heuristic: if both are text and one contains the other, use the longer one
                if let (crate::temporal::diff::DiffContent::Text { content: left_text }, 
                        crate::temporal::diff::DiffContent::Text { content: right_text }) = (left, right) {
                    
                    if left_text.contains(right_text) {
                        Some(DiffOperation::Insert {
                            position: conflict.position,
                            content: left.clone(),
                        })
                    } else if right_text.contains(left_text) {
                        Some(DiffOperation::Insert {
                            position: conflict.position,
                            content: right.clone(),
                        })
                    } else {
                        // Try to concatenate with a separator
                        let merged_content = format!("{}\n{}", left_text, right_text);
                        Some(DiffOperation::Insert {
                            position: conflict.position,
                            content: crate::temporal::diff::DiffContent::Text { 
                                content: merged_content 
                            },
                        })
                    }
                } else {
                    None
                }
            },
            _ => None,
        }
    }
    
    fn resolve_prefer_larger_change(&self, conflict: &MergeConflict) -> Option<DiffOperation> {
        match (&conflict.left_content, &conflict.right_content) {
            (Some(left), Some(right)) => {
                let left_size = left.estimate_size();
                let right_size = right.estimate_size();
                
                let preferred_content = if left_size >= right_size { left } else { right };
                
                Some(DiffOperation::Insert {
                    position: conflict.position,
                    content: preferred_content.clone(),
                })
            },
            (Some(left), None) => {
                Some(DiffOperation::Insert {
                    position: conflict.position,
                    content: left.clone(),
                })
            },
            (None, Some(right)) => {
                Some(DiffOperation::Insert {
                    position: conflict.position,
                    content: right.clone(),
                })
            },
            (None, None) => None,
        }
    }
    
    fn hash_conflict(&self, conflict: &MergeConflict) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        conflict.position.hash(&mut hasher);
        
        if let Some(ref left) = conflict.left_content {
            left.content_hash().hash(&mut hasher);
        }
        
        if let Some(ref right) = conflict.right_content {
            right.content_hash().hash(&mut hasher);
        }
        
        hasher.finish()
    }
}

#[cfg(test)]
mod merge_engine_tests {
    use super::*;
    use crate::temporal::version::types::Version;
    use crate::temporal::version::chain::VersionChain;
    
    #[test]
    fn merge_engine_creation() {
        let engine = MergeEngine::new(ConflictResolutionStrategy::TakeLeft);
        assert_eq!(engine.cache_size(), 0);
        assert_eq!(engine.get_statistics().total_merges, 0);
    }
    
    #[test]
    fn merge_statistics_calculation() {
        let mut stats = MergeStatistics::new();
        stats.total_merges = 10;
        stats.successful_merges = 8;
        stats.conflicts_resolved = 5;
        stats.conflicts_unresolved = 2;
        stats.cache_hits = 3;
        stats.cache_misses = 7;
        
        assert_eq!(stats.success_rate(), 0.8);
        assert!((stats.conflict_resolution_rate() - 5.0/7.0).abs() < 0.001);
        assert_eq!(stats.cache_hit_rate(), 0.3);
    }
    
    #[test]
    fn conflict_resolver_basic_strategies() {
        let mut resolver = ConflictResolver::new(ConflictResolutionStrategy::TakeLeft);
        
        let conflict = MergeConflict {
            conflict_id: ConflictId::new(),
            conflict_type: crate::temporal::diff::three_way::ConflictType::ContentConflict,
            position: 1,
            left_content: Some(crate::temporal::diff::DiffContent::Text { 
                content: "left_content".to_string() 
            }),
            right_content: Some(crate::temporal::diff::DiffContent::Text { 
                content: "right_content".to_string() 
            }),
            base_content: None,
            severity: ConflictSeverity::Low,
            auto_resolvable: true,
        };
        
        let resolution = resolver.resolve_single_conflict(&conflict).unwrap();
        assert!(resolution.is_some());
        
        if let Some(DiffOperation::Insert { content, .. }) = resolution {
            if let crate::temporal::diff::DiffContent::Text { content: text } = content {
                assert_eq!(text, "left_content");
            }
        }
    }
}
```

**Final Integration Test Suite (60 min):**
```rust
// tests/integration/diff_merge_integration.rs
use llmkg::temporal::diff::*;
use llmkg::temporal::merge::*;
use llmkg::temporal::version::types::*;

#[cfg(test)]
mod integration_tests {
    use super::*;
    
    #[test]
    fn complete_diff_merge_workflow() {
        // Create test versions
        let base_version = Version::new(1, None, "Base version".to_string());
        let left_version = Version::new(1, Some(base_version.id), "Left changes".to_string());
        let right_version = Version::new(1, Some(base_version.id), "Right changes".to_string());
        
        // Test Myers diff algorithm
        let myers = algorithms::MyersDiffAlgorithm::new();
        let diff_result = myers.diff_text("line1\nline2\nline3", "line1\nmodified\nline3");
        
        assert!(!diff_result.is_identical());
        assert_eq!(diff_result.operation_count(), 1);
        
        // Test three-way merge
        let merger = three_way::ThreeWayMerger::new(
            three_way::ConflictResolutionStrategy::TakeLeft
        );
        
        let base = "line1\nline2\nline3";
        let left = "line1\nmodified_left\nline3";
        let right = "line1\nline2\nline3\nadded_line";
        
        let merge_result = merger.merge(base, left, right);
        
        // Should merge successfully with no conflicts (different positions)
        assert!(merge_result.conflicts.is_empty());
        assert!(merge_result.merge_confidence > 0.8);
        
        println!("Merge completed in {}ms with confidence {:.2}", 
                merge_result.computation_time_ms, merge_result.merge_confidence);
    }
    
    #[test]
    fn conflict_resolution_workflow() {
        let merger = three_way::ThreeWayMerger::new(
            three_way::ConflictResolutionStrategy::Contextual
        );
        
        let base = "original content";
        let left = "left modified content";
        let right = "right modified content";
        
        let merge_result = merger.merge(base, left, right);
        
        // Should detect conflict
        assert!(!merge_result.conflicts.is_empty());
        
        // Test conflict resolver
        let mut resolver = conflict_resolution::ConflictResolver::new(
            three_way::ConflictResolutionStrategy::TakeLeft
        );
        
        let resolutions = resolver.resolve_conflicts(&merge_result.conflicts).unwrap();
        
        // Should resolve at least some conflicts
        assert!(!resolutions.is_empty());
    }
    
    #[test]
    fn large_scale_performance_test() {
        let start = std::time::Instant::now();
        
        // Create large content for performance testing
        let base_lines: Vec<String> = (0..1000).map(|i| format!("line_{}", i)).collect();
        let base_content = base_lines.join("\n");
        
        // Modify some lines
        let mut left_lines = base_lines.clone();
        for i in (0..1000).step_by(10) {
            left_lines[i] = format!("modified_left_{}", i);
        }
        let left_content = left_lines.join("\n");
        
        let mut right_lines = base_lines.clone();
        for i in (5..1000).step_by(10) {
            right_lines[i] = format!("modified_right_{}", i);
        }
        let right_content = right_lines.join("\n");
        
        // Test Myers algorithm performance
        let myers = algorithms::MyersDiffAlgorithm::new();
        let diff_result = myers.diff_text(&base_content, &left_content);
        
        let diff_time = start.elapsed();
        
        // Test merge performance
        let merge_start = std::time::Instant::now();
        let merger = three_way::ThreeWayMerger::new(
            three_way::ConflictResolutionStrategy::TakeLeft
        );
        let merge_result = merger.merge(&base_content, &left_content, &right_content);
        
        let merge_time = merge_start.elapsed();
        
        // Performance assertions
        assert!(diff_time.as_millis() < 1000, "Diff too slow: {:?}", diff_time);
        assert!(merge_time.as_millis() < 2000, "Merge too slow: {:?}", merge_time);
        
        assert!(!diff_result.is_identical());
        assert!(diff_result.operation_count() > 0);
        
        println!("Performance: Diff={}ms, Merge={}ms, Operations={}", 
                diff_time.as_millis(), merge_time.as_millis(), diff_result.operation_count());
    }
    
    #[test]
    fn graph_structure_diff_merge() {
        let myers = algorithms::MyersDiffAlgorithm::new();
        
        let from_nodes = vec![
            (1, vec![("name".to_string(), "alice".to_string())]),
            (2, vec![("name".to_string(), "bob".to_string())]),
            (3, vec![("type".to_string(), "user".to_string())]),
        ];
        
        let to_nodes = vec![
            (1, vec![("name".to_string(), "alice_updated".to_string())]), // Modified
            (2, vec![("name".to_string(), "bob".to_string())]),            // Unchanged
            (4, vec![("name".to_string(), "charlie".to_string())]),        // Added
        ];
        
        let operations = myers.diff_graph_nodes(&from_nodes, &to_nodes);
        
        // Should detect: 1 deletion (node 3), 1 addition (node 4), 1 modification (node 1)
        assert_eq!(operations.len(), 3);
        
        let has_delete = operations.iter().any(|op| matches!(op, DiffOperation::Delete { .. }));
        let has_insert = operations.iter().any(|op| matches!(op, DiffOperation::Insert { .. }));
        let has_modify = operations.iter().any(|op| matches!(op, DiffOperation::Modify { .. }));
        
        assert!(has_delete);
        assert!(has_insert);
        assert!(has_modify);
        
        println!("Graph diff operations: {}", operations.len());
    }
}
```

**Final Validation Sequence:**
```bash
# Complete integration test setup
mkdir -p tests/integration
touch tests/integration/diff_merge_integration.rs

# Run all tests
cargo test --lib temporal::diff::
cargo test --lib temporal::merge::
cargo test --test diff_merge_integration

# Performance validation
cargo test large_scale_performance_test --release

# Final system check
cargo check --all-targets && echo "✅ MicroPhase4 Complete"
```

## PERFORMANCE TARGETS WITH VALIDATION

| Operation | Target | Validation Command |
|-----------|--------|--------------------|
| Myers Diff (1000 lines) | <1000ms | `cargo test large_scale_performance_test --release` |
| Three-Way Merge | <2000ms | `cargo test large_scale_performance_test --release` |
| Conflict Resolution | <100ms | `cargo test conflict_resolution_workflow --release` |
| Graph Diff | <50ms | `cargo test graph_structure_diff_merge --release` |

## SUCCESS CRITERIA CHECKLIST

- [ ] Complete Myers diff algorithm implementation
- [ ] Three-way merge with conflict detection
- [ ] Multiple conflict resolution strategies
- [ ] Merge engine with caching and statistics
- [ ] Graph structure diff capabilities
- [ ] Performance targets met for large datasets
- [ ] No external diff library dependencies
- [ ] Complete error recovery procedures
- [ ] Self-contained implementations with comprehensive tests

**🎯 EXECUTION TARGET: Complete all tasks in 480 minutes with 100% self-containment and production-ready performance**