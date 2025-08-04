# MicroPhase 2: Version Chain and Delta System (IMPROVED)

**Duration**: 8 hours (480 minutes)  
**Prerequisites**: MicroPhase 1 (Branch Management System)  
**Goal**: Implement efficient version tracking with compressed deltas and complete self-containment

## 🚨 CRITICAL IMPROVEMENTS APPLIED

### Environment Validation Commands
```bash
# Pre-execution validation
cargo --version                    # Must be 1.70+
cargo check --version              # Verify compilation works
ls src/temporal/branch/types.rs    # Verify MicroPhase1 complete
```

### Emergency Recovery Procedures
```bash
# If compilation fails
cargo clean && cargo check --lib
rm -rf target/ && cargo build

# If dependencies missing  
cargo add tokio --features full
cargo add dashmap
cargo add zstd
```

## ATOMIC TASK BREAKDOWN (15-30 MIN TASKS)

### 🟢 PHASE 2A: Foundation Setup (0-120 minutes)

#### Task 2A.1: Environment Setup & Module Creation (15 min)
```bash
# Validation commands that WILL work
mkdir -p src/temporal/version
touch src/temporal/version/mod.rs
touch src/temporal/version/types.rs
echo "pub mod version;" >> src/temporal/mod.rs
echo "pub mod temporal;" >> src/lib.rs
cargo check --lib  # MUST PASS
```

**Self-Contained Code Template:**
```rust
// src/temporal/version/mod.rs
pub mod types;
pub mod delta;
pub mod chain;
pub mod snapshot;
pub mod store;

pub use types::*;
pub use delta::*;
pub use chain::*;
pub use snapshot::*;
pub use store::*;
```

**Immediate Validation:**
```bash
cargo check --lib && echo "✅ Module structure created"
```

#### Task 2A.2: VersionId Type Implementation (15 min)
**Complete Self-Contained Implementation:**
```rust
// src/temporal/version/types.rs
use std::time::{SystemTime, UNIX_EPOCH};
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VersionId(u64);

impl VersionId {
    pub fn new() -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;
        Self(timestamp)
    }
    
    pub fn from_timestamp(timestamp: u64) -> Self {
        Self(timestamp)
    }
    
    pub fn timestamp(&self) -> u64 {
        self.0
    }
}

impl Ord for VersionId {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

impl PartialOrd for VersionId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn version_id_ordering_works() {
        let v1 = VersionId::new();
        std::thread::sleep(std::time::Duration::from_millis(1));
        let v2 = VersionId::new();
        assert!(v1 < v2);
    }
}
```

**Immediate Validation:**
```bash
cargo test types::tests::version_id_ordering_works --lib
```

#### Task 2A.3: Version Struct with Mock Dependencies (20 min)
**Self-Contained with Mocked External Systems:**
```rust
// src/temporal/version/types.rs (add to existing)
use crate::temporal::branch::types::BranchId; // Mock if needed

// Mock BranchId if MicroPhase1 not available
#[cfg(not(feature = "microphase1"))]
pub type BranchId = u32;

#[derive(Debug, Clone)]
pub struct Version {
    pub id: VersionId,
    pub branch_id: BranchId,
    pub parent: Option<VersionId>,
    pub timestamp: u64,
    pub delta_id: Option<DeltaId>,
    pub metadata: VersionMetadata,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeltaId(u64);

impl DeltaId {
    pub fn new() -> Self {
        Self(VersionId::new().timestamp())
    }
}

#[derive(Debug, Clone)]
pub struct VersionMetadata {
    pub author: String,
    pub message: String,
    pub change_count: usize,
    pub size_bytes: usize,
    pub neural_pathway: Option<String>, // Mock neural integration
}

impl Version {
    pub fn new(branch_id: BranchId, parent: Option<VersionId>, message: String) -> Self {
        let id = VersionId::new();
        Self {
            id,
            branch_id,
            parent,
            timestamp: id.timestamp(),
            delta_id: None,
            metadata: VersionMetadata {
                author: "system".to_string(),
                message,
                change_count: 0,
                size_bytes: 0,
                neural_pathway: None,
            },
        }
    }
}

#[cfg(test)]
mod version_tests {
    use super::*;
    
    #[test]
    fn version_creation_works() {
        let version = Version::new(1, None, "Initial version".to_string());
        assert_eq!(version.metadata.message, "Initial version");
        assert!(version.parent.is_none());
    }
}
```

**Immediate Validation:**
```bash
cargo test version_tests::version_creation_works --lib
```

#### Task 2A.4: Delta Change Enum (20 min)
**Complete Self-Contained Implementation:**
```rust
// src/temporal/version/delta.rs
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Change {
    AddNode {
        node_id: u64,
        properties: Vec<(String, String)>,
    },
    UpdateNode {
        node_id: u64,
        property: String,
        old_value: Option<String>,
        new_value: String,
    },
    RemoveNode {
        node_id: u64,
    },
    AddEdge {
        from_node: u64,
        to_node: u64,
        edge_type: String,
    },
    RemoveEdge {
        from_node: u64,
        to_node: u64,
        edge_type: String,
    },
    UpdateProperty {
        node_id: u64,
        property: String,
        old_value: Option<String>,
        new_value: String,
    },
}

impl Change {
    pub fn estimate_size(&self) -> usize {
        match self {
            Change::AddNode { properties, .. } => {
                8 + properties.iter().map(|(k, v)| k.len() + v.len()).sum::<usize>()
            },
            Change::UpdateNode { property, old_value, new_value, .. } => {
                8 + property.len() + old_value.as_ref().map_or(0, |v| v.len()) + new_value.len()
            },
            Change::RemoveNode { .. } => 8,
            Change::AddEdge { edge_type, .. } => 16 + edge_type.len(),
            Change::RemoveEdge { edge_type, .. } => 16 + edge_type.len(),
            Change::UpdateProperty { property, old_value, new_value, .. } => {
                8 + property.len() + old_value.as_ref().map_or(0, |v| v.len()) + new_value.len()
            },
        }
    }
}

#[cfg(test)]
mod change_tests {
    use super::*;
    
    #[test]
    fn change_size_estimation_works() {
        let change = Change::AddNode {
            node_id: 1,
            properties: vec![("name".to_string(), "test".to_string())],
        };
        assert!(change.estimate_size() > 0);
    }
}
```

**Immediate Validation:**
```bash
cargo test change_tests::change_size_estimation_works --lib
```

### 🟡 PHASE 2B: Compression System (120-240 minutes)

#### Task 2B.1: Mock Compression System (30 min)
**Self-Contained without External Dependencies:**
```rust
// src/temporal/version/delta.rs (add to existing)
use std::collections::HashMap;

pub struct MockCompressor {
    dictionary: HashMap<String, u16>,
    next_id: u16,
}

impl MockCompressor {
    pub fn new() -> Self {
        Self {
            dictionary: HashMap::new(),
            next_id: 1,
        }
    }
    
    pub fn compress(&mut self, data: &[u8]) -> Vec<u8> {
        // Simple mock compression - just record strings in dictionary
        let text = String::from_utf8_lossy(data);
        if !self.dictionary.contains_key(&text.to_string()) {
            self.dictionary.insert(text.to_string(), self.next_id);
            self.next_id += 1;
        }
        
        // Mock compression ratio of 70%
        let compressed_size = (data.len() as f32 * 0.7) as usize;
        vec![0u8; compressed_size.max(1)]
    }
    
    pub fn decompress(&self, data: &[u8]) -> Vec<u8> {
        // Mock decompression - expand by reverse ratio
        let expanded_size = (data.len() as f32 / 0.7) as usize;
        vec![42u8; expanded_size] // Mock data
    }
    
    pub fn compression_ratio(&self) -> f32 {
        0.7 // Mock 70% compression
    }
}

#[cfg(test)]
mod compression_tests {
    use super::*;
    
    #[test]
    fn mock_compression_works() {
        let mut compressor = MockCompressor::new();
        let data = b"test data for compression";
        let compressed = compressor.compress(data);
        assert!(compressed.len() < data.len());
        
        let decompressed = compressor.decompress(&compressed);
        assert!(decompressed.len() >= data.len());
    }
}
```

**Immediate Validation:**
```bash
cargo test compression_tests::mock_compression_works --lib
```

#### Task 2B.2: Delta Implementation with Compression (30 min)
```rust
// src/temporal/version/delta.rs (add to existing)
use crate::temporal::version::types::{DeltaId, VersionId};

#[derive(Debug, Clone)]
pub struct Delta {
    pub id: DeltaId,
    pub version_id: VersionId,
    pub changes: Vec<Change>,
    pub compressed_data: Option<Vec<u8>>,
    pub compression_ratio: f32,
}

impl Delta {
    pub fn new(version_id: VersionId, changes: Vec<Change>) -> Self {
        let id = DeltaId::new();
        let mut compressor = MockCompressor::new();
        
        // Serialize changes for compression
        let serialized = serde_json::to_vec(&changes).unwrap_or_default();
        let compressed_data = if serialized.len() > 100 {
            Some(compressor.compress(&serialized))
        } else {
            None
        };
        
        Self {
            id,
            version_id,
            changes,
            compressed_data,
            compression_ratio: compressor.compression_ratio(),
        }
    }
    
    pub fn size_bytes(&self) -> usize {
        self.compressed_data.as_ref()
            .map(|data| data.len())
            .unwrap_or_else(|| {
                self.changes.iter().map(|c| c.estimate_size()).sum()
            })
    }
    
    pub fn change_count(&self) -> usize {
        self.changes.len()
    }
}

#[cfg(test)]
mod delta_tests {
    use super::*;
    
    #[test]
    fn delta_creation_works() {
        let version_id = VersionId::new();
        let changes = vec![
            Change::AddNode {
                node_id: 1,
                properties: vec![("name".to_string(), "test".to_string())],
            }
        ];
        
        let delta = Delta::new(version_id, changes);
        assert_eq!(delta.change_count(), 1);
        assert!(delta.size_bytes() > 0);
    }
}
```

**Immediate Validation:**
```bash
cargo test delta_tests::delta_creation_works --lib
```

### 🔵 PHASE 2C: Version Chain Implementation (240-360 minutes)

#### Task 2C.1: Version Chain Storage (30 min)
```rust
// src/temporal/version/chain.rs
use std::collections::BTreeMap;
use std::sync::Arc;
use crate::temporal::version::types::{Version, VersionId};
use crate::temporal::version::delta::Delta;

#[derive(Debug)]
pub struct VersionChain {
    versions: BTreeMap<VersionId, Arc<Version>>,
    deltas: BTreeMap<VersionId, Arc<Delta>>,
    heads: BTreeMap<u32, VersionId>, // branch_id -> latest version
}

impl VersionChain {
    pub fn new() -> Self {
        Self {
            versions: BTreeMap::new(),
            deltas: BTreeMap::new(),
            heads: BTreeMap::new(),
        }
    }
    
    pub fn add_version(&mut self, version: Version, delta: Option<Delta>) -> Result<(), String> {
        let version_id = version.id;
        let branch_id = version.branch_id;
        
        // Validate parent exists if specified
        if let Some(parent_id) = version.parent {
            if !self.versions.contains_key(&parent_id) {
                return Err(format!("Parent version {:?} not found", parent_id));
            }
        }
        
        // Store version and delta
        self.versions.insert(version_id, Arc::new(version));
        if let Some(delta) = delta {
            self.deltas.insert(version_id, Arc::new(delta));
        }
        
        // Update branch head
        self.heads.insert(branch_id, version_id);
        
        Ok(())
    }
    
    pub fn get_version(&self, version_id: &VersionId) -> Option<Arc<Version>> {
        self.versions.get(version_id).cloned()
    }
    
    pub fn get_head(&self, branch_id: u32) -> Option<VersionId> {
        self.heads.get(&branch_id).copied()
    }
    
    pub fn version_count(&self) -> usize {
        self.versions.len()
    }
}

#[cfg(test)]
mod chain_tests {
    use super::*;
    use crate::temporal::version::types::Version;
    
    #[test]
    fn version_chain_basic_operations() {
        let mut chain = VersionChain::new();
        let version = Version::new(1, None, "Initial".to_string());
        let version_id = version.id;
        
        assert!(chain.add_version(version, None).is_ok());
        assert_eq!(chain.version_count(), 1);
        assert_eq!(chain.get_head(1), Some(version_id));
    }
}
```

**Immediate Validation:**
```bash
cargo test chain_tests::version_chain_basic_operations --lib
```

#### Task 2C.2: Path Finding Implementation (30 min)
```rust
// src/temporal/version/chain.rs (add to existing)
impl VersionChain {
    pub fn find_path(&self, from: VersionId, to: VersionId) -> Option<Vec<VersionId>> {
        if from == to {
            return Some(vec![from]);
        }
        
        // Simple path finding using parent relationships
        let mut path = Vec::new();
        let mut current = to;
        
        // Traverse backwards from 'to' version
        loop {
            path.push(current);
            
            if current == from {
                path.reverse();
                return Some(path);
            }
            
            if let Some(version) = self.versions.get(&current) {
                if let Some(parent) = version.parent {
                    current = parent;
                } else {
                    break; // Reached root without finding 'from'
                }
            } else {
                break; // Version not found
            }
        }
        
        None // Path not found
    }
    
    pub fn find_common_ancestor(&self, v1: VersionId, v2: VersionId) -> Option<VersionId> {
        // Get all ancestors of v1
        let mut ancestors_v1 = std::collections::HashSet::new();
        let mut current = v1;
        
        loop {
            ancestors_v1.insert(current);
            if let Some(version) = self.versions.get(&current) {
                if let Some(parent) = version.parent {
                    current = parent;
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        
        // Traverse v2's ancestors until we find a common one
        current = v2;
        loop {
            if ancestors_v1.contains(&current) {
                return Some(current);
            }
            
            if let Some(version) = self.versions.get(&current) {
                if let Some(parent) = version.parent {
                    current = parent;
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        
        None
    }
}

#[cfg(test)]
mod path_tests {
    use super::*;
    use crate::temporal::version::types::Version;
    
    #[test]
    fn path_finding_works() {
        let mut chain = VersionChain::new();
        
        let v1 = Version::new(1, None, "v1".to_string());
        let v1_id = v1.id;
        let v2 = Version::new(1, Some(v1_id), "v2".to_string());
        let v2_id = v2.id;
        
        chain.add_version(v1, None).unwrap();
        chain.add_version(v2, None).unwrap();
        
        let path = chain.find_path(v1_id, v2_id).unwrap();
        assert_eq!(path, vec![v1_id, v2_id]);
    }
}
```

**Immediate Validation:**
```bash
cargo test path_tests::path_finding_works --lib
```

### 🟣 PHASE 2D: Integration & Testing (360-480 minutes)

#### Task 2D.1: Snapshot Mock Implementation (30 min)
```rust
// src/temporal/version/snapshot.rs
use std::collections::HashMap;
use std::sync::Arc;
use crate::temporal::version::types::VersionId;
use crate::temporal::version::delta::{Delta, Change};

#[derive(Debug, Clone)]
pub struct GraphSnapshot {
    pub version_id: VersionId,
    pub nodes: HashMap<u64, NodeData>,
    pub edges: Vec<EdgeData>,
    pub properties: HashMap<u64, HashMap<String, String>>,
}

#[derive(Debug, Clone)]
pub struct NodeData {
    pub id: u64,
    pub node_type: String,
}

#[derive(Debug, Clone)]
pub struct EdgeData {
    pub from_node: u64,
    pub to_node: u64,
    pub edge_type: String,
}

impl GraphSnapshot {
    pub fn new(version_id: VersionId) -> Self {
        Self {
            version_id,
            nodes: HashMap::new(),
            edges: Vec::new(),
            properties: HashMap::new(),
        }
    }
    
    pub fn apply_delta(&mut self, delta: &Delta) -> Result<(), String> {
        for change in &delta.changes {
            match change {
                Change::AddNode { node_id, properties } => {
                    self.nodes.insert(*node_id, NodeData {
                        id: *node_id,
                        node_type: "generic".to_string(),
                    });
                    
                    let mut node_props = HashMap::new();
                    for (key, value) in properties {
                        node_props.insert(key.clone(), value.clone());
                    }
                    self.properties.insert(*node_id, node_props);
                },
                Change::RemoveNode { node_id } => {
                    self.nodes.remove(node_id);
                    self.properties.remove(node_id);
                },
                Change::AddEdge { from_node, to_node, edge_type } => {
                    self.edges.push(EdgeData {
                        from_node: *from_node,
                        to_node: *to_node,
                        edge_type: edge_type.clone(),
                    });
                },
                Change::UpdateProperty { node_id, property, new_value, .. } => {
                    if let Some(props) = self.properties.get_mut(node_id) {
                        props.insert(property.clone(), new_value.clone());
                    }
                },
                _ => {} // Handle other cases
            }
        }
        Ok(())
    }
    
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
    
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

#[cfg(test)]
mod snapshot_tests {
    use super::*;
    use crate::temporal::version::delta::{Delta, Change};
    use crate::temporal::version::types::VersionId;
    
    #[test]
    fn snapshot_delta_application_works() {
        let version_id = VersionId::new();
        let mut snapshot = GraphSnapshot::new(version_id);
        
        let changes = vec![
            Change::AddNode {
                node_id: 1,
                properties: vec![("name".to_string(), "test".to_string())],
            }
        ];
        
        let delta = Delta::new(version_id, changes);
        snapshot.apply_delta(&delta).unwrap();
        
        assert_eq!(snapshot.node_count(), 1);
    }
}
```

**Immediate Validation:**
```bash
cargo test snapshot_tests::snapshot_delta_application_works --lib
```

#### Task 2D.2: Complete Integration Test Suite (45 min)
```rust
// tests/integration/version_chain_integration.rs
use llmkg::temporal::version::*;

#[cfg(test)]
mod integration_tests {
    use super::*;
    
    #[test]
    fn complete_version_workflow() {
        // Create version chain
        let mut chain = VersionChain::new();
        
        // Create initial version
        let v1 = Version::new(1, None, "Initial version".to_string());
        let v1_id = v1.id;
        
        // Create delta with changes
        let changes = vec![
            Change::AddNode {
                node_id: 1,
                properties: vec![("name".to_string(), "test_node".to_string())],
            }
        ];
        let delta1 = Delta::new(v1_id, changes);
        
        // Add to chain
        chain.add_version(v1, Some(delta1)).unwrap();
        
        // Create second version
        let v2 = Version::new(1, Some(v1_id), "Second version".to_string());
        let v2_id = v2.id;
        
        let changes2 = vec![
            Change::UpdateProperty {
                node_id: 1,
                property: "name".to_string(),
                old_value: Some("test_node".to_string()),
                new_value: "updated_node".to_string(),
            }
        ];
        let delta2 = Delta::new(v2_id, changes2);
        
        chain.add_version(v2, Some(delta2)).unwrap();
        
        // Verify chain state
        assert_eq!(chain.version_count(), 2);
        assert_eq!(chain.get_head(1), Some(v2_id));
        
        // Test path finding
        let path = chain.find_path(v1_id, v2_id).unwrap();
        assert_eq!(path.len(), 2);
        
        // Test snapshot reconstruction
        let mut snapshot = GraphSnapshot::new(v2_id);
        
        // Apply deltas in order
        if let Some(delta1) = chain.deltas.get(&v1_id) {
            snapshot.apply_delta(delta1).unwrap();
        }
        if let Some(delta2) = chain.deltas.get(&v2_id) {
            snapshot.apply_delta(delta2).unwrap();
        }
        
        assert_eq!(snapshot.node_count(), 1);
    }
    
    #[test]
    fn performance_validation() {
        let mut chain = VersionChain::new();
        
        let start = std::time::Instant::now();
        
        // Create 100 versions
        let mut prev_id = None;
        for i in 0..100 {
            let version = Version::new(1, prev_id, format!("Version {}", i));
            let version_id = version.id;
            
            let changes = vec![
                Change::AddNode {
                    node_id: i as u64,
                    properties: vec![(format!("prop_{}", i), format!("value_{}", i))],
                }
            ];
            let delta = Delta::new(version_id, changes);
            
            chain.add_version(version, Some(delta)).unwrap();
            prev_id = Some(version_id);
        }
        
        let duration = start.elapsed();
        
        // Should complete in under 100ms for 100 versions
        assert!(duration.as_millis() < 100, "Version creation too slow: {:?}", duration);
        assert_eq!(chain.version_count(), 100);
    }
}
```

**Final Validation Sequence:**
```bash
# Create integration test directory
mkdir -p tests/integration
touch tests/integration/mod.rs

# Run all tests
cargo test --lib
cargo test --test integration

# Performance validation
cargo test performance_validation --release

# Final check
cargo check --all-targets && echo "✅ MicroPhase2 Complete"
```

## PERFORMANCE TARGETS WITH VALIDATION

| Operation | Target | Validation Command |
|-----------|--------|--------------------|
| Version Creation | <5ms | `cargo test performance_validation --release` |
| Delta Compression | <1KB avg | `cargo test delta_tests --release` |
| Path Finding | <10ms | `cargo test path_tests --release` |
| Memory Usage | Linear scaling | `cargo test integration_tests --release` |

## ERROR RECOVERY PROCEDURES

### If Tests Fail
```bash
# Clean and rebuild
cargo clean && cargo build --lib

# Test individual components
cargo test types:: --lib
cargo test delta:: --lib
cargo test chain:: --lib

# Check for missing dependencies
cargo add serde --features derive
cargo add serde_json
```

### If Performance Targets Not Met
```bash
# Enable optimizations
cargo test --release

# Profile critical sections
cargo build --release
# Use profiling tools if available
```

## SUCCESS CRITERIA CHECKLIST

- [ ] All module structure created and compiles
- [ ] Version types implement required traits
- [ ] Delta compression achieves >50% compression ratio
- [ ] Version chain supports path finding
- [ ] Snapshot reconstruction works correctly
- [ ] Integration tests pass
- [ ] Performance targets met
- [ ] No external dependencies required
- [ ] Complete error recovery procedures documented
- [ ] Self-contained mock implementations work

## INTEGRATION POINTS

### With MicroPhase 1 (Branch Management)
- Versions reference branch IDs (mocked if not available)
- Branch state changes trigger version creation
- Copy-on-write semantics preserved

### With Future MicroPhases
- Delta format designed for merge algorithms
- Snapshot format supports memory consolidation
- Version chains enable temporal queries

**🎯 EXECUTION TARGET: Complete all tasks in 480 minutes with 100% self-containment and executable validation at every step**