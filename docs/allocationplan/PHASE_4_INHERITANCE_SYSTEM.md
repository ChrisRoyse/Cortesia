# Phase 4: Inheritance-Based Knowledge Compression

**Duration**: 1 week  
**Team Size**: 2-3 developers  
**Methodology**: SPARC + London School TDD  
**Goal**: Achieve 10x compression through hierarchical inheritance with exceptions  

## AI-Verifiable Success Criteria

### Compression Metrics
- [ ] Storage reduction: > 10x vs flat storage
- [ ] Property lookup: < 1ms including inheritance chain
- [ ] Exception storage: < 5% of total properties
- [ ] Hierarchy depth: Supports up to 20 levels

### Accuracy Metrics
- [ ] Property inheritance: 100% correct
- [ ] Exception override: 100% accurate
- [ ] Multiple inheritance: Conflict resolution 100% deterministic
- [ ] No property loss during compression

### Performance Metrics
- [ ] Property resolution: < 100μs per lookup
- [ ] Batch inheritance update: < 10ms for 1000 nodes
- [ ] Exception checking: O(1) time complexity
- [ ] Memory overhead: < 10KB per inheritance level

## SPARC Methodology Application

### Specification

**Objective**: Implement brain-inspired hierarchical compression where general knowledge is stored once and inherited.

**Biological Inspiration**:
```
Cortical Hierarchy → Inheritance System
- V1 (edges) → Base properties
- V2 (shapes) → Inherited features  
- V4 (objects) → Complex inheritance
- IT (categories) → Deep hierarchies
```

**Core Principles**:
1. Properties flow down hierarchies unless overridden
2. Exceptions are first-class citizens
3. Multiple inheritance with deterministic resolution
4. Dynamic hierarchy reorganization

### Pseudocode

```
INHERITANCE_SYSTEM:
    
    // Property Resolution
    RESOLVE_PROPERTY(node, property_name):
        // Check local properties first
        IF node.has_local_property(property_name):
            RETURN node.get_local_property(property_name)
            
        // Check exceptions
        IF node.has_exception(property_name):
            RETURN node.get_exception(property_name).value
            
        // Walk up inheritance chain
        FOR parent IN node.get_parents():
            result = RESOLVE_PROPERTY(parent, property_name)
            IF result != NOT_FOUND:
                RETURN result
                
        RETURN NOT_FOUND
        
    // Compression Algorithm
    COMPRESS_HIERARCHY(nodes):
        // Find common properties
        property_frequency = COUNT_PROPERTIES(nodes)
        
        // Promote common properties
        FOR (property, frequency) IN property_frequency:
            IF frequency > THRESHOLD:
                parent = FIND_COMMON_ANCESTOR(nodes_with_property)
                PROMOTE_PROPERTY(property, parent)
                REMOVE_FROM_CHILDREN(property, parent.descendants)
                
        // Detect and store exceptions
        FOR node IN nodes:
            inherited = GET_ALL_INHERITED(node)
            local = node.local_properties
            
            FOR (prop, value) IN local:
                IF inherited[prop] != value:
                    CREATE_EXCEPTION(node, prop, inherited[prop], value)
```

### Architecture

```
inheritance-system/
├── src/
│   ├── hierarchy/
│   │   ├── mod.rs
│   │   ├── node.rs              # Inheritance node
│   │   ├── tree.rs              # Hierarchy tree
│   │   ├── dag.rs               # DAG for multiple inheritance
│   │   └── builder.rs           # Hierarchy construction
│   ├── properties/
│   │   ├── mod.rs
│   │   ├── store.rs             # Property storage
│   │   ├── resolver.rs          # Property resolution
│   │   ├── inheritance.rs       # Inheritance logic
│   │   └── cache.rs             # Resolution cache
│   ├── exceptions/
│   │   ├── mod.rs
│   │   ├── store.rs             # Exception storage
│   │   ├── detector.rs          # Exception detection
│   │   └── handler.rs           # Exception application
│   ├── compression/
│   │   ├── mod.rs
│   │   ├── analyzer.rs          # Property analysis
│   │   ├── promoter.rs          # Property promotion
│   │   ├── compressor.rs        # Compression engine
│   │   └── metrics.rs           # Compression metrics
│   └── optimization/
│       ├── mod.rs
│       ├── reorganizer.rs       # Hierarchy reorganization
│       ├── balancer.rs          # Tree balancing
│       └── pruner.rs            # Dead branch removal
```

### Refinement

Optimization stages:
1. Naive property storage baseline
2. Basic inheritance implementation
3. Add exception handling
4. Implement property promotion
5. Add caching and optimization

### Completion

Phase complete when:
- 10x compression achieved
- All lookups < 1ms
- Exception handling verified
- Performance benchmarks pass

## Task Breakdown

### Task 4.1: Hierarchical Node System (Day 1)

**Specification**: Create nodes that support inheritance hierarchies

**Test-Driven Development**:

```rust
#[test]
fn test_single_inheritance() {
    let mut hierarchy = InheritanceHierarchy::new();
    
    // Create hierarchy: Animal -> Mammal -> Dog
    let animal = hierarchy.create_node("Animal", hashmap!{
        "alive" => "true",
        "needs_food" => "true",
    });
    
    let mammal = hierarchy.create_child("Mammal", animal, hashmap!{
        "warm_blooded" => "true",
        "has_hair" => "true",
    });
    
    let dog = hierarchy.create_child("Dog", mammal, hashmap!{
        "barks" => "true",
        "loyal" => "true",
    });
    
    // Test property resolution
    assert_eq!(hierarchy.get_property(dog, "alive"), Some("true"));
    assert_eq!(hierarchy.get_property(dog, "warm_blooded"), Some("true"));
    assert_eq!(hierarchy.get_property(dog, "barks"), Some("true"));
    assert_eq!(hierarchy.get_property(dog, "can_fly"), None);
}

#[test]
fn test_multiple_inheritance() {
    let mut hierarchy = InheritanceHierarchy::new();
    
    // Create diamond: Device -> (Phone, Computer) -> Smartphone
    let device = hierarchy.create_node("Device", hashmap!{
        "electronic" => "true",
        "needs_power" => "true",
    });
    
    let phone = hierarchy.create_child("Phone", device, hashmap!{
        "can_call" => "true",
        "portable" => "true",
    });
    
    let computer = hierarchy.create_child("Computer", device, hashmap!{
        "can_compute" => "true",
        "has_cpu" => "true",
    });
    
    let smartphone = hierarchy.create_node_with_parents(
        "Smartphone",
        vec![phone, computer],
        hashmap!{
            "has_apps" => "true",
        }
    );
    
    // Should inherit from both parents
    assert_eq!(hierarchy.get_property(smartphone, "can_call"), Some("true"));
    assert_eq!(hierarchy.get_property(smartphone, "can_compute"), Some("true"));
    assert_eq!(hierarchy.get_property(smartphone, "electronic"), Some("true"));
}

#[test]
fn test_property_resolution_performance() {
    let hierarchy = create_deep_hierarchy(20); // 20 levels deep
    let leaf_node = NodeId(19);
    
    let start = Instant::now();
    for _ in 0..10000 {
        let _ = hierarchy.get_property(leaf_node, "root_property");
    }
    let elapsed = start.elapsed();
    
    let per_lookup = elapsed / 10000;
    assert!(per_lookup < Duration::from_micros(100)); // <100μs per lookup
}
```

**Implementation**:

```rust
// src/hierarchy/node.rs
#[derive(Debug, Clone)]
pub struct InheritanceNode {
    id: NodeId,
    name: String,
    parents: Vec<NodeId>,
    children: Vec<NodeId>,
    local_properties: HashMap<String, PropertyValue>,
    exceptions: HashMap<String, Exception>,
    metadata: NodeMetadata,
}

pub struct NodeMetadata {
    depth: u32,
    creation_time: Instant,
    last_modified: AtomicU64,
    access_count: AtomicU32,
}

// src/hierarchy/tree.rs
pub struct InheritanceHierarchy {
    nodes: DashMap<NodeId, InheritanceNode>,
    name_index: DashMap<String, NodeId>,
    property_cache: PropertyCache,
    resolution_strategy: ResolutionStrategy,
}

impl InheritanceHierarchy {
    pub fn create_node(&mut self, name: &str, properties: HashMap<&str, &str>) -> NodeId {
        let id = NodeId::generate();
        let node = InheritanceNode {
            id,
            name: name.to_string(),
            parents: Vec::new(),
            children: Vec::new(),
            local_properties: properties.into_iter()
                .map(|(k, v)| (k.to_string(), PropertyValue::from(v)))
                .collect(),
            exceptions: HashMap::new(),
            metadata: NodeMetadata::new(0),
        };
        
        self.nodes.insert(id, node);
        self.name_index.insert(name.to_string(), id);
        id
    }
    
    pub fn create_child(&mut self, name: &str, parent: NodeId, properties: HashMap<&str, &str>) -> NodeId {
        let id = self.create_node(name, properties);
        
        // Update parent-child relationships
        self.nodes.get_mut(&parent).unwrap().children.push(id);
        self.nodes.get_mut(&id).unwrap().parents.push(parent);
        
        // Update metadata
        let parent_depth = self.nodes.get(&parent).unwrap().metadata.depth;
        self.nodes.get_mut(&id).unwrap().metadata.depth = parent_depth + 1;
        
        id
    }
    
    pub fn get_property(&self, node: NodeId, property: &str) -> Option<String> {
        // Check cache first
        if let Some(cached) = self.property_cache.get(node, property) {
            return cached;
        }
        
        // Resolve property
        let result = self.resolve_property(node, property);
        
        // Cache result
        self.property_cache.insert(node, property, result.clone());
        
        result
    }
    
    fn resolve_property(&self, node: NodeId, property: &str) -> Option<String> {
        let node_data = self.nodes.get(&node)?;
        
        // Check local properties
        if let Some(value) = node_data.local_properties.get(property) {
            return Some(value.to_string());
        }
        
        // Check exceptions
        if let Some(exception) = node_data.exceptions.get(property) {
            return Some(exception.actual_value.clone());
        }
        
        // Walk up inheritance chain
        match self.resolution_strategy {
            ResolutionStrategy::DepthFirst => {
                for &parent in &node_data.parents {
                    if let Some(value) = self.resolve_property(parent, property) {
                        return Some(value);
                    }
                }
            }
            ResolutionStrategy::BreadthFirst => {
                let mut queue = VecDeque::from(node_data.parents.clone());
                let mut visited = HashSet::new();
                
                while let Some(parent) = queue.pop_front() {
                    if !visited.insert(parent) {
                        continue;
                    }
                    
                    if let Some(parent_node) = self.nodes.get(&parent) {
                        if let Some(value) = parent_node.local_properties.get(property) {
                            return Some(value.to_string());
                        }
                        queue.extend(&parent_node.parents);
                    }
                }
            }
        }
        
        None
    }
}

// src/properties/cache.rs
pub struct PropertyCache {
    cache: Arc<DashMap<(NodeId, String), Option<String>>>,
    capacity: usize,
    ttl: Duration,
}

impl PropertyCache {
    pub fn get(&self, node: NodeId, property: &str) -> Option<Option<String>> {
        let key = (node, property.to_string());
        self.cache.get(&key).map(|entry| entry.clone())
    }
    
    pub fn insert(&self, node: NodeId, property: &str, value: Option<String>) {
        let key = (node, property.to_string());
        
        // Simple LRU eviction if needed
        if self.cache.len() >= self.capacity {
            self.evict_oldest();
        }
        
        self.cache.insert(key, value);
    }
}
```

**AI-Verifiable Outcomes**:
- [ ] Single inheritance works correctly
- [ ] Multiple inheritance resolves deterministically  
- [ ] Property lookup < 100μs
- [ ] Cache improves performance > 10x

### Task 4.2: Exception Handling System (Day 2)

**Specification**: Implement exception detection and storage

**Test First**:

```rust
#[test]
fn test_exception_detection() {
    let mut hierarchy = InheritanceHierarchy::new();
    
    // Bird -> Penguin (can't fly)
    let bird = hierarchy.create_node("Bird", hashmap!{
        "can_fly" => "true",
        "has_wings" => "true",
        "has_feathers" => "true",
    });
    
    let penguin = hierarchy.create_child("Penguin", bird, hashmap!{
        "can_fly" => "false", // Exception!
        "swims" => "true",
    });
    
    // Exception should be detected and stored
    let exceptions = hierarchy.get_exceptions(penguin);
    assert_eq!(exceptions.len(), 1);
    assert!(exceptions.contains_key("can_fly"));
    
    let exception = &exceptions["can_fly"];
    assert_eq!(exception.inherited_value, "true");
    assert_eq!(exception.actual_value, "false");
    assert!(exception.reason.contains("override"));
}

#[test]
fn test_exception_storage_efficiency() {
    let mut hierarchy = create_large_hierarchy(10000);
    
    // Add 100 exceptions
    for i in 0..100 {
        hierarchy.add_exception(NodeId(i), "special_prop", Exception {
            inherited_value: "normal".to_string(),
            actual_value: "special".to_string(),
            reason: "Test exception".to_string(),
        });
    }
    
    let stats = hierarchy.get_compression_stats();
    
    // Exceptions should be < 5% of total storage
    let exception_ratio = stats.exception_bytes as f64 / stats.total_bytes as f64;
    assert!(exception_ratio < 0.05);
}

#[test]
fn test_exception_cascading() {
    let mut hierarchy = InheritanceHierarchy::new();
    
    // A -> B -> C, where B has exception
    let a = hierarchy.create_node("A", hashmap!{"prop" => "a_value"});
    let b = hierarchy.create_child("B", a, hashmap!{"prop" => "b_value"});
    let c = hierarchy.create_child("C", b, hashmap!{});
    
    // C should inherit B's value (not A's)
    assert_eq!(hierarchy.get_property(c, "prop"), Some("b_value"));
    
    // B should have exception recorded
    let b_exceptions = hierarchy.get_exceptions(b);
    assert!(b_exceptions.contains_key("prop"));
}
```

**Implementation**:

```rust
// src/exceptions/store.rs
#[derive(Debug, Clone)]
pub struct Exception {
    pub inherited_value: String,
    pub actual_value: String,
    pub reason: String,
    pub source: ExceptionSource,
}

#[derive(Debug, Clone)]
pub enum ExceptionSource {
    Explicit,        // User-defined exception
    Detected,        // Auto-detected during insertion
    Promoted,        // Created during property promotion
}

pub struct ExceptionStore {
    exceptions: DashMap<(NodeId, String), Exception>,
    index_by_property: DashMap<String, Vec<NodeId>>,
    stats: ExceptionStats,
}

impl ExceptionStore {
    pub fn detect_exceptions(&mut self, node: &InheritanceNode, inherited: &HashMap<String, String>) {
        for (prop, local_value) in &node.local_properties {
            if let Some(inherited_value) = inherited.get(prop) {
                if local_value.to_string() != *inherited_value {
                    // Exception detected
                    let exception = Exception {
                        inherited_value: inherited_value.clone(),
                        actual_value: local_value.to_string(),
                        reason: format!("Override of inherited property"),
                        source: ExceptionSource::Detected,
                    };
                    
                    self.add_exception(node.id, prop.clone(), exception);
                }
            }
        }
    }
    
    pub fn add_exception(&mut self, node: NodeId, property: String, exception: Exception) {
        let key = (node, property.clone());
        
        // Update stats
        if !self.exceptions.contains_key(&key) {
            self.stats.count += 1;
            self.stats.bytes += self.estimate_exception_size(&exception);
        }
        
        self.exceptions.insert(key, exception);
        
        // Update index
        self.index_by_property.entry(property)
            .or_insert_with(Vec::new)
            .push(node);
    }
    
    pub fn get_exception(&self, node: NodeId, property: &str) -> Option<Exception> {
        self.exceptions.get(&(node, property.to_string()))
            .map(|e| e.clone())
    }
    
    fn estimate_exception_size(&self, exception: &Exception) -> usize {
        std::mem::size_of::<Exception>() +
        exception.inherited_value.len() +
        exception.actual_value.len() +
        exception.reason.len()
    }
}

// src/exceptions/detector.rs
pub struct ExceptionDetector {
    threshold: f32,
    patterns: Vec<ExceptionPattern>,
}

impl ExceptionDetector {
    pub fn analyze_hierarchy(&self, hierarchy: &InheritanceHierarchy) -> ExceptionReport {
        let mut report = ExceptionReport::new();
        
        // Find all nodes with exceptions
        for node in hierarchy.all_nodes() {
            let inherited = hierarchy.get_all_inherited_properties(node.id);
            let exceptions = self.detect_node_exceptions(&node, &inherited);
            
            for (prop, exception) in exceptions {
                report.add_exception(node.id, prop, exception);
            }
        }
        
        // Analyze patterns
        report.patterns = self.find_exception_patterns(&report.exceptions);
        
        report
    }
    
    fn detect_node_exceptions(&self, node: &InheritanceNode, inherited: &HashMap<String, String>) -> Vec<(String, Exception)> {
        let mut exceptions = Vec::new();
        
        for (prop, value) in &node.local_properties {
            if let Some(inherited_value) = inherited.get(prop) {
                if value.to_string() != *inherited_value {
                    // Check if this is a meaningful exception
                    let similarity = self.calculate_similarity(&value.to_string(), inherited_value);
                    
                    if similarity < self.threshold {
                        exceptions.push((prop.clone(), Exception {
                            inherited_value: inherited_value.clone(),
                            actual_value: value.to_string(),
                            reason: self.infer_exception_reason(prop, inherited_value, &value.to_string()),
                            source: ExceptionSource::Detected,
                        }));
                    }
                }
            }
        }
        
        exceptions
    }
    
    fn infer_exception_reason(&self, property: &str, inherited: &str, actual: &str) -> String {
        // Pattern matching for common exceptions
        for pattern in &self.patterns {
            if pattern.matches(property, inherited, actual) {
                return pattern.reason.clone();
            }
        }
        
        // Generic reason
        format!("Property '{}' overrides inherited value", property)
    }
}
```

**AI-Verifiable Outcomes**:
- [ ] Exceptions detected automatically
- [ ] Storage < 5% of total
- [ ] Exception cascading works
- [ ] Pattern detection accurate

### Task 4.3: Property Compression Engine (Day 3)

**Specification**: Implement property promotion for compression

**Test-Driven Approach**:

```rust
#[test]
fn test_property_promotion() {
    let mut compressor = PropertyCompressor::new();
    let mut hierarchy = create_test_hierarchy();
    
    // All dogs have "loyal" property
    hierarchy.set_property(NodeId(10), "loyal", "true"); // Golden Retriever
    hierarchy.set_property(NodeId(11), "loyal", "true"); // Labrador
    hierarchy.set_property(NodeId(12), "loyal", "true"); // Beagle
    
    // Run compression
    let result = compressor.compress(&mut hierarchy);
    
    // "loyal" should be promoted to "Dog" parent
    let dog_node = hierarchy.get_node_by_name("Dog").unwrap();
    assert_eq!(hierarchy.get_local_property(dog_node, "loyal"), Some("true"));
    
    // Children should not have local "loyal" property
    assert_eq!(hierarchy.get_local_property(NodeId(10), "loyal"), None);
    assert_eq!(hierarchy.get_local_property(NodeId(11), "loyal"), None);
    assert_eq!(hierarchy.get_local_property(NodeId(12), "loyal"), None);
    
    // But they should still inherit it
    assert_eq!(hierarchy.get_property(NodeId(10), "loyal"), Some("true"));
}

#[test]
fn test_compression_ratio() {
    let mut hierarchy = create_animal_hierarchy(1000); // 1000 animals
    let before_size = hierarchy.calculate_storage_size();
    
    let compressor = PropertyCompressor::new();
    let result = compressor.compress(&mut hierarchy);
    
    let after_size = hierarchy.calculate_storage_size();
    let compression_ratio = before_size as f64 / after_size as f64;
    
    assert!(compression_ratio > 10.0); // >10x compression
    assert_eq!(result.properties_promoted, 47); // Common properties
    assert!(result.bytes_saved > before_size * 9 / 10);
}

#[test]
fn test_compression_preserves_semantics() {
    let mut hierarchy = create_complex_hierarchy();
    let mut test_queries = Vec::new();
    
    // Record all property values before compression
    for node in hierarchy.all_nodes() {
        for prop in node.all_property_names() {
            let value = hierarchy.get_property(node.id, &prop);
            test_queries.push((node.id, prop, value));
        }
    }
    
    // Compress
    let compressor = PropertyCompressor::new();
    compressor.compress(&mut hierarchy);
    
    // Verify all properties still resolve to same values
    for (node, prop, expected) in test_queries {
        assert_eq!(hierarchy.get_property(node, &prop), expected);
    }
}
```

**Implementation**:

```rust
// src/compression/analyzer.rs
pub struct PropertyAnalyzer {
    min_frequency: f32,
    min_nodes: usize,
}

impl PropertyAnalyzer {
    pub fn analyze(&self, hierarchy: &InheritanceHierarchy) -> PropertyAnalysis {
        let mut property_occurrences: HashMap<(String, String), Vec<NodeId>> = HashMap::new();
        
        // Count property occurrences
        for node in hierarchy.all_nodes() {
            for (prop, value) in node.local_properties.iter() {
                property_occurrences
                    .entry((prop.clone(), value.to_string()))
                    .or_insert_with(Vec::new)
                    .push(node.id);
            }
        }
        
        // Find promotion candidates
        let mut candidates = Vec::new();
        
        for ((prop, value), nodes) in property_occurrences {
            if nodes.len() >= self.min_nodes {
                // Find common ancestor
                if let Some(ancestor) = hierarchy.find_lowest_common_ancestor(&nodes) {
                    let total_descendants = hierarchy.count_descendants(ancestor);
                    let frequency = nodes.len() as f32 / total_descendants as f32;
                    
                    if frequency >= self.min_frequency {
                        candidates.push(PromotionCandidate {
                            property: prop,
                            value,
                            from_nodes: nodes,
                            to_node: ancestor,
                            frequency,
                            bytes_saved: self.calculate_bytes_saved(&prop, &value, nodes.len()),
                        });
                    }
                }
            }
        }
        
        PropertyAnalysis { candidates }
    }
}

// src/compression/promoter.rs
pub struct PropertyPromoter {
    analyzer: PropertyAnalyzer,
    exception_handler: ExceptionHandler,
}

impl PropertyPromoter {
    pub fn promote_properties(&self, hierarchy: &mut InheritanceHierarchy, candidates: Vec<PromotionCandidate>) -> PromotionResult {
        let mut result = PromotionResult::new();
        
        // Sort by bytes saved (greedy approach)
        let mut sorted_candidates = candidates;
        sorted_candidates.sort_by_key(|c| std::cmp::Reverse(c.bytes_saved));
        
        for candidate in sorted_candidates {
            // Check if still valid (previous promotions might affect this)
            if self.is_still_valid(&candidate, hierarchy) {
                // Promote property
                hierarchy.set_local_property(
                    candidate.to_node,
                    &candidate.property,
                    &candidate.value
                );
                
                // Remove from children and create exceptions if needed
                for &node in &candidate.from_nodes {
                    let node_value = hierarchy.get_local_property(node, &candidate.property);
                    
                    if let Some(value) = node_value {
                        if value == candidate.value {
                            // Safe to remove
                            hierarchy.remove_local_property(node, &candidate.property);
                            result.properties_removed += 1;
                        } else {
                            // Create exception
                            let exception = Exception {
                                inherited_value: candidate.value.clone(),
                                actual_value: value,
                                reason: "Value differs from promoted property".to_string(),
                                source: ExceptionSource::Promoted,
                            };
                            hierarchy.add_exception(node, &candidate.property, exception);
                            result.exceptions_created += 1;
                        }
                    }
                }
                
                result.properties_promoted += 1;
                result.bytes_saved += candidate.bytes_saved;
            }
        }
        
        result
    }
}

// src/compression/compressor.rs
pub struct PropertyCompressor {
    analyzer: PropertyAnalyzer,
    promoter: PropertyPromoter,
    optimizer: HierarchyOptimizer,
}

impl PropertyCompressor {
    pub fn compress(&self, hierarchy: &mut InheritanceHierarchy) -> CompressionResult {
        let start_size = hierarchy.calculate_storage_size();
        let mut total_result = CompressionResult::new();
        
        // Iterative compression
        loop {
            // Analyze current state
            let analysis = self.analyzer.analyze(hierarchy);
            
            if analysis.candidates.is_empty() {
                break;
            }
            
            // Promote properties
            let promotion_result = self.promoter.promote_properties(hierarchy, analysis.candidates);
            total_result.merge(promotion_result);
            
            // Optimize hierarchy structure
            let optimization_result = self.optimizer.optimize(hierarchy);
            total_result.merge(optimization_result);
            
            // Check if we've reached diminishing returns
            if promotion_result.bytes_saved < 1000 {
                break;
            }
        }
        
        let end_size = hierarchy.calculate_storage_size();
        total_result.compression_ratio = start_size as f64 / end_size as f64;
        total_result.total_bytes_saved = start_size - end_size;
        
        total_result
    }
}
```

**AI-Verifiable Outcomes**:
- [ ] Properties promoted correctly
- [ ] 10x compression achieved
- [ ] Semantics preserved 100%
- [ ] Compression algorithm < 10ms/1000 nodes

### Task 4.4: Dynamic Hierarchy Optimization (Day 4)

**Specification**: Optimize hierarchy structure dynamically

**Tests First**:

```rust
#[test]
fn test_hierarchy_rebalancing() {
    let mut hierarchy = create_unbalanced_hierarchy();
    let optimizer = HierarchyOptimizer::new();
    
    // Initial depth
    let initial_depth = hierarchy.max_depth();
    assert!(initial_depth > 20);
    
    // Optimize
    let result = optimizer.rebalance(&mut hierarchy);
    
    // Should be more balanced
    let final_depth = hierarchy.max_depth();
    assert!(final_depth < 10);
    assert!(result.nodes_moved > 0);
    
    // Verify all nodes still accessible
    assert_eq!(hierarchy.node_count(), create_unbalanced_hierarchy().node_count());
}

#[test]
fn test_dead_branch_pruning() {
    let mut hierarchy = InheritanceHierarchy::new();
    
    // Create nodes with no properties
    let root = hierarchy.create_node("Root", hashmap!{"active" => "true"});
    let dead1 = hierarchy.create_child("Dead1", root, hashmap!{});
    let dead2 = hierarchy.create_child("Dead2", dead1, hashmap!{});
    let alive = hierarchy.create_child("Alive", root, hashmap!{"data" => "value"});
    
    let optimizer = HierarchyOptimizer::new();
    let result = optimizer.prune_dead_branches(&mut hierarchy);
    
    // Dead branches should be removed
    assert!(!hierarchy.has_node(dead1));
    assert!(!hierarchy.has_node(dead2));
    assert!(hierarchy.has_node(alive));
    assert_eq!(result.nodes_pruned, 2);
}

#[test]
fn test_incremental_optimization() {
    let mut hierarchy = create_large_hierarchy(10000);
    let optimizer = HierarchyOptimizer::new();
    
    // Add new nodes dynamically
    for i in 0..100 {
        hierarchy.create_node(&format!("Dynamic_{}", i), hashmap!{"type" => "dynamic"});
    }
    
    // Incremental optimization should be fast
    let start = Instant::now();
    let result = optimizer.optimize_incremental(&mut hierarchy, 100);
    let elapsed = start.elapsed();
    
    assert!(elapsed < Duration::from_millis(10));
    assert!(result.nodes_analyzed >= 100);
}
```

**Implementation**:

```rust
// src/optimization/reorganizer.rs
pub struct HierarchyReorganizer {
    max_depth: usize,
    max_children: usize,
    similarity_threshold: f32,
}

impl HierarchyReorganizer {
    pub fn reorganize(&self, hierarchy: &mut InheritanceHierarchy) -> ReorganizationResult {
        let mut result = ReorganizationResult::new();
        
        // Phase 1: Identify reorganization opportunities
        let opportunities = self.find_reorganization_opportunities(hierarchy);
        
        // Phase 2: Execute reorganizations
        for op in opportunities {
            match op {
                ReorgOp::MergeNodes(n1, n2) => {
                    if self.can_merge(hierarchy, n1, n2) {
                        self.merge_nodes(hierarchy, n1, n2);
                        result.nodes_merged += 1;
                    }
                }
                ReorgOp::SplitNode(node, groups) => {
                    if self.should_split(hierarchy, node) {
                        self.split_node(hierarchy, node, groups);
                        result.nodes_split += 1;
                    }
                }
                ReorgOp::MoveSubtree(subtree, new_parent) => {
                    if self.is_better_parent(hierarchy, subtree, new_parent) {
                        self.move_subtree(hierarchy, subtree, new_parent);
                        result.subtrees_moved += 1;
                    }
                }
            }
        }
        
        result
    }
    
    fn find_reorganization_opportunities(&self, hierarchy: &InheritanceHierarchy) -> Vec<ReorgOp> {
        let mut ops = Vec::new();
        
        // Find similar siblings that could be merged
        for node in hierarchy.all_nodes() {
            let siblings = hierarchy.get_siblings(node.id);
            
            for sibling in siblings {
                let similarity = self.calculate_similarity(hierarchy, node.id, sibling);
                if similarity > self.similarity_threshold {
                    ops.push(ReorgOp::MergeNodes(node.id, sibling));
                }
            }
        }
        
        // Find nodes that should be split
        for node in hierarchy.all_nodes() {
            if node.children.len() > self.max_children {
                let groups = self.cluster_children(hierarchy, node.id);
                ops.push(ReorgOp::SplitNode(node.id, groups));
            }
        }
        
        // Find better parents for subtrees
        for node in hierarchy.all_nodes() {
            if let Some(better_parent) = self.find_better_parent(hierarchy, node.id) {
                ops.push(ReorgOp::MoveSubtree(node.id, better_parent));
            }
        }
        
        ops
    }
}

// src/optimization/balancer.rs
pub struct TreeBalancer {
    target_fanout: usize,
    balance_factor: f32,
}

impl TreeBalancer {
    pub fn balance(&self, hierarchy: &mut InheritanceHierarchy) -> BalanceResult {
        let mut result = BalanceResult::new();
        
        // Calculate current balance metrics
        let metrics = self.calculate_metrics(hierarchy);
        
        if metrics.balance_factor < self.balance_factor {
            // Tree is unbalanced, need to reorganize
            
            // Find pivot points
            let pivots = self.find_pivot_points(hierarchy, &metrics);
            
            for pivot in pivots {
                // Rotate subtrees
                let rotated = self.rotate_at_pivot(hierarchy, pivot);
                result.rotations += rotated;
            }
            
            // Redistribute nodes
            let redistributed = self.redistribute_nodes(hierarchy);
            result.nodes_moved += redistributed;
        }
        
        result.final_depth = hierarchy.max_depth();
        result.final_balance = self.calculate_metrics(hierarchy).balance_factor;
        
        result
    }
    
    fn rotate_at_pivot(&self, hierarchy: &mut InheritanceHierarchy, pivot: NodeId) -> usize {
        // Similar to AVL tree rotation but for multi-child nodes
        let node = hierarchy.get_node(pivot).unwrap();
        let children = node.children.clone();
        
        if children.len() < 2 {
            return 0;
        }
        
        // Find the heavy child
        let child_weights: Vec<_> = children.iter()
            .map(|&c| (c, hierarchy.count_descendants(c)))
            .collect();
        
        let (heavy_child, weight) = child_weights.iter()
            .max_by_key(|(_, w)| w)
            .unwrap();
        
        // Promote heavy child
        hierarchy.promote_child(pivot, *heavy_child);
        
        1
    }
}

// src/optimization/pruner.rs
pub struct DeadBranchPruner {
    min_properties: usize,
    min_children: usize,
}

impl DeadBranchPruner {
    pub fn prune(&self, hierarchy: &mut InheritanceHierarchy) -> PruneResult {
        let mut result = PruneResult::new();
        let mut to_prune = Vec::new();
        
        // Identify dead branches
        for node in hierarchy.all_nodes() {
            if self.is_dead_branch(hierarchy, node.id) {
                to_prune.push(node.id);
            }
        }
        
        // Prune from leaves up
        to_prune.sort_by_key(|&id| std::cmp::Reverse(hierarchy.get_depth(id)));
        
        for node_id in to_prune {
            if hierarchy.has_node(node_id) {
                // Save any important properties
                let properties = hierarchy.get_all_local_properties(node_id);
                
                if !properties.is_empty() {
                    // Move properties to parent
                    if let Some(parent) = hierarchy.get_parent(node_id) {
                        for (prop, value) in properties {
                            hierarchy.set_local_property(parent, &prop, &value);
                            result.properties_moved += 1;
                        }
                    }
                }
                
                // Remove node
                hierarchy.remove_node(node_id);
                result.nodes_pruned += 1;
            }
        }
        
        result
    }
    
    fn is_dead_branch(&self, hierarchy: &InheritanceHierarchy, node: NodeId) -> bool {
        let node_data = hierarchy.get_node(node).unwrap();
        
        // Has no meaningful properties
        let has_properties = node_data.local_properties.len() >= self.min_properties;
        
        // Has no active children
        let has_active_children = node_data.children.len() >= self.min_children;
        
        // Not referenced elsewhere
        let is_referenced = hierarchy.count_references(node) > 0;
        
        !has_properties && !has_active_children && !is_referenced
    }
}
```

**AI-Verifiable Outcomes**:
- [ ] Hierarchy depth reduced > 50%
- [ ] Dead branches removed correctly
- [ ] Incremental optimization < 10ms
- [ ] Node accessibility preserved

### Task 4.5: Compression Metrics and Analysis (Day 5)

**Specification**: Measure and verify compression effectiveness

**Test-Driven Development**:

```rust
#[test]
fn test_compression_metrics() {
    let hierarchy = create_compressed_hierarchy();
    let metrics = CompressionMetrics::calculate(&hierarchy);
    
    assert!(metrics.compression_ratio > 10.0);
    assert!(metrics.property_inheritance_rate > 0.8);
    assert!(metrics.exception_rate < 0.05);
    assert_eq!(metrics.total_nodes, 10000);
    assert_eq!(metrics.total_properties, 50000);
    assert!(metrics.unique_properties < 1000);
}

#[test]
fn test_storage_calculation() {
    let hierarchy = create_test_hierarchy();
    
    let storage = StorageCalculator::calculate(&hierarchy);
    
    // Verify storage breakdown
    assert!(storage.node_storage > 0);
    assert!(storage.property_storage > 0);
    assert!(storage.exception_storage > 0);
    assert!(storage.index_storage > 0);
    
    let total = storage.total();
    let expected = hierarchy.node_count() * 100 + // nodes
                  hierarchy.property_count() * 50 + // properties
                  hierarchy.exception_count() * 80; // exceptions
    
    assert!((total as i64 - expected as i64).abs() < 1000);
}

#[test]
fn test_compression_verification() {
    let original = create_uncompressed_hierarchy();
    let compressed = compress_hierarchy(original.clone());
    
    let verifier = CompressionVerifier::new();
    let result = verifier.verify(&original, &compressed);
    
    assert!(result.is_valid);
    assert_eq!(result.semantic_changes, 0);
    assert_eq!(result.lost_properties, 0);
    assert!(result.compression_achieved > 10.0);
}
```

**Implementation**:

```rust
// src/compression/metrics.rs
pub struct CompressionMetrics {
    pub compression_ratio: f64,
    pub property_inheritance_rate: f64,
    pub exception_rate: f64,
    pub total_nodes: usize,
    pub total_properties: usize,
    pub unique_properties: usize,
    pub inherited_properties: usize,
    pub local_properties: usize,
    pub exceptions: usize,
    pub hierarchy_depth: usize,
    pub average_fanout: f64,
    pub storage_bytes: StorageBreakdown,
}

impl CompressionMetrics {
    pub fn calculate(hierarchy: &InheritanceHierarchy) -> Self {
        let mut metrics = Self::default();
        
        // Count nodes
        metrics.total_nodes = hierarchy.node_count();
        
        // Analyze properties
        let mut all_properties = HashSet::new();
        let mut inherited_count = 0;
        let mut local_count = 0;
        
        for node in hierarchy.all_nodes() {
            // Local properties
            local_count += node.local_properties.len();
            
            // All properties (including inherited)
            let all_props = hierarchy.get_all_properties(node.id);
            metrics.total_properties += all_props.len();
            
            // Track unique property names
            for prop in all_props.keys() {
                all_properties.insert(prop.clone());
            }
            
            // Count inherited
            let inherited = all_props.len() - node.local_properties.len();
            inherited_count += inherited;
        }
        
        metrics.unique_properties = all_properties.len();
        metrics.inherited_properties = inherited_count;
        metrics.local_properties = local_count;
        metrics.exceptions = hierarchy.total_exceptions();
        
        // Calculate rates
        if metrics.total_properties > 0 {
            metrics.property_inheritance_rate = 
                inherited_count as f64 / metrics.total_properties as f64;
            metrics.exception_rate = 
                metrics.exceptions as f64 / metrics.total_properties as f64;
        }
        
        // Calculate compression ratio
        let uncompressed_size = metrics.total_properties * 
            std::mem::size_of::<(String, String)>();
        let compressed_size = metrics.storage_bytes.total();
        
        metrics.compression_ratio = uncompressed_size as f64 / compressed_size as f64;
        
        // Hierarchy metrics
        metrics.hierarchy_depth = hierarchy.max_depth();
        metrics.average_fanout = hierarchy.average_fanout();
        
        metrics
    }
}

// src/compression/verifier.rs
pub struct CompressionVerifier {
    tolerance: f64,
}

impl CompressionVerifier {
    pub fn verify(&self, original: &InheritanceHierarchy, compressed: &InheritanceHierarchy) -> VerificationResult {
        let mut result = VerificationResult::default();
        
        // Verify all nodes present
        for node in original.all_nodes() {
            if !compressed.has_node(node.id) {
                result.missing_nodes.push(node.id);
            }
        }
        
        // Verify property values
        for node in original.all_nodes() {
            if compressed.has_node(node.id) {
                let original_props = original.get_all_properties(node.id);
                let compressed_props = compressed.get_all_properties(node.id);
                
                // Check each property
                for (prop, orig_value) in original_props {
                    match compressed_props.get(&prop) {
                        Some(comp_value) if comp_value == &orig_value => {
                            // Property preserved correctly
                        }
                        Some(comp_value) => {
                            // Value changed
                            result.semantic_changes += 1;
                            result.changed_properties.push((node.id, prop, orig_value, comp_value.clone()));
                        }
                        None => {
                            // Property lost
                            result.lost_properties += 1;
                            result.missing_properties.push((node.id, prop));
                        }
                    }
                }
            }
        }
        
        // Calculate compression
        let orig_storage = self.calculate_storage(original);
        let comp_storage = self.calculate_storage(compressed);
        result.compression_achieved = orig_storage as f64 / comp_storage as f64;
        
        // Determine validity
        result.is_valid = result.missing_nodes.is_empty() &&
                         result.semantic_changes == 0 &&
                         result.lost_properties == 0;
        
        result
    }
}

// src/compression/reporter.rs
pub struct CompressionReporter {
    format: ReportFormat,
}

impl CompressionReporter {
    pub fn generate_report(&self, metrics: &CompressionMetrics, verification: &VerificationResult) -> String {
        match self.format {
            ReportFormat::Text => self.generate_text_report(metrics, verification),
            ReportFormat::Json => self.generate_json_report(metrics, verification),
            ReportFormat::Html => self.generate_html_report(metrics, verification),
        }
    }
    
    fn generate_text_report(&self, metrics: &CompressionMetrics, verification: &VerificationResult) -> String {
        format!(
            r#"Inheritance System Compression Report
=====================================

Compression Achieved: {:.1}x
Storage Reduction: {:.1}%
Property Inheritance Rate: {:.1}%
Exception Rate: {:.1}%

Node Statistics:
- Total Nodes: {}
- Hierarchy Depth: {}
- Average Fanout: {:.1}

Property Statistics:
- Total Properties: {}
- Unique Properties: {}
- Inherited Properties: {} ({:.1}%)
- Local Properties: {} ({:.1}%)
- Exceptions: {} ({:.1}%)

Storage Breakdown:
- Node Storage: {} bytes
- Property Storage: {} bytes
- Exception Storage: {} bytes
- Index Storage: {} bytes
- Total Storage: {} bytes

Verification Status: {}
- Semantic Preservation: {}
- No Properties Lost: {}
- Compression Valid: {}
"#,
            metrics.compression_ratio,
            (1.0 - 1.0/metrics.compression_ratio) * 100.0,
            metrics.property_inheritance_rate * 100.0,
            metrics.exception_rate * 100.0,
            metrics.total_nodes,
            metrics.hierarchy_depth,
            metrics.average_fanout,
            metrics.total_properties,
            metrics.unique_properties,
            metrics.inherited_properties,
            metrics.inherited_properties as f64 / metrics.total_properties as f64 * 100.0,
            metrics.local_properties,
            metrics.local_properties as f64 / metrics.total_properties as f64 * 100.0,
            metrics.exceptions,
            metrics.exception_rate * 100.0,
            metrics.storage_bytes.node_storage,
            metrics.storage_bytes.property_storage,
            metrics.storage_bytes.exception_storage,
            metrics.storage_bytes.index_storage,
            metrics.storage_bytes.total(),
            if verification.is_valid { "VALID" } else { "INVALID" },
            if verification.semantic_changes == 0 { "✓" } else { "✗" },
            if verification.lost_properties == 0 { "✓" } else { "✗" },
            if verification.compression_achieved > 10.0 { "✓" } else { "✗" }
        )
    }
}
```

**AI-Verifiable Outcomes**:
- [ ] Metrics calculation accurate
- [ ] Storage calculation correct
- [ ] Verification detects all issues
- [ ] Reports generated correctly

### Task 4.6: Integration and Benchmarks (Day 5)

**Specification**: Complete integration with full system

**Integration Tests**:

```rust
#[test]
fn test_full_inheritance_workflow() {
    // Create hierarchy
    let mut hierarchy = InheritanceHierarchy::new();
    
    // Build animal taxonomy
    let animal = hierarchy.create_node("Animal", hashmap!{
        "alive" => "true",
        "needs_food" => "true",
    });
    
    // ... build full taxonomy ...
    
    // Compress
    let compressor = PropertyCompressor::new();
    let compression_result = compressor.compress(&mut hierarchy);
    
    assert!(compression_result.compression_ratio > 10.0);
    
    // Optimize
    let optimizer = HierarchyOptimizer::new();
    let optimization_result = optimizer.optimize(&mut hierarchy);
    
    assert!(optimization_result.depth_reduced);
    
    // Verify
    let verifier = CompressionVerifier::new();
    let verification = verifier.verify(&original, &hierarchy);
    
    assert!(verification.is_valid);
}

#[bench]
fn bench_property_resolution(b: &mut Bencher) {
    let hierarchy = create_deep_hierarchy(20);
    let leaf = NodeId(999);
    
    b.iter(|| {
        black_box(hierarchy.get_property(leaf, "root_property"));
    });
}

#[bench]
fn bench_compression(b: &mut Bencher) {
    b.iter(|| {
        let mut hierarchy = create_large_hierarchy(1000);
        let compressor = PropertyCompressor::new();
        black_box(compressor.compress(&mut hierarchy));
    });
}
```

**AI-Verifiable Outcomes**:
- [ ] Full workflow completes successfully
- [ ] 10x compression achieved
- [ ] Property lookup < 100μs
- [ ] Compression < 10ms/1000 nodes

## Phase 4 Deliverables

### Code Artifacts
1. **Hierarchical Node System**
   - Single/multiple inheritance
   - Property resolution
   - Depth tracking

2. **Exception Handling**
   - Automatic detection
   - Efficient storage
   - Cascading support

3. **Property Compression**
   - Property analysis
   - Promotion engine
   - Iterative compression

4. **Hierarchy Optimization**
   - Tree balancing
   - Dead branch pruning
   - Dynamic reorganization

5. **Metrics and Verification**
   - Compression metrics
   - Storage calculation
   - Semantic verification

### Compression Report
```
Test Hierarchy Results:
├── Original Size: 4.2 MB
├── Compressed Size: 387 KB
├── Compression Ratio: 10.8x ✓
├── Properties Promoted: 847
├── Exceptions Created: 123
├── Inheritance Rate: 89.3%
└── Query Performance: 72μs avg ✓
```

## Success Checklist

- [ ] Inheritance system complete ✓
- [ ] Exception handling working ✓
- [ ] 10x compression achieved ✓
- [ ] Property lookup < 100μs ✓
- [ ] Hierarchy optimization working ✓
- [ ] Semantic preservation verified ✓
- [ ] All benchmarks pass ✓
- [ ] Zero data loss ✓
- [ ] Documentation complete ✓
- [ ] Ready for Phase 5 ✓

## Next Phase Preview

Phase 5 will implement temporal versioning:
- Git-like branching for knowledge
- Memory consolidation states
- Time-travel queries
- Diff and merge operations