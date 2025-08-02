# AI Prompt: Micro Phase 1.8 - Performance Benchmarks

You are tasked with creating comprehensive performance benchmarks to validate all performance requirements for the hierarchical node system. Your goal is to create `benches/task_4_1_hierarchy_performance.rs` using Rust's criterion crate to measure and verify performance characteristics.

## Your Task
Implement detailed performance benchmarks that measure property resolution speed, cache effectiveness, multiple inheritance performance, concurrent access scaling, and memory efficiency for the complete Task 4.1 system.

## Specific Requirements
1. Create `benches/task_4_1_hierarchy_performance.rs` using criterion for precise benchmarking
2. Benchmark property resolution across various hierarchy depths (1-50 levels)
3. Measure cache performance improvement and hit/miss scenarios
4. Benchmark multiple inheritance and DAG resolution performance
5. Test concurrent access scaling from 1-16 threads
6. Measure memory allocation patterns and efficiency
7. Validate all performance requirements with quantitative measurements

## Expected Code Structure
You must implement these exact benchmark functions:

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

// Import all the modules we've built
use llmkg::hierarchy::tree::InheritanceHierarchy;
use llmkg::hierarchy::node::{NodeId, InheritanceNode};
use llmkg::properties::value::PropertyValue;
use llmkg::properties::resolver::{PropertyResolver, ResolutionStrategy};
use llmkg::properties::cache::PropertyCache;
use llmkg::hierarchy::dag::DAGManager;

fn bench_property_resolution_depth(c: &mut Criterion) {
    let mut group = c.benchmark_group("property_resolution_depth");
    
    // Test various depths from 1 to 50 levels
    for depth in [1, 5, 10, 20, 30, 50].iter() {
        group.bench_with_input(
            BenchmarkId::new("depth", depth),
            depth,
            |b, &depth| {
                let hierarchy = create_linear_hierarchy(depth);
                let resolver = PropertyResolver::new(ResolutionStrategy::DepthFirst);
                let leaf_node = NodeId(depth as u64 - 1);
                
                b.iter(|| {
                    let result = resolver.resolve_property(
                        black_box(&hierarchy),
                        black_box(leaf_node),
                        black_box("root_property")
                    );
                    black_box(result);
                });
            },
        );
    }
    
    group.finish();
}

fn bench_cache_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_performance");
    
    let hierarchy = create_linear_hierarchy(20);
    let resolver = PropertyResolver::new(ResolutionStrategy::DepthFirst);
    let cache = PropertyCache::new(10000, Duration::from_secs(300));
    let leaf_node = NodeId(19);
    
    // Benchmark uncached resolution
    group.bench_function("uncached_resolution", |b| {
        b.iter(|| {
            let result = resolver.resolve_property(
                black_box(&hierarchy),
                black_box(leaf_node),
                black_box("root_property")
            );
            black_box(result);
        });
    });
    
    // Pre-populate cache
    let resolution = resolver.resolve_property(&hierarchy, leaf_node, "root_property");
    cache.insert(leaf_node, "root_property", resolution.value, resolution.source_node);
    
    // Benchmark cached resolution
    group.bench_function("cached_resolution", |b| {
        b.iter(|| {
            let result = cache.get(black_box(leaf_node), black_box("root_property"));
            black_box(result);
        });
    });
    
    group.finish();
}

fn bench_multiple_inheritance_resolution(c: &mut Criterion) {
    let mut group = c.benchmark_group("multiple_inheritance");
    
    // Test various numbers of parents
    for num_parents in [2, 4, 8, 16].iter() {
        group.bench_with_input(
            BenchmarkId::new("parents", num_parents),
            num_parents,
            |b, &num_parents| {
                let hierarchy = create_multiple_inheritance_hierarchy(num_parents);
                let resolver = PropertyResolver::new(ResolutionStrategy::C3Linearization);
                let dag_manager = DAGManager::new();
                let child_node = NodeId(num_parents as u64); // Child has ID after all parents
                
                b.iter(|| {
                    let mro = dag_manager.compute_mro(
                        black_box(&hierarchy),
                        black_box(child_node)
                    );
                    black_box(mro);
                });
            },
        );
    }
    
    group.finish();
}

fn bench_concurrent_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_access");
    
    let hierarchy = Arc::new(create_linear_hierarchy(10));
    let resolver = Arc::new(PropertyResolver::new(ResolutionStrategy::DepthFirst));
    let cache = Arc::new(PropertyCache::new(10000, Duration::from_secs(300)));
    
    // Test various thread counts
    for thread_count in [1, 2, 4, 8, 16].iter() {
        group.bench_with_input(
            BenchmarkId::new("threads", thread_count),
            thread_count,
            |b, &thread_count| {
                b.iter(|| {
                    let handles: Vec<_> = (0..thread_count).map(|thread_id| {
                        let hierarchy = hierarchy.clone();
                        let resolver = resolver.clone();
                        let cache = cache.clone();
                        
                        thread::spawn(move || {
                            let node = NodeId(thread_id as u64 % 10);
                            
                            // Try cache first
                            if cache.get(node, "root_property").is_none() {
                                // Resolve and cache
                                let resolution = resolver.resolve_property(&hierarchy, node, "root_property");
                                cache.insert(node, "root_property", resolution.value, resolution.source_node);
                            }
                        })
                    }).collect();
                    
                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            },
        );
    }
    
    group.finish();
}

fn bench_memory_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_allocation");
    
    // Benchmark hierarchy creation
    group.bench_function("hierarchy_creation", |b| {
        b.iter(|| {
            let hierarchy = InheritanceHierarchy::new();
            for i in 0..1000 {
                let node = hierarchy.create_node(&format!("Node{}", i));
                black_box(node);
            }
            black_box(hierarchy);
        });
    });
    
    // Benchmark node creation with relationships
    group.bench_function("relationship_creation", |b| {
        b.iter(|| {
            let hierarchy = InheritanceHierarchy::new();
            let root = hierarchy.create_node("Root").unwrap();
            
            for i in 0..100 {
                let child = hierarchy.create_child(&format!("Child{}", i), root);
                black_box(child);
            }
            black_box(hierarchy);
        });
    });
    
    // Benchmark property addition
    group.bench_function("property_addition", |b| {
        let hierarchy = InheritanceHierarchy::new();
        let node = hierarchy.create_node("TestNode").unwrap();
        
        b.iter(|| {
            if let Some(mut node_ref) = hierarchy.get_node(node) {
                for i in 0..100 {
                    node_ref.add_property(
                        format!("prop{}", i),
                        PropertyValue::String(format!("value{}", i))
                    );
                }
            }
        });
    });
    
    group.finish();
}

fn bench_dag_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("dag_operations");
    
    // Benchmark cycle detection
    group.bench_function("cycle_detection", |b| {
        let hierarchy = create_complex_dag(50);
        let dag_manager = DAGManager::new();
        
        b.iter(|| {
            let validation = dag_manager.validate_dag(black_box(&hierarchy));
            black_box(validation);
        });
    });
    
    // Benchmark C3 linearization
    group.bench_function("c3_linearization", |b| {
        let hierarchy = create_diamond_hierarchy_deep(10);
        let dag_manager = DAGManager::new();
        let leaf = NodeId(100); // Leaf node ID
        
        b.iter(|| {
            let mro = dag_manager.compute_mro(black_box(&hierarchy), black_box(leaf));
            black_box(mro);
        });
    });
    
    group.finish();
}

fn bench_real_world_scenarios(c: &mut Criterion) {
    let mut group = c.benchmark_group("real_world");
    
    // Animal taxonomy benchmark
    group.bench_function("animal_taxonomy", |b| {
        let hierarchy = create_animal_taxonomy();
        let resolver = PropertyResolver::new(ResolutionStrategy::DepthFirst);
        let golden_retriever = hierarchy.get_node_by_name("Golden Retriever").unwrap();
        
        b.iter(|| {
            let warm_blooded = resolver.resolve_property(&hierarchy, golden_retriever, "warm_blooded");
            let kingdom = resolver.resolve_property(&hierarchy, golden_retriever, "kingdom");
            let loyalty = resolver.resolve_property(&hierarchy, golden_retriever, "loyalty");
            black_box((warm_blooded, kingdom, loyalty));
        });
    });
    
    // Software class hierarchy benchmark
    group.bench_function("software_classes", |b| {
        let hierarchy = create_software_class_hierarchy();
        let resolver = PropertyResolver::new(ResolutionStrategy::C3Linearization);
        let gui_button = hierarchy.get_node_by_name("GUIButton").unwrap();
        
        b.iter(|| {
            let clickable = resolver.resolve_property(&hierarchy, gui_button, "clickable");
            let drawable = resolver.resolve_property(&hierarchy, gui_button, "drawable");
            let interactive = resolver.resolve_property(&hierarchy, gui_button, "interactive");
            black_box((clickable, drawable, interactive));
        });
    });
    
    group.finish();
}

// Helper functions to create test hierarchies

fn create_linear_hierarchy(depth: usize) -> InheritanceHierarchy {
    let hierarchy = InheritanceHierarchy::new();
    
    let root = hierarchy.create_node("Root").unwrap();
    if let Some(mut root_node) = hierarchy.get_node(root) {
        root_node.add_property("root_property".to_string(), PropertyValue::String("root_value".to_string()));
    }
    
    let mut current = root;
    for i in 1..depth {
        current = hierarchy.create_child(&format!("Node{}", i), current).unwrap();
    }
    
    hierarchy
}

fn create_multiple_inheritance_hierarchy(num_parents: usize) -> InheritanceHierarchy {
    let hierarchy = InheritanceHierarchy::new();
    
    // Create multiple parent nodes
    let mut parents = Vec::new();
    for i in 0..num_parents {
        let parent = hierarchy.create_node(&format!("Parent{}", i)).unwrap();
        if let Some(mut parent_node) = hierarchy.get_node(parent) {
            parent_node.add_property(
                format!("property{}", i),
                PropertyValue::String(format!("value{}", i))
            );
        }
        parents.push(parent);
    }
    
    // Create child with multiple inheritance
    let child = hierarchy.create_node("Child").unwrap();
    for &parent in &parents {
        hierarchy.add_parent(child, parent).unwrap();
    }
    
    hierarchy
}

fn create_complex_dag(nodes: usize) -> InheritanceHierarchy {
    let hierarchy = InheritanceHierarchy::new();
    
    // Create a complex but valid DAG
    let mut node_ids = Vec::new();
    for i in 0..nodes {
        let node = hierarchy.create_node(&format!("Node{}", i)).unwrap();
        node_ids.push(node);
        
        // Add relationships to create complex DAG without cycles
        if i > 0 {
            let parent_count = std::cmp::min(i, 3); // At most 3 parents
            for j in 0..parent_count {
                let parent_idx = (i - 1 - j) % i;
                if parent_idx < node_ids.len() {
                    hierarchy.add_parent(node, node_ids[parent_idx]).unwrap();
                }
            }
        }
    }
    
    hierarchy
}

fn create_diamond_hierarchy_deep(levels: usize) -> InheritanceHierarchy {
    let hierarchy = InheritanceHierarchy::new();
    
    let root = hierarchy.create_node("Root").unwrap();
    let mut current_level = vec![root];
    
    for level in 1..levels {
        let mut next_level = Vec::new();
        
        // Create diamond pattern at each level
        for i in 0..2 {
            let node = hierarchy.create_node(&format!("L{}N{}", level, i)).unwrap();
            
            // Connect to all nodes in previous level
            for &parent in &current_level {
                hierarchy.add_parent(node, parent).unwrap();
            }
            next_level.push(node);
        }
        current_level = next_level;
    }
    
    // Create final leaf node
    let leaf = hierarchy.create_node("Leaf").unwrap();
    for &parent in &current_level {
        hierarchy.add_parent(leaf, parent).unwrap();
    }
    
    hierarchy
}

fn create_animal_taxonomy() -> InheritanceHierarchy {
    let hierarchy = InheritanceHierarchy::new();
    
    // Create realistic animal taxonomy
    let animal = hierarchy.create_node("Animal").unwrap();
    let vertebrate = hierarchy.create_child("Vertebrate", animal).unwrap();
    let mammal = hierarchy.create_child("Mammal", vertebrate).unwrap();
    let carnivore = hierarchy.create_child("Carnivore", mammal).unwrap();
    let canidae = hierarchy.create_child("Canidae", carnivore).unwrap();
    let dog = hierarchy.create_child("Dog", canidae).unwrap();
    let golden_retriever = hierarchy.create_child("Golden Retriever", dog).unwrap();
    
    // Add properties
    if let Some(mut animal_node) = hierarchy.get_node(animal) {
        animal_node.add_property("kingdom".to_string(), PropertyValue::String("Animalia".to_string()));
        animal_node.add_property("alive".to_string(), PropertyValue::Boolean(true));
    }
    
    if let Some(mut mammal_node) = hierarchy.get_node(mammal) {
        mammal_node.add_property("warm_blooded".to_string(), PropertyValue::Boolean(true));
        mammal_node.add_property("has_fur".to_string(), PropertyValue::Boolean(true));
    }
    
    if let Some(mut dog_node) = hierarchy.get_node(dog) {
        dog_node.add_property("domesticated".to_string(), PropertyValue::Boolean(true));
        dog_node.add_property("loyalty".to_string(), PropertyValue::Float(0.95));
    }
    
    hierarchy
}

fn create_software_class_hierarchy() -> InheritanceHierarchy {
    let hierarchy = InheritanceHierarchy::new();
    
    // Create software class hierarchy with multiple inheritance
    let object = hierarchy.create_node("Object").unwrap();
    let drawable = hierarchy.create_child("Drawable", object).unwrap();
    let interactive = hierarchy.create_child("Interactive", object).unwrap();
    let widget = hierarchy.create_node("Widget").unwrap();
    let button = hierarchy.create_node("Button").unwrap();
    let gui_button = hierarchy.create_node("GUIButton").unwrap();
    
    // Set up multiple inheritance
    hierarchy.add_parent(widget, drawable).unwrap();
    hierarchy.add_parent(widget, interactive).unwrap();
    hierarchy.add_parent(button, widget).unwrap();
    hierarchy.add_parent(gui_button, button).unwrap();
    
    // Add properties
    if let Some(mut drawable_node) = hierarchy.get_node(drawable) {
        drawable_node.add_property("drawable".to_string(), PropertyValue::Boolean(true));
    }
    
    if let Some(mut interactive_node) = hierarchy.get_node(interactive) {
        interactive_node.add_property("interactive".to_string(), PropertyValue::Boolean(true));
    }
    
    if let Some(mut button_node) = hierarchy.get_node(button) {
        button_node.add_property("clickable".to_string(), PropertyValue::Boolean(true));
    }
    
    hierarchy
}

criterion_group!(
    benches,
    bench_property_resolution_depth,
    bench_cache_performance,
    bench_multiple_inheritance_resolution,
    bench_concurrent_access,
    bench_memory_allocation,
    bench_dag_operations,
    bench_real_world_scenarios
);

criterion_main!(benches);
```

## Success Criteria (You must verify these)
- [ ] Property resolution < 100μs for 20-level depth (measured and verified)
- [ ] Cache provides > 10x speedup for repeated lookups (quantified)
- [ ] Multiple inheritance resolution is deterministic and performant
- [ ] Concurrent access scales linearly up to 8 threads
- [ ] Memory allocation is bounded and predictable
- [ ] All benchmarks run successfully and produce green results
- [ ] Performance characteristics meet or exceed requirements
- [ ] Benchmarks are reproducible and stable

## File to Create
Create exactly this file: `benches/task_4_1_hierarchy_performance.rs`

## Dependencies Required
Add to Cargo.toml:
```toml
[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }

[[bench]]
name = "task_4_1_hierarchy_performance"
harness = false
```

## Performance Targets to Validate
1. **Property Resolution**: < 100μs for 20-level depth
2. **Cache Speedup**: > 10x improvement for cached vs uncached
3. **Multiple Inheritance**: < 1ms for complex DAG resolution
4. **Concurrent Scaling**: Linear scaling up to 8 threads
5. **Memory Efficiency**: < 1KB per node for basic properties

## When Complete
Respond with "MICRO PHASE 1.8 COMPLETE" and a brief summary of what you implemented, including:
- Performance benchmarks achieved for each test
- Cache speedup measurements
- Concurrent scaling characteristics
- Memory usage patterns observed
- Any performance optimizations discovered
- Confirmation that all benchmarks pass and meet requirements