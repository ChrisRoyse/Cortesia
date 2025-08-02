# Task 12: Property Inheritance Implementation

**Estimated Time**: 25-30 minutes  
**Dependencies**: 11_inheritance_hierarchy.md  
**Stage**: Inheritance System  

## Objective
Implement the property inheritance mechanism that allows concepts to inherit properties from parent concepts in the hierarchy, with support for overrides, exceptions, and inheritance chains.

## Specific Requirements

### 1. Inheritance Mechanisms
- Property propagation down inheritance chains
- Override and exception handling for property values
- Priority-based inheritance resolution
- Lazy vs eager inheritance evaluation

### 2. Inheritance Cache System
- Cache resolved inheritance chains for performance
- Invalidate caches on structural changes
- Support for partial cache updates
- Memory-efficient cache storage

### 3. Resolution Algorithms
- Depth-first property resolution
- Multiple inheritance handling
- Conflict resolution between multiple parents
- Default value propagation

## Implementation Steps

### 1. Create Property Inheritance Engine
```rust
// src/inheritance/property_inheritance_engine.rs
pub struct PropertyInheritanceEngine {
    connection_manager: Arc<Neo4jConnectionManager>,
    inheritance_cache: Arc<RwLock<InheritanceCache>>,
    resolution_cache: Arc<RwLock<LRUCache<String, ResolvedProperties>>>,
    performance_monitor: Arc<InheritancePerformanceMonitor>,
    config: InheritanceConfig,
}

impl PropertyInheritanceEngine {
    pub async fn new(
        connection_manager: Arc<Neo4jConnectionManager>,
        config: InheritanceConfig,
    ) -> Result<Self, InheritanceError> {
        Ok(Self {
            connection_manager,
            inheritance_cache: Arc::new(RwLock::new(InheritanceCache::new())),
            resolution_cache: Arc::new(RwLock::new(LRUCache::new(10000))),
            performance_monitor: Arc::new(InheritancePerformanceMonitor::new()),
            config,
        })
    }
    
    pub async fn resolve_properties(
        &self,
        concept_id: &str,
        include_inherited: bool,
    ) -> Result<ResolvedProperties, InheritanceError> {
        let resolution_start = Instant::now();
        
        // Check resolution cache first
        let cache_key = format!("{}:{}", concept_id, include_inherited);
        if let Some(cached_properties) = self.resolution_cache.read().await.get(&cache_key) {
            return Ok(cached_properties.clone());
        }
        
        // Get direct properties
        let direct_properties = self.get_direct_properties(concept_id).await?;
        
        let resolved_properties = if include_inherited {
            // Get inheritance chain
            let inheritance_chain = self.get_inheritance_chain(concept_id).await?;
            
            // Resolve inherited properties
            let inherited_properties = self.resolve_inherited_properties(
                &inheritance_chain,
                &direct_properties,
            ).await?;
            
            // Merge direct and inherited properties
            self.merge_properties(direct_properties, inherited_properties).await?
        } else {
            ResolvedProperties::from_direct(direct_properties)
        };
        
        // Cache the resolved properties
        self.resolution_cache.write().await.put(cache_key, resolved_properties.clone());
        
        let resolution_time = resolution_start.elapsed();
        self.performance_monitor.record_resolution_time(resolution_time).await;
        
        Ok(resolved_properties)
    }
    
    async fn get_inheritance_chain(
        &self,
        concept_id: &str,
    ) -> Result<InheritanceChain, InheritanceError> {
        // Check inheritance cache first
        if let Some(cached_chain) = self.inheritance_cache.read().await.get_chain(concept_id) {
            return Ok(cached_chain);
        }
        
        let session = self.connection_manager.get_session().await?;
        
        let query = r#"
            MATCH path = (child:Concept {id: $concept_id})-[:INHERITS_FROM*]->(ancestor:Concept)
            WITH path, length(path) as depth
            ORDER BY depth
            RETURN 
                nodes(path) as inheritance_nodes,
                relationships(path) as inheritance_relationships,
                depth
        "#;
        
        let parameters = hashmap!["concept_id".to_string() => concept_id.into()];
        
        let result = session.run(query, Some(parameters)).await?;
        
        let mut inheritance_chain = InheritanceChain::new(concept_id.to_string());
        
        for record in result {
            let nodes: Vec<Value> = record.get("inheritance_nodes")?;
            let relationships: Vec<Value> = record.get("inheritance_relationships")?;
            let depth: i32 = record.get("depth")?;
            
            // Process each level of inheritance
            for (i, node) in nodes.iter().enumerate() {
                if i == 0 { continue; } // Skip the child node itself
                
                let ancestor_id = node.get("id").unwrap().as_str().unwrap();
                let relationship = &relationships[i - 1];
                
                let inheritance_metadata = InheritanceMetadata {
                    depth: i as i32,
                    strength: relationship.get("strength").unwrap().as_f64().unwrap() as f32,
                    property_mask: relationship.get("property_mask")
                        .map(|v| v.as_list().unwrap().iter()
                            .map(|s| s.as_str().unwrap().to_string())
                            .collect())
                        .unwrap_or_default(),
                    established_at: relationship.get("established_at").unwrap().try_into()?,
                    last_validated: relationship.get("last_validated").unwrap().try_into()?,
                };
                
                inheritance_chain.add_ancestor(ancestor_id.to_string(), inheritance_metadata);
            }
        }
        
        // Cache the inheritance chain
        self.inheritance_cache.write().await.store_chain(concept_id.to_string(), inheritance_chain.clone());
        
        Ok(inheritance_chain)
    }
    
    async fn resolve_inherited_properties(
        &self,
        inheritance_chain: &InheritanceChain,
        direct_properties: &[PropertyNode],
    ) -> Result<Vec<InheritedProperty>, InheritanceError> {
        let mut inherited_properties = Vec::new();
        let direct_property_names: HashSet<String> = direct_properties
            .iter()
            .map(|p| p.name.clone())
            .collect();
        
        // Process inheritance chain in order (closest ancestor first)
        for ancestor in inheritance_chain.ancestors() {
            let ancestor_properties = self.get_direct_properties(&ancestor.concept_id).await?;
            
            for property in ancestor_properties {
                // Skip if property is already defined directly or by closer ancestor
                if direct_property_names.contains(&property.name) ||
                   inherited_properties.iter().any(|ip: &InheritedProperty| ip.property.name == property.name) {
                    continue;
                }
                
                // Check if property is inheritable
                if !property.is_inheritable {
                    continue;
                }
                
                // Check property mask filter
                if !ancestor.metadata.property_mask.is_empty() &&
                   !ancestor.metadata.property_mask.contains(&property.name) {
                    continue;
                }
                
                // Check for exceptions that might override this inheritance
                let exceptions = self.get_property_exceptions(
                    &inheritance_chain.child_id,
                    &property.name,
                ).await?;
                
                let final_value = if let Some(exception) = exceptions.first() {
                    exception.exception_value.clone()
                } else {
                    property.value.clone()
                };
                
                inherited_properties.push(InheritedProperty {
                    property: PropertyNode {
                        value: final_value,
                        ..property
                    },
                    source_concept_id: ancestor.concept_id.clone(),
                    inheritance_depth: ancestor.metadata.depth,
                    inheritance_strength: ancestor.metadata.strength,
                    has_exception: exceptions.first().is_some(),
                    exception_reason: exceptions.first().map(|e| e.exception_reason.clone()),
                });
            }
        }
        
        // Sort by inheritance priority and depth
        inherited_properties.sort_by(|a, b| {
            a.property.inheritance_priority
                .cmp(&b.property.inheritance_priority)
                .then(a.inheritance_depth.cmp(&b.inheritance_depth))
        });
        
        Ok(inherited_properties)
    }
    
    async fn merge_properties(
        &self,
        direct_properties: Vec<PropertyNode>,
        inherited_properties: Vec<InheritedProperty>,
    ) -> Result<ResolvedProperties, InheritanceError> {
        let mut resolved = ResolvedProperties::new();
        
        // Add direct properties (highest priority)
        for property in direct_properties {
            resolved.add_direct_property(property);
        }
        
        // Add inherited properties (only if not overridden)
        for inherited in inherited_properties {
            if !resolved.has_property(&inherited.property.name) {
                resolved.add_inherited_property(inherited);
            }
        }
        
        // Calculate inheritance statistics
        resolved.calculate_statistics();
        
        Ok(resolved)
    }
}
```

### 2. Implement Property Exception Handling
```rust
// src/inheritance/property_exceptions.rs
pub struct PropertyExceptionHandler {
    connection_manager: Arc<Neo4jConnectionManager>,
    exception_cache: Arc<RwLock<HashMap<String, Vec<ExceptionNode>>>>,
}

impl PropertyExceptionHandler {
    pub async fn get_property_exceptions(
        &self,
        concept_id: &str,
        property_name: &str,
    ) -> Result<Vec<ExceptionNode>, ExceptionError> {
        let cache_key = format!("{}:{}", concept_id, property_name);
        
        // Check cache first
        if let Some(cached_exceptions) = self.exception_cache.read().await.get(&cache_key) {
            return Ok(cached_exceptions.clone());
        }
        
        let session = self.connection_manager.get_session().await?;
        
        let query = r#"
            MATCH (c:Concept {id: $concept_id})-[:HAS_EXCEPTION]->(e:Exception)
            WHERE e.property_name = $property_name
            RETURN e
            ORDER BY e.precedence DESC, e.confidence DESC
        "#;
        
        let parameters = hashmap![
            "concept_id".to_string() => concept_id.into(),
            "property_name".to_string() => property_name.into()
        ];
        
        let result = session.run(query, Some(parameters)).await?;
        
        let mut exceptions = Vec::new();
        for record in result {
            let exception_data: Value = record.get("e")?;
            let exception = ExceptionNode::from_neo4j_value(exception_data)?;
            exceptions.push(exception);
        }
        
        // Cache the exceptions
        self.exception_cache.write().await.insert(cache_key, exceptions.clone());
        
        Ok(exceptions)
    }
    
    pub async fn create_property_exception(
        &self,
        concept_id: &str,
        property_name: &str,
        original_value: PropertyValue,
        exception_value: PropertyValue,
        reason: &str,
        confidence: f32,
    ) -> Result<String, ExceptionError> {
        let exception = ExceptionNodeBuilder::new()
            .property_name(property_name)
            .original_value(original_value)
            .exception_value(exception_value)
            .exception_reason(reason)
            .confidence(confidence)
            .build()?;
        
        let exception_id = self.create_exception_node(&exception).await?;
        
        // Create relationship to concept
        self.create_exception_relationship(concept_id, &exception_id).await?;
        
        // Invalidate related caches
        self.invalidate_exception_cache(concept_id, property_name).await;
        
        Ok(exception_id)
    }
}
```

### 3. Implement Inheritance Cache Management
```rust
// src/inheritance/inheritance_cache.rs
pub struct InheritanceCache {
    inheritance_chains: HashMap<String, InheritanceChain>,
    property_resolutions: HashMap<String, CachedResolution>,
    last_updated: HashMap<String, DateTime<Utc>>,
    cache_stats: CacheStatistics,
}

impl InheritanceCache {
    pub fn new() -> Self {
        Self {
            inheritance_chains: HashMap::new(),
            property_resolutions: HashMap::new(),
            last_updated: HashMap::new(),
            cache_stats: CacheStatistics::new(),
        }
    }
    
    pub fn get_chain(&self, concept_id: &str) -> Option<InheritanceChain> {
        self.cache_stats.record_access();
        
        if let Some(chain) = self.inheritance_chains.get(concept_id) {
            // Check if cache entry is still valid
            if let Some(last_update) = self.last_updated.get(concept_id) {
                if Utc::now().signed_duration_since(*last_update).num_minutes() < 30 {
                    self.cache_stats.record_hit();
                    return Some(chain.clone());
                }
            }
        }
        
        self.cache_stats.record_miss();
        None
    }
    
    pub fn store_chain(&mut self, concept_id: String, chain: InheritanceChain) {
        self.inheritance_chains.insert(concept_id.clone(), chain);
        self.last_updated.insert(concept_id, Utc::now());
        self.cache_stats.record_store();
    }
    
    pub fn invalidate_concept(&mut self, concept_id: &str) {
        // Invalidate the concept itself
        self.inheritance_chains.remove(concept_id);
        self.property_resolutions.remove(concept_id);
        self.last_updated.remove(concept_id);
        
        // Invalidate all descendants (they might inherit from this concept)
        let descendants = self.find_descendants(concept_id);
        for descendant in descendants {
            self.inheritance_chains.remove(&descendant);
            self.property_resolutions.remove(&descendant);
            self.last_updated.remove(&descendant);
        }
        
        self.cache_stats.record_invalidation();
    }
    
    pub fn invalidate_property(&mut self, concept_id: &str, property_name: &str) {
        // Remove property-specific resolutions
        let property_key = format!("{}:{}", concept_id, property_name);
        self.property_resolutions.remove(&property_key);
        
        // Invalidate descendants that might inherit this property
        let descendants = self.find_descendants(concept_id);
        for descendant in descendants {
            let descendant_key = format!("{}:{}", descendant, property_name);
            self.property_resolutions.remove(&descendant_key);
        }
    }
    
    fn find_descendants(&self, concept_id: &str) -> Vec<String> {
        // Find all concepts that have this concept in their inheritance chain
        self.inheritance_chains
            .iter()
            .filter_map(|(child_id, chain)| {
                if chain.ancestors().iter().any(|a| a.concept_id == concept_id) {
                    Some(child_id.clone())
                } else {
                    None
                }
            })
            .collect()
    }
    
    pub fn get_statistics(&self) -> &CacheStatistics {
        &self.cache_stats
    }
}
```

### 4. Add Property Resolution Performance Optimization
```rust
// src/inheritance/property_resolution_optimizer.rs
pub struct PropertyResolutionOptimizer {
    batch_resolver: BatchPropertyResolver,
    precomputation_engine: PrecomputationEngine,
    materialized_views: MaterializedInheritanceViews,
}

impl PropertyResolutionOptimizer {
    pub async fn optimize_resolution_for_concept(
        &self,
        concept_id: &str,
    ) -> Result<OptimizationResult, OptimizationError> {
        // Analyze inheritance patterns
        let inheritance_analysis = self.analyze_inheritance_patterns(concept_id).await?;
        
        // Determine if materialization would be beneficial
        if inheritance_analysis.should_materialize() {
            self.materialize_inheritance_chain(concept_id).await?;
        }
        
        // Precompute frequently accessed property combinations
        if inheritance_analysis.has_frequent_access_patterns() {
            self.precompute_property_combinations(concept_id).await?;
        }
        
        Ok(OptimizationResult {
            optimization_applied: true,
            expected_performance_improvement: inheritance_analysis.estimated_improvement(),
            materialization_created: inheritance_analysis.should_materialize(),
        })
    }
    
    pub async fn batch_resolve_properties(
        &self,
        concept_ids: &[String],
        property_names: &[String],
    ) -> Result<BatchResolutionResult, ResolutionError> {
        // Group concepts by inheritance patterns for efficient batch processing
        let concept_groups = self.group_concepts_by_inheritance_pattern(concept_ids).await?;
        
        let mut batch_results = HashMap::new();
        
        for group in concept_groups {
            // Resolve inheritance chains for the group
            let group_chains = self.batch_resolve_inheritance_chains(&group.concept_ids).await?;
            
            // Resolve properties for the group
            let group_properties = self.batch_resolve_group_properties(
                &group_chains,
                property_names,
            ).await?;
            
            for (concept_id, properties) in group_properties {
                batch_results.insert(concept_id, properties);
            }
        }
        
        Ok(BatchResolutionResult {
            resolved_properties: batch_results,
            processing_time: Instant::now().elapsed(),
            cache_utilization: self.calculate_cache_utilization(),
        })
    }
}
```

## Acceptance Criteria

### Functional Requirements
- [ ] Property inheritance works correctly through multiple levels
- [ ] Overrides and exceptions are properly handled
- [ ] Priority-based inheritance resolution functions correctly
- [ ] Cache invalidation maintains data consistency
- [ ] Batch property resolution works efficiently

### Performance Requirements
- [ ] Single property resolution < 5ms
- [ ] Inheritance chain resolution < 10ms for depth 10
- [ ] Cache hit rate > 80% for frequent property accesses
- [ ] Batch operations handle 100+ concepts efficiently

### Testing Requirements
- [ ] Unit tests for inheritance algorithms
- [ ] Integration tests with complex inheritance hierarchies
- [ ] Performance tests for large inheritance chains
- [ ] Cache consistency tests

## Validation Steps

1. **Test property inheritance**:
   ```rust
   let inheritance_engine = PropertyInheritanceEngine::new(connection_manager, config).await?;
   let properties = inheritance_engine.resolve_properties("concept_id", true).await?;
   ```

2. **Test exception handling**:
   ```rust
   let exception_id = exception_handler.create_property_exception(
       "concept_id", "property_name", original_value, exception_value, "reason", 0.9
   ).await?;
   ```

3. **Run inheritance tests**:
   ```bash
   cargo test property_inheritance_tests
   ```

## Files to Create/Modify
- `src/inheritance/property_inheritance_engine.rs` - Main inheritance engine
- `src/inheritance/property_exceptions.rs` - Exception handling
- `src/inheritance/inheritance_cache.rs` - Cache management
- `src/inheritance/property_resolution_optimizer.rs` - Performance optimization
- `tests/inheritance/property_inheritance_tests.rs` - Test suite

## Error Handling
- Circular inheritance detection
- Invalid property override attempts
- Cache corruption recovery
- Performance degradation detection
- Exception conflict resolution

## Success Metrics
- Property inheritance accuracy: 100%
- Cache efficiency > 80%
- Performance requirements met
- Exception handling correctness: 100%

## Next Task
Upon completion, proceed to **13_inheritance_cache.md** to build the inheritance chain caching system.