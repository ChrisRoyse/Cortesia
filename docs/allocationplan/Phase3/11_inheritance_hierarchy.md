# Task 11: Inheritance Hierarchy

**Estimated Time**: 12-18 minutes  
**Dependencies**: 10_spike_pattern_processing.md  
**Stage**: Inheritance System  

## Objective
Create the foundational inheritance relationship structure to enable hierarchical concept organization with parent-child relationships and property propagation.

## Specific Requirements

### 1. Inheritance Relationship Model
- Define parent-child concept relationships
- Support multiple inheritance patterns
- Implement inheritance depth tracking
- Enable inheritance chain validation

### 2. Basic Hierarchy Operations
- Create inheritance relationships
- Query inheritance chains
- Validate hierarchy consistency
- Support hierarchy traversal

### 3. Foundation for Property Propagation
- Prepare structure for property inheritance
- Define inheritance precedence rules
- Support inheritance override mechanisms
- Enable inheritance caching foundations

## Implementation Steps

### 1. Create Inheritance Data Structures
```rust
// src/inheritance/hierarchy_types.rs
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InheritanceRelationship {
    pub id: String,
    pub parent_concept_id: String,
    pub child_concept_id: String,
    pub inheritance_type: InheritanceType,
    pub inheritance_weight: f32,
    pub created_at: DateTime<Utc>,
    pub modified_at: DateTime<Utc>,
    pub is_active: bool,
    pub inheritance_depth: u32,
    pub precedence: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InheritanceType {
    ClassInheritance,      // Traditional class-based inheritance
    PrototypeInheritance,  // Prototype-based inheritance
    MixinInheritance,      // Mixin pattern
    InterfaceInheritance,  // Interface implementation
    CompositionInheritance, // Composition-based
    Custom(String),        // Custom inheritance pattern
}

#[derive(Debug, Clone)]
pub struct InheritanceChain {
    pub child_concept_id: String,
    pub chain: Vec<InheritanceLink>,
    pub total_depth: u32,
    pub is_valid: bool,
    pub has_cycles: bool,
}

#[derive(Debug, Clone)]
pub struct InheritanceLink {
    pub parent_concept_id: String,
    pub relationship_id: String,
    pub inheritance_type: InheritanceType,
    pub depth_from_child: u32,
    pub weight: f32,
}

#[derive(Debug, Clone)]
pub struct HierarchyValidationResult {
    pub is_valid: bool,
    pub validation_errors: Vec<ValidationError>,
    pub cycle_detected: bool,
    pub max_depth_exceeded: bool,
    pub orphaned_concepts: Vec<String>,
}
```

### 2. Implement Inheritance Hierarchy Manager
```rust
// src/inheritance/hierarchy_manager.rs
pub struct InheritanceHierarchyManager {
    connection_manager: Arc<Neo4jConnectionManager>,
    hierarchy_cache: Arc<RwLock<LRUCache<String, InheritanceChain>>>,
    validation_cache: Arc<RwLock<LRUCache<String, HierarchyValidationResult>>>,
    performance_monitor: Arc<HierarchyPerformanceMonitor>,
}

impl InheritanceHierarchyManager {
    pub async fn new(
        connection_manager: Arc<Neo4jConnectionManager>,
    ) -> Result<Self, HierarchyManagerError> {
        Ok(Self {
            connection_manager,
            hierarchy_cache: Arc::new(RwLock::new(LRUCache::new(5000))),
            validation_cache: Arc::new(RwLock::new(LRUCache::new(1000))),
            performance_monitor: Arc::new(HierarchyPerformanceMonitor::new()),
        })
    }
    
    pub async fn create_inheritance_relationship(
        &self,
        parent_concept_id: &str,
        child_concept_id: &str,
        inheritance_type: InheritanceType,
        inheritance_weight: f32,
    ) -> Result<String, InheritanceCreationError> {
        let creation_start = Instant::now();
        
        // Validate that this relationship won't create cycles
        if self.would_create_cycle(parent_concept_id, child_concept_id).await? {
            return Err(InheritanceCreationError::CycleDetected {
                parent: parent_concept_id.to_string(),
                child: child_concept_id.to_string(),
            });
        }
        
        // Calculate inheritance depth
        let parent_depth = self.get_concept_depth(parent_concept_id).await?;
        let child_depth = parent_depth + 1;
        
        // Check maximum depth limit
        if child_depth > 20 { // Configurable max depth
            return Err(InheritanceCreationError::MaxDepthExceeded(child_depth));
        }
        
        // Generate relationship ID
        let relationship_id = format!("inh_{}_{}", parent_concept_id, child_concept_id);
        
        // Create inheritance relationship in Neo4j
        let session = self.connection_manager.get_session().await?;
        let query = r#"
            MATCH (parent:Concept {id: $parent_id})
            MATCH (child:Concept {id: $child_id})
            CREATE (child)-[r:INHERITS_FROM {
                relationship_id: $relationship_id,
                inheritance_type: $inheritance_type,
                inheritance_weight: $inheritance_weight,
                inheritance_depth: $inheritance_depth,
                precedence: $precedence,
                created_at: $created_at,
                modified_at: $modified_at,
                is_active: true
            }]->(parent)
            RETURN r
        "#;
        
        let precedence = self.calculate_precedence(child_concept_id, &inheritance_type).await?;
        let now = Utc::now();
        
        let parameters = hashmap![
            "parent_id".to_string() => parent_concept_id.into(),
            "child_id".to_string() => child_concept_id.into(),
            "relationship_id".to_string() => relationship_id.clone().into(),
            "inheritance_type".to_string() => format!("{:?}", inheritance_type).into(),
            "inheritance_weight".to_string() => inheritance_weight.into(),
            "inheritance_depth".to_string() => (child_depth as i64).into(),
            "precedence".to_string() => (precedence as i64).into(),
            "created_at".to_string() => now.into(),
            "modified_at".to_string() => now.into(),
        ];
        
        session.run(query, Some(parameters)).await?;
        
        // Update depth for all descendants
        self.update_descendant_depths(child_concept_id, child_depth).await?;
        
        // Invalidate relevant caches
        self.invalidate_inheritance_caches(child_concept_id).await;
        
        // Record performance metrics
        let creation_time = creation_start.elapsed();
        self.performance_monitor.record_relationship_creation_time(creation_time).await;
        
        Ok(relationship_id)
    }
    
    pub async fn get_inheritance_chain(
        &self,
        concept_id: &str,
    ) -> Result<InheritanceChain, ChainRetrievalError> {
        let retrieval_start = Instant::now();
        
        // Check cache first
        if let Some(cached_chain) = self.hierarchy_cache.read().await.get(concept_id) {
            return Ok(cached_chain.clone());
        }
        
        // Build inheritance chain from database
        let chain = self.build_inheritance_chain_from_db(concept_id).await?;
        
        // Cache the result
        self.hierarchy_cache.write().await.put(concept_id.to_string(), chain.clone());
        
        // Record performance metrics
        let retrieval_time = retrieval_start.elapsed();
        self.performance_monitor.record_chain_retrieval_time(
            chain.chain.len(),
            retrieval_time,
        ).await;
        
        Ok(chain)
    }
    
    async fn build_inheritance_chain_from_db(
        &self,
        concept_id: &str,
    ) -> Result<InheritanceChain, ChainBuildError> {
        let session = self.connection_manager.get_session().await?;
        
        // Query to get full inheritance chain
        let query = r#"
            MATCH path = (child:Concept {id: $concept_id})-[r:INHERITS_FROM*]->(ancestor:Concept)
            WITH child, relationships(path) as rels, nodes(path) as concepts
            UNWIND range(0, length(rels)-1) as i
            WITH child, rels[i] as rel, concepts[i+1] as parent_concept, i
            RETURN parent_concept.id as parent_id,
                   rel.relationship_id as relationship_id,
                   rel.inheritance_type as inheritance_type,
                   rel.inheritance_weight as weight,
                   i as depth_from_child
            ORDER BY depth_from_child
        "#;
        
        let parameters = hashmap![
            "concept_id".to_string() => concept_id.into(),
        ];
        
        let result = session.run(query, Some(parameters)).await?;
        
        let mut chain_links = Vec::new();
        let mut max_depth = 0;
        
        for record in result {
            let depth: i64 = record.get("depth_from_child")?;
            let link = InheritanceLink {
                parent_concept_id: record.get("parent_id")?,
                relationship_id: record.get("relationship_id")?,
                inheritance_type: self.parse_inheritance_type(&record.get::<String>("inheritance_type")?)?,
                depth_from_child: depth as u32,
                weight: record.get("weight")?,
            };
            
            max_depth = max_depth.max(depth as u32);
            chain_links.push(link);
        }
        
        // Check for cycles (simplified check)
        let has_cycles = self.detect_cycles_in_chain(&chain_links);
        
        Ok(InheritanceChain {
            child_concept_id: concept_id.to_string(),
            chain: chain_links,
            total_depth: max_depth,
            is_valid: !has_cycles && max_depth <= 20,
            has_cycles,
        })
    }
    
    pub async fn get_direct_children(
        &self,
        parent_concept_id: &str,
    ) -> Result<Vec<String>, ChildRetrievalError> {
        let session = self.connection_manager.get_session().await?;
        
        let query = r#"
            MATCH (parent:Concept {id: $parent_id})<-[r:INHERITS_FROM]-(child:Concept)
            WHERE r.is_active = true
            RETURN child.id as child_id
            ORDER BY r.precedence, r.created_at
        "#;
        
        let parameters = hashmap![
            "parent_id".to_string() => parent_concept_id.into(),
        ];
        
        let result = session.run(query, Some(parameters)).await?;
        
        let mut children = Vec::new();
        for record in result {
            children.push(record.get("child_id")?);
        }
        
        Ok(children)
    }
    
    pub async fn get_all_descendants(
        &self,
        ancestor_concept_id: &str,
        max_depth: Option<u32>,
    ) -> Result<Vec<String>, DescendantRetrievalError> {
        let session = self.connection_manager.get_session().await?;
        
        let depth_filter = match max_depth {
            Some(depth) => format!("WHERE length(path) <= {}", depth),
            None => String::new(),
        };
        
        let query = format!(
            r#"
            MATCH path = (ancestor:Concept {{id: $ancestor_id}})<-[r:INHERITS_FROM*]-(descendant:Concept)
            {}
            RETURN DISTINCT descendant.id as descendant_id,
                   length(path) as inheritance_depth
            ORDER BY inheritance_depth, descendant_id
            "#,
            depth_filter
        );
        
        let parameters = hashmap![
            "ancestor_id".to_string() => ancestor_concept_id.into(),
        ];
        
        let result = session.run(&query, Some(parameters)).await?;
        
        let mut descendants = Vec::new();
        for record in result {
            descendants.push(record.get("descendant_id")?);
        }
        
        Ok(descendants)
    }
    
    async fn would_create_cycle(
        &self,
        parent_concept_id: &str,
        child_concept_id: &str,
    ) -> Result<bool, CycleDetectionError> {
        // Check if parent_concept_id is already a descendant of child_concept_id
        let descendants = self.get_all_descendants(child_concept_id, Some(20)).await?;
        Ok(descendants.contains(&parent_concept_id.to_string()))
    }
    
    async fn calculate_precedence(
        &self,
        child_concept_id: &str,
        inheritance_type: &InheritanceType,
    ) -> Result<u32, PrecedenceCalculationError> {
        // Get current inheritance relationships for this child
        let session = self.connection_manager.get_session().await?;
        
        let query = r#"
            MATCH (child:Concept {id: $child_id})-[r:INHERITS_FROM]->(parent:Concept)
            WHERE r.is_active = true
            RETURN MAX(r.precedence) as max_precedence
        "#;
        
        let parameters = hashmap![
            "child_id".to_string() => child_concept_id.into(),
        ];
        
        let result = session.run(query, Some(parameters)).await?;
        
        let max_precedence: Option<i64> = result.next().await?
            .and_then(|record| record.get("max_precedence").ok());
        
        // Assign precedence based on inheritance type and current max
        let base_precedence = match inheritance_type {
            InheritanceType::ClassInheritance => 0,
            InheritanceType::InterfaceInheritance => 10,
            InheritanceType::MixinInheritance => 20,
            InheritanceType::CompositionInheritance => 30,
            InheritanceType::PrototypeInheritance => 40,
            InheritanceType::Custom(_) => 50,
        };
        
        let next_precedence = max_precedence.unwrap_or(-1) + 1;
        Ok((base_precedence + next_precedence as u32))
    }
}
```

### 3. Add Hierarchy Validation
```rust
// src/inheritance/hierarchy_validator.rs
pub struct HierarchyValidator {
    hierarchy_manager: Arc<InheritanceHierarchyManager>,
    validation_rules: ValidationRules,
}

impl HierarchyValidator {
    pub async fn validate_full_hierarchy(&self) -> Result<HierarchyValidationResult, ValidationError> {
        let validation_start = Instant::now();
        let mut validation_errors = Vec::new();
        let mut orphaned_concepts = Vec::new();
        
        // Check for cycles
        let cycle_detected = self.detect_global_cycles().await?;
        if cycle_detected {
            validation_errors.push(ValidationError::CycleDetected);
        }
        
        // Check for orphaned concepts
        orphaned_concepts = self.find_orphaned_concepts().await?;
        
        // Check maximum depth violations
        let max_depth_exceeded = self.check_max_depth_violations().await?;
        if max_depth_exceeded {
            validation_errors.push(ValidationError::MaxDepthExceeded);
        }
        
        // Check for invalid inheritance types
        let invalid_types = self.check_invalid_inheritance_types().await?;
        validation_errors.extend(invalid_types);
        
        let is_valid = validation_errors.is_empty() && !cycle_detected && !max_depth_exceeded;
        
        Ok(HierarchyValidationResult {
            is_valid,
            validation_errors,
            cycle_detected,
            max_depth_exceeded,
            orphaned_concepts,
        })
    }
    
    async fn detect_global_cycles(&self) -> Result<bool, CycleDetectionError> {
        let session = self.hierarchy_manager.connection_manager.get_session().await?;
        
        // Use graph algorithms to detect cycles
        let query = r#"
            MATCH (c:Concept)-[r:INHERITS_FROM*]->(c)
            WHERE r.is_active = true
            RETURN count(*) as cycle_count
        "#;
        
        let result = session.run(query, None).await?;
        let cycle_count: i64 = result.next().await?
            .map(|record| record.get("cycle_count").unwrap_or(0))
            .unwrap_or(0);
        
        Ok(cycle_count > 0)
    }
    
    async fn find_orphaned_concepts(&self) -> Result<Vec<String>, OrphanDetectionError> {
        let session = self.hierarchy_manager.connection_manager.get_session().await?;
        
        // Find concepts with no parents and no children
        let query = r#"
            MATCH (c:Concept)
            WHERE NOT (c)-[:INHERITS_FROM]->() AND NOT ()<-[:INHERITS_FROM]-(c)
            RETURN c.id as concept_id
        "#;
        
        let result = session.run(query, None).await?;
        
        let mut orphaned = Vec::new();
        for record in result {
            orphaned.push(record.get("concept_id")?);
        }
        
        Ok(orphaned)
    }
}
```

## Acceptance Criteria

### Functional Requirements
- [ ] Inheritance relationships can be created between concepts
- [ ] Inheritance chains can be retrieved correctly
- [ ] Cycle detection prevents invalid hierarchies
- [ ] Inheritance depth tracking works accurately
- [ ] Hierarchy validation identifies problems

### Performance Requirements
- [ ] Relationship creation time < 10ms
- [ ] Chain retrieval time < 15ms
- [ ] Cycle detection completes within 100ms
- [ ] Cache hit rate > 75% for inheritance queries

### Testing Requirements
- [ ] Unit tests for hierarchy data structures
- [ ] Integration tests for inheritance operations
- [ ] Validation tests for cycle detection
- [ ] Performance tests for large hierarchies

## Validation Steps

1. **Test inheritance relationship creation**:
   ```rust
   let relationship_id = hierarchy_manager.create_inheritance_relationship(
       "parent_concept", "child_concept", InheritanceType::ClassInheritance, 1.0
   ).await?;
   ```

2. **Test inheritance chain retrieval**:
   ```rust
   let chain = hierarchy_manager.get_inheritance_chain("child_concept").await?;
   assert!(!chain.chain.is_empty());
   ```

3. **Run hierarchy tests**:
   ```bash
   cargo test inheritance_hierarchy_tests
   ```

## Files to Create/Modify
- `src/inheritance/hierarchy_types.rs` - Data structures
- `src/inheritance/hierarchy_manager.rs` - Main hierarchy manager
- `src/inheritance/hierarchy_validator.rs` - Validation logic
- `tests/inheritance/hierarchy_tests.rs` - Test suite

## Error Handling
- Cycle detection and prevention
- Maximum depth validation
- Orphaned concept detection
- Invalid inheritance type errors
- Database consistency issues

## Success Metrics
- Inheritance relationship creation success rate: 100%
- Cycle detection accuracy: 100%
- Average hierarchy traversal time < 15ms
- Cache efficiency > 75%

## Next Task
Upon completion, proceed to **12_property_inheritance.md** to implement property inheritance mechanisms.