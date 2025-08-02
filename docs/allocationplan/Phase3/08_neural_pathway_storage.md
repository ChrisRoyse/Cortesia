# Task 08: Neural Pathway Storage

**Estimated Time**: 18-22 minutes  
**Dependencies**: 07_cortical_column_integration.md  
**Stage**: Neural Integration  

## Objective
Implement storage and management of neural pathway metadata to track concept relationships through neuromorphic pathways and enable pathway-based traversal.

## Specific Requirements

### 1. Neural Pathway Data Model
- Define pathway metadata structure
- Store pathway activation patterns
- Track pathway strength and usage frequency
- Support pathway decay and reinforcement

### 2. Pathway-Relationship Mapping
- Map graph relationships to neural pathways
- Store bidirectional pathway associations
- Track pathway creation and modification times
- Support pathway hierarchies and sub-pathways

### 3. Pathway Performance Tracking
- Monitor pathway activation frequencies
- Track pathway traversal performance
- Implement pathway strength calculations
- Support pathway pruning based on usage

## Implementation Steps

### 1. Create Neural Pathway Data Structures
```rust
// src/neural_pathways/pathway_types.rs
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralPathway {
    pub id: String,
    pub source_concept_id: String,
    pub target_concept_id: String,
    pub pathway_type: PathwayType,
    pub activation_pattern: Vec<f32>,
    pub strength: f32,
    pub usage_count: u64,
    pub last_activated: DateTime<Utc>,
    pub created_at: DateTime<Utc>,
    pub modified_at: DateTime<Utc>,
    pub decay_rate: f32,
    pub reinforcement_factor: f32,
    pub metadata: PathwayMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PathwayType {
    Association,
    Causality,
    Similarity,
    Hierarchy,
    Temporal,
    Spatial,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathwayMetadata {
    pub cortical_column_source: Option<ColumnId>,
    pub cortical_column_target: Option<ColumnId>,
    pub ttfs_correlation: Option<f32>,
    pub semantic_weight: f32,
    pub neural_efficiency: f32,
    pub pathway_tags: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct PathwayActivationResult {
    pub pathway_id: String,
    pub activation_strength: f32,
    pub traversal_time: Duration,
    pub source_activation: f32,
    pub target_activation: f32,
}
```

### 2. Implement Pathway Storage Service
```rust
// src/neural_pathways/pathway_storage.rs
pub struct NeuralPathwayStorage {
    connection_manager: Arc<Neo4jConnectionManager>,
    pathway_cache: Arc<RwLock<LRUCache<String, NeuralPathway>>>,
    activation_tracker: Arc<ActivationTracker>,
    performance_monitor: Arc<PathwayPerformanceMonitor>,
}

impl NeuralPathwayStorage {
    pub async fn new(
        connection_manager: Arc<Neo4jConnectionManager>,
    ) -> Result<Self, PathwayStorageError> {
        Ok(Self {
            connection_manager,
            pathway_cache: Arc::new(RwLock::new(LRUCache::new(10000))),
            activation_tracker: Arc::new(ActivationTracker::new()),
            performance_monitor: Arc::new(PathwayPerformanceMonitor::new()),
        })
    }
    
    pub async fn create_pathway(
        &self,
        source_concept_id: &str,
        target_concept_id: &str,
        pathway_type: PathwayType,
        initial_strength: f32,
    ) -> Result<String, PathwayCreationError> {
        let creation_start = Instant::now();
        
        // Generate pathway ID
        let pathway_id = self.generate_pathway_id(source_concept_id, target_concept_id);
        
        // Create pathway object
        let pathway = NeuralPathway {
            id: pathway_id.clone(),
            source_concept_id: source_concept_id.to_string(),
            target_concept_id: target_concept_id.to_string(),
            pathway_type,
            activation_pattern: vec![0.0; 128], // Initialize with zeros
            strength: initial_strength,
            usage_count: 0,
            last_activated: Utc::now(),
            created_at: Utc::now(),
            modified_at: Utc::now(),
            decay_rate: 0.01, // Default decay rate
            reinforcement_factor: 1.1, // Default reinforcement
            metadata: PathwayMetadata {
                cortical_column_source: None,
                cortical_column_target: None,
                ttfs_correlation: None,
                semantic_weight: initial_strength,
                neural_efficiency: 1.0,
                pathway_tags: Vec::new(),
            },
        };
        
        // Store in Neo4j
        let session = self.connection_manager.get_session().await?;
        let query = r#"
            MATCH (source:Concept {id: $source_id})
            MATCH (target:Concept {id: $target_id})
            CREATE (source)-[p:NEURAL_PATHWAY {
                pathway_id: $pathway_id,
                pathway_type: $pathway_type,
                strength: $strength,
                usage_count: $usage_count,
                last_activated: $last_activated,
                created_at: $created_at,
                decay_rate: $decay_rate,
                reinforcement_factor: $reinforcement_factor,
                activation_pattern: $activation_pattern,
                semantic_weight: $semantic_weight,
                neural_efficiency: $neural_efficiency
            }]->(target)
            RETURN p
        "#;
        
        let parameters = hashmap![
            "source_id".to_string() => source_concept_id.into(),
            "target_id".to_string() => target_concept_id.into(),
            "pathway_id".to_string() => pathway_id.clone().into(),
            "pathway_type".to_string() => format!("{:?}", pathway.pathway_type).into(),
            "strength".to_string() => pathway.strength.into(),
            "usage_count".to_string() => (pathway.usage_count as i64).into(),
            "last_activated".to_string() => pathway.last_activated.into(),
            "created_at".to_string() => pathway.created_at.into(),
            "decay_rate".to_string() => pathway.decay_rate.into(),
            "reinforcement_factor".to_string() => pathway.reinforcement_factor.into(),
            "activation_pattern".to_string() => pathway.activation_pattern.into(),
            "semantic_weight".to_string() => pathway.metadata.semantic_weight.into(),
            "neural_efficiency".to_string() => pathway.metadata.neural_efficiency.into(),
        ];
        
        session.run(query, Some(parameters)).await?;
        
        // Cache the pathway
        self.pathway_cache.write().await.put(pathway_id.clone(), pathway);
        
        // Record performance metrics
        let creation_time = creation_start.elapsed();
        self.performance_monitor.record_pathway_creation_time(creation_time).await;
        
        Ok(pathway_id)
    }
    
    pub async fn activate_pathway(
        &self,
        pathway_id: &str,
        activation_strength: f32,
    ) -> Result<PathwayActivationResult, PathwayActivationError> {
        let activation_start = Instant::now();
        
        // Retrieve pathway from cache or database
        let mut pathway = match self.pathway_cache.read().await.get(pathway_id) {
            Some(cached_pathway) => cached_pathway.clone(),
            None => self.load_pathway_from_db(pathway_id).await?,
        };
        
        // Update pathway activation
        pathway.last_activated = Utc::now();
        pathway.usage_count += 1;
        
        // Apply reinforcement
        if activation_strength > 0.5 {
            pathway.strength = (pathway.strength * pathway.reinforcement_factor).min(1.0);
        }
        
        // Update activation pattern (simple moving average)
        let pattern_index = (pathway.usage_count % 128) as usize;
        pathway.activation_pattern[pattern_index] = activation_strength;
        
        // Update in database
        self.update_pathway_in_db(&pathway).await?;
        
        // Update cache
        self.pathway_cache.write().await.put(pathway_id.to_string(), pathway.clone());
        
        // Record activation
        let traversal_time = activation_start.elapsed();
        self.activation_tracker.record_activation(
            pathway_id,
            activation_strength,
            traversal_time,
        ).await;
        
        Ok(PathwayActivationResult {
            pathway_id: pathway_id.to_string(),
            activation_strength,
            traversal_time,
            source_activation: activation_strength * 0.9, // Simulated
            target_activation: activation_strength * 0.8, // Simulated
        })
    }
    
    pub async fn get_pathways_from_concept(
        &self,
        concept_id: &str,
        pathway_type: Option<PathwayType>,
        min_strength: f32,
        limit: usize,
    ) -> Result<Vec<NeuralPathway>, PathwayQueryError> {
        let session = self.connection_manager.get_session().await?;
        
        let pathway_type_filter = match pathway_type {
            Some(ptype) => format!("AND p.pathway_type = '{:?}'", ptype),
            None => String::new(),
        };
        
        let query = format!(
            r#"
            MATCH (source:Concept {{id: $concept_id}})-[p:NEURAL_PATHWAY]->(target:Concept)
            WHERE p.strength >= $min_strength {}
            RETURN p, target.id as target_id
            ORDER BY p.strength DESC, p.last_activated DESC
            LIMIT $limit
            "#,
            pathway_type_filter
        );
        
        let parameters = hashmap![
            "concept_id".to_string() => concept_id.into(),
            "min_strength".to_string() => min_strength.into(),
            "limit".to_string() => (limit as i64).into(),
        ];
        
        let result = session.run(&query, Some(parameters)).await?;
        
        let mut pathways = Vec::new();
        for record in result {
            let pathway = NeuralPathway::from_neo4j_record(record)?;
            pathways.push(pathway);
        }
        
        Ok(pathways)
    }
    
    async fn apply_decay(&self) -> Result<usize, PathwayDecayError> {
        let decay_start = Instant::now();
        let cutoff_time = Utc::now() - Duration::hours(24); // Decay pathways not used in 24h
        
        let session = self.connection_manager.get_session().await?;
        let query = r#"
            MATCH ()-[p:NEURAL_PATHWAY]->()
            WHERE p.last_activated < $cutoff_time
            SET p.strength = p.strength * (1.0 - p.decay_rate)
            WITH p
            WHERE p.strength < 0.1
            DELETE p
            RETURN count(p) as decayed_count
        "#;
        
        let parameters = hashmap![
            "cutoff_time".to_string() => cutoff_time.into(),
        ];
        
        let result = session.run(query, Some(parameters)).await?;
        let decayed_count = result.next().await?
            .map(|record| record.get::<i64>("decayed_count").unwrap_or(0) as usize)
            .unwrap_or(0);
        
        let decay_time = decay_start.elapsed();
        self.performance_monitor.record_decay_operation_time(decay_time).await;
        
        Ok(decayed_count)
    }
}
```

### 3. Implement Pathway Performance Monitoring
```rust
// src/neural_pathways/pathway_performance.rs
pub struct PathwayPerformanceMonitor {
    activation_times: Arc<RwLock<Vec<Duration>>>,
    creation_times: Arc<RwLock<Vec<Duration>>>,
    pathway_stats: Arc<RwLock<HashMap<String, PathwayStats>>>,
    decay_stats: Arc<RwLock<DecayStats>>,
}

impl PathwayPerformanceMonitor {
    pub async fn record_pathway_activation(
        &self,
        pathway_id: &str,
        activation_time: Duration,
        strength: f32,
    ) {
        // Record timing
        self.activation_times.write().await.push(activation_time);
        
        // Update pathway-specific stats
        let mut stats = self.pathway_stats.write().await;
        let pathway_stat = stats.entry(pathway_id.to_string()).or_insert(PathwayStats::new());
        pathway_stat.total_activations += 1;
        pathway_stat.total_activation_time += activation_time;
        pathway_stat.last_strength = strength;
        pathway_stat.strength_history.push(strength);
        
        // Keep only last 100 strength values
        if pathway_stat.strength_history.len() > 100 {
            pathway_stat.strength_history.remove(0);
        }
    }
    
    pub async fn get_pathway_efficiency(&self, pathway_id: &str) -> Option<f32> {
        let stats = self.pathway_stats.read().await;
        stats.get(pathway_id).map(|stat| {
            if stat.total_activations == 0 {
                return 0.0;
            }
            
            let avg_activation_time = stat.total_activation_time.as_millis() as f32 
                / stat.total_activations as f32;
            let avg_strength = stat.strength_history.iter().sum::<f32>() 
                / stat.strength_history.len() as f32;
            
            // Efficiency = strength / (time_factor + 1)
            avg_strength / (avg_activation_time / 10.0 + 1.0)
        })
    }
    
    pub async fn get_system_performance_summary(&self) -> PerformanceSummary {
        let activation_times = self.activation_times.read().await;
        let pathway_stats = self.pathway_stats.read().await;
        
        let total_pathways = pathway_stats.len();
        let total_activations: u64 = pathway_stats.values()
            .map(|stat| stat.total_activations)
            .sum();
        
        let avg_activation_time = if activation_times.is_empty() {
            Duration::from_millis(0)
        } else {
            activation_times.iter().sum::<Duration>() / activation_times.len() as u32
        };
        
        PerformanceSummary {
            total_pathways,
            total_activations,
            average_activation_time: avg_activation_time,
            cache_hit_rate: self.calculate_cache_hit_rate().await,
        }
    }
}
```

## Acceptance Criteria

### Functional Requirements
- [ ] Neural pathway creation and storage works correctly
- [ ] Pathway activation updates strength and usage statistics
- [ ] Pathway decay removes unused pathways automatically
- [ ] Pathway queries return relevant results with performance data
- [ ] Pathway metadata is accurately stored and retrieved

### Performance Requirements
- [ ] Pathway creation time < 5ms
- [ ] Pathway activation time < 2ms
- [ ] Pathway queries complete within 15ms
- [ ] Cache hit rate > 85% for frequently accessed pathways

### Testing Requirements
- [ ] Unit tests for pathway data structures
- [ ] Integration tests for pathway storage operations
- [ ] Performance tests for activation and decay
- [ ] Concurrency tests for pathway access

## Validation Steps

1. **Test pathway creation**:
   ```rust
   let pathway_id = pathway_storage.create_pathway(
       "concept_1", "concept_2", PathwayType::Association, 0.8
   ).await?;
   ```

2. **Test pathway activation**:
   ```rust
   let result = pathway_storage.activate_pathway(&pathway_id, 0.9).await?;
   assert!(result.activation_strength > 0.0);
   ```

3. **Run pathway tests**:
   ```bash
   cargo test neural_pathway_tests
   ```

## Files to Create/Modify
- `src/neural_pathways/pathway_types.rs` - Pathway data structures
- `src/neural_pathways/pathway_storage.rs` - Storage service
- `src/neural_pathways/pathway_performance.rs` - Performance monitoring
- `tests/neural_pathways/pathway_tests.rs` - Test suite

## Error Handling
- Pathway creation failures
- Database connectivity issues
- Cache inconsistency errors
- Performance degradation detection
- Pathway corruption detection

## Success Metrics
- Pathway creation success rate: 100%
- Average pathway activation time < 2ms
- Pathway decay efficiency: Removes 90%+ of unused pathways
- Cache efficiency > 85%

## Next Task
Upon completion, proceed to **09_allocation_guided_placement.md** to implement allocation engine guided node placement.