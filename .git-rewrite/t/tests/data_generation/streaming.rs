//! Streaming Data Generation
//! 
//! Provides temporal data generation with streaming updates and batch processing capabilities.

use crate::infrastructure::deterministic_rng::DeterministicRng;
use crate::data_generation::{
    TestGraph, TestEntity, TestEdge, GraphProperties,
    EmbeddingTestSet, TraversalQuery, RagQuery, SimilarityQuery,
    ComprehensiveDataGenerator, GenerationParameters
};
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// Streaming data generator for temporal scenarios
pub struct StreamingDataGenerator {
    rng: DeterministicRng,
    base_generator: ComprehensiveDataGenerator,
    stream_state: StreamingState,
    temporal_config: TemporalConfig,
}

/// Configuration for temporal data generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalConfig {
    pub time_steps: u64,
    pub entities_per_step: u64,
    pub edges_per_step: u64,
    pub update_probability: f64,
    pub deletion_probability: f64,
    pub batch_size: usize,
    pub temporal_locality: f64, // How much new data relates to recent data
}

/// Current state of the streaming generation
#[derive(Debug, Clone)]
pub struct StreamingState {
    pub current_timestamp: u64,
    pub active_entities: HashMap<u32, u64>, // entity_id -> creation_timestamp
    pub active_edges: HashMap<(u32, u32), u64>, // edge -> creation_timestamp
    pub entity_lifecycle: EntityLifecycleTracker,
    pub temporal_patterns: TemporalPatterns,
}

/// Tracks entity creation, updates, and deletion patterns
#[derive(Debug, Clone)]
pub struct EntityLifecycleTracker {
    pub creation_times: HashMap<u32, u64>,
    pub update_times: HashMap<u32, Vec<u64>>,
    pub deletion_times: HashMap<u32, u64>,
    pub activity_patterns: HashMap<u32, ActivityPattern>,
}

/// Temporal patterns in the data
#[derive(Debug, Clone)]
pub struct TemporalPatterns {
    pub growth_rate: f64,
    pub seasonal_patterns: Vec<SeasonalPattern>,
    pub burst_events: Vec<BurstEvent>,
    pub trend_direction: TrendDirection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalPattern {
    pub period: u64,
    pub amplitude: f64,
    pub phase_offset: f64,
    pub pattern_type: SeasonalType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SeasonalType {
    Sinusoidal,
    Square,
    Sawtooth,
    Custom(Vec<f64>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BurstEvent {
    pub timestamp: u64,
    pub duration: u64,
    pub intensity_multiplier: f64,
    pub affected_entities: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Growing,
    Declining,
    Stable,
    Cyclical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivityPattern {
    pub creation_frequency: f64,
    pub update_frequency: f64,
    pub connection_tendency: f64,
    pub lifespan_expectation: u64,
}

/// A temporal batch of generated data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalBatch {
    pub timestamp: u64,
    pub batch_id: u64,
    pub new_entities: Vec<TestEntity>,
    pub new_edges: Vec<TestEdge>,
    pub updated_entities: Vec<EntityUpdate>,
    pub deleted_entities: Vec<u32>,
    pub embedding_updates: HashMap<String, Vec<EmbeddingUpdate>>,
    pub temporal_queries: Vec<TemporalQuery>,
    pub batch_metadata: BatchMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityUpdate {
    pub entity_id: u32,
    pub property_changes: HashMap<String, String>,
    pub embedding_drift: Option<Vec<f64>>,
    pub update_reason: UpdateReason,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UpdateReason {
    PeriodicUpdate,
    EventTriggered,
    UserInteraction,
    SystemMaintenance,
    DataCorrection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingUpdate {
    pub entity_id: u32,
    pub old_embedding: Vec<f64>,
    pub new_embedding: Vec<f64>,
    pub drift_magnitude: f64,
    pub update_method: EmbeddingUpdateMethod,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmbeddingUpdateMethod {
    GradualDrift,
    AbruptChange,
    SeasonalVariation,
    NoiseInjection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalQuery {
    pub query_id: u64,
    pub query_type: TemporalQueryType,
    pub time_window: TimeWindow,
    pub expected_temporal_behavior: TemporalBehavior,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemporalQueryType {
    SnapshotQuery { snapshot_time: u64 },
    RangeQuery { start_time: u64, end_time: u64 },
    WindowQuery { window_size: u64, slide_size: u64 },
    VersionQuery { entity_id: u32, version_time: u64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeWindow {
    pub start: u64,
    pub end: u64,
    pub window_type: WindowType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WindowType {
    Tumbling,
    Sliding,
    Session,
    Landmark,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalBehavior {
    pub expected_growth_rate: f64,
    pub expected_update_frequency: f64,
    pub expected_deletion_rate: f64,
    pub temporal_consistency_requirements: Vec<ConsistencyRequirement>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyRequirement {
    pub requirement_type: ConsistencyType,
    pub tolerance: f64,
    pub validation_method: ValidationMethod,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsistencyType {
    MonotonicGrowth,
    BoundedVariation,
    EventualConsistency,
    CausalConsistency,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationMethod {
    Statistical,
    Algorithmic,
    Manual,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchMetadata {
    pub generation_time_ms: u128,
    pub entities_created: u64,
    pub entities_updated: u64,
    pub entities_deleted: u64,
    pub edges_created: u64,
    pub edges_deleted: u64,
    pub memory_usage_bytes: u64,
    pub validation_passed: bool,
}

impl StreamingDataGenerator {
    /// Create a new streaming data generator
    pub fn new(seed: u64, temporal_config: TemporalConfig) -> Self {
        let mut rng = DeterministicRng::new(seed);
        rng.set_label("streaming_generator".to_string());
        
        let base_generator = ComprehensiveDataGenerator::new(seed + 1);
        
        let stream_state = StreamingState {
            current_timestamp: 0,
            active_entities: HashMap::new(),
            active_edges: HashMap::new(),
            entity_lifecycle: EntityLifecycleTracker {
                creation_times: HashMap::new(),
                update_times: HashMap::new(),
                deletion_times: HashMap::new(),
                activity_patterns: HashMap::new(),
            },
            temporal_patterns: TemporalPatterns {
                growth_rate: 1.0,
                seasonal_patterns: Vec::new(),
                burst_events: Vec::new(),
                trend_direction: TrendDirection::Growing,
            },
        };

        Self {
            rng,
            base_generator,
            stream_state,
            temporal_config,
        }
    }

    /// Generate a complete temporal stream
    pub fn generate_temporal_stream(&mut self, duration: u64) -> Result<Vec<TemporalBatch>> {
        let mut batches = Vec::new();
        let mut batch_id = 0;

        for timestamp in 0..duration {
            self.stream_state.current_timestamp = timestamp;
            
            let batch = self.generate_temporal_batch(batch_id)?;
            self.update_stream_state(&batch)?;
            
            batches.push(batch);
            batch_id += 1;
        }

        Ok(batches)
    }

    /// Generate a single temporal batch
    pub fn generate_temporal_batch(&mut self, batch_id: u64) -> Result<TemporalBatch> {
        let start_time = std::time::Instant::now();
        
        // Calculate batch size based on temporal patterns
        let adjusted_batch_size = self.calculate_adjusted_batch_size()?;
        
        // Generate new entities
        let new_entities = self.generate_new_entities(adjusted_batch_size)?;
        
        // Generate new edges
        let new_edges = self.generate_new_edges(&new_entities)?;
        
        // Generate entity updates
        let updated_entities = self.generate_entity_updates()?;
        
        // Generate deletions
        let deleted_entities = self.generate_entity_deletions()?;
        
        // Generate embedding updates
        let embedding_updates = self.generate_embedding_updates(&updated_entities)?;
        
        // Generate temporal queries
        let temporal_queries = self.generate_temporal_queries()?;
        
        let generation_time = start_time.elapsed();
        
        let batch_metadata = BatchMetadata {
            generation_time_ms: generation_time.as_millis(),
            entities_created: new_entities.len() as u64,
            entities_updated: updated_entities.len() as u64,
            entities_deleted: deleted_entities.len() as u64,
            edges_created: new_edges.len() as u64,
            edges_deleted: 0, // Simplified for now
            memory_usage_bytes: self.estimate_memory_usage(&new_entities, &new_edges),
            validation_passed: true, // Will be validated separately
        };

        Ok(TemporalBatch {
            timestamp: self.stream_state.current_timestamp,
            batch_id,
            new_entities,
            new_edges,
            updated_entities,
            deleted_entities,
            embedding_updates,
            temporal_queries,
            batch_metadata,
        })
    }

    /// Process streaming data in real-time simulation
    pub fn simulate_realtime_processing<F>(&mut self, duration: u64, mut processor: F) -> Result<Vec<ProcessingResult>>
    where
        F: FnMut(&TemporalBatch) -> Result<ProcessingResult>,
    {
        let mut results = Vec::new();
        
        for timestamp in 0..duration {
            self.stream_state.current_timestamp = timestamp;
            
            // Generate batch
            let batch = self.generate_temporal_batch(timestamp)?;
            
            // Process batch
            let processing_start = std::time::Instant::now();
            let result = processor(&batch)?;
            let processing_time = processing_start.elapsed();
            
            // Update state
            self.update_stream_state(&batch)?;
            
            // Track results
            results.push(ProcessingResult {
                timestamp,
                batch_size: batch.new_entities.len() + batch.updated_entities.len(),
                processing_time_ms: processing_time.as_millis(),
                memory_usage: batch.batch_metadata.memory_usage_bytes,
                result,
            });
            
            // Simulate real-time delay
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        
        Ok(results)
    }

    // Private implementation methods

    fn calculate_adjusted_batch_size(&mut self) -> Result<usize> {
        let base_size = self.temporal_config.batch_size as f64;
        
        // Apply seasonal patterns
        let seasonal_multiplier = self.calculate_seasonal_multiplier()?;
        
        // Apply trend
        let trend_multiplier = match self.stream_state.temporal_patterns.trend_direction {
            TrendDirection::Growing => 1.0 + (self.stream_state.current_timestamp as f64 * 0.01),
            TrendDirection::Declining => 1.0 - (self.stream_state.current_timestamp as f64 * 0.01),
            TrendDirection::Stable => 1.0,
            TrendDirection::Cyclical => 1.0 + 0.1 * ((self.stream_state.current_timestamp as f64 * 0.1).sin()),
        };
        
        // Apply burst events
        let burst_multiplier = self.calculate_burst_multiplier()?;
        
        let adjusted_size = base_size * seasonal_multiplier * trend_multiplier * burst_multiplier;
        Ok(adjusted_size.max(1.0) as usize)
    }

    fn calculate_seasonal_multiplier(&self) -> Result<f64> {
        let mut multiplier = 1.0;
        
        for pattern in &self.stream_state.temporal_patterns.seasonal_patterns {
            let phase = (self.stream_state.current_timestamp as f64 * 2.0 * std::f64::consts::PI / pattern.period as f64) + pattern.phase_offset;
            
            let seasonal_value = match &pattern.pattern_type {
                SeasonalType::Sinusoidal => phase.sin(),
                SeasonalType::Square => if phase.sin() > 0.0 { 1.0 } else { -1.0 },
                SeasonalType::Sawtooth => 2.0 * (phase / (2.0 * std::f64::consts::PI) - (phase / (2.0 * std::f64::consts::PI) + 0.5).floor()),
                SeasonalType::Custom(values) => {
                    let index = (self.stream_state.current_timestamp % pattern.period) as usize % values.len();
                    values[index]
                },
            };
            
            multiplier *= 1.0 + pattern.amplitude * seasonal_value;
        }
        
        Ok(multiplier.max(0.1))
    }

    fn calculate_burst_multiplier(&self) -> Result<f64> {
        for burst in &self.stream_state.temporal_patterns.burst_events {
            if self.stream_state.current_timestamp >= burst.timestamp && 
               self.stream_state.current_timestamp < burst.timestamp + burst.duration {
                return Ok(burst.intensity_multiplier);
            }
        }
        Ok(1.0)
    }

    fn generate_new_entities(&mut self, count: usize) -> Result<Vec<TestEntity>> {
        let mut entities = Vec::new();
        
        for i in 0..count {
            let entity_id = self.generate_unique_entity_id();
            
            let entity = TestEntity {
                id: entity_id,
                entity_type: format!("temporal_entity_{}", i % 5),
                properties: {
                    let mut props = HashMap::new();
                    props.insert("created_at".to_string(), self.stream_state.current_timestamp.to_string());
                    props.insert("batch_origin".to_string(), "streaming".to_string());
                    props.insert("temporal_locality".to_string(), format!("{:.3}", self.temporal_config.temporal_locality));
                    props
                },
            };
            
            // Track in lifecycle
            self.stream_state.entity_lifecycle.creation_times.insert(entity_id, self.stream_state.current_timestamp);
            self.stream_state.active_entities.insert(entity_id, self.stream_state.current_timestamp);
            
            entities.push(entity);
        }
        
        Ok(entities)
    }

    fn generate_new_edges(&mut self, new_entities: &[TestEntity]) -> Result<Vec<TestEdge>> {
        let mut edges = Vec::new();
        let active_entity_ids: Vec<u32> = self.stream_state.active_entities.keys().cloned().collect();
        
        if active_entity_ids.len() < 2 {
            return Ok(edges);
        }
        
        let edges_to_create = (self.temporal_config.edges_per_step as f64 * 
                              self.calculate_seasonal_multiplier()?) as usize;
        
        for _ in 0..edges_to_create {
            let source_idx = self.rng.range(0, active_entity_ids.len());
            let target_idx = self.rng.range(0, active_entity_ids.len());
            
            if source_idx != target_idx {
                let source = active_entity_ids[source_idx];
                let target = active_entity_ids[target_idx];
                
                // Apply temporal locality - prefer connecting to recent entities
                let source_age = self.stream_state.current_timestamp - 
                    self.stream_state.active_entities.get(&source).unwrap_or(&0);
                let target_age = self.stream_state.current_timestamp - 
                    self.stream_state.active_entities.get(&target).unwrap_or(&0);
                
                let locality_bonus = if source_age < 10 || target_age < 10 { 2.0 } else { 1.0 };
                
                if self.rng.probability(self.temporal_config.temporal_locality * locality_bonus) {
                    let edge = TestEdge {
                        source,
                        target,
                        edge_type: "temporal_connection".to_string(),
                        properties: {
                            let mut props = HashMap::new();
                            props.insert("created_at".to_string(), self.stream_state.current_timestamp.to_string());
                            props.insert("temporal_score".to_string(), format!("{:.3}", locality_bonus));
                            props
                        },
                    };
                    
                    self.stream_state.active_edges.insert((source, target), self.stream_state.current_timestamp);
                    edges.push(edge);
                }
            }
        }
        
        Ok(edges)
    }

    fn generate_entity_updates(&mut self) -> Result<Vec<EntityUpdate>> {
        let mut updates = Vec::new();
        
        for &entity_id in self.stream_state.active_entities.keys() {
            if self.rng.probability(self.temporal_config.update_probability) {
                let mut property_changes = HashMap::new();
                property_changes.insert("last_updated".to_string(), self.stream_state.current_timestamp.to_string());
                property_changes.insert("update_count".to_string(), 
                    (self.stream_state.entity_lifecycle.update_times.get(&entity_id).map(|v| v.len()).unwrap_or(0) + 1).to_string());
                
                let update = EntityUpdate {
                    entity_id,
                    property_changes,
                    embedding_drift: Some(self.generate_embedding_drift()?),
                    update_reason: self.select_update_reason(),
                };
                
                // Track update
                self.stream_state.entity_lifecycle.update_times
                    .entry(entity_id)
                    .or_insert_with(Vec::new)
                    .push(self.stream_state.current_timestamp);
                
                updates.push(update);
            }
        }
        
        Ok(updates)
    }

    fn generate_entity_deletions(&mut self) -> Result<Vec<u32>> {
        let mut deletions = Vec::new();
        
        let entities_to_check: Vec<u32> = self.stream_state.active_entities.keys().cloned().collect();
        
        for entity_id in entities_to_check {
            if self.rng.probability(self.temporal_config.deletion_probability) {
                // Track deletion
                self.stream_state.entity_lifecycle.deletion_times.insert(entity_id, self.stream_state.current_timestamp);
                self.stream_state.active_entities.remove(&entity_id);
                
                deletions.push(entity_id);
            }
        }
        
        Ok(deletions)
    }

    fn generate_embedding_updates(&mut self, updated_entities: &[EntityUpdate]) -> Result<HashMap<String, Vec<EmbeddingUpdate>>> {
        let mut embedding_updates = HashMap::new();
        
        for entity_update in updated_entities {
            if let Some(drift) = &entity_update.embedding_drift {
                let update = EmbeddingUpdate {
                    entity_id: entity_update.entity_id,
                    old_embedding: vec![0.0; 64], // Simplified
                    new_embedding: drift.clone(),
                    drift_magnitude: self.calculate_drift_magnitude(drift),
                    update_method: EmbeddingUpdateMethod::GradualDrift,
                };
                
                embedding_updates.entry("default".to_string())
                    .or_insert_with(Vec::new)
                    .push(update);
            }
        }
        
        Ok(embedding_updates)
    }

    fn generate_temporal_queries(&mut self) -> Result<Vec<TemporalQuery>> {
        let mut queries = Vec::new();
        
        // Generate snapshot queries
        if self.rng.probability(0.3) {
            let query = TemporalQuery {
                query_id: self.rng.next() as u64,
                query_type: TemporalQueryType::SnapshotQuery { 
                    snapshot_time: self.stream_state.current_timestamp 
                },
                time_window: TimeWindow {
                    start: self.stream_state.current_timestamp,
                    end: self.stream_state.current_timestamp,
                    window_type: WindowType::Landmark,
                },
                expected_temporal_behavior: TemporalBehavior {
                    expected_growth_rate: self.stream_state.temporal_patterns.growth_rate,
                    expected_update_frequency: self.temporal_config.update_probability,
                    expected_deletion_rate: self.temporal_config.deletion_probability,
                    temporal_consistency_requirements: vec![
                        ConsistencyRequirement {
                            requirement_type: ConsistencyType::MonotonicGrowth,
                            tolerance: 0.1,
                            validation_method: ValidationMethod::Statistical,
                        }
                    ],
                },
            };
            queries.push(query);
        }
        
        // Generate range queries
        if self.rng.probability(0.2) {
            let window_size = 10;
            let start_time = if self.stream_state.current_timestamp >= window_size { 
                self.stream_state.current_timestamp - window_size 
            } else { 
                0 
            };
            
            let query = TemporalQuery {
                query_id: self.rng.next() as u64,
                query_type: TemporalQueryType::RangeQuery { 
                    start_time, 
                    end_time: self.stream_state.current_timestamp 
                },
                time_window: TimeWindow {
                    start: start_time,
                    end: self.stream_state.current_timestamp,
                    window_type: WindowType::Sliding,
                },
                expected_temporal_behavior: TemporalBehavior {
                    expected_growth_rate: self.stream_state.temporal_patterns.growth_rate,
                    expected_update_frequency: self.temporal_config.update_probability,
                    expected_deletion_rate: self.temporal_config.deletion_probability,
                    temporal_consistency_requirements: vec![],
                },
            };
            queries.push(query);
        }
        
        Ok(queries)
    }

    fn update_stream_state(&mut self, batch: &TemporalBatch) -> Result<()> {
        // Update growth rate based on actual generation
        let actual_growth = batch.new_entities.len() as f64;
        let expected_growth = self.temporal_config.entities_per_step as f64;
        
        self.stream_state.temporal_patterns.growth_rate = 
            0.9 * self.stream_state.temporal_patterns.growth_rate + 
            0.1 * (actual_growth / expected_growth);
        
        Ok(())
    }

    // Helper methods

    fn generate_unique_entity_id(&mut self) -> u32 {
        loop {
            let id = self.rng.range(1000000, 9999999) as u32;
            if !self.stream_state.active_entities.contains_key(&id) {
                return id;
            }
        }
    }

    fn generate_embedding_drift(&mut self) -> Result<Vec<f64>> {
        let dimension = 64;
        let mut drift = Vec::with_capacity(dimension);
        
        for _ in 0..dimension {
            drift.push(self.rng.gaussian(0.0, 0.01));
        }
        
        Ok(drift)
    }

    fn calculate_drift_magnitude(&self, drift: &[f64]) -> f64 {
        drift.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }

    fn select_update_reason(&mut self) -> UpdateReason {
        match self.rng.range(0, 5) {
            0 => UpdateReason::PeriodicUpdate,
            1 => UpdateReason::EventTriggered,
            2 => UpdateReason::UserInteraction,
            3 => UpdateReason::SystemMaintenance,
            _ => UpdateReason::DataCorrection,
        }
    }

    fn estimate_memory_usage(&self, entities: &[TestEntity], edges: &[TestEdge]) -> u64 {
        let entity_size = 256; // Estimated bytes per entity
        let edge_size = 128;   // Estimated bytes per edge
        
        (entities.len() * entity_size + edges.len() * edge_size) as u64
    }
}

/// Result of processing a temporal batch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingResult {
    pub timestamp: u64,
    pub batch_size: usize,
    pub processing_time_ms: u128,
    pub memory_usage: u64,
    pub result: ProcessingOutcome,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingOutcome {
    Success { entities_processed: u64 },
    PartialSuccess { entities_processed: u64, errors: Vec<String> },
    Failure { error: String },
}

/// Create default temporal configuration
pub fn create_default_temporal_config() -> TemporalConfig {
    TemporalConfig {
        time_steps: 100,
        entities_per_step: 10,
        edges_per_step: 15,
        update_probability: 0.1,
        deletion_probability: 0.05,
        batch_size: 50,
        temporal_locality: 0.7,
    }
}

/// Create configuration for high-frequency streaming
pub fn create_high_frequency_config() -> TemporalConfig {
    TemporalConfig {
        time_steps: 1000,
        entities_per_step: 100,
        edges_per_step: 200,
        update_probability: 0.2,
        deletion_probability: 0.02,
        batch_size: 500,
        temporal_locality: 0.8,
    }
}

/// Create configuration with seasonal patterns
pub fn create_seasonal_config() -> TemporalConfig {
    let mut config = create_default_temporal_config();
    config.time_steps = 365; // One year
    config
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_generator_creation() {
        let config = create_default_temporal_config();
        let generator = StreamingDataGenerator::new(42, config);
        
        assert_eq!(generator.stream_state.current_timestamp, 0);
        assert!(generator.stream_state.active_entities.is_empty());
    }

    #[test]
    fn test_temporal_batch_generation() {
        let config = create_default_temporal_config();
        let mut generator = StreamingDataGenerator::new(42, config);
        
        let batch = generator.generate_temporal_batch(0).unwrap();
        
        assert_eq!(batch.timestamp, 0);
        assert_eq!(batch.batch_id, 0);
        assert!(!batch.new_entities.is_empty());
        assert!(batch.batch_metadata.validation_passed);
    }

    #[test]
    fn test_temporal_stream_generation() {
        let config = create_default_temporal_config();
        let mut generator = StreamingDataGenerator::new(42, config);
        
        let stream = generator.generate_temporal_stream(10).unwrap();
        
        assert_eq!(stream.len(), 10);
        assert_eq!(stream[0].timestamp, 0);
        assert_eq!(stream[9].timestamp, 9);
    }

    #[test]
    fn test_entity_lifecycle_tracking() {
        let config = create_default_temporal_config();
        let mut generator = StreamingDataGenerator::new(42, config);
        
        // Generate multiple batches
        for i in 0..5 {
            let batch = generator.generate_temporal_batch(i).unwrap();
            generator.update_stream_state(&batch).unwrap();
            generator.stream_state.current_timestamp = i + 1;
        }
        
        // Check that lifecycle is tracked
        assert!(!generator.stream_state.entity_lifecycle.creation_times.is_empty());
    }

    #[test]
    fn test_seasonal_multiplier_calculation() {
        let config = create_default_temporal_config();
        let mut generator = StreamingDataGenerator::new(42, config);
        
        // Add seasonal pattern
        generator.stream_state.temporal_patterns.seasonal_patterns.push(SeasonalPattern {
            period: 10,
            amplitude: 0.5,
            phase_offset: 0.0,
            pattern_type: SeasonalType::Sinusoidal,
        });
        
        let multiplier = generator.calculate_seasonal_multiplier().unwrap();
        assert!(multiplier > 0.0);
    }

    #[test]
    fn test_temporal_query_generation() {
        let config = create_default_temporal_config();
        let mut generator = StreamingDataGenerator::new(42, config);
        
        // Set up some active entities first
        generator.stream_state.current_timestamp = 5;
        for i in 0..10 {
            generator.stream_state.active_entities.insert(i, 0);
        }
        
        let queries = generator.generate_temporal_queries().unwrap();
        
        // Should generate some queries with reasonable probability
        // (Note: may be empty due to randomness, that's okay)
        for query in queries {
            assert!(query.time_window.start <= query.time_window.end);
        }
    }
}