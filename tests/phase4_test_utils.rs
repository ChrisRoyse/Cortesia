use anyhow::Result;
use std::sync::Arc;
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

use llmkg::core::{
    brain_enhanced_graph::BrainEnhancedGraph,
    brain_types::{EntityKey, BrainInspiredEntity, ActivationPattern, BrainInspiredRelationship, RelationshipType},
};

use llmkg::learning::types::*;

/// Utilities for realistic Phase 4 testing
pub struct TestMetrics {
    pub start_time: std::time::Instant,
    pub measurements: HashMap<String, Vec<f32>>,
    pub events: Vec<String>,
}

impl TestMetrics {
    pub fn new() -> Self {
        Self {
            start_time: std::time::Instant::now(),
            measurements: HashMap::new(),
            events: Vec::new(),
        }
    }

    pub fn record(&mut self, metric_name: &str, value: f32) {
        self.measurements
            .entry(metric_name.to_string())
            .or_insert_with(Vec::new)
            .push(value);
    }

    pub fn log_event(&mut self, event: &str) {
        let elapsed = self.start_time.elapsed();
        self.events.push(format!("[{:?}] {}", elapsed, event));
    }

    pub fn get_average(&self, metric_name: &str) -> Option<f32> {
        self.measurements.get(metric_name).map(|values| {
            if values.is_empty() {
                0.0
            } else {
                values.iter().sum::<f32>() / values.len() as f32
            }
        })
    }

    pub fn get_trend(&self, metric_name: &str) -> Option<f32> {
        self.measurements.get(metric_name).and_then(|values| {
            if values.len() < 2 {
                None
            } else {
                // Simple linear regression slope
                let n = values.len() as f32;
                let x_mean = (n - 1.0) / 2.0;
                let y_mean = values.iter().sum::<f32>() / n;
                
                let mut numerator = 0.0;
                let mut denominator = 0.0;
                
                for (i, y) in values.iter().enumerate() {
                    let x = i as f32;
                    numerator += (x - x_mean) * (y - y_mean);
                    denominator += (x - x_mean).powi(2);
                }
                
                if denominator > 0.0 {
                    Some(numerator / denominator)
                } else {
                    None
                }
            }
        })
    }

    pub fn print_summary(&self) {
        println!("\n📊 Test Metrics Summary:");
        println!("Total duration: {:?}", self.start_time.elapsed());
        
        for (metric, values) in &self.measurements {
            if let Some(avg) = self.get_average(metric) {
                if let Some(trend) = self.get_trend(metric) {
                    println!("  {}: avg={:.3}, trend={:+.3}, samples={}", 
                             metric, avg, trend, values.len());
                } else {
                    println!("  {}: avg={:.3}, samples={}", metric, avg, values.len());
                }
            }
        }
        
        if !self.events.is_empty() {
            println!("\n📝 Event Log:");
            for event in &self.events {
                println!("  {}", event);
            }
        }
    }
}

/// Graph validation utilities
pub struct GraphValidator {
    pub max_orphan_entities: usize,
    pub min_connectivity: f32,
    pub max_weight_sum: f32,
}

impl Default for GraphValidator {
    fn default() -> Self {
        Self {
            max_orphan_entities: 5,
            min_connectivity: 0.1,
            max_weight_sum: 1000.0,
        }
    }
}

impl GraphValidator {
    pub async fn validate(&self, graph: &BrainEnhancedGraph) -> Result<ValidationReport> {
        let entities = graph.get_all_entities().await?;
        let relationships = graph.get_all_relationships().await?;
        
        // Check for orphan entities
        let mut connected_entities = std::collections::HashSet::new();
        for rel in &relationships {
            connected_entities.insert(rel.source);
            connected_entities.insert(rel.target);
        }
        
        let orphan_count = entities.iter()
            .filter(|e| !connected_entities.contains(&e.key))
            .count();
        
        // Calculate connectivity
        let possible_connections = entities.len() * (entities.len() - 1);
        let connectivity = if possible_connections > 0 {
            relationships.len() as f32 / possible_connections as f32
        } else {
            0.0
        };
        
        // Sum weights
        let total_weight: f32 = relationships.iter().map(|r| r.weight).sum();
        
        // Check for invalid weights
        let invalid_weights = relationships.iter()
            .filter(|r| r.weight < 0.0 || r.weight > 1.0)
            .count();
        
        let issues = self.check_issues(orphan_count, connectivity, total_weight, invalid_weights);
        
        Ok(ValidationReport {
            entity_count: entities.len(),
            relationship_count: relationships.len(),
            orphan_count,
            connectivity,
            total_weight,
            invalid_weights,
            issues,
            is_valid: issues.is_empty(),
        })
    }

    fn check_issues(&self, orphan_count: usize, connectivity: f32, total_weight: f32, invalid_weights: usize) -> Vec<String> {
        let mut issues = Vec::new();
        
        if orphan_count > self.max_orphan_entities {
            issues.push(format!("Too many orphan entities: {} > {}", orphan_count, self.max_orphan_entities));
        }
        
        if connectivity < self.min_connectivity {
            issues.push(format!("Connectivity too low: {:.3} < {:.3}", connectivity, self.min_connectivity));
        }
        
        if total_weight > self.max_weight_sum {
            issues.push(format!("Total weight too high: {:.1} > {:.1}", total_weight, self.max_weight_sum));
        }
        
        if invalid_weights > 0 {
            issues.push(format!("Found {} invalid weights", invalid_weights));
        }
        
        issues
    }
}

#[derive(Debug)]
pub struct ValidationReport {
    pub entity_count: usize,
    pub relationship_count: usize,
    pub orphan_count: usize,
    pub connectivity: f32,
    pub total_weight: f32,
    pub invalid_weights: usize,
    pub issues: Vec<String>,
    pub is_valid: bool,
}

/// Test data builder for creating realistic scenarios
pub struct TestDataBuilder {
    entities: Vec<BrainInspiredEntity>,
    relationships: Vec<(EntityKey, EntityKey, f32, RelationshipType)>,
    entity_index: HashMap<String, EntityKey>,
}

impl TestDataBuilder {
    pub fn new() -> Self {
        Self {
            entities: Vec::new(),
            relationships: Vec::new(),
            entity_index: HashMap::new(),
        }
    }

    pub fn add_entity(&mut self, name: &str, entity_type: &str) -> EntityKey {
        let key = EntityKey::new();
        
        let entity = BrainInspiredEntity {
            key,
            name: name.to_string(),
            entity_type: entity_type.to_string(),
            attributes: HashMap::new(),
            semantic_embedding: vec![0.1; 768],
            activation_pattern: ActivationPattern {
                current_activation: 0.5,
                activation_history: vec![0.5],
                decay_rate: 0.1,
                last_activated: SystemTime::now(),
            },
            temporal_aspects: Default::default(),
            ingestion_time: SystemTime::now(),
        };
        
        self.entities.push(entity);
        self.entity_index.insert(name.to_string(), key);
        key
    }

    pub fn add_relationship(&mut self, from: &str, to: &str, weight: f32, rel_type: RelationshipType) -> Result<()> {
        let source = self.entity_index.get(from)
            .ok_or_else(|| anyhow::anyhow!("Entity '{}' not found", from))?;
        let target = self.entity_index.get(to)
            .ok_or_else(|| anyhow::anyhow!("Entity '{}' not found", to))?;
        
        self.relationships.push((*source, *target, weight, rel_type));
        Ok(())
    }

    pub async fn build_graph(self) -> Result<BrainEnhancedGraph> {
        let graph = BrainEnhancedGraph::new().await?;
        
        // Add entities
        for entity in self.entities {
            graph.insert_entity(entity).await?;
        }
        
        // Add relationships
        for (source, target, weight, rel_type) in self.relationships {
            let relationship = BrainInspiredRelationship {
                source,
                target,
                relation_type: rel_type,
                weight,
                is_inhibitory: weight < 0.3,
                temporal_decay: 0.01,
                last_strengthened: SystemTime::now(),
                activation_count: 1,
                creation_time: SystemTime::now(),
                ingestion_time: SystemTime::now(),
            };
            graph.insert_relationship(relationship).await?;
        }
        
        Ok(graph)
    }

    pub fn create_hierarchical_structure(&mut self, depth: usize, branching_factor: usize) {
        self._create_hierarchy("root", "concept", depth, branching_factor, 0);
    }

    fn _create_hierarchy(&mut self, parent_name: &str, entity_type: &str, depth: usize, branching: usize, current_depth: usize) {
        if current_depth >= depth {
            return;
        }

        let parent_key = if current_depth == 0 {
            self.add_entity(parent_name, entity_type)
        } else {
            self.entity_index[parent_name]
        };

        for i in 0..branching {
            let child_name = format!("{}_{}", parent_name, i);
            let child_type = if current_depth == depth - 1 { "instance" } else { "concept" };
            
            self.add_entity(&child_name, child_type);
            self.add_relationship(parent_name, &child_name, 0.8, RelationshipType::Contains).ok();
            
            self._create_hierarchy(&child_name, child_type, depth, branching, current_depth + 1);
        }
    }
}

/// Performance profiler for identifying bottlenecks
pub struct PerformanceProfiler {
    checkpoints: Vec<(String, std::time::Instant)>,
    start_time: std::time::Instant,
}

impl PerformanceProfiler {
    pub fn new() -> Self {
        let now = std::time::Instant::now();
        Self {
            checkpoints: vec![("start".to_string(), now)],
            start_time: now,
        }
    }

    pub fn checkpoint(&mut self, name: &str) {
        self.checkpoints.push((name.to_string(), std::time::Instant::now()));
    }

    pub fn report(&self) -> ProfileReport {
        let mut segments = Vec::new();
        
        for i in 1..self.checkpoints.len() {
            let (name, time) = &self.checkpoints[i];
            let (_, prev_time) = &self.checkpoints[i-1];
            let duration = time.duration_since(*prev_time);
            
            segments.push(ProfileSegment {
                name: name.clone(),
                duration,
                percentage: (duration.as_secs_f32() / self.start_time.elapsed().as_secs_f32()) * 100.0,
            });
        }
        
        ProfileReport {
            total_duration: self.start_time.elapsed(),
            segments,
        }
    }
}

#[derive(Debug)]
pub struct ProfileReport {
    pub total_duration: Duration,
    pub segments: Vec<ProfileSegment>,
}

#[derive(Debug)]
pub struct ProfileSegment {
    pub name: String,
    pub duration: Duration,
    pub percentage: f32,
}

impl ProfileReport {
    pub fn print(&self) {
        println!("\n⏱️ Performance Profile:");
        println!("Total duration: {:?}", self.total_duration);
        println!("Breakdown:");
        
        for segment in &self.segments {
            println!("  {} - {:?} ({:.1}%)", segment.name, segment.duration, segment.percentage);
        }
        
        // Find bottlenecks
        if let Some(slowest) = self.segments.iter().max_by(|a, b| a.duration.cmp(&b.duration)) {
            println!("\n🐌 Slowest operation: {} ({:?})", slowest.name, slowest.duration);
        }
    }
}

/// Assertion helpers with better error messages
#[macro_export]
macro_rules! assert_in_range {
    ($value:expr, $min:expr, $max:expr, $metric:expr) => {
        assert!(
            $value >= $min && $value <= $max,
            "{} value {} outside expected range [{}, {}]",
            $metric, $value, $min, $max
        );
    };
}

#[macro_export]
macro_rules! assert_improvement {
    ($before:expr, $after:expr, $metric:expr) => {
        assert!(
            $after > $before,
            "{} did not improve: before={:.3}, after={:.3}",
            $metric, $before, $after
        );
    };
}

#[macro_export]
macro_rules! assert_convergence {
    ($values:expr, $threshold:expr, $metric:expr) => {
        if $values.len() >= 2 {
            let last_idx = $values.len() - 1;
            let change = ($values[last_idx] - $values[last_idx - 1]).abs();
            assert!(
                change < $threshold,
                "{} not converging: last change={:.3} > threshold={:.3}",
                $metric, change, $threshold
            );
        }
    };
}

/// Test scenario templates
pub mod scenarios {
    use super::*;

    pub async fn create_knowledge_domain(name: &str, size: usize) -> Result<BrainEnhancedGraph> {
        let mut builder = TestDataBuilder::new();
        
        // Create domain root
        let root_name = format!("{}_root", name);
        builder.add_entity(&root_name, "domain");
        
        // Add concepts
        for i in 0..size {
            let concept_name = format!("{}_concept_{}", name, i);
            builder.add_entity(&concept_name, "concept");
            builder.add_relationship(&root_name, &concept_name, 0.7, RelationshipType::Contains)?;
            
            // Add some inter-concept relationships
            if i > 0 {
                let prev_concept = format!("{}_concept_{}", name, i - 1);
                builder.add_relationship(&prev_concept, &concept_name, 0.5, RelationshipType::RelatedTo)?;
            }
        }
        
        builder.build_graph().await
    }

    pub fn create_learning_sequence(domain: &str, complexity: f32) -> Vec<ActivationEvent> {
        let mut events = Vec::new();
        let base_time = std::time::Instant::now();
        
        for i in 0..50 {
            let pattern = match (i / 10) % 4 {
                0 => CognitivePatternType::Convergent,
                1 => CognitivePatternType::Divergent,
                2 => CognitivePatternType::Transform,
                _ => CognitivePatternType::AbstractThinking,
            };
            
            events.push(ActivationEvent {
                entity_key: EntityKey::new(),
                activation_strength: 0.4 + (complexity * 0.4),
                timestamp: base_time + Duration::from_millis(i * 100),
                context: ActivationContext {
                    query_id: format!("{}_query_{}", domain, i / 5),
                    cognitive_pattern: pattern,
                    user_session: Some(format!("{}_session", domain)),
                    outcome_quality: Some(0.6 + (i as f32 / 100.0)),
                },
            });
        }
        
        events
    }
}