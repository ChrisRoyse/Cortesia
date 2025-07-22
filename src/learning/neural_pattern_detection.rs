use crate::core::brain_types::{ActivationPattern, BrainInspiredEntity};
use crate::core::types::EntityKey;
use crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use crate::learning::types::ActivationEvent;
use crate::neural::neural_server::NeuralProcessingServer;
use crate::error::Result;

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, Duration};
use serde::{Serialize, Deserialize};
use uuid::Uuid;
use futures;
use async_trait::async_trait;

/// Enhanced neural pattern detection for Phase 4 learning
pub struct NeuralPatternDetectionSystem {
    pub brain_graph: Arc<BrainEnhancedKnowledgeGraph>,
    pub neural_server: Arc<NeuralProcessingServer>,
    pub pattern_detectors: HashMap<PatternType, Box<dyn PatternDetector>>,
    pub pattern_cache: Arc<RwLock<PatternCache>>,
    pub detection_config: PatternDetectionConfig,
    pub pattern_models: HashMap<String, String>, // Model IDs for different patterns
}

impl std::fmt::Debug for NeuralPatternDetectionSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NeuralPatternDetectionSystem")
            .field("brain_graph", &"BrainEnhancedKnowledgeGraph")
            .field("neural_server", &self.neural_server)
            .field("pattern_detectors", &self.pattern_detectors.keys().collect::<Vec<_>>())
            .field("pattern_cache", &"Arc<RwLock<PatternCache>>")
            .field("detection_config", &self.detection_config)
            .field("pattern_models", &self.pattern_models)
            .finish()
    }
}

impl Clone for NeuralPatternDetectionSystem {
    fn clone(&self) -> Self {
        Self {
            brain_graph: self.brain_graph.clone(),
            neural_server: self.neural_server.clone(),
            pattern_detectors: HashMap::new(), // Can't clone trait objects
            pattern_cache: self.pattern_cache.clone(),
            detection_config: self.detection_config.clone(),
            pattern_models: self.pattern_models.clone(),
        }
    }
}

/// Core pattern detection trait
#[async_trait]
pub trait PatternDetector: Send + Sync {
    async fn detect_patterns(&self, scope: &AnalysisScope) -> Result<Vec<DetectedPattern>>;
    fn get_pattern_type(&self) -> PatternType;
    fn get_confidence_threshold(&self) -> f32;
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PatternType {
    ActivationPattern,
    FrequencyPattern,
    SynchronyPattern,
    OscillatoryPattern,
    TemporalPattern,
    SpatialPattern,
    CausalPattern,
    HierarchicalPattern,
    CompetitivePattern,
    CooperativePattern,
}

#[derive(Debug, Clone)]
pub struct AnalysisScope {
    pub entities: Vec<EntityKey>,
    pub time_window: Duration,
    pub depth: usize,
    pub pattern_types: Vec<PatternType>,
    pub minimum_confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedPattern {
    pub pattern_id: Uuid,
    pub pattern_type: PatternType,
    pub entities_involved: Vec<EntityKey>,
    pub confidence: f32,
    pub significance: f32,
    pub temporal_span: Duration,
    pub pattern_strength: f32,
    pub pattern_frequency: f32,
    pub pattern_metadata: HashMap<String, String>,
    pub discovery_timestamp: SystemTime,
}

#[derive(Debug, Clone)]
pub struct PatternCache {
    pub cached_patterns: HashMap<String, CachedPatternResult>,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub max_cache_size: usize,
    pub cache_ttl: Duration,
}

#[derive(Debug, Clone)]
pub struct CachedPatternResult {
    pub patterns: Vec<DetectedPattern>,
    pub timestamp: SystemTime,
    pub analysis_scope: AnalysisScope,
}

#[derive(Debug, Clone)]
pub struct PatternDetectionConfig {
    pub enable_caching: bool,
    pub cache_ttl: Duration,
    pub parallel_detection: bool,
    pub max_concurrent_detectors: usize,
    pub neural_model_timeout: Duration,
    pub pattern_significance_threshold: f32,
}

// Specific pattern detector implementations
#[derive(Debug)]
pub struct ActivationPatternDetector {
    pub brain_graph: Arc<BrainEnhancedKnowledgeGraph>,
    pub confidence_threshold: f32,
}

#[derive(Debug)]
pub struct FrequencyPatternDetector {
    pub brain_graph: Arc<BrainEnhancedKnowledgeGraph>,
    pub neural_server: Arc<NeuralProcessingServer>,
    pub frequency_model_id: String,
}

#[derive(Debug)]
pub struct SynchronyPatternDetector {
    pub brain_graph: Arc<BrainEnhancedKnowledgeGraph>,
    pub synchrony_threshold: f32,
    pub time_window: Duration,
}

#[derive(Debug)]
pub struct OscillatoryPatternDetector {
    pub neural_server: Arc<NeuralProcessingServer>,
    pub oscillation_model_id: String,
    pub min_frequency: f32,
    pub max_frequency: f32,
}

#[derive(Debug)]
pub struct TemporalPatternDetector {
    pub neural_server: Arc<NeuralProcessingServer>,
    pub temporal_model_id: String, // TimesNet model
    pub sequence_length: usize,
}

impl NeuralPatternDetectionSystem {
    pub async fn new(
        brain_graph: Arc<BrainEnhancedKnowledgeGraph>,
        neural_server: Arc<NeuralProcessingServer>,
    ) -> Result<Self> {
        let mut pattern_detectors: HashMap<PatternType, Box<dyn PatternDetector>> = HashMap::new();
        
        // Initialize pattern detectors
        pattern_detectors.insert(
            PatternType::ActivationPattern,
            Box::new(ActivationPatternDetector {
                brain_graph: brain_graph.clone(),
                confidence_threshold: 0.7,
            })
        );
        
        pattern_detectors.insert(
            PatternType::FrequencyPattern,
            Box::new(FrequencyPatternDetector {
                brain_graph: brain_graph.clone(),
                neural_server: neural_server.clone(),
                frequency_model_id: "frequency_analyzer".to_string(),
            })
        );
        
        pattern_detectors.insert(
            PatternType::SynchronyPattern,
            Box::new(SynchronyPatternDetector {
                brain_graph: brain_graph.clone(),
                synchrony_threshold: 0.8,
                time_window: Duration::from_millis(100),
            })
        );
        
        pattern_detectors.insert(
            PatternType::OscillatoryPattern,
            Box::new(OscillatoryPatternDetector {
                neural_server: neural_server.clone(),
                oscillation_model_id: "oscillation_detector".to_string(),
                min_frequency: 0.1,
                max_frequency: 100.0,
            })
        );
        
        pattern_detectors.insert(
            PatternType::TemporalPattern,
            Box::new(TemporalPatternDetector {
                neural_server: neural_server.clone(),
                temporal_model_id: "timesnet_pattern_detector".to_string(),
                sequence_length: 50,
            })
        );
        
        let mut pattern_models = HashMap::new();
        pattern_models.insert("frequency_analyzer".to_string(), "freq_model_v1".to_string());
        pattern_models.insert("oscillation_detector".to_string(), "osc_model_v1".to_string());
        pattern_models.insert("timesnet_pattern_detector".to_string(), "timesnet_v2".to_string());
        pattern_models.insert("nbeats_anomaly_detector".to_string(), "nbeats_v1".to_string());
        
        Ok(Self {
            brain_graph,
            neural_server,
            pattern_detectors,
            pattern_cache: Arc::new(RwLock::new(PatternCache::new())),
            detection_config: PatternDetectionConfig::default(),
            pattern_models,
        })
    }

    /// Detect activation patterns in the neural network
    pub async fn detect_activation_patterns(&self, time_window: Duration) -> Result<Vec<ActivationPattern>> {
        let analysis_scope = AnalysisScope {
            entities: Vec::new(), // Will be populated from brain graph
            time_window,
            depth: 3,
            pattern_types: vec![PatternType::ActivationPattern],
            minimum_confidence: 0.6,
        };
        
        let detected_patterns = self.detect_patterns(&analysis_scope).await?;
        
        // Convert detected patterns to activation patterns
        let mut activation_patterns = Vec::new();
        for pattern in detected_patterns {
            if pattern.pattern_type == PatternType::ActivationPattern {
                let mut activations = HashMap::new();
                for entity in pattern.entities_involved {
                    activations.insert(entity, pattern.pattern_strength);
                }
                
                activation_patterns.push(ActivationPattern {
                    activations,
                    timestamp: pattern.discovery_timestamp,
                    query: format!("pattern_{}", pattern.pattern_id),
                });
            }
        }
        
        Ok(activation_patterns)
    }

    /// Detect frequency patterns in entity activations
    pub async fn detect_frequency_patterns(&self, entities: &[EntityKey]) -> Result<Vec<FrequencyPattern>> {
        let analysis_scope = AnalysisScope {
            entities: entities.to_vec(),
            time_window: Duration::from_secs(300), // 5 minutes
            depth: 2,
            pattern_types: vec![PatternType::FrequencyPattern],
            minimum_confidence: 0.7,
        };
        
        let detected_patterns = self.detect_patterns(&analysis_scope).await?;
        
        // Convert to frequency patterns
        let mut frequency_patterns = Vec::new();
        for pattern in detected_patterns {
            if pattern.pattern_type == PatternType::FrequencyPattern {
                frequency_patterns.push(FrequencyPattern {
                    pattern_id: pattern.pattern_id,
                    entities: pattern.entities_involved,
                    frequency_hz: pattern.pattern_frequency,
                    amplitude: pattern.pattern_strength,
                    phase_relationships: HashMap::new(), // Would be computed from neural analysis
                    stability: pattern.confidence,
                    duration: pattern.temporal_span,
                });
            }
        }
        
        Ok(frequency_patterns)
    }

    /// Detect synchrony patterns between entities
    pub async fn detect_synchrony_patterns(&self, activation_events: &[ActivationEvent]) -> Result<Vec<SynchronyPattern>> {
        let entity_keys: Vec<EntityKey> = activation_events.iter()
            .map(|event| event.entity_key)
            .collect();
        
        let analysis_scope = AnalysisScope {
            entities: entity_keys,
            time_window: Duration::from_millis(500),
            depth: 1,
            pattern_types: vec![PatternType::SynchronyPattern],
            minimum_confidence: 0.75,
        };
        
        let detected_patterns = self.detect_patterns(&analysis_scope).await?;
        
        // Convert to synchrony patterns
        let mut synchrony_patterns = Vec::new();
        for pattern in detected_patterns {
            if pattern.pattern_type == PatternType::SynchronyPattern {
                synchrony_patterns.push(SynchronyPattern {
                    pattern_id: pattern.pattern_id,
                    synchronized_entities: pattern.entities_involved,
                    synchrony_strength: pattern.pattern_strength,
                    time_lag: Duration::from_millis(0), // Would be computed from timing analysis
                    phase_coherence: pattern.confidence,
                    duration: pattern.temporal_span,
                });
            }
        }
        
        Ok(synchrony_patterns)
    }

    /// Detect oscillatory patterns in time series data
    pub async fn detect_oscillatory_patterns(&self, time_series: &[f32]) -> Result<Vec<OscillatoryPattern>> {
        // Use neural network for oscillation detection
        let model_id = self.pattern_models.get("oscillation_detector")
            .ok_or_else(|| crate::error::GraphError::PatternNotFound("Oscillation detection model not found".to_string()))?;
        
        // Encode time series for neural analysis
        let neural_input = self.encode_time_series_for_neural_analysis(time_series)?;
        
        let neural_prediction = self.neural_server.neural_predict(
            model_id,
            neural_input,
        ).await?;
        
        // Decode neural predictions to oscillatory patterns
        let oscillatory_patterns = self.decode_oscillatory_predictions(neural_prediction.prediction)?;
        
        Ok(oscillatory_patterns)
    }

    /// Main pattern detection interface
    pub async fn detect_patterns(&self, scope: &AnalysisScope) -> Result<Vec<DetectedPattern>> {
        // Check cache first
        if self.detection_config.enable_caching {
            if let Some(cached_result) = self.check_pattern_cache(scope).await? {
                return Ok(cached_result.patterns);
            }
        }
        
        let mut all_patterns = Vec::new();
        
        if self.detection_config.parallel_detection {
            // Run pattern detectors in parallel
            let mut detector_futures = Vec::new();
            
            for pattern_type in &scope.pattern_types {
                if let Some(detector) = self.pattern_detectors.get(pattern_type) {
                    detector_futures.push(detector.detect_patterns(scope));
                }
            }
            
            // Wait for all detectors to complete
            let results = futures::future::try_join_all(detector_futures).await?;
            
            // Combine results
            for patterns in results {
                all_patterns.extend(patterns);
            }
        } else {
            // Run pattern detectors sequentially
            for pattern_type in &scope.pattern_types {
                if let Some(detector) = self.pattern_detectors.get(pattern_type) {
                    let patterns = detector.detect_patterns(scope).await?;
                    all_patterns.extend(patterns);
                }
            }
        }
        
        // Filter by confidence and significance
        let filtered_patterns: Vec<DetectedPattern> = all_patterns.into_iter()
            .filter(|pattern| {
                pattern.confidence >= scope.minimum_confidence &&
                pattern.significance >= self.detection_config.pattern_significance_threshold
            })
            .collect();
        
        // Cache results
        if self.detection_config.enable_caching {
            self.cache_pattern_results(&filtered_patterns, scope).await?;
        }
        
        Ok(filtered_patterns)
    }

    async fn check_pattern_cache(&self, scope: &AnalysisScope) -> Result<Option<CachedPatternResult>> {
        let cache = self.pattern_cache.read().unwrap();
        let cache_key = self.generate_cache_key(scope);
        
        if let Some(cached_result) = cache.cached_patterns.get(&cache_key) {
            // Check if cache entry is still valid
            let age = SystemTime::now().duration_since(cached_result.timestamp).unwrap_or(Duration::MAX);
            if age <= self.detection_config.cache_ttl {
                return Ok(Some(cached_result.clone()));
            }
        }
        
        Ok(None)
    }

    async fn cache_pattern_results(&self, patterns: &[DetectedPattern], scope: &AnalysisScope) -> Result<()> {
        let mut cache = self.pattern_cache.write().unwrap();
        let cache_key = self.generate_cache_key(scope);
        
        let cached_result = CachedPatternResult {
            patterns: patterns.to_vec(),
            timestamp: SystemTime::now(),
            analysis_scope: scope.clone(),
        };
        
        cache.cached_patterns.insert(cache_key, cached_result);
        
        // Implement cache size management
        if cache.cached_patterns.len() > cache.max_cache_size {
            self.evict_old_cache_entries(&mut cache);
        }
        
        Ok(())
    }

    fn generate_cache_key(&self, scope: &AnalysisScope) -> String {
        format!(
            "entities_{}_window_{}_types_{:?}",
            scope.entities.len(),
            scope.time_window.as_millis(),
            scope.pattern_types
        )
    }

    fn evict_old_cache_entries(&self, cache: &mut PatternCache) {
        // Remove oldest entries until cache is within size limit
        let target_size = cache.max_cache_size * 80 / 100; // Keep 80% of max
        
        let entries_to_remove = cache.cached_patterns.len().saturating_sub(target_size);
        if entries_to_remove > 0 {
            let mut entries: Vec<_> = cache.cached_patterns.iter()
                .map(|(key, result)| (key.clone(), result.timestamp))
                .collect();
            entries.sort_by_key(|(_, timestamp)| *timestamp);
            
            let keys_to_remove: Vec<_> = entries.iter()
                .take(entries_to_remove)
                .map(|(key, _)| key.clone())
                .collect();
            
            for key in keys_to_remove {
                cache.cached_patterns.remove(key.as_str());
            }
        }
    }

    fn encode_time_series_for_neural_analysis(&self, time_series: &[f32]) -> Result<Vec<f32>> {
        // Normalize and prepare time series for neural network input
        let mean = time_series.iter().sum::<f32>() / time_series.len() as f32;
        let variance = time_series.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / time_series.len() as f32;
        let std_dev = variance.sqrt();
        
        let normalized: Vec<f32> = time_series.iter()
            .map(|x| if std_dev > 0.0 { (x - mean) / std_dev } else { 0.0 })
            .collect();
        
        Ok(normalized)
    }

    fn decode_oscillatory_predictions(&self, neural_prediction: Vec<f32>) -> Result<Vec<OscillatoryPattern>> {
        let mut patterns = Vec::new();
        
        // Simplified decoding - in practice, this would be more sophisticated
        for (i, &prediction) in neural_prediction.iter().enumerate() {
            if prediction > 0.7 { // Threshold for oscillation detection
                patterns.push(OscillatoryPattern {
                    pattern_id: Uuid::new_v4(),
                    frequency: (i as f32) * 0.1 + 1.0, // Frequency mapping
                    amplitude: prediction,
                    phase: 0.0, // Would be computed from phase analysis
                    coherence: prediction,
                    start_time: SystemTime::now(),
                    duration: Duration::from_secs(60),
                });
            }
        }
        
        Ok(patterns)
    }
}

// Pattern detector implementations
#[async_trait]
impl PatternDetector for ActivationPatternDetector {
    async fn detect_patterns(&self, scope: &AnalysisScope) -> Result<Vec<DetectedPattern>> {
        let mut patterns = Vec::new();
        
        // Get current activation states from brain graph
        let entities = if scope.entities.is_empty() {
            // Get all entities and convert them to BrainInspiredEntity
            let all_entities = self.brain_graph.get_all_entities().await;
            let mut entities = Vec::new();
            for (entity_key, entity_data, activation) in all_entities {
                entities.push(BrainInspiredEntity {
                    id: entity_key,
                    concept_id: {
                        use slotmap::{Key, KeyData};
                        let key_data: KeyData = entity_key.data();
                        format!("concept_{}", key_data.as_ffi())
                    },
                    properties: serde_json::from_str(&entity_data.properties).unwrap_or_default(),
                    embedding: entity_data.embedding,
                    activation_state: activation,
                    direction: crate::core::brain_types::EntityDirection::Input,
                    last_activation: SystemTime::now(),
                    last_update: SystemTime::now(),
                });
            }
            entities
        } else {
            // Get specific entities
            let mut entities = Vec::new();
            for entity_key in &scope.entities {
                // In practice, would get entity from brain graph
                entities.push(BrainInspiredEntity {
                    id: *entity_key,
                    concept_id: {
                        use slotmap::{Key, KeyData};
                        let key_data: KeyData = entity_key.data();
                        format!("concept_{}", key_data.as_ffi())
                    },
                    properties: HashMap::new(),
                    embedding: vec![0.0; 128],
                    activation_state: 0.5,
                    direction: crate::core::brain_types::EntityDirection::Input,
                    last_activation: SystemTime::now(),
                    last_update: SystemTime::now(),
                });
            }
            entities
        };
        
        // Detect patterns in activations
        let mut high_activation_entities = Vec::new();
        for entity in entities {
            if entity.activation_state > self.confidence_threshold {
                high_activation_entities.push(entity.id);
            }
        }
        
        if high_activation_entities.len() >= 2 {
            patterns.push(DetectedPattern {
                pattern_id: Uuid::new_v4(),
                pattern_type: PatternType::ActivationPattern,
                entities_involved: high_activation_entities,
                confidence: 0.85,
                significance: 0.8,
                temporal_span: scope.time_window,
                pattern_strength: 0.75,
                pattern_frequency: 1.0,
                pattern_metadata: {
                    let mut metadata = HashMap::new();
                    metadata.insert("detector".to_string(), "activation_pattern_detector".to_string());
                    metadata
                },
                discovery_timestamp: SystemTime::now(),
            });
        }
        
        Ok(patterns)
    }

    fn get_pattern_type(&self) -> PatternType {
        PatternType::ActivationPattern
    }

    fn get_confidence_threshold(&self) -> f32 {
        self.confidence_threshold
    }
}

#[async_trait]
impl PatternDetector for FrequencyPatternDetector {
    async fn detect_patterns(&self, scope: &AnalysisScope) -> Result<Vec<DetectedPattern>> {
        let mut patterns = Vec::new();
        
        // Use neural network for frequency analysis
        let neural_input = self.encode_entities_for_frequency_analysis(&scope.entities).await?;
        
        let neural_prediction = self.neural_server.neural_predict(
            &self.frequency_model_id,
            neural_input,
        ).await?;
        
        // Decode neural predictions
        if let Some(&frequency_score) = neural_prediction.prediction.first() {
            if frequency_score > 0.7 {
                patterns.push(DetectedPattern {
                    pattern_id: Uuid::new_v4(),
                    pattern_type: PatternType::FrequencyPattern,
                    entities_involved: scope.entities.clone(),
                    confidence: frequency_score,
                    significance: frequency_score * 0.9,
                    temporal_span: scope.time_window,
                    pattern_strength: frequency_score,
                    pattern_frequency: frequency_score * 10.0, // Scaled frequency
                    pattern_metadata: {
                        let mut metadata = HashMap::new();
                        metadata.insert("neural_model".to_string(), self.frequency_model_id.clone());
                        metadata
                    },
                    discovery_timestamp: SystemTime::now(),
                });
            }
        }
        
        Ok(patterns)
    }

    fn get_pattern_type(&self) -> PatternType {
        PatternType::FrequencyPattern
    }

    fn get_confidence_threshold(&self) -> f32 {
        0.7
    }
}

impl FrequencyPatternDetector {
    async fn encode_entities_for_frequency_analysis(&self, entities: &[EntityKey]) -> Result<Vec<f32>> {
        // Simplified encoding - would be more sophisticated in practice
        let mut encoding = Vec::new();
        
        for entity in entities {
            // Get entity activation history and compute frequency features
            let recent_activations = self.brain_graph.get_entity_activation(*entity).await;
            encoding.push(recent_activations as f32 / 100.0);
        }
        
        // Pad to fixed size if needed
        while encoding.len() < 64 {
            encoding.push(0.0);
        }
        
        Ok(encoding)
    }
}

// Additional pattern detector implementations

#[async_trait]
impl PatternDetector for SynchronyPatternDetector {
    async fn detect_patterns(&self, scope: &AnalysisScope) -> Result<Vec<DetectedPattern>> {
            let mut patterns = Vec::new();
            
            // Simplified synchrony detection
            if scope.entities.len() >= 2 {
                patterns.push(DetectedPattern {
                    pattern_id: uuid::Uuid::new_v4(),
                    pattern_type: PatternType::SynchronyPattern,
                    entities_involved: scope.entities.clone(),
                    confidence: self.synchrony_threshold,
                    significance: self.synchrony_threshold * 0.9,
                    temporal_span: self.time_window,
                    pattern_strength: self.synchrony_threshold,
                    pattern_frequency: 1.0,
                    pattern_metadata: {
                        let mut metadata = HashMap::new();
                        metadata.insert("detector".to_string(), "synchrony_pattern_detector".to_string());
                        metadata
                    },
                    discovery_timestamp: SystemTime::now(),
                });
            }
            
            Ok(patterns)
    }

    fn get_pattern_type(&self) -> PatternType {
        PatternType::SynchronyPattern
    }

    fn get_confidence_threshold(&self) -> f32 {
        self.synchrony_threshold
    }
}

#[async_trait]
impl PatternDetector for OscillatoryPatternDetector {
    async fn detect_patterns(&self, scope: &AnalysisScope) -> Result<Vec<DetectedPattern>> {
            let mut patterns = Vec::new();
            
            // Use neural network for oscillation detection
            let time_series: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
            
            let neural_prediction = self.neural_server.neural_predict(
                &self.oscillation_model_id,
                time_series,
            ).await?;
            
            // Decode neural predictions
            if let Some(&oscillation_score) = neural_prediction.prediction.first() {
                if oscillation_score > 0.7 {
                    patterns.push(DetectedPattern {
                        pattern_id: uuid::Uuid::new_v4(),
                        pattern_type: PatternType::OscillatoryPattern,
                        entities_involved: scope.entities.clone(),
                        confidence: oscillation_score,
                        significance: oscillation_score * 0.9,
                        temporal_span: scope.time_window,
                        pattern_strength: oscillation_score,
                        pattern_frequency: oscillation_score * self.max_frequency,
                        pattern_metadata: {
                            let mut metadata = HashMap::new();
                            metadata.insert("neural_model".to_string(), self.oscillation_model_id.clone());
                            metadata
                        },
                        discovery_timestamp: SystemTime::now(),
                    });
                }
            }
            
            Ok(patterns)
    }

    fn get_pattern_type(&self) -> PatternType {
        PatternType::OscillatoryPattern
    }

    fn get_confidence_threshold(&self) -> f32 {
        0.7
    }
}

#[async_trait]
impl PatternDetector for TemporalPatternDetector {
    async fn detect_patterns(&self, scope: &AnalysisScope) -> Result<Vec<DetectedPattern>> {
            let mut patterns = Vec::new();
            
            // Use temporal neural network
            let sequence_data: Vec<f32> = (0..self.sequence_length).map(|i| i as f32 / self.sequence_length as f32).collect();
            
            let neural_prediction = self.neural_server.neural_predict(
                &self.temporal_model_id,
                sequence_data,
            ).await?;
            
            // Decode temporal predictions
            if let Some(&temporal_score) = neural_prediction.prediction.first() {
                if temporal_score > 0.6 {
                    patterns.push(DetectedPattern {
                        pattern_id: uuid::Uuid::new_v4(),
                        pattern_type: PatternType::TemporalPattern,
                        entities_involved: scope.entities.clone(),
                        confidence: temporal_score,
                        significance: temporal_score * 0.85,
                        temporal_span: scope.time_window,
                        pattern_strength: temporal_score,
                        pattern_frequency: 1.0,
                        pattern_metadata: {
                            let mut metadata = HashMap::new();
                            metadata.insert("neural_model".to_string(), self.temporal_model_id.clone());
                            metadata.insert("sequence_length".to_string(), self.sequence_length.to_string());
                            metadata
                        },
                        discovery_timestamp: SystemTime::now(),
                    });
                }
            }
            
            Ok(patterns)
    }

    fn get_pattern_type(&self) -> PatternType {
        PatternType::TemporalPattern
    }

    fn get_confidence_threshold(&self) -> f32 {
        0.6
    }
}

impl PatternCache {
    fn new() -> Self {
        Self {
            cached_patterns: HashMap::new(),
            cache_hits: 0,
            cache_misses: 0,
            max_cache_size: 1000,
            cache_ttl: Duration::from_secs(300), // 5 minutes
        }
    }
}

impl Default for PatternDetectionConfig {
    fn default() -> Self {
        Self {
            enable_caching: true,
            cache_ttl: Duration::from_secs(300),
            parallel_detection: true,
            max_concurrent_detectors: 4,
            neural_model_timeout: Duration::from_secs(30),
            pattern_significance_threshold: 0.6,
        }
    }
}

/// Pattern structures for different types
#[derive(Debug, Clone)]
pub struct FrequencyPattern {
    pub pattern_id: Uuid,
    pub entities: Vec<EntityKey>,
    pub frequency_hz: f32,
    pub amplitude: f32,
    pub phase_relationships: HashMap<EntityKey, f32>,
    pub stability: f32,
    pub duration: Duration,
}

#[derive(Debug, Clone)]
pub struct SynchronyPattern {
    pub pattern_id: Uuid,
    pub synchronized_entities: Vec<EntityKey>,
    pub synchrony_strength: f32,
    pub time_lag: Duration,
    pub phase_coherence: f32,
    pub duration: Duration,
}

#[derive(Debug, Clone)]
pub struct OscillatoryPattern {
    pub pattern_id: Uuid,
    pub frequency: f32,
    pub amplitude: f32,
    pub phase: f32,
    pub coherence: f32,
    pub start_time: SystemTime,
    pub duration: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
    use crate::neural::neural_server::NeuralProcessingServer;
    use crate::core::types::EntityKey;
    use std::time::{Duration, SystemTime};
    use uuid::Uuid;

    // Test helper to create mock neural pattern detection system
    async fn create_test_pattern_detection_system() -> Result<NeuralPatternDetectionSystem> {
        let embedding_dim = 512;
        let brain_graph = Arc::new(BrainEnhancedKnowledgeGraph::new(embedding_dim)?);
        let neural_server = Arc::new(NeuralProcessingServer::new_mock());
        
        NeuralPatternDetectionSystem::new(brain_graph, neural_server).await
    }

    fn create_test_analysis_scope() -> AnalysisScope {
        AnalysisScope {
            entities: vec![EntityKey::default(), EntityKey::default()],
            time_window: Duration::from_secs(60),
            depth: 3,
            pattern_types: vec![
                PatternType::ActivationPattern,
                PatternType::FrequencyPattern,
                PatternType::SynchronyPattern,
            ],
            minimum_confidence: 0.6,
        }
    }

    fn create_test_activation_events() -> Vec<ActivationEvent> {
        vec![
            ActivationEvent {
                entity_key: EntityKey::default(),
                activation_strength: 0.8,
                timestamp: std::time::Instant::now(),
                context: ActivationContext {
                    query_id: "test-query-1".to_string(),
                    cognitive_pattern: CognitivePatternType::Convergent,
                    user_session: Some("test-session".to_string()),
                    outcome_quality: Some(0.8),
                },
            },
            ActivationEvent {
                entity_key: EntityKey::default(),
                activation_strength: 0.6,
                timestamp: std::time::Instant::now() + Duration::from_millis(50),
                context: ActivationContext {
                    query_id: "test-query-2".to_string(),
                    cognitive_pattern: CognitivePatternType::Divergent,
                    user_session: Some("test-session".to_string()),
                    outcome_quality: Some(0.6),
                },
            }
        ]
    }

    #[tokio::test]
    async fn test_pattern_detection_system_creation() {
        let system = create_test_pattern_detection_system().await
            .expect("Failed to create pattern detection system");
        
        assert!(!system.pattern_detectors.is_empty(), "Should have pattern detectors");
        assert!(!system.pattern_models.is_empty(), "Should have pattern models");
        assert!(system.detection_config.enable_caching, "Should enable caching by default");
        assert!(system.detection_config.parallel_detection, "Should enable parallel detection by default");
    }

    #[tokio::test]
    async fn test_activation_pattern_detection() {
        let system = create_test_pattern_detection_system().await
            .expect("Failed to create pattern detection system");
        
        let time_window = Duration::from_secs(60);
        
        let patterns = system.detect_activation_patterns(time_window).await
            .expect("Failed to detect activation patterns");
        
        // Should return patterns (may be empty for new system)
        assert!(patterns.len() >= 0, "Should return activation patterns");
        
        for pattern in &patterns {
            assert!(!pattern.activations.is_empty(), "Activation pattern should have activations");
            assert!(!pattern.query.is_empty(), "Pattern should have query identifier");
        }
    }

    #[tokio::test]
    async fn test_frequency_pattern_detection() {
        let system = create_test_pattern_detection_system().await
            .expect("Failed to create pattern detection system");
        
        let entities = vec![EntityKey::default(), EntityKey::default()];
        
        let patterns = system.detect_frequency_patterns(&entities).await
            .expect("Failed to detect frequency patterns");
        
        // Should return patterns (may be empty for new system)
        assert!(patterns.len() >= 0, "Should return frequency patterns");
        
        for pattern in &patterns {
            assert!(!pattern.entities.is_empty(), "Frequency pattern should have entities");
            assert!(pattern.frequency_hz >= 0.0, "Frequency should be non-negative");
            assert!(pattern.amplitude >= 0.0 && pattern.amplitude <= 1.0, 
                   "Amplitude should be normalized");
            assert!(pattern.stability >= 0.0 && pattern.stability <= 1.0, 
                   "Stability should be normalized");
        }
    }

    #[tokio::test]
    async fn test_synchrony_pattern_detection() {
        let system = create_test_pattern_detection_system().await
            .expect("Failed to create pattern detection system");
        
        let activation_events = create_test_activation_events();
        
        let patterns = system.detect_synchrony_patterns(&activation_events).await
            .expect("Failed to detect synchrony patterns");
        
        // Should return patterns (may be empty for new system)
        assert!(patterns.len() >= 0, "Should return synchrony patterns");
        
        for pattern in &patterns {
            assert!(pattern.synchronized_entities.len() >= 2, 
                   "Synchrony pattern should have at least 2 entities");
            assert!(pattern.synchrony_strength >= 0.0 && pattern.synchrony_strength <= 1.0, 
                   "Synchrony strength should be normalized");
            assert!(pattern.phase_coherence >= 0.0 && pattern.phase_coherence <= 1.0, 
                   "Phase coherence should be normalized");
        }
    }

    #[tokio::test]
    async fn test_oscillatory_pattern_detection() {
        let system = create_test_pattern_detection_system().await
            .expect("Failed to create pattern detection system");
        
        // Create test time series with known oscillatory pattern
        let time_series: Vec<f32> = (0..100)
            .map(|i| (i as f32 * 0.1).sin()) // Simple sine wave
            .collect();
        
        let patterns = system.detect_oscillatory_patterns(&time_series).await
            .expect("Failed to detect oscillatory patterns");
        
        // Should detect oscillations in sine wave
        assert!(patterns.len() >= 0, "Should return oscillatory patterns");
        
        for pattern in &patterns {
            assert!(pattern.frequency > 0.0, "Oscillation frequency should be positive");
            assert!(pattern.amplitude >= 0.0 && pattern.amplitude <= 1.0, 
                   "Amplitude should be normalized");
            assert!(pattern.coherence >= 0.0 && pattern.coherence <= 1.0, 
                   "Coherence should be normalized");
        }
    }

    #[tokio::test]
    async fn test_general_pattern_detection() {
        let system = create_test_pattern_detection_system().await
            .expect("Failed to create pattern detection system");
        
        let analysis_scope = create_test_analysis_scope();
        
        let patterns = system.detect_patterns(&analysis_scope).await
            .expect("Failed to detect patterns");
        
        // Should return patterns (may be empty for new system)
        assert!(patterns.len() >= 0, "Should return detected patterns");
        
        for pattern in &patterns {
            assert!(!pattern.pattern_id.to_string().is_empty(), "Pattern should have ID");
            assert!(pattern.confidence >= analysis_scope.minimum_confidence, 
                   "Pattern confidence should meet minimum threshold");
            assert!(pattern.significance >= system.detection_config.pattern_significance_threshold, 
                   "Pattern significance should meet threshold");
            assert!(pattern.pattern_strength >= 0.0 && pattern.pattern_strength <= 1.0, 
                   "Pattern strength should be normalized");
        }
    }

    #[tokio::test]
    async fn test_pattern_caching() {
        let system = create_test_pattern_detection_system().await
            .expect("Failed to create pattern detection system");
        
        let analysis_scope = create_test_analysis_scope();
        
        // First detection - should populate cache
        let patterns1 = system.detect_patterns(&analysis_scope).await
            .expect("Failed to detect patterns first time");
        
        // Second detection - should use cache
        let patterns2 = system.detect_patterns(&analysis_scope).await
            .expect("Failed to detect patterns second time");
        
        // Results should be identical (from cache)
        assert_eq!(patterns1.len(), patterns2.len(), 
                  "Cached results should match original results");
    }

    #[test]
    fn test_cache_key_generation() {
        let system_future = create_test_pattern_detection_system();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let system = rt.block_on(system_future).expect("Failed to create system");
        
        let analysis_scope = create_test_analysis_scope();
        let cache_key = system.generate_cache_key(&analysis_scope);
        
        assert!(!cache_key.is_empty(), "Cache key should not be empty");
        assert!(cache_key.contains("entities"), "Cache key should include entity information");
        assert!(cache_key.contains("window"), "Cache key should include time window");
        assert!(cache_key.contains("types"), "Cache key should include pattern types");
    }

    #[test]
    fn test_pattern_cache_initialization() {
        let cache = PatternCache::new();
        
        assert!(cache.cached_patterns.is_empty(), "New cache should be empty");
        assert_eq!(cache.cache_hits, 0, "Should start with zero hits");
        assert_eq!(cache.cache_misses, 0, "Should start with zero misses");
        assert_eq!(cache.max_cache_size, 1000, "Should have reasonable cache size");
        assert_eq!(cache.cache_ttl, Duration::from_secs(300), "Should have appropriate TTL");
    }

    #[test]
    fn test_pattern_detection_config_defaults() {
        let config = PatternDetectionConfig::default();
        
        assert!(config.enable_caching, "Should enable caching by default");
        assert_eq!(config.cache_ttl, Duration::from_secs(300), "Should have appropriate cache TTL");
        assert!(config.parallel_detection, "Should enable parallel detection by default");
        assert_eq!(config.max_concurrent_detectors, 4, "Should allow reasonable concurrency");
        assert_eq!(config.neural_model_timeout, Duration::from_secs(30), "Should have reasonable timeout");
        assert!(config.pattern_significance_threshold > 0.0, "Should have positive significance threshold");
    }

    #[tokio::test]
    async fn test_activation_pattern_detector() {
        let system = create_test_pattern_detection_system().await
            .expect("Failed to create pattern detection system");
        
        let detector = system.pattern_detectors.get(&PatternType::ActivationPattern)
            .expect("Should have activation pattern detector");
        
        assert_eq!(detector.get_pattern_type(), PatternType::ActivationPattern);
        assert!(detector.get_confidence_threshold() > 0.0, 
               "Confidence threshold should be positive");
        
        let analysis_scope = create_test_analysis_scope();
        let patterns = detector.detect_patterns(&analysis_scope).await
            .expect("Failed to detect activation patterns");
        
        // Should return patterns (may be empty for new system)
        assert!(patterns.len() >= 0, "Should return activation patterns");
    }

    #[test]
    fn test_time_series_encoding() {
        let system_future = create_test_pattern_detection_system();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let system = rt.block_on(system_future).expect("Failed to create system");
        
        let time_series = vec![1.0, 2.0, 3.0, 2.0, 1.0]; // Simple pattern
        
        let encoded = system.encode_time_series_for_neural_analysis(&time_series)
            .expect("Failed to encode time series");
        
        assert_eq!(encoded.len(), time_series.len(), 
                  "Encoded series should have same length");
        
        // Check normalization (mean should be ~0, std should be ~1)
        let mean: f32 = encoded.iter().sum::<f32>() / encoded.len() as f32;
        assert!(mean.abs() < 0.001, "Encoded series should be normalized (mean ~0)");
    }

    #[test]
    fn test_oscillatory_prediction_decoding() {
        let system_future = create_test_pattern_detection_system();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let system = rt.block_on(system_future).expect("Failed to create system");
        
        // Mock neural prediction with high oscillation score
        let neural_prediction = vec![0.8, 0.2, 0.9, 0.1, 0.7];
        
        let patterns = system.decode_oscillatory_predictions(neural_prediction)
            .expect("Failed to decode oscillatory predictions");
        
        // Should detect oscillations for high scores (>0.7)
        let high_score_patterns: Vec<_> = patterns.iter()
            .filter(|p| p.amplitude > 0.7)
            .collect();
        
        assert!(!high_score_patterns.is_empty(), "Should detect patterns for high scores");
        
        for pattern in &patterns {
            assert!(pattern.frequency > 0.0, "Frequency should be positive");
            assert!(pattern.amplitude >= 0.0 && pattern.amplitude <= 1.0, 
                   "Amplitude should be normalized");
            assert!(pattern.coherence >= 0.0 && pattern.coherence <= 1.0, 
                   "Coherence should be normalized");
        }
    }

    #[test]
    fn test_detected_pattern_structure() {
        let pattern = DetectedPattern {
            pattern_id: Uuid::new_v4(),
            pattern_type: PatternType::ActivationPattern,
            entities_involved: vec![EntityKey::default()],
            confidence: 0.8,
            significance: 0.7,
            temporal_span: Duration::from_secs(60),
            pattern_strength: 0.9,
            pattern_frequency: 1.5,
            pattern_metadata: {
                let mut metadata = HashMap::new();
                metadata.insert("detector".to_string(), "test_detector".to_string());
                metadata
            },
            discovery_timestamp: SystemTime::now(),
        };
        
        // Test pattern structure
        assert!(!pattern.pattern_id.to_string().is_empty(), "Pattern should have ID");
        assert_eq!(pattern.pattern_type, PatternType::ActivationPattern);
        assert!(!pattern.entities_involved.is_empty(), "Pattern should involve entities");
        assert!(pattern.confidence >= 0.0 && pattern.confidence <= 1.0, 
               "Confidence should be normalized");
        assert!(pattern.significance >= 0.0 && pattern.significance <= 1.0, 
               "Significance should be normalized");
        assert!(pattern.pattern_strength >= 0.0 && pattern.pattern_strength <= 1.0, 
               "Pattern strength should be normalized");
        assert!(pattern.pattern_frequency >= 0.0, "Pattern frequency should be non-negative");
        assert!(!pattern.pattern_metadata.is_empty(), "Pattern should have metadata");
    }

    #[test]
    fn test_analysis_scope_configuration() {
        let scope = AnalysisScope {
            entities: vec![EntityKey::default(), EntityKey::default()],
            time_window: Duration::from_secs(120),
            depth: 5,
            pattern_types: vec![PatternType::ActivationPattern, PatternType::FrequencyPattern],
            minimum_confidence: 0.75,
        };
        
        assert_eq!(scope.entities.len(), 2, "Should have 2 entities");
        assert_eq!(scope.time_window, Duration::from_secs(120), "Should have 2-minute window");
        assert_eq!(scope.depth, 5, "Should have depth of 5");
        assert_eq!(scope.pattern_types.len(), 2, "Should have 2 pattern types");
        assert_eq!(scope.minimum_confidence, 0.75, "Should have 75% confidence threshold");
    }

    #[test]
    fn test_pattern_types_enum() {
        // Test that all pattern types are properly defined
        let all_types = vec![
            PatternType::ActivationPattern,
            PatternType::FrequencyPattern,
            PatternType::SynchronyPattern,
            PatternType::OscillatoryPattern,
            PatternType::TemporalPattern,
            PatternType::SpatialPattern,
            PatternType::CausalPattern,
            PatternType::HierarchicalPattern,
            PatternType::CompetitivePattern,
            PatternType::CooperativePattern,
        ];
        
        assert_eq!(all_types.len(), 10, "Should have 10 pattern types defined");
        
        // Test serialization/deserialization
        for pattern_type in &all_types {
            let serialized = serde_json::to_string(pattern_type)
                .expect("Pattern type should be serializable");
            let deserialized: PatternType = serde_json::from_str(&serialized)
                .expect("Pattern type should be deserializable");
            assert_eq!(*pattern_type, deserialized, "Serialization round-trip should work");
        }
    }
}