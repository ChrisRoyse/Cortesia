# Task 13: Neuromorphic Concept Structure

## Metadata
- **Micro-Phase**: 2.13
- **Duration**: 15-20 minutes
- **Dependencies**: Task 05 (ValidatedFact), Task 12 (TTFSSpikePattern)
- **Output**: `src/ttfs_encoding/neuromorphic_concept.rs`

## Description
Create the NeuromorphicConcept structure that bridges validated facts from Phase 0A to TTFS spike patterns, providing the semantic and contextual information needed for neuromorphic encoding.

## Test Requirements
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::quality_integration::{ValidatedFact, FactContent, ConfidenceComponents};
    use crate::ttfs_encoding::ConceptId;

    #[test]
    fn test_neuromorphic_concept_creation() {
        let concept = NeuromorphicConcept::new(
            "elephant",
            ConceptType::Entity,
            vec![0.8, 0.6, 0.9],
            Some("Large African mammal with trunk".to_string())
        );
        
        assert_eq!(concept.name(), "elephant");
        assert_eq!(concept.concept_type(), &ConceptType::Entity);
        assert_eq!(concept.semantic_features().len(), 3);
        assert!(concept.description().is_some());
    }
    
    #[test]
    fn test_concept_from_validated_fact() {
        let content = FactContent::new("Dolphins are intelligent marine mammals");
        let confidence = ConfidenceComponents::new(0.95, 0.90, 0.88);
        let fact = ValidatedFact::new(content, confidence, vec!["syntax", "semantic", "logical"]);
        
        let concept = NeuromorphicConcept::from_validated_fact(&fact);
        
        assert_eq!(concept.name(), "Dolphins are intelligent marine mammals");
        assert!(concept.semantic_features().len() > 0);
        assert!(concept.confidence() > 0.8);
    }
    
    #[test]
    fn test_semantic_feature_extraction() {
        let concept = NeuromorphicConcept::new(
            "artificial intelligence",
            ConceptType::AbstractConcept,
            vec![0.9, 0.8, 0.7, 0.6],
            Some("Computer systems that perform tasks requiring human intelligence".to_string())
        );
        
        let features = concept.semantic_features();
        assert_eq!(features.len(), 4);
        assert!(features.iter().all(|&f| f >= 0.0 && f <= 1.0));
        
        let normalized = concept.get_normalized_features();
        let norm: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001); // Should be unit normalized
    }
    
    #[test]
    fn test_temporal_context_handling() {
        let mut concept = NeuromorphicConcept::new(
            "event",
            ConceptType::Event,
            vec![0.8, 0.7],
            None
        );
        
        concept.set_temporal_context(Some(Duration::from_millis(100)));
        assert!(concept.has_temporal_context());
        assert_eq!(concept.temporal_context(), Some(Duration::from_millis(100)));
        
        let temporal_features = concept.extract_temporal_features();
        assert!(!temporal_features.is_empty());
    }
    
    #[test]
    fn test_concept_similarity() {
        let concept1 = NeuromorphicConcept::new(
            "cat",
            ConceptType::Entity,
            vec![0.8, 0.6, 0.9, 0.7],
            Some("Domestic feline".to_string())
        );
        
        let concept2 = NeuromorphicConcept::new(
            "dog",
            ConceptType::Entity,
            vec![0.7, 0.8, 0.8, 0.6],
            Some("Domestic canine".to_string())
        );
        
        let concept3 = NeuromorphicConcept::new(
            "car",
            ConceptType::Entity,
            vec![0.1, 0.2, 0.1, 0.3],
            Some("Vehicle".to_string())
        );
        
        let similarity_cat_dog = concept1.semantic_similarity(&concept2);
        let similarity_cat_car = concept1.semantic_similarity(&concept3);
        
        assert!(similarity_cat_dog > similarity_cat_car);
        assert!(similarity_cat_dog > 0.5);
        assert!(similarity_cat_car < 0.5);
    }
    
    #[test]
    fn test_concept_hierarchy_relationships() {
        let mut animal_concept = NeuromorphicConcept::new(
            "animal",
            ConceptType::Category,
            vec![0.9, 0.8, 0.7],
            Some("Living organism".to_string())
        );
        
        let mammal_concept = NeuromorphicConcept::new(
            "mammal",
            ConceptType::Category,
            vec![0.9, 0.8, 0.9],
            Some("Warm-blooded vertebrate".to_string())
        );
        
        animal_concept.add_child_concept(mammal_concept.id().clone());
        assert!(animal_concept.has_children());
        assert_eq!(animal_concept.child_concepts().len(), 1);
        
        let relationship_strength = animal_concept.calculate_hierarchical_strength(&mammal_concept);
        assert!(relationship_strength > 0.0);
    }
    
    #[test]
    fn test_encoding_readiness() {
        let ready_concept = NeuromorphicConcept::new(
            "ready_concept",
            ConceptType::Entity,
            vec![0.9, 0.8, 0.7, 0.6],
            Some("Well-defined concept".to_string())
        );
        
        assert!(ready_concept.is_ready_for_encoding());
        
        let unready_concept = NeuromorphicConcept::new(
            "",
            ConceptType::Unknown,
            vec![],
            None
        );
        
        assert!(!unready_concept.is_ready_for_encoding());
    }
}
```

## Implementation
```rust
use crate::quality_integration::ValidatedFact;
use crate::ttfs_encoding::ConceptId;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use std::collections::{HashMap, HashSet};

/// Type classification for neuromorphic concepts
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConceptType {
    /// Concrete entities (people, objects, places)
    Entity,
    /// Abstract concepts (ideas, emotions, qualities)
    AbstractConcept,
    /// Events or actions
    Event,
    /// Categories or classifications
    Category,
    /// Relationships between entities
    Relationship,
    /// Properties or attributes
    Property,
    /// Temporal concepts (time, duration, sequence)
    Temporal,
    /// Unknown or unclassified
    Unknown,
}

impl ConceptType {
    /// Get type-specific encoding parameters
    pub fn encoding_parameters(&self) -> EncodingParameters {
        match self {
            ConceptType::Entity => EncodingParameters {
                base_frequency: 50.0,
                amplitude_scale: 1.0,
                temporal_scaling: 1.0,
                spike_density: 0.8,
            },
            ConceptType::AbstractConcept => EncodingParameters {
                base_frequency: 30.0,
                amplitude_scale: 0.8,
                temporal_scaling: 1.2,
                spike_density: 0.6,
            },
            ConceptType::Event => EncodingParameters {
                base_frequency: 70.0,
                amplitude_scale: 1.1,
                temporal_scaling: 0.8,
                spike_density: 1.0,
            },
            ConceptType::Category => EncodingParameters {
                base_frequency: 40.0,
                amplitude_scale: 0.9,
                temporal_scaling: 1.1,
                spike_density: 0.7,
            },
            ConceptType::Relationship => EncodingParameters {
                base_frequency: 60.0,
                amplitude_scale: 0.85,
                temporal_scaling: 0.9,
                spike_density: 0.9,
            },
            ConceptType::Property => EncodingParameters {
                base_frequency: 45.0,
                amplitude_scale: 0.75,
                temporal_scaling: 1.0,
                spike_density: 0.5,
            },
            ConceptType::Temporal => EncodingParameters {
                base_frequency: 80.0,
                amplitude_scale: 1.2,
                temporal_scaling: 0.7,
                spike_density: 1.2,
            },
            ConceptType::Unknown => EncodingParameters {
                base_frequency: 35.0,
                amplitude_scale: 0.7,
                temporal_scaling: 1.0,
                spike_density: 0.5,
            },
        }
    }
}

/// Encoding parameters specific to concept types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodingParameters {
    pub base_frequency: f32,     // Hz
    pub amplitude_scale: f32,    // Multiplier for spike amplitudes
    pub temporal_scaling: f32,   // Time scale factor
    pub spike_density: f32,      // Spikes per unit time
}

/// Neuromorphic concept representation for TTFS encoding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuromorphicConcept {
    /// Unique identifier
    id: ConceptId,
    
    /// Human-readable name
    name: String,
    
    /// Type classification
    concept_type: ConceptType,
    
    /// Semantic feature vector for encoding
    semantic_features: Vec<f32>,
    
    /// Optional textual description
    description: Option<String>,
    
    /// Confidence score from validation (0.0-1.0)
    confidence: f32,
    
    /// Temporal context if applicable
    temporal_context: Option<Duration>,
    
    /// Hierarchical relationships
    parent_concepts: Vec<ConceptId>,
    child_concepts: Vec<ConceptId>,
    
    /// Associated properties
    properties: HashMap<String, f32>,
    
    /// Encoding metadata
    encoding_metadata: HashMap<String, String>,
    
    /// Activation level for spreading activation
    activation_level: f32,
}

impl NeuromorphicConcept {
    /// Create new neuromorphic concept
    pub fn new(
        name: impl Into<String>,
        concept_type: ConceptType,
        semantic_features: Vec<f32>,
        description: Option<String>,
    ) -> Self {
        let name = name.into();
        let id = ConceptId::new(format!("concept_{}", name.to_lowercase().replace(' ', "_")));
        
        Self {
            id,
            name,
            concept_type,
            semantic_features,
            description,
            confidence: 1.0,
            temporal_context: None,
            parent_concepts: Vec::new(),
            child_concepts: Vec::new(),
            properties: HashMap::new(),
            encoding_metadata: HashMap::new(),
            activation_level: 0.0,
        }
    }
    
    /// Create neuromorphic concept from validated fact
    pub fn from_validated_fact(fact: &ValidatedFact) -> Self {
        let name = fact.content().text().to_string();
        let confidence = fact.quality_score();
        
        // Extract semantic features from fact content
        let semantic_features = Self::extract_semantic_features_from_fact(fact);
        
        // Classify concept type based on content
        let concept_type = Self::classify_concept_type(fact);
        
        let mut concept = Self::new(
            name,
            concept_type,
            semantic_features,
            Some(fact.content().text().to_string()),
        );
        
        concept.confidence = confidence;
        
        // Add entities as properties
        for entity in fact.content().entities() {
            concept.properties.insert(
                format!("entity_{}", entity.name),
                entity.confidence,
            );
        }
        
        // Add relationships as metadata
        for (i, relationship) in fact.content().relationships().iter().enumerate() {
            concept.encoding_metadata.insert(
                format!("relationship_{}", i),
                format!("{}:{}:{}", relationship.subject, relationship.predicate, relationship.object),
            );
        }
        
        concept
    }
    
    /// Get concept identifier
    pub fn id(&self) -> &ConceptId {
        &self.id
    }
    
    /// Get concept name
    pub fn name(&self) -> &str {
        &self.name
    }
    
    /// Get concept type
    pub fn concept_type(&self) -> &ConceptType {
        &self.concept_type
    }
    
    /// Get semantic features
    pub fn semantic_features(&self) -> &[f32] {
        &self.semantic_features
    }
    
    /// Get description
    pub fn description(&self) -> Option<&String> {
        self.description.as_ref()
    }
    
    /// Get confidence score
    pub fn confidence(&self) -> f32 {
        self.confidence
    }
    
    /// Get temporal context
    pub fn temporal_context(&self) -> Option<Duration> {
        self.temporal_context
    }
    
    /// Check if concept has temporal context
    pub fn has_temporal_context(&self) -> bool {
        self.temporal_context.is_some()
    }
    
    /// Set temporal context
    pub fn set_temporal_context(&mut self, context: Option<Duration>) {
        self.temporal_context = context;
    }
    
    /// Get normalized semantic features (unit vector)
    pub fn get_normalized_features(&self) -> Vec<f32> {
        if self.semantic_features.is_empty() {
            return vec![];
        }
        
        let norm: f32 = self.semantic_features.iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        
        if norm == 0.0 {
            return self.semantic_features.clone();
        }
        
        self.semantic_features.iter()
            .map(|x| x / norm)
            .collect()
    }
    
    /// Extract temporal features for encoding
    pub fn extract_temporal_features(&self) -> Vec<f32> {
        let mut features = Vec::new();
        
        if let Some(duration) = self.temporal_context {
            features.push(duration.as_millis() as f32 / 1000.0); // Duration in seconds
            features.push(1.0); // Has temporal context flag
        } else {
            features.push(0.0); // No duration
            features.push(0.0); // No temporal context flag
        }
        
        // Add type-specific temporal characteristics
        match self.concept_type {
            ConceptType::Event => {
                features.push(1.0); // Event temporality
                features.push(0.8); // High temporal relevance
            },
            ConceptType::Temporal => {
                features.push(0.5); // Temporal concept
                features.push(1.0); // Maximum temporal relevance
            },
            _ => {
                features.push(0.0); // Non-temporal
                features.push(0.2); // Low temporal relevance
            },
        }
        
        features
    }
    
    /// Calculate semantic similarity with another concept
    pub fn semantic_similarity(&self, other: &NeuromorphicConcept) -> f32 {
        let features1 = self.get_normalized_features();
        let features2 = other.get_normalized_features();
        
        if features1.is_empty() || features2.is_empty() {
            return 0.0;
        }
        
        let min_len = features1.len().min(features2.len());
        
        // Cosine similarity
        let dot_product: f32 = features1.iter()
            .zip(features2.iter())
            .take(min_len)
            .map(|(a, b)| a * b)
            .sum();
        
        dot_product.clamp(0.0, 1.0)
    }
    
    /// Add parent concept relationship
    pub fn add_parent_concept(&mut self, parent_id: ConceptId) {
        if !self.parent_concepts.contains(&parent_id) {
            self.parent_concepts.push(parent_id);
        }
    }
    
    /// Add child concept relationship
    pub fn add_child_concept(&mut self, child_id: ConceptId) {
        if !self.child_concepts.contains(&child_id) {
            self.child_concepts.push(child_id);
        }
    }
    
    /// Get parent concepts
    pub fn parent_concepts(&self) -> &[ConceptId] {
        &self.parent_concepts
    }
    
    /// Get child concepts
    pub fn child_concepts(&self) -> &[ConceptId] {
        &self.child_concepts
    }
    
    /// Check if concept has children
    pub fn has_children(&self) -> bool {
        !self.child_concepts.is_empty()
    }
    
    /// Calculate hierarchical relationship strength with another concept
    pub fn calculate_hierarchical_strength(&self, other: &NeuromorphicConcept) -> f32 {
        // Direct parent-child relationship
        if self.child_concepts.contains(other.id()) || 
           other.child_concepts.contains(self.id()) {
            return 0.9;
        }
        
        // Shared parent
        let self_parents: HashSet<_> = self.parent_concepts.iter().collect();
        let other_parents: HashSet<_> = other.parent_concepts.iter().collect();
        let shared_parents = self_parents.intersection(&other_parents).count();
        
        if shared_parents > 0 {
            return 0.7 * (shared_parents as f32 / self_parents.len().max(1) as f32);
        }
        
        // Type-based relationship
        let type_similarity = match (&self.concept_type, &other.concept_type) {
            (ConceptType::Entity, ConceptType::Entity) => 0.5,
            (ConceptType::Category, ConceptType::Entity) => 0.6,
            (ConceptType::Entity, ConceptType::Category) => 0.6,
            (a, b) if a == b => 0.4,
            _ => 0.1,
        };
        
        type_similarity
    }
    
    /// Check if concept is ready for TTFS encoding
    pub fn is_ready_for_encoding(&self) -> bool {
        !self.name.is_empty() &&
        !self.semantic_features.is_empty() &&
        self.concept_type != ConceptType::Unknown &&
        self.confidence > 0.5
    }
    
    /// Get encoding parameters based on concept type
    pub fn get_encoding_parameters(&self) -> EncodingParameters {
        self.concept_type.encoding_parameters()
    }
    
    /// Add property to concept
    pub fn add_property(&mut self, name: impl Into<String>, value: f32) {
        self.properties.insert(name.into(), value);
    }
    
    /// Get properties
    pub fn properties(&self) -> &HashMap<String, f32> {
        &self.properties
    }
    
    /// Set activation level
    pub fn set_activation_level(&mut self, level: f32) {
        self.activation_level = level.clamp(0.0, 1.0);
    }
    
    /// Get activation level
    pub fn activation_level(&self) -> f32 {
        self.activation_level
    }
    
    /// Extract semantic features from validated fact
    fn extract_semantic_features_from_fact(fact: &ValidatedFact) -> Vec<f32> {
        let mut features = Vec::new();
        
        let confidence_components = fact.confidence_components();
        
        // Basic confidence features
        features.push(confidence_components.syntax_confidence);
        features.push(confidence_components.entity_confidence);
        features.push(confidence_components.semantic_confidence);
        features.push(confidence_components.logical_confidence);
        
        // Content-based features
        let text = fact.content().text();
        features.push((text.len() as f32).log10() / 3.0); // Text length (log-scaled)
        features.push(fact.content().entities().len() as f32 / 10.0); // Entity count
        features.push(fact.content().relationships().len() as f32 / 5.0); // Relationship count
        
        // Validation features
        features.push(fact.validation_chain().len() as f32 / 4.0); // Validation completeness
        features.push(if fact.is_fully_validated() { 1.0 } else { 0.0 });
        features.push(if fact.is_ambiguity_resolved() { 1.0 } else { 0.0 });
        
        // Pad or truncate to standard size (16 features)
        features.resize(16, 0.0);
        
        features
    }
    
    /// Classify concept type from validated fact
    fn classify_concept_type(fact: &ValidatedFact) -> ConceptType {
        let text = fact.content().text().to_lowercase();
        let entities = fact.content().entities();
        
        // Rule-based classification
        if text.contains("is a") || text.contains("are") {
            ConceptType::Category
        } else if text.contains("when") || text.contains("then") || text.contains("during") {
            ConceptType::Event
        } else if entities.len() >= 2 {
            ConceptType::Relationship
        } else if entities.len() == 1 {
            ConceptType::Entity
        } else if text.len() < 50 && text.split_whitespace().count() < 8 {
            ConceptType::Property
        } else {
            ConceptType::AbstractConcept
        }
    }
    
    /// Create a test concept for validation
    pub fn create_test_concept(name: &str, concept_type: ConceptType) -> Self {
        let features = match concept_type {
            ConceptType::Entity => vec![0.8, 0.6, 0.9, 0.7],
            ConceptType::AbstractConcept => vec![0.5, 0.8, 0.3, 0.9],
            ConceptType::Event => vec![0.9, 0.4, 0.7, 0.8],
            _ => vec![0.6, 0.6, 0.6, 0.6],
        };
        
        Self::new(
            name,
            concept_type,
            features,
            Some(format!("Test concept: {}", name)),
        )
    }
}

impl PartialEq for NeuromorphicConcept {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl std::hash::Hash for NeuromorphicConcept {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}
```

## Verification Steps
1. Create NeuromorphicConcept structure with semantic features
2. Implement conversion from ValidatedFact to concept
3. Add hierarchical relationship support
4. Implement semantic similarity calculation
5. Ensure all tests pass

## Success Criteria
- [ ] NeuromorphicConcept struct compiles without errors
- [ ] Conversion from ValidatedFact works correctly
- [ ] Semantic similarity calculation produces meaningful results
- [ ] Hierarchical relationships are properly managed
- [ ] All tests pass