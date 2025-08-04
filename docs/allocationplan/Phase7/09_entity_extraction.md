# Micro Task 09: Entity Extraction

**Priority**: CRITICAL  
**Estimated Time**: 45 minutes  
**Dependencies**: 08_intent_parser_llm.md  
**Skills Required**: NLP entity recognition, pattern matching

## Objective

Implement comprehensive entity extraction and classification from natural language queries to identify organisms, concepts, locations, objects, and attributes.

## Context

Entity extraction is a critical component of query understanding that identifies and classifies the key nouns, concepts, and entities mentioned in user queries. This enables precise activation of relevant graph nodes and improves query processing accuracy.

## Specifications

### Entity Classification System

1. **Primary Entity Types**
   - ORGANISM (animals, plants, microorganisms, biological entities)
   - CONCEPT (abstract ideas, processes, theories, properties)
   - LOCATION (geographic places, environments, habitats)
   - OBJECT (physical things, substances, materials)
   - ATTRIBUTE (characteristics, properties, measurements)

2. **Extraction Methods**
   - Pattern-based recognition (fast path)
   - LLM-enhanced extraction (complex cases)
   - Hybrid approach with confidence scoring
   - Context-aware disambiguation

3. **Output Requirements**
   - Text span identification
   - Entity type classification
   - Confidence scoring
   - Alias and synonym detection

## Implementation Guide

### Step 1: Core Entity Types
```rust
// File: src/query/entity_extraction.rs

use std::collections::HashMap;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EntityType {
    Organism,
    Concept,
    Location,
    Object,
    Attribute,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEntity {
    pub text: String,
    pub entity_type: EntityType,
    pub start_pos: usize,
    pub end_pos: usize,
    pub confidence: f32,
    pub aliases: Vec<String>,
    pub context_clues: Vec<String>,
    pub modifiers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityExtractionResult {
    pub entities: Vec<ExtractedEntity>,
    pub extraction_method: ExtractionMethod,
    pub processing_time_ms: u64,
    pub confidence_distribution: HashMap<EntityType, f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExtractionMethod {
    PatternOnly,
    LLMOnly,
    HybridPatternLLM,
    ContextEnhanced,
}
```

### Step 2: Pattern-Based Extraction
```rust
pub struct PatternEntityExtractor {
    organism_patterns: Vec<EntityPattern>,
    concept_patterns: Vec<EntityPattern>,
    location_patterns: Vec<EntityPattern>,
    object_patterns: Vec<EntityPattern>,
    attribute_patterns: Vec<EntityPattern>,
    modifier_patterns: Vec<ModifierPattern>,
}

#[derive(Debug, Clone)]
pub struct EntityPattern {
    pub pattern: regex::Regex,
    pub entity_type: EntityType,
    pub confidence_base: f32,
    pub context_boost: HashMap<String, f32>,
}

#[derive(Debug, Clone)]
pub struct ModifierPattern {
    pub pattern: regex::Regex,
    pub modifier_type: ModifierType,
    pub position: ModifierPosition,
}

#[derive(Debug, Clone)]
pub enum ModifierType {
    Size,      // large, small, tiny
    Color,     // red, blue, green
    Quantity,  // many, few, several
    Quality,   // good, bad, excellent
    Temporal,  // recent, old, ancient
}

#[derive(Debug, Clone)]
pub enum ModifierPosition {
    Before,
    After,
    Either,
}

impl PatternEntityExtractor {
    pub fn new() -> Self {
        Self {
            organism_patterns: Self::create_organism_patterns(),
            concept_patterns: Self::create_concept_patterns(),
            location_patterns: Self::create_location_patterns(),
            object_patterns: Self::create_object_patterns(),
            attribute_patterns: Self::create_attribute_patterns(),
            modifier_patterns: Self::create_modifier_patterns(),
        }
    }
    
    fn create_organism_patterns() -> Vec<EntityPattern> {
        vec![
            EntityPattern {
                pattern: regex::Regex::new(r"\b(?:animals?|mammals?|birds?|fish|reptiles?|amphibians?|insects?)\b").unwrap(),
                entity_type: EntityType::Organism,
                confidence_base: 0.9,
                context_boost: HashMap::from([
                    ("species".to_string(), 0.1),
                    ("biology".to_string(), 0.1),
                ]),
            },
            EntityPattern {
                pattern: regex::Regex::new(r"\b(?:plants?|trees?|flowers?|grass|fungi|bacteria|viruses?)\b").unwrap(),
                entity_type: EntityType::Organism,
                confidence_base: 0.85,
                context_boost: HashMap::new(),
            },
            EntityPattern {
                pattern: regex::Regex::new(r"\b(?:dogs?|cats?|horses?|cows?|birds?|lions?|tigers?|wolves?|bears?)\b").unwrap(),
                entity_type: EntityType::Organism,
                confidence_base: 0.95,
                context_boost: HashMap::new(),
            },
        ]
    }
    
    fn create_concept_patterns() -> Vec<EntityPattern> {
        vec![
            EntityPattern {
                pattern: regex::Regex::new(r"\b(?:evolution|adaptation|migration|photosynthesis|metabolism|reproduction)\b").unwrap(),
                entity_type: EntityType::Concept,
                confidence_base: 0.9,
                context_boost: HashMap::new(),
            },
            EntityPattern {
                pattern: regex::Regex::new(r"\b(?:behavior|intelligence|consciousness|learning|memory|communication)\b").unwrap(),
                entity_type: EntityType::Concept,
                confidence_base: 0.85,
                context_boost: HashMap::new(),
            },
        ]
    }
    
    fn create_location_patterns() -> Vec<EntityPattern> {
        vec![
            EntityPattern {
                pattern: regex::Regex::new(r"\b(?:forest|ocean|desert|mountain|river|lake|habitat|ecosystem)\b").unwrap(),
                entity_type: EntityType::Location,
                confidence_base: 0.85,
                context_boost: HashMap::new(),
            },
            EntityPattern {
                pattern: regex::Regex::new(r"\b(?:Africa|Asia|Europe|Americas?|Antarctica|Australia)\b").unwrap(),
                entity_type: EntityType::Location,
                confidence_base: 0.95,
                context_boost: HashMap::new(),
            },
        ]
    }
    
    pub fn extract_entities(&self, text: &str) -> Vec<ExtractedEntity> {
        let mut entities = Vec::new();
        let text_lower = text.to_lowercase();
        
        // Extract each entity type
        entities.extend(self.extract_by_patterns(&text_lower, &self.organism_patterns));
        entities.extend(self.extract_by_patterns(&text_lower, &self.concept_patterns));
        entities.extend(self.extract_by_patterns(&text_lower, &self.location_patterns));
        entities.extend(self.extract_by_patterns(&text_lower, &self.object_patterns));
        entities.extend(self.extract_by_patterns(&text_lower, &self.attribute_patterns));
        
        // Remove overlaps and enhance with modifiers
        let entities = self.resolve_overlaps(entities);
        self.add_modifiers(&text_lower, entities)
    }
    
    fn extract_by_patterns(&self, text: &str, patterns: &[EntityPattern]) -> Vec<ExtractedEntity> {
        let mut entities = Vec::new();
        
        for pattern in patterns {
            for mat in pattern.pattern.find_iter(text) {
                let entity = ExtractedEntity {
                    text: mat.as_str().to_string(),
                    entity_type: pattern.entity_type.clone(),
                    start_pos: mat.start(),
                    end_pos: mat.end(),
                    confidence: pattern.confidence_base,
                    aliases: Vec::new(),
                    context_clues: Vec::new(),
                    modifiers: Vec::new(),
                };
                entities.push(entity);
            }
        }
        
        entities
    }
    
    fn resolve_overlaps(&self, mut entities: Vec<ExtractedEntity>) -> Vec<ExtractedEntity> {
        entities.sort_by_key(|e| e.start_pos);
        
        let mut resolved = Vec::new();
        for entity in entities {
            // Check for overlaps with existing entities
            if !resolved.iter().any(|existing: &ExtractedEntity| {
                entity.start_pos < existing.end_pos && entity.end_pos > existing.start_pos
            }) {
                resolved.push(entity);
            }
        }
        
        resolved
    }
    
    fn add_modifiers(&self, text: &str, mut entities: Vec<ExtractedEntity>) -> Vec<ExtractedEntity> {
        for entity in &mut entities {
            // Look for modifiers before and after each entity
            let before_text = &text[..entity.start_pos];
            let after_text = &text[entity.end_pos..];
            
            for modifier_pattern in &self.modifier_patterns {
                match modifier_pattern.position {
                    ModifierPosition::Before | ModifierPosition::Either => {
                        if let Some(words) = before_text.split_whitespace().last() {
                            if modifier_pattern.pattern.is_match(words) {
                                entity.modifiers.push(words.to_string());
                            }
                        }
                    }
                    ModifierPosition::After | ModifierPosition::Either => {
                        if let Some(words) = after_text.split_whitespace().next() {
                            if modifier_pattern.pattern.is_match(words) {
                                entity.modifiers.push(words.to_string());
                            }
                        }
                    }
                }
            }
        }
        
        entities
    }
}
```

### Step 3: LLM-Enhanced Extraction
```rust
pub struct LLMEntityExtractor {
    llm: Arc<dyn LanguageModel + Send + Sync>,
    prompt_template: String,
}

impl LLMEntityExtractor {
    pub fn new(llm: Arc<dyn LanguageModel + Send + Sync>) -> Self {
        Self {
            llm,
            prompt_template: Self::create_extraction_prompt_template(),
        }
    }
    
    fn create_extraction_prompt_template() -> String {
        r#"Extract and classify all entities from the following text. Focus on identifying:

1. ORGANISM - Living things: animals, plants, microorganisms, biological entities
2. CONCEPT - Abstract ideas: processes, theories, properties, behaviors
3. LOCATION - Places: geographic locations, environments, habitats  
4. OBJECT - Physical things: substances, materials, tools
5. ATTRIBUTE - Characteristics: properties, measurements, qualities

Text: "{text}"

For each entity found, provide:
- The exact text span
- Classification (ORGANISM/CONCEPT/LOCATION/OBJECT/ATTRIBUTE)
- Start and end character positions
- Confidence score (0.0-1.0)
- Any modifiers or qualifiers

Response format (JSON):
{{
  "entities": [
    {{
      "text": "entity_name",
      "type": "ORGANISM",
      "start": 0,
      "end": 10,
      "confidence": 0.95,
      "modifiers": ["large", "african"],
      "context_clues": ["habitat", "species"]
    }}
  ]
}}

Response:"#.to_string()
    }
    
    pub async fn extract_entities(&self, text: &str) -> Result<Vec<ExtractedEntity>> {
        let prompt = self.prompt_template.replace("{text}", text);
        
        let response = self.llm.generate(&prompt).await
            .map_err(|e| Error::LLMExtractionFailed(e.to_string()))?;
        
        self.parse_llm_response(&response)
    }
    
    fn parse_llm_response(&self, response: &str) -> Result<Vec<ExtractedEntity>> {
        #[derive(Deserialize)]
        struct LLMEntityResponse {
            entities: Vec<LLMEntity>,
        }
        
        #[derive(Deserialize)]
        struct LLMEntity {
            text: String,
            #[serde(rename = "type")]
            entity_type: String,
            start: usize,
            end: usize,
            confidence: f32,
            modifiers: Option<Vec<String>>,
            context_clues: Option<Vec<String>>,
        }
        
        let parsed: LLMEntityResponse = serde_json::from_str(response)
            .map_err(|e| Error::ResponseParsing(e.to_string()))?;
        
        let mut entities = Vec::new();
        for llm_entity in parsed.entities {
            let entity_type = match llm_entity.entity_type.as_str() {
                "ORGANISM" => EntityType::Organism,
                "CONCEPT" => EntityType::Concept,
                "LOCATION" => EntityType::Location,
                "OBJECT" => EntityType::Object,
                "ATTRIBUTE" => EntityType::Attribute,
                _ => EntityType::Unknown,
            };
            
            entities.push(ExtractedEntity {
                text: llm_entity.text,
                entity_type,
                start_pos: llm_entity.start,
                end_pos: llm_entity.end,
                confidence: llm_entity.confidence.clamp(0.0, 1.0),
                aliases: Vec::new(),
                context_clues: llm_entity.context_clues.unwrap_or_default(),
                modifiers: llm_entity.modifiers.unwrap_or_default(),
            });
        }
        
        Ok(entities)
    }
}
```

### Step 4: Hybrid Entity Extractor
```rust
pub struct HybridEntityExtractor {
    pattern_extractor: PatternEntityExtractor,
    llm_extractor: LLMEntityExtractor,
    config: ExtractionConfig,
}

#[derive(Debug, Clone)]
pub struct ExtractionConfig {
    pub use_llm_when_pattern_confidence_below: f32,
    pub merge_strategy: MergeStrategy,
    pub max_entities_per_query: usize,
    pub min_entity_confidence: f32,
}

#[derive(Debug, Clone)]
pub enum MergeStrategy {
    PatternPriority,    // Prefer pattern matches
    LLMPriority,        // Prefer LLM extractions
    HighestConfidence,  // Choose highest confidence
    Ensemble,           // Combine both with weighting
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            use_llm_when_pattern_confidence_below: 0.7,
            merge_strategy: MergeStrategy::HighestConfidence,
            max_entities_per_query: 10,
            min_entity_confidence: 0.5,
        }
    }
}

impl HybridEntityExtractor {
    pub fn new(llm: Arc<dyn LanguageModel + Send + Sync>) -> Self {
        Self {
            pattern_extractor: PatternEntityExtractor::new(),
            llm_extractor: LLMEntityExtractor::new(llm),
            config: ExtractionConfig::default(),
        }
    }
    
    pub async fn extract_entities(&self, text: &str) -> Result<EntityExtractionResult> {
        let start_time = std::time::Instant::now();
        
        // First pass: pattern-based extraction
        let pattern_entities = self.pattern_extractor.extract_entities(text);
        
        // Decide if LLM enhancement is needed
        let needs_llm = self.should_use_llm(&pattern_entities);
        
        let (final_entities, method) = if needs_llm {
            // Second pass: LLM extraction
            let llm_entities = self.llm_extractor.extract_entities(text).await?;
            
            // Merge results
            let merged = self.merge_extractions(pattern_entities, llm_entities)?;
            (merged, ExtractionMethod::HybridPatternLLM)
        } else {
            (pattern_entities, ExtractionMethod::PatternOnly)
        };
        
        // Filter and validate
        let filtered_entities = self.filter_and_validate(final_entities)?;
        
        // Calculate confidence distribution
        let confidence_dist = self.calculate_confidence_distribution(&filtered_entities);
        
        Ok(EntityExtractionResult {
            entities: filtered_entities,
            extraction_method: method,
            processing_time_ms: start_time.elapsed().as_millis() as u64,
            confidence_distribution: confidence_dist,
        })
    }
    
    fn should_use_llm(&self, pattern_entities: &[ExtractedEntity]) -> bool {
        if pattern_entities.is_empty() {
            return true;
        }
        
        let avg_confidence = pattern_entities.iter()
            .map(|e| e.confidence)
            .sum::<f32>() / pattern_entities.len() as f32;
        
        avg_confidence < self.config.use_llm_when_pattern_confidence_below
    }
    
    fn merge_extractions(
        &self,
        pattern_entities: Vec<ExtractedEntity>,
        llm_entities: Vec<ExtractedEntity>,
    ) -> Result<Vec<ExtractedEntity>> {
        match self.config.merge_strategy {
            MergeStrategy::PatternPriority => Ok(pattern_entities),
            MergeStrategy::LLMPriority => Ok(llm_entities),
            MergeStrategy::HighestConfidence => {
                Ok(self.merge_by_highest_confidence(pattern_entities, llm_entities))
            }
            MergeStrategy::Ensemble => {
                Ok(self.ensemble_merge(pattern_entities, llm_entities))
            }
        }
    }
    
    fn merge_by_highest_confidence(
        &self,
        pattern_entities: Vec<ExtractedEntity>,
        llm_entities: Vec<ExtractedEntity>,
    ) -> Vec<ExtractedEntity> {
        let mut all_entities = pattern_entities;
        all_entities.extend(llm_entities);
        
        // Sort by confidence and remove overlaps
        all_entities.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        
        let mut final_entities = Vec::new();
        for entity in all_entities {
            if !final_entities.iter().any(|existing: &ExtractedEntity| {
                self.entities_overlap(&entity, existing)
            }) {
                final_entities.push(entity);
            }
        }
        
        final_entities
    }
    
    fn entities_overlap(&self, a: &ExtractedEntity, b: &ExtractedEntity) -> bool {
        a.start_pos < b.end_pos && a.end_pos > b.start_pos
    }
    
    fn filter_and_validate(&self, mut entities: Vec<ExtractedEntity>) -> Result<Vec<ExtractedEntity>> {
        // Filter by minimum confidence
        entities.retain(|e| e.confidence >= self.config.min_entity_confidence);
        
        // Limit number of entities
        entities.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        entities.truncate(self.config.max_entities_per_query);
        
        // Sort by position for final output
        entities.sort_by_key(|e| e.start_pos);
        
        Ok(entities)
    }
    
    fn calculate_confidence_distribution(&self, entities: &[ExtractedEntity]) -> HashMap<EntityType, f32> {
        let mut distribution = HashMap::new();
        
        for entity in entities {
            let entry = distribution.entry(entity.entity_type.clone()).or_insert(0.0);
            *entry += entity.confidence;
        }
        
        // Normalize by count
        for (entity_type, total_confidence) in &mut distribution {
            let count = entities.iter()
                .filter(|e| e.entity_type == *entity_type)
                .count() as f32;
            if count > 0.0 {
                *total_confidence /= count;
            }
        }
        
        distribution
    }
}
```

## File Locations

- `src/query/entity_extraction.rs` - Main implementation
- `src/query/entity_patterns.rs` - Pattern definitions
- `src/query/entity_types.rs` - Type definitions
- `src/query/hybrid_extractor.rs` - Hybrid extraction logic
- `tests/query/entity_extraction_tests.rs` - Test implementation

## Success Criteria

- [ ] Extracts entities from 95%+ of queries
- [ ] Entity type classification > 85% accuracy
- [ ] Pattern matching covers common entities
- [ ] LLM enhancement working correctly
- [ ] Hybrid approach balances speed/accuracy
- [ ] Confidence scoring is meaningful
- [ ] All tests pass

## Test Requirements

```rust
#[test]
fn test_pattern_entity_extraction() {
    let extractor = PatternEntityExtractor::new();
    
    let test_cases = vec![
        ("What animals can fly?", vec!["animals"]),
        ("Compare lions and tigers", vec!["lions", "tigers"]),
        ("Plants in the forest", vec!["Plants", "forest"]),
    ];
    
    for (query, expected) in test_cases {
        let entities = extractor.extract_entities(query);
        
        for expected_entity in expected {
            assert!(entities.iter().any(|e| e.text.contains(expected_entity)));
        }
    }
}

#[tokio::test]
async fn test_llm_entity_extraction() {
    let llm = create_test_llm().await;
    let extractor = LLMEntityExtractor::new(llm);
    
    let query = "How does photosynthesis work in green plants?";
    let result = extractor.extract_entities(query).await.unwrap();
    
    assert!(!result.is_empty());
    assert!(result.iter().any(|e| matches!(e.entity_type, EntityType::Concept)));
    assert!(result.iter().any(|e| matches!(e.entity_type, EntityType::Organism)));
}

#[tokio::test]
async fn test_hybrid_extraction_performance() {
    let llm = create_test_llm().await;
    let extractor = HybridEntityExtractor::new(llm);
    
    let queries = vec![
        "Simple query with animals",
        "Complex evolutionary relationship between primates and humans",
        "How do neural networks process information in the brain?",
    ];
    
    for query in queries {
        let start = std::time::Instant::now();
        let result = extractor.extract_entities(query).await.unwrap();
        let elapsed = start.elapsed();
        
        assert!(!result.entities.is_empty());
        assert!(elapsed.as_millis() < 1000); // Performance requirement
        assert!(result.entities.iter().all(|e| e.confidence >= 0.5));
    }
}

#[test]
fn test_entity_overlap_resolution() {
    let extractor = PatternEntityExtractor::new();
    
    // Create overlapping entities
    let mut entities = vec![
        ExtractedEntity {
            text: "large animals".to_string(),
            entity_type: EntityType::Organism,
            start_pos: 0,
            end_pos: 13,
            confidence: 0.8,
            aliases: vec![],
            context_clues: vec![],
            modifiers: vec![],
        },
        ExtractedEntity {
            text: "animals".to_string(),
            entity_type: EntityType::Organism,
            start_pos: 6,
            end_pos: 13,
            confidence: 0.9,
            aliases: vec![],
            context_clues: vec![],
            modifiers: vec![],
        },
    ];
    
    let resolved = extractor.resolve_overlaps(entities);
    assert_eq!(resolved.len(), 1);
    assert_eq!(resolved[0].text, "large animals");
}

#[test]
fn test_modifier_detection() {
    let extractor = PatternEntityExtractor::new();
    
    let queries = vec![
        "large mammals",
        "small red birds",
        "ancient trees",
    ];
    
    for query in queries {
        let entities = extractor.extract_entities(query);
        assert!(entities.iter().any(|e| !e.modifiers.is_empty()));
    }
}

#[test]
fn test_confidence_distribution() {
    let extractor = HybridEntityExtractor::new(create_mock_llm());
    
    let entities = vec![
        ExtractedEntity {
            text: "cat".to_string(),
            entity_type: EntityType::Organism,
            start_pos: 0,
            end_pos: 3,
            confidence: 0.9,
            aliases: vec![],
            context_clues: vec![],
            modifiers: vec![],
        },
        ExtractedEntity {
            text: "behavior".to_string(),
            entity_type: EntityType::Concept,
            start_pos: 4,
            end_pos: 12,
            confidence: 0.8,
            aliases: vec![],
            context_clues: vec![],
            modifiers: vec![],
        },
    ];
    
    let distribution = extractor.calculate_confidence_distribution(&entities);
    assert!(distribution.contains_key(&EntityType::Organism));
    assert!(distribution.contains_key(&EntityType::Concept));
}
```

## Quality Gates

- [ ] No false positive entity extractions > 10%
- [ ] Entity type classification stable across different text styles
- [ ] Performance under 500ms for complex queries
- [ ] Memory usage reasonable for concurrent extractions
- [ ] Modifier detection accuracy > 80%

## Next Task

Upon completion, proceed to **10_context_analysis.md**