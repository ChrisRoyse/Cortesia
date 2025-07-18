use crate::core::triple::Triple;
use crate::error::{GraphError, Result};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use regex::Regex;

/// Advanced NLP-based entity and relation extraction
pub struct AdvancedEntityExtractor {
    ner_models: HashMap<String, Arc<dyn NERModel>>,
    entity_linker: EntityLinker,
    coreference_resolver: CoreferenceResolver,
    relation_extractor: RelationExtractor,
}

impl AdvancedEntityExtractor {
    pub fn new() -> Self {
        let mut ner_models = HashMap::new();
        ner_models.insert("person".to_string(), Arc::new(PersonNERModel::new()) as Arc<dyn NERModel>);
        ner_models.insert("location".to_string(), Arc::new(LocationNERModel::new()) as Arc<dyn NERModel>);
        ner_models.insert("organization".to_string(), Arc::new(OrganizationNERModel::new()) as Arc<dyn NERModel>);
        ner_models.insert("misc".to_string(), Arc::new(MiscNERModel::new()) as Arc<dyn NERModel>);
        
        Self {
            ner_models,
            entity_linker: EntityLinker::new(),
            coreference_resolver: CoreferenceResolver::new(),
            relation_extractor: RelationExtractor::new(),
        }
    }

    pub async fn extract_entities(&self, text: &str) -> Result<Vec<Entity>> {
        // Step 1: Resolve coreferences
        let resolved_text = self.coreference_resolver.resolve(text).await?;
        
        // Step 2: Extract named entities using multiple models
        let mut all_entities = Vec::new();
        
        for (model_name, model) in &self.ner_models {
            let entities = model.extract_entities(&resolved_text).await?;
            for mut entity in entities {
                entity.source_model = model_name.clone();
                all_entities.push(entity);
            }
        }
        
        // Step 3: Merge and deduplicate entities
        let merged_entities = self.merge_entities(all_entities)?;
        
        // Step 4: Link entities to existing knowledge graph nodes
        let linked_entities = self.entity_linker.link_entities(merged_entities).await?;
        
        Ok(linked_entities)
    }

    pub async fn extract_relations(&self, text: &str, entities: &[Entity]) -> Result<Vec<Relation>> {
        self.relation_extractor.extract_relations(text, entities).await
    }

    pub async fn extract_triples(&self, text: &str) -> Result<Vec<Triple>> {
        // Extract entities first
        let entities = self.extract_entities(text).await?;
        
        // Extract relations between entities
        let relations = self.extract_relations(text, &entities).await?;
        
        // Convert relations to triples
        let mut triples = Vec::new();
        for relation in relations {
            let subject_entity = entities.iter().find(|e| e.id == relation.subject_id);
            let object_entity = entities.iter().find(|e| e.id == relation.object_id);
            
            if let (Some(subject), Some(object)) = (subject_entity, object_entity) {
                let triple = Triple::with_metadata(
                    subject.canonical_name.clone(),
                    relation.predicate.clone(),
                    object.canonical_name.clone(),
                    relation.confidence,
                    Some(relation.evidence.clone()),
                )?;
                triples.push(triple);
            }
        }
        
        Ok(triples)
    }

    fn merge_entities(&self, entities: Vec<Entity>) -> Result<Vec<Entity>> {
        let mut merged = HashMap::new();
        let mut entity_groups = HashMap::new();
        
        // Group entities by text span
        for entity in entities {
            let key = (entity.start_pos, entity.end_pos);
            entity_groups.entry(key).or_insert_with(Vec::new).push(entity);
        }
        
        // Merge entities with same span
        for ((start, end), group) in entity_groups {
            let merged_entity = self.merge_entity_group(group)?;
            merged.insert((start, end), merged_entity);
        }
        
        Ok(merged.into_values().collect())
    }

    fn merge_entity_group(&self, mut entities: Vec<Entity>) -> Result<Entity> {
        if entities.is_empty() {
            return Err(GraphError::InvalidInput("Empty entity group".to_string()));
        }
        
        if entities.len() == 1 {
            return Ok(entities.pop().unwrap());
        }
        
        // Take the entity with highest confidence as base
        entities.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        let mut base_entity = entities.remove(0);
        
        // Merge entity types and increase confidence
        let mut all_types = vec![base_entity.entity_type.clone()];
        let mut total_confidence = base_entity.confidence;
        
        for entity in entities {
            if !all_types.contains(&entity.entity_type) {
                all_types.push(entity.entity_type);
            }
            total_confidence += entity.confidence;
        }
        
        // Update merged entity
        base_entity.entity_type = all_types.join(",");
        base_entity.confidence = (total_confidence / all_types.len() as f32).min(1.0);
        
        Ok(base_entity)
    }
}

/// Trait for Named Entity Recognition models
#[async_trait::async_trait]
pub trait NERModel: Send + Sync {
    async fn extract_entities(&self, text: &str) -> Result<Vec<Entity>>;
    fn get_model_name(&self) -> &str;
    fn get_supported_types(&self) -> Vec<&str>;
}

/// Person NER model
pub struct PersonNERModel {
    patterns: Vec<Regex>,
}

impl PersonNERModel {
    fn new() -> Self {
        let patterns = vec![
            Regex::new(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b").unwrap(), // First Last
            Regex::new(r"\b[A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+\b").unwrap(), // First M. Last
            Regex::new(r"\bDr\. [A-Z][a-z]+\b").unwrap(), // Dr. Name
            Regex::new(r"\bProf\. [A-Z][a-z]+\b").unwrap(), // Prof. Name
        ];
        
        Self { patterns }
    }
}

#[async_trait::async_trait]
impl NERModel for PersonNERModel {
    async fn extract_entities(&self, text: &str) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();
        
        for pattern in &self.patterns {
            for mat in pattern.find_iter(text) {
                entities.push(Entity {
                    id: format!("person_{}", entities.len()),
                    text: mat.as_str().to_string(),
                    canonical_name: mat.as_str().to_string(),
                    entity_type: "PERSON".to_string(),
                    start_pos: mat.start(),
                    end_pos: mat.end(),
                    confidence: 0.8,
                    source_model: "person".to_string(),
                    linked_id: None,
                    properties: HashMap::new(),
                });
            }
        }
        
        Ok(entities)
    }
    
    fn get_model_name(&self) -> &str {
        "person"
    }
    
    fn get_supported_types(&self) -> Vec<&str> {
        vec!["PERSON"]
    }
}

/// Location NER model
pub struct LocationNERModel {
    patterns: Vec<Regex>,
}

impl LocationNERModel {
    fn new() -> Self {
        let patterns = vec![
            Regex::new(r"\b[A-Z][a-z]+ [A-Z][a-z]+, [A-Z][a-z]+\b").unwrap(), // City, State
            Regex::new(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b").unwrap(), // City Name
            Regex::new(r"\b[A-Z][a-z]+ University\b").unwrap(), // Universities
        ];
        
        Self { patterns }
    }
}

#[async_trait::async_trait]
impl NERModel for LocationNERModel {
    async fn extract_entities(&self, text: &str) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();
        
        for pattern in &self.patterns {
            for mat in pattern.find_iter(text) {
                entities.push(Entity {
                    id: format!("location_{}", entities.len()),
                    text: mat.as_str().to_string(),
                    canonical_name: mat.as_str().to_string(),
                    entity_type: "LOCATION".to_string(),
                    start_pos: mat.start(),
                    end_pos: mat.end(),
                    confidence: 0.7,
                    source_model: "location".to_string(),
                    linked_id: None,
                    properties: HashMap::new(),
                });
            }
        }
        
        Ok(entities)
    }
    
    fn get_model_name(&self) -> &str {
        "location"
    }
    
    fn get_supported_types(&self) -> Vec<&str> {
        vec!["LOCATION", "GPE"]
    }
}

/// Organization NER model
pub struct OrganizationNERModel {
    patterns: Vec<Regex>,
}

impl OrganizationNERModel {
    fn new() -> Self {
        let patterns = vec![
            Regex::new(r"\b[A-Z][a-z]+ (Inc|Corp|LLC|Ltd)\b").unwrap(),
            Regex::new(r"\b[A-Z][a-z]+ [A-Z][a-z]+ (Company|Corporation)\b").unwrap(),
            Regex::new(r"\b[A-Z][a-z]+ (Institute|Foundation)\b").unwrap(),
        ];
        
        Self { patterns }
    }
}

#[async_trait::async_trait]
impl NERModel for OrganizationNERModel {
    async fn extract_entities(&self, text: &str) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();
        
        for pattern in &self.patterns {
            for mat in pattern.find_iter(text) {
                entities.push(Entity {
                    id: format!("org_{}", entities.len()),
                    text: mat.as_str().to_string(),
                    canonical_name: mat.as_str().to_string(),
                    entity_type: "ORGANIZATION".to_string(),
                    start_pos: mat.start(),
                    end_pos: mat.end(),
                    confidence: 0.75,
                    source_model: "organization".to_string(),
                    linked_id: None,
                    properties: HashMap::new(),
                });
            }
        }
        
        Ok(entities)
    }
    
    fn get_model_name(&self) -> &str {
        "organization"
    }
    
    fn get_supported_types(&self) -> Vec<&str> {
        vec!["ORGANIZATION", "ORG"]
    }
}

/// Miscellaneous NER model
pub struct MiscNERModel {
    patterns: Vec<Regex>,
}

impl MiscNERModel {
    fn new() -> Self {
        let patterns = vec![
            Regex::new(r"\b\d{4}\b").unwrap(), // Years
            Regex::new(r"\b[A-Z][a-z]+ [A-Z][a-z]+ (Theory|Principle|Law)\b").unwrap(), // Scientific concepts
            Regex::new(r"\b[A-Z][a-z]+ (Prize|Award|Medal)\b").unwrap(), // Awards
        ];
        
        Self { patterns }
    }
}

#[async_trait::async_trait]
impl NERModel for MiscNERModel {
    async fn extract_entities(&self, text: &str) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();
        
        for pattern in &self.patterns {
            for mat in pattern.find_iter(text) {
                let entity_type = if mat.as_str().chars().all(|c| c.is_numeric()) {
                    "DATE"
                } else if mat.as_str().contains("Theory") || mat.as_str().contains("Principle") || mat.as_str().contains("Law") {
                    "CONCEPT"
                } else {
                    "MISC"
                };
                
                entities.push(Entity {
                    id: format!("misc_{}", entities.len()),
                    text: mat.as_str().to_string(),
                    canonical_name: mat.as_str().to_string(),
                    entity_type: entity_type.to_string(),
                    start_pos: mat.start(),
                    end_pos: mat.end(),
                    confidence: 0.6,
                    source_model: "misc".to_string(),
                    linked_id: None,
                    properties: HashMap::new(),
                });
            }
        }
        
        Ok(entities)
    }
    
    fn get_model_name(&self) -> &str {
        "misc"
    }
    
    fn get_supported_types(&self) -> Vec<&str> {
        vec!["MISC", "DATE", "CONCEPT"]
    }
}

/// Entity linking to existing knowledge graph
pub struct EntityLinker {
    similarity_threshold: f32,
}

impl EntityLinker {
    fn new() -> Self {
        Self {
            similarity_threshold: 0.8,
        }
    }

    pub async fn link_entities(&self, entities: Vec<Entity>) -> Result<Vec<Entity>> {
        let mut linked_entities = Vec::new();
        
        for entity in entities {
            let linked_entity = self.link_single_entity(entity).await?;
            linked_entities.push(linked_entity);
        }
        
        Ok(linked_entities)
    }

    async fn link_single_entity(&self, mut entity: Entity) -> Result<Entity> {
        // In a real implementation, this would search the knowledge graph
        // for similar entities and link them
        
        // For now, just normalize the canonical name
        entity.canonical_name = self.normalize_entity_name(&entity.text);
        
        Ok(entity)
    }

    fn normalize_entity_name(&self, name: &str) -> String {
        // Simple normalization - remove common prefixes/suffixes
        let normalized = name
            .replace("Dr. ", "")
            .replace("Prof. ", "")
            .trim()
            .to_string();
        
        // Title case
        let mut result = String::new();
        let mut capitalize_next = true;
        
        for ch in normalized.chars() {
            if ch.is_whitespace() {
                result.push(ch);
                capitalize_next = true;
            } else if capitalize_next {
                result.push(ch.to_uppercase().next().unwrap());
                capitalize_next = false;
            } else {
                result.push(ch.to_lowercase().next().unwrap());
            }
        }
        
        result
    }
}

/// Coreference resolution
pub struct CoreferenceResolver {
    pronoun_patterns: Vec<Regex>,
}

impl CoreferenceResolver {
    fn new() -> Self {
        let pronoun_patterns = vec![
            Regex::new(r"\bhe\b").unwrap(),
            Regex::new(r"\bshe\b").unwrap(),
            Regex::new(r"\bit\b").unwrap(),
            Regex::new(r"\bthey\b").unwrap(),
            Regex::new(r"\bthis\b").unwrap(),
            Regex::new(r"\bthat\b").unwrap(),
        ];
        
        Self { pronoun_patterns }
    }

    pub async fn resolve(&self, text: &str) -> Result<String> {
        // Simple coreference resolution
        // In a real implementation, this would use sophisticated NLP models
        
        let resolved = text.to_string();
        
        // For now, just return the original text
        // A real implementation would resolve pronouns to their antecedents
        
        Ok(resolved)
    }
}

/// Relation extraction between entities
pub struct RelationExtractor {
    relation_models: Vec<Arc<dyn RelationModel>>,
    predicate_normalizer: PredicateNormalizer,
    confidence_scorer: ConfidenceScorer,
}

impl RelationExtractor {
    fn new() -> Self {
        let relation_models = vec![
            Arc::new(PatternBasedRelationModel::new()) as Arc<dyn RelationModel>,
            Arc::new(DependencyRelationModel::new()) as Arc<dyn RelationModel>,
        ];
        
        Self {
            relation_models,
            predicate_normalizer: PredicateNormalizer::new(),
            confidence_scorer: ConfidenceScorer::new(),
        }
    }

    pub async fn extract_relations(&self, text: &str, entities: &[Entity]) -> Result<Vec<Relation>> {
        let mut all_relations = Vec::new();
        
        // Extract relations using each model
        for model in &self.relation_models {
            let relations = model.extract_relations(text, entities).await?;
            all_relations.extend(relations);
        }
        
        // Normalize predicates
        for relation in &mut all_relations {
            relation.predicate = self.predicate_normalizer.normalize(&relation.predicate);
        }
        
        // Score confidence
        for relation in &mut all_relations {
            relation.confidence = self.confidence_scorer.score_relation(relation, text);
        }
        
        // Remove duplicates and low-confidence relations
        let filtered_relations = self.filter_relations(all_relations);
        
        Ok(filtered_relations)
    }

    fn filter_relations(&self, relations: Vec<Relation>) -> Vec<Relation> {
        let mut filtered = Vec::new();
        let mut seen = HashSet::new();
        
        for relation in relations {
            let key = (relation.subject_id.clone(), relation.predicate.clone(), relation.object_id.clone());
            
            if !seen.contains(&key) && relation.confidence > 0.3 {
                seen.insert(key);
                filtered.push(relation);
            }
        }
        
        filtered
    }
}

/// Trait for relation extraction models
#[async_trait::async_trait]
pub trait RelationModel: Send + Sync {
    async fn extract_relations(&self, text: &str, entities: &[Entity]) -> Result<Vec<Relation>>;
}

/// Pattern-based relation extraction
pub struct PatternBasedRelationModel {
    patterns: Vec<RelationPattern>,
}

impl PatternBasedRelationModel {
    fn new() -> Self {
        let patterns = vec![
            RelationPattern {
                pattern: Regex::new(r"(.+) (?:is|was) (?:a|an) (.+)").unwrap(),
                predicate: "is".to_string(),
                subject_group: 1,
                object_group: 2,
            },
            RelationPattern {
                pattern: Regex::new(r"(.+) (?:invented|created|developed) (.+)").unwrap(),
                predicate: "invented".to_string(),
                subject_group: 1,
                object_group: 2,
            },
            RelationPattern {
                pattern: Regex::new(r"(.+) (?:works at|employed by) (.+)").unwrap(),
                predicate: "works_at".to_string(),
                subject_group: 1,
                object_group: 2,
            },
            RelationPattern {
                pattern: Regex::new(r"(.+) (?:born in|from) (.+)").unwrap(),
                predicate: "born_in".to_string(),
                subject_group: 1,
                object_group: 2,
            },
        ];
        
        Self { patterns }
    }
}

#[async_trait::async_trait]
impl RelationModel for PatternBasedRelationModel {
    async fn extract_relations(&self, text: &str, entities: &[Entity]) -> Result<Vec<Relation>> {
        let mut relations = Vec::new();
        
        for pattern in &self.patterns {
            for captures in pattern.pattern.captures_iter(text) {
                if let (Some(subject_match), Some(object_match)) = (
                    captures.get(pattern.subject_group),
                    captures.get(pattern.object_group),
                ) {
                    // Find matching entities
                    let subject_entity = entities.iter().find(|e| 
                        subject_match.as_str().contains(&e.text) || e.text.contains(subject_match.as_str().trim())
                    );
                    let object_entity = entities.iter().find(|e| 
                        object_match.as_str().contains(&e.text) || e.text.contains(object_match.as_str().trim())
                    );
                    
                    if let (Some(subject), Some(object)) = (subject_entity, object_entity) {
                        relations.push(Relation {
                            subject_id: subject.id.clone(),
                            predicate: pattern.predicate.clone(),
                            object_id: object.id.clone(),
                            confidence: 0.7,
                            evidence: captures.get(0).unwrap().as_str().to_string(),
                            extraction_model: "pattern_based".to_string(),
                        });
                    }
                }
            }
        }
        
        Ok(relations)
    }
}

/// Dependency parsing-based relation extraction
pub struct DependencyRelationModel;

impl DependencyRelationModel {
    fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl RelationModel for DependencyRelationModel {
    async fn extract_relations(&self, text: &str, entities: &[Entity]) -> Result<Vec<Relation>> {
        // In a real implementation, this would use dependency parsing
        // For now, return empty relations
        Ok(Vec::new())
    }
}

/// Predicate normalization
pub struct PredicateNormalizer {
    normalizations: HashMap<String, String>,
}

impl PredicateNormalizer {
    fn new() -> Self {
        let mut normalizations = HashMap::new();
        normalizations.insert("is a".to_string(), "is".to_string());
        normalizations.insert("was a".to_string(), "is".to_string());
        normalizations.insert("are".to_string(), "is".to_string());
        normalizations.insert("were".to_string(), "is".to_string());
        normalizations.insert("works at".to_string(), "works_at".to_string());
        normalizations.insert("employed by".to_string(), "works_at".to_string());
        normalizations.insert("born in".to_string(), "born_in".to_string());
        normalizations.insert("from".to_string(), "born_in".to_string());
        
        Self { normalizations }
    }

    pub fn normalize(&self, predicate: &str) -> String {
        let cleaned = predicate.trim().to_lowercase();
        
        self.normalizations.get(&cleaned)
            .cloned()
            .unwrap_or_else(|| cleaned.replace(' ', "_"))
    }
}

/// Confidence scoring for relations
pub struct ConfidenceScorer;

impl ConfidenceScorer {
    fn new() -> Self {
        Self
    }

    pub fn score_relation(&self, relation: &Relation, text: &str) -> f32 {
        let mut score = 0.5f32; // Base score
        
        // Increase score based on evidence length
        if relation.evidence.len() > 20 {
            score += 0.1;
        }
        
        // Increase score for common predicates
        match relation.predicate.as_str() {
            "is" | "works_at" | "born_in" | "invented" => score += 0.2,
            _ => {}
        }
        
        // Decrease score for very long predicates
        if relation.predicate.len() > 20 {
            score -= 0.2;
        }
        
        score.max(0.0).min(1.0)
    }
}

/// Data structures

#[derive(Debug, Clone)]
pub struct Entity {
    pub id: String,
    pub text: String,
    pub canonical_name: String,
    pub entity_type: String,
    pub start_pos: usize,
    pub end_pos: usize,
    pub confidence: f32,
    pub source_model: String,
    pub linked_id: Option<String>,
    pub properties: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct Relation {
    pub subject_id: String,
    pub predicate: String,
    pub object_id: String,
    pub confidence: f32,
    pub evidence: String,
    pub extraction_model: String,
}

struct RelationPattern {
    pattern: Regex,
    predicate: String,
    subject_group: usize,
    object_group: usize,
}

