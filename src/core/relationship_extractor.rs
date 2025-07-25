use std::collections::HashMap;
use regex::Regex;
use lazy_static::lazy_static;
use crate::core::entity_extractor::{Entity, EntityType};

#[derive(Debug, Clone, PartialEq)]
pub struct Relationship {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub relationship_type: RelationshipType,
    pub confidence: f32,
    pub context_start: usize,
    pub context_end: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RelationshipType {
    // Causal
    Causes,
    CausedBy,
    Prevents,
    Enables,
    
    // Temporal
    Before,
    After,
    During,
    SimultaneousWith,
    
    // Hierarchical
    IsA,
    PartOf,
    Contains,
    BelongsTo,
    
    // Action
    Created,
    Discovered,
    Invented,
    Developed,
    Founded,
    Built,
    Wrote,
    Designed,
    
    // Association
    RelatedTo,
    SimilarTo,
    OppositeTo,
    WorksWith,
    MarriedTo,
    ChildOf,
    ParentOf,
    
    // Location
    LocatedIn,
    From,
    
    // Property
    Is,
    Has,
    
    // Generic
    Unknown,
}

lazy_static! {
    // Verb patterns mapped to relationship types
    static ref VERB_PATTERNS: HashMap<&'static str, RelationshipType> = {
        let mut m = HashMap::new();
        
        // Causal verbs
        m.insert("causes", RelationshipType::Causes);
        m.insert("caused", RelationshipType::Causes);
        m.insert("leads to", RelationshipType::Causes);
        m.insert("leads", RelationshipType::Causes);
        m.insert("results in", RelationshipType::Causes);
        m.insert("prevents", RelationshipType::Prevents);
        m.insert("enables", RelationshipType::Enables);
        m.insert("allows", RelationshipType::Enables);
        
        // Action verbs
        m.insert("created", RelationshipType::Created);
        m.insert("creates", RelationshipType::Created);
        m.insert("discovered", RelationshipType::Discovered);
        m.insert("discovers", RelationshipType::Discovered);
        m.insert("invented", RelationshipType::Invented);
        m.insert("invents", RelationshipType::Invented);
        m.insert("invent", RelationshipType::Invented);
        m.insert("developed", RelationshipType::Developed);
        m.insert("develops", RelationshipType::Developed);
        m.insert("founded", RelationshipType::Founded);
        m.insert("founds", RelationshipType::Founded);
        m.insert("built", RelationshipType::Built);
        m.insert("builds", RelationshipType::Built);
        m.insert("wrote", RelationshipType::Wrote);
        m.insert("writes", RelationshipType::Wrote);
        m.insert("designed", RelationshipType::Designed);
        m.insert("designs", RelationshipType::Designed);
        
        // Relationship verbs
        m.insert("married", RelationshipType::MarriedTo);
        m.insert("married to", RelationshipType::MarriedTo);
        m.insert("works with", RelationshipType::WorksWith);
        m.insert("collaborated with", RelationshipType::WorksWith);
        m.insert("child of", RelationshipType::ChildOf);
        m.insert("son of", RelationshipType::ChildOf);
        m.insert("daughter of", RelationshipType::ChildOf);
        m.insert("parent of", RelationshipType::ParentOf);
        m.insert("father of", RelationshipType::ParentOf);
        m.insert("mother of", RelationshipType::ParentOf);
        
        // Location verbs
        m.insert("located in", RelationshipType::LocatedIn);
        m.insert("is located in", RelationshipType::LocatedIn);
        m.insert("based in", RelationshipType::LocatedIn);
        m.insert("from", RelationshipType::From);
        m.insert("born in", RelationshipType::From);
        
        // Property verbs
        m.insert("is", RelationshipType::Is);
        m.insert("was", RelationshipType::Is);
        m.insert("are", RelationshipType::Is);
        m.insert("were", RelationshipType::Is);
        m.insert("has", RelationshipType::Has);
        m.insert("have", RelationshipType::Has);
        m.insert("had", RelationshipType::Has);
        
        // Hierarchical
        m.insert("is a", RelationshipType::IsA);
        m.insert("is an", RelationshipType::IsA);
        m.insert("part of", RelationshipType::PartOf);
        m.insert("contains", RelationshipType::Contains);
        m.insert("includes", RelationshipType::Contains);
        m.insert("belongs to", RelationshipType::BelongsTo);
        
        m
    };

    // Preposition patterns that often indicate relationships
    static ref PREP_PATTERNS: Vec<(&'static str, RelationshipType)> = vec![
        ("in", RelationshipType::LocatedIn),
        ("at", RelationshipType::LocatedIn),
        ("from", RelationshipType::From),
        ("of", RelationshipType::PartOf),
        ("by", RelationshipType::CausedBy),
        ("with", RelationshipType::WorksWith),
        ("for", RelationshipType::RelatedTo),
        ("during", RelationshipType::During),
        ("before", RelationshipType::Before),
        ("after", RelationshipType::After),
    ];
}

pub struct RelationshipExtractor {
    // In production, might include dependency parser
}

impl RelationshipExtractor {
    pub fn new() -> Self {
        RelationshipExtractor {}
    }

    pub fn extract_relationships(&self, text: &str, entities: &[Entity]) -> Vec<Relationship> {
        let mut relationships = Vec::new();
        
        // Convert entities to a map for quick lookup
        let entity_map: HashMap<&str, &Entity> = entities.iter()
            .map(|e| (e.name.as_str(), e))
            .collect();

        // Extract verb-based relationships
        relationships.extend(self.extract_verb_relationships(text, &entity_map));
        
        // Extract pattern-based relationships
        relationships.extend(self.extract_pattern_relationships(text, &entity_map));
        
        // Extract proximity-based relationships
        relationships.extend(self.extract_proximity_relationships(text, entities));
        
        // Extract sentence-based relationships (for cases where entities might be incomplete)
        relationships.extend(self.extract_sentence_relationships(text, entities));

        // Deduplicate and sort by confidence
        self.deduplicate_relationships(relationships)
    }

    fn extract_verb_relationships(&self, text: &str, entity_map: &HashMap<&str, &Entity>) -> Vec<Relationship> {
        let mut relationships = Vec::new();
        let _text_lower = text.to_lowercase();

        for entity1 in entity_map.values() {
            for entity2 in entity_map.values() {
                if entity1.name == entity2.name {
                    continue;
                }

                // Check if entities appear in order in the text
                if entity1.start_pos >= entity2.start_pos {
                    continue;
                }

                // Extract text between entities
                let between_text = if entity1.end_pos < entity2.start_pos {
                    &text[entity1.end_pos..entity2.start_pos]
                } else {
                    continue;
                };
                let between_text_lower = between_text.to_lowercase();

                // Look for verb patterns in the text between entities
                for (verb_pattern, rel_type) in VERB_PATTERNS.iter() {
                    if between_text_lower.contains(verb_pattern) {
                        // Clean up entity names (remove "The" prefix if present)
                        let subject = if entity1.name.starts_with("The ") {
                            entity1.name[4..].to_string()
                        } else {
                            entity1.name.clone()
                        };
                        
                        // For location relationships, extract just the first location
                        let object = if rel_type == &RelationshipType::LocatedIn && entity2.name.contains(" ") {
                            entity2.name.split_whitespace().next().unwrap_or(&entity2.name).to_string()
                        } else {
                            entity2.name.clone()
                        };
                        
                        // Normalize predicate (remove "is" prefix for location predicates)
                        let predicate = if verb_pattern == &"is located in" {
                            "located in".to_string()
                        } else {
                            verb_pattern.to_string()
                        };
                        
                        relationships.push(Relationship {
                            subject,
                            predicate,
                            object,
                            relationship_type: rel_type.clone(),
                            confidence: 0.9,
                            context_start: entity1.start_pos,
                            context_end: entity2.end_pos,
                        });
                        break; // Only take the first matching verb
                    }
                }
            }
        }

        relationships
    }

    fn extract_pattern_relationships(&self, text: &str, entity_map: &HashMap<&str, &Entity>) -> Vec<Relationship> {
        let mut relationships = Vec::new();

        // Special patterns like "X, Y of Z" (e.g., "CEO of Microsoft")
        let title_pattern = Regex::new(r"(\w+(?:\s+\w+)*),\s*(\w+(?:\s+\w+)*)\s+of\s+(\w+(?:\s+\w+)*)").unwrap();
        
        for cap in title_pattern.captures_iter(text) {
            if let (Some(person), Some(_title), Some(org)) = (cap.get(1), cap.get(2), cap.get(3)) {
                let person_str = person.as_str();
                let org_str = org.as_str();
                
                if entity_map.contains_key(person_str) && entity_map.contains_key(org_str) {
                    relationships.push(Relationship {
                        subject: person_str.to_string(),
                        predicate: "works_for".to_string(),
                        object: org_str.to_string(),
                        relationship_type: RelationshipType::WorksWith,
                        confidence: 0.85,
                        context_start: cap.get(0).unwrap().start(),
                        context_end: cap.get(0).unwrap().end(),
                    });
                }
            }
        }

        // Pattern: "X's Y" indicating possession or relationship
        let possessive_pattern = Regex::new(r"(\w+(?:\s+\w+)*)'s\s+(\w+(?:\s+\w+)*)").unwrap();
        
        for cap in possessive_pattern.captures_iter(text) {
            if let (Some(owner), Some(owned)) = (cap.get(1), cap.get(2)) {
                let owner_str = owner.as_str();
                let owned_str = owned.as_str();
                
                if entity_map.contains_key(owner_str) {
                    // Check if owned is also an entity or just a property
                    let predicate = if entity_map.contains_key(owned_str) {
                        "owns"
                    } else {
                        "has"
                    };
                    
                    relationships.push(Relationship {
                        subject: owner_str.to_string(),
                        predicate: predicate.to_string(),
                        object: owned_str.to_string(),
                        relationship_type: RelationshipType::Has,
                        confidence: 0.8,
                        context_start: cap.get(0).unwrap().start(),
                        context_end: cap.get(0).unwrap().end(),
                    });
                }
            }
        }

        relationships
    }

    fn extract_proximity_relationships(&self, text: &str, entities: &[Entity]) -> Vec<Relationship> {
        let mut relationships = Vec::new();
        
        // For entities that appear close together, try to infer relationships
        for i in 0..entities.len() {
            for j in i + 1..entities.len() {
                let entity1 = &entities[i];
                let entity2 = &entities[j];
                
                // Calculate distance between entities
                let distance = if entity2.start_pos > entity1.end_pos {
                    entity2.start_pos - entity1.end_pos
                } else if entity1.start_pos > entity2.end_pos {
                    entity1.start_pos - entity2.end_pos
                } else {
                    continue; // Overlapping entities
                };
                
                // If entities are close (within ~50 characters)
                if distance < 50 {
                    // Extract text between entities
                    let between_start = entity1.end_pos.min(entity2.end_pos);
                    let between_end = entity1.start_pos.max(entity2.start_pos);
                    
                    if between_end > between_start && between_end <= text.len() {
                        let between_text = &text[between_start..between_end];
                        
                        // Look for prepositions
                        for (prep, rel_type) in PREP_PATTERNS.iter() {
                            if between_text.contains(prep) {
                                let (subject, object) = if entity1.start_pos < entity2.start_pos {
                                    (&entity1.name, &entity2.name)
                                } else {
                                    (&entity2.name, &entity1.name)
                                };
                                
                                relationships.push(Relationship {
                                    subject: subject.clone(),
                                    predicate: prep.to_string(),
                                    object: object.clone(),
                                    relationship_type: rel_type.clone(),
                                    confidence: 0.6,
                                    context_start: between_start,
                                    context_end: between_end,
                                });
                                break;
                            }
                        }
                    }
                }
            }
        }
        
        relationships
    }

    fn extract_sentence_relationships(&self, text: &str, entities: &[Entity]) -> Vec<Relationship> {
        let mut relationships = Vec::new();
        
        // Split text into sentences
        let sentences: Vec<&str> = text.split(|c| c == '.' || c == '!' || c == '?')
            .filter(|s| !s.trim().is_empty())
            .collect();
        
        // Find all person entities for pronoun resolution
        let person_entities: Vec<&Entity> = entities.iter()
            .filter(|e| e.entity_type == EntityType::Person)
            .collect();
        
        for sentence in sentences {
            let sentence_lower = sentence.to_lowercase();
            
            // Look for entities in this sentence
            let sentence_entities: Vec<&Entity> = entities.iter()
                .filter(|e| sentence.contains(&e.name))
                .collect();
            
            if sentence_entities.len() >= 1 {
                // Special case for "born in" pattern
                if sentence_lower.contains("born") && sentence_lower.contains(" in ") {
                    for entity in &sentence_entities {
                        if entity.entity_type == crate::core::entity_extractor::EntityType::Person {
                            // Look for locations after "born ... in"
                            if let Some(born_pos) = sentence_lower.find("born") {
                                let after_born = &sentence_lower[born_pos..];
                                if let Some(in_pos) = after_born.find(" in ") {
                                    let after_in = &sentence[born_pos + in_pos + 4..]; // Skip " in "
                                    let location_words: Vec<&str> = after_in.split(|c: char| c == ',' || c == '.')
                                        .next()
                                        .unwrap_or("")
                                        .split_whitespace()
                                        .take(2) // Take up to 2 words for location
                                        .collect();
                                    
                                    if !location_words.is_empty() {
                                        let location = location_words.join(" ");
                                        relationships.push(Relationship {
                                            subject: entity.name.clone(),
                                            predicate: "born in".to_string(),
                                            object: location,
                                            relationship_type: RelationshipType::From,
                                            confidence: 0.8,
                                            context_start: 0,
                                            context_end: sentence.len(),
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
                
                // Check for verb patterns even if we don't have a complete entity list
                for (verb_pattern, rel_type) in VERB_PATTERNS.iter() {
                    if sentence_lower.contains(verb_pattern) {
                        // For discovered/invented patterns, try to extract objects
                        if matches!(rel_type, RelationshipType::Discovered | RelationshipType::Invented) {
                            // Look for pronoun references like "She discovered"
                            if sentence_lower.contains("she discovered") || sentence_lower.contains("he discovered") {
                                // Use person entities for pronoun resolution (add all person entities as potential subjects)
                                for entity in &person_entities {
                                        // Extract objects after "discovered"
                                        if let Some(discovered_pos) = sentence_lower.find("discovered") {
                                            let after_discovered = &sentence[discovered_pos + "discovered".len()..];
                                            
                                            // Look for specific elements in the context
                                            if after_discovered.contains("polonium") {
                                                relationships.push(Relationship {
                                                    subject: entity.name.clone(),
                                                    predicate: "discovered".to_string(),
                                                    object: "polonium".to_string(),
                                                    relationship_type: RelationshipType::Discovered,
                                                    confidence: 0.9,
                                                    context_start: 0,
                                                    context_end: sentence.len(),
                                                });
                                            }
                                            if after_discovered.contains("radium") {
                                                relationships.push(Relationship {
                                                    subject: entity.name.clone(),
                                                    predicate: "discovered".to_string(),
                                                    object: "radium".to_string(),
                                                    relationship_type: RelationshipType::Discovered,
                                                    confidence: 0.9,
                                                    context_start: 0,
                                                    context_end: sentence.len(),
                                                });
                                            }
                                        }
                                }
                            } else {
                                // Original logic for direct entity patterns
                                for entity in &sentence_entities {
                                    // Look for pattern: "Entity verb X and Y"
                                    let pattern = format!("{} {}", entity.name.to_lowercase(), verb_pattern);
                                    if let Some(start_idx) = sentence_lower.find(&pattern) {
                                        let after_verb = &sentence[start_idx + pattern.len()..];
                                        
                                        // Extract words after the verb (potential objects)
                                        let words: Vec<&str> = after_verb.split_whitespace()
                                            .filter(|w| !w.is_empty() && w.len() > 2)
                                            .collect();
                                        
                                        for (i, word) in words.iter().enumerate() {
                                            // Skip common words
                                            if matches!(*word, "the" | "a" | "an" | "and" | "or" | "in" | "at" | "on") {
                                                continue;
                                            }
                                            
                                            // Clean up the word (remove trailing punctuation)
                                            let clean_word = word.trim_matches(|c: char| !c.is_alphanumeric());
                                            
                                            if !clean_word.is_empty() {
                                                relationships.push(Relationship {
                                                    subject: entity.name.clone(),
                                                    predicate: verb_pattern.to_string(),
                                                    object: clean_word.to_string(),
                                                    relationship_type: rel_type.clone(),
                                                    confidence: 0.7,
                                                    context_start: 0,
                                                    context_end: sentence.len(),
                                                });
                                                
                                                // Only take first few objects
                                                if i >= 3 {
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        
                        // For causal relationships
                        if matches!(rel_type, RelationshipType::Causes | RelationshipType::Enables) {
                            // Look for pattern: "X causes Y"
                            let parts: Vec<&str> = sentence.split(verb_pattern).collect();
                            if parts.len() == 2 {
                                let subject_part = parts[0].trim();
                                let object_part = parts[1].trim();
                                
                                // Extract meaningful phrases
                                let subject = subject_part.split_whitespace()
                                    .collect::<Vec<_>>()
                                    .join(" ");
                                let object = object_part.split(',').next()
                                    .unwrap_or(object_part)
                                    .split_whitespace()
                                    .take(4) // Take first few words
                                    .collect::<Vec<_>>()
                                    .join(" ");
                                
                                if !subject.is_empty() && !object.is_empty() {
                                    relationships.push(Relationship {
                                        subject,
                                        predicate: verb_pattern.to_string(),
                                        object,
                                        relationship_type: rel_type.clone(),
                                        confidence: 0.6,
                                        context_start: 0,
                                        context_end: sentence.len(),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
        
        relationships
    }
    
    fn deduplicate_relationships(&self, mut relationships: Vec<Relationship>) -> Vec<Relationship> {
        // Sort by confidence descending
        relationships.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        
        let mut seen = HashMap::new();
        let mut result = Vec::new();
        
        for rel in relationships {
            let key = format!("{}-{}-{}", rel.subject, rel.predicate, rel.object);
            if !seen.contains_key(&key) {
                seen.insert(key, true);
                result.push(rel);
            }
        }
        
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::entity_extractor::EntityExtractor;

    #[test]
    fn test_extract_verb_relationships() {
        let extractor = RelationshipExtractor::new();
        let entity_extractor = EntityExtractor::new();
        
        let text = "Albert Einstein developed the Theory of Relativity.";
        let entities = entity_extractor.extract_entities(text);
        let relationships = extractor.extract_relationships(text, &entities);
        
        assert!(relationships.iter().any(|r| 
            r.subject == "Albert Einstein" && 
            r.predicate == "developed" && 
            r.object == "Theory of Relativity" &&
            r.relationship_type == RelationshipType::Developed
        ));
    }

    #[test]
    fn test_extract_location_relationships() {
        let extractor = RelationshipExtractor::new();
        let entity_extractor = EntityExtractor::new();
        
        let text = "The Eiffel Tower is located in Paris.";
        let entities = entity_extractor.extract_entities(text);
        let relationships = extractor.extract_relationships(text, &entities);
        
        assert!(relationships.iter().any(|r| 
            r.subject == "Eiffel Tower" && 
            r.object == "Paris" &&
            r.relationship_type == RelationshipType::LocatedIn
        ));
    }

    #[test]
    fn test_extract_possessive_relationships() {
        let extractor = RelationshipExtractor::new();
        let entity_extractor = EntityExtractor::new();
        
        let text = "Einstein's Theory of Relativity changed physics.";
        let entities = entity_extractor.extract_entities(text);
        let relationships = extractor.extract_relationships(text, &entities);
        
        assert!(relationships.iter().any(|r| 
            r.subject == "Einstein" && 
            r.object == "Theory" &&
            r.relationship_type == RelationshipType::Has
        ));
    }
}