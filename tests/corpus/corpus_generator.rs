use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use serde_json;
use std::fs;
use rand::Rng;
use chrono::{DateTime, Utc};

use llmkg::core::entity_extractor::{EntityExtractor, Entity};
use llmkg::core::relationship_extractor::{RelationshipExtractor, Relationship};

use super::accuracy_tests::{TestCorpus, TestSample, GroundTruthEntity, GroundTruthRelationship, TestCorpusMetadata};

#[derive(Debug, Serialize, Deserialize)]
pub struct TemplatePattern {
    pub pattern: String,
    pub variables: Vec<TemplateVariable>,
    pub domain: String,
    pub expected_entities: Vec<EntityTemplate>,
    pub expected_relationships: Vec<RelationshipTemplate>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TemplateVariable {
    pub name: String,
    pub var_type: String, // "person", "place", "organization", "concept", "time", "quantity"
    pub examples: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EntityTemplate {
    pub variable: String,
    pub entity_type: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RelationshipTemplate {
    pub subject_var: String,
    pub predicate: String,
    pub object_var: String,
    pub relationship_type: String,
}

pub struct CorpusGenerator {
    entity_extractor: EntityExtractor,
    relationship_extractor: RelationshipExtractor,
    templates: Vec<TemplatePattern>,
}

impl CorpusGenerator {
    pub fn new() -> Self {
        let mut generator = Self {
            entity_extractor: EntityExtractor::new(),
            relationship_extractor: RelationshipExtractor::new(),
            templates: Vec::new(),
        };
        generator.initialize_templates();
        generator
    }

    fn initialize_templates(&mut self) {
        // Scientific discovery templates
        self.templates.push(TemplatePattern {
            pattern: "{person} discovered {concept} in {year}.".to_string(),
            variables: vec![
                TemplateVariable {
                    name: "person".to_string(),
                    var_type: "person".to_string(),
                    examples: vec![
                        "Marie Curie".to_string(), "Alexander Fleming".to_string(),
                        "Gregor Mendel".to_string(), "Rosalind Franklin".to_string(),
                    ],
                },
                TemplateVariable {
                    name: "concept".to_string(),
                    var_type: "concept".to_string(),
                    examples: vec![
                        "radium".to_string(), "penicillin".to_string(),
                        "genetic inheritance".to_string(), "DNA structure".to_string(),
                    ],
                },
                TemplateVariable {
                    name: "year".to_string(),
                    var_type: "time".to_string(),
                    examples: vec!["1898".to_string(), "1928".to_string(), "1865".to_string(), "1951".to_string()],
                },
            ],
            domain: "science".to_string(),
            expected_entities: vec![
                EntityTemplate { variable: "person".to_string(), entity_type: "Person".to_string() },
                EntityTemplate { variable: "concept".to_string(), entity_type: "Concept".to_string() },
                EntityTemplate { variable: "year".to_string(), entity_type: "Time".to_string() },
            ],
            expected_relationships: vec![
                RelationshipTemplate {
                    subject_var: "person".to_string(),
                    predicate: "discovered".to_string(),
                    object_var: "concept".to_string(),
                    relationship_type: "Discovered".to_string(),
                },
            ],
        });

        // Business founding templates
        self.templates.push(TemplatePattern {
            pattern: "{organization} was founded by {person} in {year}.".to_string(),
            variables: vec![
                TemplateVariable {
                    name: "organization".to_string(),
                    var_type: "organization".to_string(),
                    examples: vec![
                        "OpenAI".to_string(), "SpaceX".to_string(), "Stripe".to_string(),
                        "Palantir".to_string(), "Coinbase".to_string(),
                    ],
                },
                TemplateVariable {
                    name: "person".to_string(),
                    var_type: "person".to_string(),
                    examples: vec![
                        "Sam Altman".to_string(), "Elon Musk".to_string(), "Patrick Collison".to_string(),
                        "Peter Thiel".to_string(), "Brian Armstrong".to_string(),
                    ],
                },
                TemplateVariable {
                    name: "year".to_string(),
                    var_type: "time".to_string(),
                    examples: vec!["2015".to_string(), "2002".to_string(), "2010".to_string(), "2003".to_string(), "2012".to_string()],
                },
            ],
            domain: "business".to_string(),
            expected_entities: vec![
                EntityTemplate { variable: "organization".to_string(), entity_type: "Organization".to_string() },
                EntityTemplate { variable: "person".to_string(), entity_type: "Person".to_string() },
                EntityTemplate { variable: "year".to_string(), entity_type: "Time".to_string() },
            ],
            expected_relationships: vec![
                RelationshipTemplate {
                    subject_var: "person".to_string(),
                    predicate: "founded".to_string(),
                    object_var: "organization".to_string(),
                    relationship_type: "Founded".to_string(),
                },
            ],
        });

        // Geographic location templates
        self.templates.push(TemplatePattern {
            pattern: "{landmark} is located in {city}, {country}.".to_string(),
            variables: vec![
                TemplateVariable {
                    name: "landmark".to_string(),
                    var_type: "place".to_string(),
                    examples: vec![
                        "Big Ben".to_string(), "Sagrada Familia".to_string(), "Christ the Redeemer".to_string(),
                        "Burj Khalifa".to_string(), "CN Tower".to_string(),
                    ],
                },
                TemplateVariable {
                    name: "city".to_string(),
                    var_type: "place".to_string(),
                    examples: vec![
                        "London".to_string(), "Barcelona".to_string(), "Rio de Janeiro".to_string(),
                        "Dubai".to_string(), "Toronto".to_string(),
                    ],
                },
                TemplateVariable {
                    name: "country".to_string(),
                    var_type: "place".to_string(),
                    examples: vec![
                        "England".to_string(), "Spain".to_string(), "Brazil".to_string(),
                        "UAE".to_string(), "Canada".to_string(),
                    ],
                },
            ],
            domain: "geography".to_string(),
            expected_entities: vec![
                EntityTemplate { variable: "landmark".to_string(), entity_type: "Place".to_string() },
                EntityTemplate { variable: "city".to_string(), entity_type: "Place".to_string() },
                EntityTemplate { variable: "country".to_string(), entity_type: "Place".to_string() },
            ],
            expected_relationships: vec![
                RelationshipTemplate {
                    subject_var: "landmark".to_string(),
                    predicate: "located in".to_string(),
                    object_var: "city".to_string(),
                    relationship_type: "LocatedIn".to_string(),
                },
            ],
        });

        // Literature creation templates
        self.templates.push(TemplatePattern {
            pattern: "{author} wrote '{book}' in {year}.".to_string(),
            variables: vec![
                TemplateVariable {
                    name: "author".to_string(),
                    var_type: "person".to_string(),
                    examples: vec![
                        "Toni Morrison".to_string(), "Gabriel García Márquez".to_string(),
                        "Chimamanda Ngozi Adichie".to_string(), "Haruki Murakami".to_string(),
                    ],
                },
                TemplateVariable {
                    name: "book".to_string(),
                    var_type: "concept".to_string(),
                    examples: vec![
                        "Beloved".to_string(), "One Hundred Years of Solitude".to_string(),
                        "Americanah".to_string(), "Norwegian Wood".to_string(),
                    ],
                },
                TemplateVariable {
                    name: "year".to_string(),
                    var_type: "time".to_string(),
                    examples: vec!["1987".to_string(), "1967".to_string(), "2013".to_string(), "1987".to_string()],
                },
            ],
            domain: "literature".to_string(),
            expected_entities: vec![
                EntityTemplate { variable: "author".to_string(), entity_type: "Person".to_string() },
                EntityTemplate { variable: "book".to_string(), entity_type: "Concept".to_string() },
                EntityTemplate { variable: "year".to_string(), entity_type: "Time".to_string() },
            ],
            expected_relationships: vec![
                RelationshipTemplate {
                    subject_var: "author".to_string(),
                    predicate: "wrote".to_string(),
                    object_var: "book".to_string(),
                    relationship_type: "Wrote".to_string(),
                },
            ],
        });

        // Scientific measurements templates
        self.templates.push(TemplatePattern {
            pattern: "{element} has the atomic number {number}.".to_string(),
            variables: vec![
                TemplateVariable {
                    name: "element".to_string(),
                    var_type: "concept".to_string(),
                    examples: vec![
                        "Uranium".to_string(), "Plutonium".to_string(), "Thorium".to_string(),
                        "Radium".to_string(), "Cesium".to_string(),
                    ],
                },
                TemplateVariable {
                    name: "number".to_string(),
                    var_type: "quantity".to_string(),
                    examples: vec!["92".to_string(), "94".to_string(), "90".to_string(), "88".to_string(), "55".to_string()],
                },
            ],
            domain: "science".to_string(),
            expected_entities: vec![
                EntityTemplate { variable: "element".to_string(), entity_type: "Concept".to_string() },
                EntityTemplate { variable: "number".to_string(), entity_type: "Quantity".to_string() },
            ],
            expected_relationships: vec![
                RelationshipTemplate {
                    subject_var: "element".to_string(),
                    predicate: "has atomic number".to_string(),
                    object_var: "number".to_string(),
                    relationship_type: "Has".to_string(),
                },
            ],
        });

        // Technology invention templates
        self.templates.push(TemplatePattern {
            pattern: "{inventor} invented the {invention} in {year}.".to_string(),
            variables: vec![
                TemplateVariable {
                    name: "inventor".to_string(),
                    var_type: "person".to_string(),
                    examples: vec![
                        "Tim Berners-Lee".to_string(), "Grace Hopper".to_string(),
                        "Ada Lovelace".to_string(), "Katherine Johnson".to_string(),
                    ],
                },
                TemplateVariable {
                    name: "invention".to_string(),
                    var_type: "concept".to_string(),
                    examples: vec![
                        "World Wide Web".to_string(), "compiler".to_string(),
                        "first computer program".to_string(), "orbital mechanics calculations".to_string(),
                    ],
                },
                TemplateVariable {
                    name: "year".to_string(),
                    var_type: "time".to_string(),
                    examples: vec!["1989".to_string(), "1952".to_string(), "1843".to_string(), "1961".to_string()],
                },
            ],
            domain: "technology".to_string(),
            expected_entities: vec![
                EntityTemplate { variable: "inventor".to_string(), entity_type: "Person".to_string() },
                EntityTemplate { variable: "invention".to_string(), entity_type: "Concept".to_string() },
                EntityTemplate { variable: "year".to_string(), entity_type: "Time".to_string() },
            ],
            expected_relationships: vec![
                RelationshipTemplate {
                    subject_var: "inventor".to_string(),
                    predicate: "invented".to_string(),
                    object_var: "invention".to_string(),
                    relationship_type: "Invented".to_string(),
                },
            ],
        });

        // Historical events templates
        self.templates.push(TemplatePattern {
            pattern: "The {event} occurred in {location} during {time_period}.".to_string(),
            variables: vec![
                TemplateVariable {
                    name: "event".to_string(),
                    var_type: "event".to_string(),
                    examples: vec![
                        "Industrial Revolution".to_string(), "Meiji Restoration".to_string(),
                        "Cultural Revolution".to_string(), "Arab Spring".to_string(),
                    ],
                },
                TemplateVariable {
                    name: "location".to_string(),
                    var_type: "place".to_string(),
                    examples: vec![
                        "Britain".to_string(), "Japan".to_string(), "China".to_string(), "Middle East".to_string(),
                    ],
                },
                TemplateVariable {
                    name: "time_period".to_string(),
                    var_type: "time".to_string(),
                    examples: vec![
                        "the 18th century".to_string(), "the 1860s".to_string(),
                        "the 1960s".to_string(), "2010-2012".to_string(),
                    ],
                },
            ],
            domain: "history".to_string(),
            expected_entities: vec![
                EntityTemplate { variable: "event".to_string(), entity_type: "Event".to_string() },
                EntityTemplate { variable: "location".to_string(), entity_type: "Place".to_string() },
                EntityTemplate { variable: "time_period".to_string(), entity_type: "Time".to_string() },
            ],
            expected_relationships: vec![
                RelationshipTemplate {
                    subject_var: "event".to_string(),
                    predicate: "occurred in".to_string(),
                    object_var: "location".to_string(),
                    relationship_type: "LocatedIn".to_string(),
                },
                RelationshipTemplate {
                    subject_var: "event".to_string(),
                    predicate: "during".to_string(),
                    object_var: "time_period".to_string(),
                    relationship_type: "During".to_string(),
                },
            ],
        });
    }

    pub fn generate_samples(&self, num_samples: usize) -> Vec<TestSample> {
        let mut samples = Vec::new();
        let mut rng = rand::thread_rng();
        let mut id_counter = 1;

        for _ in 0..num_samples {
            // Randomly select a template
            let template_idx = rng.gen_range(0..self.templates.len());
            let template = &self.templates[template_idx];

            // Generate variable assignments
            let assignments = self.generate_variable_assignments(template, &mut rng);

            // Fill in the template
            let text = self.fill_template(&template.pattern, &assignments);

            // Generate ground truth entities and relationships
            let entities = self.generate_ground_truth_entities(template, &assignments, &text);
            let relationships = self.generate_ground_truth_relationships(template, &assignments);

            samples.push(TestSample {
                id: id_counter,
                text,
                domain: template.domain.clone(),
                entities,
                relationships,
            });

            id_counter += 1;
        }

        samples
    }

    fn generate_variable_assignments(&self, template: &TemplatePattern, rng: &mut impl Rng) -> HashMap<String, String> {
        let mut assignments = HashMap::new();

        for variable in &template.variables {
            let example_idx = rng.gen_range(0..variable.examples.len());
            assignments.insert(variable.name.clone(), variable.examples[example_idx].clone());
        }

        assignments
    }

    fn fill_template(&self, pattern: &str, assignments: &HashMap<String, String>) -> String {
        let mut result = pattern.to_string();

        for (var_name, value) in assignments {
            let placeholder = format!("{{{}}}", var_name);
            result = result.replace(&placeholder, value);
        }

        result
    }

    fn generate_ground_truth_entities(
        &self,
        template: &TemplatePattern,
        assignments: &HashMap<String, String>,
        text: &str,
    ) -> Vec<GroundTruthEntity> {
        let mut entities = Vec::new();

        for entity_template in &template.expected_entities {
            if let Some(value) = assignments.get(&entity_template.variable) {
                // Find the position of this value in the text
                if let Some(start_pos) = text.find(value) {
                    entities.push(GroundTruthEntity {
                        text: value.clone(),
                        entity_type: entity_template.entity_type.clone(),
                        start: start_pos,
                        end: start_pos + value.len(),
                    });
                }
            }
        }

        entities
    }

    fn generate_ground_truth_relationships(
        &self,
        template: &TemplatePattern,
        assignments: &HashMap<String, String>,
    ) -> Vec<GroundTruthRelationship> {
        let mut relationships = Vec::new();

        for rel_template in &template.expected_relationships {
            if let (Some(subject), Some(object)) = (
                assignments.get(&rel_template.subject_var),
                assignments.get(&rel_template.object_var),
            ) {
                relationships.push(GroundTruthRelationship {
                    subject: subject.clone(),
                    predicate: rel_template.predicate.clone(),
                    object: object.clone(),
                    relationship_type: rel_template.relationship_type.clone(),
                });
            }
        }

        relationships
    }

    pub fn extend_corpus(&self, existing_corpus: &mut TestCorpus, additional_samples: usize) {
        let new_samples = self.generate_samples(additional_samples);
        let mut max_id = existing_corpus.samples.iter().map(|s| s.id).max().unwrap_or(0);

        for mut sample in new_samples {
            max_id += 1;
            sample.id = max_id;
            existing_corpus.samples.push(sample);
        }

        // Update metadata
        existing_corpus.metadata.total_samples = existing_corpus.samples.len();
    }

    pub fn validate_generated_samples(&self, samples: &[TestSample]) -> Vec<String> {
        let mut issues = Vec::new();

        for sample in samples {
            // Test if the system can extract the expected entities
            let extracted_entities = self.entity_extractor.extract_entities(&sample.text);
            let extracted_relationships = self.relationship_extractor.extract_relationships(&sample.text, &extracted_entities);

            // Check if we're missing expected entities
            for gt_entity in &sample.entities {
                let found = extracted_entities.iter().any(|e| {
                    e.name.to_lowercase() == gt_entity.text.to_lowercase() ||
                    e.name.to_lowercase().contains(&gt_entity.text.to_lowercase()) ||
                    gt_entity.text.to_lowercase().contains(&e.name.to_lowercase())
                });

                if !found {
                    issues.push(format!(
                        "Sample {}: Expected entity '{}' ({}) not found in extraction",
                        sample.id, gt_entity.text, gt_entity.entity_type
                    ));
                }
            }

            // Check if we're missing expected relationships
            for gt_rel in &sample.relationships {
                let found = extracted_relationships.iter().any(|r| {
                    r.subject.to_lowercase() == gt_rel.subject.to_lowercase() &&
                    r.object.to_lowercase() == gt_rel.object.to_lowercase()
                });

                if !found {
                    issues.push(format!(
                        "Sample {}: Expected relationship '{}--{}-->{} not found in extraction",
                        sample.id, gt_rel.subject, gt_rel.predicate, gt_rel.object
                    ));
                }
            }
        }

        issues
    }

    pub fn create_balanced_corpus(&self, samples_per_domain: usize) -> TestCorpus {
        let domains: Vec<String> = self.templates.iter()
            .map(|t| t.domain.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        let mut all_samples = Vec::new();
        let mut id_counter = 1;

        for domain in &domains {
            let domain_templates: Vec<&TemplatePattern> = self.templates.iter()
                .filter(|t| t.domain == *domain)
                .collect();

            let samples_per_template = samples_per_domain / domain_templates.len().max(1);
            let mut rng = rand::thread_rng();

            for template in domain_templates {
                for _ in 0..samples_per_template {
                    let assignments = self.generate_variable_assignments(template, &mut rng);
                    let text = self.fill_template(&template.pattern, &assignments);
                    let entities = self.generate_ground_truth_entities(template, &assignments, &text);
                    let relationships = self.generate_ground_truth_relationships(template, &assignments);

                    all_samples.push(TestSample {
                        id: id_counter,
                        text,
                        domain: template.domain.clone(),
                        entities,
                        relationships,
                    });

                    id_counter += 1;
                }
            }
        }

        // Shuffle samples
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        all_samples.shuffle(&mut rng);

        TestCorpus {
            metadata: TestCorpusMetadata {
                name: "Generated LLMKG Test Corpus".to_string(),
                version: "1.0.0".to_string(),
                description: "Auto-generated test corpus with ground truth annotations".to_string(),
                total_samples: all_samples.len(),
                domains,
                entity_types: vec![
                    "Person".to_string(), "Place".to_string(), "Organization".to_string(),
                    "Concept".to_string(), "Event".to_string(), "Time".to_string(),
                    "Quantity".to_string(), "Unknown".to_string(),
                ],
                relationship_types: vec![
                    "Created".to_string(), "Discovered".to_string(), "Invented".to_string(),
                    "Developed".to_string(), "Founded".to_string(), "Built".to_string(),
                    "Wrote".to_string(), "Designed".to_string(), "LocatedIn".to_string(),
                    "From".to_string(), "WorksWith".to_string(), "Is".to_string(),
                    "Has".to_string(), "During".to_string(),
                ],
            },
            samples: all_samples,
        }
    }

    pub fn save_corpus_to_file(&self, corpus: &TestCorpus, filepath: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json_content = serde_json::to_string_pretty(corpus)?;
        fs::write(filepath, json_content)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_corpus_generator_initialization() {
        let generator = CorpusGenerator::new();
        assert!(!generator.templates.is_empty());
        assert!(generator.templates.len() >= 7); // Should have at least 7 templates
    }

    #[test]
    fn test_generate_samples() {
        let generator = CorpusGenerator::new();
        let samples = generator.generate_samples(10);
        
        assert_eq!(samples.len(), 10);
        
        for sample in &samples {
            assert!(!sample.text.is_empty());
            assert!(!sample.domain.is_empty());
            assert!(!sample.entities.is_empty()); // Should have at least some entities
        }
    }

    #[test]
    fn test_variable_assignment() {
        let generator = CorpusGenerator::new();
        let mut rng = rand::thread_rng();
        
        let template = &generator.templates[0]; // Use first template
        let assignments = generator.generate_variable_assignments(template, &mut rng);
        
        // Should have assignments for all variables
        assert_eq!(assignments.len(), template.variables.len());
        
        for variable in &template.variables {
            assert!(assignments.contains_key(&variable.name));
            assert!(variable.examples.contains(&assignments[&variable.name]));
        }
    }

    #[test]
    fn test_fill_template() {
        let generator = CorpusGenerator::new();
        let pattern = "{person} invented {invention} in {year}.";
        
        let mut assignments = HashMap::new();
        assignments.insert("person".to_string(), "Tesla".to_string());
        assignments.insert("invention".to_string(), "AC motor".to_string());
        assignments.insert("year".to_string(), "1887".to_string());
        
        let result = generator.fill_template(pattern, &assignments);
        assert_eq!(result, "Tesla invented AC motor in 1887.");
    }

    #[test]
    fn test_create_balanced_corpus() {
        let generator = CorpusGenerator::new();
        let corpus = generator.create_balanced_corpus(2); // 2 samples per domain
        
        assert!(!corpus.samples.is_empty());
        assert!(corpus.metadata.total_samples > 0);
        assert_eq!(corpus.samples.len(), corpus.metadata.total_samples);
        
        // Check that we have diverse domains
        let domains: std::collections::HashSet<String> = corpus.samples.iter()
            .map(|s| s.domain.clone())
            .collect();
        assert!(domains.len() > 1);
    }

    #[test]
    fn test_validation() {
        let generator = CorpusGenerator::new();
        let samples = generator.generate_samples(5);
        let issues = generator.validate_generated_samples(&samples);
        
        // Should have minimal issues for well-formed templates
        println!("Validation issues: {:?}", issues);
        // We expect some issues since the extraction system may not be perfect
        // but should not be too many
        assert!(issues.len() < samples.len() * 2); // Less than 2 issues per sample on average
    }
}