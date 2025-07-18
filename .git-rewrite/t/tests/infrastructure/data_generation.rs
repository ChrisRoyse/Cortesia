//! Data Generation System
//! 
//! Generates synthetic test data for various simulation scenarios.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::infrastructure::DeterministicRng;

/// Size categories for generated data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataSize {
    Small,
    Medium, 
    Large,
    XLarge,
}

/// Parameters for data generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationParams {
    pub size: DataSize,
    pub entity_count: usize,
    pub relationship_count: usize,
    pub embedding_dimension: usize,
    pub tags: Vec<String>,
}

/// Generated test data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedData {
    pub entities: Vec<GeneratedEntity>,
    pub relationships: Vec<GeneratedRelationship>,
    pub embeddings: Vec<GeneratedEmbedding>,
    pub metadata: DataMetadata,
}

/// Generated entity structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedEntity {
    pub id: String,
    pub name: String,
    pub entity_type: String,
    pub attributes: HashMap<String, String>,
}

/// Generated relationship structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedRelationship {
    pub source_id: String,
    pub target_id: String,
    pub relationship_type: String,
    pub weight: f32,
    pub attributes: HashMap<String, String>,
}

/// Generated embedding structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedEmbedding {
    pub entity_id: String,
    pub vector: Vec<f32>,
    pub dimension: usize,
}

/// Metadata about generated data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataMetadata {
    pub generation_seed: u64,
    pub entity_count: usize,
    pub relationship_count: usize,
    pub embedding_dimension: usize,
    pub size_category: DataSize,
    pub generation_time_ms: u64,
}

/// Data generator implementation
pub struct DataGenerator {
    entity_types: Vec<String>,
    relationship_types: Vec<String>,
    attribute_keys: Vec<String>,
}

impl DataGenerator {
    /// Create a new data generator
    pub fn new() -> Result<Self> {
        Ok(Self {
            entity_types: vec![
                "Person".to_string(),
                "Organization".to_string(),
                "Location".to_string(),
                "Event".to_string(),
                "Document".to_string(),
                "Concept".to_string(),
            ],
            relationship_types: vec![
                "knows".to_string(),
                "works_for".to_string(),
                "located_in".to_string(),
                "participated_in".to_string(),
                "mentions".to_string(),
                "related_to".to_string(),
                "part_of".to_string(),
                "created_by".to_string(),
            ],
            attribute_keys: vec![
                "description".to_string(),
                "category".to_string(),
                "importance".to_string(),
                "confidence".to_string(),
                "source".to_string(),
                "timestamp".to_string(),
            ],
        })
    }

    /// Generate test data based on parameters
    pub async fn generate(&self, params: &GenerationParams) -> Result<GeneratedData> {
        let start_time = std::time::Instant::now();
        let generation_seed = self.calculate_seed(params);
        let mut rng = DeterministicRng::new(generation_seed);

        // Generate entities
        let entities = self.generate_entities(&mut rng, params)?;

        // Generate relationships
        let relationships = self.generate_relationships(&mut rng, &entities, params)?;

        // Generate embeddings
        let embeddings = self.generate_embeddings(&mut rng, &entities, params)?;

        let generation_time = start_time.elapsed();

        let metadata = DataMetadata {
            generation_seed,
            entity_count: entities.len(),
            relationship_count: relationships.len(),
            embedding_dimension: params.embedding_dimension,
            size_category: params.size.clone(),
            generation_time_ms: generation_time.as_millis() as u64,
        };

        Ok(GeneratedData {
            entities,
            relationships,
            embeddings,
            metadata,
        })
    }

    /// Generate entities
    fn generate_entities(&self, rng: &mut DeterministicRng, params: &GenerationParams) -> Result<Vec<GeneratedEntity>> {
        let mut entities = Vec::new();

        for i in 0..params.entity_count {
            let entity_type = &self.entity_types[rng.gen_range(0..self.entity_types.len())];
            let id = format!("entity_{:06}", i);
            let name = self.generate_entity_name(rng, entity_type, i);
            
            let mut attributes = HashMap::new();
            
            // Add some random attributes
            let attr_count = rng.gen_range(1..=5);
            for _ in 0..attr_count {
                let key = &self.attribute_keys[rng.gen_range(0..self.attribute_keys.len())];
                let value = self.generate_attribute_value(rng, key);
                attributes.insert(key.clone(), value);
            }

            entities.push(GeneratedEntity {
                id,
                name,
                entity_type: entity_type.clone(),
                attributes,
            });
        }

        Ok(entities)
    }

    /// Generate relationships
    fn generate_relationships(
        &self, 
        rng: &mut DeterministicRng, 
        entities: &[GeneratedEntity], 
        params: &GenerationParams
    ) -> Result<Vec<GeneratedRelationship>> {
        let mut relationships = Vec::new();

        for _ in 0..params.relationship_count {
            let source_idx = rng.gen_range(0..entities.len());
            let target_idx = rng.gen_range(0..entities.len());
            
            // Avoid self-loops
            if source_idx == target_idx {
                continue;
            }

            let relationship_type = &self.relationship_types[rng.gen_range(0..self.relationship_types.len())];
            let weight = rng.gen_range(0.1..1.0);

            let mut attributes = HashMap::new();
            attributes.insert("confidence".to_string(), format!("{:.2}", rng.gen_range(0.5..1.0)));
            
            relationships.push(GeneratedRelationship {
                source_id: entities[source_idx].id.clone(),
                target_id: entities[target_idx].id.clone(),
                relationship_type: relationship_type.clone(),
                weight,
                attributes,
            });
        }

        Ok(relationships)
    }

    /// Generate embeddings
    fn generate_embeddings(
        &self, 
        rng: &mut DeterministicRng, 
        entities: &[GeneratedEntity], 
        params: &GenerationParams
    ) -> Result<Vec<GeneratedEmbedding>> {
        let mut embeddings = Vec::new();

        for entity in entities {
            let mut vector = Vec::with_capacity(params.embedding_dimension);
            
            // Generate normalized random vector
            for _ in 0..params.embedding_dimension {
                vector.push(rng.gen_range(-1.0..1.0));
            }

            // Normalize vector
            let magnitude: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
            if magnitude > 0.0 {
                for val in &mut vector {
                    *val /= magnitude;
                }
            }

            embeddings.push(GeneratedEmbedding {
                entity_id: entity.id.clone(),
                vector,
                dimension: params.embedding_dimension,
            });
        }

        Ok(embeddings)
    }

    /// Generate entity name based on type
    fn generate_entity_name(&self, rng: &mut DeterministicRng, entity_type: &str, index: usize) -> String {
        match entity_type {
            "Person" => {
                let first_names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"];
                let last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis"];
                let first = first_names[rng.gen_range(0..first_names.len())];
                let last = last_names[rng.gen_range(0..last_names.len())];
                format!("{} {} #{}", first, last, index)
            }
            "Organization" => {
                let prefixes = ["Tech", "Global", "Advanced", "Future", "Smart", "Dynamic"];
                let suffixes = ["Corp", "Inc", "LLC", "Ltd", "Systems", "Solutions"];
                let prefix = prefixes[rng.gen_range(0..prefixes.len())];
                let suffix = suffixes[rng.gen_range(0..suffixes.len())];
                format!("{} {} #{}", prefix, suffix, index)
            }
            "Location" => {
                let places = ["City", "Town", "Village", "District", "Region", "Area"];
                let adjectives = ["New", "Old", "North", "South", "East", "West", "Central"];
                let place = places[rng.gen_range(0..places.len())];
                let adj = adjectives[rng.gen_range(0..adjectives.len())];
                format!("{} {} #{}", adj, place, index)
            }
            _ => format!("{} #{}", entity_type, index)
        }
    }

    /// Generate attribute value
    fn generate_attribute_value(&self, rng: &mut DeterministicRng, key: &str) -> String {
        match key {
            "description" => {
                let descriptions = [
                    "Important entity in the knowledge graph",
                    "Key component of the system",
                    "Central node with many connections",
                    "Peripheral entity with specific role",
                    "Bridge connecting different clusters",
                ];
                descriptions[rng.gen_range(0..descriptions.len())].to_string()
            }
            "category" => {
                let categories = ["Primary", "Secondary", "Auxiliary", "Reference", "Core"];
                categories[rng.gen_range(0..categories.len())].to_string()
            }
            "importance" => {
                let levels = ["High", "Medium", "Low"];
                levels[rng.gen_range(0..levels.len())].to_string()
            }
            "confidence" => format!("{:.2}", rng.gen_range(0.5..1.0)),
            "source" => {
                let sources = ["Database", "API", "Manual", "Import", "Generated"];
                sources[rng.gen_range(0..sources.len())].to_string()
            }
            "timestamp" => {
                let base_time = 1640995200; // 2022-01-01 UTC
                let random_offset = rng.gen_range(0..31536000); // +1 year
                (base_time + random_offset).to_string()
            }
            _ => format!("value_{}", rng.gen_range(1000..9999))
        }
    }

    /// Calculate deterministic seed from parameters
    fn calculate_seed(&self, params: &GenerationParams) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        params.entity_count.hash(&mut hasher);
        params.relationship_count.hash(&mut hasher);
        params.embedding_dimension.hash(&mut hasher);
        for tag in &params.tags {
            tag.hash(&mut hasher);
        }
        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_data_generator_creation() {
        let generator = DataGenerator::new();
        assert!(generator.is_ok());
    }

    #[tokio::test]
    async fn test_small_data_generation() {
        let generator = DataGenerator::new().unwrap();
        
        let params = GenerationParams {
            size: DataSize::Small,
            entity_count: 10,
            relationship_count: 15,
            embedding_dimension: 64,
            tags: vec!["test".to_string()],
        };

        let data = generator.generate(&params).await.unwrap();
        
        assert_eq!(data.entities.len(), 10);
        assert!(data.relationships.len() <= 15); // Some may be filtered out
        assert_eq!(data.embeddings.len(), 10);
        assert_eq!(data.metadata.embedding_dimension, 64);
    }

    #[tokio::test]
    async fn test_deterministic_generation() {
        let generator = DataGenerator::new().unwrap();
        
        let params = GenerationParams {
            size: DataSize::Small,
            entity_count: 5,
            relationship_count: 10,
            embedding_dimension: 32,
            tags: vec!["deterministic".to_string()],
        };

        let data1 = generator.generate(&params).await.unwrap();
        let data2 = generator.generate(&params).await.unwrap();
        
        // Should be identical due to deterministic seed
        assert_eq!(data1.entities.len(), data2.entities.len());
        assert_eq!(data1.embeddings.len(), data2.embeddings.len());
        assert_eq!(data1.metadata.generation_seed, data2.metadata.generation_seed);
        
        // Check first entity is identical
        assert_eq!(data1.entities[0].name, data2.entities[0].name);
        assert_eq!(data1.entities[0].entity_type, data2.entities[0].entity_type);
    }

    #[test]
    fn test_entity_name_generation() {
        let generator = DataGenerator::new().unwrap();
        let mut rng = DeterministicRng::new(12345);
        
        let person_name = generator.generate_entity_name(&mut rng, "Person", 1);
        assert!(person_name.contains("#1"));
        
        let org_name = generator.generate_entity_name(&mut rng, "Organization", 2);
        assert!(org_name.contains("#2"));
        
        let location_name = generator.generate_entity_name(&mut rng, "Location", 3);
        assert!(location_name.contains("#3"));
    }

    #[test]
    fn test_attribute_value_generation() {
        let generator = DataGenerator::new().unwrap();
        let mut rng = DeterministicRng::new(54321);
        
        let confidence = generator.generate_attribute_value(&mut rng, "confidence");
        assert!(confidence.parse::<f32>().is_ok());
        
        let category = generator.generate_attribute_value(&mut rng, "category");
        assert!(!category.is_empty());
        
        let unknown = generator.generate_attribute_value(&mut rng, "unknown_key");
        assert!(unknown.starts_with("value_"));
    }
}