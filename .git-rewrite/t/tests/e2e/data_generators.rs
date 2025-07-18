//! Data Generators for E2E Simulations
//! 
//! Specialized data generators for creating realistic test datasets for end-to-end simulations.

use crate::data_generation::{ComprehensiveDataGenerator, TestEntity, TestRelationship};
use crate::infrastructure::DeterministicRng;
use anyhow::{Result, anyhow};
use std::collections::HashMap;
use std::time::Duration;
use serde_json;

/// Entity key type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EntityKey(u64);

impl EntityKey {
    pub fn new(s: String) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        Self(hasher.finish())
    }
    
    pub fn from_hash(s: &str) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        Self(hasher.finish())
    }
    
    pub fn to_string(&self) -> String {
        format!("entity_{}", self.0)
    }
}

/// System update types for long-running simulations
#[derive(Debug, Clone)]
pub enum SystemUpdate {
    AddEntity(TestEntity),
    UpdateEntity(EntityKey, HashMap<String, String>),
    AddRelationship(EntityKey, EntityKey, TestRelationship),
    AddEmbedding(EntityKey, Vec<f32>),
}

/// User pattern for simulations
#[derive(Debug, Clone)]
pub struct UserPattern {
    pub queries: Vec<MockMcpToolRequest>,
    pub think_time_ms: u64,
    pub session_duration: Duration,
}

/// Mock MCP tool request for testing
#[derive(Debug, Clone)]
pub struct MockMcpToolRequest {
    pub tool_name: String,
    pub arguments: serde_json::Value,
}

/// Specification for academic knowledge base
#[derive(Debug, Clone)]
pub struct AcademicKbSpec {
    pub papers: u64,
    pub authors: u64,
    pub venues: u64,
    pub fields: u64,
    pub citation_years: std::ops::Range<i32>,
    pub embedding_dim: usize,
}

/// Specification for content creation knowledge base
#[derive(Debug, Clone)]
pub struct ContentKbSpec {
    pub topics: u64,
    pub articles: u64,
    pub entities: u64,
    pub facts: u64,
    pub relationships: u64,
    pub embedding_dim: usize,
}

/// Specification for multi-user knowledge base
#[derive(Debug, Clone)]
pub struct MultiUserKbSpec {
    pub entities: u64,
    pub relationships: u64,
    pub embedding_dim: usize,
    pub user_scenarios: u64,
}

/// Specification for production-scale knowledge base
#[derive(Debug, Clone)]
pub struct ProductionKbSpec {
    pub entities: u64,
    pub relationships: u64,
    pub embedding_dim: usize,
    pub update_frequency: Duration,
    pub user_load: u64,
}

/// Academic knowledge base for research simulations
#[derive(Debug, Clone)]
pub struct AcademicKnowledgeBase {
    pub entities: Vec<TestEntity>,
    pub relationships: Vec<(EntityKey, EntityKey, TestRelationship)>,
    pub embeddings: HashMap<EntityKey, Vec<f32>>,
    pub ground_truth_citations: HashMap<EntityKey, Vec<EntityKey>>,
    pub entity_count: u64,
    pub relationship_count: u64,
    pub embedding_count: u64,
}

/// Content knowledge base for content creation simulations
#[derive(Debug, Clone)]
pub struct ContentKnowledgeBase {
    pub entities: Vec<TestEntity>,
    pub relationships: Vec<(EntityKey, EntityKey, TestRelationship)>,
    pub embeddings: HashMap<EntityKey, Vec<f32>>,
    pub test_claims: Vec<String>,
    pub claim_truth_values: HashMap<String, bool>,
    pub entity_count: u64,
    pub relationship_count: u64,
    pub embedding_count: u64,
}

/// Multi-user knowledge base for concurrent testing
#[derive(Debug, Clone)]
pub struct MultiUserKnowledgeBase {
    pub entities: Vec<TestEntity>,
    pub relationships: Vec<(EntityKey, EntityKey, TestRelationship)>,
    pub embeddings: HashMap<EntityKey, Vec<f32>>,
    pub user_queries: Vec<Vec<String>>,
    pub sample_entities: Vec<EntityKey>,
    pub entity_count: u64,
    pub relationship_count: u64,
    pub embedding_count: u64,
}

/// Production-scale knowledge base for stress testing
#[derive(Debug, Clone)]
pub struct ProductionKnowledgeBase {
    pub initial_entities: Vec<TestEntity>,
    pub initial_relationships: Vec<(EntityKey, EntityKey, TestRelationship)>,
    pub initial_embeddings: HashMap<EntityKey, Vec<f32>>,
    pub update_stream: Vec<SystemUpdate>,
    pub user_patterns: Vec<UserPattern>,
    pub entities: u64,
    pub relationships: u64,
    pub embeddings: u64,
}

/// E2E-specific data generator
pub struct E2EDataGenerator {
    rng: DeterministicRng,
    base_generator: ComprehensiveDataGenerator,
}

impl E2EDataGenerator {
    /// Create a new E2E data generator
    pub fn new(seed: u64) -> Self {
        let mut rng = DeterministicRng::new(seed);
        rng.set_label("e2e_data_generator".to_string());
        let base_generator = ComprehensiveDataGenerator::new(seed);

        Self {
            rng,
            base_generator,
        }
    }

    /// Generate academic knowledge base for research simulations
    pub fn generate_academic_knowledge_base(&mut self, spec: AcademicKbSpec) -> Result<AcademicKnowledgeBase> {
        let mut entities = Vec::new();
        let mut relationships = Vec::new();
        let mut embeddings = HashMap::new();
        let mut ground_truth_citations = HashMap::new();

        // Generate papers
        for i in 0..spec.papers {
            let paper_key = EntityKey::new(format!("paper_{}", i));
            let paper = TestEntity {
                key: paper_key,
                entity_type: "paper".to_string(),
                attributes: [
                    ("title".to_string(), format!("Paper Title {}", i)),
                    ("year".to_string(), self.rng.gen_range(spec.citation_years.clone()).to_string()),
                    ("field".to_string(), format!("field_{}", self.rng.gen_range(0..spec.fields))),
                ].into_iter().collect(),
            };
            entities.push(paper);

            // Generate embedding for paper
            let embedding: Vec<f32> = (0..spec.embedding_dim)
                .map(|_| self.rng.gen_normal(0.0, 1.0) as f32)
                .collect();
            embeddings.insert(paper_key, embedding);

            // Generate ground truth citations (for validation)
            let citation_count = self.rng.gen_range(0..=20);
            let mut citations = Vec::new();
            for _ in 0..citation_count {
                let cited_paper_id = self.rng.gen_range(0..spec.papers);
                if cited_paper_id != i {
                    citations.push(EntityKey::new(format!("paper_{}", cited_paper_id)));
                }
            }
            ground_truth_citations.insert(paper_key, citations);
        }

        // Generate authors
        for i in 0..spec.authors {
            let author_key = EntityKey::new(format!("author_{}", i));
            let author = TestEntity {
                key: author_key,
                entity_type: "author".to_string(),
                attributes: [
                    ("name".to_string(), format!("Author {}", i)),
                    ("institution".to_string(), format!("Institution {}", self.rng.gen_range(0..100))),
                    ("field".to_string(), format!("field_{}", self.rng.gen_range(0..spec.fields))),
                ].into_iter().collect(),
            };
            entities.push(author);

            // Generate embedding for author
            let embedding: Vec<f32> = (0..spec.embedding_dim)
                .map(|_| self.rng.gen_normal(0.0, 1.0) as f32)
                .collect();
            embeddings.insert(author_key, embedding);
        }

        // Generate venues
        for i in 0..spec.venues {
            let venue_key = EntityKey::new(format!("venue_{}", i));
            let venue = TestEntity {
                key: venue_key,
                entity_type: "venue".to_string(),
                attributes: [
                    ("name".to_string(), format!("Venue {}", i)),
                    ("type".to_string(), if self.rng.gen_bool(0.7) { "conference" } else { "journal" }.to_string()),
                    ("field".to_string(), format!("field_{}", self.rng.gen_range(0..spec.fields))),
                ].into_iter().collect(),
            };
            entities.push(venue);

            // Generate embedding for venue
            let embedding: Vec<f32> = (0..spec.embedding_dim)
                .map(|_| self.rng.gen_normal(0.0, 1.0) as f32)
                .collect();
            embeddings.insert(venue_key, embedding);
        }

        // Generate relationships
        // Paper-Author relationships
        for i in 0..spec.papers {
            let paper_key = EntityKey::new(format!("paper_{}", i));
            let author_count = self.rng.gen_range(1..=5);
            
            for _ in 0..author_count {
                let author_id = self.rng.gen_range(0..spec.authors);
                let author_key = EntityKey::new(format!("author_{}", author_id));
                
                relationships.push((
                    paper_key,
                    author_key,
                    TestRelationship {
                        name: "authored_by".to_string(),
                        properties: HashMap::new(),
                    }
                ));
            }
        }

        // Paper-Venue relationships
        for i in 0..spec.papers {
            let paper_key = EntityKey::new(format!("paper_{}", i));
            let venue_id = self.rng.gen_range(0..spec.venues);
            let venue_key = EntityKey::new(format!("venue_{}", venue_id));
            
            relationships.push((
                paper_key,
                venue_key,
                TestRelationship {
                    name: "published_in".to_string(),
                    properties: HashMap::new(),
                }
            ));
        }

        // Citation relationships
        for (paper_key, cited_papers) in &ground_truth_citations {
            for cited_paper_key in cited_papers {
                relationships.push((
                    *paper_key,
                    *cited_paper_key,
                    TestRelationship {
                        name: "cites".to_string(),
                        properties: HashMap::new(),
                    }
                ));
                
                // Reverse relationship
                relationships.push((
                    *cited_paper_key,
                    *paper_key,
                    TestRelationship {
                        name: "cited_by".to_string(),
                        properties: HashMap::new(),
                    }
                ));
            }
        }

        Ok(AcademicKnowledgeBase {
            entity_count: entities.len() as u64,
            relationship_count: relationships.len() as u64,
            embedding_count: embeddings.len() as u64,
            entities,
            relationships,
            embeddings,
            ground_truth_citations,
        })
    }

    /// Generate content knowledge base for content creation simulations
    pub fn generate_content_knowledge_base(&mut self, spec: ContentKbSpec) -> Result<ContentKnowledgeBase> {
        let mut entities = Vec::new();
        let mut relationships = Vec::new();
        let mut embeddings = HashMap::new();
        let mut test_claims = Vec::new();
        let mut claim_truth_values = HashMap::new();

        // Generate topics
        for i in 0..spec.topics {
            let topic_key = EntityKey::new(format!("topic_{}", i));
            let topic = TestEntity {
                key: topic_key,
                entity_type: "topic".to_string(),
                attributes: [
                    ("name".to_string(), format!("Topic {}", i)),
                    ("category".to_string(), format!("category_{}", self.rng.gen_range(0..10))),
                    ("importance".to_string(), self.rng.gen_range(1..=10).to_string()),
                ].into_iter().collect(),
            };
            entities.push(topic);

            // Generate embedding
            let embedding: Vec<f32> = (0..spec.embedding_dim)
                .map(|_| self.rng.gen_normal(0.0, 1.0) as f32)
                .collect();
            embeddings.insert(topic_key, embedding);
        }

        // Generate articles
        for i in 0..spec.articles {
            let article_key = EntityKey::new(format!("article_{}", i));
            let article = TestEntity {
                key: article_key,
                entity_type: "article".to_string(),
                attributes: [
                    ("title".to_string(), format!("Article {}", i)),
                    ("content_type".to_string(), ["news", "analysis", "opinion", "tutorial"][self.rng.gen_range(0..4)].to_string()),
                    ("reliability".to_string(), self.rng.gen_range(0.5..1.0).to_string()),
                ].into_iter().collect(),
            };
            entities.push(article);

            // Generate embedding
            let embedding: Vec<f32> = (0..spec.embedding_dim)
                .map(|_| self.rng.gen_normal(0.0, 1.0) as f32)
                .collect();
            embeddings.insert(article_key, embedding);
        }

        // Generate other entities (people, organizations, concepts)
        let remaining_entities = spec.entities.saturating_sub(spec.topics + spec.articles);
        for i in 0..remaining_entities {
            let entity_key = EntityKey::new(format!("entity_{}", i));
            let entity_types = ["person", "organization", "concept", "location", "event"];
            let entity_type = entity_types[self.rng.gen_range(0..entity_types.len())];
            
            let entity = TestEntity {
                key: entity_key,
                entity_type: entity_type.to_string(),
                attributes: [
                    ("name".to_string(), format!("{} {}", entity_type, i)),
                    ("verified".to_string(), self.rng.gen_bool(0.8).to_string()),
                ].into_iter().collect(),
            };
            entities.push(entity);

            // Generate embedding
            let embedding: Vec<f32> = (0..spec.embedding_dim)
                .map(|_| self.rng.gen_normal(0.0, 1.0) as f32)
                .collect();
            embeddings.insert(entity_key, embedding);
        }

        // Generate relationships
        for _ in 0..spec.relationships {
            let source_idx = self.rng.gen_range(0..entities.len());
            let target_idx = self.rng.gen_range(0..entities.len());
            
            if source_idx != target_idx {
                let source_key = entities[source_idx].key;
                let target_key = entities[target_idx].key;
                
                let relationship_types = ["mentions", "relates_to", "contains", "supports", "contradicts"];
                let rel_type = relationship_types[self.rng.gen_range(0..relationship_types.len())];
                
                relationships.push((
                    source_key,
                    target_key,
                    TestRelationship {
                        name: rel_type.to_string(),
                        properties: [
                            ("strength".to_string(), self.rng.gen_range(0.1..1.0).to_string()),
                        ].into_iter().collect(),
                    }
                ));
            }
        }

        // Generate test claims for fact checking
        for i in 0..100 { // Generate 100 test claims
            let claim = format!("Test claim {}: This is a factual statement about topic {}.", 
                               i, self.rng.gen_range(0..spec.topics));
            let is_true = self.rng.gen_bool(0.7); // 70% of claims are true
            
            test_claims.push(claim.clone());
            claim_truth_values.insert(claim, is_true);
        }

        Ok(ContentKnowledgeBase {
            entity_count: entities.len() as u64,
            relationship_count: relationships.len() as u64,
            embedding_count: embeddings.len() as u64,
            entities,
            relationships,
            embeddings,
            test_claims,
            claim_truth_values,
        })
    }

    /// Generate multi-user knowledge base for concurrent testing
    pub fn generate_multi_user_knowledge_base(&mut self, spec: MultiUserKbSpec) -> Result<MultiUserKnowledgeBase> {
        let mut entities = Vec::new();
        let mut relationships = Vec::new();
        let mut embeddings = HashMap::new();
        let mut user_queries = Vec::new();
        let mut sample_entities = Vec::new();

        // Generate entities
        for i in 0..spec.entities {
            let entity_key = EntityKey::new(format!("entity_{}", i));
            let entity_types = ["document", "user", "product", "category", "tag"];
            let entity_type = entity_types[self.rng.gen_range(0..entity_types.len())];
            
            let entity = TestEntity {
                key: entity_key,
                entity_type: entity_type.to_string(),
                attributes: [
                    ("name".to_string(), format!("{} {}", entity_type, i)),
                    ("created_at".to_string(), format!("2024-01-{:02}", (i % 28) + 1)),
                    ("popularity".to_string(), self.rng.gen_range(0..1000).to_string()),
                ].into_iter().collect(),
            };
            entities.push(entity);

            // Generate embedding
            let embedding: Vec<f32> = (0..spec.embedding_dim)
                .map(|_| self.rng.gen_normal(0.0, 1.0) as f32)
                .collect();
            embeddings.insert(entity_key, embedding);

            // Add to sample entities for consistency checking
            if i < 1000 {
                sample_entities.push(entity_key);
            }
        }

        // Generate relationships
        for _ in 0..spec.relationships {
            let source_idx = self.rng.gen_range(0..entities.len());
            let target_idx = self.rng.gen_range(0..entities.len());
            
            if source_idx != target_idx {
                let source_key = entities[source_idx].key;
                let target_key = entities[target_idx].key;
                
                let relationship_types = ["interacts_with", "similar_to", "contains", "belongs_to"];
                let rel_type = relationship_types[self.rng.gen_range(0..relationship_types.len())];
                
                relationships.push((
                    source_key,
                    target_key,
                    TestRelationship {
                        name: rel_type.to_string(),
                        properties: HashMap::new(),
                    }
                ));
            }
        }

        // Generate user query patterns
        for scenario_id in 0..spec.user_scenarios {
            let mut scenario_queries = Vec::new();
            let queries_per_scenario = self.rng.gen_range(10..=50);
            
            for query_id in 0..queries_per_scenario {
                let query_types = [
                    "search for similar entities",
                    "find connections between entities",
                    "get entity details",
                    "list entities by type",
                    "find popular entities",
                ];
                
                let query_type = query_types[self.rng.gen_range(0..query_types.len())];
                let entity_ref = self.rng.gen_range(0..spec.entities);
                
                let query = format!("{} related to entity_{}", query_type, entity_ref);
                scenario_queries.push(query);
            }
            
            user_queries.push(scenario_queries);
        }

        Ok(MultiUserKnowledgeBase {
            entity_count: entities.len() as u64,
            relationship_count: relationships.len() as u64,
            embedding_count: embeddings.len() as u64,
            entities,
            relationships,
            embeddings,
            user_queries,
            sample_entities,
        })
    }

    /// Generate production-scale knowledge base for stress testing
    pub fn generate_production_scale_kb(&mut self, spec: ProductionKbSpec) -> Result<ProductionKnowledgeBase> {
        // Generate initial state
        let mut initial_entities = Vec::new();
        let mut initial_relationships = Vec::new();
        let mut initial_embeddings = HashMap::new();
        let mut update_stream = Vec::new();
        let mut user_patterns = Vec::new();

        // Generate initial entities (smaller initial set, will be expanded via updates)
        let initial_entity_count = spec.entities / 4; // Start with 25% of final size
        for i in 0..initial_entity_count {
            let entity_key = EntityKey::new(format!("prod_entity_{}", i));
            let entity = TestEntity {
                key: entity_key,
                entity_type: "document".to_string(),
                attributes: [
                    ("title".to_string(), format!("Production Document {}", i)),
                    ("timestamp".to_string(), "2024-01-01T00:00:00Z".to_string()),
                    ("version".to_string(), "1.0".to_string()),
                ].into_iter().collect(),
            };
            initial_entities.push(entity);

            // Generate embedding
            let embedding: Vec<f32> = (0..spec.embedding_dim)
                .map(|_| self.rng.gen_normal(0.0, 1.0) as f32)
                .collect();
            initial_embeddings.insert(entity_key, embedding);
        }

        // Generate initial relationships
        let initial_relationship_count = spec.relationships / 4;
        for _ in 0..initial_relationship_count {
            let source_idx = self.rng.gen_range(0..initial_entities.len());
            let target_idx = self.rng.gen_range(0..initial_entities.len());
            
            if source_idx != target_idx {
                let source_key = initial_entities[source_idx].key;
                let target_key = initial_entities[target_idx].key;
                
                initial_relationships.push((
                    source_key,
                    target_key,
                    TestRelationship {
                        name: "references".to_string(),
                        properties: HashMap::new(),
                    }
                ));
            }
        }

        // Generate update stream for incremental growth
        let remaining_entities = spec.entities - initial_entity_count;
        for i in initial_entity_count..(initial_entity_count + remaining_entities) {
            let entity_key = EntityKey::new(format!("prod_entity_{}", i));
            let entity = TestEntity {
                key: entity_key,
                entity_type: "document".to_string(),
                attributes: [
                    ("title".to_string(), format!("Production Document {}", i)),
                    ("timestamp".to_string(), "2024-01-01T00:00:00Z".to_string()),
                    ("version".to_string(), "1.0".to_string()),
                ].into_iter().collect(),
            };

            update_stream.push(SystemUpdate::AddEntity(entity));

            // Add embedding update
            let embedding: Vec<f32> = (0..spec.embedding_dim)
                .map(|_| self.rng.gen_normal(0.0, 1.0) as f32)
                .collect();
            update_stream.push(SystemUpdate::AddEmbedding(entity_key, embedding));
        }

        // Generate user patterns for load simulation
        for user_id in 0..spec.user_load {
            let queries_count = self.rng.gen_range(10..=100);
            let mut pattern_queries = Vec::new();
            
            for _ in 0..queries_count {
                let query_type = match self.rng.gen_range(0..4) {
                    0 => "search",
                    1 => "similar",
                    2 => "traverse",
                    _ => "aggregate",
                };
                
                pattern_queries.push(McpToolRequest {
                    tool_name: format!("knowledge_{}", query_type),
                    arguments: serde_json::json!({
                        "query": format!("production query {} for user {}", query_type, user_id),
                        "max_results": 10
                    }),
                });
            }

            user_patterns.push(UserPattern {
                queries: pattern_queries,
                think_time_ms: self.rng.gen_range(100..=2000),
                session_duration: Duration::from_minutes(self.rng.gen_range(5..=60)),
            });
        }

        Ok(ProductionKnowledgeBase {
            entities: spec.entities,
            relationships: spec.relationships,
            embeddings: spec.entities, // One embedding per entity
            initial_entities,
            initial_relationships,
            initial_embeddings,
            update_stream,
            user_patterns,
        })
    }
}

// Supporting types and structures

/// Entity key for test entities
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EntityKey(u64);

impl EntityKey {
    pub fn new(name: String) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        name.hash(&mut hasher);
        Self(hasher.finish())
    }

    pub fn from_hash(hash_str: &str) -> Self {
        // Parse hex string or use string hash
        if let Ok(hash) = u64::from_str_radix(hash_str, 16) {
            Self(hash)
        } else {
            Self::new(hash_str.to_string())
        }
    }

    pub fn to_string(&self) -> String {
        format!("{:016x}", self.0)
    }
}

/// Test entity structure
#[derive(Debug, Clone)]
pub struct TestEntity {
    pub key: EntityKey,
    pub entity_type: String,
    pub attributes: HashMap<String, String>,
}

/// Test relationship structure
#[derive(Debug, Clone)]
pub struct TestRelationship {
    pub name: String,
    pub properties: HashMap<String, String>,
}

/// System update for incremental testing
#[derive(Debug, Clone)]
pub enum SystemUpdate {
    AddEntity(TestEntity),
    UpdateEntity(EntityKey, HashMap<String, String>),
    AddRelationship(EntityKey, EntityKey, TestRelationship),
    AddEmbedding(EntityKey, Vec<f32>),
}

/// User pattern for load testing
#[derive(Debug, Clone)]
pub struct UserPattern {
    pub queries: Vec<McpToolRequest>,
    pub think_time_ms: u64,
    pub session_duration: Duration,
}

/// MCP tool request structure
#[derive(Debug, Clone)]
pub struct McpToolRequest {
    pub tool_name: String,
    pub arguments: serde_json::Value,
}

impl Default for AcademicKbSpec {
    fn default() -> Self {
        Self {
            papers: 1000,
            authors: 500,
            venues: 50,
            fields: 20,
            citation_years: 2000..2024,
            embedding_dim: 256,
        }
    }
}

impl Default for ContentKbSpec {
    fn default() -> Self {
        Self {
            topics: 100,
            articles: 1000,
            entities: 5000,
            facts: 10000,
            relationships: 15000,
            embedding_dim: 256,
        }
    }
}

impl Default for MultiUserKbSpec {
    fn default() -> Self {
        Self {
            entities: 10000,
            relationships: 25000,
            embedding_dim: 256,
            user_scenarios: 20,
        }
    }
}

impl Default for ProductionKbSpec {
    fn default() -> Self {
        Self {
            entities: 50000,
            relationships: 125000,
            embedding_dim: 512,
            update_frequency: Duration::from_secs(60),
            user_load: 50,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_e2e_data_generator_creation() {
        let generator = E2EDataGenerator::new(42);
        // Should create without errors
    }

    #[test]
    fn test_academic_kb_generation() {
        let mut generator = E2EDataGenerator::new(42);
        let spec = AcademicKbSpec::default();
        
        let kb = generator.generate_academic_knowledge_base(spec.clone()).unwrap();
        
        assert_eq!(kb.entity_count, spec.papers + spec.authors + spec.venues);
        assert!(!kb.embeddings.is_empty());
        assert!(!kb.ground_truth_citations.is_empty());
    }

    #[test]
    fn test_content_kb_generation() {
        let mut generator = E2EDataGenerator::new(42);
        let spec = ContentKbSpec::default();
        
        let kb = generator.generate_content_knowledge_base(spec).unwrap();
        
        assert!(kb.entity_count > 0);
        assert!(!kb.test_claims.is_empty());
        assert_eq!(kb.test_claims.len(), kb.claim_truth_values.len());
    }

    #[test]
    fn test_multi_user_kb_generation() {
        let mut generator = E2EDataGenerator::new(42);
        let spec = MultiUserKbSpec::default();
        
        let kb = generator.generate_multi_user_knowledge_base(spec.clone()).unwrap();
        
        assert_eq!(kb.entity_count, spec.entities);
        assert_eq!(kb.user_queries.len(), spec.user_scenarios as usize);
        assert!(!kb.sample_entities.is_empty());
    }

    #[test]
    fn test_production_kb_generation() {
        let mut generator = E2EDataGenerator::new(42);
        let spec = ProductionKbSpec::default();
        
        let kb = generator.generate_production_scale_kb(spec.clone()).unwrap();
        
        assert!(kb.initial_entities.len() < spec.entities as usize);
        assert!(!kb.update_stream.is_empty());
        assert_eq!(kb.user_patterns.len(), spec.user_load as usize);
    }

    #[test]
    fn test_entity_key_creation() {
        let key1 = EntityKey::new("test_entity".to_string());
        let key2 = EntityKey::new("test_entity".to_string());
        let key3 = EntityKey::new("different_entity".to_string());
        
        assert_eq!(key1, key2); // Same string should produce same key
        assert_ne!(key1, key3); // Different strings should produce different keys
    }

    #[test]
    fn test_entity_key_conversion() {
        let key = EntityKey::new("test".to_string());
        let key_string = key.to_string();
        let recovered_key = EntityKey::from_hash(&key_string);
        
        assert_eq!(key, recovered_key);
    }

    #[test]
    fn test_default_specifications() {
        let academic_spec = AcademicKbSpec::default();
        assert_eq!(academic_spec.papers, 1000);
        assert_eq!(academic_spec.embedding_dim, 256);
        
        let content_spec = ContentKbSpec::default();
        assert_eq!(content_spec.entities, 5000);
        assert_eq!(content_spec.embedding_dim, 256);
        
        let multi_user_spec = MultiUserKbSpec::default();
        assert_eq!(multi_user_spec.entities, 10000);
        assert_eq!(multi_user_spec.user_scenarios, 20);
        
        let production_spec = ProductionKbSpec::default();
        assert_eq!(production_spec.entities, 50000);
        assert_eq!(production_spec.user_load, 50);
    }
}