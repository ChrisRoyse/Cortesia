// Test Data Generator
// Generates synthetic test data for integration testing

use std::collections::{HashMap, HashSet};
use rand::{Rng, SeedableRng};
use rand::distributions::{Distribution, Uniform};
use rand::rngs::StdRng;
use rand_distr::{Normal, Zipf};

use crate::entity::{Entity, EntityKey};
use crate::relationship::{Relationship, RelationshipType};
use crate::embedding::EmbeddingVector;

/// Test data generator for creating synthetic test scenarios
pub struct TestDataGenerator {
    rng: StdRng,
}

/// Graph specification for generation
#[derive(Debug, Clone)]
pub struct GraphSpec {
    pub entity_count: u64,
    pub relationship_count: u64,
    pub topology: TopologyType,
    pub clustering_coefficient: f32,
}

/// Graph topology types
#[derive(Debug, Clone)]
pub enum TopologyType {
    Random,
    ScaleFree { exponent: f64 },
    SmallWorld { rewiring_prob: f64 },
    Hierarchical { levels: u32 },
}

/// Generated test graph data
pub struct TestGraphData {
    pub entities: HashMap<EntityKey, Entity>,
    pub relationships: Vec<(EntityKey, EntityKey, Relationship)>,
}

/// Academic scenario data
pub struct AcademicScenario {
    pub entities: HashMap<EntityKey, Entity>,
    pub relationships: Vec<(EntityKey, EntityKey, Relationship)>,
    pub embeddings: HashMap<EntityKey, Vec<f32>>,
    pub central_entities: Vec<EntityKey>,
}

/// Membership test data
pub struct MembershipTestData {
    pub entities: Vec<Entity>,
    pub relationships: Vec<(EntityKey, EntityKey, Relationship)>,
    pub known_entities: HashSet<EntityKey>,
}

/// Attributed graph data
pub struct AttributedGraphData {
    pub entities: HashMap<EntityKey, Entity>,
    pub relationships: Vec<(EntityKey, EntityKey, Relationship)>,
}

/// Embedding test scenario
pub struct EmbeddingTestScenario {
    pub embeddings: HashMap<EntityKey, Vec<f32>>,
    pub query_embeddings: Vec<Vec<f32>>,
    pub test_entities: Vec<EntityKey>,
}

/// Correlated graph and embeddings
pub struct CorrelatedGraphEmbeddings {
    pub entities: HashMap<EntityKey, Entity>,
    pub relationships: Vec<(EntityKey, EntityKey, Relationship)>,
    pub embeddings: HashMap<EntityKey, Vec<f32>>,
    pub test_entity_pairs: Vec<(EntityKey, EntityKey)>,
}

/// Performance test scenario
pub struct PerformanceTestScenario {
    pub entities: Vec<Entity>,
    pub relationships: Vec<(EntityKey, EntityKey, Relationship)>,
    pub relationship_count: u64,
}

/// Federation scenario data
pub struct FederationScenario {
    pub shards: Vec<DatabaseShard>,
    pub cross_shard_relationships: Vec<CrossShardRelationship>,
}

/// Database shard data
pub struct DatabaseShard {
    pub entities: HashSet<EntityKey>,
    pub relationships: Vec<ShardRelationship>,
}

/// Shard relationship
pub struct ShardRelationship {
    pub source: EntityKey,
    pub target: EntityKey,
    pub relationship: Relationship,
}

/// Cross-shard relationship
pub struct CrossShardRelationship {
    pub source_shard: usize,
    pub target_shard: usize,
    pub source_entity: EntityKey,
    pub target_entity: EntityKey,
    pub relationship: Relationship,
}

/// Memory test scenario
pub struct MemoryTestScenario {
    pub entities: Vec<Entity>,
    pub relationships: Vec<(EntityKey, EntityKey, Relationship)>,
    pub embeddings: HashMap<EntityKey, Vec<f32>>,
    pub relationship_count: u64,
}

/// Concurrent test scenario
pub struct ConcurrentTestScenario {
    pub entities: Vec<Entity>,
    pub relationships: Vec<(EntityKey, EntityKey, Relationship)>,
    pub embeddings: HashMap<EntityKey, Vec<f32>>,
    pub test_entities: Vec<EntityKey>,
}

/// Performance graph data
pub struct PerformanceGraphData {
    pub entities: Vec<Entity>,
    pub relationships: Vec<(EntityKey, EntityKey, Relationship)>,
    pub embeddings: HashMap<EntityKey, Vec<f32>>,
}

impl TestDataGenerator {
    /// Create a new test data generator
    pub fn new() -> Self {
        Self::with_seed(42)
    }

    /// Create a generator with a specific seed
    pub fn with_seed(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Generate a graph with specified properties
    pub fn generate_graph(&mut self, spec: &GraphSpec) -> TestGraphData {
        let mut entities = HashMap::new();
        let mut relationships = Vec::new();

        // Create entities
        for i in 0..spec.entity_count {
            let key = EntityKey::from_hash(format!("entity_{}", i));
            let entity = Entity::new(key, format!("Entity {}", i))
                .with_attribute("index", i.to_string())
                .with_attribute("type", self.random_entity_type());
            entities.insert(key, entity);
        }

        // Create relationships based on topology
        match &spec.topology {
            TopologyType::Random => {
                self.generate_random_relationships(
                    &entities,
                    spec.relationship_count,
                    &mut relationships,
                );
            }
            TopologyType::ScaleFree { exponent } => {
                self.generate_scale_free_relationships(
                    &entities,
                    spec.relationship_count,
                    *exponent,
                    &mut relationships,
                );
            }
            TopologyType::SmallWorld { rewiring_prob } => {
                self.generate_small_world_relationships(
                    &entities,
                    spec.relationship_count,
                    *rewiring_prob,
                    &mut relationships,
                );
            }
            TopologyType::Hierarchical { levels } => {
                self.generate_hierarchical_relationships(
                    &entities,
                    spec.relationship_count,
                    *levels,
                    &mut relationships,
                );
            }
        }

        TestGraphData {
            entities,
            relationships,
        }
    }

    /// Generate random relationships
    fn generate_random_relationships(
        &mut self,
        entities: &HashMap<EntityKey, Entity>,
        count: u64,
        relationships: &mut Vec<(EntityKey, EntityKey, Relationship)>,
    ) {
        let entity_keys: Vec<_> = entities.keys().cloned().collect();
        let uniform = Uniform::new(0, entity_keys.len());

        for i in 0..count {
            let source_idx = uniform.sample(&mut self.rng);
            let mut target_idx = uniform.sample(&mut self.rng);
            
            // Avoid self-loops
            while target_idx == source_idx {
                target_idx = uniform.sample(&mut self.rng);
            }

            let relationship = Relationship::new(
                format!("rel_{}", i),
                RelationshipType::Directed,
                self.rng.gen_range(0.1..1.0),
            );

            relationships.push((
                entity_keys[source_idx],
                entity_keys[target_idx],
                relationship,
            ));
        }
    }

    /// Generate scale-free (power-law) relationships
    fn generate_scale_free_relationships(
        &mut self,
        entities: &HashMap<EntityKey, Entity>,
        count: u64,
        exponent: f64,
        relationships: &mut Vec<(EntityKey, EntityKey, Relationship)>,
    ) {
        let entity_keys: Vec<_> = entities.keys().cloned().collect();
        let n = entity_keys.len();
        
        // Use preferential attachment
        let mut degrees: Vec<usize> = vec![1; n]; // Start with degree 1 for all
        
        for i in 0..count {
            // Select source with probability proportional to degree
            let total_degree: usize = degrees.iter().sum();
            let mut rnd = self.rng.gen_range(0..total_degree);
            let mut source_idx = 0;
            
            for (idx, &degree) in degrees.iter().enumerate() {
                if rnd < degree {
                    source_idx = idx;
                    break;
                }
                rnd -= degree;
            }
            
            // Select target similarly
            let mut target_idx = source_idx;
            while target_idx == source_idx {
                let mut rnd = self.rng.gen_range(0..total_degree);
                target_idx = 0;
                
                for (idx, &degree) in degrees.iter().enumerate() {
                    if rnd < degree {
                        target_idx = idx;
                        break;
                    }
                    rnd -= degree;
                }
            }
            
            // Update degrees
            degrees[source_idx] += 1;
            degrees[target_idx] += 1;
            
            let relationship = Relationship::new(
                format!("rel_{}", i),
                RelationshipType::Directed,
                self.rng.gen_range(0.1..1.0),
            );
            
            relationships.push((
                entity_keys[source_idx],
                entity_keys[target_idx],
                relationship,
            ));
        }
    }

    /// Generate small-world relationships
    fn generate_small_world_relationships(
        &mut self,
        entities: &HashMap<EntityKey, Entity>,
        count: u64,
        rewiring_prob: f64,
        relationships: &mut Vec<(EntityKey, EntityKey, Relationship)>,
    ) {
        let entity_keys: Vec<_> = entities.keys().cloned().collect();
        let n = entity_keys.len();
        
        // Start with ring lattice
        let k = (count as usize / n).max(2);
        
        for i in 0..n {
            for j in 1..=k/2 {
                let target = (i + j) % n;
                
                // Rewire with probability
                let actual_target = if self.rng.gen::<f64>() < rewiring_prob {
                    self.rng.gen_range(0..n)
                } else {
                    target
                };
                
                if actual_target != i {
                    let relationship = Relationship::new(
                        format!("rel_{}_{}", i, j),
                        RelationshipType::Directed,
                        self.rng.gen_range(0.1..1.0),
                    );
                    
                    relationships.push((
                        entity_keys[i],
                        entity_keys[actual_target],
                        relationship,
                    ));
                }
            }
        }
    }

    /// Generate hierarchical relationships
    fn generate_hierarchical_relationships(
        &mut self,
        entities: &HashMap<EntityKey, Entity>,
        count: u64,
        levels: u32,
        relationships: &mut Vec<(EntityKey, EntityKey, Relationship)>,
    ) {
        let entity_keys: Vec<_> = entities.keys().cloned().collect();
        let n = entity_keys.len();
        let entities_per_level = n / levels as usize;
        
        let mut rel_count = 0;
        
        // Connect within levels
        for level in 0..levels {
            let start = (level as usize) * entities_per_level;
            let end = ((level + 1) as usize * entities_per_level).min(n);
            
            for i in start..end {
                for j in (i+1)..end {
                    if self.rng.gen::<f64>() < 0.1 && rel_count < count {
                        let relationship = Relationship::new(
                            format!("rel_{}", rel_count),
                            RelationshipType::Undirected,
                            self.rng.gen_range(0.5..1.0),
                        );
                        
                        relationships.push((
                            entity_keys[i],
                            entity_keys[j],
                            relationship,
                        ));
                        rel_count += 1;
                    }
                }
            }
        }
        
        // Connect between levels
        for level in 0..(levels-1) {
            let start1 = (level as usize) * entities_per_level;
            let end1 = ((level + 1) as usize * entities_per_level).min(n);
            let start2 = ((level + 1) as usize) * entities_per_level;
            let end2 = ((level + 2) as usize * entities_per_level).min(n);
            
            for i in start1..end1 {
                for j in start2..end2 {
                    if self.rng.gen::<f64>() < 0.2 && rel_count < count {
                        let relationship = Relationship::new(
                            format!("rel_{}", rel_count),
                            RelationshipType::Directed,
                            self.rng.gen_range(0.3..0.8),
                        );
                        
                        relationships.push((
                            entity_keys[i],
                            entity_keys[j],
                            relationship,
                        ));
                        rel_count += 1;
                    }
                }
            }
        }
    }

    /// Generate random entity type
    fn random_entity_type(&mut self) -> String {
        let types = ["paper", "author", "venue", "topic", "institution"];
        types[self.rng.gen_range(0..types.len())].to_string()
    }

    /// Generate academic scenario with papers, authors, venues
    pub fn generate_academic_scenario(
        &mut self,
        paper_count: usize,
        author_count: usize,
        venue_count: usize,
        embedding_dim: usize,
    ) -> AcademicScenario {
        let mut entities = HashMap::new();
        let mut relationships = Vec::new();
        let mut embeddings = HashMap::new();
        let mut central_entities = Vec::new();

        // Create papers
        let mut paper_keys = Vec::new();
        for i in 0..paper_count {
            let key = EntityKey::from_hash(format!("paper_{}", i));
            let entity = Entity::new(key, format!("Paper {}", i))
                .with_attribute("type", "paper")
                .with_attribute("year", self.rng.gen_range(2000..2025).to_string())
                .with_attribute("topic", self.random_topic());
            
            entities.insert(key, entity);
            paper_keys.push(key);
            
            // Generate embedding
            let embedding = self.generate_embedding(embedding_dim);
            embeddings.insert(key, embedding);
        }

        // Create authors
        let mut author_keys = Vec::new();
        for i in 0..author_count {
            let key = EntityKey::from_hash(format!("author_{}", i));
            let entity = Entity::new(key, format!("Author {}", i))
                .with_attribute("type", "author")
                .with_attribute("institution", format!("Institution {}", i % 20));
            
            entities.insert(key, entity);
            author_keys.push(key);
            
            // Generate embedding
            let embedding = self.generate_embedding(embedding_dim);
            embeddings.insert(key, embedding);
        }

        // Create venues
        let mut venue_keys = Vec::new();
        for i in 0..venue_count {
            let key = EntityKey::from_hash(format!("venue_{}", i));
            let entity = Entity::new(key, format!("Venue {}", i))
                .with_attribute("type", "venue")
                .with_attribute("impact_factor", self.rng.gen_range(1.0..10.0).to_string());
            
            entities.insert(key, entity);
            venue_keys.push(key);
            
            // Generate embedding
            let embedding = self.generate_embedding(embedding_dim);
            embeddings.insert(key, embedding);
        }

        // Create authorship relationships
        for (i, &paper_key) in paper_keys.iter().enumerate() {
            let author_count = self.rng.gen_range(1..5);
            for j in 0..author_count {
                let author_idx = self.rng.gen_range(0..author_keys.len());
                let author_key = author_keys[author_idx];
                
                let relationship = Relationship::new(
                    format!("authored_{}_{}",  i, j),
                    RelationshipType::Directed,
                    1.0,
                )
                .with_attribute("order", j.to_string());
                
                relationships.push((author_key, paper_key, relationship));
            }
        }

        // Create publication relationships
        for (i, &paper_key) in paper_keys.iter().enumerate() {
            let venue_idx = self.rng.gen_range(0..venue_keys.len());
            let venue_key = venue_keys[venue_idx];
            
            let relationship = Relationship::new(
                format!("published_in_{}", i),
                RelationshipType::Directed,
                1.0,
            );
            
            relationships.push((paper_key, venue_key, relationship));
        }

        // Create citation relationships
        for i in 0..paper_count {
            let citing_paper = paper_keys[i];
            let citation_count = self.rng.gen_range(0..10);
            
            for _ in 0..citation_count {
                let cited_idx = self.rng.gen_range(0..paper_count);
                if cited_idx != i {
                    let cited_paper = paper_keys[cited_idx];
                    
                    let relationship = Relationship::new(
                        format!("cites_{}_{}", i, cited_idx),
                        RelationshipType::Directed,
                        self.rng.gen_range(0.5..1.0),
                    );
                    
                    relationships.push((citing_paper, cited_paper, relationship));
                }
            }
        }

        // Select central entities (high-degree nodes)
        let mut degree_count: HashMap<EntityKey, usize> = HashMap::new();
        for (source, target, _) in &relationships {
            *degree_count.entry(*source).or_insert(0) += 1;
            *degree_count.entry(*target).or_insert(0) += 1;
        }
        
        let mut degrees: Vec<_> = degree_count.into_iter().collect();
        degrees.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
        
        central_entities = degrees.into_iter()
            .take(10)
            .map(|(key, _)| key)
            .collect();

        AcademicScenario {
            entities,
            relationships,
            embeddings,
            central_entities,
        }
    }

    /// Generate random topic
    fn random_topic(&mut self) -> String {
        let topics = [
            "machine learning",
            "natural language processing", 
            "computer vision",
            "robotics",
            "algorithms",
            "databases",
            "networks",
            "security",
        ];
        topics[self.rng.gen_range(0..topics.len())].to_string()
    }

    /// Generate random embedding vector
    pub fn generate_embedding(&mut self, dim: usize) -> Vec<f32> {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut embedding: Vec<f32> = (0..dim)
            .map(|_| normal.sample(&mut self.rng) as f32)
            .collect();
        
        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut embedding {
                *x /= norm;
            }
        }
        
        embedding
    }

    /// Generate membership test data
    pub fn generate_membership_test_data(
        &mut self,
        entity_count: u64,
        relationship_count: u64,
    ) -> MembershipTestData {
        let mut entities = Vec::new();
        let mut relationships = Vec::new();
        let mut known_entities = HashSet::new();

        // Create entities
        for i in 0..entity_count {
            let key = EntityKey::from_hash(format!("member_entity_{}", i));
            let entity = Entity::new(key, format!("Member Entity {}", i));
            entities.push(entity);
            known_entities.insert(key);
        }

        // Create relationships
        let uniform = Uniform::new(0, entity_count as usize);
        for i in 0..relationship_count {
            let source_idx = uniform.sample(&mut self.rng);
            let mut target_idx = uniform.sample(&mut self.rng);
            
            while target_idx == source_idx {
                target_idx = uniform.sample(&mut self.rng);
            }

            let relationship = Relationship::new(
                format!("member_rel_{}", i),
                RelationshipType::Directed,
                1.0,
            );

            relationships.push((
                entities[source_idx].key(),
                entities[target_idx].key(),
                relationship,
            ));
        }

        MembershipTestData {
            entities,
            relationships,
            known_entities,
        }
    }

    /// Generate attributed graph data
    pub fn generate_attributed_graph(
        &mut self,
        entity_count: usize,
        relationship_count: usize,
        attribute_names: Vec<&str>,
    ) -> AttributedGraphData {
        let mut entities = HashMap::new();
        let mut relationships = Vec::new();

        // Create entities with attributes
        for i in 0..entity_count {
            let key = EntityKey::from_hash(format!("attributed_entity_{}", i));
            let mut entity = Entity::new(key, format!("Attributed Entity {}", i));
            
            for attr_name in &attribute_names {
                let value = match *attr_name {
                    "type" => self.random_entity_type(),
                    "category" => ["AI", "Systems", "Theory", "Applications"][self.rng.gen_range(0..4)].to_string(),
                    "year" => self.rng.gen_range(2000..2025).to_string(),
                    "score" => self.rng.gen_range(0.0..100.0).to_string(),
                    _ => format!("{}_value_{}", attr_name, i),
                };
                entity = entity.with_attribute(attr_name, value);
            }
            
            entities.insert(key, entity);
        }

        // Create relationships
        let entity_keys: Vec<_> = entities.keys().cloned().collect();
        let uniform = Uniform::new(0, entity_keys.len());
        
        for i in 0..relationship_count {
            let source_idx = uniform.sample(&mut self.rng);
            let mut target_idx = uniform.sample(&mut self.rng);
            
            while target_idx == source_idx {
                target_idx = uniform.sample(&mut self.rng);
            }

            let relationship = Relationship::new(
                format!("attr_rel_{}", i),
                RelationshipType::Directed,
                self.rng.gen_range(0.1..1.0),
            );

            relationships.push((
                entity_keys[source_idx],
                entity_keys[target_idx],
                relationship,
            ));
        }

        AttributedGraphData {
            entities,
            relationships,
        }
    }

    /// Generate embedding test scenario
    pub fn generate_embedding_test_scenario(
        &mut self,
        entity_count: usize,
        embedding_dim: usize,
        query_count: usize,
    ) -> EmbeddingTestScenario {
        let mut embeddings = HashMap::new();
        let mut test_entities = Vec::new();

        // Generate entity embeddings
        for i in 0..entity_count {
            let key = EntityKey::from_hash(format!("embed_entity_{}", i));
            let embedding = self.generate_embedding(embedding_dim);
            embeddings.insert(key, embedding);
            
            if i < 100 {
                test_entities.push(key);
            }
        }

        // Generate query embeddings
        let query_embeddings = (0..query_count)
            .map(|_| self.generate_embedding(embedding_dim))
            .collect();

        EmbeddingTestScenario {
            embeddings,
            query_embeddings,
            test_entities,
        }
    }

    /// Generate correlated graph and embeddings
    pub fn generate_correlated_graph_embeddings(
        &mut self,
        entity_count: usize,
        relationship_count: usize,
        embedding_dim: usize,
        correlation_strength: f32,
    ) -> CorrelatedGraphEmbeddings {
        let spec = GraphSpec {
            entity_count: entity_count as u64,
            relationship_count: relationship_count as u64,
            topology: TopologyType::SmallWorld { rewiring_prob: 0.1 },
            clustering_coefficient: 0.3,
        };
        
        let graph_data = self.generate_graph(&spec);
        let mut embeddings = HashMap::new();
        
        // First, generate random embeddings
        for &key in graph_data.entities.keys() {
            let embedding = self.generate_embedding(embedding_dim);
            embeddings.insert(key, embedding);
        }
        
        // Then, adjust embeddings based on graph structure
        for _ in 0..3 { // Multiple iterations to propagate similarity
            let mut new_embeddings = embeddings.clone();
            
            for (source, target, _) in &graph_data.relationships {
                let source_embed = &embeddings[source];
                let target_embed = &embeddings[target];
                
                // Make connected nodes more similar
                let new_source = new_embeddings.get_mut(source).unwrap();
                for i in 0..embedding_dim {
                    new_source[i] = (1.0 - correlation_strength) * new_source[i]
                        + correlation_strength * target_embed[i];
                }
                
                let new_target = new_embeddings.get_mut(target).unwrap();
                for i in 0..embedding_dim {
                    new_target[i] = (1.0 - correlation_strength) * new_target[i]
                        + correlation_strength * source_embed[i];
                }
            }
            
            embeddings = new_embeddings;
        }
        
        // Normalize embeddings
        for embedding in embeddings.values_mut() {
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in embedding {
                    *x /= norm;
                }
            }
        }
        
        // Generate test pairs
        let entity_keys: Vec<_> = graph_data.entities.keys().cloned().collect();
        let mut test_entity_pairs = Vec::new();
        
        for _ in 0..200 {
            let idx1 = self.rng.gen_range(0..entity_keys.len());
            let idx2 = self.rng.gen_range(0..entity_keys.len());
            
            if idx1 != idx2 {
                test_entity_pairs.push((entity_keys[idx1], entity_keys[idx2]));
            }
        }
        
        CorrelatedGraphEmbeddings {
            entities: graph_data.entities,
            relationships: graph_data.relationships,
            embeddings,
            test_entity_pairs,
        }
    }

    /// Generate performance test scenario
    pub fn generate_performance_test_scenario(
        &mut self,
        entity_count: usize,
        relationship_count: usize,
    ) -> PerformanceTestScenario {
        let spec = GraphSpec {
            entity_count: entity_count as u64,
            relationship_count: relationship_count as u64,
            topology: TopologyType::Random,
            clustering_coefficient: 0.0,
        };
        
        let graph_data = self.generate_graph(&spec);
        
        PerformanceTestScenario {
            entities: graph_data.entities.into_values().collect(),
            relationships: graph_data.relationships,
            relationship_count: relationship_count as u64,
        }
    }

    /// Generate federation scenario
    pub fn generate_federation_scenario(
        &mut self,
        shard_count: usize,
        entities_per_shard: usize,
    ) -> FederationScenario {
        let mut shards = Vec::new();
        let mut cross_shard_relationships = Vec::new();
        
        // Create shards
        for shard_id in 0..shard_count {
            let mut shard_entities = HashSet::new();
            let mut shard_relationships = Vec::new();
            
            // Create entities for this shard
            for i in 0..entities_per_shard {
                let key = EntityKey::from_hash(format!("shard_{}_entity_{}", shard_id, i));
                shard_entities.insert(key);
            }
            
            // Create intra-shard relationships
            let entity_vec: Vec<_> = shard_entities.iter().cloned().collect();
            let rel_count = entities_per_shard * 2; // Average degree of 2
            
            for i in 0..rel_count {
                let source_idx = self.rng.gen_range(0..entity_vec.len());
                let target_idx = self.rng.gen_range(0..entity_vec.len());
                
                if source_idx != target_idx {
                    let relationship = Relationship::new(
                        format!("shard_{}_rel_{}", shard_id, i),
                        RelationshipType::Directed,
                        1.0,
                    );
                    
                    shard_relationships.push(ShardRelationship {
                        source: entity_vec[source_idx],
                        target: entity_vec[target_idx],
                        relationship,
                    });
                }
            }
            
            shards.push(DatabaseShard {
                entities: shard_entities,
                relationships: shard_relationships,
            });
        }
        
        // Create cross-shard relationships
        let cross_shard_count = shard_count * 10; // Some cross-shard connections
        
        for i in 0..cross_shard_count {
            let source_shard = self.rng.gen_range(0..shard_count);
            let mut target_shard = self.rng.gen_range(0..shard_count);
            
            while target_shard == source_shard {
                target_shard = self.rng.gen_range(0..shard_count);
            }
            
            let source_entities: Vec<_> = shards[source_shard].entities.iter().cloned().collect();
            let target_entities: Vec<_> = shards[target_shard].entities.iter().cloned().collect();
            
            if !source_entities.is_empty() && !target_entities.is_empty() {
                let source_entity = source_entities[self.rng.gen_range(0..source_entities.len())];
                let target_entity = target_entities[self.rng.gen_range(0..target_entities.len())];
                
                let relationship = Relationship::new(
                    format!("cross_shard_rel_{}", i),
                    RelationshipType::Directed,
                    0.5,
                );
                
                cross_shard_relationships.push(CrossShardRelationship {
                    source_shard,
                    target_shard,
                    source_entity,
                    target_entity,
                    relationship,
                });
            }
        }
        
        FederationScenario {
            shards,
            cross_shard_relationships,
        }
    }

    /// Generate memory test scenario
    pub fn generate_memory_test_scenario(&mut self, size: u64) -> MemoryTestScenario {
        let spec = GraphSpec {
            entity_count: size,
            relationship_count: size * 2,
            topology: TopologyType::Random,
            clustering_coefficient: 0.0,
        };
        
        let graph_data = self.generate_graph(&spec);
        let mut embeddings = HashMap::new();
        
        // Generate embeddings
        for &key in graph_data.entities.keys() {
            let embedding = self.generate_embedding(128);
            embeddings.insert(key, embedding);
        }
        
        MemoryTestScenario {
            entities: graph_data.entities.into_values().collect(),
            relationships: graph_data.relationships,
            embeddings,
            relationship_count: size * 2,
        }
    }

    /// Generate concurrent test scenario
    pub fn generate_concurrent_test_scenario(
        &mut self,
        entity_count: usize,
        relationship_count: usize,
    ) -> ConcurrentTestScenario {
        let spec = GraphSpec {
            entity_count: entity_count as u64,
            relationship_count: relationship_count as u64,
            topology: TopologyType::Random,
            clustering_coefficient: 0.0,
        };
        
        let graph_data = self.generate_graph(&spec);
        let mut embeddings = HashMap::new();
        
        // Generate embeddings
        for &key in graph_data.entities.keys() {
            let embedding = self.generate_embedding(128);
            embeddings.insert(key, embedding);
        }
        
        // Select test entities
        let test_entities: Vec<_> = graph_data.entities.keys()
            .take(100)
            .cloned()
            .collect();
        
        ConcurrentTestScenario {
            entities: graph_data.entities.into_values().collect(),
            relationships: graph_data.relationships,
            embeddings,
            test_entities,
        }
    }

    /// Generate random embeddings
    pub fn generate_random_embeddings(
        &mut self,
        count: usize,
        dim: usize,
    ) -> HashMap<EntityKey, Vec<f32>> {
        let mut embeddings = HashMap::new();
        
        for i in 0..count {
            let key = EntityKey::from_hash(format!("random_embed_{}", i));
            let embedding = self.generate_embedding(dim);
            embeddings.insert(key, embedding);
        }
        
        embeddings
    }

    /// Generate performance graph
    pub fn generate_performance_graph(
        &mut self,
        entity_count: u64,
        relationship_count: u64,
        embedding_dim: usize,
    ) -> PerformanceGraphData {
        let spec = GraphSpec {
            entity_count,
            relationship_count,
            topology: TopologyType::ScaleFree { exponent: 2.5 },
            clustering_coefficient: 0.2,
        };
        
        let graph_data = self.generate_graph(&spec);
        let mut embeddings = HashMap::new();
        
        // Generate embeddings
        for &key in graph_data.entities.keys() {
            let embedding = self.generate_embedding(embedding_dim);
            embeddings.insert(key, embedding);
        }
        
        PerformanceGraphData {
            entities: graph_data.entities.into_values().collect(),
            relationships: graph_data.relationships,
            embeddings,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_generation() {
        let mut generator = TestDataGenerator::new();
        
        let spec = GraphSpec {
            entity_count: 100,
            relationship_count: 200,
            topology: TopologyType::Random,
            clustering_coefficient: 0.0,
        };
        
        let graph_data = generator.generate_graph(&spec);
        
        assert_eq!(graph_data.entities.len(), 100);
        assert_eq!(graph_data.relationships.len(), 200);
    }

    #[test]
    fn test_academic_scenario() {
        let mut generator = TestDataGenerator::new();
        
        let scenario = generator.generate_academic_scenario(50, 20, 5, 64);
        
        assert_eq!(scenario.entities.len(), 75); // 50 + 20 + 5
        assert!(!scenario.relationships.is_empty());
        assert_eq!(scenario.embeddings.len(), 75);
        assert!(!scenario.central_entities.is_empty());
    }

    #[test]
    fn test_embedding_generation() {
        let mut generator = TestDataGenerator::new();
        
        let embedding = generator.generate_embedding(128);
        
        assert_eq!(embedding.len(), 128);
        
        // Check normalization
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }
}