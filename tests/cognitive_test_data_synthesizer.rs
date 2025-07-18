use std::collections::HashMap;
use std::sync::Arc;
use llmkg::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use llmkg::core::brain_types::{BrainInspiredEntity, EntityDirection, BrainInspiredRelationship, RelationType};
use llmkg::core::types::EntityKey;
use llmkg::error::Result;

/// Comprehensive test data synthesizer for cognitive pattern testing
pub struct CognitiveTestDataSynthesizer {
    pub graph: Arc<BrainEnhancedKnowledgeGraph>,
    pub entity_registry: HashMap<String, EntityKey>,
    pub relationship_count: usize,
}

impl CognitiveTestDataSynthesizer {
    pub async fn new() -> Result<Self> {
        let graph = Arc::new(BrainEnhancedKnowledgeGraph::new_for_test());
        
        Ok(Self {
            graph,
            entity_registry: HashMap::new(),
            relationship_count: 0,
        })
    }
    
    /// Create a comprehensive knowledge hierarchy for testing all cognitive patterns
    pub async fn synthesize_comprehensive_knowledge_base(&mut self) -> Result<()> {
        // 1. Create biological taxonomy for systems thinking
        self.create_biological_taxonomy().await?;
        
        // 2. Create technology domain for convergent/divergent thinking
        self.create_technology_domain().await?;
        
        // 3. Create art and culture domain for lateral thinking
        self.create_art_culture_domain().await?;
        
        // 4. Create contradictory facts for critical thinking
        self.create_contradictory_scenarios().await?;
        
        // 5. Create pattern-rich structures for abstract thinking
        self.create_pattern_structures().await?;
        
        // 6. Create complex scenarios for adaptive thinking
        self.create_adaptive_scenarios().await?;
        
        println!("Synthesized comprehensive knowledge base:");
        println!("- Total entities: {}", self.entity_registry.len());
        println!("- Total relationships: {}", self.relationship_count);
        
        Ok(())
    }
    
    /// Create biological taxonomy with proper inheritance
    async fn create_biological_taxonomy(&mut self) -> Result<()> {
        // Root taxonomy
        let living_things = self.create_entity("living_things", EntityDirection::Input).await?;
        let animals = self.create_entity("animals", EntityDirection::Gate).await?;
        let plants = self.create_entity("plants", EntityDirection::Gate).await?;
        
        // Animal taxonomy
        let vertebrates = self.create_entity("vertebrates", EntityDirection::Gate).await?;
        let invertebrates = self.create_entity("invertebrates", EntityDirection::Gate).await?;
        let mammals = self.create_entity("mammals", EntityDirection::Gate).await?;
        let birds = self.create_entity("birds", EntityDirection::Gate).await?;
        let reptiles = self.create_entity("reptiles", EntityDirection::Gate).await?;
        let fish = self.create_entity("fish", EntityDirection::Gate).await?;
        
        // Mammal subcategories
        let carnivores = self.create_entity("carnivores", EntityDirection::Gate).await?;
        let herbivores = self.create_entity("herbivores", EntityDirection::Gate).await?;
        let omnivores = self.create_entity("omnivores", EntityDirection::Gate).await?;
        let primates = self.create_entity("primates", EntityDirection::Gate).await?;
        
        // Specific animals
        let dogs = self.create_entity("dogs", EntityDirection::Gate).await?;
        let cats = self.create_entity("cats", EntityDirection::Gate).await?;
        let elephants = self.create_entity("elephants", EntityDirection::Gate).await?;
        let humans = self.create_entity("humans", EntityDirection::Output).await?;
        let golden_retriever = self.create_entity("golden_retriever", EntityDirection::Output).await?;
        let persian_cat = self.create_entity("persian_cat", EntityDirection::Output).await?;
        let african_elephant = self.create_entity("african_elephant", EntityDirection::Output).await?;
        
        // Special case entities for testing
        let tripper = self.create_entity("tripper_three_legged_dog", EntityDirection::Output).await?;
        let manx_cat = self.create_entity("manx_tailless_cat", EntityDirection::Output).await?;
        
        // Create hierarchical relationships (IsA)
        self.create_relationship(animals, living_things, RelationType::IsA, 0.95).await?;
        self.create_relationship(plants, living_things, RelationType::IsA, 0.95).await?;
        self.create_relationship(vertebrates, animals, RelationType::IsA, 0.9).await?;
        self.create_relationship(invertebrates, animals, RelationType::IsA, 0.9).await?;
        self.create_relationship(mammals, vertebrates, RelationType::IsA, 0.95).await?;
        self.create_relationship(birds, vertebrates, RelationType::IsA, 0.95).await?;
        self.create_relationship(reptiles, vertebrates, RelationType::IsA, 0.95).await?;
        self.create_relationship(fish, vertebrates, RelationType::IsA, 0.95).await?;
        
        self.create_relationship(carnivores, mammals, RelationType::IsA, 0.9).await?;
        self.create_relationship(herbivores, mammals, RelationType::IsA, 0.9).await?;
        self.create_relationship(omnivores, mammals, RelationType::IsA, 0.9).await?;
        self.create_relationship(primates, mammals, RelationType::IsA, 0.9).await?;
        
        self.create_relationship(dogs, carnivores, RelationType::IsA, 0.95).await?;
        self.create_relationship(cats, carnivores, RelationType::IsA, 0.95).await?;
        self.create_relationship(elephants, herbivores, RelationType::IsA, 0.95).await?;
        self.create_relationship(humans, primates, RelationType::IsA, 0.95).await?;
        
        self.create_relationship(golden_retriever, dogs, RelationType::IsA, 0.98).await?;
        self.create_relationship(persian_cat, cats, RelationType::IsA, 0.98).await?;
        self.create_relationship(african_elephant, elephants, RelationType::IsA, 0.98).await?;
        self.create_relationship(tripper, dogs, RelationType::IsA, 0.98).await?;
        self.create_relationship(manx_cat, cats, RelationType::IsA, 0.98).await?;
        
        // Create property relationships
        let warm_blooded = self.create_entity("warm_blooded", EntityDirection::Output).await?;
        let has_fur = self.create_entity("has_fur", EntityDirection::Output).await?;
        let has_tail = self.create_entity("has_tail", EntityDirection::Output).await?;
        let four_legs = self.create_entity("four_legs", EntityDirection::Output).await?;
        let domesticated = self.create_entity("domesticated", EntityDirection::Output).await?;
        let large_size = self.create_entity("large_size", EntityDirection::Output).await?;
        let intelligent = self.create_entity("intelligent", EntityDirection::Output).await?;
        let social = self.create_entity("social", EntityDirection::Output).await?;
        
        // Property inheritance
        self.create_relationship(mammals, warm_blooded, RelationType::HasProperty, 0.98).await?;
        self.create_relationship(mammals, has_fur, RelationType::HasProperty, 0.8).await?;
        self.create_relationship(mammals, four_legs, RelationType::HasProperty, 0.85).await?;
        self.create_relationship(dogs, domesticated, RelationType::HasProperty, 0.95).await?;
        self.create_relationship(cats, domesticated, RelationType::HasProperty, 0.8).await?;
        self.create_relationship(elephants, large_size, RelationType::HasProperty, 0.98).await?;
        self.create_relationship(elephants, intelligent, RelationType::HasProperty, 0.9).await?;
        self.create_relationship(dogs, social, RelationType::HasProperty, 0.9).await?;
        self.create_relationship(humans, intelligent, RelationType::HasProperty, 0.98).await?;
        self.create_relationship(humans, social, RelationType::HasProperty, 0.95).await?;
        
        // Exception cases for critical thinking
        let three_legs = self.create_entity("three_legs", EntityDirection::Output).await?;
        let no_tail = self.create_entity("no_tail", EntityDirection::Output).await?;
        self.create_relationship(tripper, three_legs, RelationType::HasProperty, 0.98).await?;
        self.create_relationship(manx_cat, no_tail, RelationType::HasProperty, 0.98).await?;
        
        Ok(())
    }
    
    /// Create technology domain for convergent/divergent testing
    async fn create_technology_domain(&mut self) -> Result<()> {
        // Technology taxonomy
        let technology = self.create_entity("technology", EntityDirection::Input).await?;
        let computers = self.create_entity("computers", EntityDirection::Gate).await?;
        let ai = self.create_entity("artificial_intelligence", EntityDirection::Gate).await?;
        let hardware = self.create_entity("hardware", EntityDirection::Gate).await?;
        let software = self.create_entity("software", EntityDirection::Gate).await?;
        
        // AI subcategories
        let machine_learning = self.create_entity("machine_learning", EntityDirection::Gate).await?;
        let deep_learning = self.create_entity("deep_learning", EntityDirection::Gate).await?;
        let neural_networks = self.create_entity("neural_networks", EntityDirection::Gate).await?;
        let natural_language_processing = self.create_entity("natural_language_processing", EntityDirection::Gate).await?;
        let computer_vision = self.create_entity("computer_vision", EntityDirection::Gate).await?;
        
        // Specific technologies
        let transformers = self.create_entity("transformer_models", EntityDirection::Output).await?;
        let gpt = self.create_entity("gpt_models", EntityDirection::Output).await?;
        let cnns = self.create_entity("convolutional_neural_networks", EntityDirection::Output).await?;
        let rnns = self.create_entity("recurrent_neural_networks", EntityDirection::Output).await?;
        
        // Create relationships
        self.create_relationship(computers, technology, RelationType::IsA, 0.95).await?;
        self.create_relationship(ai, computers, RelationType::IsA, 0.9).await?;
        self.create_relationship(hardware, computers, RelationType::IsA, 0.9).await?;
        self.create_relationship(software, computers, RelationType::IsA, 0.9).await?;
        
        self.create_relationship(machine_learning, ai, RelationType::IsA, 0.95).await?;
        self.create_relationship(deep_learning, machine_learning, RelationType::IsA, 0.95).await?;
        self.create_relationship(neural_networks, deep_learning, RelationType::IsA, 0.95).await?;
        self.create_relationship(natural_language_processing, ai, RelationType::IsA, 0.9).await?;
        self.create_relationship(computer_vision, ai, RelationType::IsA, 0.9).await?;
        
        self.create_relationship(transformers, neural_networks, RelationType::IsA, 0.95).await?;
        self.create_relationship(gpt, transformers, RelationType::IsA, 0.95).await?;
        self.create_relationship(cnns, neural_networks, RelationType::IsA, 0.95).await?;
        self.create_relationship(rnns, neural_networks, RelationType::IsA, 0.95).await?;
        
        // Properties for technology domain
        let computational = self.create_entity("computational", EntityDirection::Output).await?;
        let digital = self.create_entity("digital", EntityDirection::Output).await?;
        let automated = self.create_entity("automated", EntityDirection::Output).await?;
        let learning_capable = self.create_entity("learning_capable", EntityDirection::Output).await?;
        let data_driven = self.create_entity("data_driven", EntityDirection::Output).await?;
        
        self.create_relationship(computers, computational, RelationType::HasProperty, 0.98).await?;
        self.create_relationship(computers, digital, RelationType::HasProperty, 0.98).await?;
        self.create_relationship(ai, automated, RelationType::HasProperty, 0.9).await?;
        self.create_relationship(machine_learning, learning_capable, RelationType::HasProperty, 0.95).await?;
        self.create_relationship(machine_learning, data_driven, RelationType::HasProperty, 0.95).await?;
        
        Ok(())
    }
    
    /// Create art and culture domain for lateral thinking
    async fn create_art_culture_domain(&mut self) -> Result<()> {
        // Art taxonomy
        let art = self.create_entity("art", EntityDirection::Input).await?;
        let visual_art = self.create_entity("visual_art", EntityDirection::Gate).await?;
        let performing_art = self.create_entity("performing_art", EntityDirection::Gate).await?;
        let literature = self.create_entity("literature", EntityDirection::Gate).await?;
        let music = self.create_entity("music", EntityDirection::Gate).await?;
        
        // Visual art types
        let painting = self.create_entity("painting", EntityDirection::Gate).await?;
        let sculpture = self.create_entity("sculpture", EntityDirection::Gate).await?;
        let photography = self.create_entity("photography", EntityDirection::Gate).await?;
        let digital_art = self.create_entity("digital_art", EntityDirection::Gate).await?;
        
        // Specific works/styles
        let impressionism = self.create_entity("impressionism", EntityDirection::Output).await?;
        let cubism = self.create_entity("cubism", EntityDirection::Output).await?;
        let abstract_expressionism = self.create_entity("abstract_expressionism", EntityDirection::Output).await?;
        let photorealism = self.create_entity("photorealism", EntityDirection::Output).await?;
        
        // Create relationships
        self.create_relationship(visual_art, art, RelationType::IsA, 0.95).await?;
        self.create_relationship(performing_art, art, RelationType::IsA, 0.95).await?;
        self.create_relationship(literature, art, RelationType::IsA, 0.9).await?;
        self.create_relationship(music, art, RelationType::IsA, 0.95).await?;
        
        self.create_relationship(painting, visual_art, RelationType::IsA, 0.95).await?;
        self.create_relationship(sculpture, visual_art, RelationType::IsA, 0.95).await?;
        self.create_relationship(photography, visual_art, RelationType::IsA, 0.9).await?;
        self.create_relationship(digital_art, visual_art, RelationType::IsA, 0.85).await?;
        
        self.create_relationship(impressionism, painting, RelationType::IsA, 0.95).await?;
        self.create_relationship(cubism, painting, RelationType::IsA, 0.95).await?;
        self.create_relationship(abstract_expressionism, painting, RelationType::IsA, 0.95).await?;
        self.create_relationship(photorealism, painting, RelationType::IsA, 0.95).await?;
        
        // Creative cross-domain connections for lateral thinking
        let creativity = self.create_entity("creativity", EntityDirection::Output).await?;
        let expression = self.create_entity("expression", EntityDirection::Output).await?;
        let innovation = self.create_entity("innovation", EntityDirection::Output).await?;
        let beauty = self.create_entity("beauty", EntityDirection::Output).await?;
        let emotion = self.create_entity("emotion", EntityDirection::Output).await?;
        
        self.create_relationship(art, creativity, RelationType::HasProperty, 0.95).await?;
        self.create_relationship(art, expression, RelationType::HasProperty, 0.95).await?;
        self.create_relationship(art, beauty, RelationType::HasProperty, 0.8).await?;
        self.create_relationship(art, emotion, RelationType::HasProperty, 0.9).await?;
        
        // Cross-domain creative connections (AI + Art)
        let ai = *self.entity_registry.get("artificial_intelligence").unwrap();
        self.create_relationship(digital_art, ai, RelationType::RelatedTo, 0.7).await?;
        self.create_relationship(ai, creativity, RelationType::RelatedTo, 0.6).await?;
        self.create_relationship(ai, innovation, RelationType::HasProperty, 0.8).await?;
        
        Ok(())
    }
    
    /// Create contradictory scenarios for critical thinking
    async fn create_contradictory_scenarios(&mut self) -> Result<()> {
        // Get existing entities
        let tripper = *self.entity_registry.get("tripper_three_legged_dog").unwrap();
        let dogs = *self.entity_registry.get("dogs").unwrap();
        let four_legs = *self.entity_registry.get("four_legs").unwrap();
        let three_legs = *self.entity_registry.get("three_legs").unwrap();
        
        // Create explicit contradiction
        self.create_relationship(dogs, four_legs, RelationType::HasProperty, 0.9).await?;
        self.create_relationship(tripper, three_legs, RelationType::HasProperty, 0.95).await?;
        
        // Additional contradictory scenarios
        let flying_mammals = self.create_entity("flying_mammals", EntityDirection::Output).await?;
        let bats = self.create_entity("bats", EntityDirection::Output).await?;
        let mammals = *self.entity_registry.get("mammals").unwrap();
        let flight_capable = self.create_entity("flight_capable", EntityDirection::Output).await?;
        let terrestrial = self.create_entity("terrestrial", EntityDirection::Output).await?;
        
        self.create_relationship(bats, flying_mammals, RelationType::IsA, 0.95).await?;
        self.create_relationship(flying_mammals, mammals, RelationType::IsA, 0.95).await?;
        self.create_relationship(mammals, terrestrial, RelationType::HasProperty, 0.8).await?;
        self.create_relationship(bats, flight_capable, RelationType::HasProperty, 0.95).await?;
        
        // Temperature contradiction
        let penguins = self.create_entity("penguins", EntityDirection::Output).await?;
        let birds = *self.entity_registry.get("birds").unwrap();
        let cold_adapted = self.create_entity("cold_adapted", EntityDirection::Output).await?;
        let heat_adapted = self.create_entity("heat_adapted", EntityDirection::Output).await?;
        
        self.create_relationship(penguins, birds, RelationType::IsA, 0.95).await?;
        self.create_relationship(birds, heat_adapted, RelationType::HasProperty, 0.7).await?;
        self.create_relationship(penguins, cold_adapted, RelationType::HasProperty, 0.95).await?;
        
        Ok(())
    }
    
    /// Create pattern-rich structures for abstract thinking
    async fn create_pattern_structures(&mut self) -> Result<()> {
        // Create repeating hierarchical patterns
        for domain in ["science", "business", "sports", "education"] {
            let domain_entity = self.create_entity(domain, EntityDirection::Input).await?;
            
            // Create 3-level hierarchy pattern
            for category in ["theory", "practice", "innovation"] {
                let category_name = format!("{}_{}", domain, category);
                let category_entity = self.create_entity(&category_name, EntityDirection::Gate).await?;
                self.create_relationship(category_entity, domain_entity, RelationType::IsA, 0.9).await?;
                
                // Create subcategories with consistent naming pattern
                for level in ["basic", "intermediate", "advanced"] {
                    let sublevel_name = format!("{}_{}_{}", domain, category, level);
                    let sublevel_entity = self.create_entity(&sublevel_name, EntityDirection::Output).await?;
                    self.create_relationship(sublevel_entity, category_entity, RelationType::IsA, 0.85).await?;
                }
            }
        }
        
        // Create connection patterns
        let collaboration = self.create_entity("collaboration", EntityDirection::Output).await?;
        let research = self.create_entity("research", EntityDirection::Output).await?;
        let development = self.create_entity("development", EntityDirection::Output).await?;
        
        // Connect across domains with similar patterns
        for domain in ["science", "business", "education"] {
            let theory_name = format!("{}_theory", domain);
            if let Some(&theory_entity) = self.entity_registry.get(&theory_name) {
                self.create_relationship(theory_entity, research, RelationType::RelatedTo, 0.8).await?;
            }
            
            let practice_name = format!("{}_practice", domain);
            if let Some(&practice_entity) = self.entity_registry.get(&practice_name) {
                self.create_relationship(practice_entity, collaboration, RelationType::RelatedTo, 0.7).await?;
            }
            
            let innovation_name = format!("{}_innovation", domain);
            if let Some(&innovation_entity) = self.entity_registry.get(&innovation_name) {
                self.create_relationship(innovation_entity, development, RelationType::RelatedTo, 0.9).await?;
            }
        }
        
        Ok(())
    }
    
    /// Create complex scenarios for adaptive thinking
    async fn create_adaptive_scenarios(&mut self) -> Result<()> {
        // Multi-domain query scenarios that require different patterns
        
        // Scenario 1: Convergent + Systems (factual inheritance query)
        let domesticated = *self.entity_registry.get("domesticated").unwrap();
        let warm_blooded = *self.entity_registry.get("warm_blooded").unwrap();
        let golden_retriever = *self.entity_registry.get("golden_retriever").unwrap();
        self.create_relationship(golden_retriever, domesticated, RelationType::HasProperty, 0.95).await?;
        self.create_relationship(golden_retriever, warm_blooded, RelationType::HasProperty, 0.95).await?;
        
        // Scenario 2: Divergent + Lateral (creative exploration)
        let art = *self.entity_registry.get("art").unwrap();
        let ai = *self.entity_registry.get("artificial_intelligence").unwrap();
        let creativity = *self.entity_registry.get("creativity").unwrap();
        let innovation = *self.entity_registry.get("innovation").unwrap();
        
        // Bridge connections for lateral thinking
        self.create_relationship(ai, innovation, RelationType::HasProperty, 0.8).await?;
        self.create_relationship(art, creativity, RelationType::HasProperty, 0.95).await?;
        self.create_relationship(creativity, innovation, RelationType::RelatedTo, 0.7).await?;
        
        // Scenario 3: Critical + Systems (contradiction in hierarchy)
        let tripper = *self.entity_registry.get("tripper_three_legged_dog").unwrap();
        let dogs = *self.entity_registry.get("dogs").unwrap();
        let four_legs = *self.entity_registry.get("four_legs").unwrap();
        let three_legs = *self.entity_registry.get("three_legs").unwrap();
        
        // Ensure both properties are linked for contradiction testing
        self.create_relationship(dogs, four_legs, RelationType::HasProperty, 0.9).await?;
        self.create_relationship(tripper, three_legs, RelationType::HasProperty, 0.95).await?;
        
        // Scenario 4: Abstract pattern detection across domains
        let pattern_entity = self.create_entity("learning_pattern", EntityDirection::Output).await?;
        let machine_learning = *self.entity_registry.get("machine_learning").unwrap();
        let education = *self.entity_registry.get("education").unwrap();
        
        self.create_relationship(machine_learning, pattern_entity, RelationType::RelatedTo, 0.8).await?;
        self.create_relationship(education, pattern_entity, RelationType::RelatedTo, 0.7).await?;
        
        Ok(())
    }
    
    /// Helper method to create entity
    async fn create_entity(&mut self, concept: &str, direction: EntityDirection) -> Result<EntityKey> {
        let entity = BrainInspiredEntity::new(concept.to_string(), direction);
        let key = self.graph.insert_brain_entity(entity).await?;
        self.entity_registry.insert(concept.to_string(), key);
        Ok(key)
    }
    
    /// Helper method to create relationship
    async fn create_relationship(
        &mut self, 
        source: EntityKey, 
        target: EntityKey, 
        relation_type: RelationType, 
        weight: f32
    ) -> Result<()> {
        let relationship = BrainInspiredRelationship {
            source,
            target,
            relation_type,
            weight,
            is_inhibitory: false,
            temporal_decay: 0.95,
            last_strengthened: std::time::SystemTime::now(),
            activation_count: 0,
            creation_time: std::time::SystemTime::now(),
            ingestion_time: std::time::SystemTime::now(),
        };
        self.graph.insert_brain_relationship(relationship).await?;
        self.relationship_count += 1;
        Ok(())
    }
    
    /// Get entity key by concept name
    pub fn get_entity(&self, concept: &str) -> Option<EntityKey> {
        self.entity_registry.get(concept).copied()
    }
    
    /// Get statistics about the synthesized data
    pub fn get_statistics(&self) -> TestDataStatistics {
        TestDataStatistics {
            total_entities: self.entity_registry.len(),
            total_relationships: self.relationship_count,
            entity_breakdown: self.count_entities_by_direction(),
            relationship_breakdown: HashMap::new(), // Would need to track this during creation
        }
    }
    
    fn count_entities_by_direction(&self) -> HashMap<String, usize> {
        // This would require iterating through actual entities
        // For now, return estimated counts
        let mut breakdown = HashMap::new();
        breakdown.insert("Input".to_string(), self.entity_registry.len() / 4);
        breakdown.insert("Gate".to_string(), self.entity_registry.len() / 2);
        breakdown.insert("Output".to_string(), self.entity_registry.len() / 4);
        breakdown
    }
    
    /// Create specific test scenarios for each cognitive pattern
    pub async fn create_pattern_specific_scenarios(&mut self) -> Result<PatternTestScenarios> {
        let mut scenarios = PatternTestScenarios::new();
        
        // Convergent thinking scenarios
        scenarios.convergent_scenarios.push(ConvergentTestCase {
            query: "What properties do golden retrievers have?".to_string(),
            expected_properties: vec!["warm_blooded".to_string(), "domesticated".to_string(), "has_fur".to_string()],
            confidence_threshold: 0.8,
        });
        
        scenarios.convergent_scenarios.push(ConvergentTestCase {
            query: "How many legs do dogs have?".to_string(),
            expected_properties: vec!["four_legs".to_string()],
            confidence_threshold: 0.8,
        });
        
        // Divergent thinking scenarios
        scenarios.divergent_scenarios.push(DivergentTestCase {
            query: "What are types of animals?".to_string(),
            expected_explorations: vec!["mammals".to_string(), "birds".to_string(), "reptiles".to_string(), "fish".to_string()],
            min_exploration_count: 3,
        });
        
        scenarios.divergent_scenarios.push(DivergentTestCase {
            query: "What are examples of AI technologies?".to_string(),
            expected_explorations: vec!["machine_learning".to_string(), "deep_learning".to_string(), "neural_networks".to_string()],
            min_exploration_count: 2,
        });
        
        // Systems thinking scenarios
        scenarios.systems_scenarios.push(SystemsTestCase {
            query: "What properties do mammals inherit?".to_string(),
            expected_attributes: vec!["warm_blooded".to_string(), "has_fur".to_string()],
            hierarchy_depth: 2,
        });
        
        // Critical thinking scenarios
        scenarios.critical_scenarios.push(CriticalTestCase {
            query: "How many legs does Tripper have?".to_string(),
            expected_contradictions: vec![
                ("dogs have four legs".to_string(), "tripper has three legs".to_string())
            ],
            resolution_strategy: "prefer_specific".to_string(),
        });
        
        // Lateral thinking scenarios
        scenarios.lateral_scenarios.push(LateralTestCase {
            query_a: "artificial intelligence".to_string(),
            query_b: "art".to_string(),
            expected_bridges: vec!["creativity".to_string(), "innovation".to_string()],
            max_bridge_length: 3,
        });
        
        // Abstract thinking scenarios
        scenarios.abstract_scenarios.push(AbstractTestCase {
            scope: "global".to_string(),
            expected_patterns: vec!["hierarchical_structure".to_string(), "domain_specialization".to_string()],
            pattern_frequency_threshold: 0.5,
        });
        
        // Adaptive thinking scenarios
        scenarios.adaptive_scenarios.push(AdaptiveTestCase {
            query: "What creative connections exist between AI and art?".to_string(),
            expected_patterns: vec!["Lateral".to_string(), "Divergent".to_string()],
            confidence_threshold: 0.7,
        });
        
        Ok(scenarios)
    }
}

#[derive(Debug, Clone)]
pub struct TestDataStatistics {
    pub total_entities: usize,
    pub total_relationships: usize,
    pub entity_breakdown: HashMap<String, usize>,
    pub relationship_breakdown: HashMap<String, usize>,
}

#[derive(Debug, Clone)]
pub struct PatternTestScenarios {
    pub convergent_scenarios: Vec<ConvergentTestCase>,
    pub divergent_scenarios: Vec<DivergentTestCase>,
    pub systems_scenarios: Vec<SystemsTestCase>,
    pub critical_scenarios: Vec<CriticalTestCase>,
    pub lateral_scenarios: Vec<LateralTestCase>,
    pub abstract_scenarios: Vec<AbstractTestCase>,
    pub adaptive_scenarios: Vec<AdaptiveTestCase>,
}

impl PatternTestScenarios {
    pub fn new() -> Self {
        Self {
            convergent_scenarios: Vec::new(),
            divergent_scenarios: Vec::new(),
            systems_scenarios: Vec::new(),
            critical_scenarios: Vec::new(),
            lateral_scenarios: Vec::new(),
            abstract_scenarios: Vec::new(),
            adaptive_scenarios: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConvergentTestCase {
    pub query: String,
    pub expected_properties: Vec<String>,
    pub confidence_threshold: f32,
}

#[derive(Debug, Clone)]
pub struct DivergentTestCase {
    pub query: String,
    pub expected_explorations: Vec<String>,
    pub min_exploration_count: usize,
}

#[derive(Debug, Clone)]
pub struct SystemsTestCase {
    pub query: String,
    pub expected_attributes: Vec<String>,
    pub hierarchy_depth: usize,
}

#[derive(Debug, Clone)]
pub struct CriticalTestCase {
    pub query: String,
    pub expected_contradictions: Vec<(String, String)>,
    pub resolution_strategy: String,
}

#[derive(Debug, Clone)]
pub struct LateralTestCase {
    pub query_a: String,
    pub query_b: String,
    pub expected_bridges: Vec<String>,
    pub max_bridge_length: usize,
}

#[derive(Debug, Clone)]
pub struct AbstractTestCase {
    pub scope: String,
    pub expected_patterns: Vec<String>,
    pub pattern_frequency_threshold: f32,
}

#[derive(Debug, Clone)]
pub struct AdaptiveTestCase {
    pub query: String,
    pub expected_patterns: Vec<String>,
    pub confidence_threshold: f32,
}