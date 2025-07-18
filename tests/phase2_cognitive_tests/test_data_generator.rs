use std::sync::Arc;
use llmkg::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use llmkg::core::brain_types::{BrainInspiredEntity, BrainInspiredRelationship, EntityDirection, RelationType};
use llmkg::core::types::{EntityKey, AttributeValue};
use llmkg::neural::neural_server::NeuralProcessingServer;
use llmkg::error::Result;
use std::collections::HashMap;

/// Test data generator for Phase 2 cognitive pattern testing
pub struct TestDataGenerator {
    pub graph: Arc<BrainEnhancedKnowledgeGraph>,
    pub neural_server: Arc<NeuralProcessingServer>,
    pub entity_keys: HashMap<String, EntityKey>,
}

impl TestDataGenerator {
    pub async fn new() -> Result<Self> {
        let base_graph = llmkg::core::graph::KnowledgeGraph::new(384)?;
        let temporal_graph = llmkg::versioning::temporal_graph::TemporalKnowledgeGraph::new(base_graph);
        let graph = Arc::new(BrainEnhancedKnowledgeGraph::new(temporal_graph));
        let neural_server = Arc::new(NeuralProcessingServer::new_test().await?);
        
        Ok(Self {
            graph,
            neural_server,
            entity_keys: HashMap::new(),
        })
    }
    
    /// Generate comprehensive test data for cognitive patterns
    pub async fn generate_comprehensive_data(&mut self) -> Result<()> {
        // Generate animal kingdom data
        self.generate_animal_kingdom().await?;
        
        // Generate technology domain data
        self.generate_technology_domain().await?;
        
        // Generate contradictory data for critical thinking
        self.generate_contradictory_data().await?;
        
        // Generate pattern data for abstract thinking
        self.generate_pattern_data().await?;
        
        // Generate bridge data for lateral thinking
        self.generate_bridge_data().await?;
        
        Ok(())
    }
    
    /// Generate animal kingdom data for hierarchical reasoning
    async fn generate_animal_kingdom(&mut self) -> Result<()> {
        // Top-level categories
        let living_thing = self.create_input_output_pair("living_thing", "A living organism").await?;
        let animal = self.create_input_output_pair("animal", "A living creature that moves and breathes").await?;
        let mammal = self.create_input_output_pair("mammal", "A warm-blooded animal with hair or fur").await?;
        let bird = self.create_input_output_pair("bird", "A warm-blooded animal with feathers and wings").await?;
        let fish = self.create_input_output_pair("fish", "A cold-blooded animal that lives in water").await?;
        
        // Specific animals
        let dog = self.create_input_output_pair("dog", "A domesticated carnivorous mammal").await?;
        let cat = self.create_input_output_pair("cat", "A small carnivorous mammal").await?;
        let elephant = self.create_input_output_pair("elephant", "A large mammal with a trunk").await?;
        let eagle = self.create_input_output_pair("eagle", "A large bird of prey").await?;
        let salmon = self.create_input_output_pair("salmon", "A fish that swims upstream to spawn").await?;
        
        // Special case for critical thinking
        let tripper = self.create_input_output_pair("tripper", "A three-legged dog").await?;
        
        // Properties
        let warm_blooded = self.create_input_output_pair("warm_blooded", "Maintains constant body temperature").await?;
        let cold_blooded = self.create_input_output_pair("cold_blooded", "Body temperature varies with environment").await?;
        let four_legs = self.create_input_output_pair("four_legs", "Has four legs for locomotion").await?;
        let three_legs = self.create_input_output_pair("three_legs", "Has three legs for locomotion").await?;
        let wings = self.create_input_output_pair("wings", "Has wings for flight").await?;
        let fins = self.create_input_output_pair("fins", "Has fins for swimming").await?;
        
        // Create hierarchical relationships
        self.create_relationship(animal.0, living_thing.0, RelationType::IsA).await?;
        self.create_relationship(mammal.0, animal.0, RelationType::IsA).await?;
        self.create_relationship(bird.0, animal.0, RelationType::IsA).await?;
        self.create_relationship(fish.0, animal.0, RelationType::IsA).await?;
        
        // Specific animals to categories
        self.create_relationship(dog.0, mammal.0, RelationType::IsA).await?;
        self.create_relationship(cat.0, mammal.0, RelationType::IsA).await?;
        self.create_relationship(elephant.0, mammal.0, RelationType::IsA).await?;
        self.create_relationship(eagle.0, bird.0, RelationType::IsA).await?;
        self.create_relationship(salmon.0, fish.0, RelationType::IsA).await?;
        self.create_relationship(tripper.0, dog.0, RelationType::IsA).await?;
        
        // Property relationships
        self.create_relationship(mammal.0, warm_blooded.0, RelationType::HasProperty).await?;
        self.create_relationship(bird.0, warm_blooded.0, RelationType::HasProperty).await?;
        self.create_relationship(fish.0, cold_blooded.0, RelationType::HasProperty).await?;
        self.create_relationship(dog.0, four_legs.0, RelationType::HasProperty).await?;
        self.create_relationship(cat.0, four_legs.0, RelationType::HasProperty).await?;
        self.create_relationship(elephant.0, four_legs.0, RelationType::HasProperty).await?;
        self.create_relationship(eagle.0, wings.0, RelationType::HasProperty).await?;
        self.create_relationship(salmon.0, fins.0, RelationType::HasProperty).await?;
        
        // Contradictory relationship for critical thinking
        self.create_relationship(tripper.0, three_legs.0, RelationType::HasProperty).await?;
        
        Ok(())
    }
    
    /// Generate technology domain data
    async fn generate_technology_domain(&mut self) -> Result<()> {
        // Technology categories
        let technology = self.create_input_output_pair("technology", "Applied scientific knowledge").await?;
        let computer = self.create_input_output_pair("computer", "Electronic device for processing data").await?;
        let software = self.create_input_output_pair("software", "Programs and applications").await?;
        let ai = self.create_input_output_pair("ai", "Artificial intelligence systems").await?;
        let machine_learning = self.create_input_output_pair("machine_learning", "AI that learns from data").await?;
        
        // Specific technologies
        let neural_network = self.create_input_output_pair("neural_network", "AI inspired by brain neurons").await?;
        let deep_learning = self.create_input_output_pair("deep_learning", "Neural networks with many layers").await?;
        let llm = self.create_input_output_pair("llm", "Large language model").await?;
        
        // Properties
        let intelligent = self.create_input_output_pair("intelligent", "Capable of reasoning and learning").await?;
        let automated = self.create_input_output_pair("automated", "Operates without human intervention").await?;
        let scalable = self.create_input_output_pair("scalable", "Can handle increasing workload").await?;
        
        // Create hierarchical relationships
        self.create_relationship(computer.0, technology.0, RelationType::IsA).await?;
        self.create_relationship(software.0, technology.0, RelationType::IsA).await?;
        self.create_relationship(ai.0, software.0, RelationType::IsA).await?;
        self.create_relationship(machine_learning.0, ai.0, RelationType::IsA).await?;
        self.create_relationship(neural_network.0, machine_learning.0, RelationType::IsA).await?;
        self.create_relationship(deep_learning.0, neural_network.0, RelationType::IsA).await?;
        self.create_relationship(llm.0, deep_learning.0, RelationType::IsA).await?;
        
        // Property relationships
        self.create_relationship(ai.0, intelligent.0, RelationType::HasProperty).await?;
        self.create_relationship(machine_learning.0, automated.0, RelationType::HasProperty).await?;
        self.create_relationship(neural_network.0, scalable.0, RelationType::HasProperty).await?;
        
        Ok(())
    }
    
    /// Generate contradictory data for critical thinking tests
    async fn generate_contradictory_data(&mut self) -> Result<()> {
        // Create a special entity with contradictory properties
        let special_dog = self.create_input_output_pair("special_dog", "A dog with unusual properties").await?;
        let normal_dog = self.create_input_output_pair("normal_dog", "A typical dog").await?;
        
        // Properties
        let five_legs = self.create_input_output_pair("five_legs", "Has five legs").await?;
        let can_fly = self.create_input_output_pair("can_fly", "Capable of flight").await?;
        let cannot_fly = self.create_input_output_pair("cannot_fly", "Cannot fly").await?;
        
        // Create contradictory relationships
        if let Some(dog_key) = self.entity_keys.get("dog") {
            self.create_relationship(special_dog.0, *dog_key, RelationType::IsA).await?;
            self.create_relationship(normal_dog.0, *dog_key, RelationType::IsA).await?;
        }
        
        self.create_relationship(special_dog.0, five_legs.0, RelationType::HasProperty).await?;
        self.create_relationship(special_dog.0, can_fly.0, RelationType::HasProperty).await?;
        self.create_relationship(normal_dog.0, cannot_fly.0, RelationType::HasProperty).await?;
        
        Ok(())
    }
    
    /// Generate pattern data for abstract thinking
    async fn generate_pattern_data(&mut self) -> Result<()> {
        // Create patterns in naming
        for i in 1..=5 {
            let pattern_entity = self.create_input_output_pair(
                &format!("pattern_{}", i), 
                &format!("Pattern instance number {}", i)
            ).await?;
            
            // Create similar structures to detect patterns
            let structure_entity = self.create_input_output_pair(
                &format!("structure_{}", i), 
                &format!("Structure instance number {}", i)
            ).await?;
            
            self.create_relationship(pattern_entity.0, structure_entity.0, RelationType::Similar).await?;
        }
        
        // Create temporal patterns
        for i in 1..=3 {
            let event_entity = self.create_input_output_pair(
                &format!("event_{}", i), 
                &format!("Event that occurred at time {}", i)
            ).await?;
            
            if i > 1 {
                if let Some(prev_event) = self.entity_keys.get(&format!("event_{}", i - 1)) {
                    self.create_relationship(event_entity.0, *prev_event, RelationType::Temporal).await?;
                }
            }
        }
        
        Ok(())
    }
    
    /// Generate bridge data for lateral thinking
    async fn generate_bridge_data(&mut self) -> Result<()> {
        // Create concepts that can be bridged
        let art = self.create_input_output_pair("art", "Creative expression").await?;
        let creativity = self.create_input_output_pair("creativity", "Ability to create new ideas").await?;
        let innovation = self.create_input_output_pair("innovation", "New methods or ideas").await?;
        let problem_solving = self.create_input_output_pair("problem_solving", "Finding solutions").await?;
        
        // Create bridge relationships
        self.create_relationship(art.0, creativity.0, RelationType::RelatedTo).await?;
        self.create_relationship(creativity.0, innovation.0, RelationType::RelatedTo).await?;
        self.create_relationship(innovation.0, problem_solving.0, RelationType::RelatedTo).await?;
        
        // Connect to AI (creating a bridge path: art -> creativity -> innovation -> problem_solving -> ai)
        if let Some(ai_key) = self.entity_keys.get("ai") {
            self.create_relationship(problem_solving.0, *ai_key, RelationType::RelatedTo).await?;
        }
        
        Ok(())
    }
    
    /// Create an input-output entity pair
    async fn create_input_output_pair(&mut self, concept: &str, description: &str) -> Result<(EntityKey, EntityKey)> {
        // Create input entity
        let input_entity = BrainInspiredEntity::new(concept.to_string(), EntityDirection::Input);
        let input_key = self.graph.insert_brain_entity(input_entity).await?;
        
        // Create output entity
        let output_entity = BrainInspiredEntity::new(description.to_string(), EntityDirection::Output);
        let output_key = self.graph.insert_brain_entity(output_entity).await?;
        
        // Create relationship between input and output
        self.create_relationship(input_key, output_key, RelationType::RelatedTo).await?;
        
        // Store keys for later reference
        self.entity_keys.insert(concept.to_string(), input_key);
        self.entity_keys.insert(format!("{}_output", concept), output_key);
        
        Ok((input_key, output_key))
    }
    
    /// Create a relationship between entities
    async fn create_relationship(&self, source: EntityKey, target: EntityKey, relation_type: RelationType) -> Result<()> {
        let relationship = BrainInspiredRelationship::new(source, target, relation_type);
        self.graph.insert_brain_relationship(relationship).await?;
        Ok(())
    }
    
    /// Get entity key by concept name
    pub fn get_entity_key(&self, concept: &str) -> Option<EntityKey> {
        self.entity_keys.get(concept).copied()
    }
    
    /// Get all entity keys
    pub fn get_all_entity_keys(&self) -> &HashMap<String, EntityKey> {
        &self.entity_keys
    }
    
    /// Add advanced scientific knowledge for complex reasoning
    pub async fn add_advanced_scientific_knowledge(&mut self) -> Result<()> {
        // Add quantum mechanics concepts
        let quantum_key = self.add_entity("quantum_mechanics", EntityDirection::Input).await?;
        let consciousness_key = self.add_entity("consciousness", EntityDirection::Input).await?;
        let entanglement_key = self.add_entity("quantum_entanglement", EntityDirection::Input).await?;
        
        // Add relationships
        self.add_relationship(quantum_key, entanglement_key, RelationType::HasProperty).await?;
        self.add_relationship(entanglement_key, consciousness_key, RelationType::RelatedTo).await?;
        
        // Add AI concepts
        let agi_key = self.add_entity("artificial_general_intelligence", EntityDirection::Input).await?;
        let computation_key = self.add_entity("computation", EntityDirection::Input).await?;
        let information_key = self.add_entity("information", EntityDirection::Input).await?;
        
        self.add_relationship(agi_key, computation_key, RelationType::RelatedTo).await?;
        self.add_relationship(computation_key, information_key, RelationType::RelatedTo).await?;
        self.add_relationship(information_key, consciousness_key, RelationType::RelatedTo).await?;
        
        Ok(())
    }
    
    /// Add complex temporal relationships
    pub async fn add_complex_temporal_relationships(&mut self) -> Result<()> {
        // Add temporal scale concepts
        let nanosecond_key = self.add_entity("nanosecond_scale", EntityDirection::Input).await?;
        let century_key = self.add_entity("century_scale", EntityDirection::Input).await?;
        let evolution_key = self.add_entity("civilizational_evolution", EntityDirection::Input).await?;
        
        self.add_relationship(nanosecond_key, century_key, RelationType::Temporal).await?;
        self.add_relationship(century_key, evolution_key, RelationType::RelatedTo).await?;
        
        Ok(())
    }
    
    /// Add contradictory edge cases
    pub async fn add_contradictory_edge_cases(&mut self) -> Result<()> {
        // Add philosophical contradictions
        let determinism_key = self.add_entity("determinism", EntityDirection::Input).await?;
        let free_will_key = self.add_entity("free_will", EntityDirection::Input).await?;
        let paradox_key = self.add_entity("philosophical_paradox", EntityDirection::Output).await?;
        
        self.add_relationship(determinism_key, free_will_key, RelationType::RelatedTo).await?;
        self.add_relationship(free_will_key, paradox_key, RelationType::RelatedTo).await?;
        
        Ok(())
    }
    
    /// Add abstract concepts
    pub async fn add_abstract_concepts(&mut self) -> Result<()> {
        // Add meta-cognitive concepts
        let meta_cognition_key = self.add_entity("meta_cognition", EntityDirection::Input).await?;
        let self_reference_key = self.add_entity("self_reference", EntityDirection::Input).await?;
        let infinite_regress_key = self.add_entity("infinite_regress", EntityDirection::Output).await?;
        
        self.add_relationship(meta_cognition_key, self_reference_key, RelationType::RelatedTo).await?;
        self.add_relationship(self_reference_key, infinite_regress_key, RelationType::RelatedTo).await?;
        
        Ok(())
    }
    
    /// Add cultural and social knowledge
    pub async fn add_cultural_and_social_knowledge(&mut self) -> Result<()> {
        // Add social system concepts
        let collective_intelligence_key = self.add_entity("collective_intelligence", EntityDirection::Input).await?;
        let cultural_identity_key = self.add_entity("cultural_identity", EntityDirection::Input).await?;
        let social_protocol_key = self.add_entity("social_protocol", EntityDirection::Output).await?;
        
        self.add_relationship(collective_intelligence_key, cultural_identity_key, RelationType::RelatedTo).await?;
        self.add_relationship(cultural_identity_key, social_protocol_key, RelationType::RelatedTo).await?;
        
        Ok(())
    }
    
    /// Helper method to add entity
    async fn add_entity(&mut self, concept: &str, direction: EntityDirection) -> Result<EntityKey> {
        let entity = BrainInspiredEntity::new(concept.to_string(), direction);
        let key = self.graph.insert_brain_entity(entity).await?;
        self.entity_keys.insert(concept.to_string(), key);
        Ok(key)
    }
    
    /// Helper method to add relationship
    async fn add_relationship(&mut self, source: EntityKey, target: EntityKey, relation_type: RelationType) -> Result<()> {
        let mut relationship = BrainInspiredRelationship::new(source, target, relation_type);
        relationship.weight = 0.8;
        self.graph.insert_brain_relationship(relationship).await?;
        Ok(())
    }
    
    /// Get statistics about the generated data
    pub async fn get_statistics(&self) -> Result<TestDataStatistics> {
        let entities = self.graph.brain_entities.read().await;
        let relationships = self.graph.brain_relationships.read().await;
        
        let mut input_count = 0;
        let mut output_count = 0;
        let mut gate_count = 0;
        
        for entity in entities.values() {
            match entity.direction {
                EntityDirection::Input => input_count += 1,
                EntityDirection::Output => output_count += 1,
                EntityDirection::Gate => gate_count += 1,
            }
        }
        
        let mut relationship_types = HashMap::new();
        for relationship in relationships.values() {
            *relationship_types.entry(relationship.relation_type).or_insert(0) += 1;
        }
        
        Ok(TestDataStatistics {
            total_entities: entities.len(),
            input_entities: input_count,
            output_entities: output_count,
            gate_entities: gate_count,
            total_relationships: relationships.len(),
            relationship_types,
        })
    }
}

/// Statistics about generated test data
#[derive(Debug)]
pub struct TestDataStatistics {
    pub total_entities: usize,
    pub input_entities: usize,
    pub output_entities: usize,
    pub gate_entities: usize,
    pub total_relationships: usize,
    pub relationship_types: HashMap<RelationType, usize>,
}

impl TestDataStatistics {
    pub fn print_summary(&self) {
        println!("=== Test Data Statistics ===");
        println!("Total entities: {}", self.total_entities);
        println!("  Input entities: {}", self.input_entities);
        println!("  Output entities: {}", self.output_entities);
        println!("  Gate entities: {}", self.gate_entities);
        println!("Total relationships: {}", self.total_relationships);
        println!("Relationship types:");
        for (rel_type, count) in &self.relationship_types {
            println!("  {:?}: {}", rel_type, count);
        }
        println!("============================");
    }
}