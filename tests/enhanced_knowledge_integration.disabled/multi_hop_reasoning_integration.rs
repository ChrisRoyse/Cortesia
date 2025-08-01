//! Multi-Hop Reasoning Integration Tests
//! 
//! Tests for complex reasoning that requires multiple steps and connections:
//! - Multi-step logical reasoning chains
//! - Cross-domain knowledge connections
//! - Inference and deduction capabilities
//! - Reasoning validation and explanation

use llmkg::core::graph::KnowledgeGraph;
use llmkg::core::types::{EntityData, Relationship, AttributeValue};
use llmkg::extraction::AdvancedEntityExtractor;
use llmkg::embedding::store::EmbeddingStore;
use llmkg::cognitive::{CognitiveOrchestrator, CognitiveOrchestratorConfig};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{Duration, Instant};

#[cfg(test)]
mod tests {
    use super::*;

    struct MultiHopReasoningSystem {
        graph: Arc<RwLock<KnowledgeGraph>>,
        extractor: AdvancedEntityExtractor,
        embedding_store: EmbeddingStore,
        orchestrator: CognitiveOrchestrator,
    }

    impl MultiHopReasoningSystem {
        async fn new() -> Self {
            let graph = Arc::new(RwLock::new(KnowledgeGraph::new(384).expect("Failed to create graph")));
            let extractor = AdvancedEntityExtractor::new();
            let embedding_store = EmbeddingStore::new(384, 8).expect("Failed to create embedding store");
            let config = CognitiveOrchestratorConfig {
                max_reasoning_depth: 10, // Allow deeper reasoning
                confidence_threshold: 0.3, // Lower threshold for exploration
                enable_creative_reasoning: true,
                timeout_seconds: 60, // Longer timeout for complex reasoning
            };
            let orchestrator = CognitiveOrchestrator::new(config);

            Self {
                graph,
                extractor,
                embedding_store,
                orchestrator,
            }
        }

        async fn build_knowledge_base(&mut self) -> Result<(), String> {
            // Create a rich knowledge base with interconnected facts
            let knowledge_documents = vec![
                // Physics and Einstein
                ("Einstein developed the theory of relativity in 1905 and 1915.", "physics_einstein"),
                ("The theory of relativity explains the relationship between space, time, and gravity.", "physics_concepts"),
                ("Einstein's equation E=mc² shows mass-energy equivalence.", "physics_equations"),
                
                // GPS and Applications
                ("GPS satellites orbit Earth and provide positioning data.", "technology_gps"),
                ("GPS satellites must account for relativistic time dilation to maintain accuracy.", "gps_relativity"),
                ("Without relativistic corrections, GPS would have errors of several kilometers per day.", "gps_precision"),
                
                // Nuclear physics
                ("Nuclear reactions convert mass to energy according to E=mc².", "nuclear_physics"),
                ("Nuclear power plants use controlled nuclear reactions to generate electricity.", "nuclear_power"),
                ("The atomic bomb demonstrated the destructive potential of mass-energy conversion.", "nuclear_weapons"),
                
                // Modern technology
                ("Quantum computers use quantum mechanical principles for computation.", "quantum_computing"),
                ("Quantum entanglement allows for quantum communication and cryptography.", "quantum_communication"),
                ("Lasers use stimulated emission, a concept Einstein helped develop.", "laser_technology"),
                
                // Medicine and radiation
                ("Medical imaging uses radioactive isotopes for diagnosis.", "medical_imaging"),
                ("Radiation therapy uses high-energy radiation to treat cancer.", "radiation_therapy"),
                ("PET scans use positron emission based on antimatter annihilation.", "pet_scanning"),
                
                // Space exploration
                ("Space missions require precise navigation using gravitational calculations.", "space_navigation"),
                ("Time dilation affects astronauts on long space missions.", "space_time_effects"),
                ("Gravitational lensing helps astronomers observe distant galaxies.", "space_observation"),
                
                // Historical connections
                ("Einstein wrote to President Roosevelt about the possibility of atomic weapons.", "einstein_roosevelt"),
                ("The Manhattan Project developed the atomic bomb during World War II.", "manhattan_project"),
                ("Many physicists who worked on the atomic bomb later advocated for nuclear disarmament.", "physicist_activism"),
            ];

            for (document, category) in knowledge_documents {
                let extraction = self.extractor.extract_entities_and_relations(document).await
                    .map_err(|e| format!("Failed to extract from {}: {:?}", category, e))?;

                // Store in knowledge graph with category information
                {
                    let mut graph_lock = self.graph.write().await;
                    for entity in &extraction.entities {
                        let entity_data = EntityData {
                            attributes: [
                                ("name".to_string(), AttributeValue::String(entity.name.clone())),
                                ("type".to_string(), AttributeValue::String(entity.entity_type.clone())),
                                ("confidence".to_string(), AttributeValue::Float(entity.confidence)),
                                ("category".to_string(), AttributeValue::String(category.to_string())),
                                ("source_document".to_string(), AttributeValue::String(document.to_string())),
                            ].into_iter().collect(),
                        };

                        graph_lock.add_entity(
                            format!("{}_{}", category, entity.id),
                            entity_data
                        ).map_err(|e| format!("Failed to add entity from {}: {:?}", category, e))?;
                    }

                    for relation in &extraction.relations {
                        let relationship = Relationship {
                            target: format!("{}_{}", category, relation.object_id),
                            relationship_type: relation.predicate.clone(),
                            weight: relation.confidence,
                            properties: [
                                ("category".to_string(), AttributeValue::String(category.to_string())),
                                ("source_document".to_string(), AttributeValue::String(document.to_string())),
                            ].into_iter().collect(),
                        };

                        graph_lock.add_relationship(
                            format!("{}_{}", category, relation.subject_id),
                            relationship
                        ).map_err(|e| format!("Failed to add relationship from {}: {:?}", category, e))?;
                    }
                }

                // Create contextual embeddings
                for entity in &extraction.entities {
                    let embedding_context = format!("{} {} {} {}", 
                        entity.name, entity.entity_type, document, category);
                    
                    let embedding: Vec<f32> = (0..384).map(|i| {
                        (embedding_context.len() as f32 * i as f32 * 0.0001 + 
                         entity.confidence + 
                         category.len() as f32 * 0.01).sin()
                    }).collect();

                    self.embedding_store.add_embedding(
                        &format!("{}_{}", category, entity.id),
                        embedding
                    ).map_err(|e| format!("Failed to add embedding for {}: {:?}", category, e))?;
                }
            }

            Ok(())
        }

        async fn perform_multi_hop_reasoning(&self, query: &str, expected_hops: usize) -> Result<MultiHopResult, String> {
            let reasoning_start = Instant::now();

            // Gather entities from knowledge base for reasoning
            let graph_lock = self.graph.read().await;
            let stats = graph_lock.get_stats();
            drop(graph_lock);

            // Create comprehensive entity set for complex reasoning
            let reasoning_entities: Vec<_> = (0..stats.entity_count.min(50)).map(|i| {
                llmkg::extraction::Entity {
                    id: format!("reasoning_entity_{}", i),
                    name: format!("Knowledge Entity {}", i),
                    entity_type: "knowledge_concept".to_string(),
                    confidence: 0.8,
                }
            }).collect();

            let reasoning_result = self.orchestrator.process_complex_query(
                query,
                &reasoning_entities,
                &[]
            ).await.map_err(|e| format!("Multi-hop reasoning failed: {:?}", e))?;

            let reasoning_time = reasoning_start.elapsed();

            // Validate reasoning chain
            let actual_hops = reasoning_result.reasoning_steps.len();
            let hop_validation = if actual_hops >= expected_hops {
                "sufficient"
            } else if actual_hops > 0 {
                "partial"
            } else {
                "insufficient"
            };

            Ok(MultiHopResult {
                query: query.to_string(),
                confidence: reasoning_result.confidence,
                reasoning_steps: actual_hops,
                expected_hops,
                hop_validation: hop_validation.to_string(),
                explanation: reasoning_result.explanation,
                reasoning_time,
                reasoning_chain: reasoning_result.reasoning_steps,
            })
        }
    }

    #[derive(Debug)]
    struct MultiHopResult {
        query: String,
        confidence: f32,
        reasoning_steps: usize,
        expected_hops: usize,
        hop_validation: String,
        explanation: String,
        reasoning_time: Duration,
        reasoning_chain: Vec<String>,
    }

    #[tokio::test]
    async fn test_basic_multi_hop_reasoning() {
        let mut system = MultiHopReasoningSystem::new().await;
        
        // Build comprehensive knowledge base
        system.build_knowledge_base().await
            .expect("Failed to build knowledge base");

        // Test basic 2-hop reasoning
        let basic_queries = vec![
            ("How is Einstein connected to GPS technology?", 2),
            ("What is the relationship between Einstein's theories and nuclear power?", 3),
            ("How does relativity theory impact modern technology?", 2),
        ];

        for (query, expected_hops) in basic_queries {
            let result = system.perform_multi_hop_reasoning(query, expected_hops).await
                .expect(&format!("Failed to process basic multi-hop query: {}", query));

            // Validate basic reasoning
            assert!(result.confidence > 0.3,
                   "Basic multi-hop reasoning should be confident: {:.3} for '{}'", 
                   result.confidence, query);

            assert!(result.reasoning_steps >= 1,
                   "Should perform at least 1 reasoning step for '{}'", query);

            assert!(result.reasoning_time < Duration::from_secs(10),
                   "Basic multi-hop reasoning should be reasonably fast: {:?} for '{}'", 
                   result.reasoning_time, query);

            assert!(!result.explanation.is_empty(),
                   "Should provide explanation for multi-hop reasoning: '{}'", query);

            println!("✓ Basic multi-hop: '{}' -> {} steps, confidence: {:.3}, time: {:?}",
                    query, result.reasoning_steps, result.confidence, result.reasoning_time);
        }
    }

    #[tokio::test]
    async fn test_complex_multi_hop_reasoning() {
        let mut system = MultiHopReasoningSystem::new().await;
        
        system.build_knowledge_base().await
            .expect("Failed to build knowledge base");

        // Test complex multi-hop reasoning requiring deeper connections
        let complex_queries = vec![
            ("How did Einstein's work lead to modern medical imaging technology?", 4),
            ("What is the connection between Einstein's letter to Roosevelt and modern GPS systems?", 5),
            ("How do Einstein's theories connect nuclear physics to space exploration?", 4),
            ("What is the path from Einstein's relativity theory to quantum computing?", 3),
        ];

        let mut complex_results = Vec::new();

        for (query, expected_hops) in complex_queries {
            let result = system.perform_multi_hop_reasoning(query, expected_hops).await
                .expect(&format!("Failed to process complex multi-hop query: {}", query));

            // Complex reasoning should show sophisticated connections
            assert!(result.confidence > 0.2,
                   "Complex multi-hop reasoning should show some confidence: {:.3} for '{}'", 
                   result.confidence, query);

            assert!(result.reasoning_steps >= 2,
                   "Complex queries should require multiple steps: {} for '{}'", 
                   result.reasoning_steps, query);

            assert!(result.reasoning_time < Duration::from_secs(30),
                   "Complex reasoning should complete within reasonable time: {:?} for '{}'", 
                   result.reasoning_time, query);

            // Validate reasoning depth
            match result.hop_validation.as_str() {
                "sufficient" => {
                    assert!(result.reasoning_steps >= expected_hops,
                           "Sufficient reasoning should meet expected hops for '{}'", query);
                },
                "partial" => {
                    assert!(result.reasoning_steps > 0 && result.reasoning_steps < expected_hops,
                           "Partial reasoning should have some but fewer than expected hops for '{}'", query);
                },
                "insufficient" => {
                    // Even insufficient reasoning should provide some explanation
                    assert!(!result.explanation.is_empty(),
                           "Even insufficient reasoning should provide explanation for '{}'", query);
                }
            }

            complex_results.push(result);
            println!("✓ Complex multi-hop: '{}' -> {} steps (expected: {}), confidence: {:.3}, validation: {}",
                    query, complex_results.last().unwrap().reasoning_steps, expected_hops,
                    complex_results.last().unwrap().confidence, complex_results.last().unwrap().hop_validation);
        }

        // Analyze complex reasoning patterns
        let avg_complex_steps: f32 = complex_results.iter()
            .map(|r| r.reasoning_steps as f32)
            .sum::<f32>() / complex_results.len() as f32;

        let avg_complex_confidence: f32 = complex_results.iter()
            .map(|r| r.confidence)
            .sum::<f32>() / complex_results.len() as f32;

        assert!(avg_complex_steps >= 2.0,
               "Complex queries should average at least 2 reasoning steps: {:.1}", avg_complex_steps);

        assert!(avg_complex_confidence > 0.25,
               "Complex reasoning should maintain reasonable average confidence: {:.3}", avg_complex_confidence);

        println!("✓ Complex multi-hop reasoning analysis:");
        println!("  - Average reasoning steps: {:.1}", avg_complex_steps);
        println!("  - Average confidence: {:.3}", avg_complex_confidence);
    }

    #[tokio::test]
    async fn test_cross_domain_reasoning() {
        let mut system = MultiHopReasoningSystem::new().await;
        
        system.build_knowledge_base().await
            .expect("Failed to build knowledge base");

        // Test reasoning that crosses multiple domains
        let cross_domain_queries = vec![
            ("How do physics principles apply to medical technology?", vec!["physics", "medicine"]),
            ("What connects theoretical physics to space exploration?", vec!["physics", "space"]),
            ("How does nuclear physics relate to modern power generation?", vec!["nuclear", "technology"]),
            ("What is the connection between Einstein's work and modern computing?", vec!["physics", "computing"]),
        ];

        for (query, expected_domains) in cross_domain_queries {
            let result = system.perform_multi_hop_reasoning(query, 3).await
                .expect(&format!("Failed to process cross-domain query: {}", query));

            // Cross-domain reasoning should connect different areas
            assert!(result.confidence > 0.2,
                   "Cross-domain reasoning should show confidence: {:.3} for '{}'", 
                   result.confidence, query);

            assert!(result.reasoning_steps >= 2,
                   "Cross-domain queries should require multiple steps: {} for '{}'", 
                   result.reasoning_steps, query);

            // Check that explanation mentions multiple domains
            let explanation_lower = result.explanation.to_lowercase();
            let domain_mentions = expected_domains.iter()
                .filter(|domain| explanation_lower.contains(&domain.to_lowercase()))
                .count();

            assert!(domain_mentions >= 1,
                   "Cross-domain reasoning should mention relevant domains for '{}': found {}/{}", 
                   query, domain_mentions, expected_domains.len());

            println!("✓ Cross-domain: '{}' -> {} steps, confidence: {:.3}, domains mentioned: {}/{}",
                    query, result.reasoning_steps, result.confidence, domain_mentions, expected_domains.len());
        }
    }

    #[tokio::test]
    async fn test_causal_reasoning_chains() {
        let mut system = MultiHopReasoningSystem::new().await;
        
        system.build_knowledge_base().await
            .expect("Failed to build knowledge base");

        // Test causal reasoning that follows cause-and-effect chains
        let causal_queries = vec![
            ("What chain of events led from Einstein's theories to GPS accuracy?", "Einstein -> relativity -> time dilation -> GPS corrections"),
            ("How did theoretical physics lead to nuclear weapons?", "Einstein -> E=mc² -> nuclear reactions -> atomic bomb"),
            ("What is the causal path from Einstein's work to medical imaging?", "Einstein -> nuclear physics -> radioactive isotopes -> medical imaging"),
        ];

        for (query, expected_chain_description) in causal_queries {
            let result = system.perform_multi_hop_reasoning(query, 4).await
                .expect(&format!("Failed to process causal reasoning query: {}", query));

            // Causal reasoning should show clear logical progression
            assert!(result.confidence > 0.3,
                   "Causal reasoning should be confident: {:.3} for '{}'", 
                   result.confidence, query);

            assert!(result.reasoning_steps >= 3,
                   "Causal chains should have multiple steps: {} for '{}'", 
                   result.reasoning_steps, query);

            // Check for causal language in explanation
            let explanation_lower = result.explanation.to_lowercase();
            let causal_indicators = vec!["because", "therefore", "leads to", "results in", "causes", "due to"];
            let causal_language_count = causal_indicators.iter()
                .filter(|indicator| explanation_lower.contains(*indicator))
                .count();

            assert!(causal_language_count >= 1,
                   "Causal reasoning should use causal language for '{}': found {} indicators", 
                   query, causal_language_count);

            println!("✓ Causal reasoning: '{}' -> {} steps, confidence: {:.3}, causal indicators: {}",
                    query, result.reasoning_steps, result.confidence, causal_language_count);
            println!("  Expected chain: {}", expected_chain_description);
            println!("  Reasoning steps: {:?}", result.reasoning_chain.iter().take(3).collect::<Vec<_>>());
        }
    }

    #[tokio::test]
    async fn test_analogical_reasoning() {
        let mut system = MultiHopReasoningSystem::new().await;
        
        system.build_knowledge_base().await
            .expect("Failed to build knowledge base");

        // Test analogical reasoning that finds similarities across different contexts
        let analogical_queries = vec![
            ("How are GPS satellites similar to atomic clocks in terms of Einstein's theories?", "time_measurement"),
            ("What analogies exist between nuclear reactors and the sun in terms of energy production?", "energy_conversion"),
            ("How is the precision required for GPS similar to the precision needed in medical imaging?", "precision_requirements"),
        ];

        for (query, analogy_theme) in analogical_queries {
            let result = system.perform_multi_hop_reasoning(query, 3).await
                .expect(&format!("Failed to process analogical reasoning query: {}", query));

            // Analogical reasoning should make connections
            assert!(result.confidence > 0.2,
                   "Analogical reasoning should show confidence: {:.3} for '{}'", 
                   result.confidence, query);

            assert!(result.reasoning_steps >= 2,
                   "Analogical reasoning should involve multiple steps: {} for '{}'", 
                   result.reasoning_steps, query);

            // Check for analogical language
            let explanation_lower = result.explanation.to_lowercase();
            let analogical_indicators = vec!["similar", "like", "analogous", "comparable", "parallel", "both"];
            let analogical_language_count = analogical_indicators.iter()
                .filter(|indicator| explanation_lower.contains(*indicator))
                .count();

            assert!(analogical_language_count >= 1,
                   "Analogical reasoning should use comparative language for '{}': found {} indicators", 
                   query, analogical_language_count);

            println!("✓ Analogical reasoning: '{}' -> {} steps, confidence: {:.3}, analogical indicators: {}",
                    query, result.reasoning_steps, result.confidence, analogical_language_count);
            println!("  Theme: {}", analogy_theme);
        }
    }

    #[tokio::test]
    async fn test_reasoning_performance_under_complexity() {
        let mut system = MultiHopReasoningSystem::new().await;
        
        system.build_knowledge_base().await
            .expect("Failed to build knowledge base");

        // Test performance with increasingly complex reasoning requirements
        let complexity_levels = vec![
            (2, "How is Einstein connected to GPS?"),
            (4, "What is the path from Einstein's theories through nuclear physics to modern medical technology?"),
            (6, "How did Einstein's work lead to the development of technologies that eventually enabled modern quantum computing and space exploration?"),
            (8, "What is the complete causal chain from Einstein's theoretical work to modern applications in medicine, technology, space exploration, and quantum computing?"),
        ];

        let mut performance_metrics = Vec::new();

        for (expected_complexity, query) in complexity_levels {
            let complexity_start = Instant::now();
            
            let result = system.perform_multi_hop_reasoning(query, expected_complexity).await
                .expect(&format!("Failed to process complexity level {}", expected_complexity));

            let complexity_time = complexity_start.elapsed();

            // Validate performance scaling
            assert!(complexity_time < Duration::from_secs(20),
                   "Complex reasoning should complete within reasonable time: {:?} for complexity {}", 
                   complexity_time, expected_complexity);

            // Quality should degrade gracefully with complexity
            let quality_threshold = match expected_complexity {
                2 => 0.5,
                4 => 0.4,
                6 => 0.3,
                8 => 0.2,
                _ => 0.1,
            };

            assert!(result.confidence >= quality_threshold,
                   "Reasoning quality should degrade gracefully: {:.3} >= {:.3} for complexity {}", 
                   result.confidence, quality_threshold, expected_complexity);

            performance_metrics.push((expected_complexity, result.reasoning_steps, result.confidence, complexity_time));

            println!("✓ Complexity level {}: {} steps, confidence: {:.3}, time: {:?}",
                    expected_complexity, result.reasoning_steps, result.confidence, complexity_time);
        }

        // Analyze performance trends
        let max_time = performance_metrics.iter().map(|(_, _, _, time)| *time).max().unwrap();
        let min_confidence = performance_metrics.iter().map(|(_, _, conf, _)| *conf).fold(1.0, f32::min);

        assert!(max_time < Duration::from_secs(15),
               "Maximum reasoning time should be reasonable: {:?}", max_time);

        assert!(min_confidence > 0.15,
               "Minimum confidence should be acceptable: {:.3}", min_confidence);

        println!("✓ Multi-hop reasoning performance test passed:");
        println!("  - Maximum reasoning time: {:?}", max_time);
        println!("  - Minimum confidence: {:.3}", min_confidence);
        
        for (complexity, steps, confidence, time) in performance_metrics {
            println!("  - Complexity {}: {} steps, {:.3} confidence, {:?}", complexity, steps, confidence, time);
        }
    }
}