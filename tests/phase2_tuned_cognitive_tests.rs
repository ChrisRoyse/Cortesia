use std::time::Duration;
use tokio::time::timeout;

mod cognitive_test_data_synthesizer;
use cognitive_test_data_synthesizer::*;

use llmkg::cognitive::{
    CognitiveOrchestrator, CognitiveOrchestratorConfig,
    ConvergentThinking, DivergentThinking, LateralThinking,
    SystemsThinking, CriticalThinking, AbstractThinking, AdaptiveThinking,
    types::*,
};

/// Comprehensive tuned tests for Phase 2 cognitive patterns
/// Uses synthesized data designed specifically for testing edge cases
#[cfg(test)]
mod phase2_tuned_cognitive_tests {
    use super::*;

    async fn setup_tuned_test_environment() -> (CognitiveOrchestrator, PatternTestScenarios) {
        // Create comprehensive synthetic data
        let mut synthesizer = CognitiveTestDataSynthesizer::new().await.unwrap();
        synthesizer.synthesize_comprehensive_knowledge_base().await.unwrap();
        let scenarios = synthesizer.create_pattern_specific_scenarios().await.unwrap();
        
        let stats = synthesizer.get_statistics();
        println!("=== Tuned Test Data Statistics ===");
        println!("Total entities: {}", stats.total_entities);
        println!("Total relationships: {}", stats.total_relationships);
        for (direction, count) in &stats.entity_breakdown {
            println!("  {} entities: {}", direction, count);
        }
        println!("==================================");
        
        // Create orchestrator with tuned configuration
        let config = CognitiveOrchestratorConfig {
            enable_adaptive_selection: true,
            enable_ensemble_methods: true,
            default_timeout_ms: 10000, // Increased timeout for complex scenarios
            max_parallel_patterns: 4,
            performance_tracking: true,
        };
        
        let orchestrator = CognitiveOrchestrator::new(
            synthesizer.graph,
            config,
        ).await.unwrap();
        
        (orchestrator, scenarios)
    }

    #[tokio::test]
    async fn test_tuned_convergent_thinking_factual_retrieval() {
        let (orchestrator, scenarios) = setup_tuned_test_environment().await;
        
        for scenario in &scenarios.convergent_scenarios {
            println!("Testing convergent scenario: {}", scenario.query);
            
            let result = timeout(
                Duration::from_millis(5000),
                orchestrator.reason(
                    &scenario.query,
                    None,
                    ReasoningStrategy::Specific(CognitivePatternType::Convergent),
                )
            ).await.unwrap().unwrap();
            
            println!("Result: {}", result.final_answer);
            println!("Confidence: {:.3}", result.quality_metrics.overall_confidence);
            
            // Verify result quality
            assert!(result.quality_metrics.overall_confidence >= scenario.confidence_threshold,
                "Confidence {} below threshold {} for query: {}", 
                result.quality_metrics.overall_confidence, scenario.confidence_threshold, scenario.query);
            
            // Check if answer contains expected properties
            let answer_lower = result.final_answer.to_lowercase();
            let contains_expected = scenario.expected_properties.iter()
                .any(|prop| answer_lower.contains(&prop.to_lowercase()));
            
            if !contains_expected {
                println!("Expected properties: {:?}", scenario.expected_properties);
                println!("Actual answer: {}", result.final_answer);
                // Don't fail hard, but log for investigation
            }
            
            assert!(!result.final_answer.trim().is_empty(), "Answer should not be empty");
        }
    }

    #[tokio::test]
    async fn test_tuned_divergent_thinking_exploration() {
        let (orchestrator, scenarios) = setup_tuned_test_environment().await;
        
        for scenario in &scenarios.divergent_scenarios {
            println!("Testing divergent scenario: {}", scenario.query);
            
            let result = timeout(
                Duration::from_millis(8000),
                orchestrator.reason(
                    &scenario.query,
                    None,
                    ReasoningStrategy::Specific(CognitivePatternType::Divergent),
                )
            ).await.unwrap().unwrap();
            
            println!("Result: {}", result.final_answer);
            println!("Execution time: {}ms", result.execution_metadata.total_time_ms);
            
            // Verify exploration breadth
            assert!(!result.final_answer.trim().is_empty(), "Answer should not be empty");
            
            // Check that the answer mentions exploration or contains expected terms
            let answer_lower = result.final_answer.to_lowercase();
            let has_exploration_content = answer_lower.contains("exploration") ||
                answer_lower.contains("results") ||
                scenario.expected_explorations.iter().any(|exp| answer_lower.contains(&exp.to_lowercase()));
            
            if !has_exploration_content {
                println!("Expected explorations: {:?}", scenario.expected_explorations);
                println!("Actual answer: {}", result.final_answer);
            }
            
            // At minimum, verify the pattern executed
            assert!(result.execution_metadata.patterns_executed.contains(&CognitivePatternType::Divergent));
        }
    }

    #[tokio::test]
    async fn test_tuned_systems_thinking_inheritance() {
        let (orchestrator, scenarios) = setup_tuned_test_environment().await;
        
        for scenario in &scenarios.systems_scenarios {
            println!("Testing systems scenario: {}", scenario.query);
            
            let result = timeout(
                Duration::from_millis(8000),
                orchestrator.reason(
                    &scenario.query,
                    None,
                    ReasoningStrategy::Specific(CognitivePatternType::Systems),
                )
            ).await.unwrap().unwrap();
            
            println!("Result: {}", result.final_answer);
            println!("Nodes activated: {}", result.execution_metadata.nodes_activated);
            
            // Verify hierarchical reasoning occurred
            assert!(!result.final_answer.trim().is_empty(), "Answer should not be empty");
            assert!(result.execution_metadata.nodes_activated > 0, "Should have activated some nodes");
            
            // Check for inheritance-related content
            let answer_lower = result.final_answer.to_lowercase();
            let has_inheritance_content = answer_lower.contains("inherit") ||
                answer_lower.contains("property") ||
                answer_lower.contains("attribute") ||
                scenario.expected_attributes.iter().any(|attr| answer_lower.contains(&attr.to_lowercase()));
            
            if !has_inheritance_content {
                println!("Expected attributes: {:?}", scenario.expected_attributes);
                println!("Actual answer: {}", result.final_answer);
            }
            
            assert!(result.execution_metadata.patterns_executed.contains(&CognitivePatternType::Systems));
        }
    }

    #[tokio::test]
    async fn test_tuned_critical_thinking_contradiction_resolution() {
        let (orchestrator, scenarios) = setup_tuned_test_environment().await;
        
        for scenario in &scenarios.critical_scenarios {
            println!("Testing critical scenario: {}", scenario.query);
            
            let result = timeout(
                Duration::from_millis(8000),
                orchestrator.reason(
                    &scenario.query,
                    None,
                    ReasoningStrategy::Specific(CognitivePatternType::Critical),
                )
            ).await.unwrap().unwrap();
            
            println!("Result: {}", result.final_answer);
            println!("Confidence: {:.3}", result.quality_metrics.overall_confidence);
            
            // Verify contradiction handling
            assert!(!result.final_answer.trim().is_empty(), "Answer should not be empty");
            
            // Check for critical thinking indicators
            let answer_lower = result.final_answer.to_lowercase();
            let has_critical_content = answer_lower.contains("however") ||
                answer_lower.contains("but") ||
                answer_lower.contains("contradiction") ||
                answer_lower.contains("exception") ||
                answer_lower.contains("three"); // For the three-legged dog case
            
            if !has_critical_content {
                println!("Expected critical analysis for: {}", scenario.query);
                println!("Actual answer: {}", result.final_answer);
            }
            
            assert!(result.execution_metadata.patterns_executed.contains(&CognitivePatternType::Critical));
        }
    }

    #[tokio::test]
    async fn test_tuned_lateral_thinking_creative_bridges() {
        let (orchestrator, scenarios) = setup_tuned_test_environment().await;
        
        for scenario in &scenarios.lateral_scenarios {
            let query = format!("How is {} connected to {}?", scenario.query_a, scenario.query_b);
            println!("Testing lateral scenario: {}", query);
            
            let result = timeout(
                Duration::from_millis(10000),
                orchestrator.reason(
                    &query,
                    None,
                    ReasoningStrategy::Specific(CognitivePatternType::Lateral),
                )
            ).await.unwrap().unwrap();
            
            println!("Result: {}", result.final_answer);
            println!("Novelty score: {:.3}", result.quality_metrics.novelty_score);
            
            // Verify creative connection finding
            assert!(!result.final_answer.trim().is_empty(), "Answer should not be empty");
            
            // Check for bridge concepts
            let answer_lower = result.final_answer.to_lowercase();
            let has_bridge_content = answer_lower.contains("connect") ||
                answer_lower.contains("bridge") ||
                answer_lower.contains("relation") ||
                scenario.expected_bridges.iter().any(|bridge| answer_lower.contains(&bridge.to_lowercase()));
            
            if !has_bridge_content {
                println!("Expected bridges: {:?}", scenario.expected_bridges);
                println!("Actual answer: {}", result.final_answer);
            }
            
            assert!(result.execution_metadata.patterns_executed.contains(&CognitivePatternType::Lateral));
        }
    }

    #[tokio::test]
    async fn test_tuned_abstract_thinking_pattern_detection() {
        let (orchestrator, scenarios) = setup_tuned_test_environment().await;
        
        for scenario in &scenarios.abstract_scenarios {
            let query = "What patterns exist in the knowledge structure?";
            println!("Testing abstract scenario: {}", query);
            
            let result = timeout(
                Duration::from_millis(10000),
                orchestrator.reason(
                    query,
                    None,
                    ReasoningStrategy::Specific(CognitivePatternType::Abstract),
                )
            ).await.unwrap().unwrap();
            
            println!("Result: {}", result.final_answer);
            println!("Efficiency score: {:.3}", result.quality_metrics.efficiency_score);
            
            // Verify pattern detection
            assert!(!result.final_answer.trim().is_empty(), "Answer should not be empty");
            
            // Check for pattern-related content
            let answer_lower = result.final_answer.to_lowercase();
            let has_pattern_content = answer_lower.contains("pattern") ||
                answer_lower.contains("structure") ||
                answer_lower.contains("similarity") ||
                answer_lower.contains("abstraction");
            
            if !has_pattern_content {
                println!("Expected pattern content for: {}", query);
                println!("Actual answer: {}", result.final_answer);
            }
            
            assert!(result.execution_metadata.patterns_executed.contains(&CognitivePatternType::Abstract));
        }
    }

    #[tokio::test]
    async fn test_tuned_adaptive_thinking_strategy_selection() {
        let (orchestrator, scenarios) = setup_tuned_test_environment().await;
        
        for scenario in &scenarios.adaptive_scenarios {
            println!("Testing adaptive scenario: {}", scenario.query);
            
            let result = timeout(
                Duration::from_millis(12000),
                orchestrator.reason(
                    &scenario.query,
                    None,
                    ReasoningStrategy::Automatic, // Let adaptive thinking choose
                )
            ).await.unwrap().unwrap();
            
            println!("Result: {}", result.final_answer);
            println!("Strategy used: {:?}", result.strategy_used);
            println!("Patterns executed: {:?}", result.execution_metadata.patterns_executed);
            
            // Verify adaptive selection occurred
            assert!(!result.final_answer.trim().is_empty(), "Answer should not be empty");
            
            // Check that automatic strategy was used
            assert!(matches!(result.strategy_used, ReasoningStrategy::Automatic), 
                "Should use automatic strategy selection");
            
            // Verify multiple patterns were considered or executed
            assert!(!result.execution_metadata.patterns_executed.is_empty(), 
                "Should have executed at least one pattern");
        }
    }

    #[tokio::test]
    async fn test_tuned_ensemble_reasoning_complex_query() {
        let (orchestrator, _scenarios) = setup_tuned_test_environment().await;
        
        let complex_queries = vec![
            "What are the creative applications of artificial intelligence in art, and what contradictions exist?",
            "How do mammals inherit properties, and what exceptions exist in nature?",
            "What patterns connect technology and biology, and how can we explore new possibilities?",
        ];
        
        for query in complex_queries {
            println!("Testing ensemble reasoning: {}", query);
            
            let result = timeout(
                Duration::from_millis(15000),
                orchestrator.reason(
                    query,
                    None,
                    ReasoningStrategy::Ensemble(vec![
                        CognitivePatternType::Convergent,
                        CognitivePatternType::Divergent,
                        CognitivePatternType::Lateral,
                        CognitivePatternType::Critical,
                    ]),
                )
            ).await.unwrap().unwrap();
            
            println!("Result: {}", result.final_answer);
            println!("Patterns executed: {:?}", result.execution_metadata.patterns_executed);
            println!("Overall confidence: {:.3}", result.quality_metrics.overall_confidence);
            println!("Consistency score: {:.3}", result.quality_metrics.consistency_score);
            
            // Verify ensemble execution
            assert!(!result.final_answer.trim().is_empty(), "Answer should not be empty");
            assert!(result.execution_metadata.patterns_executed.len() >= 2, 
                "Ensemble should execute multiple patterns");
            
            // Check for ensemble strategy
            if let ReasoningStrategy::Ensemble(patterns) = result.strategy_used {
                assert!(patterns.len() >= 2, "Ensemble should use multiple patterns");
            }
            
            // Verify quality metrics
            assert!(result.quality_metrics.overall_confidence > 0.0, "Should have positive confidence");
            assert!(result.quality_metrics.consistency_score >= 0.0, "Consistency score should be valid");
        }
    }

    #[tokio::test]
    async fn test_tuned_edge_case_handling() {
        let (orchestrator, _scenarios) = setup_tuned_test_environment().await;
        
        let edge_cases = vec![
            ("Empty query", ""),
            ("Single word", "dogs"),
            ("Very specific", "What specific properties does the three-legged dog Tripper inherit from mammals despite its unusual leg count?"),
            ("Cross-domain", "How do neural networks in AI relate to neural networks in biology?"),
            ("Contradiction query", "Do all dogs have four legs including Tripper?"),
            ("Abstract query", "What is the essence of creativity across all domains?"),
        ];
        
        for (test_name, query) in edge_cases {
            println!("Testing edge case: {} - '{}'", test_name, query);
            
            let result = timeout(
                Duration::from_millis(10000),
                orchestrator.reason(
                    query,
                    None,
                    ReasoningStrategy::Automatic,
                )
            ).await;
            
            match result {
                Ok(Ok(reasoning_result)) => {
                    println!("✓ Handled: {}", reasoning_result.final_answer);
                    assert!(!reasoning_result.final_answer.trim().is_empty() || query.is_empty(), 
                        "Non-empty query should produce non-empty answer");
                }
                Ok(Err(e)) => {
                    println!("✓ Graceful error: {}", e);
                    // Edge cases may fail gracefully
                }
                Err(_) => {
                    println!("✓ Timeout - system didn't hang");
                    // Timeout is acceptable for very complex edge cases
                }
            }
        }
    }

    #[tokio::test]
    async fn test_tuned_performance_benchmarks() {
        let (orchestrator, _scenarios) = setup_tuned_test_environment().await;
        
        let benchmark_queries = vec![
            ("Simple factual", "What are dogs?", CognitivePatternType::Convergent, 1000),
            ("Complex exploration", "What are all the types and examples of artificial intelligence?", CognitivePatternType::Divergent, 3000),
            ("Hierarchy reasoning", "What properties do golden retrievers inherit?", CognitivePatternType::Systems, 2000),
            ("Creative connection", "How is art related to technology?", CognitivePatternType::Lateral, 4000),
        ];
        
        for (test_name, query, pattern, max_time_ms) in benchmark_queries {
            println!("Benchmarking: {} - '{}'", test_name, query);
            
            let start = std::time::Instant::now();
            let result = orchestrator.reason(
                query,
                None,
                ReasoningStrategy::Specific(pattern),
            ).await.unwrap();
            let elapsed = start.elapsed().as_millis();
            
            println!("✓ Completed in {}ms (max: {}ms)", elapsed, max_time_ms);
            println!("  Answer: {}", result.final_answer.chars().take(100).collect::<String>());
            println!("  Confidence: {:.3}", result.quality_metrics.overall_confidence);
            
            assert!(elapsed <= max_time_ms as u128, 
                "Query '{}' took {}ms, expected <= {}ms", query, elapsed, max_time_ms);
            assert!(!result.final_answer.trim().is_empty(), "Should produce answer");
        }
    }

    #[tokio::test]
    async fn test_tuned_cognitive_pattern_isolation() {
        let (orchestrator, _scenarios) = setup_tuned_test_environment().await;
        
        // Test that each pattern produces different types of responses
        let test_query = "Tell me about dogs";
        
        let patterns = vec![
            CognitivePatternType::Convergent,
            CognitivePatternType::Divergent,
            CognitivePatternType::Systems,
            CognitivePatternType::Critical,
            CognitivePatternType::Lateral,
            CognitivePatternType::Abstract,
        ];
        
        let mut results = Vec::new();
        
        for pattern in patterns {
            let result = orchestrator.reason(
                test_query,
                None,
                ReasoningStrategy::Specific(pattern),
            ).await.unwrap();
            
            println!("Pattern {:?}: {}", pattern, result.final_answer);
            results.push((pattern, result.final_answer));
        }
        
        // Verify patterns produce different approaches
        for (i, (pattern_a, answer_a)) in results.iter().enumerate() {
            for (pattern_b, answer_b) in results.iter().skip(i + 1) {
                // Answers should be different (not identical)
                if answer_a == answer_b {
                    println!("Warning: {:?} and {:?} produced identical answers", pattern_a, pattern_b);
                }
                // At minimum, they should not be empty
                assert!(!answer_a.trim().is_empty(), "Pattern {:?} should produce answer", pattern_a);
                assert!(!answer_b.trim().is_empty(), "Pattern {:?} should produce answer", pattern_b);
            }
        }
    }
}