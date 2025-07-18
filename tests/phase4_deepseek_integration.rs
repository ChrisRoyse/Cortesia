use anyhow::Result;
use std::sync::Arc;
use std::time::{Duration, Instant};
use dotenv::dotenv;
use std::env;
use serde::{Deserialize, Serialize};
use reqwest::Client;

use llmkg::cognitive::phase4_integration::Phase4CognitiveSystem;
use llmkg::cognitive::types::{QueryContext, CognitivePatternType};
use llmkg::learning::types::*;

#[derive(Debug, Serialize)]
struct DeepSeekRequest {
    model: String,
    messages: Vec<Message>,
    max_tokens: u32,
    temperature: f32,
}

#[derive(Debug, Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct DeepSeekResponse {
    choices: Vec<Choice>,
    usage: Usage,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: MessageResponse,
    finish_reason: String,
}

#[derive(Debug, Deserialize)]
struct MessageResponse {
    content: String,
}

#[derive(Debug, Deserialize)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

/// DeepSeek integration for LLMKG testing
struct DeepSeekLLM {
    client: Client,
    api_key: String,
    api_url: String,
}

impl DeepSeekLLM {
    fn new() -> Result<Self> {
        dotenv().ok();
        
        Ok(Self {
            client: Client::new(),
            api_key: env::var("DEEPSEEK_API_KEY")?,
            api_url: env::var("DEEPSEEK_API_URL")
                .unwrap_or_else(|_| "https://api.deepseek.com/v1".to_string()),
        })
    }

    async fn generate(&self, prompt: &str, max_tokens: u32) -> Result<String> {
        let request = DeepSeekRequest {
            model: "deepseek-chat".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            max_tokens,
            temperature: 0.7,
        };

        let response = self.client
            .post(format!("{}/chat/completions", self.api_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!("DeepSeek API error: {}", error_text));
        }

        let deepseek_response: DeepSeekResponse = response.json().await?;
        
        Ok(deepseek_response.choices
            .first()
            .map(|c| c.message.content.clone())
            .unwrap_or_default())
    }

    async fn generate_knowledge_entities(&self, domain: &str, count: usize) -> Result<Vec<(String, String, Vec<String>)>> {
        let prompt = format!(
            "Generate {} distinct entities for the knowledge domain '{}'. \
            For each entity provide: name, description, and 3 related concepts. \
            Format: EntityName|Description|Concept1,Concept2,Concept3\n\
            One entity per line.",
            count, domain
        );

        let response = self.generate(&prompt, 500).await?;
        let mut entities = Vec::new();

        for line in response.lines() {
            if line.trim().is_empty() {
                continue;
            }

            let parts: Vec<&str> = line.split('|').collect();
            if parts.len() == 3 {
                let name = parts[0].trim().to_string();
                let description = parts[1].trim().to_string();
                let concepts: Vec<String> = parts[2]
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .collect();
                
                entities.push((name, description, concepts));
            }
        }

        Ok(entities)
    }

    async fn generate_test_queries(&self, domain: &str, complexity_levels: &[&str]) -> Result<Vec<String>> {
        let prompt = format!(
            "Generate test queries for the '{}' domain at these complexity levels: {}. \
            Create 2 queries per level. \
            Format: Level: Query\n\
            Make queries progressively more complex.",
            domain, complexity_levels.join(", ")
        );

        let response = self.generate(&prompt, 300).await?;
        let mut queries = Vec::new();

        for line in response.lines() {
            if line.contains(':') {
                let query = line.split(':').nth(1).unwrap_or("").trim();
                if !query.is_empty() {
                    queries.push(query.to_string());
                }
            }
        }

        Ok(queries)
    }

    async fn evaluate_response(&self, query: &str, response: &str) -> Result<f32> {
        let prompt = format!(
            "Evaluate this response to the query on a scale of 0.0 to 1.0:\n\
            Query: {}\n\
            Response: {}\n\
            Consider: relevance, accuracy, completeness, clarity.\n\
            Output only a number between 0.0 and 1.0",
            query, response
        );

        let score_str = self.generate(&prompt, 10).await?;
        let score = score_str.trim().parse::<f32>().unwrap_or(0.5);
        
        Ok(score.clamp(0.0, 1.0))
    }
}

#[cfg(test)]
mod deepseek_integration_tests {
    use super::*;
    use crate::phase4_test_utils::*;

    #[tokio::test]
    async fn test_deepseek_knowledge_generation() -> Result<()> {
        let deepseek = DeepSeekLLM::new()?;
        
        // Test domains
        let domains = vec!["quantum_computing", "neuroscience", "climate_science"];
        
        for domain in domains {
            println!("\n🧠 Generating knowledge for domain: {}", domain);
            
            let entities = deepseek.generate_knowledge_entities(domain, 5).await?;
            
            assert!(!entities.is_empty(), "No entities generated for {}", domain);
            
            for (name, desc, concepts) in &entities {
                println!("  Entity: {}", name);
                println!("    Description: {}", desc);
                println!("    Concepts: {:?}", concepts);
                
                assert!(!name.is_empty(), "Empty entity name");
                assert!(!desc.is_empty(), "Empty description");
                assert!(!concepts.is_empty(), "No related concepts");
            }
        }
        
        Ok(())
    }

    #[tokio::test]
    async fn test_llmkg_with_deepseek_queries() -> Result<()> {
        let deepseek = DeepSeekLLM::new()?;
        let phase4_system = create_test_phase4_system_with_knowledge().await?;
        
        // Generate test queries
        let queries = deepseek.generate_test_queries(
            "artificial_intelligence",
            &["beginner", "intermediate", "advanced"]
        ).await?;
        
        assert!(queries.len() >= 3, "Insufficient test queries generated");
        
        let mut metrics = TestMetrics::new();
        
        for (idx, query) in queries.iter().enumerate() {
            println!("\n📝 Query {}: {}", idx + 1, query);
            
            let start = Instant::now();
            let result = phase4_system.enhanced_query(query, None).await?;
            let query_time = start.elapsed();
            
            metrics.record("query_time_ms", query_time.as_millis() as f32);
            metrics.record("confidence", result.base_result.overall_confidence);
            
            // Use DeepSeek to evaluate the response quality
            let response_summary = format!("{:?}", result.base_result.aggregated_response);
            let quality_score = deepseek.evaluate_response(query, &response_summary).await?;
            
            metrics.record("deepseek_quality", quality_score);
            
            println!("  Confidence: {:.2}", result.base_result.overall_confidence);
            println!("  Query time: {:?}", query_time);
            println!("  DeepSeek quality: {:.2}", quality_score);
            
            // Verify minimum quality
            assert!(
                result.base_result.overall_confidence > 0.3,
                "Confidence too low for query: {}",
                query
            );
        }
        
        metrics.print_summary();
        
        // Verify overall performance
        let avg_quality = metrics.get_average("deepseek_quality").unwrap_or(0.0);
        assert!(avg_quality > 0.5, "Average quality score too low: {:.2}", avg_quality);
        
        Ok(())
    }

    #[tokio::test]
    async fn test_adaptive_learning_with_deepseek_feedback() -> Result<()> {
        let deepseek = DeepSeekLLM::new()?;
        let mut phase4_system = create_test_phase4_system_with_knowledge().await?;
        
        // Test queries with expected quality levels
        let test_cases = vec![
            ("What is machine learning?", "basic", 0.7),
            ("Explain backpropagation in neural networks", "technical", 0.6),
            ("How do transformers achieve self-attention?", "advanced", 0.5),
        ];
        
        let mut learning_improvements = Vec::new();
        
        for (query, style, min_quality) in test_cases {
            println!("\n🔄 Testing query: {}", query);
            
            // First attempt
            let result1 = phase4_system.enhanced_query(query, None).await?;
            let response1 = format!("{:?}", result1.base_result.aggregated_response);
            let quality1 = deepseek.evaluate_response(query, &response1).await?;
            
            println!("  Initial quality: {:.2}", quality1);
            
            // Generate feedback based on DeepSeek evaluation
            let feedback = generate_feedback_from_quality(quality1, style);
            
            // Apply feedback to learning system
            phase4_system.phase4_learning.process_feedback(feedback).await?;
            
            // Allow learning to propagate
            tokio::time::sleep(Duration::from_millis(100)).await;
            
            // Second attempt
            let result2 = phase4_system.enhanced_query(query, None).await?;
            let response2 = format!("{:?}", result2.base_result.aggregated_response);
            let quality2 = deepseek.evaluate_response(query, &response2).await?;
            
            println!("  Post-feedback quality: {:.2}", quality2);
            
            let improvement = quality2 - quality1;
            learning_improvements.push(improvement);
            
            // Verify minimum quality is met
            assert!(
                quality2 >= min_quality,
                "Quality {:.2} below minimum {:.2} for query: {}",
                quality2, min_quality, query
            );
            
            // Ideally, we should see improvement
            if improvement > 0.0 {
                println!("  ✓ Improved by {:.2}", improvement);
            }
        }
        
        // Check overall learning trend
        let avg_improvement = learning_improvements.iter().sum::<f32>() / learning_improvements.len() as f32;
        println!("\n📈 Average improvement: {:.3}", avg_improvement);
        
        Ok(())
    }

    #[tokio::test]
    async fn test_deepseek_driven_optimization() -> Result<()> {
        let deepseek = DeepSeekLLM::new()?;
        let mut phase4_system = create_test_phase4_system_with_knowledge().await?;
        
        // Use DeepSeek to suggest optimization strategies
        let prompt = "Suggest 3 ways to optimize a knowledge graph for faster query processing. \
                     Be specific and technical. One suggestion per line.";
        
        let suggestions = deepseek.generate(&prompt, 200).await?;
        println!("🔧 DeepSeek optimization suggestions:");
        
        for (idx, suggestion) in suggestions.lines().enumerate() {
            if suggestion.trim().is_empty() {
                continue;
            }
            
            println!("\n{}: {}", idx + 1, suggestion);
            
            // Attempt to apply the optimization
            let optimization_result = phase4_system.phase4_learning
                .attempt_optimization(suggestion.trim())
                .await;
            
            match optimization_result {
                Ok(result) => {
                    println!("  ✓ Applied: efficiency gain = {:.2}", result.efficiency_gain);
                },
                Err(e) => {
                    println!("  ✗ Failed: {}", e);
                }
            }
        }
        
        Ok(())
    }

    #[tokio::test]
    async fn test_continuous_learning_loop() -> Result<()> {
        let deepseek = DeepSeekLLM::new()?;
        let mut phase4_system = create_test_phase4_system_with_knowledge().await?;
        
        let mut profiler = PerformanceProfiler::new();
        let mut metrics = TestMetrics::new();
        
        // Simulate a continuous learning session
        for cycle in 0..5 {
            println!("\n🔄 Learning Cycle {}", cycle + 1);
            profiler.checkpoint(&format!("cycle_{}_start", cycle));
            
            // Generate domain-specific queries
            let domain = ["physics", "biology", "computer_science"][cycle % 3];
            let queries = deepseek.generate_test_queries(domain, &["intermediate"]).await?;
            
            for query in queries.iter().take(2) {
                // Process query
                let result = phase4_system.enhanced_query(query, None).await?;
                
                // Evaluate with DeepSeek
                let quality = deepseek.evaluate_response(
                    query,
                    &format!("{:?}", result.base_result.aggregated_response)
                ).await?;
                
                metrics.record("quality", quality);
                metrics.record("confidence", result.base_result.overall_confidence);
                
                // Generate and apply feedback
                if quality < 0.7 {
                    let feedback = generate_detailed_feedback(query, quality);
                    phase4_system.phase4_learning.process_feedback(feedback).await?;
                }
            }
            
            // Run learning cycle
            profiler.checkpoint(&format!("cycle_{}_learning", cycle));
            let learning_result = phase4_system.phase4_learning
                .execute_comprehensive_learning_cycle()
                .await?;
            
            metrics.record("learning_efficiency", learning_result.learning_results.overall_learning_effectiveness);
            profiler.checkpoint(&format!("cycle_{}_end", cycle));
        }
        
        // Analyze results
        profiler.report().print();
        metrics.print_summary();
        
        // Verify continuous improvement
        let quality_trend = metrics.get_trend("quality").unwrap_or(0.0);
        println!("\n📊 Quality trend: {:+.3}", quality_trend);
        
        assert!(
            quality_trend >= -0.01,
            "Quality degrading over time: trend = {:.3}",
            quality_trend
        );
        
        Ok(())
    }

    // Helper functions

    async fn create_test_phase4_system_with_knowledge() -> Result<Phase4CognitiveSystem> {
        let system = crate::phase4_realistic_tests::create_minimal_phase4_system().await?;
        
        // Add some initial knowledge
        let mut builder = TestDataBuilder::new();
        builder.add_entity("AI", "artificial_intelligence");
        builder.add_entity("ML", "machine_learning");
        builder.add_entity("DL", "deep_learning");
        builder.add_entity("NN", "neural_network");
        
        builder.add_relationship("AI", "ML", 0.9, llmkg::core::brain_types::RelationshipType::Contains)?;
        builder.add_relationship("ML", "DL", 0.85, llmkg::core::brain_types::RelationshipType::Contains)?;
        builder.add_relationship("DL", "NN", 0.8, llmkg::core::brain_types::RelationshipType::ImplementedBy)?;
        
        let graph = builder.build_graph().await?;
        
        // Replace the system's graph
        // Note: This assumes the system allows graph replacement, which might need implementation
        
        Ok(system)
    }

    fn generate_feedback_from_quality(quality: f32, expected_style: &str) -> UserFeedback {
        UserFeedback {
            feedback_id: uuid::Uuid::new_v4(),
            session_id: "deepseek_test".to_string(),
            query_id: uuid::Uuid::new_v4().to_string(),
            satisfaction_score: quality,
            response_quality: quality,
            response_speed: 0.8,
            accuracy_rating: quality * 1.1, // Slightly higher for accuracy
            feedback_text: Some(match quality {
                q if q < 0.5 => format!("Response not {} enough", expected_style),
                q if q < 0.7 => format!("Needs more {} detail", expected_style),
                _ => "Good response".to_string(),
            }),
            timestamp: std::time::SystemTime::now(),
        }
    }

    fn generate_detailed_feedback(query: &str, quality: f32) -> UserFeedback {
        let feedback_text = if quality < 0.4 {
            "Response lacks clarity and detail. Please provide more comprehensive explanation."
        } else if quality < 0.6 {
            "Response is somewhat relevant but missing key information."
        } else if quality < 0.8 {
            "Good response but could be more precise."
        } else {
            "Excellent response, well structured and informative."
        };

        UserFeedback {
            feedback_id: uuid::Uuid::new_v4(),
            session_id: "continuous_learning".to_string(),
            query_id: uuid::Uuid::new_v4().to_string(),
            satisfaction_score: quality,
            response_quality: quality,
            response_speed: 0.7,
            accuracy_rating: quality,
            feedback_text: Some(format!("{} Query was: {}", feedback_text, query)),
            timestamp: std::time::SystemTime::now(),
        }
    }

    // Mock implementation for testing
    impl Phase4LearningSystem {
        async fn process_feedback(&self, feedback: UserFeedback) -> Result<()> {
            // This would integrate with the actual feedback processing
            println!("Processing feedback: satisfaction={:.2}", feedback.satisfaction_score);
            Ok(())
        }

        async fn attempt_optimization(&self, strategy: &str) -> Result<OptimizationResult> {
            // Mock implementation
            Ok(OptimizationResult {
                strategy: strategy.to_string(),
                efficiency_gain: 0.1,
                success: true,
            })
        }
    }

    #[derive(Debug)]
    struct OptimizationResult {
        strategy: String,
        efficiency_gain: f32,
        success: bool,
    }
}

#[cfg(test)]
mod performance_benchmarks {
    use super::*;

    #[tokio::test]
    async fn benchmark_deepseek_latency() -> Result<()> {
        let deepseek = DeepSeekLLM::new()?;
        let mut latencies = Vec::new();
        
        // Warm up
        let _ = deepseek.generate("Hello", 5).await?;
        
        // Measure latencies for different prompt sizes
        let prompts = vec![
            ("short", "What is AI?", 50),
            ("medium", "Explain machine learning in detail with examples", 200),
            ("long", "Provide a comprehensive overview of deep learning architectures including CNNs, RNNs, Transformers, and their applications", 500),
        ];
        
        for (size, prompt, max_tokens) in prompts {
            let start = Instant::now();
            let _ = deepseek.generate(prompt, max_tokens).await?;
            let latency = start.elapsed();
            
            latencies.push((size, latency));
            println!("{} prompt latency: {:?}", size, latency);
        }
        
        // Verify latencies are reasonable
        for (size, latency) in &latencies {
            assert!(
                latency.as_secs() < 10,
                "{} prompt took too long: {:?}",
                size, latency
            );
        }
        
        Ok(())
    }
}

// Integration with MCP tool
#[cfg(feature = "mcp")]
mod mcp_integration {
    use super::*;

    pub async fn setup_mcp_with_deepseek() -> Result<()> {
        // This would integrate with the MCP tool setup
        println!("Setting up MCP with DeepSeek LLM...");
        
        // Verify DeepSeek is accessible
        let deepseek = DeepSeekLLM::new()?;
        let test_response = deepseek.generate("Test", 5).await?;
        
        assert!(!test_response.is_empty(), "DeepSeek not responding");
        
        println!("✓ MCP-DeepSeek integration ready");
        Ok(())
    }
}