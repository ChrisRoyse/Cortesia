use llmkg::core::knowledge_engine::KnowledgeEngine;
use llmkg::core::triple::Triple;
use llmkg::mcp::llm_friendly_server::{LLMFriendlyMCPServer, LLMMCPRequest};
use std::collections::HashMap;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🤖 LLM-Friendly Knowledge Graph Demo");
    println!("=====================================\\n");
    
    // Create the LLM-friendly MCP server
    let server = LLMFriendlyMCPServer::new()?;
    
    println!("🔧 Available Tools for LLMs:");
    println!("============================");
    let tools = server.get_tools();
    for tool in &tools {
        println!("📋 {}: {}", tool.name, tool.description);
        if !tool.examples.is_empty() {
            println!("   Example: {}", tool.examples[0].description);
        }
        println!();
    }
    
    // Demonstrate how an LLM would use the system
    demonstrate_llm_usage(&server).await?;
    
    // Show memory efficiency and anti-bloat measures
    demonstrate_memory_efficiency(&server).await?;
    
    // Show advanced features
    demonstrate_advanced_features(&server).await?;
    
    Ok(())
}

async fn demonstrate_llm_usage(server: &LLMFriendlyMCPServer) -> Result<(), Box<dyn std::error::Error>> {
    println!("🤖 Simulating LLM Usage Patterns:");
    println!("==================================\\n");
    
    // Step 1: LLM stores basic facts
    println!("Step 1: Storing basic facts about Einstein...");
    let facts = vec![
        ("Einstein", "is", "physicist"),
        ("Einstein", "invented", "relativity"),
        ("Einstein", "born_in", "Germany"),
        ("Einstein", "won", "Nobel_Prize"),
        ("relativity", "is", "theory"),
        ("relativity", "explains", "spacetime"),
        ("Nobel_Prize", "awarded_for", "photoelectric_effect"),
    ];
    
    for (subject, predicate, object) in facts {
        let request = LLMMCPRequest {
            method: "store_fact".to_string(),
            params: json!({
                "subject": subject,
                "predicate": predicate,
                "object": object
            }),
        };
        
        let response = server.handle_request(request).await;
        if response.success {
            println!("✅ {}", response.message);
        } else {
            println!("❌ {}", response.message);
        }
    }
    
    println!();
    
    // Step 2: LLM stores knowledge chunks
    println!("Step 2: Storing detailed knowledge...");
    let knowledge_chunks = vec![
        "Albert Einstein was a German theoretical physicist who developed the theory of relativity. He was born in 1879 and died in 1955.",
        "The theory of relativity has two parts: special relativity and general relativity. Special relativity deals with objects moving at constant speeds.",
        "Einstein won the Nobel Prize in Physics in 1921 for his work on the photoelectric effect, not for relativity as many people think."
    ];
    
    for chunk in knowledge_chunks {
        let request = LLMMCPRequest {
            method: "store_knowledge".to_string(),
            params: json!({
                "text": chunk,
                "tags": ["Einstein", "physics", "science"]
            }),
        };
        
        let response = server.handle_request(request).await;
        if response.success {
            println!("✅ {}", response.message);
        }
    }
    
    println!();
    
    // Step 3: LLM queries the knowledge
    println!("Step 3: LLM querying stored knowledge...");
    
    // Query 1: Find facts about Einstein
    println!("\\n🔍 Query: Find all facts about Einstein");
    let request = LLMMCPRequest {
        method: "find_facts".to_string(),
        params: json!({
            "subject": "Einstein",
            "limit": 10
        }),
    };
    
    let response = server.handle_request(request).await;
    println!("Response: {}", response.message);
    if let Some(facts) = response.data.get("facts_text") {
        if let Some(facts_array) = facts.as_array() {
            for fact in facts_array {
                println!("  • {}", fact.as_str().unwrap_or(""));
            }
        }
    }
    
    // Query 2: Ask a natural language question
    println!("\\n❓ Query: What did Einstein discover?");
    let request = LLMMCPRequest {
        method: "ask_question".to_string(),
        params: json!({
            "question": "What did Einstein discover?",
            "max_facts": 15
        }),
    };
    
    let response = server.handle_request(request).await;
    println!("Response: {}", response.message);
    if let Some(facts) = response.data.get("relevant_facts") {
        if let Some(facts_array) = facts.as_array() {
            for fact in facts_array.iter().take(5) {
                println!("  • {}", fact.as_str().unwrap_or(""));
            }
        }
    }
    
    // Query 3: Explore connections
    println!("\\n🌐 Query: Explore Einstein's connections");
    let request = LLMMCPRequest {
        method: "explore_connections".to_string(),
        params: json!({
            "entity": "Einstein",
            "max_hops": 2,
            "max_connections": 20
        }),
    };
    
    let response = server.handle_request(request).await;
    println!("Response: {}", response.message);
    if let Some(connections) = response.data.get("grouped_by_predicate") {
        if let Some(obj) = connections.as_object() {
            for (predicate, relationships) in obj {
                println!("  {} relationships:", predicate);
                if let Some(rels) = relationships.as_array() {
                    for rel in rels.iter().take(3) {
                        println!("    → {}", rel.as_str().unwrap_or(""));
                    }
                }
            }
        }
    }
    
    println!();
    
    Ok(())
}

async fn demonstrate_memory_efficiency(server: &LLMFriendlyMCPServer) -> Result<(), Box<dyn std::error::Error>> {
    println!("💾 Memory Efficiency & Anti-Bloat Demo:");
    println!("=======================================\\n");
    
    // Get current stats
    let request = LLMMCPRequest {
        method: "get_stats".to_string(),
        params: json!({}),
    };
    
    let response = server.handle_request(request).await;
    println!("📊 Current System Stats:");
    println!("{}", response.message);
    
    if let Some(memory) = response.data.get("memory") {
        if let Some(bytes_per_node) = memory.get("bytes_per_node") {
            println!("\\n🎯 Memory Efficiency:");
            println!("  • Bytes per node: {:.1} (target: <60)", bytes_per_node.as_f64().unwrap_or(0.0));
            println!("  • Status: {}", if bytes_per_node.as_f64().unwrap_or(0.0) < 60.0 { "✅ Excellent" } else { "⚠️ Needs optimization" });
        }
    }
    
    // Demonstrate chunk size validation
    println!("\\n🛡️ Anti-Bloat Protection:");
    println!("Testing chunk size limits...");
    
    let large_text = "x".repeat(3000); // Exceeds 2048 byte limit
    let request = LLMMCPRequest {
        method: "store_knowledge".to_string(),
        params: json!({
            "text": large_text
        }),
    };
    
    let response = server.handle_request(request).await;
    if !response.success {
        println!("✅ Properly rejected oversized chunk: {}", response.message);
    }
    
    // Demonstrate predicate validation
    println!("\\nTesting predicate length limits...");
    let request = LLMMCPRequest {
        method: "store_fact".to_string(),
        params: json!({
            "subject": "Test",
            "predicate": "this_is_way_too_long_predicate_that_should_be_rejected",
            "object": "Something"
        }),
    };
    
    let response = server.handle_request(request).await;
    if !response.success {
        println!("✅ Properly rejected oversized predicate: {}", response.message);
    }
    
    println!();
    
    Ok(())
}

async fn demonstrate_advanced_features(server: &LLMFriendlyMCPServer) -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Advanced LLM Features:");
    println!("=========================\\n");
    
    // Demonstrate predicate suggestions
    println!("💡 Getting predicate suggestions...");
    let request = LLMMCPRequest {
        method: "get_suggestions".to_string(),
        params: json!({
            "context": "I want to express that someone works at a company",
            "suggestion_type": "predicates"
        }),
    };
    
    let response = server.handle_request(request).await;
    println!("Response: {}", response.message);
    if let Some(predicates) = response.data.get("predicates") {
        println!("Suggested predicates: {:?}", predicates);
    }
    
    // Demonstrate helpful error messages
    println!("\\n🆘 Error handling for LLMs...");
    let request = LLMMCPRequest {
        method: "store_fact".to_string(),
        params: json!({
            "subject": "Test",
            // Missing predicate and object
        }),
    };
    
    let response = server.handle_request(request).await;
    println!("Error response: {}", response.message);
    println!("Helpful info: {}", response.helpful_info.unwrap_or_default());
    println!("Suggestions: {:?}", response.suggestions);
    
    // Show performance metrics
    println!("\\n⚡ Performance Metrics:");
    if let Some(perf) = response.data.get("performance") {
        println!("Performance data available for monitoring");
    }
    println!("Response time: {}ms", response.performance.response_time_ms);
    println!("Efficiency score: {:.2}", response.performance.efficiency_score);
    
    println!("\\n🎉 Demo Complete! The system is optimized for:");
    println!("  ✅ LLM-friendly SPO triple storage");
    println!("  ✅ Automatic chunk size validation (<512 tokens)");
    println!("  ✅ Memory efficiency (<60 bytes per entity)");
    println!("  ✅ Intuitive error messages and suggestions");
    println!("  ✅ Natural language query processing");
    println!("  ✅ Automatic fact extraction from text");
    
    Ok(())
}

// Example usage for different LLM scenarios
#[tokio::test]
async fn test_llm_scenarios() -> Result<(), Box<dyn std::error::Error>> {
    let server = LLMFriendlyMCPServer::new()?;
    
    // Scenario 1: LLM learning about a new topic
    println!("\\n📚 Scenario 1: LLM learning about quantum physics");
    
    let facts = vec![
        ("quantum_mechanics", "is", "physics_branch"),
        ("quantum_mechanics", "studies", "subatomic_particles"),
        ("Heisenberg", "created", "uncertainty_principle"),
        ("uncertainty_principle", "part_of", "quantum_mechanics"),
        ("Schrödinger", "created", "wave_equation"),
    ];
    
    for (s, p, o) in facts {
        let request = LLMMCPRequest {
            method: "store_fact".to_string(),
            params: json!({"subject": s, "predicate": p, "object": o}),
        };
        let response = server.handle_request(request).await;
        assert!(response.success);
    }
    
    // Scenario 2: LLM retrieving context for a question
    let request = LLMMCPRequest {
        method: "ask_question".to_string(),
        params: json!({
            "question": "What is quantum mechanics?",
            "max_facts": 10
        }),
    };
    
    let response = server.handle_request(request).await;
    assert!(response.success);
    assert!(response.data.get("relevant_facts").is_some());
    
    Ok(())
}

// Demonstrate the chunking strategy
fn demonstrate_optimal_chunk_sizes() {
    println!("\\n📏 Optimal Chunk Size Guidelines:");
    println!("==================================");
    println!("• Target: 512 tokens (~400 words, ~2KB)");
    println!("• Rationale: Balances context richness with memory efficiency");
    println!("• LLM Context Windows (2024):");
    println!("  - GPT-4: 32K tokens");
    println!("  - Claude: 200K tokens");
    println!("  - Gemini 1.5: 2M tokens");
    println!("• Our chunks fit efficiently in all context windows");
    println!("• Overlap strategy: 50% stride for context preservation");
}