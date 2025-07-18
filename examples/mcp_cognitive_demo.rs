use llmkg::mcp::brain_inspired_server::{BrainInspiredMCPServer, MCPRequest};
use llmkg::versioning::temporal_graph::TemporalKnowledgeGraph;
use llmkg::neural::neural_server::NeuralProcessingServer;
use llmkg::cognitive::orchestrator::CognitiveOrchestrator;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== LLMKG MCP Server with Cognitive Reasoning Demo ===\n");
    
    // Initialize components
    let graph = Arc::new(RwLock::new(TemporalKnowledgeGraph::new()));
    let neural_server = Arc::new(NeuralProcessingServer::new_test().await?);
    let cognitive_orchestrator = Arc::new(
        CognitiveOrchestrator::new(graph.clone(), neural_server.clone())
            .await?
    );
    
    // Create MCP server with cognitive capabilities
    let mut server = BrainInspiredMCPServer::new(
        graph.clone(),
        neural_server.clone(),
        Some(cognitive_orchestrator),
    ).await?;
    
    // Demonstrate neural-powered graph construction
    demonstrate_neural_graph_construction(&mut server).await?;
    
    // Demonstrate each cognitive pattern through MCP
    demonstrate_convergent_reasoning(&mut server).await?;
    demonstrate_divergent_reasoning(&mut server).await?;
    demonstrate_lateral_reasoning(&mut server).await?;
    demonstrate_systems_reasoning(&mut server).await?;
    demonstrate_critical_reasoning(&mut server).await?;
    demonstrate_abstract_reasoning(&mut server).await?;
    demonstrate_adaptive_reasoning(&mut server).await?;
    
    // Demonstrate neural query
    demonstrate_neural_query(&mut server).await?;
    
    println!("\n=== Demo Complete ===");
    Ok(())
}

async fn demonstrate_neural_graph_construction(server: &mut BrainInspiredMCPServer) -> Result<(), Box<dyn std::error::Error>> {
    println!("1. NEURAL-POWERED GRAPH CONSTRUCTION");
    println!("   Building knowledge graph with brain-inspired structure...\n");
    
    // Store facts using neural construction
    let facts = vec![
        "Dogs and cats are mammals. Mammals are warm-blooded animals with hair or fur.",
        "Dogs typically have four legs and bark. They are loyal companions.",
        "Cats also have four legs and meow. They are independent pets.",
        "Sparrows are small birds that can fly. Birds have wings and lay eggs.",
        "Fido is a 5-year-old brown dog who loves to play fetch.",
        "Whiskers is a gray cat who enjoys napping in sunny spots.",
        "Tripper is a special three-legged dog who needs extra care.",
        "AI can be used in art to create generative designs and enhance creativity.",
        "Machine learning models can analyze patterns in animal behavior.",
    ];
    
    for fact in facts {
        println!("   Storing: '{}'", fact);
        let request = MCPRequest::CallTool {
            tool: "store_fact".to_string(),
            arguments: json!({
                "text": fact,
                "use_neural_construction": true,
                "context": "Building demo knowledge base"
            }),
        };
        
        let response = server.handle_request(request).await?;
        match response {
            llmkg::mcp::brain_inspired_server::MCPResponse { content, is_error } => {
                if !is_error && !content.is_empty() {
                    println!("   ✓ {}", content[0].text);
                }
            }
        }
    }
    
    println!("\n   Knowledge graph construction complete!\n");
    Ok(())
}

async fn demonstrate_convergent_reasoning(server: &mut BrainInspiredMCPServer) -> Result<(), Box<dyn std::error::Error>> {
    println!("2. CONVERGENT REASONING (Direct Answers)");
    
    let queries = vec![
        "What sound does a dog make?",
        "How many legs does a typical cat have?",
        "What color is Fido?",
    ];
    
    for query in queries {
        println!("   Query: '{}'", query);
        let request = MCPRequest::CallTool {
            tool: "cognitive_reasoning".to_string(),
            arguments: json!({
                "query": query,
                "pattern": "convergent"
            }),
        };
        
        let response = server.handle_request(request).await?;
        match response {
            llmkg::mcp::brain_inspired_server::MCPResponse { content, is_error } => {
                if !is_error && !content.is_empty() {
                    println!("   Answer: {}\n", content[0].text);
                }
            }
        }
    }
    
    Ok(())
}

async fn demonstrate_divergent_reasoning(server: &mut BrainInspiredMCPServer) -> Result<(), Box<dyn std::error::Error>> {
    println!("3. DIVERGENT REASONING (Exploration)");
    
    let queries = vec![
        ("What are examples of mammals?", "instances"),
        ("What types of pets are there?", "categories"),
        ("What can dogs do?", "properties"),
    ];
    
    for (query, exploration_type) in queries {
        println!("   Query: '{}' ({})", query, exploration_type);
        let request = MCPRequest::CallTool {
            tool: "cognitive_reasoning".to_string(),
            arguments: json!({
                "query": query,
                "pattern": "divergent"
            }),
        };
        
        let response = server.handle_request(request).await?;
        match response {
            llmkg::mcp::brain_inspired_server::MCPResponse { content, is_error } => {
                if !is_error && !content.is_empty() {
                    println!("   Results: {}\n", content[0].text);
                }
            }
        }
    }
    
    Ok(())
}

async fn demonstrate_lateral_reasoning(server: &mut BrainInspiredMCPServer) -> Result<(), Box<dyn std::error::Error>> {
    println!("4. LATERAL REASONING (Creative Connections)");
    
    let connections = vec![
        ("dog", "cat", "How are dogs and cats related?"),
        ("AI", "art", "What connects AI to art?"),
        ("pet", "creativity", "How might pets relate to creativity?"),
    ];
    
    for (concept_a, concept_b, query) in connections {
        println!("   Finding connection: {} ↔ {}", concept_a, concept_b);
        println!("   Query: '{}'", query);
        
        let request = MCPRequest::CallTool {
            tool: "cognitive_reasoning".to_string(),
            arguments: json!({
                "query": query,
                "pattern": "lateral"
            }),
        };
        
        let response = server.handle_request(request).await?;
        match response {
            llmkg::mcp::brain_inspired_server::MCPResponse { content, is_error } => {
                if !is_error && !content.is_empty() {
                    println!("   Connection: {}\n", content[0].text);
                }
            }
        }
    }
    
    Ok(())
}

async fn demonstrate_systems_reasoning(server: &mut BrainInspiredMCPServer) -> Result<(), Box<dyn std::error::Error>> {
    println!("5. SYSTEMS REASONING (Hierarchical Analysis)");
    
    let queries = vec![
        "What properties do dogs inherit from being mammals?",
        "Where does Fido fit in the animal hierarchy?",
        "What characteristics are shared by all animals?",
    ];
    
    for query in queries {
        println!("   Query: '{}'", query);
        let request = MCPRequest::CallTool {
            tool: "cognitive_reasoning".to_string(),
            arguments: json!({
                "query": query,
                "pattern": "systems"
            }),
        };
        
        let response = server.handle_request(request).await?;
        match response {
            llmkg::mcp::brain_inspired_server::MCPResponse { content, is_error } => {
                if !is_error && !content.is_empty() {
                    println!("   Analysis: {}\n", content[0].text);
                }
            }
        }
    }
    
    Ok(())
}

async fn demonstrate_critical_reasoning(server: &mut BrainInspiredMCPServer) -> Result<(), Box<dyn std::error::Error>> {
    println!("6. CRITICAL REASONING (Contradiction Resolution)");
    
    println!("   Scenario: Dogs typically have 4 legs, but Tripper has only 3 legs.");
    
    let queries = vec![
        "How many legs does Tripper have?",
        "Is Tripper still a dog even with 3 legs?",
        "What makes Tripper special compared to other dogs?",
    ];
    
    for query in queries {
        println!("   Query: '{}'", query);
        let request = MCPRequest::CallTool {
            tool: "cognitive_reasoning".to_string(),
            arguments: json!({
                "query": query,
                "pattern": "critical"
            }),
        };
        
        let response = server.handle_request(request).await?;
        match response {
            llmkg::mcp::brain_inspired_server::MCPResponse { content, is_error } => {
                if !is_error && !content.is_empty() {
                    println!("   Resolution: {}\n", content[0].text);
                }
            }
        }
    }
    
    Ok(())
}

async fn demonstrate_abstract_reasoning(server: &mut BrainInspiredMCPServer) -> Result<(), Box<dyn std::error::Error>> {
    println!("7. ABSTRACT REASONING (Pattern Recognition)");
    
    let queries = vec![
        "What patterns exist in the pet data?",
        "What common attributes do animals share?",
        "Are there any recurring relationships in the knowledge graph?",
    ];
    
    for query in queries {
        println!("   Query: '{}'", query);
        let request = MCPRequest::CallTool {
            tool: "cognitive_reasoning".to_string(),
            arguments: json!({
                "query": query,
                "pattern": "abstract"
            }),
        };
        
        let response = server.handle_request(request).await?;
        match response {
            llmkg::mcp::brain_inspired_server::MCPResponse { content, is_error } => {
                if !is_error && !content.is_empty() {
                    println!("   Patterns: {}\n", content[0].text);
                }
            }
        }
    }
    
    Ok(())
}

async fn demonstrate_adaptive_reasoning(server: &mut BrainInspiredMCPServer) -> Result<(), Box<dyn std::error::Error>> {
    println!("8. ADAPTIVE REASONING (Automatic Pattern Selection)");
    
    let complex_queries = vec![
        "Tell me everything about Fido",
        "How do pets relate to human creativity and AI?",
        "Compare and contrast dogs and cats as pets",
        "What makes each animal in our database unique?",
    ];
    
    for query in complex_queries {
        println!("   Complex Query: '{}'", query);
        let request = MCPRequest::CallTool {
            tool: "cognitive_reasoning".to_string(),
            arguments: json!({
                "query": query,
                "pattern": "adaptive"
            }),
        };
        
        let response = server.handle_request(request).await?;
        match response {
            llmkg::mcp::brain_inspired_server::MCPResponse { content, is_error } => {
                if !is_error && !content.is_empty() {
                    println!("   Comprehensive Answer: {}\n", content[0].text);
                }
            }
        }
    }
    
    Ok(())
}

async fn demonstrate_neural_query(server: &mut BrainInspiredMCPServer) -> Result<(), Box<dyn std::error::Error>> {
    println!("9. NEURAL QUERY (Semantic Search)");
    
    let queries = vec![
        ("furry companions", "Find entities related to furry companions"),
        ("flying animals", "Search for flying animals"),
        ("special needs", "Find entities with special needs"),
    ];
    
    for (query, description) in queries {
        println!("   {}: '{}'", description, query);
        let request = MCPRequest::CallTool {
            tool: "neural_query".to_string(),
            arguments: json!({
                "query": query,
                "query_type": "semantic",
                "top_k": 3
            }),
        };
        
        let response = server.handle_request(request).await?;
        match response {
            llmkg::mcp::brain_inspired_server::MCPResponse { content, is_error } => {
                if !is_error && !content.is_empty() {
                    println!("   Results: {}\n", content[0].text);
                }
            }
        }
    }
    
    Ok(())
}