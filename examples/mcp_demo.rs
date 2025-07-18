#[cfg(feature = "native")]
use llmkg::mcp::{LLMKGMCPServer, MCPRequest};
use llmkg::error::Result;
use tokio;

#[cfg(feature = "native")]
#[tokio::main]
async fn main() -> Result<()> {
    println!("🔧 LLMKG MCP Integration Demo");
    println!("=============================");
    
    // Create MCP server
    let mcp_server = LLMKGMCPServer::new(96)?;
    println!("✅ MCP server initialized");
    
    // Get available tools
    let tools = mcp_server.get_tools();
    println!("\n📚 Available MCP Tools:");
    for (i, tool) in tools.iter().enumerate() {
        println!("  {}. {} - {}", i + 1, tool.name, tool.description);
    }
    
    // Test knowledge search
    println!("\n🔍 Testing knowledge_search tool...");
    let search_request = MCPRequest {
        method: "knowledge_search".to_string(),
        params: serde_json::json!({
            "query": "artificial intelligence machine learning",
            "max_entities": 10,
            "max_depth": 2
        }),
    };
    
    let search_response = mcp_server.handle_request(search_request).await;
    if !search_response.is_error {
        println!("✅ Knowledge search executed successfully");
        println!("  Response length: {} characters", search_response.content[0].text.len());
    } else {
        println!("⚠️  Knowledge search returned: {}", search_response.content[0].text);
    }
    
    // Test entity lookup
    println!("\n🔍 Testing entity_lookup tool...");
    let lookup_request = MCPRequest {
        method: "entity_lookup".to_string(),
        params: serde_json::json!({
            "description": "programming language"
        }),
    };
    
    let lookup_response = mcp_server.handle_request(lookup_request).await;
    if !lookup_response.is_error {
        println!("✅ Entity lookup executed successfully");
    } else {
        println!("⚠️  Entity lookup returned: {}", lookup_response.content[0].text);
    }
    
    // Test graph statistics
    println!("\n📊 Testing graph_statistics tool...");
    let stats_request = MCPRequest {
        method: "graph_statistics".to_string(),
        params: serde_json::json!({}),
    };
    
    let stats_response = mcp_server.handle_request(stats_request).await;
    if !stats_response.is_error {
        println!("✅ Graph statistics executed successfully");
        println!("  Statistics: {}", stats_response.content[0].text);
    } else {
        println!("⚠️  Graph statistics returned: {}", stats_response.content[0].text);
    }
    
    println!("\n🎉 MCP Integration Demo Complete!");
    println!("🔧 LLMs can now discover and use these tools automatically");
    println!("📡 Ready for production deployment with LLM systems");
    
    Ok(())
}

#[cfg(not(feature = "native"))]
fn main() {
    println!("❌ MCP demo requires 'native' feature. Run with: cargo run --example mcp_demo --features native");
}