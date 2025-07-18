//! MCP Server Unit Tests

use crate::unit::*;
use crate::unit::test_utils::*;
use crate::mcp::llm_friendly_server::*;

#[cfg(test)]
mod server_tests {
    use super::*;

    #[test]
    fn test_mcp_server_initialization() {
        let server = LlmFriendlyServer::new().unwrap();
        
        assert!(server.is_running());
        assert_eq!(server.protocol_version(), "2024-11-05");
    }

    #[test]
    fn test_mcp_tool_registration() {
        let mut server = LlmFriendlyServer::new().unwrap();
        
        // Register a test tool
        server.register_tool("query_graph", |args| {
            Ok(format!("Queried with: {:?}", args))
        }).unwrap();
        
        let tools = server.list_tools();
        assert!(tools.contains(&"query_graph".to_string()));
    }
}