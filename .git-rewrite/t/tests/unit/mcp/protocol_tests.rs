//! MCP Protocol Unit Tests

use crate::unit::*;
use crate::unit::test_utils::*;
use crate::mcp::protocol::*;

#[cfg(test)]
mod protocol_tests {
    use super::*;

    #[test]
    fn test_mcp_message_serialization() {
        let message = McpMessage {
            jsonrpc: "2.0".to_string(),
            method: "initialize".to_string(),
            params: serde_json::json!({"protocol_version": "2024-11-05"}),
            id: Some(1),
        };
        
        let serialized = serde_json::to_string(&message).unwrap();
        let deserialized: McpMessage = serde_json::from_str(&serialized).unwrap();
        
        assert_eq!(message.method, deserialized.method);
        assert_eq!(message.id, deserialized.id);
    }
}