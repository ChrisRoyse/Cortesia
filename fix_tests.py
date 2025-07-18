#!/usr/bin/env python3

# Simple script to fix the test file by replacing all occurrences
import re

# Read the file
with open('tests/phase2_mcp_tests.rs', 'r') as f:
    content = f.read()

# Replace all CallTool patterns
content = re.sub(r'MCPRequest::CallTool\s*\{', 'MCPRequest {', content)

# Replace all handle_request calls
content = re.sub(r'\.handle_request\(', '.handle_tool_call(', content)

# Replace store_fact with store_knowledge
content = re.sub(r'"store_fact"', '"store_knowledge"', content)

# Replace ListTools pattern
content = re.sub(
    r'let request = MCPRequest::ListTools;\s*let response = server\.handle_request\(request\)\.await\.unwrap\(\);',
    'let tools = server.get_tools();\n        assert!(!tools.is_empty());\n        assert!(tools.iter().any(|t| t.name == "store_knowledge"));\n        assert!(tools.iter().any(|t| t.name == "neural_query"));',
    content
)

# Write the file back
with open('tests/phase2_mcp_tests.rs', 'w') as f:
    f.write(content)

print("Fixed test file")