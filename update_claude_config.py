import json
import sys

# Read the current configuration
with open(r'C:\Users\hotra\.claude.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# Add LLMKG MCP server configuration
config['mcpServers']['llmkg'] = {
    "type": "stdio",
    "command": "C:\\\\code\\\\LLMKG\\\\target\\\\release\\\\llmkg_mcp_server.exe",
    "args": [
        "--data-dir", "C:\\\\code\\\\LLMKG\\\\llmkg_data",
        "--embedding-dim", "96"
    ]
}

# Write the updated configuration
with open(r'C:\Users\hotra\.claude.json', 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=2)

print("LLMKG MCP server configuration added successfully!")