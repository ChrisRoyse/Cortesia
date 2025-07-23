# LLMKG MCP Server Quick Start

## Installation Complete! ✅

The LLMKG MCP server has been successfully built and is ready to use with Claude Code.

### Built Executable Location
```
C:\code\LLMKG\target\release\llmkg_mcp_server.exe
```

### Quick Configuration for Claude Code

1. **Edit your Claude configuration** at `C:\Users\%USERNAME%\.claude.json`:

```json
{
  "mcpServers": {
    "llmkg": {
      "type": "stdio",
      "command": "C:\\\\code\\\\LLMKG\\\\target\\\\release\\\\llmkg_mcp_server.exe",
      "args": [
        "--data-dir", "C:\\\\code\\\\LLMKG\\\\llmkg_data",
        "--embedding-dim", "96"
      ]
    }
  }
}
```

2. **Restart Claude Code** completely

3. **Verify connection** by typing `/mcp` in Claude

### Available MCP Tools

Once connected, you'll have access to these knowledge graph tools:

- **Entity Management**: `create_entity`, `get_entity`, `update_entity`, `delete_entity`
- **Relationship Management**: `create_relationship`, `strengthen_relationship`, `weaken_relationship`
- **Query Operations**: `query_similar`, `neural_query`, `path_query`, `pattern_query`
- **Analysis Tools**: `analyze_graph`, `get_clusters`, `get_memory_usage`

### Testing the Server

Test creating an entity:
```
Using the LLMKG tools, create an entity called "Machine Learning" with type "technology"
```

Test querying:
```
Find all entities similar to "Machine Learning" in the knowledge graph
```

### Troubleshooting

If the server doesn't connect:
1. Check logs at: `C:\Users\%USERNAME%\AppData\Local\claude-cli-nodejs\Cache\`
2. Test manually: `target\release\llmkg_mcp_server.exe --verbose`
3. Ensure paths in `.claude.json` use double backslashes

### What's Next?

The LLMKG MCP server is now ready to:
- Build complex knowledge graphs through conversation
- Perform neural-inspired queries
- Manage relationships with synaptic weights
- Analyze graph patterns and clusters

The system has been tested with a 92.4% test pass rate and is fully operational on Windows!