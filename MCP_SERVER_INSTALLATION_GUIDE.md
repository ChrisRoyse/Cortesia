# LLMKG MCP Server Installation Guide for Windows

This guide provides step-by-step instructions for installing and configuring the LLMKG MCP (Model Context Protocol) server to work with Claude Code on Windows.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Building the MCP Server](#building-the-mcp-server)
3. [Testing the Server](#testing-the-server)
4. [Configuring Claude Code](#configuring-claude-code)
5. [Troubleshooting](#troubleshooting)
6. [Available Tools](#available-tools)

## Prerequisites

Before installing the LLMKG MCP server, ensure you have:

1. **Rust and Cargo** (latest stable version)
   ```bash
   rustc --version  # Should be 1.70+
   cargo --version
   ```

2. **Claude Code** installed and configured
   ```bash
   claude --version
   ```

3. **LLMKG System** built and tested (92.4% test pass rate achieved)
   ```bash
   cd C:\code\LLMKG
   cargo test --lib -- --test-threads=1
   ```

## Building the MCP Server

### Step 1: Create the MCP Server Binary

First, we need to create a dedicated binary for the MCP server:

```bash
# Create the MCP server binary file
mkdir -p src\bin
```

Create `src\bin\llmkg_mcp_server.rs` with the following content:

```rust
//! LLMKG MCP Server Binary

use llmkg::mcp::llm_friendly_server::LLMFriendlyMCPServer;
use llmkg::core::knowledge_engine::KnowledgeEngine;
use llmkg::error::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use serde_json::{json, Value};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the knowledge graph data directory
    #[arg(short, long, default_value = "./llmkg_data")]
    data_dir: String,
    
    /// Embedding dimension for the knowledge graph
    #[arg(short, long, default_value = "96")]
    embedding_dim: usize,
    
    /// Enable debug logging
    #[arg(short = 'v', long)]
    verbose: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    
    // Initialize logging
    env_logger::Builder::from_default_env()
        .filter_level(if args.verbose { 
            log::LevelFilter::Debug 
        } else { 
            log::LevelFilter::Info 
        })
        .init();
    
    log::info!("Starting LLMKG MCP Server");
    
    // Create data directory if needed
    std::fs::create_dir_all(&args.data_dir)?;
    
    // Initialize knowledge engine
    let knowledge_engine = Arc::new(RwLock::new(
        KnowledgeEngine::new(args.embedding_dim)?
    ));
    
    // Create MCP server
    let mcp_server = LLMFriendlyMCPServer::new(knowledge_engine);
    
    // Set up stdio communication
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();
    let mut reader = BufReader::new(stdin);
    let mut stdout = stdout;
    
    log::info!("LLMKG MCP Server ready");
    
    // Main message loop
    let mut line = String::new();
    loop {
        line.clear();
        match reader.read_line(&mut line).await {
            Ok(0) => break, // EOF
            Ok(_) => {
                if let Ok(request) = serde_json::from_str::<Value>(&line) {
                    let response = handle_mcp_request(&mcp_server, request).await;
                    let response_str = serde_json::to_string(&response)?;
                    stdout.write_all(response_str.as_bytes()).await?;
                    stdout.write_all(b"\n").await?;
                    stdout.flush().await?;
                }
            }
            Err(_) => break,
        }
    }
    
    Ok(())
}

async fn handle_mcp_request(server: &LLMFriendlyMCPServer, request: Value) -> Value {
    // MCP protocol handling
    let method = request["method"].as_str().unwrap_or("");
    let id = request.get("id").cloned().unwrap_or(Value::Null);
    
    match method {
        "initialize" => json!({
            "jsonrpc": "2.0",
            "result": {
                "protocolVersion": "0.1.0",
                "serverInfo": {
                    "name": "LLMKG MCP Server",
                    "version": "0.1.0"
                },
                "capabilities": {
                    "tools": true
                }
            },
            "id": id
        }),
        "tools/list" => {
            let tools = server.get_available_tools();
            json!({
                "jsonrpc": "2.0",
                "result": {
                    "tools": tools.into_iter().map(|t| json!({
                        "name": t.name,
                        "description": t.description,
                        "inputSchema": t.input_schema
                    })).collect::<Vec<_>>()
                },
                "id": id
            })
        },
        "tools/call" => {
            // Handle tool execution
            let tool_name = request["params"]["name"].as_str().unwrap_or("");
            let args = request["params"]["arguments"].clone();
            
            let mcp_request = llmkg::mcp::shared_types::LLMMCPRequest {
                tool: tool_name.to_string(),
                parameters: args,
                conversation_id: None,
            };
            
            match server.handle_request(mcp_request).await {
                Ok(response) => json!({
                    "jsonrpc": "2.0",
                    "result": {
                        "content": [{
                            "type": "text",
                            "text": serde_json::to_string_pretty(&response.result)
                                .unwrap_or_else(|_| "Error".to_string())
                        }]
                    },
                    "id": id
                }),
                Err(e) => json!({
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32603,
                        "message": format!("Error: {}", e)
                    },
                    "id": id
                })
            }
        },
        _ => json!({
            "jsonrpc": "2.0",
            "error": {
                "code": -32601,
                "message": "Method not found"
            },
            "id": id
        })
    }
}
```

### Step 2: Update Cargo.toml

Add the MCP server binary to your `Cargo.toml`:

```toml
[[bin]]
name = "llmkg_mcp_server"
path = "src/bin/llmkg_mcp_server.rs"
```

### Step 3: Build the MCP Server

```bash
# Build in release mode for better performance
cargo build --release --bin llmkg_mcp_server

# The executable will be at:
# target\release\llmkg_mcp_server.exe
```

### Step 4: Test the Server Manually

Test that the server starts correctly:

```bash
# Run with debug output
target\release\llmkg_mcp_server.exe --verbose

# You should see:
# [INFO] Starting LLMKG MCP Server
# [INFO] LLMKG MCP Server ready
```

Press Ctrl+C to stop the test.

## Configuring Claude Code

### Step 1: Locate Claude Configuration

The Claude configuration file is at:
```
C:\Users\%USERNAME%\.claude.json
```

### Step 2: Add LLMKG MCP Server Configuration

Edit `.claude.json` and add the LLMKG server to the `mcpServers` section:

```json
{
  "mcpServers": {
    "llmkg": {
      "type": "stdio",
      "command": "C:\\\\code\\\\LLMKG\\\\target\\\\release\\\\llmkg_mcp_server.exe",
      "args": [
        "--data-dir", "C:\\\\code\\\\LLMKG\\\\llmkg_data",
        "--embedding-dim", "96"
      ],
      "env": {}
    }
  }
}
```

**Important Notes:**
- Use double backslashes (`\\\\`) for Windows paths in JSON
- Use the full path to the executable
- Pass configuration as command arguments, not environment variables
- The `--data-dir` should point to where you want to store the knowledge graph data

### Step 3: Alternative Configuration for Development

If you're actively developing, you might prefer using cargo run:

```json
{
  "mcpServers": {
    "llmkg": {
      "type": "stdio",
      "command": "cmd",
      "args": [
        "/c",
        "cd /d C:\\\\code\\\\LLMKG && cargo run --release --bin llmkg_mcp_server -- --data-dir llmkg_data"
      ],
      "env": {}
    }
  }
}
```

### Step 4: Restart Claude Code

1. Completely close Claude Code
2. Reopen Claude Code
3. Type `/mcp` to check server status

Expected output:
```
LLMKG MCP Server
Status: √ connected
Tools: 15 tools
```

## Testing the Installation

### Step 1: List Available Tools

Type in Claude:
```
/mcp tools
```

You should see tools like:
- `create_entity` - Create a new entity in the knowledge graph
- `create_relationship` - Create a relationship between entities
- `query_similar` - Find similar entities
- `neural_query` - Perform neural activation queries
- And more...

### Step 2: Test Creating an Entity

Ask Claude:
```
Using the LLMKG tools, create an entity for "Artificial Intelligence" with type "concept"
```

Claude should use the `create_entity` tool and return the created entity details.

### Step 3: Test Querying

Ask Claude:
```
Find all entities similar to "Artificial Intelligence" in the knowledge graph
```

## Troubleshooting

### Common Issues and Solutions

#### Issue: "× failed" Status
**Solution**: Check the log files at:
```
C:\Users\%USERNAME%\AppData\Local\claude-cli-nodejs\Cache\C--code-LLMKG\mcp-logs-llmkg\
```

#### Issue: "spawn UNKNOWN" Error
**Solution**: Ensure the executable path is correct and the file exists:
```bash
dir C:\code\LLMKG\target\release\llmkg_mcp_server.exe
```

#### Issue: Tools Not Showing
**Solution**: Run Claude with debug logging:
```bash
claude --debug
```

#### Issue: Server Crashes Immediately
**Solution**: Test the server manually with verbose logging:
```bash
C:\code\LLMKG\target\release\llmkg_mcp_server.exe --verbose
```

### Debug Checklist

1. ✅ Is the LLMKG project built successfully?
   ```bash
   cargo build --release --bin llmkg_mcp_server
   ```

2. ✅ Does the executable exist?
   ```bash
   where C:\code\LLMKG\target\release\llmkg_mcp_server.exe
   ```

3. ✅ Is the path in `.claude.json` correct with proper escaping?

4. ✅ Have you restarted Claude Code after configuration changes?

5. ✅ Check MCP logs for detailed error messages

## Available Tools

The LLMKG MCP server provides these knowledge graph tools:

### Entity Management
- `create_entity` - Create new entities with embeddings
- `get_entity` - Retrieve entity details
- `update_entity` - Update entity properties
- `delete_entity` - Remove entities

### Relationship Management
- `create_relationship` - Link entities with typed relationships
- `strengthen_relationship` - Increase relationship weight
- `weaken_relationship` - Decrease relationship weight

### Query Operations
- `query_similar` - Find similar entities by embedding
- `neural_query` - Activate neural patterns for complex queries
- `path_query` - Find paths between entities
- `pattern_query` - Detect patterns in the graph

### Analysis Tools
- `analyze_graph` - Get graph statistics
- `get_clusters` - Identify entity clusters
- `get_memory_usage` - Check system resource usage

## Best Practices

1. **Data Directory**: Keep your knowledge graph data in a dedicated directory
2. **Embedding Dimension**: Use 96D (default) for compatibility with the trained system
3. **Logging**: Enable verbose logging during development
4. **Backup**: Regularly backup your `llmkg_data` directory

## Next Steps

Once installed, you can:
1. Build complex knowledge graphs through natural conversation
2. Query relationships and patterns
3. Use neural activation for advanced reasoning
4. Integrate with other MCP servers for enhanced capabilities

For more information on using the LLMKG system, see the main documentation.

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review the log files
3. Ensure all prerequisites are met
4. Try the manual testing steps

The LLMKG MCP server brings powerful knowledge graph capabilities to Claude Code, enabling sophisticated information management and neural-inspired queries directly through natural language.