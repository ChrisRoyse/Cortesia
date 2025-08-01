//! LLMKG MCP Server Binary
//! 
//! This binary provides a Model Context Protocol (MCP) server for the LLMKG system,
//! allowing Claude and other LLMs to interact with the knowledge graph.

use llmkg::mcp::llm_friendly_server::LLMFriendlyMCPServer;
use llmkg::core::knowledge_engine::KnowledgeEngine;
use llmkg::cli::Args;
use llmkg::error::{Result, GraphError};
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use serde_json::{json, Value};
use clap::Parser;


#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    
    // Initialize structured logging (tracing) for enhanced knowledge storage
    // and fallback to env_logger for other components
    if let Err(e) = llmkg::enhanced_knowledge_storage::logging::init_logging() {
        eprintln!("Failed to initialize structured logging: {e}");
        // Fallback to env_logger
        env_logger::Builder::from_default_env()
            .filter_level(if args.verbose { 
                log::LevelFilter::Debug 
            } else { 
                log::LevelFilter::Info 
            })
            .init();
    }
    
    tracing::info!("Starting LLMKG MCP Server with enhanced logging");
    log::info!("Data directory: {}", args.data_dir);
    log::info!("Embedding dimension: {}", args.embedding_dim);
    
    // Create data directory if needed
    std::fs::create_dir_all(&args.data_dir)?;
    
    // Initialize knowledge engine with max_nodes
    let knowledge_engine = Arc::new(RwLock::new(
        KnowledgeEngine::new(args.embedding_dim, 1_000_000)?
    ));
    
    // Create MCP server
    let mcp_server = LLMFriendlyMCPServer::new(knowledge_engine)?;
    
    // Set up stdio communication
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();
    let mut reader = BufReader::new(stdin);
    let mut stdout = stdout;
    
    log::info!("LLMKG MCP Server ready, listening on stdio");
    
    // Main message loop
    let mut line = String::new();
    loop {
        line.clear();
        match reader.read_line(&mut line).await {
            Ok(0) => {
                log::info!("EOF received, shutting down");
                break;
            }
            Ok(_) => {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                
                log::debug!("Received: {line}");
                
                // Parse JSON-RPC request
                match serde_json::from_str::<Value>(line) {
                    Ok(request) => {
                        let response = handle_mcp_request(&mcp_server, request).await;
                        let response_str = serde_json::to_string(&response)
                            .map_err(|e| GraphError::SerializationError(format!("JSON serialization failed: {e}")))?;
                        
                        log::debug!("Sending: {response_str}");
                        
                        stdout.write_all(response_str.as_bytes()).await?;
                        stdout.write_all(b"\n").await?;
                        stdout.flush().await?;
                    }
                    Err(e) => {
                        log::error!("Failed to parse request: {e}");
                        let error_response = json!({
                            "jsonrpc": "2.0",
                            "error": {
                                "code": -32700,
                                "message": "Parse error"
                            },
                            "id": null
                        });
                        
                        let error_str = serde_json::to_string(&error_response)
                            .map_err(|e| GraphError::SerializationError(format!("JSON serialization failed: {e}")))?;
                        stdout.write_all(error_str.as_bytes()).await?;
                        stdout.write_all(b"\n").await?;
                        stdout.flush().await?;
                    }
                }
            }
            Err(e) => {
                log::error!("Error reading stdin: {e}");
                break;
            }
        }
    }
    
    log::info!("LLMKG MCP Server shutting down");
    Ok(())
}

async fn handle_mcp_request(server: &LLMFriendlyMCPServer, request: Value) -> Value {
    let id = request.get("id").cloned().unwrap_or(Value::Null);
    let method = request.get("method").and_then(|m| m.as_str()).unwrap_or("");
    let params = request.get("params").cloned().unwrap_or(Value::Object(Default::default()));
    
    match method {
        "initialize" => {
            json!({
                "jsonrpc": "2.0",
                "result": {
                    "protocolVersion": "2025-06-18",
                    "serverInfo": {
                        "name": "LLMKG MCP Server",
                        "version": env!("CARGO_PKG_VERSION"),
                        "vendor": "LLMKG Project"
                    },
                    "capabilities": {
                        "tools": {}
                    }
                },
                "id": id
            })
        }
        "tools/list" => {
            let tools = server.get_available_tools();
            let tool_list: Vec<Value> = tools.into_iter().map(|tool| {
                json!({
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.input_schema
                })
            }).collect();
            
            json!({
                "jsonrpc": "2.0",
                "result": {
                    "tools": tool_list
                },
                "id": id
            })
        }
        "tools/call" => {
            // Convert params to MCP request format
            let tool_name = params.get("name").and_then(|n| n.as_str()).unwrap_or("");
            let tool_params = params.get("arguments").cloned().unwrap_or(Value::Object(Default::default()));
            
            let mcp_request = llmkg::mcp::shared_types::LLMMCPRequest {
                method: tool_name.to_string(),
                params: tool_params,
            };
            
            match server.handle_request(mcp_request).await {
                Ok(response) => {
                    json!({
                        "jsonrpc": "2.0",
                        "result": {
                            "content": [{
                                "type": "text",
                                "text": serde_json::to_string_pretty(&response.data)
                                    .unwrap_or_else(|_| "Error formatting response".to_string())
                            }]
                        },
                        "id": id
                    })
                }
                Err(e) => {
                    json!({
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32603,
                            "message": format!("Internal error: {}", e)
                        },
                        "id": id
                    })
                }
            }
        }
        _ => {
            json!({
                "jsonrpc": "2.0",
                "error": {
                    "code": -32601,
                    "message": "Method not found"
                },
                "id": id
            })
        }
    }
}