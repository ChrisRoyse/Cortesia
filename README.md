# LLMKG - Lightning-fast Knowledge Graph for LLM Integration

A high-performance knowledge graph system optimized for Large Language Model integration, built in Rust with WebAssembly support.

## Features

- **Lightning-fast performance**: Optimized for sub-millisecond query times
- **LLM-optimized**: Designed specifically for knowledge retrieval in AI applications
- **Multi-modal support**: Text processing, semantic embeddings, and graph operations
- **Cognitive architecture**: Brain-inspired processing with attention mechanisms
- **WebAssembly ready**: Cross-platform deployment including browsers
- **MCP server integration**: Compatible with Model Context Protocol
- **Neural model support**: Integration with HuggingFace Candle for ML inference

## Architecture

### Core Components

- **Knowledge Engine**: Core graph storage and retrieval system
- **Cognitive Systems**: Brain-inspired processing modules including attention, memory, and reasoning
- **Embedding Store**: Semantic similarity search with SIMD optimization
- **MCP Servers**: Multiple server implementations for different use cases
- **Neural Models**: Integration with SmolLM, TinyLlama, OpenELM, and MiniLM models
- **Production Systems**: Health checks, monitoring, and graceful shutdown

### Key Modules

- `src/core/` - Core knowledge graph and engine implementations
- `src/cognitive/` - Brain-inspired processing systems
- `src/mcp/` - Model Context Protocol server implementations
- `src/models/` - Neural model integration and registry
- `src/embedding/` - Semantic search and similarity operations
- `src/monitoring/` - Performance metrics and observability
- `src/production/` - Production-ready systems and error recovery

## Installation

### Prerequisites

- Rust 1.70+ with cargo
- Node.js 18+ (for WebAssembly features)
- Neo4j (optional, for graph persistence)

### Build from Source

```bash
git clone <repository-url>
cd LLMKG
cargo build --release
```

### WebAssembly Build

```bash
cargo build --target wasm32-unknown-unknown --features wasm
```

## Usage

### Basic Knowledge Graph Operations

```rust
use llmkg::core::KnowledgeEngine;

let mut engine = KnowledgeEngine::new();
engine.add_fact("Einstein", "invented", "relativity")?;
let results = engine.query("What did Einstein invent?")?;
```

### MCP Server

Run the MCP server for LLM integration:

```bash
cargo run --bin llmkg_mcp_server
```

### API Server

Start the REST API server:

```bash
cargo run --bin llmkg_api_server
```

## Configuration

The system supports various configuration options through:

- Environment variables
- Configuration files in `examples/unified_config_example.rs`
- Command-line arguments

## Model Integration

LLMKG supports multiple neural models:

### SmolLM Family
- SmolLM-135M, 360M, 1.7B variants
- SmolLM2 series with enhanced capabilities
- Instruct-tuned versions

### TinyLlama Family
- TinyLlama-1.1B base and chat variants
- Optimized for edge deployment

### OpenELM Family
- OpenELM-270M to 3B parameter variants
- Efficient language models

### MiniLM Family
- Embedding-focused models for semantic search

## Development

### Running Tests

```bash
cargo test
```

### Development Servers

```bash
# Brain-inspired server
cargo run --bin llmkg-brain-server

# Standard server
cargo run --bin llmkg-server
```

### Documentation Generation

```bash
cargo doc --open
```

## Performance

- **Query latency**: <1ms for typical knowledge retrieval
- **Inference latency**: <10ms for neural model operations
- **Memory efficiency**: Optimized memory layouts with string interning
- **Concurrent operations**: Lock-free data structures where possible

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Built with Rust for performance and safety
- Integrated with HuggingFace Candle for ML capabilities
- Inspired by cognitive neuroscience research
- Optimized for modern AI workflows