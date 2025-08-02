use clap::Parser;

/// Command line arguments for MCP server binaries
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Path to the knowledge graph data directory
    #[arg(short, long, default_value = "./llmkg_data")]
    pub data_dir: String,
    
    /// Embedding dimension for the knowledge graph
    #[arg(short, long, default_value = "96")]
    pub embedding_dim: usize,
    
    /// Enable debug logging
    #[arg(short = 'v', long)]
    pub verbose: bool,
}