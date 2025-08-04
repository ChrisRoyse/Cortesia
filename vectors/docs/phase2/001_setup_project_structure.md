# Task 001: Setup Project Structure and Dependencies

## Prerequisites
- Rust toolchain installed (1.70+)
- Empty vectors/ directory exists
- No prior Phase 2 implementation

## Context
You are implementing Phase 2 of a vector search system that adds boolean logic (AND/OR/NOT) functionality. This system will use Tantivy (a full-text search library) to index and search documents.

**Starting from scratch - no existing code assumed.**

## Your Task (10 minutes max)
Set up the foundational project structure for boolean search functionality by creating the necessary source files and updating dependencies.

## Success Criteria
1. Create `src/boolean.rs` file with module structure
2. Create `src/cross_chunk.rs` file with module structure  
3. Create `src/validator.rs` file with module structure
4. Add these modules to `src/lib.rs` or `src/main.rs`
5. Verify all files compile without errors using `cargo check`

## Implementation Steps

### 1. Create Project Structure
```bash
mkdir -p src
touch src/lib.rs
touch src/boolean.rs
touch src/cross_chunk.rs
touch src/validator.rs
```

### 2. Update src/lib.rs
```rust
// src/lib.rs
pub mod boolean;
pub mod cross_chunk;
pub mod validator;

// Re-export key types
pub use boolean::BooleanSearchEngine;
// Note: SearchResult will be defined in Task 005
// Note: DocumentResult will be defined in Task 002
pub use cross_chunk::CrossChunkBooleanHandler;
pub use validator::DocumentLevelValidator;
// Note: BooleanQueryStructure will be defined in Task 002
```

### 3. Create boolean.rs with failing test
```rust
// src/boolean.rs
use anyhow::Result;

pub struct BooleanSearchEngine {
    // Fields will be added in Task 003
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_boolean_search_engine_creation() {
        // This test should fail initially (RED phase)
        // Will be implemented in Task 003
        // let _engine = BooleanSearchEngine::new(&Path::new("test"));
    }
}
```

### 4. Create other module files
```rust
// src/cross_chunk.rs
pub struct CrossChunkBooleanHandler {
    // Fields will be added in Task 012
}

// src/validator.rs  
pub struct DocumentLevelValidator {
    // Fields will be added in Task 010
}
```

## Expected Output Structure
```rust
// src/boolean.rs
pub struct BooleanSearchEngine {
    // TODO: Add fields in next task
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_boolean_search_engine_creation() {
        // This test should fail initially
        let _engine = BooleanSearchEngine::new();
    }
}
```

## Cargo.toml Setup
Create or update `Cargo.toml` with:
```toml
[package]
name = "vector-search"
version = "0.1.0"
edition = "2021"

[dependencies]
tantivy = "0.21"
anyhow = "1.0"
serde = { version = "1.0", features = ["derive"] }

[dev-dependencies]
tempfile = "3.0"
tokio-test = "0.4"
```

## Module Declaration Updates
Update `src/lib.rs` (or create if it doesn't exist) to include:
```rust
// Add to src/lib.rs
pub mod boolean;
pub mod cross_chunk; 
pub mod validator;

// Re-export key types for easier access
pub use boolean::{BooleanSearchEngine};
pub use cross_chunk::CrossChunkBooleanHandler;
pub use validator::DocumentLevelValidator;
```

## Compilation Verification
```bash
# Verify project structure
ls -la src/
# Should show: lib.rs, boolean.rs, cross_chunk.rs, validator.rs

# Verify compilation
cargo check
# Should compile with warnings about unused items

# Run tests (should compile but test is commented out)
cargo test
```

## Creates for Future Tasks
- `src/boolean.rs` - Empty module file with basic BooleanSearchEngine struct
- `src/cross_chunk.rs` - Empty module file  
- `src/validator.rs` - Empty module file
- Updated module declarations in lib.rs

## Exports for Other Tasks
- BooleanSearchEngine struct (placeholder)
- Proper module structure for Phase 2 development

## Context for Next Task
Task 002 will define the core data structures:
- DocumentResult struct (in boolean.rs) 
- BooleanQueryStructure enum (in validator.rs)
- These will be the CANONICAL definitions used throughout Phase 2