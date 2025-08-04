//! Standardized test utilities for vector search system
//! 
//! Provides consistent schema definitions, test index creation,
//! and common helper functions to prevent code duplication across tests.

use anyhow::Result;
use std::path::Path;
use tantivy::schema::{Schema, Field, TEXT, STORED};
use tantivy::{Index, IndexWriter, doc};
use tempfile::TempDir;

/// Standard schema configuration used across all tests
#[derive(Debug, Clone)]
pub struct StandardSchema {
    pub schema: Schema,
    pub file_path_field: Field,
    pub content_field: Field,
    pub raw_content_field: Field,
    pub chunk_index_field: Field,
}

impl StandardSchema {
    /// Create the standard schema used by all vector search components
    pub fn new() -> Self {
        let mut schema_builder = Schema::builder();
        
        // Standardized field definitions - ALL tasks must use these exact configurations
        let file_path_field = schema_builder.add_text_field("file_path", TEXT | STORED);
        let content_field = schema_builder.add_text_field("content", TEXT | STORED);
        let raw_content_field = schema_builder.add_text_field("raw_content", TEXT | STORED);
        let chunk_index_field = schema_builder.add_u64_field("chunk_index", STORED);
        
        let schema = schema_builder.build();
        
        StandardSchema {
            schema,
            file_path_field,
            content_field,
            raw_content_field,
            chunk_index_field,
        }
    }
}

/// Test index builder for consistent test setup
pub struct TestIndexBuilder {
    temp_dir: TempDir,
    index: Index,
    schema: StandardSchema,
}

impl TestIndexBuilder {
    /// Create a new test index with standard schema
    pub fn new() -> Result<Self> {
        let temp_dir = TempDir::new()?;
        let schema = StandardSchema::new();
        let index = Index::create_in_dir(temp_dir.path(), schema.schema.clone())?;
        
        Ok(TestIndexBuilder {
            temp_dir,
            index,
            schema,
        })
    }
    
    /// Add a document to the test index
    pub fn add_document(
        &self,
        writer: &mut IndexWriter,
        file_path: &str,
        content: &str,
        raw_content: &str,
        chunk_index: u64,
    ) -> Result<()> {
        let doc = doc!(
            self.schema.file_path_field => file_path,
            self.schema.content_field => content,
            self.schema.raw_content_field => raw_content,
            self.schema.chunk_index_field => chunk_index
        );
        
        writer.add_document(doc)?;
        Ok(())
    }
    
    /// Create multiple test documents with standard data
    pub fn create_test_documents(
        &self,
        documents: Vec<(String, String, String, u64)>,
    ) -> Result<()> {
        let mut writer = self.index.writer(50_000_000)?;
        
        for (file_path, content, raw_content, chunk_index) in documents {
            self.add_document(&mut writer, &file_path, &content, &raw_content, chunk_index)?;
        }
        
        writer.commit()?;
        Ok(())
    }
    
    /// Get the underlying index for testing
    pub fn get_index(&self) -> &Index {
        &self.index
    }
    
    /// Get the schema for testing
    pub fn get_schema(&self) -> &StandardSchema {
        &self.schema
    }
    
    /// Get the temporary directory path
    pub fn get_path(&self) -> &Path {
        self.temp_dir.path()
    }
}

/// Common test data generators
pub struct TestDataGenerator;

impl TestDataGenerator {
    /// Generate standard Rust code test documents
    pub fn rust_code_documents() -> Vec<(String, String, String, u64)> {
        vec![
            (
                "src/main.rs".to_string(),
                "fn main() { println!(\"Hello, world!\"); }".to_string(),
                "fn main() {\n    println!(\"Hello, world!\");\n}".to_string(),
                0,
            ),
            (
                "src/lib.rs".to_string(),
                "pub mod utils; pub use utils::*;".to_string(),
                "pub mod utils;\npub use utils::*;".to_string(),
                0,
            ),
            (
                "Cargo.toml".to_string(),
                "[workspace] members = [\"crate1\", \"crate2\"]".to_string(),
                "[workspace]\nmembers = [\"crate1\", \"crate2\"]".to_string(),
                0,
            ),
        ]
    }
    
    /// Generate documents with special characters commonly found in code
    pub fn special_character_documents() -> Vec<(String, String, String, u64)> {
        vec![
            (
                "test.rs".to_string(),
                "Result<T, E> #[derive(Debug)] Vec<Option<String>>".to_string(),
                "fn test() -> Result<T, E> {\n    #[derive(Debug)]\n    struct Test { field: Vec<Option<String>> }\n}".to_string(),
                0,
            ),
            (
                "config.toml".to_string(),
                "[workspace] dependencies = { version = \"1.0\" }".to_string(),
                "[workspace]\ndependencies = { version = \"1.0\" }".to_string(),
                0,
            ),
        ]
    }
    
    /// Generate large documents for performance testing
    pub fn large_documents() -> Vec<(String, String, String, u64)> {
        let large_content = "fn test() { println!(\"test\"); }\n".repeat(1000);
        vec![
            (
                "large_file.rs".to_string(),
                large_content.clone(),
                large_content,
                0,
            ),
        ]
    }
}

/// Create a test search engine with sample data
pub async fn create_test_search_engine() -> (crate::SearchEngine, TempDir) {
    let temp_dir = TempDir::new().unwrap();
    
    // Create SearchEngine using the existing constructor
    let engine = crate::SearchEngine::new().expect("Failed to create search engine");
    
    (engine, temp_dir)
}

/// Create a test boolean search engine
pub async fn create_test_boolean_engine() -> crate::boolean_logic::BooleanSearchEngine {
    let (base_engine, _temp_dir) = create_test_search_engine().await;
    crate::boolean_logic::BooleanSearchEngine::new(base_engine)
}

/// Sample test documents for boolean search testing
pub fn sample_documents() -> Vec<(String, String)> {
    vec![
        ("doc1".to_string(), "Rust programming language is memory safe".to_string()),
        ("doc2".to_string(), "Python programming with dynamic typing".to_string()),
        ("doc3".to_string(), "Rust struct implementation with methods".to_string()),
        ("doc4".to_string(), "Deprecated Python 2 code examples".to_string()),
        ("doc5".to_string(), "Modern Rust async programming patterns".to_string()),
    ]
}

/// Compilation verification helpers
pub struct CompilationVerifier;

impl CompilationVerifier {
    /// Verify that a crate compiles successfully
    pub fn verify_crate_compilation(crate_name: &str) -> Result<()> {
        use std::process::Command;
        
        let output = Command::new("cargo")
            .args(&["check", "-p", crate_name])
            .output()?;
        
        if !output.status.success() {
            return Err(anyhow::anyhow!(
                "Compilation failed for crate {}: {}",
                crate_name,
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        
        Ok(())
    }
    
    /// Verify that specific tests pass
    pub fn verify_tests(crate_name: &str, test_name: &str) -> Result<()> {
        use std::process::Command;
        
        let output = Command::new("cargo")
            .args(&["test", "-p", crate_name, test_name])
            .output()?;
        
        if !output.status.success() {
            return Err(anyhow::anyhow!(
                "Tests failed for {}: {}",
                test_name,
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_standard_schema_creation() {
        let schema = StandardSchema::new();
        assert_eq!(schema.schema.fields().count(), 4);
    }
    
    #[test]
    fn test_index_builder() -> Result<()> {
        let builder = TestIndexBuilder::new()?;
        let documents = TestDataGenerator::rust_code_documents();
        builder.create_test_documents(documents)?;
        
        let reader = builder.get_index().reader()?;
        let searcher = reader.searcher();
        assert_eq!(searcher.num_docs(), 3);
        
        Ok(())
    }
    
    #[test]
    fn test_special_characters() -> Result<()> {
        let builder = TestIndexBuilder::new()?;
        let documents = TestDataGenerator::special_character_documents();
        builder.create_test_documents(documents)?;
        
        let reader = builder.get_index().reader()?;
        let searcher = reader.searcher();
        assert_eq!(searcher.num_docs(), 2);
        
        Ok(())
    }
}