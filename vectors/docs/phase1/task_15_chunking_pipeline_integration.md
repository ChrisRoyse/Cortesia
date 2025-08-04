# Task 15: Integrate Complete Chunking Pipeline

**Estimated Time:** 10 minutes  
**Prerequisites:** Task 14 (Chunk validation)  
**Dependencies:** Tasks 01-14 must be completed

## Objective
Create a unified chunking pipeline that integrates all components: chunking, metadata enrichment, validation, and indexing into a single, efficient workflow.

## Context
All the individual components are now complete. This task creates the main pipeline that orchestrates the entire process from raw files to indexed, searchable chunks with proper error handling, logging, and performance monitoring.

## Task Details

### What You Need to Do

1. **Create the integrated pipeline in `src/pipeline.rs`:**

   ```rust
   use crate::chunker::{SmartChunker, TextChunk};
   use crate::metadata::{MetadataEnricher, EnrichedChunk};
   use crate::validation::{ChunkValidator, ValidationConfig, ValidationResult};
   use crate::indexing::ChunkIndexer;
   use crate::batch::{BatchProcessor, BatchConfig};
   use crate::incremental::IncrementalIndexer;
   use std::path::{Path, PathBuf};
   use std::time::{Duration, Instant};
   use anyhow::Result;
   use std::sync::atomic::{AtomicUsize, Ordering};
   use std::sync::Arc;
   
   #[derive(Debug, Clone)]
   pub struct PipelineConfig {
       pub chunker_config: crate::chunker::ChunkConfig,
       pub batch_config: BatchConfig,
       pub validation_config: ValidationConfig,
       pub enable_validation: bool,
       pub enable_metadata_enrichment: bool,
       pub enable_incremental: bool,
       pub log_rejected_chunks: bool,
       pub parallel_processing: bool,
   }
   
   impl Default for PipelineConfig {
       fn default() -> Self {
           Self {
               chunker_config: crate::chunker::ChunkConfig::default(),
               batch_config: BatchConfig::default(),
               validation_config: ValidationConfig::default(),
               enable_validation: true,
               enable_metadata_enrichment: true,
               enable_incremental: false,
               log_rejected_chunks: true,
               parallel_processing: true,
           }
       }
   }
   
   #[derive(Debug, Clone)]
   pub struct PipelineStats {
       pub files_processed: usize,
       pub files_failed: usize,
       pub chunks_created: usize,
       pub chunks_enriched: usize,
       pub chunks_validated: usize,
       pub chunks_rejected: usize,
       pub chunks_indexed: usize,
       pub total_processing_time: Duration,
       pub average_chunks_per_file: f32,
       pub validation_pass_rate: f32,
       pub enrichment_success_rate: f32,
   }
   
   #[derive(Debug, Clone)]
   pub struct ProcessedChunk {
       pub chunk: TextChunk,
       pub enriched: Option<EnrichedChunk>,
       pub validation: Option<ValidationResult>,
       pub indexed: bool,
       pub processing_time: Duration,
   }
   
   pub struct ChunkingPipeline {
       chunker: SmartChunker,
       enricher: Option<MetadataEnricher>,
       validator: Option<ChunkValidator>,
       indexer: ChunkIndexer,
       config: PipelineConfig,
       stats: Arc<AtomicStats>,
   }
   
   #[derive(Debug)]
   struct AtomicStats {
       files_processed: AtomicUsize,
       files_failed: AtomicUsize,
       chunks_created: AtomicUsize,
       chunks_enriched: AtomicUsize,
       chunks_validated: AtomicUsize,
       chunks_rejected: AtomicUsize,
       chunks_indexed: AtomicUsize,
   }
   
   impl AtomicStats {
       fn new() -> Self {
           Self {
               files_processed: AtomicUsize::new(0),
               files_failed: AtomicUsize::new(0),
               chunks_created: AtomicUsize::new(0),
               chunks_enriched: AtomicUsize::new(0),
               chunks_validated: AtomicUsize::new(0),
               chunks_rejected: AtomicUsize::new(0),
               chunks_indexed: AtomicUsize::new(0),
           }
       }
       
       fn to_stats(&self, total_time: Duration) -> PipelineStats {
           let files_processed = self.files_processed.load(Ordering::Relaxed);
           let chunks_created = self.chunks_created.load(Ordering::Relaxed);
           let chunks_validated = self.chunks_validated.load(Ordering::Relaxed);
           let chunks_enriched = self.chunks_enriched.load(Ordering::Relaxed);
           
           PipelineStats {
               files_processed,
               files_failed: self.files_failed.load(Ordering::Relaxed),
               chunks_created,
               chunks_enriched,
               chunks_validated,
               chunks_rejected: self.chunks_rejected.load(Ordering::Relaxed),
               chunks_indexed: self.chunks_indexed.load(Ordering::Relaxed),
               total_processing_time: total_time,
               average_chunks_per_file: if files_processed > 0 {
                   chunks_created as f32 / files_processed as f32
               } else {
                   0.0
               },
               validation_pass_rate: if chunks_validated > 0 {
                   1.0 - (self.chunks_rejected.load(Ordering::Relaxed) as f32 / chunks_validated as f32)
               } else {
                   0.0
               },
               enrichment_success_rate: if chunks_created > 0 {
                   chunks_enriched as f32 / chunks_created as f32
               } else {
                   0.0
               },
           }
       }
   }
   
   impl ChunkingPipeline {
       /// Create a new chunking pipeline
       pub fn new(indexer: ChunkIndexer, config: PipelineConfig) -> Result<Self> {
           let chunker = SmartChunker::with_config(config.chunker_config.clone())?;
           
           let enricher = if config.enable_metadata_enrichment {
               Some(MetadataEnricher::new()?)
           } else {
               None
           };
           
           let validator = if config.enable_validation {
               Some(ChunkValidator::new(config.validation_config.clone()))
           } else {
               None
           };
           
           let stats = Arc::new(AtomicStats::new());
           
           Ok(Self {
               chunker,
               enricher,
               validator,
               indexer,
               config,
               stats,
           })
       }
       
       /// Process a single file through the complete pipeline
       pub fn process_file<P: AsRef<Path>>(&mut self, file_path: P) -> Result<Vec<ProcessedChunk>> {
           let start_time = Instant::now();
           let file_path = file_path.as_ref();
           
           // Read file content
           let content = std::fs::read_to_string(file_path)?;
           let file_path_str = file_path.to_string_lossy().to_string();
           
           // Step 1: Create chunks
           let chunks = match self.chunker.create_chunks(&content, &file_path_str) {
               Ok(chunks) => {
                   self.stats.chunks_created.fetch_add(chunks.len(), Ordering::Relaxed);
                   chunks
               }
               Err(e) => {
                   self.stats.files_failed.fetch_add(1, Ordering::Relaxed);
                   return Err(e);
               }
           };
           
           let mut processed_chunks = Vec::new();
           
           // Process each chunk through the pipeline
           for chunk in chunks {
               let chunk_start = Instant::now();
               let processed = self.process_single_chunk(chunk)?;
               processed_chunks.push(processed);
           }
           
           self.stats.files_processed.fetch_add(1, Ordering::Relaxed);
           
           Ok(processed_chunks)
       }
       
       /// Process a single chunk through enrichment, validation, and indexing
       fn process_single_chunk(&mut self, chunk: TextChunk) -> Result<ProcessedChunk> {
           let start_time = Instant::now();
           let mut processed = ProcessedChunk {
               chunk,
               enriched: None,
               validation: None,
               indexed: false,
               processing_time: Duration::new(0, 0),
           };
           
           // Step 2: Enrich with metadata (if enabled)
           if let Some(ref mut enricher) = self.enricher {
               match enricher.enrich_chunk(processed.chunk.clone()) {
                   Ok(enriched) => {
                       processed.enriched = Some(enriched);
                       self.stats.chunks_enriched.fetch_add(1, Ordering::Relaxed);
                   }
                   Err(e) => {
                       if self.config.log_rejected_chunks {
                           eprintln!("Enrichment failed for chunk {}: {}", processed.chunk.id, e);
                       }
                       // Continue without enrichment
                   }
               }
           }
           
           // Step 3: Validate chunk quality (if enabled)
           let should_index = if let Some(ref validator) = self.validator {
               if let Some(ref enriched_chunk) = processed.enriched {
                   let validation_result = validator.validate_chunk(enriched_chunk);
                   let should_index = validation_result.is_valid;
                   
                   if !should_index && self.config.log_rejected_chunks {
                       eprintln!("Chunk {} rejected: {}", 
                                processed.chunk.id, 
                                validation_result.issues.iter()
                                   .map(|i| &i.message)
                                   .collect::<Vec<_>>()
                                   .join(", "));
                   }
                   
                   processed.validation = Some(validation_result);
                   self.stats.chunks_validated.fetch_add(1, Ordering::Relaxed);
                   
                   if !should_index {
                       self.stats.chunks_rejected.fetch_add(1, Ordering::Relaxed);
                   }
                   
                   should_index
               } else {
                   // No enriched chunk, create basic validation
                   let basic_enriched = EnrichedChunk {
                       chunk: processed.chunk.clone(),
                       metadata: crate::metadata::ChunkMetadata {
                           complexity_score: 1.0,
                           has_documentation: false,
                           doc_comment_count: 0,
                           imports: Vec::new(),
                           exports: Vec::new(),
                           function_signatures: Vec::new(),
                           type_definitions: Vec::new(),
                           keywords: std::collections::HashSet::new(),
                           identifiers: std::collections::HashSet::new(),
                           string_literals: Vec::new(),
                           comment_ratio: 0.0,
                           line_count: processed.chunk.content.lines().count(),
                           cyclomatic_complexity: 1,
                       },
                   };
                   
                   let validation_result = validator.validate_chunk(&basic_enriched);
                   let should_index = validation_result.is_valid;
                   
                   processed.validation = Some(validation_result);
                   self.stats.chunks_validated.fetch_add(1, Ordering::Relaxed);
                   
                   if !should_index {
                       self.stats.chunks_rejected.fetch_add(1, Ordering::Relaxed);
                   }
                   
                   should_index
               }
           } else {
               true // No validation, always index
           };
           
           // Step 4: Index chunk (if it passed validation)
           if should_index {
               match self.indexer.index_chunk(processed.chunk.clone()) {
                   Ok(()) => {
                       processed.indexed = true;
                       self.stats.chunks_indexed.fetch_add(1, Ordering::Relaxed);
                   }
                   Err(e) => {
                       eprintln!("Indexing failed for chunk {}: {}", processed.chunk.id, e);
                       // Don't fail the entire process for indexing errors
                   }
               }
           }
           
           processed.processing_time = start_time.elapsed();
           Ok(processed)
       }
       
       /// Process a directory using batch processing
       pub fn process_directory<P: AsRef<Path>>(&mut self, dir_path: P) -> Result<PipelineStats> {
           let start_time = Instant::now();
           
           if self.config.enable_incremental {
               // Use incremental processing
               self.process_directory_incremental(dir_path)
           } else {
               // Use batch processing
               self.process_directory_batch(dir_path)
           }
       }
       
       /// Process directory using batch processing
       fn process_directory_batch<P: AsRef<Path>>(&mut self, dir_path: P) -> Result<PipelineStats> {
           let start_time = Instant::now();
           
           // Find all files
           let files = self.find_files(dir_path.as_ref())?;
           
           // Process files
           for file_path in files {
               match self.process_file(&file_path) {
                   Ok(_) => {} // Success tracked in process_file
                   Err(e) => {
                       self.stats.files_failed.fetch_add(1, Ordering::Relaxed);
                       if self.config.log_rejected_chunks {
                           eprintln!("Failed to process file {}: {}", file_path.display(), e);
                       }
                   }
               }
           }
           
           // Commit all changes
           self.indexer.commit()?;
           
           let total_time = start_time.elapsed();
           Ok(self.stats.to_stats(total_time))
       }
       
       /// Process directory using incremental processing
       fn process_directory_incremental<P: AsRef<Path>>(&mut self, _dir_path: P) -> Result<PipelineStats> {
           // This would integrate with IncrementalIndexer
           // For now, fall back to batch processing
           // TODO: Implement incremental pipeline integration
           Err(anyhow::anyhow!("Incremental processing not yet integrated"))
       }
       
       /// Find all processable files in directory
       fn find_files(&self, dir_path: &Path) -> Result<Vec<PathBuf>> {
           let mut files = Vec::new();
           self.find_files_recursive(dir_path, &mut files, 0)?;
           Ok(files)
       }
       
       /// Recursively find files
       fn find_files_recursive(&self, dir_path: &Path, files: &mut Vec<PathBuf>, depth: usize) -> Result<()> {
           if let Some(max_depth) = self.config.batch_config.max_depth {
               if depth > max_depth {
                   return Ok(());
               }
           }
           
           let entries = std::fs::read_dir(dir_path)?;
           
           for entry in entries {
               let entry = entry?;
               let path = entry.path();
               
               if path.is_dir() {
                   if self.config.batch_config.follow_symlinks || !self.is_symlink(&path) {
                       self.find_files_recursive(&path, files, depth + 1)?;
                   }
               } else if self.should_process_file(&path) {
                   files.push(path);
               }
           }
           
           Ok(())
       }
       
       /// Check if file should be processed
       fn should_process_file(&self, file_path: &Path) -> bool {
           // Check extension
           if let Some(extension) = file_path.extension() {
               let ext = extension.to_string_lossy().to_lowercase();
               self.config.batch_config.supported_extensions.contains(&ext)
           } else {
               false
           }
       }
       
       /// Check if path is a symlink
       fn is_symlink(&self, path: &Path) -> bool {
           path.symlink_metadata()
               .map(|m| m.file_type().is_symlink())
               .unwrap_or(false)
       }
       
       /// Get current pipeline statistics
       pub fn get_stats(&self) -> PipelineStats {
           self.stats.to_stats(Duration::new(0, 0))
       }
       
       /// Reset statistics
       pub fn reset_stats(&self) {
           self.stats.files_processed.store(0, Ordering::Relaxed);
           self.stats.files_failed.store(0, Ordering::Relaxed);
           self.stats.chunks_created.store(0, Ordering::Relaxed);
           self.stats.chunks_enriched.store(0, Ordering::Relaxed);
           self.stats.chunks_validated.store(0, Ordering::Relaxed);
           self.stats.chunks_rejected.store(0, Ordering::Relaxed);
           self.stats.chunks_indexed.store(0, Ordering::Relaxed);
       }
       
       /// Configure pipeline for different use cases
       pub fn configure_for_speed(&mut self) {
           self.config.enable_metadata_enrichment = false;
           self.config.enable_validation = false;
           self.config.parallel_processing = true;
       }
       
       pub fn configure_for_quality(&mut self) {
           self.config.enable_metadata_enrichment = true;
           self.config.enable_validation = true;
           self.config.validation_config.require_documentation = true;
       }
       
       pub fn configure_for_development(&mut self) {
           self.config.log_rejected_chunks = true;
           self.config.enable_incremental = true;
       }
   }
   ```

2. **Add module declaration to `src/lib.rs`:**

   ```rust
   pub mod pipeline;
   ```

3. **Create integration tests for the complete pipeline:**

   ```rust
   #[cfg(test)]
   mod pipeline_integration_tests {
       use super::*;
       use crate::schema::create_schema;
       use tempfile::TempDir;
       use tantivy::Index;
       use std::fs;
       
       fn create_test_pipeline() -> Result<(TempDir, ChunkingPipeline)> {
           let temp_dir = TempDir::new()?;
           let schema = create_schema();
           let index = Index::create_in_dir(temp_dir.path().join("index"), schema)?;
           let indexer = ChunkIndexer::new(index)?;
           let config = PipelineConfig::default();
           
           let pipeline = ChunkingPipeline::new(indexer, config)?;
           
           Ok((temp_dir, pipeline))
       }
       
       fn create_test_files(temp_dir: &TempDir) -> Result<()> {
           // Good Rust file
           fs::write(temp_dir.path().join("good.rs"), r#"
               /// Well documented function
               pub fn calculate(x: i32, y: i32) -> i32 {
                   x + y
               }
               
               /// Another good function
               pub fn process_data(data: &[String]) -> Vec<String> {
                   data.iter().map(|s| s.to_uppercase()).collect()
               }
           "#)?;
           
           // Poor quality file (should be rejected)
           fs::write(temp_dir.path().join("poor.rs"), r#"
               fn x() {}
               fn y() {}
           "#)?;
           
           // Python file
           fs::write(temp_dir.path().join("script.py"), r#"
               def hello_world():
                   """Simple greeting function."""
                   print("Hello, world!")
               
               class Calculator:
                   def add(self, a: int, b: int) -> int:
                       return a + b
           "#)?;
           
           // Text file
           fs::write(temp_dir.path().join("readme.txt"), r#"
               This is a documentation file that contains
               useful information about the project.
               It should be processed as plain text.
           "#)?;
           
           Ok(())
       }
       
       #[test]
       fn test_pipeline_creation() -> Result<()> {
           let (_temp_dir, _pipeline) = create_test_pipeline()?;
           Ok(())
       }
       
       #[test]
       fn test_single_file_processing() -> Result<()> {
           let (temp_dir, mut pipeline) = create_test_pipeline()?;
           
           let test_file = temp_dir.path().join("test.rs");
           fs::write(&test_file, r#"
               /// Calculate sum of two numbers
               pub fn add(a: i32, b: i32) -> i32 {
                   a + b
               }
           "#)?;
           
           let processed_chunks = pipeline.process_file(&test_file)?;
           
           assert!(!processed_chunks.is_empty(), "Should process chunks");
           
           let chunk = &processed_chunks[0];
           assert!(chunk.enriched.is_some(), "Should have enriched metadata");
           assert!(chunk.validation.is_some(), "Should have validation result");
           assert!(chunk.indexed, "Should be indexed");
           
           Ok(())
       }
       
       #[test]
       fn test_directory_processing() -> Result<()> {
           let (temp_dir, mut pipeline) = create_test_pipeline()?;
           create_test_files(&temp_dir)?;
           
           let stats = pipeline.process_directory(temp_dir.path())?;
           
           assert!(stats.files_processed > 0, "Should process files");
           assert!(stats.chunks_created > 0, "Should create chunks");
           assert!(stats.chunks_indexed > 0, "Should index some chunks");
           
           // Some chunks should be rejected due to quality issues
           if stats.chunks_validated > 0 {
               assert!(stats.validation_pass_rate < 1.0, "Should reject some poor quality chunks");
           }
           
           Ok(())
       }
       
       #[test]
       fn test_pipeline_with_validation_disabled() -> Result<()> {
           let (temp_dir, mut pipeline) = create_test_pipeline()?;
           pipeline.config.enable_validation = false;
           
           let test_file = temp_dir.path().join("poor.rs");
           fs::write(&test_file, "fn empty() {}")?; // Poor quality code
           
           let processed_chunks = pipeline.process_file(&test_file)?;
           
           assert!(!processed_chunks.is_empty(), "Should process chunks");
           assert!(processed_chunks[0].indexed, "Should index even poor quality chunks when validation disabled");
           assert!(processed_chunks[0].validation.is_none(), "Should not have validation result");
           
           Ok(())
       }
       
       #[test]
       fn test_pipeline_with_enrichment_disabled() -> Result<()> {
           let (temp_dir, mut pipeline) = create_test_pipeline()?;
           pipeline.config.enable_metadata_enrichment = false;
           
           let test_file = temp_dir.path().join("test.rs");
           fs::write(&test_file, "pub fn test() { println!(\"test\"); }")?;
           
           let processed_chunks = pipeline.process_file(&test_file)?;
           
           assert!(!processed_chunks.is_empty(), "Should process chunks");
           assert!(processed_chunks[0].enriched.is_none(), "Should not have enriched metadata");
           
           Ok(())
       }
       
       #[test]
       fn test_pipeline_statistics_tracking() -> Result<()> {
           let (temp_dir, mut pipeline) = create_test_pipeline()?;
           create_test_files(&temp_dir)?;
           
           // Process directory and check stats
           let stats = pipeline.process_directory(temp_dir.path())?;
           
           assert!(stats.total_processing_time > Duration::new(0, 0), "Should track processing time");
           assert!(stats.average_chunks_per_file > 0.0, "Should calculate average chunks per file");
           
           if stats.chunks_validated > 0 {
               assert!(stats.validation_pass_rate >= 0.0 && stats.validation_pass_rate <= 1.0, 
                      "Validation pass rate should be between 0 and 1");
           }
           
           if stats.chunks_created > 0 {
               assert!(stats.enrichment_success_rate >= 0.0 && stats.enrichment_success_rate <= 1.0,
                      "Enrichment success rate should be between 0 and 1");
           }
           
           Ok(())
       }
       
       #[test]
       fn test_pipeline_configuration_presets() -> Result<()> {
           let (temp_dir, mut pipeline) = create_test_pipeline()?;
           
           // Test speed configuration
           pipeline.configure_for_speed();
           assert!(!pipeline.config.enable_metadata_enrichment, "Speed config should disable enrichment");
           assert!(!pipeline.config.enable_validation, "Speed config should disable validation");
           
           // Test quality configuration
           pipeline.configure_for_quality();
           assert!(pipeline.config.enable_metadata_enrichment, "Quality config should enable enrichment");
           assert!(pipeline.config.enable_validation, "Quality config should enable validation");
           assert!(pipeline.config.validation_config.require_documentation, "Quality config should require docs");
           
           // Test development configuration
           pipeline.configure_for_development();
           assert!(pipeline.config.log_rejected_chunks, "Development config should enable logging");
           assert!(pipeline.config.enable_incremental, "Development config should enable incremental");
           
           Ok(())
       }
       
       #[test]
       fn test_error_handling_in_pipeline() -> Result<()> {
           let (temp_dir, mut pipeline) = create_test_pipeline()?;
           
           // Create a file that will cause parsing issues
           let problem_file = temp_dir.path().join("problem.rs");
           fs::write(&problem_file, "invalid rust syntax {{{{")?;
           
           // Pipeline should handle the error gracefully
           let result = pipeline.process_file(&problem_file);
           
           // Depending on implementation, this might succeed with empty chunks or fail gracefully
           match result {
               Ok(chunks) => {
                   // If it succeeds, chunks might be empty or contain text-only chunks
                   println!("Processed {} chunks from problematic file", chunks.len());
               }
               Err(_) => {
                   // If it fails, that's also acceptable for this test
                   let stats = pipeline.get_stats();
                   assert!(stats.files_failed > 0, "Should track failed files");
               }
           }
           
           Ok(())
       }
       
       #[test]
       fn test_stats_reset() -> Result<()> {
           let (temp_dir, mut pipeline) = create_test_pipeline()?;
           create_test_files(&temp_dir)?;
           
           // Process some files
           let _stats = pipeline.process_directory(temp_dir.path())?;
           let initial_stats = pipeline.get_stats();
           assert!(initial_stats.files_processed > 0, "Should have processed files");
           
           // Reset stats
           pipeline.reset_stats();
           let reset_stats = pipeline.get_stats();
           assert_eq!(reset_stats.files_processed, 0, "Stats should be reset");
           assert_eq!(reset_stats.chunks_created, 0, "Stats should be reset");
           
           Ok(())
       }
   }
   ```

## Success Criteria
- [ ] Pipeline integration compiles without errors
- [ ] All pipeline integration tests pass with `cargo test pipeline_integration_tests`
- [ ] Single file processing works through complete pipeline
- [ ] Directory processing handles multiple files correctly
- [ ] Statistics tracking provides accurate metrics
- [ ] Configuration presets work as expected
- [ ] Error handling prevents pipeline crashes
- [ ] Validation and enrichment can be enabled/disabled
- [ ] Performance monitoring tracks processing times
- [ ] Memory usage remains reasonable during processing

## Common Pitfalls to Avoid
- Don't fail entire pipeline for single chunk failures
- Handle memory usage carefully with large numbers of chunks
- Ensure atomic operations for statistics in multi-threaded scenarios
- Don't block pipeline for logging or non-critical operations
- Handle file system errors gracefully (permissions, missing files)
- Ensure proper cleanup of resources after processing
- Validate configuration before starting processing

## Context for Next Task
Task 16 will begin implementing the search engine components, starting with query parsing and preparation for full-text search across the indexed chunks.