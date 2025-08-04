# Micro-Task 015: Setup Logging Configuration

## Objective
Configure structured logging for the vector search system development and testing.

## Context
Proper logging is essential for debugging performance issues and understanding system behavior. This task creates a consistent logging configuration that will be used across all crates.

## Prerequisites  
- Task 014 completed (Development tools installed)
- Environment variables configured
- Data directories created

## Time Estimate
8 minutes

## Instructions
1. Create `logging.toml` in project root:
   ```toml
   [appenders.stdout]
   kind = "console"
   
   [appenders.file]
   kind = "file"
   path = "data/logs/vector_search.log"
   append = true
   
   [loggers.vector_search]
   level = "info"
   appenders = ["stdout", "file"]
   additive = false
   
   [root]
   level = "warn"
   appenders = ["stdout"]
   ```
2. Create log configuration test file `test_logging.rs`:
   ```rust
   fn main() {
       println!("Testing log directory structure...");
       
       let log_dir = std::path::Path::new("data/logs");
       if log_dir.exists() {
           println!("✓ Log directory exists");
       } else {
           println!("✗ Log directory missing");
       }
   }
   ```
3. Compile and run test: `rustc test_logging.rs && test_logging.exe`
4. Clean up test: `del test_logging.exe test_logging.rs`
5. Commit configuration: `git add logging.toml && git commit -m "Add logging configuration"`

## Expected Output
- Logging configuration file created
- Log directory structure validated
- Configuration committed to version control

## Success Criteria
- [ ] `logging.toml` created with console and file appenders
- [ ] Log directory validation test passes
- [ ] Test artifacts cleaned up properly
- [ ] Configuration committed to Git

## Next Task
task_016_create_makefile_scripts.md