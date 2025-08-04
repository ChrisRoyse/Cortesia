# Micro-Task 011: Setup Environment Variables

## Objective
Configure environment variables for development and testing consistency.

## Context
Environment variables provide configuration for different deployment environments and enable consistent behavior across development machines. This is particularly important for Windows path handling and test data locations.

## Prerequisites
- Task 010 completed (Windows compatibility validated)
- Command prompt or PowerShell access

## Time Estimate
7 minutes

## Instructions
1. Create `.env.example` file in project root:
   ```
   # Vector Search System Configuration
   LLMKG_DATA_DIR=./data
   LLMKG_INDEX_DIR=./data/indices
   LLMKG_LOG_LEVEL=info
   LLMKG_MAX_MEMORY_MB=1024
   LLMKG_PARALLEL_WORKERS=4
   ```
2. Create `.env` file by copying example: `copy .env.example .env`
3. Add `.env` to `.gitignore` (append): `echo .env >> .gitignore`
4. Test environment variable loading by creating simple test:
   ```rust
   // test_env.rs
   fn main() {
       println!("Data dir: {:?}", std::env::var("LLMKG_DATA_DIR"));
   }
   ```
5. Compile and test: `rustc test_env.rs && test_env.exe`
6. Clean up: `del test_env.exe test_env.rs`
7. Commit: `git add .env.example .gitignore && git commit -m "Add environment configuration"`

## Expected Output
- Environment configuration template created
- Local environment file ready for customization
- Environment variable access tested
- Configuration committed to Git

## Success Criteria
- [ ] `.env.example` file created with all required variables
- [ ] `.env` file created and working locally
- [ ] `.env` added to `.gitignore` to prevent committing secrets
- [ ] Environment variable loading tested successfully
- [ ] Changes committed to Git

## Next Task
task_012_create_data_directories.md