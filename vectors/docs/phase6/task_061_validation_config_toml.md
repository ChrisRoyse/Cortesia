# Task 061: Create Validation Configuration File

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. The validation system needs a configuration file to define test parameters, thresholds, and system settings.

## Project Structure
```
validation_config.toml  <- Create this file (in project root)
```

## Task Description
Create a comprehensive TOML configuration file that defines all validation parameters, performance thresholds, test settings, and system configuration for the validation suite.

## Requirements
1. Create `validation_config.toml` in project root
2. Define performance thresholds and targets
3. Configure test data generation parameters
4. Set up validation criteria and tolerances
5. Include Windows-specific settings

## Expected File Content
```toml
[validation]
title = "LLMKG Vector Indexing System Validation Configuration"
version = "1.0.0"
target_accuracy = 100.0  # Required accuracy percentage

[test_data]
output_directory = "./test_data"
cleanup_after_tests = true
generate_synthetic_files = 100
large_file_size_mb = 10
unicode_test_enabled = true

[test_data.file_types]
special_characters = true
boolean_logic = true
proximity_search = true
wildcard_patterns = true
regex_patterns = true
vector_similarity = true
hybrid_search = true

[performance.targets]
# Latency targets (milliseconds)
p50_latency_ms = 50
p95_latency_ms = 100
p99_latency_ms = 200
max_acceptable_latency_ms = 1000

# Throughput targets
min_queries_per_second = 100
target_queries_per_second = 500
concurrent_users = 100

# Resource targets
max_memory_usage_mb = 1024
max_cpu_usage_percent = 80
index_rate_files_per_minute = 1000

[performance.benchmarks]
latency_test_iterations = 100
throughput_test_duration_secs = 60
warmup_queries = ["pub", "struct", "impl", "fn", "let"]
stress_test_duration_secs = 300
concurrent_test_users = [1, 10, 50, 100]

[correctness]
# Accuracy requirements
require_100_percent_accuracy = true
allow_false_positives = false
allow_false_negatives = false

# Content validation
validate_must_contain = true
validate_must_not_contain = true
case_sensitive_content = false

# Query type validation weights
[correctness.query_types]
special_characters_weight = 1.0
boolean_and_weight = 1.0
boolean_or_weight = 1.0
boolean_not_weight = 1.0
proximity_weight = 1.0
wildcard_weight = 1.0
regex_weight = 1.0
phrase_weight = 1.0
vector_weight = 1.0
hybrid_weight = 2.0  # Hybrid queries are most important

[ground_truth]
dataset_file = "ground_truth.json"
auto_generate_missing = true
validate_expected_files = true
require_content_validation = true

# Test case categories and minimum counts
[ground_truth.minimum_test_cases]
special_characters = 25
boolean_and = 20
boolean_or = 20
boolean_not = 15
proximity = 15
wildcard = 15
phrase = 10
regex = 10
vector = 20
hybrid = 25

[system]
# Index paths
text_index_path = "./validation_text_index"
vector_db_path = "./validation_vector.lance"

# System resources
max_indexing_threads = 0  # 0 = use all available cores
max_search_threads = 0    # 0 = use all available cores
memory_limit_mb = 2048

# Windows-specific settings
[system.windows]
handle_long_paths = true
normalize_path_separators = true
handle_unicode_filenames = true
respect_file_attributes = true

[logging]
level = "info"
enable_performance_logging = true
enable_validation_logging = true
log_failed_queries = true
log_file = "validation.log"

[reporting]
output_directory = "./validation_reports"
generate_markdown_report = true
generate_json_report = true
include_detailed_metrics = true
include_error_analysis = true

# Report sections to include
[reporting.sections]
executive_summary = true
accuracy_metrics = true
performance_metrics = true
stress_test_results = true
security_audit = true
recommendations = true

[security]
# Security validation settings
test_malicious_queries = true
test_injection_attacks = true
test_path_traversal = true
test_buffer_overflow = false  # Not applicable to Rust, but test anyway
max_query_length = 10000
sanitize_log_output = true

[experimental]
# Experimental features for testing
enable_ai_generated_queries = false
enable_mutation_testing = false
enable_property_based_testing = false
parallel_validation = true
```

## Additional Configuration Notes
Create a comment block at the top explaining key configuration options:

```toml
# LLMKG Vector Indexing System - Validation Configuration
# 
# This file configures all aspects of the validation system including:
# - Performance targets and thresholds
# - Test data generation parameters  
# - Correctness validation criteria
# - Ground truth dataset requirements
# - System resource limits
# - Reporting and logging options
#
# Performance Targets:
# - Latency: P50 < 50ms, P95 < 100ms, P99 < 200ms
# - Throughput: > 100 QPS sustained
# - Memory: < 1GB for 100K documents
# - Accuracy: 100% on all query types
#
# Modify these values based on your hardware and requirements.
```

## Success Criteria
- TOML file is valid and parseable
- All validation parameters are covered
- Performance thresholds are realistic and measurable
- Windows-specific settings are included
- Configuration supports all test types mentioned in the validation document
- File is well-documented with comments

## Time Limit
10 minutes maximum