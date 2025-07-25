# LLMKG (LLM Knowledge Graph) - Phase 1 Foundation Fixes

## Project Overview
A sophisticated Large Language Model Knowledge Graph system implementing cognitive-enhanced AI processing with federation support, neural processing, and advanced storage optimization. This project builds upon extensive existing infrastructure including CognitiveOrchestrator, FederationCoordinator, NeuralProcessingServer, and 28 MCP tools.

## Phase 1 Mission Statement
**Goal**: Fix critical foundation issues preventing basic functionality while establishing cognitive-enhanced event store foundation with full MCP tool integration.

**Duration**: 4 weeks  
**Priority**: CRITICAL  
**Target Performance**: <10ms per operation on Intel i9 processor with cognitive orchestration

## Core Objectives

### Primary Objectives (Phase 1)
1. **Cognitive-Enhanced Entity Extraction**: Implement AI-powered entity recognition with neural server integration, attention management, and working memory
2. **Neural-Federation Relationship Extraction**: Build relationship extraction with cross-database coordination and cognitive pattern selection  
3. **Orchestrated Question Answering**: Develop cognitive reasoning-guided question parsing and answer generation
4. **Enhanced MCP Tool Integration**: Upgrade 28 existing MCP tools with cognitive metadata and federation support
5. **Cognitive-Federation Migration**: Complete data migration with neural embeddings and attention weight preservation

### Success Criteria (Measurable)
- **Entity Extraction**: >95% accuracy with cognitive orchestration and neural processing
- **Relationship Extraction**: 30+ relationship types, 90% accuracy via federation
- **Question Answering**: >90% relevance with cognitive reasoning patterns
- **Performance Targets**:
  - Entity extraction: <8ms per sentence with neural processing
  - Relationship extraction: <12ms per sentence with federation
  - Question answering: <20ms total with cognitive reasoning
  - Federation storage: <3ms with cross-database coordination
- **Test Coverage**: >90% including cognitive integration and federation tests
- **Migration**: Complete with cognitive metadata and working memory preservation

### Technical Infrastructure (Existing)
**Discovered Advanced Systems**:
- **CognitiveOrchestrator**: Complete cognitive pattern orchestration with 7 thinking patterns
- **Neural Processing Server**: Advanced neural model execution with training/prediction capabilities  
- **Federation Coordinator**: Cross-database transaction management with 2-phase commit
- **28 MCP Tools**: Comprehensive knowledge storage, neural processing, cognitive reasoning suite
- **15+ Pre-trained Models**: DistilBERT, TinyBERT, T5, MiniLM, dependency parsers, classifiers
- **Advanced Storage**: Zero-copy MMAP, string interning, HNSW indexing, product quantization
- **Comprehensive Monitoring**: ObservabilityEngine, BrainMetricsCollector, PerformanceMonitor
- **Working Memory Systems**: AttentionManager, CompetitiveInhibitionSystem, HebbianLearning

### Process for Task Execution

#### Step 1: Task Analysis

*   **Understand the Task**: Analyze the user's request to identify all requirements, constraints, and edge cases.
*   **Clarify if Needed**: If any part of the request is ambiguous, ask specific questions to ensure clarity (e.g., "Do you prefer a specific framework?" or "Should this handle [specific edge case]?").
*   **Define Success Criteria**: Outline measurable criteria for task completion (e.g., functionality, performance, code readability).

#### Step 2: Parallel Subagent Delegation

*   **Break Down Tasks**: Decompose the task into independent, atomic subtasks to enable parallel processing.
*   **Assign Subagents**: Spawn subagents to handle each subtask in parallel, ensuring no overlap or loss of context.
*   **Subagent Instructions**:
    *   Provide each subagent with:
        *   **Task Description**: Specific issue or feature to implement.
        *   **Context**: Relevant code, requirements, or dependencies.
        *   **Success Criteria**: Exact expected outcome (e.g., "Function returns correct output for inputs X, Y, Z").
        *   **Example**: Before/after code snippets or expected behavior (e.g., "Input: `vec![1, 2]`, Output: `3`").

***

**Example Subagent Prompt:**
**Subagent Task**: Implement a function to validate email format.
**Context**: Part of a user registration system in Rust.
**Success Criteria**: Return `true` for valid emails (e.g., "user@domain.com"), `false` for invalid (e.g., "user@domain", "user.domain.com").
**Example**:
*   **Before**: No validation exists.
*   **After**:
    ```rust
    use regex::Regex;

    fn is_valid_email(email: &str) -> bool {
        let email_regex = Regex::new(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$").unwrap();
        email_regex.is_match(email)
    }
    ```

***

*   **Isolation**: Ensure each subagent operates on a specific, isolated scope to avoid conflicts.

#### Step 3: Implementation

*   **Write Code**: Each subagent implements its assigned subtask, adhering to best practices for the language/framework.
*   **Standards**:
    *   Use clear, descriptive variable/function names.
    *   Include comments for complex logic.
    *   Handle all edge cases specified in the success criteria.
    *   Optimize for performance where applicable.
*   **Output Format**: Wrap all code in a single `<xaiArtifact>` tag with a unique UUID, appropriate title, and correct contentType (e.g., `text/rust`, `text/html`).

#### Step 4: Quality Assurance & Iterative Improvement

*   **Self-Assessment**:
    *   After completing a task or subtask, evaluate the work against the original requirements.
    *   Rate the quality on a 1-100 scale based on:
        *   **Functionality**: Does it meet all requirements and handle edge cases?
        *   **Code Quality**: Is it readable, maintainable, and following best practices?
        *   **Performance**: Is it optimized for the use case?
        *   **User Intent**: Does it fully align with the user's expectations?
    *   Document the score and rationale (e.g., "Score: 90/100. Missed edge case for empty input").
*   **Identify Gaps**: List any bugs, missing features, or deviations from requirements.
*   **Iterate if Score < 100**:
    *   For each identified issue, spawn a new subagent to fix it.
        *   **Task**: Specific issue to resolve.
        *   **Context**: Relevant code and original requirements.
        *   **Success Criteria**: Fix closes the gap without introducing regressions.

***

**Example Fix Prompt:**
**Subagent Task**: Fix missing edge case for empty input in `is_valid_email`.
**Context**: Current function: `fn is_valid_email(email: &str) -> bool { ... }`
**Success Criteria**: Return `false` for an empty string.
**Example**:
*   **Before**: `is_valid_email("")` returns `false` (by regex luck), but it's not explicit.
*   **After**:
    ```rust
    use regex::Regex;

    fn is_valid_email(email: &str) -> bool {
        if email.is_empty() {
            return false;
        }
        let email_regex = Regex::new(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$").unwrap();
        email_regex.is_match(email)
    }
    ```

***

*   **Verification Loop**:
    *   Spawn a verification subagent to validate each fix.
    *   Check for regressions or new issues.
    *   Update the quality score after verification.
*   **Stop and Iterate**: Do not proceed to the next task until the current task scores 100/100.

#### Step 5: Final Delivery

*   **Consolidate Output**: Combine all subtask outputs into a single, cohesive artifact.
*   **Document Changes**: Include a brief summary of iterations and improvements made (e.g., "Fixed edge case for empty input, optimized loop performance").
*   **Present to User**: Deliver the final artifact with the quality score and confirmation of 100/100 alignment with intent.

### Example Workflow

**User Request**: "Create a Rust function to sort a list of integers in ascending order, handling duplicates and negative numbers."

**Analysis**:

*   **Requirements**: Sort integers, handle duplicates, negative numbers, return sorted list.
*   **Edge cases**: Empty list, single element, all duplicates, mixed positive/negative.
*   **Success Criteria**: Correctly sorted `Vec<i32>`, O(n log n) time complexity, readable idiomatic Rust.

**Delegation**:

*   **Subagent 1**: Implement core sorting logic.
*   **Subagent 2**: Handle edge cases (empty `Vec`).
*   **Subagent 3**: Optimize for performance and readability.

**Implementation**:

*   **Subagent 1 produces**:
    ```rust
    fn sort_integers(numbers: &mut Vec<i32>) {
        numbers.sort_unstable();
    }
    ```

*   **Subagent 2 improves handling (no real change needed as `sort` handles empty `Vec` gracefully, but showing iteration):**
    ```rust
    fn sort_integers(numbers: &mut Vec<i32>) {
        if numbers.is_empty() {
            return; // Explicitly handle empty case
        }
        numbers.sort_unstable();
    }
    ```

**Quality Assurance**:

*   **Initial Score**: 95/100 (The function modifies the input `Vec` in place, but the user might have expected a new `Vec` to be returned. The name implies sorting, but ownership is a key detail).
*   **Subagent 4 fixes by returning a new, sorted `Vec` to better align with functional expectations:**
    ```rust
    fn sort_integers(numbers: &[i32]) -> Vec<i32> {
        let mut sorted_numbers = numbers.to_vec();
        sorted_numbers.sort_unstable();
        sorted_numbers
    }
    ```
*   **Verification Subagent** confirms the fix, no regressions.
*   **Final Score**: 100/100.

**Delivery**:

*   Artifact with final code, score, and summary of iterations.

### Constraints

*   **Context Preservation**: Maintain full context across all subagents and iterations.
*   **No Artifact Tag Mentions**: Never reference `<xaiArtifact>` outside the tag itself.
*   **UUID Usage**: Use the same UUID for updated artifacts, new UUIDs for unrelated artifacts.
*   **Language-Specific Guidelines**:
    *   For Python/Pygame: Follow Pyodide compatibility guidelines.
    *   For React/JSX: Use CDN-hosted React, Tailwind CSS, and JSX syntax.
    *   For LaTeX: Use PDFLaTeX, `texlive-full` packages, and correct font configurations.

### Final Notes

*   Iterate relentlessly until every task achieves a 100/100 score.
*   Ensure all subagents communicate clearly and maintain isolation to avoid conflicts.
*   Deliver a single, polished artifact that fully satisfies the user's intent.