# Comprehensive Testing Prompt for LLMKG Cognitive Algorithms

## Overview

You are tasked with creating a comprehensive test suite for the LLMKG (Lightning-Fast Knowledge Graph) system, specifically focusing on the 7 advanced cognitive algorithms. This test suite will consist of 56 tests total: 28 tests with mocked LLM calls and 28 tests using the DeepSeek API for real LLM interactions.

## System Architecture Summary

LLMKG is a high-performance knowledge graph system with brain-inspired cognitive capabilities. Key components include:

- **Core Graph**: Slotmap-based entity storage with relationships supporting directed/undirected/weighted edges
- **Storage Layers**: Multiple backends including memory-mapped files, CSR format, and zero-copy operations
- **Cognitive Module**: 7 distinct algorithms for different types of reasoning
- **Learning Systems**: Hebbian learning, homeostasis, and adaptive optimization
- **Federation**: Multi-database support with temporal versioning
- **Neural Integration**: WASM-based neural computations with SIMD optimization

## The 7 Cognitive Algorithms Requiring Testing

### 1. **Convergent Thinking** (`src/cognitive/convergent.rs`)
- **Purpose**: Find single best answer through focused search
- **Key Methods**: `execute_convergent_query()`, `focused_propagation()`, `extract_best_answer()`
- **Parameters**: `activation_threshold` (0.0-1.0), `max_depth`, `beam_width`
- **Test Focus**: Beam search accuracy, activation propagation, answer quality

### 2. **Divergent Thinking** (`src/cognitive/divergent.rs`)
- **Purpose**: Creative exploration and brainstorming
- **Key Methods**: `execute_divergent_exploration()`, `spread_activation()`, `rank_by_creativity()`
- **Parameters**: `exploration_breadth`, `creativity_threshold`, `novelty_weight`
- **Test Focus**: Exploration coverage, creativity scoring, novel connection discovery

### 3. **Critical Thinking** (`src/cognitive/critical.rs`)
- **Purpose**: Contradiction detection and resolution
- **Key Methods**: `execute_critical_analysis()`, `identify_contradictions()`, `validate_information_sources()`
- **Parameters**: `validation_threshold`, `ValidationLevel` enum
- **Test Focus**: Contradiction detection accuracy, resolution strategies, source validation

### 4. **Systems Thinking** (`src/cognitive/systems.rs`)
- **Purpose**: Hierarchical reasoning and inheritance
- **Key Methods**: `execute_hierarchical_reasoning()`, `traverse_hierarchy()`, `apply_inheritance_rules()`
- **Parameters**: `max_inheritance_depth`
- **Test Focus**: Hierarchy traversal, inheritance correctness, exception handling

### 5. **Lateral Thinking** (`src/cognitive/lateral.rs`)
- **Purpose**: Find unexpected connections between concepts
- **Key Methods**: `find_creative_connections()`, `neural_bridge_search()`
- **Parameters**: `novelty_threshold`, `max_bridge_length`, `creativity_boost`
- **Test Focus**: Bridge discovery, path uniqueness, cross-domain connections

### 6. **Abstract Thinking** (`src/cognitive/abstract_pattern.rs`)
- **Purpose**: Pattern detection and meta-analysis
- **Key Methods**: `execute_pattern_analysis()`, `identify_abstractions()`, `suggest_refactoring()`
- **Pattern Types**: Structural, Temporal, Semantic, Usage
- **Test Focus**: Pattern recognition accuracy, abstraction quality, refactoring suggestions

### 7. **Adaptive Thinking** (`src/cognitive/adaptive.rs`)
- **Purpose**: Dynamic strategy selection based on query type
- **Key Methods**: `execute_adaptive_reasoning()`, `select_cognitive_strategies()`, `merge_pattern_results()`
- **Test Focus**: Strategy selection logic, ensemble coordination, learning from feedback

## Test Requirements

### Data Scale Requirements
- **Nodes**: 300,000+ entities representing real-world concepts
- **Relationships**: 500,000+ connections with varied types and weights
- **Databases**: Multiple layered databases with federation
- **Temporal Data**: Time-series data for temporal calculations

### Test Categories

#### A. Mock LLM Tests (28 tests - 4 per algorithm)
Each algorithm needs 4 rigorous tests that:
1. **Basic Functionality Test**: Verify core algorithm behavior with simple inputs
2. **Edge Case Test**: Push algorithm limits with extreme parameters
3. **Performance Stress Test**: Handle maximum scale (300k+ nodes)
4. **Integration Test**: Verify interaction with other system components

#### B. DeepSeek API Tests (28 tests - 4 per algorithm)
Each algorithm needs 4 tests that:
1. **Real-World Query Test**: Natural language queries processed by DeepSeek
2. **Complex Reasoning Test**: Multi-step reasoning requiring LLM understanding
3. **Context Integration Test**: Combine LLM responses with graph knowledge
4. **Adaptive Learning Test**: Improve performance based on LLM feedback

## Data Generation Strategy

### 1. Domain-Specific Data Sets

#### Scientific Knowledge Domain (50k nodes, 100k relationships)
- Hierarchical taxonomy: Biology → Animals → Mammals → Primates → Humans
- Cross-references: Genetic relationships, evolutionary paths
- Temporal data: Discovery dates, evolution timelines
- Contradictions: Competing theories, disputed classifications

#### Technology Domain (50k nodes, 100k relationships)
- Programming languages, frameworks, libraries
- Dependency graphs, version histories
- Performance benchmarks, compatibility matrices
- Design patterns and anti-patterns

#### Cultural Knowledge Domain (50k nodes, 100k relationships)
- Art movements, literary genres, music styles
- Influence networks, artistic lineages
- Geographic distributions, temporal progressions
- Cross-cultural connections and bridges

#### Business & Economics Domain (50k nodes, 100k relationships)
- Company hierarchies, market sectors
- Financial relationships, supply chains
- Temporal market data, trend analysis
- Competitive landscapes, merger histories

#### Geographic & Political Domain (50k nodes, 100k relationships)
- Country/city hierarchies, political boundaries
- Trade relationships, diplomatic ties
- Historical changes, border evolution
- Resource distributions, demographic data

#### Medical & Health Domain (50k nodes, 100k relationships)
- Disease taxonomies, symptom networks
- Drug interactions, treatment protocols
- Research citations, clinical trials
- Temporal progression, patient pathways

### 2. Relationship Types
- **Hierarchical**: is-a, part-of, belongs-to
- **Associative**: related-to, similar-to, co-occurs-with
- **Causal**: causes, prevents, influences
- **Temporal**: before, after, during, evolved-from
- **Contradictory**: disputes, contradicts, challenges
- **Weighted**: strength values 0.0-1.0 based on confidence/relevance

### 3. Multi-Layer Database Structure
- **Layer 1**: Core ontology (stable, rarely changing)
- **Layer 2**: Instance data (frequent updates)
- **Layer 3**: Temporal snapshots (historical versions)
- **Layer 4**: User annotations (collaborative knowledge)
- **Layer 5**: Learned patterns (ML-derived relationships)

## Mock LLM Test Specifications

### Test Structure Template
```rust
#[tokio::test]
async fn test_{algorithm}_{test_type}_mock() {
    // Setup
    let mock_llm = MockLLMClient::new();
    let graph = create_test_graph_with_data().await;
    let cognitive_system = Phase4CognitiveSystem::new(graph.clone());
    
    // Configure mock responses
    mock_llm.expect_call()
        .with_query("expected query pattern")
        .returns("mocked LLM response")
        .times(1);
    
    // Execute algorithm
    let result = cognitive_system.{algorithm}_query(
        query,
        parameters,
        Some(mock_llm)
    ).await;
    
    // Assertions
    assert!(result.is_ok());
    assert_performance_metrics(&result);
    assert_correctness(&result);
}
```

### Convergent Thinking Mock Tests (4 tests)

1. **Basic Convergent Search Mock**
   - Query: "What is the capital of France?"
   - Mock LLM: Returns structured extraction hints
   - Verify: Single correct answer (Paris) with high confidence

2. **Deep Hierarchy Convergent Mock**
   - Query: "What enzyme breaks down lactose?"
   - Graph depth: 10+ levels of biochemical hierarchy
   - Mock LLM: Provides domain-specific guidance
   - Verify: Correct traversal through enzyme categories

3. **Performance Convergent Mock**
   - Query: "Find the CEO of the largest tech company"
   - Data: Full 300k node graph
   - Mock LLM: Returns query optimization hints
   - Verify: < 100ms response time, correct result

4. **Ambiguous Resolution Mock**
   - Query: "What is the fastest animal?"
   - Ambiguity: Land vs air vs water
   - Mock LLM: Helps disambiguate context
   - Verify: Correct contextual answer selection

### Divergent Thinking Mock Tests (4 tests)

1. **Creative Exploration Mock**
   - Query: "What can I make with eggs?"
   - Mock LLM: Suggests creative categories
   - Verify: 20+ diverse suggestions across cooking, art, science

2. **Cross-Domain Divergent Mock**
   - Query: "How is music like mathematics?"
   - Mock LLM: Provides abstraction hints
   - Verify: Unexpected connections (frequencies, patterns, harmony)

3. **Maximum Breadth Mock**
   - Query: "Applications of artificial intelligence"
   - Exploration breadth: 1000+
   - Mock LLM: Categorizes applications
   - Verify: Coverage across all major domains

4. **Novel Combination Mock**
   - Query: "Combine features of dolphins and helicopters"
   - Mock LLM: Suggests combination strategies
   - Verify: Creative yet logical combinations

### Critical Thinking Mock Tests (4 tests)

1. **Basic Contradiction Mock**
   - Setup: "Coffee is healthy" vs "Coffee is unhealthy"
   - Mock LLM: Provides resolution framework
   - Verify: Contextual resolution with evidence

2. **Source Validation Mock**
   - Query: "Is this medical claim valid?"
   - Sources: Mix of peer-reviewed and blogs
   - Mock LLM: Rates source credibility
   - Verify: Correct source ranking and filtering

3. **Complex Dispute Mock**
   - Topic: "Climate change causes"
   - Contradictions: 10+ competing theories
   - Mock LLM: Identifies consensus vs outliers
   - Verify: Weighted resolution based on evidence

4. **Temporal Contradiction Mock**
   - Facts: Historical data with updates
   - Mock LLM: Handles temporal context
   - Verify: Correct handling of fact evolution

### Systems Thinking Mock Tests (4 tests)

1. **Basic Inheritance Mock**
   - Query: "Properties of golden retrievers"
   - Hierarchy: Dog → Mammal → Animal
   - Mock LLM: Confirms inheritance rules
   - Verify: All inherited properties included

2. **Exception Handling Mock**
   - Query: "Can penguins fly?"
   - Exception: Flightless birds
   - Mock LLM: Identifies exceptions
   - Verify: Correct exception override

3. **Deep Hierarchy Mock**
   - Query: "Corporate structure of subsidiary X"
   - Depth: 15+ organizational levels
   - Mock LLM: Navigates complex structures
   - Verify: Complete hierarchy traversal

4. **Multiple Inheritance Mock**
   - Entity: "Smartphone" (Computer + Phone)
   - Mock LLM: Resolves inheritance conflicts
   - Verify: Correct property precedence

### Lateral Thinking Mock Tests (4 tests)

1. **Simple Bridge Mock**
   - Connect: "Einstein" to "Rock Music"
   - Mock LLM: Suggests bridge concepts
   - Verify: Valid unexpected path found

2. **Multi-Hop Lateral Mock**
   - Connect: "Quantum Physics" to "Cooking"
   - Max bridges: 5
   - Mock LLM: Provides creative links
   - Verify: Semantically valid connections

3. **Cross-Culture Bridge Mock**
   - Connect: "Japanese Tea Ceremony" to "Computer Programming"
   - Mock LLM: Identifies cultural patterns
   - Verify: Meaningful cultural bridges

4. **Invention Inspiration Mock**
   - Query: "Nature-inspired solutions for urban transport"
   - Mock LLM: Suggests biomimicry connections
   - Verify: Novel yet practical connections

### Abstract Thinking Mock Tests (4 tests)

1. **Pattern Detection Mock**
   - Data: Sales patterns across industries
   - Mock LLM: Identifies pattern types
   - Verify: Correct pattern categorization

2. **Structural Abstraction Mock**
   - Analyze: Software architecture patterns
   - Mock LLM: Suggests abstractions
   - Verify: Valid refactoring recommendations

3. **Temporal Pattern Mock**
   - Data: 10-year market cycles
   - Mock LLM: Predicts future patterns
   - Verify: Pattern projection accuracy

4. **Meta-Pattern Mock**
   - Query: "Patterns in how patterns emerge"
   - Mock LLM: Provides meta-analysis
   - Verify: Higher-order pattern recognition

### Adaptive Thinking Mock Tests (4 tests)

1. **Strategy Selection Mock**
   - Queries: Mix of 10 different types
   - Mock LLM: Recommends strategies
   - Verify: Optimal algorithm selection

2. **Ensemble Coordination Mock**
   - Complex query requiring 3+ algorithms
   - Mock LLM: Coordinates approaches
   - Verify: Effective result merging

3. **Learning Adaptation Mock**
   - Run 100 queries with feedback
   - Mock LLM: Improves suggestions
   - Verify: Performance improvement curve

4. **Context Switching Mock**
   - Rapid context changes
   - Mock LLM: Adapts strategies
   - Verify: Smooth transitions

## DeepSeek API Test Specifications

### Configuration
```rust
const DEEPSEEK_API_KEY: &str = env::var("DEEPSEEK_API_KEY");
const DEEPSEEK_MODEL: &str = "deepseek-chat";
const MAX_TOKENS: u32 = 2000;
const TEMPERATURE: f32 = 0.7;
```

### Test Structure Template
```rust
#[tokio::test]
async fn test_{algorithm}_{test_type}_deepseek() {
    // Setup
    let deepseek_client = DeepSeekClient::new(DEEPSEEK_API_KEY);
    let graph = create_production_graph().await;
    let cognitive_system = Phase4CognitiveSystem::new(graph.clone());
    
    // Execute with real LLM
    let result = cognitive_system.{algorithm}_query_with_llm(
        natural_language_query,
        parameters,
        deepseek_client
    ).await;
    
    // Validate results
    assert!(result.is_ok());
    validate_llm_enhanced_results(&result);
}
```

### Convergent Thinking DeepSeek Tests (4 tests)

1. **Natural Language Fact Extraction**
   - Query: "What year did the first human land on the moon and who was it?"
   - DeepSeek: Extracts structured facts from response
   - Verify: Correct date (1969) and person (Neil Armstrong)

2. **Technical Specification Query**
   - Query: "What are the memory requirements for running Kubernetes?"
   - DeepSeek: Parses technical documentation
   - Verify: Accurate technical specifications

3. **Multi-Source Convergence**
   - Query: "What is the consensus on optimal sleep duration?"
   - DeepSeek: Synthesizes multiple sources
   - Verify: Evidence-based answer with citations

4. **Real-time Data Integration**
   - Query: "Current CEO of major tech companies"
   - DeepSeek: Provides up-to-date information
   - Verify: Accurate current information

### Divergent Thinking DeepSeek Tests (4 tests)

1. **Creative Business Ideas**
   - Query: "Innovative business ideas combining AI and agriculture"
   - DeepSeek: Generates creative suggestions
   - Verify: Minimum 15 unique, viable ideas

2. **Cross-Industry Innovation**
   - Query: "How can gaming technology improve healthcare?"
   - DeepSeek: Explores unexpected applications
   - Verify: Practical cross-domain innovations

3. **Future Scenario Planning**
   - Query: "Possible futures for urban transportation in 2050"
   - DeepSeek: Envisions multiple scenarios
   - Verify: Diverse, plausible futures

4. **Problem-Solving Alternatives**
   - Query: "Alternative solutions to plastic pollution"
   - DeepSeek: Generates comprehensive solutions
   - Verify: Technical, social, and economic approaches

### Critical Thinking DeepSeek Tests (4 tests)

1. **Fact-Checking Complex Claims**
   - Query: "Verify: 'Quantum computers will replace all classical computers by 2030'"
   - DeepSeek: Analyzes claim validity
   - Verify: Nuanced analysis with evidence

2. **Bias Detection**
   - Query: "Analyze potential biases in this news article about climate change"
   - DeepSeek: Identifies bias indicators
   - Verify: Comprehensive bias assessment

3. **Argument Evaluation**
   - Query: "Evaluate arguments for and against universal basic income"
   - DeepSeek: Weighs competing arguments
   - Verify: Balanced critical analysis

4. **Research Validation**
   - Query: "Assess the validity of this correlation between coffee and longevity"
   - DeepSeek: Evaluates research methodology
   - Verify: Scientific rigor assessment

### Systems Thinking DeepSeek Tests (4 tests)

1. **Organizational Analysis**
   - Query: "Explain the UN organizational structure and decision flow"
   - DeepSeek: Maps complex hierarchies
   - Verify: Accurate structural representation

2. **Ecosystem Interactions**
   - Query: "How do changes in bee populations affect agriculture?"
   - DeepSeek: Traces system-wide impacts
   - Verify: Complete causal chains

3. **Technology Stack Dependencies**
   - Query: "What depends on OpenSSL in a typical web application?"
   - DeepSeek: Identifies dependencies
   - Verify: Comprehensive dependency tree

4. **Economic System Effects**
   - Query: "Trace the effects of interest rate changes through the economy"
   - DeepSeek: Models systemic impacts
   - Verify: Multi-order effects captured

### Lateral Thinking DeepSeek Tests (4 tests)

1. **Innovation Through Analogy**
   - Query: "How can ant colony behavior improve network routing?"
   - DeepSeek: Finds biological inspirations
   - Verify: Practical technical applications

2. **Cross-Cultural Solutions**
   - Query: "What can Silicon Valley learn from Japanese manufacturing?"
   - DeepSeek: Bridges cultural practices
   - Verify: Actionable insights

3. **Historical Parallels**
   - Query: "Modern situations similar to the Library of Alexandria"
   - DeepSeek: Identifies historical patterns
   - Verify: Meaningful parallels

4. **Interdisciplinary Connections**
   - Query: "Connect principles of music theory to data visualization"
   - DeepSeek: Finds deep connections
   - Verify: Practical applications

### Abstract Thinking DeepSeek Tests (4 tests)

1. **Trend Meta-Analysis**
   - Query: "What patterns exist across all technology hype cycles?"
   - DeepSeek: Abstracts common patterns
   - Verify: Valid meta-patterns

2. **Conceptual Framework Generation**
   - Query: "Create a framework for evaluating AI ethics proposals"
   - DeepSeek: Builds abstract frameworks
   - Verify: Comprehensive, applicable framework

3. **Pattern Transfer**
   - Query: "Apply patterns from evolution to software development"
   - DeepSeek: Transfers patterns across domains
   - Verify: Valid pattern applications

4. **Philosophical Abstraction**
   - Query: "What do all successful communities have in common?"
   - DeepSeek: Identifies abstract principles
   - Verify: Universal principles

### Adaptive Thinking DeepSeek Tests (4 tests)

1. **Multi-Modal Query Processing**
   - Query: "Analyze this company: financial health, market position, and innovation"
   - DeepSeek: Coordinates multiple analyses
   - Verify: Integrated multi-faceted response

2. **Dynamic Strategy Adjustment**
   - Series: 10 queries with increasing complexity
   - DeepSeek: Adapts approach per query
   - Verify: Appropriate strategy evolution

3. **Contextual Intelligence**
   - Query: "What should I know about X?" (X varies: person/place/concept)
   - DeepSeek: Tailors response to entity type
   - Verify: Context-appropriate depth

4. **Learning From Interaction**
   - Interactive: 20 queries with feedback
   - DeepSeek: Improves based on feedback
   - Verify: Measurable improvement

## Test Data Requirements

### Synthetic Data Generation
```rust
pub struct TestDataGenerator {
    pub nodes_per_domain: usize,  // 50,000
    pub relationships_per_domain: usize,  // 100,000
    pub domains: Vec<KnowledgeDomain>,
    pub temporal_range: Duration,  // 10 years
    pub contradiction_rate: f32,  // 0.05 (5%)
}

impl TestDataGenerator {
    pub async fn generate_full_dataset(&self) -> TestDataset {
        // Generate 300k+ nodes across 6 domains
        // Create 500k+ relationships with varied types
        // Add temporal data for 10-year range
        // Inject contradictions for critical thinking
        // Create multi-layer database structure
    }
}
```

### Performance Requirements
- Query response time: < 100ms for focused queries
- Exploration breadth: Handle 1000+ activations
- Memory usage: < 4GB for full dataset
- Concurrency: Support 100+ parallel queries

## Deliverables

### 1. Test Suite Structure
```
tests/
├── phase1_cognitive_tests/
│   ├── mock_tests/
│   │   ├── convergent_mock_tests.rs
│   │   ├── divergent_mock_tests.rs
│   │   ├── critical_mock_tests.rs
│   │   ├── systems_mock_tests.rs
│   │   ├── lateral_mock_tests.rs
│   │   ├── abstract_mock_tests.rs
│   │   └── adaptive_mock_tests.rs
│   └── deepseek_tests/
│       ├── convergent_deepseek_tests.rs
│       ├── divergent_deepseek_tests.rs
│       ├── critical_deepseek_tests.rs
│       ├── systems_deepseek_tests.rs
│       ├── lateral_deepseek_tests.rs
│       ├── abstract_deepseek_tests.rs
│       └── adaptive_deepseek_tests.rs
├── test_data/
│   ├── synthetic_data_generator.rs
│   ├── mock_llm_client.rs
│   └── test_fixtures/
└── integration/
    ├── multi_database_tests.rs
    └── temporal_calculation_tests.rs
```

### 2. Mock LLM Infrastructure
```rust
pub trait LLMClient: Send + Sync {
    async fn query(&self, prompt: &str) -> Result<String>;
}

pub struct MockLLMClient {
    expectations: Vec<Expectation>,
}

pub struct DeepSeekClient {
    api_key: String,
    client: reqwest::Client,
}
```

### 3. Test Execution Framework
- Parallel test execution
- Performance benchmarking
- Result aggregation
- Failure analysis
- Coverage reporting

### 4. Documentation
- Test strategy document
- Coverage matrix
- Performance benchmarks
- Failure analysis guide
- CI/CD integration guide

## Implementation Guidelines

1. **Test Independence**: Each test must be completely independent
2. **Deterministic Results**: Mock tests must be reproducible
3. **Performance Tracking**: Record execution time and resource usage
4. **Error Scenarios**: Test error handling and edge cases
5. **Real-World Relevance**: Use realistic data and queries
6. **Scalability**: Tests must work with full 300k/500k dataset
7. **Maintainability**: Clear naming, good documentation

## Success Criteria

1. **Coverage**: All 7 algorithms thoroughly tested
2. **Reliability**: 100% pass rate for mock tests
3. **Performance**: Meet all performance requirements
4. **Scalability**: Handle full dataset without degradation
5. **Integration**: Seamless DeepSeek API integration
6. **Documentation**: Comprehensive test documentation

This comprehensive testing strategy ensures that the LLMKG cognitive algorithms are rigorously validated for both correctness and performance, with both controlled (mock) and real-world (DeepSeek) scenarios.