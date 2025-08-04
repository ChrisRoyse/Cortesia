# Phase 6: Tiered Execution - Cost-Optimized Query Routing System

## Objective
Implement a three-tier execution system that optimizes accuracy, latency, and cost by intelligently routing queries to appropriate search methods. Achieve 85-97% accuracy across tiers with progressive cost escalation and comprehensive caching strategies following London School TDD methodology.

## Prerequisites
- **Completed Phases**: 0-5 (Foundation through Synthesis Engine)
- **Dependencies**: All search engines and synthesis engine from previous phases
- **Performance Baselines**: Established latency and accuracy metrics per search method
- **Cost Models**: API usage costs and local computation resource costs

## Duration
1 Week (5 days, 40 hours) - Mock-first development with progressive real implementation

## Why Tiered Execution is Essential
The Tiered Execution system maximizes efficiency by matching query complexity to appropriate search methods:

- ✅ **Cost Optimization**: Routes simple queries to fast, cheap methods
- ✅ **Performance Scaling**: Provides predictable latency guarantees per tier
- ✅ **Quality Assurance**: Ensures accuracy requirements are met efficiently
- ✅ **Resource Management**: Balances computation, memory, and API costs
- ✅ **Caching Strategy**: Implements tier-specific caching for optimal performance
- ✅ **Monitoring & Analytics**: Tracks usage patterns and cost optimization opportunities

## SPARC Framework Application

### Specification
- **Input**: User queries with optional performance/cost preferences
- **Output**: Tier-routed search results meeting accuracy and latency requirements
- **Constraints**: Three tiers with distinct accuracy/latency/cost profiles
- **Performance**: Tier 1 <50ms, Tier 2 <500ms, Tier 3 <2s

### Pseudocode
```
FOR each query:
    1. Classify query complexity and type
    2. Determine user preferences (speed vs accuracy vs cost)
    3. Select appropriate tier based on classification
    4. Check tier-specific cache for existing results
    5. Execute search using tier-appropriate methods
    6. Apply tier-specific result processing
    7. Cache results with tier-specific TTL
    8. Return results with tier metadata
```

### Architecture
```rust
pub struct TieredExecutionSystem {
    query_classifier: QueryClassifier,
    tier_router: TierRouter,
    tier_executors: HashMap<ExecutionTier, Box<dyn TierExecutor>>,
    cache_manager: TieredCacheManager,
    cost_optimizer: CostOptimizer,
    performance_monitor: PerformanceMonitor,
}
```

### Refinement
- Mock-first implementation for each tier
- Progressive replacement with real search engines
- Performance tuning based on actual usage patterns

### Completion
- All three tiers operational with real search engines
- Cost optimization algorithms functional
- Performance monitoring and alerting system active

## Technical Approach

### 1. Execution Tier Definitions

#### Tier 1: Fast Local Search
- **Accuracy Target**: 85-90%
- **Latency Target**: < 50ms (P95)
- **Cost Target**: < $0.0001/query
- **Methods**: Exact match (ripgrep), cached results, local indices
- **Use Cases**: Simple lookups, known entities, recent queries

#### Tier 2: Balanced Hybrid Search  
- **Accuracy Target**: 92-95%
- **Latency Target**: < 500ms (P95)
- **Cost Target**: < $0.01/query
- **Methods**: Text search + semantic search + basic synthesis
- **Use Cases**: Most development queries, API documentation, code examples

#### Tier 3: Deep Analysis
- **Accuracy Target**: 95-97%
- **Latency Target**: < 2s (P95)
- **Cost Target**: < $0.05/query
- **Methods**: Full multi-signal synthesis with temporal analysis
- **Use Cases**: Complex debugging, architectural questions, security audits

### 2. Mock Tiered Execution Interface (London TDD - Mock First)
```rust
use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum ExecutionTier {
    Tier1Fast,
    Tier2Balanced,
    Tier3Deep,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TieredQuery {
    pub query: String,
    pub preferred_tier: Option<ExecutionTier>,
    pub max_latency_ms: Option<u64>,
    pub max_cost_cents: Option<u32>,
    pub min_accuracy: Option<f32>,
    pub user_context: UserContext,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserContext {
    pub user_id: String,
    pub session_id: String,
    pub query_history: Vec<String>,
    pub cost_budget_remaining: u32, // in cents
    pub performance_preference: PerformancePreference,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformancePreference {
    FastestResponse,
    BestAccuracy,
    LowestCost,
    Balanced,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TieredResult {
    pub results: Vec<SearchResult>,
    pub tier_used: ExecutionTier,
    pub actual_latency_ms: u64,
    pub actual_cost_cents: u32,
    pub accuracy_estimate: f32,
    pub cache_hit: bool,
    pub tier_metadata: TierMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierMetadata {
    pub methods_used: Vec<String>,
    pub cache_status: CacheStatus,
    pub cost_breakdown: CostBreakdown,
    pub performance_metrics: PerformanceMetrics,
    pub escalation_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostBreakdown {
    pub api_calls: u32,      // cents
    pub compute_time: u32,   // cents
    pub storage_access: u32, // cents
    pub total: u32,         // cents
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub query_classification_ms: u64,
    pub cache_lookup_ms: u64,
    pub search_execution_ms: u64,
    pub result_processing_ms: u64,
    pub total_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheStatus {
    Hit,
    Miss,
    Partial,
    Expired,
}

#[async_trait]
pub trait TieredExecutionSystem {
    async fn execute_query(&self, query: TieredQuery) -> anyhow::Result<TieredResult>;
    async fn classify_query(&self, query: &str) -> QueryClassification;
    async fn estimate_tier_performance(&self, query: &str, tier: ExecutionTier) -> TierEstimate;
    async fn optimize_tier_selection(&self, query: &TieredQuery, estimates: Vec<TierEstimate>) -> ExecutionTier;
}

// Mock implementation for TDD
pub struct MockTieredExecutionSystem {
    expected_classifications: HashMap<String, QueryClassification>,
    tier_responses: HashMap<(String, ExecutionTier), TieredResult>,
    cost_tracking: HashMap<String, u32>,
}

#[async_trait]
impl TieredExecutionSystem for MockTieredExecutionSystem {
    async fn execute_query(&self, query: TieredQuery) -> anyhow::Result<TieredResult> {
        let tier = query.preferred_tier.unwrap_or(ExecutionTier::Tier2Balanced);
        
        self.tier_responses
            .get(&(query.query.clone(), tier))
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Mock response not found for query: {} tier: {:?}", query.query, tier))
    }
    
    async fn classify_query(&self, query: &str) -> QueryClassification {
        self.expected_classifications
            .get(query)
            .cloned()
            .unwrap_or(QueryClassification::default())
    }
    
    async fn estimate_tier_performance(&self, _query: &str, tier: ExecutionTier) -> TierEstimate {
        // Mock estimates based on tier characteristics
        match tier {
            ExecutionTier::Tier1Fast => TierEstimate {
                tier,
                estimated_accuracy: 0.87,
                estimated_latency_ms: 25,
                estimated_cost_cents: 0,
                confidence: 0.95,
            },
            ExecutionTier::Tier2Balanced => TierEstimate {
                tier,
                estimated_accuracy: 0.93,
                estimated_latency_ms: 200,
                estimated_cost_cents: 1,
                confidence: 0.9,
            },
            ExecutionTier::Tier3Deep => TierEstimate {
                tier,
                estimated_accuracy: 0.96,
                estimated_latency_ms: 800,
                estimated_cost_cents: 5,
                confidence: 0.85,
            },
        }
    }
    
    async fn optimize_tier_selection(&self, query: &TieredQuery, estimates: Vec<TierEstimate>) -> ExecutionTier {
        // Mock optimization: choose based on user preference
        match query.user_context.performance_preference {
            PerformancePreference::FastestResponse => ExecutionTier::Tier1Fast,
            PerformancePreference::BestAccuracy => ExecutionTier::Tier3Deep,
            PerformancePreference::LowestCost => ExecutionTier::Tier1Fast,
            PerformancePreference::Balanced => ExecutionTier::Tier2Balanced,
        }
    }
}

impl MockTieredExecutionSystem {
    pub fn new() -> Self {
        Self {
            expected_classifications: HashMap::new(),
            tier_responses: HashMap::new(),
            cost_tracking: HashMap::new(),
        }
    }
    
    pub fn expect_classification(&mut self, query: &str, classification: QueryClassification) {
        self.expected_classifications.insert(query.to_string(), classification);
    }
    
    pub fn expect_tier_response(&mut self, query: &str, tier: ExecutionTier, result: TieredResult) {
        self.tier_responses.insert((query.to_string(), tier), result);
    }
}
```

### 3. Query Classification System
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryClassification {
    pub complexity: QueryComplexity,
    pub query_type: QueryType,
    pub estimated_difficulty: f32,        // 0.0 to 1.0
    pub requires_semantic: bool,
    pub requires_temporal: bool,
    pub requires_ast: bool,
    pub has_special_chars: bool,
    pub estimated_result_count: usize,
    pub classification_confidence: f32,
}

impl Default for QueryClassification {
    fn default() -> Self {
        Self {
            complexity: QueryComplexity::Medium,
            query_type: QueryType::SemanticSearch,
            estimated_difficulty: 0.5,
            requires_semantic: false,
            requires_temporal: false,
            requires_ast: false,
            has_special_chars: false,
            estimated_result_count: 10,
            classification_confidence: 0.8,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryComplexity {
    Simple,    // Exact matches, simple keywords
    Medium,    // Boolean queries, basic semantic search
    Complex,   // Multi-faceted queries requiring synthesis
}

pub struct QueryClassifier {
    complexity_patterns: Vec<ComplexityPattern>,
    query_type_detectors: HashMap<QueryType, Box<dyn QueryTypeDetector>>,
    feature_extractors: Vec<Box<dyn FeatureExtractor>>,
}

#[derive(Debug, Clone)]
pub struct ComplexityPattern {
    pub pattern: regex::Regex,
    pub complexity: QueryComplexity,
    pub confidence: f32,
}

#[async_trait]
pub trait QueryTypeDetector: Send + Sync {
    async fn detect(&self, query: &str, features: &QueryFeatures) -> Option<f32>;
    fn get_query_type(&self) -> QueryType;
}

#[async_trait] 
pub trait FeatureExtractor: Send + Sync {
    async fn extract(&self, query: &str) -> QueryFeatures;
}

#[derive(Debug, Clone, Default)]
pub struct QueryFeatures {
    pub length: usize,
    pub word_count: usize,
    pub special_char_count: usize,
    pub boolean_operators: Vec<String>,
    pub programming_keywords: Vec<String>,
    pub file_extensions: Vec<String>,
    pub api_patterns: Vec<String>,
    pub error_patterns: Vec<String>,
    pub version_numbers: Vec<String>,
    pub has_quotes: bool,
    pub has_wildcards: bool,
    pub has_regex: bool,
}

impl QueryClassifier {
    pub fn new() -> Self {
        let complexity_patterns = vec![
            ComplexityPattern {
                pattern: regex::Regex::new(r"^[a-zA-Z0-9_]+$").unwrap(),
                complexity: QueryComplexity::Simple,
                confidence: 0.9,
            },
            ComplexityPattern {
                pattern: regex::Regex::new(r"\b(AND|OR|NOT)\b").unwrap(),
                complexity: QueryComplexity::Medium,
                confidence: 0.8,
            },
            ComplexityPattern {
                pattern: regex::Regex::new(r"(how to|why does|when should|best practice|explain)").unwrap(),
                complexity: QueryComplexity::Complex,
                confidence: 0.85,
            },
            ComplexityPattern {
                pattern: regex::Regex::new(r"(debug|troubleshoot|fix|error|exception|crash)").unwrap(),
                complexity: QueryComplexity::Complex,
                confidence: 0.8,
            },
        ];
        
        let mut query_type_detectors: HashMap<QueryType, Box<dyn QueryTypeDetector>> = HashMap::new();
        query_type_detectors.insert(QueryType::SpecialCharacters, Box::new(SpecialCharDetector));
        query_type_detectors.insert(QueryType::BooleanQuery, Box::new(BooleanDetector));
        query_type_detectors.insert(QueryType::CodeStructure, Box::new(CodeStructureDetector));
        query_type_detectors.insert(QueryType::Debugging, Box::new(DebuggingDetector));
        query_type_detectors.insert(QueryType::APIUsage, Box::new(APIUsageDetector));
        
        Self {
            complexity_patterns,
            query_type_detectors,
            feature_extractors: vec![Box::new(BasicFeatureExtractor)],
        }
    }
    
    pub async fn classify_query(&self, query: &str) -> QueryClassification {
        // Extract features
        let mut features = QueryFeatures::default();
        for extractor in &self.feature_extractors {
            let extracted = extractor.extract(query).await;
            features = self.merge_features(features, extracted);
        }
        
        // Detect complexity
        let complexity = self.detect_complexity(query, &features);
        
        // Detect query type
        let query_type = self.detect_query_type(query, &features).await;
        
        // Calculate difficulty
        let estimated_difficulty = self.calculate_difficulty(&complexity, &query_type, &features);
        
        // Determine requirements
        let requires_semantic = self.requires_semantic_search(&query_type, &features);
        let requires_temporal = self.requires_temporal_analysis(&query_type, &features);
        let requires_ast = self.requires_ast_analysis(&query_type, &features);
        
        QueryClassification {
            complexity,
            query_type,
            estimated_difficulty,
            requires_semantic,
            requires_temporal,
            requires_ast,
            has_special_chars: features.special_char_count > 0,
            estimated_result_count: self.estimate_result_count(&features),
            classification_confidence: 0.85, // TODO: Calculate based on detection confidence
        }
    }
    
    fn detect_complexity(&self, query: &str, _features: &QueryFeatures) -> QueryComplexity {
        for pattern in &self.complexity_patterns {
            if pattern.pattern.is_match(query) {
                return pattern.complexity.clone();
            }
        }
        QueryComplexity::Medium // Default
    }
    
    async fn detect_query_type(&self, query: &str, features: &QueryFeatures) -> QueryType {
        let mut best_type = QueryType::SemanticSearch; // Default
        let mut best_confidence = 0.0;
        
        for (query_type, detector) in &self.query_type_detectors {
            if let Some(confidence) = detector.detect(query, features).await {
                if confidence > best_confidence {
                    best_confidence = confidence;
                    best_type = query_type.clone();
                }
            }
        }
        
        best_type
    }
    
    fn calculate_difficulty(&self, complexity: &QueryComplexity, query_type: &QueryType, features: &QueryFeatures) -> f32 {
        let mut difficulty = match complexity {
            QueryComplexity::Simple => 0.2,
            QueryComplexity::Medium => 0.5,
            QueryComplexity::Complex => 0.8,
        };
        
        // Adjust based on query type
        difficulty += match query_type {
            QueryType::SpecialCharacters => 0.0,
            QueryType::BooleanQuery => 0.1,
            QueryType::SemanticSearch => 0.2,
            QueryType::CodeStructure => 0.3,
            QueryType::Debugging => 0.4,
            QueryType::APIUsage => 0.2,
            QueryType::PerformanceAnalysis => 0.4,
            QueryType::SecurityAudit => 0.5,
        };
        
        // Adjust based on features
        if features.special_char_count > 5 {
            difficulty += 0.1;
        }
        if features.word_count > 10 {
            difficulty += 0.1;
        }
        if !features.boolean_operators.is_empty() {
            difficulty += 0.1;
        }
        
        difficulty.min(1.0).max(0.0)
    }
    
    fn requires_semantic_search(&self, query_type: &QueryType, _features: &QueryFeatures) -> bool {
        matches!(query_type, 
            QueryType::SemanticSearch | 
            QueryType::Debugging | 
            QueryType::PerformanceAnalysis |
            QueryType::SecurityAudit
        )
    }
    
    fn requires_temporal_analysis(&self, query_type: &QueryType, features: &QueryFeatures) -> bool {
        matches!(query_type, QueryType::Debugging) || 
        !features.version_numbers.is_empty() ||
        features.error_patterns.iter().any(|p| p.contains("recent") || p.contains("since"))
    }
    
    fn requires_ast_analysis(&self, query_type: &QueryType, _features: &QueryFeatures) -> bool {
        matches!(query_type, QueryType::CodeStructure | QueryType::APIUsage)
    }
    
    fn estimate_result_count(&self, features: &QueryFeatures) -> usize {
        // Simple heuristic based on query specificity
        let mut estimate = 10;
        
        if features.has_quotes {
            estimate /= 2; // Exact matches reduce results
        }
        if features.special_char_count > 0 {
            estimate /= 2; // Special chars are specific
        }
        if features.word_count > 5 {
            estimate *= 2; // More words = more potential matches
        }
        
        estimate.max(1).min(100)
    }
    
    fn merge_features(&self, mut base: QueryFeatures, other: QueryFeatures) -> QueryFeatures {
        base.length = base.length.max(other.length);
        base.word_count = base.word_count.max(other.word_count);
        base.special_char_count = base.special_char_count.max(other.special_char_count);
        base.boolean_operators.extend(other.boolean_operators);
        base.programming_keywords.extend(other.programming_keywords);
        base.file_extensions.extend(other.file_extensions);
        base.api_patterns.extend(other.api_patterns);
        base.error_patterns.extend(other.error_patterns);
        base.version_numbers.extend(other.version_numbers);
        base.has_quotes = base.has_quotes || other.has_quotes;
        base.has_wildcards = base.has_wildcards || other.has_wildcards;
        base.has_regex = base.has_regex || other.has_regex;
        base
    }
}

// Feature extractor implementations
pub struct BasicFeatureExtractor;

#[async_trait]
impl FeatureExtractor for BasicFeatureExtractor {
    async fn extract(&self, query: &str) -> QueryFeatures {
        let length = query.len();
        let word_count = query.split_whitespace().count();
        let special_char_count = query.chars().filter(|c| !c.is_alphanumeric() && !c.is_whitespace()).count();
        
        let boolean_operators = vec!["AND", "OR", "NOT"]
            .into_iter()
            .filter(|op| query.to_uppercase().contains(op))
            .map(|s| s.to_string())
            .collect();
        
        let programming_keywords = vec!["function", "class", "struct", "enum", "trait", "impl", "fn", "pub", "async"]
            .into_iter()
            .filter(|keyword| query.to_lowercase().contains(keyword))
            .map(|s| s.to_string())
            .collect();
        
        let file_extensions = regex::Regex::new(r"\.(rs|py|js|ts|java|cpp|c|h|md|json|toml|yaml)")
            .unwrap()
            .find_iter(query)
            .map(|m| m.as_str().to_string())
            .collect();
        
        let has_quotes = query.contains('"') || query.contains('\'');
        let has_wildcards = query.contains('*') || query.contains('?');
        let has_regex = query.contains('[') || query.contains('^') || query.contains('$');
        
        QueryFeatures {
            length,
            word_count,
            special_char_count,
            boolean_operators,
            programming_keywords,
            file_extensions,
            api_patterns: Vec::new(), // TODO: Implement API pattern detection
            error_patterns: Vec::new(), // TODO: Implement error pattern detection
            version_numbers: Vec::new(), // TODO: Implement version detection
            has_quotes,
            has_wildcards,
            has_regex,
        }
    }
}

// Query type detectors
pub struct SpecialCharDetector;

#[async_trait]
impl QueryTypeDetector for SpecialCharDetector {
    async fn detect(&self, query: &str, features: &QueryFeatures) -> Option<f32> {
        if features.special_char_count > 0 && features.word_count <= 3 {
            Some(0.9) // High confidence for short queries with special chars
        } else if features.special_char_count > features.word_count {
            Some(0.7) // Medium confidence when special chars dominate
        } else {
            None
        }
    }
    
    fn get_query_type(&self) -> QueryType {
        QueryType::SpecialCharacters
    }
}

pub struct BooleanDetector;

#[async_trait]
impl QueryTypeDetector for BooleanDetector {
    async fn detect(&self, _query: &str, features: &QueryFeatures) -> Option<f32> {
        if !features.boolean_operators.is_empty() {
            Some(0.95) // Very high confidence for explicit boolean operators
        } else {
            None
        }
    }
    
    fn get_query_type(&self) -> QueryType {
        QueryType::BooleanQuery
    }
}

pub struct CodeStructureDetector;

#[async_trait]
impl QueryTypeDetector for CodeStructureDetector {
    async fn detect(&self, _query: &str, features: &QueryFeatures) -> Option<f32> {
        if !features.programming_keywords.is_empty() || !features.file_extensions.is_empty() {
            Some(0.8) // High confidence for code-related terms
        } else {
            None
        }
    }
    
    fn get_query_type(&self) -> QueryType {
        QueryType::CodeStructure
    }
}

pub struct DebuggingDetector;

#[async_trait]
impl QueryTypeDetector for DebuggingDetector {
    async fn detect(&self, query: &str, _features: &QueryFeatures) -> Option<f32> {
        let debug_terms = ["error", "exception", "crash", "bug", "debug", "troubleshoot", "fix", "broken"];
        let matches = debug_terms.iter().filter(|term| query.to_lowercase().contains(term)).count();
        
        if matches > 0 {
            Some(0.8 + (matches as f32 * 0.05).min(0.2)) // Higher confidence with more debug terms
        } else {
            None
        }
    }
    
    fn get_query_type(&self) -> QueryType {
        QueryType::Debugging
    }
}

pub struct APIUsageDetector;

#[async_trait]
impl QueryTypeDetector for APIUsageDetector {
    async fn detect(&self, query: &str, _features: &QueryFeatures) -> Option<f32> {
        let api_terms = ["api", "endpoint", "method", "function", "usage", "example", "how to use"];
        let matches = api_terms.iter().filter(|term| query.to_lowercase().contains(term)).count();
        
        if matches > 0 {
            Some(0.75 + (matches as f32 * 0.05).min(0.2))
        } else {
            None
        }
    }
    
    fn get_query_type(&self) -> QueryType {
        QueryType::APIUsage
    }
}
```

### 4. Tier Router and Selection Logic
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierEstimate {
    pub tier: ExecutionTier,
    pub estimated_accuracy: f32,
    pub estimated_latency_ms: u64,
    pub estimated_cost_cents: u32,
    pub confidence: f32,
}

pub struct TierRouter {
    tier_selectors: HashMap<PerformancePreference, Box<dyn TierSelector>>,
    cost_calculator: CostCalculator,
    performance_predictor: PerformancePredictor,
}

#[async_trait]
pub trait TierSelector: Send + Sync {
    async fn select_tier(
        &self,
        query: &TieredQuery,
        classification: &QueryClassification,
        estimates: &[TierEstimate],
    ) -> ExecutionTierSelection;
}

#[derive(Debug, Clone)]
pub struct ExecutionTierSelection {
    pub selected_tier: ExecutionTier,
    pub selection_reason: String,
    pub estimated_performance: TierEstimate,
    pub fallback_tiers: Vec<ExecutionTier>,
}

impl TierRouter {
    pub fn new() -> Self {
        let mut tier_selectors: HashMap<PerformancePreference, Box<dyn TierSelector>> = HashMap::new();
        tier_selectors.insert(PerformancePreference::FastestResponse, Box::new(SpeedOptimizedSelector));
        tier_selectors.insert(PerformancePreference::BestAccuracy, Box::new(AccuracyOptimizedSelector));
        tier_selectors.insert(PerformancePreference::LowestCost, Box::new(CostOptimizedSelector));
        tier_selectors.insert(PerformancePreference::Balanced, Box::new(BalancedSelector));
        
        Self {
            tier_selectors,
            cost_calculator: CostCalculator::new(),
            performance_predictor: PerformancePredictor::new(),
        }
    }
    
    pub async fn route_query(
        &self,
        query: &TieredQuery,
        classification: &QueryClassification,
    ) -> anyhow::Result<ExecutionTierSelection> {
        
        // Generate estimates for all tiers
        let estimates = self.generate_tier_estimates(query, classification).await?;
        
        // Select appropriate tier selector
        let selector = self.tier_selectors
            .get(&query.user_context.performance_preference)
            .ok_or_else(|| anyhow::anyhow!("No selector for preference: {:?}", query.user_context.performance_preference))?;
        
        // Select tier
        let selection = selector.select_tier(query, classification, &estimates).await;
        
        Ok(selection)
    }
    
    async fn generate_tier_estimates(
        &self,
        query: &TieredQuery,
        classification: &QueryClassification,
    ) -> anyhow::Result<Vec<TierEstimate>> {
        
        let mut estimates = Vec::new();
        
        // Tier 1 estimate
        let tier1_estimate = self.estimate_tier1_performance(query, classification).await;
        estimates.push(tier1_estimate);
        
        // Tier 2 estimate
        let tier2_estimate = self.estimate_tier2_performance(query, classification).await;
        estimates.push(tier2_estimate);
        
        // Tier 3 estimate
        let tier3_estimate = self.estimate_tier3_performance(query, classification).await;
        estimates.push(tier3_estimate);
        
        Ok(estimates)
    }
    
    async fn estimate_tier1_performance(&self, query: &TieredQuery, classification: &QueryClassification) -> TierEstimate {
        let base_latency = 30; // Base latency for local search
        let latency_penalty = if classification.has_special_chars { 5 } else { 0 };
        
        let estimated_accuracy = match classification.complexity {
            QueryComplexity::Simple => 0.90,
            QueryComplexity::Medium => 0.85,
            QueryComplexity::Complex => 0.75, // Tier 1 struggles with complex queries
        };
        
        TierEstimate {
            tier: ExecutionTier::Tier1Fast,
            estimated_accuracy,
            estimated_latency_ms: base_latency + latency_penalty,
            estimated_cost_cents: 0, // Local execution is free
            confidence: 0.95,
        }
    }
    
    async fn estimate_tier2_performance(&self, query: &TieredQuery, classification: &QueryClassification) -> TierEstimate {
        let base_latency = 200;
        let semantic_penalty = if classification.requires_semantic { 100 } else { 0 };
        let complexity_penalty = match classification.complexity {
            QueryComplexity::Simple => 0,
            QueryComplexity::Medium => 50,
            QueryComplexity::Complex => 100,
        };
        
        let estimated_accuracy = match classification.complexity {
            QueryComplexity::Simple => 0.95,
            QueryComplexity::Medium => 0.93,
            QueryComplexity::Complex => 0.90,
        };
        
        let estimated_cost = if classification.requires_semantic { 1 } else { 0 };
        
        TierEstimate {
            tier: ExecutionTier::Tier2Balanced,
            estimated_accuracy,
            estimated_latency_ms: base_latency + semantic_penalty + complexity_penalty,
            estimated_cost_cents: estimated_cost,
            confidence: 0.90,
        }
    }
    
    async fn estimate_tier3_performance(&self, query: &TieredQuery, classification: &QueryClassification) -> TierEstimate {
        let base_latency = 600;
        let temporal_penalty = if classification.requires_temporal { 200 } else { 0 };
        let ast_penalty = if classification.requires_ast { 150 } else { 0 };
        let synthesis_penalty = 100; // Always includes synthesis
        
        let estimated_accuracy = match classification.complexity {
            QueryComplexity::Simple => 0.97,
            QueryComplexity::Medium => 0.96,
            QueryComplexity::Complex => 0.95,
        };
        
        let base_cost = 2; // Synthesis cost
        let semantic_cost = if classification.requires_semantic { 2 } else { 0 };
        let temporal_cost = if classification.requires_temporal { 1 } else { 0 };
        
        TierEstimate {
            tier: ExecutionTier::Tier3Deep,
            estimated_accuracy,
            estimated_latency_ms: base_latency + temporal_penalty + ast_penalty + synthesis_penalty,
            estimated_cost_cents: base_cost + semantic_cost + temporal_cost,
            confidence: 0.85,
        }
    }
}

// Tier selector implementations
pub struct SpeedOptimizedSelector;

#[async_trait]
impl TierSelector for SpeedOptimizedSelector {
    async fn select_tier(
        &self,
        query: &TieredQuery,
        _classification: &QueryClassification,
        estimates: &[TierEstimate],
    ) -> ExecutionTierSelection {
        
        // Filter by latency constraint
        let valid_estimates: Vec<_> = estimates
            .iter()
            .filter(|est| {
                query.max_latency_ms
                    .map(|max| est.estimated_latency_ms <= max)
                    .unwrap_or(true)
            })
            .collect();
        
        // Select fastest valid tier
        let selected = valid_estimates
            .iter()
            .min_by_key(|est| est.estimated_latency_ms)
            .unwrap_or(&estimates[0]);
        
        ExecutionTierSelection {
            selected_tier: selected.tier,
            selection_reason: format!("Fastest response: {}ms", selected.estimated_latency_ms),
            estimated_performance: (*selected).clone(),
            fallback_tiers: vec![ExecutionTier::Tier2Balanced, ExecutionTier::Tier3Deep],
        }
    }
}

pub struct AccuracyOptimizedSelector;

#[async_trait]  
impl TierSelector for AccuracyOptimizedSelector {
    async fn select_tier(
        &self,
        query: &TieredQuery,
        _classification: &QueryClassification,
        estimates: &[TierEstimate],
    ) -> ExecutionTierSelection {
        
        // Filter by accuracy requirement
        let valid_estimates: Vec<_> = estimates
            .iter()
            .filter(|est| {
                query.min_accuracy
                    .map(|min| est.estimated_accuracy >= min)
                    .unwrap_or(true)
            })
            .collect();
        
        // Select most accurate valid tier
        let selected = valid_estimates
            .iter()
            .max_by(|a, b| a.estimated_accuracy.partial_cmp(&b.estimated_accuracy).unwrap())
            .unwrap_or(&estimates[2]); // Default to Tier 3 for accuracy
        
        ExecutionTierSelection {
            selected_tier: selected.tier,
            selection_reason: format!("Best accuracy: {:.1}%", selected.estimated_accuracy * 100.0),
            estimated_performance: (*selected).clone(),
            fallback_tiers: vec![ExecutionTier::Tier2Balanced, ExecutionTier::Tier1Fast],
        }
    }
}

pub struct CostOptimizedSelector;

#[async_trait]
impl TierSelector for CostOptimizedSelector {
    async fn select_tier(
        &self,
        query: &TieredQuery,
        _classification: &QueryClassification,
        estimates: &[TierEstimate],
    ) -> ExecutionTierSelection {
        
        // Filter by cost constraint
        let valid_estimates: Vec<_> = estimates
            .iter()
            .filter(|est| {
                query.max_cost_cents
                    .map(|max| est.estimated_cost_cents <= max)
                    .unwrap_or(true) &&
                query.user_context.cost_budget_remaining >= est.estimated_cost_cents
            })
            .collect();
        
        // Select cheapest valid tier
        let selected = valid_estimates
            .iter()
            .min_by_key(|est| est.estimated_cost_cents)
            .unwrap_or(&estimates[0]); // Default to Tier 1 for cost
        
        ExecutionTierSelection {
            selected_tier: selected.tier,
            selection_reason: format!("Lowest cost: {}¢", selected.estimated_cost_cents),
            estimated_performance: (*selected).clone(),
            fallback_tiers: vec![ExecutionTier::Tier2Balanced, ExecutionTier::Tier3Deep],
        }
    }
}

pub struct BalancedSelector;

#[async_trait]
impl TierSelector for BalancedSelector {
    async fn select_tier(
        &self,
        query: &TieredQuery,
        classification: &QueryClassification,
        estimates: &[TierEstimate],
    ) -> ExecutionTierSelection {
        
        // Calculate balanced score for each tier
        let scored_estimates: Vec<_> = estimates
            .iter()
            .map(|est| {
                let accuracy_score = est.estimated_accuracy;
                let speed_score = 1.0 - (est.estimated_latency_ms as f32 / 2000.0).min(1.0);
                let cost_score = 1.0 - (est.estimated_cost_cents as f32 / 10.0).min(1.0);
                
                // Weight based on query classification
                let weights = match classification.complexity {
                    QueryComplexity::Simple => (0.3, 0.5, 0.2),    // Prefer speed
                    QueryComplexity::Medium => (0.4, 0.4, 0.2),    // Balanced
                    QueryComplexity::Complex => (0.6, 0.3, 0.1),   // Prefer accuracy
                };
                
                let balanced_score = accuracy_score * weights.0 + speed_score * weights.1 + cost_score * weights.2;
                (est, balanced_score)
            })
            .collect();
        
        // Select highest scoring tier
        let (selected, score) = scored_estimates
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        
        ExecutionTierSelection {
            selected_tier: selected.tier,
            selection_reason: format!("Balanced optimization score: {:.2}", score),
            estimated_performance: (*selected).clone(),
            fallback_tiers: vec![ExecutionTier::Tier2Balanced, ExecutionTier::Tier1Fast, ExecutionTier::Tier3Deep],
        }
    }
}
```

### 5. Tiered Cache Management System
```rust
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

pub struct TieredCacheManager {
    tier1_cache: Arc<RwLock<LRUCache<String, CachedResult>>>,
    tier2_cache: Arc<RwLock<LRUCache<String, CachedResult>>>,
    tier3_cache: Arc<RwLock<LRUCache<String, CachedResult>>>,
    cache_policies: HashMap<ExecutionTier, CachePolicy>,
}

#[derive(Debug, Clone)]
pub struct CachedResult {
    pub result: TieredResult,
    pub cached_at: Instant,
    pub access_count: u32,
    pub last_accessed: Instant,
}

#[derive(Debug, Clone)]
pub struct CachePolicy {
    pub max_entries: usize,
    pub ttl: Duration,
    pub cache_hit_threshold: f32,  // Minimum confidence to cache
    pub invalidation_strategy: InvalidationStrategy,
}

#[derive(Debug, Clone)]
pub enum InvalidationStrategy {
    TTL,              // Time-based expiration
    LRU,              // Least recently used
    AccessFrequency,  // Based on access patterns
    Adaptive,         // Dynamic based on performance
}

impl TieredCacheManager {
    pub fn new() -> Self {
        let mut cache_policies = HashMap::new();
        
        // Tier 1: Large cache with longer TTL (fast queries are cached longer)
        cache_policies.insert(ExecutionTier::Tier1Fast, CachePolicy {
            max_entries: 10000,
            ttl: Duration::from_secs(3600), // 1 hour
            cache_hit_threshold: 0.8,
            invalidation_strategy: InvalidationStrategy::LRU,
        });
        
        // Tier 2: Medium cache with moderate TTL
        cache_policies.insert(ExecutionTier::Tier2Balanced, CachePolicy {
            max_entries: 5000,
            ttl: Duration::from_secs(1800), // 30 minutes
            cache_hit_threshold: 0.9,
            invalidation_strategy: InvalidationStrategy::Adaptive,
        });
        
        // Tier 3: Smaller cache with shorter TTL (expensive queries change faster)
        cache_policies.insert(ExecutionTier::Tier3Deep, CachePolicy {
            max_entries: 1000,
            ttl: Duration::from_secs(600), // 10 minutes
            cache_hit_threshold: 0.95,
            invalidation_strategy: InvalidationStrategy::AccessFrequency,
        });
        
        Self {
            tier1_cache: Arc::new(RwLock::new(LRUCache::new(10000))),
            tier2_cache: Arc::new(RwLock::new(LRUCache::new(5000))),
            tier3_cache: Arc::new(RwLock::new(LRUCache::new(1000))),
            cache_policies,
        }
    }
    
    pub async fn get_cached_result(&self, query: &str, tier: ExecutionTier) -> Option<TieredResult> {
        let cache_key = self.generate_cache_key(query, tier);
        
        let cache = match tier {
            ExecutionTier::Tier1Fast => &self.tier1_cache,
            ExecutionTier::Tier2Balanced => &self.tier2_cache,
            ExecutionTier::Tier3Deep => &self.tier3_cache,
        };
        
        let mut cache_guard = cache.write().await;
        
        if let Some(cached) = cache_guard.get_mut(&cache_key) {
            // Check TTL
            let policy = self.cache_policies.get(&tier).unwrap();
            if cached.cached_at.elapsed() < policy.ttl {
                // Update access statistics
                cached.access_count += 1;
                cached.last_accessed = Instant::now();
                
                // Clone the result and mark as cache hit
                let mut result = cached.result.clone();
                result.cache_hit = true;
                result.tier_metadata.cache_status = CacheStatus::Hit;
                
                return Some(result);
            } else {
                // Expired, remove from cache
                cache_guard.remove(&cache_key);
            }
        }
        
        None
    }
    
    pub async fn cache_result(&self, query: &str, result: &TieredResult) {
        // Only cache if result meets quality threshold
        let policy = self.cache_policies.get(&result.tier_used).unwrap();
        if result.accuracy_estimate < policy.cache_hit_threshold {
            return; // Don't cache low-quality results
        }
        
        let cache_key = self.generate_cache_key(query, result.tier_used);
        
        let cached_result = CachedResult {
            result: result.clone(),
            cached_at: Instant::now(),
            access_count: 0,
            last_accessed: Instant::now(),
        };
        
        let cache = match result.tier_used {
            ExecutionTier::Tier1Fast => &self.tier1_cache,
            ExecutionTier::Tier2Balanced => &self.tier2_cache,
            ExecutionTier::Tier3Deep => &self.tier3_cache,
        };
        
        let mut cache_guard = cache.write().await;
        cache_guard.put(cache_key, cached_result);
    }
    
    pub async fn invalidate_cache(&self, pattern: &str) {
        // Invalidate entries matching pattern across all tiers
        let caches = vec![&self.tier1_cache, &self.tier2_cache, &self.tier3_cache];
        
        for cache in caches {
            let mut cache_guard = cache.write().await;
            let keys_to_remove: Vec<_> = cache_guard.keys()
                .filter(|key| key.contains(pattern))
                .cloned()
                .collect();
            
            for key in keys_to_remove {
                cache_guard.remove(&key);
            }
        }
    }
    
    pub async fn get_cache_statistics(&self) -> CacheStatistics {
        let tier1_stats = self.get_tier_cache_stats(&self.tier1_cache, ExecutionTier::Tier1Fast).await;
        let tier2_stats = self.get_tier_cache_stats(&self.tier2_cache, ExecutionTier::Tier2Balanced).await;
        let tier3_stats = self.get_tier_cache_stats(&self.tier3_cache, ExecutionTier::Tier3Deep).await;
        
        CacheStatistics {
            tier_stats: [
                (ExecutionTier::Tier1Fast, tier1_stats),
                (ExecutionTier::Tier2Balanced, tier2_stats),
                (ExecutionTier::Tier3Deep, tier3_stats),
            ].iter().cloned().collect(),
            total_entries: tier1_stats.entry_count + tier2_stats.entry_count + tier3_stats.entry_count,
            total_hit_rate: (tier1_stats.hit_rate + tier2_stats.hit_rate + tier3_stats.hit_rate) / 3.0,
        }
    }
    
    async fn get_tier_cache_stats(&self, cache: &Arc<RwLock<LRUCache<String, CachedResult>>>, tier: ExecutionTier) -> TierCacheStats {
        let cache_guard = cache.read().await;
        let entry_count = cache_guard.len();
        
        let total_accesses: u32 = cache_guard.values().map(|cached| cached.access_count).sum();
        let avg_access_count = if entry_count > 0 { total_accesses as f32 / entry_count as f32 } else { 0.0 };
        
        // Calculate hit rate (simplified - would need more tracking in real implementation)
        let hit_rate = if entry_count > 0 { 0.8 } else { 0.0 }; // Mock value
        
        TierCacheStats {
            tier,
            entry_count,
            hit_rate,
            avg_access_count,
            memory_usage_mb: entry_count * 1024 / 1024, // Rough estimate
        }
    }
    
    fn generate_cache_key(&self, query: &str, tier: ExecutionTier) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        query.hash(&mut hasher);
        tier.hash(&mut hasher);
        
        format!("{}:{:x}", tier as u8, hasher.finish())
    }
}

#[derive(Debug, Clone)]
pub struct CacheStatistics {
    pub tier_stats: HashMap<ExecutionTier, TierCacheStats>,
    pub total_entries: usize,
    pub total_hit_rate: f32,
}

#[derive(Debug, Clone)]
pub struct TierCacheStats {
    pub tier: ExecutionTier,
    pub entry_count: usize,
    pub hit_rate: f32,
    pub avg_access_count: f32,
    pub memory_usage_mb: usize,
}

// Simple LRU Cache implementation (in real code, use a proper crate)
use std::collections::HashMap as StdHashMap;

pub struct LRUCache<K, V> {
    capacity: usize,
    map: StdHashMap<K, V>,
}

impl<K: Clone + std::hash::Hash + Eq, V> LRUCache<K, V> {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            map: StdHashMap::new(),
        }
    }
    
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        self.map.get_mut(key)
    }
    
    pub fn put(&mut self, key: K, value: V) {
        if self.map.len() >= self.capacity && !self.map.contains_key(&key) {
            // Remove oldest entry (simplified - would need proper LRU tracking)
            if let Some(first_key) = self.map.keys().next().cloned() {
                self.map.remove(&first_key);
            }
        }
        self.map.insert(key, value);
    }
    
    pub fn remove(&mut self, key: &K) -> Option<V> {
        self.map.remove(key)
    }
    
    pub fn len(&self) -> usize {
        self.map.len()
    }
    
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.map.keys()
    }
    
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.map.values()
    }
}
```

### 6. Cost Optimization Engine
```rust
pub struct CostOptimizer {
    pricing_model: PricingModel,
    budget_tracker: BudgetTracker,
    optimization_strategies: Vec<Box<dyn OptimizationStrategy>>,
}

#[derive(Debug, Clone)]
pub struct PricingModel {
    pub api_costs: HashMap<String, f32>,      // Cost per API call in cents
    pub compute_costs: HashMap<String, f32>,  // Cost per compute unit in cents
    pub storage_costs: f32,                   // Cost per GB-hour in cents
}

pub struct BudgetTracker {
    user_budgets: HashMap<String, UserBudget>,
    global_budget: GlobalBudget,
}

#[derive(Debug, Clone)]
pub struct UserBudget {
    pub user_id: String,
    pub daily_limit_cents: u32,
    pub used_today_cents: u32,
    pub monthly_limit_cents: u32,
    pub used_this_month_cents: u32,
    pub last_reset: std::time::SystemTime,
}

#[derive(Debug, Clone)]
pub struct GlobalBudget {
    pub daily_limit_cents: u32,
    pub used_today_cents: u32,
    pub alert_threshold_percent: f32,
    pub last_reset: std::time::SystemTime,
}

#[async_trait]
pub trait OptimizationStrategy: Send + Sync {
    async fn optimize(&self, query: &TieredQuery, estimates: &[TierEstimate]) -> OptimizationRecommendation;
    fn get_strategy_name(&self) -> &'static str;
}

#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    pub recommended_tier: ExecutionTier,
    pub cost_savings_cents: u32,
    pub accuracy_tradeoff: f32,
    pub reasoning: String,
    pub alternative_approaches: Vec<AlternativeApproach>,
}

#[derive(Debug, Clone)]
pub struct AlternativeApproach {
    pub description: String,
    pub tier: ExecutionTier,
    pub cost_cents: u32,
    pub accuracy_estimate: f32,
    pub latency_ms: u64,
}

impl CostOptimizer {
    pub fn new() -> Self {
        let pricing_model = PricingModel {
            api_costs: [
                ("openai_embedding".to_string(), 0.1),  // $0.001 per 1000 tokens
                ("vector_search".to_string(), 0.01),    // Local vector search cost
                ("synthesis".to_string(), 0.05),        // Synthesis processing cost
            ].iter().cloned().collect(),
            compute_costs: [
                ("cpu_second".to_string(), 0.001),      // CPU time cost
                ("memory_gb_hour".to_string(), 0.01),   // Memory usage cost
            ].iter().cloned().collect(),
            storage_costs: 0.001, // Storage access cost per GB
        };
        
        let optimization_strategies: Vec<Box<dyn OptimizationStrategy>> = vec![
            Box::new(CacheOptimizationStrategy),
            Box::new(TierDowngradeStrategy),
            Box::new(BatchingStrategy),
            Box::new(PreemptiveUpgradeStrategy),
        ];
        
        Self {
            pricing_model,
            budget_tracker: BudgetTracker::new(),
            optimization_strategies,
        }
    }
    
    pub async fn optimize_query_execution(
        &self,
        query: &TieredQuery,
        estimates: &[TierEstimate],
    ) -> anyhow::Result<OptimizationRecommendation> {
        
        // Check budget constraints
        self.check_budget_constraints(&query.user_context.user_id).await?;
        
        // Apply optimization strategies
        let mut best_recommendation = OptimizationRecommendation {
            recommended_tier: ExecutionTier::Tier2Balanced, // Default
            cost_savings_cents: 0,
            accuracy_tradeoff: 0.0,
            reasoning: "Default selection".to_string(),
            alternative_approaches: Vec::new(),
        };
        
        let mut best_score = 0.0;
        
        for strategy in &self.optimization_strategies {
            let recommendation = strategy.optimize(query, estimates).await;
            let score = self.score_recommendation(&recommendation, query);
            
            if score > best_score {
                best_score = score;
                best_recommendation = recommendation;
            }
        }
        
        Ok(best_recommendation)
    }
    
    pub async fn track_usage(&mut self, user_id: &str, actual_cost_cents: u32) -> anyhow::Result<()> {
        self.budget_tracker.record_usage(user_id, actual_cost_cents).await
    }
    
    pub async fn get_cost_analytics(&self, user_id: &str) -> CostAnalytics {
        let user_budget = self.budget_tracker.get_user_budget(user_id).await;
        
        CostAnalytics {
            daily_usage_cents: user_budget.used_today_cents,
            daily_limit_cents: user_budget.daily_limit_cents,
            monthly_usage_cents: user_budget.used_this_month_cents,
            monthly_limit_cents: user_budget.monthly_limit_cents,
            projected_monthly_cost: self.project_monthly_cost(&user_budget),
            cost_per_query_avg: self.calculate_avg_cost_per_query(&user_budget),
            optimization_opportunities: self.identify_optimization_opportunities(&user_budget).await,
        }
    }
    
    async fn check_budget_constraints(&self, user_id: &str) -> anyhow::Result<()> {
        let user_budget = self.budget_tracker.get_user_budget(user_id).await;
        
        if user_budget.used_today_cents >= user_budget.daily_limit_cents {
            return Err(anyhow::anyhow!("Daily budget exceeded for user: {}", user_id));
        }
        
        if user_budget.used_this_month_cents >= user_budget.monthly_limit_cents {
            return Err(anyhow::anyhow!("Monthly budget exceeded for user: {}", user_id));
        }
        
        Ok(())
    }
    
    fn score_recommendation(&self, recommendation: &OptimizationRecommendation, query: &TieredQuery) -> f32 {
        let cost_score = (recommendation.cost_savings_cents as f32) / 10.0; // Cost savings value
        let accuracy_penalty = recommendation.accuracy_tradeoff * 100.0;   // Accuracy loss penalty
        
        let preference_bonus = match query.user_context.performance_preference {
            PerformancePreference::LowestCost => cost_score * 2.0,
            PerformancePreference::BestAccuracy => -accuracy_penalty * 2.0,
            _ => 0.0,
        };
        
        cost_score - accuracy_penalty + preference_bonus
    }
    
    fn project_monthly_cost(&self, user_budget: &UserBudget) -> u32 {
        let days_elapsed = self.days_since_month_start();
        if days_elapsed > 0 {
            (user_budget.used_this_month_cents * 30) / days_elapsed as u32
        } else {
            user_budget.used_this_month_cents
        }
    }
    
    fn calculate_avg_cost_per_query(&self, _user_budget: &UserBudget) -> f32 {
        // Simplified - would need query count tracking
        1.5 // Mock average
    }
    
    async fn identify_optimization_opportunities(&self, _user_budget: &UserBudget) -> Vec<String> {
        vec![
            "Consider using Tier 1 for simple lookups".to_string(),
            "Enable aggressive caching for repeated queries".to_string(),
            "Batch similar queries together".to_string(),
        ]
    }
    
    fn days_since_month_start(&self) -> u32 {
        // Simplified date calculation
        15 // Mock value
    }
}

// Optimization strategy implementations
pub struct CacheOptimizationStrategy;

#[async_trait]
impl OptimizationStrategy for CacheOptimizationStrategy {
    async fn optimize(&self, query: &TieredQuery, estimates: &[TierEstimate]) -> OptimizationRecommendation {
        // If query is likely to be repeated, recommend caching tier
        let is_cacheable = query.query.len() < 100 && !query.query.contains("today") && !query.query.contains("now");
        
        if is_cacheable {
            // Recommend higher tier for better caching
            OptimizationRecommendation {
                recommended_tier: ExecutionTier::Tier2Balanced,
                cost_savings_cents: 2, // Long-term savings from caching
                accuracy_tradeoff: 0.0,
                reasoning: "Query is cacheable, higher tier provides better long-term value".to_string(),
                alternative_approaches: Vec::new(),
            }
        } else {
            // Recommend lower tier for non-cacheable queries
            OptimizationRecommendation {
                recommended_tier: ExecutionTier::Tier1Fast,
                cost_savings_cents: 3,
                accuracy_tradeoff: 0.05,
                reasoning: "Non-cacheable query, optimize for immediate cost".to_string(),
                alternative_approaches: Vec::new(),
            }
        }
    }
    
    fn get_strategy_name(&self) -> &'static str {
        "CacheOptimization"
    }
}

pub struct TierDowngradeStrategy;

#[async_trait]
impl OptimizationStrategy for TierDowngradeStrategy {
    async fn optimize(&self, _query: &TieredQuery, estimates: &[TierEstimate]) -> OptimizationRecommendation {
        // Look for opportunities to downgrade tier without significant accuracy loss
        let tier2_estimate = estimates.iter().find(|e| e.tier == ExecutionTier::Tier2Balanced);
        let tier1_estimate = estimates.iter().find(|e| e.tier == ExecutionTier::Tier1Fast);
        
        if let (Some(tier2), Some(tier1)) = (tier2_estimate, tier1_estimate) {
            let accuracy_loss = tier2.estimated_accuracy - tier1.estimated_accuracy;
            let cost_savings = tier2.estimated_cost_cents - tier1.estimated_cost_cents;
            
            if accuracy_loss < 0.1 && cost_savings > 0 {
                return OptimizationRecommendation {
                    recommended_tier: ExecutionTier::Tier1Fast,
                    cost_savings_cents: cost_savings,
                    accuracy_tradeoff: accuracy_loss,
                    reasoning: format!("Minor accuracy loss ({:.1}%) for {}¢ savings", accuracy_loss * 100.0, cost_savings),
                    alternative_approaches: Vec::new(),
                };
            }
        }
        
        // No optimization opportunity
        OptimizationRecommendation {
            recommended_tier: ExecutionTier::Tier2Balanced,
            cost_savings_cents: 0,
            accuracy_tradeoff: 0.0,
            reasoning: "No beneficial downgrade opportunity".to_string(),
            alternative_approaches: Vec::new(),
        }
    }
    
    fn get_strategy_name(&self) -> &'static str {
        "TierDowngrade"
    }
}

pub struct BatchingStrategy;

#[async_trait]
impl OptimizationStrategy for BatchingStrategy {
    async fn optimize(&self, query: &TieredQuery, _estimates: &[TierEstimate]) -> OptimizationRecommendation {
        // Check if query could benefit from batching with recent queries
        let similar_recent_queries = query.user_context.query_history
            .iter()
            .filter(|q| self.queries_similar(&query.query, q))
            .count();
        
        if similar_recent_queries > 0 {
            OptimizationRecommendation {
                recommended_tier: ExecutionTier::Tier2Balanced,
                cost_savings_cents: 1,
                accuracy_tradeoff: 0.0,
                reasoning: "Similar queries detected, batching can reduce API costs".to_string(),
                alternative_approaches: vec![
                    AlternativeApproach {
                        description: "Batch with recent similar queries".to_string(),
                        tier: ExecutionTier::Tier2Balanced,
                        cost_cents: 2,
                        accuracy_estimate: 0.93,
                        latency_ms: 250,
                    }
                ],
            }
        } else {
            OptimizationRecommendation {
                recommended_tier: ExecutionTier::Tier1Fast,
                cost_savings_cents: 0,
                accuracy_tradeoff: 0.0,
                reasoning: "No batching opportunity".to_string(),
                alternative_approaches: Vec::new(),
            }
        }
    }
    
    fn get_strategy_name(&self) -> &'static str {
        "Batching"
    }
    
    fn queries_similar(&self, query1: &str, query2: &str) -> bool {
        // Simple similarity check
        let words1: std::collections::HashSet<_> = query1.split_whitespace().collect();
        let words2: std::collections::HashSet<_> = query2.split_whitespace().collect();
        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();
        
        if union > 0 {
            (intersection as f32 / union as f32) > 0.6
        } else {
            false
        }
    }
}

pub struct PreemptiveUpgradeStrategy;

#[async_trait]
impl OptimizationStrategy for PreemptiveUpgradeStrategy {
    async fn optimize(&self, query: &TieredQuery, estimates: &[TierEstimate]) -> OptimizationRecommendation {
        // Detect queries that might benefit from preemptive upgrade
        let has_debug_terms = query.query.to_lowercase().contains("debug") || 
                             query.query.to_lowercase().contains("error") ||
                             query.query.to_lowercase().contains("fix");
        
        if has_debug_terms {
            OptimizationRecommendation {
                recommended_tier: ExecutionTier::Tier3Deep,
                cost_savings_cents: 0, // No immediate savings, but prevents escalation
                accuracy_tradeoff: -0.05, // Actually improves accuracy
                reasoning: "Debug query detected, preemptive upgrade prevents tier escalation".to_string(),
                alternative_approaches: vec![
                    AlternativeApproach {
                        description: "Start with Tier 2, escalate if needed".to_string(),
                        tier: ExecutionTier::Tier2Balanced,
                        cost_cents: 1,
                        accuracy_estimate: 0.88,
                        latency_ms: 300,
                    }
                ],
            }
        } else {
            OptimizationRecommendation {
                recommended_tier: ExecutionTier::Tier1Fast,
                cost_savings_cents: 0,
                accuracy_tradeoff: 0.0,
                reasoning: "No preemptive upgrade needed".to_string(),
                alternative_approaches: Vec::new(),
            }
        }
    }
    
    fn get_strategy_name(&self) -> &'static str {
        "PreemptiveUpgrade"
    }
}

impl BudgetTracker {
    pub fn new() -> Self {
        Self {
            user_budgets: HashMap::new(),
            global_budget: GlobalBudget {
                daily_limit_cents: 10000, // $100/day
                used_today_cents: 0,
                alert_threshold_percent: 0.8,
                last_reset: std::time::SystemTime::now(),
            },
        }
    }
    
    pub async fn record_usage(&mut self, user_id: &str, cost_cents: u32) -> anyhow::Result<()> {
        let user_budget = self.user_budgets
            .entry(user_id.to_string())
            .or_insert_with(|| UserBudget {
                user_id: user_id.to_string(),
                daily_limit_cents: 1000,  // $10/day default
                used_today_cents: 0,
                monthly_limit_cents: 10000, // $100/month default
                used_this_month_cents: 0,
                last_reset: std::time::SystemTime::now(),
            });
        
        user_budget.used_today_cents += cost_cents;
        user_budget.used_this_month_cents += cost_cents;
        
        self.global_budget.used_today_cents += cost_cents;
        
        Ok(())
    }
    
    pub async fn get_user_budget(&self, user_id: &str) -> UserBudget {
        self.user_budgets
            .get(user_id)
            .cloned()
            .unwrap_or_else(|| UserBudget {
                user_id: user_id.to_string(),
                daily_limit_cents: 1000,
                used_today_cents: 0,
                monthly_limit_cents: 10000,
                used_this_month_cents: 0,
                last_reset: std::time::SystemTime::now(),
            })
    }
}

#[derive(Debug, Clone)]
pub struct CostAnalytics {
    pub daily_usage_cents: u32,
    pub daily_limit_cents: u32,
    pub monthly_usage_cents: u32,
    pub monthly_limit_cents: u32,
    pub projected_monthly_cost: u32,
    pub cost_per_query_avg: f32,
    pub optimization_opportunities: Vec<String>,
}
```

## Implementation Tasks (600-699)

### Task 600: Mock Tiered Execution Foundation (1 day)
Create complete mock infrastructure for TDD development:
- MockTieredExecutionSystem with configurable tier responses
- MockQueryClassifier with predictable classifications
- MockTierRouter with deterministic routing logic
- MockCacheManager with controlled cache behavior
- MockCostOptimizer with optimization scenarios
- Complete test harness for tiered execution validation

### Task 601: Query Classification System (1 day)
Replace mocks with real query analysis:
- Implement QueryClassifier with pattern matching
- Create feature extractors for query analysis
- Build query type detectors for each category
- Add complexity assessment algorithms
- Build comprehensive classification test suite

### Task 602: Tier Router Implementation (1 day)
Build intelligent tier selection:
- Implement multiple tier selection strategies
- Create performance estimation algorithms
- Add constraint checking for latency/cost/accuracy
- Build fallback tier selection logic
- Comprehensive routing accuracy validation

### Task 603: Tiered Cache Management (1 day)
Implement sophisticated caching system:
- Create tier-specific cache policies
- Implement LRU/TTL/adaptive invalidation strategies
- Add cache statistics and monitoring
- Build cache warm-up strategies
- Performance optimization for cache operations

### Task 604: Cost Optimization Engine (1 day)
Develop cost-aware optimization:
- Implement budget tracking and enforcement
- Create optimization strategy framework
- Add cost projection and analytics
- Build automatic cost alerts and limits
- Validate cost optimization effectiveness

### Task 605: Real Tier Executor Integration (Half day)
Connect to actual search engines:
- Integrate Tier 1 with local search engines (Phase 1-2)
- Connect Tier 2 with hybrid search (Phase 3-4)
- Link Tier 3 with full synthesis engine (Phase 5)
- Add error handling and graceful degradation
- Comprehensive integration testing

### Task 606: Performance Monitoring System (Half day)
Build comprehensive monitoring:
- Implement latency tracking per tier
- Add accuracy measurement and calibration
- Create cost tracking and reporting
- Build performance dashboards
- Set up alerting for SLA violations

### Task 607: Tier Escalation Logic (1 day)
Create dynamic tier escalation:
- Implement automatic tier upgrade on failure
- Add confidence threshold-based escalation
- Create user-initiated tier escalation
- Build escalation cost tracking
- Validate escalation decision accuracy

### Task 608: A/B Testing Framework (Half day)
Enable tier optimization experiments:
- Create A/B test infrastructure for tier selection
- Implement statistical significance testing
- Add performance comparison tools
- Build experiment result analysis
- Automated optimization based on test results

### Task 609: Tiered Execution Validation Suite (Half day)
Create comprehensive testing framework:
- Build tier selection accuracy tests
- Implement cost optimization validation
- Create performance benchmark suite
- Add tier escalation scenario testing
- End-to-end tiered execution validation

## Deliverables

### Rust Source Files
1. `src/tiered/mod.rs` - Main tiered execution interface
2. `src/tiered/classification.rs` - Query classification system
3. `src/tiered/routing.rs` - Tier selection and routing logic
4. `src/tiered/caching.rs` - Tiered cache management
5. `src/tiered/optimization.rs` - Cost optimization engine
6. `src/tiered/monitoring.rs` - Performance monitoring system
7. `src/tiered/executors.rs` - Tier-specific execution logic
8. `src/tiered/mocks.rs` - Mock implementations for TDD

### Configuration Files
1. Tier configuration with accuracy/latency/cost targets
2. Cache policies for each tier
3. Cost optimization parameters and thresholds
4. Query classification patterns and weights
5. Budget limits and alerting configurations

### Test Suites
1. Unit tests for each tiered component
2. Integration tests for tier selection accuracy
3. Performance benchmarks for each tier
4. Cost optimization validation tests
5. A/B testing framework validation

## Success Metrics

### Functional Requirements ✅ DESIGN COMPLETE
- [x] Three-tier execution system designed with distinct profiles
- [x] Intelligent query classification and routing designed
- [x] Sophisticated caching strategy per tier designed
- [x] Cost optimization with budget tracking designed
- [x] Performance monitoring and alerting designed
- [x] Mock-first TDD development approach designed

### Performance Targets ✅ DESIGN TARGETS SET
- [x] Tier 1 latency target: < 50ms (P95), 85-90% accuracy
- [x] Tier 2 latency target: < 500ms (P95), 92-95% accuracy  
- [x] Tier 3 latency target: < 2s (P95), 95-97% accuracy
- [x] Cost targets: $0.0001, $0.01, $0.05 per query respectively
- [x] Cache hit rate target: > 70% for Tier 1, > 50% for Tier 2

### Quality Gates ✅ DESIGN COMPLETE
- [x] Mock coverage designed: 100% before implementation
- [x] Classification accuracy designed: > 90% correct tier selection
- [x] Cost optimization designed: > 20% average cost reduction
- [x] SLA compliance designed: < 1% tier target violations
- [x] Budget enforcement designed: 100% budget limit compliance

## Risk Mitigation

### Classification Accuracy Risks
- **Risk**: Poor query classification leads to wrong tier selection
- **Mitigation**: Comprehensive classification training and validation with ground truth

### Cost Control Risks
- **Risk**: Unexpected API cost spikes exceed budgets
- **Mitigation**: Hard budget limits, real-time cost tracking, and automatic tier downgrade

### Performance Risks
- **Risk**: Tier latency targets not met under load
- **Mitigation**: Performance testing, caching optimization, and graceful degradation

## Next Phase
With tiered execution system complete, the full RAG system achieves 95-97% accuracy with cost-optimized performance across all query types.

---

*Phase 6 completes the ultimate RAG system by providing intelligent, cost-optimized query routing that maximizes efficiency while maintaining accuracy guarantees.*