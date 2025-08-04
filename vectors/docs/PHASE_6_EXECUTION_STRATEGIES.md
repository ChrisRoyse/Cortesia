# Phase 6: Execution Strategies - Algorithmic Foundations for Tiered Query Routing

## Overview
This document provides the concrete algorithmic implementations, decision trees, and mathematical formulas that form the foundation of the Phase 6 Tiered Execution System. These strategies enable cost-optimized query routing with accuracy guarantees.

## Critical Gap Addresses

### 1. Query Classification - Specific Heuristics and Patterns

#### Pattern-Based Classification Algorithm
```rust
struct ClassificationRule {
    pattern: Regex,
    weight: f32,
    target_type: QueryType,
    confidence: f32,
}

const CLASSIFICATION_RULES: &[ClassificationRule] = &[
    // Special Characters Detection
    ClassificationRule {
        pattern: regex!(r"[{}[\]().*+?^$|\\&%@#!~`]"),
        weight: 0.9,
        target_type: QueryType::SpecialCharacters,
        confidence: 0.95,
    },
    
    // Boolean Query Detection  
    ClassificationRule {
        pattern: regex!(r"\b(AND|OR|NOT|&|\||!)\b"),
        weight: 0.95,
        target_type: QueryType::BooleanQuery,
        confidence: 0.98,
    },
    
    // Code Structure Detection
    ClassificationRule {
        pattern: regex!(r"\b(fn|function|class|struct|impl|trait|enum|interface|def)\b"),
        weight: 0.85,
        target_type: QueryType::CodeStructure,
        confidence: 0.90,
    },
    
    // Debugging Query Detection
    ClassificationRule {
        pattern: regex!(r"\b(error|exception|crash|bug|debug|troubleshoot|fix|broken|panic|segfault)\b"),
        weight: 0.80,
        target_type: QueryType::Debugging,
        confidence: 0.85,
    },
    
    // API Usage Detection
    ClassificationRule {
        pattern: regex!(r"\b(api|endpoint|method|usage|example|how to use|documentation)\b"),
        weight: 0.75,
        target_type: QueryType::APIUsage,
        confidence: 0.80,
    },
    
    // Performance Analysis Detection
    ClassificationRule {
        pattern: regex!(r"\b(performance|optimization|speed|latency|throughput|benchmark|profiling)\b"),
        weight: 0.70,
        target_type: QueryType::PerformanceAnalysis,
        confidence: 0.75,
    },
    
    // Security Audit Detection
    ClassificationRule {
        pattern: regex!(r"\b(security|vulnerability|exploit|authentication|authorization|sql injection|xss)\b"),
        weight: 0.85,
        target_type: QueryType::SecurityAudit,
        confidence: 0.88,
    },
];
```

#### Feature-Based Classification Scoring
```rust
fn calculate_classification_score(query: &str, features: &QueryFeatures) -> HashMap<QueryType, f32> {
    let mut scores = HashMap::new();
    
    // Pattern matching scores
    for rule in CLASSIFICATION_RULES {
        if rule.pattern.is_match(query) {
            let current_score = scores.entry(rule.target_type).or_insert(0.0);
            *current_score += rule.weight * rule.confidence;
        }
    }
    
    // Feature-based scoring adjustments
    
    // Special character density bonus
    if features.special_char_count > features.word_count {
        let special_chars_score = scores.entry(QueryType::SpecialCharacters).or_insert(0.0);
        *special_chars_score += 0.2;
    }
    
    // Programming keyword density bonus
    if !features.programming_keywords.is_empty() {
        let keyword_density = features.programming_keywords.len() as f32 / features.word_count as f32;
        let code_score = scores.entry(QueryType::CodeStructure).or_insert(0.0);
        *code_score += keyword_density * 0.3;
    }
    
    // File extension bonus
    if !features.file_extensions.is_empty() {
        let code_score = scores.entry(QueryType::CodeStructure).or_insert(0.0);
        *code_score += 0.15;
    }
    
    // Error pattern bonus
    if !features.error_patterns.is_empty() {
        let debug_score = scores.entry(QueryType::Debugging).or_insert(0.0);
        *debug_score += 0.25;
    }
    
    // Question pattern detection for semantic search
    if query.starts_with("what") || query.starts_with("how") || query.starts_with("why") {
        let semantic_score = scores.entry(QueryType::SemanticSearch).or_insert(0.0);
        *semantic_score += 0.2;
    }
    
    scores
}
```

#### Complexity Assessment Algorithm
```rust
fn assess_query_complexity(query: &str, features: &QueryFeatures, query_type: &QueryType) -> QueryComplexity {
    let mut complexity_score = 0.0;
    
    // Base complexity from query length and word count
    complexity_score += (query.len() as f32 / 100.0).min(0.3);
    complexity_score += (features.word_count as f32 / 20.0).min(0.2);
    
    // Boolean operators increase complexity
    complexity_score += features.boolean_operators.len() as f32 * 0.1;
    
    // Special characters can increase complexity
    if features.special_char_count > 5 {
        complexity_score += 0.15;
    }
    
    // Question words suggest complex semantic queries
    let question_words = ["what", "how", "why", "when", "where", "which"];
    for word in question_words {
        if query.to_lowercase().contains(word) {
            complexity_score += 0.1;
        }
    }
    
    // Query type modifiers
    complexity_score += match query_type {
        QueryType::SpecialCharacters => 0.0,     // Usually simple
        QueryType::BooleanQuery => 0.1,          // Moderately complex
        QueryType::SemanticSearch => 0.2,        // More complex
        QueryType::CodeStructure => 0.15,        // Moderately complex
        QueryType::Debugging => 0.3,             // Very complex
        QueryType::APIUsage => 0.1,              // Usually straightforward
        QueryType::PerformanceAnalysis => 0.25,  // Quite complex
        QueryType::SecurityAudit => 0.3,         // Very complex
    };
    
    // Programming language complexity
    if features.programming_keywords.len() > 3 {
        complexity_score += 0.1;
    }
    
    // Convert to discrete complexity levels
    if complexity_score < 0.3 {
        QueryComplexity::Simple
    } else if complexity_score < 0.7 {
        QueryComplexity::Medium
    } else {
        QueryComplexity::Complex
    }
}
```

### 2. Cache Invalidation - TTL and Event-Based Strategies

#### TTL-Based Cache Policy
```rust
struct TierCachePolicy {
    tier: ExecutionTier,
    base_ttl: Duration,
    dynamic_ttl_factors: Vec<TtlFactor>,
    max_ttl: Duration,
    min_ttl: Duration,
}

struct TtlFactor {
    condition: TtlCondition,
    multiplier: f32,
}

enum TtlCondition {
    HighConfidence(f32),      // confidence > threshold
    LowConfidence(f32),       // confidence < threshold
    QueryType(QueryType),     // specific query types
    AccessFrequency(u32),     // accesses > threshold
    ResultStability(f32),     // result consistency score
}

const TIER_CACHE_POLICIES: [TierCachePolicy; 3] = [
    // Tier 1: Fast local search - aggressive caching
    TierCachePolicy {
        tier: ExecutionTier::Tier1Fast,
        base_ttl: Duration::from_secs(3600), // 1 hour
        dynamic_ttl_factors: vec![
            TtlFactor { condition: TtlCondition::HighConfidence(0.9), multiplier: 2.0 },
            TtlFactor { condition: TtlCondition::LowConfidence(0.7), multiplier: 0.5 },
            TtlFactor { condition: TtlCondition::QueryType(QueryType::SpecialCharacters), multiplier: 3.0 },
            TtlFactor { condition: TtlCondition::AccessFrequency(10), multiplier: 1.5 },
        ],
        max_ttl: Duration::from_secs(86400), // 24 hours
        min_ttl: Duration::from_secs(300),   // 5 minutes
    },
    
    // Tier 2: Balanced hybrid search - moderate caching
    TierCachePolicy {
        tier: ExecutionTier::Tier2Balanced,
        base_ttl: Duration::from_secs(1800), // 30 minutes
        dynamic_ttl_factors: vec![
            TtlFactor { condition: TtlCondition::HighConfidence(0.95), multiplier: 1.5 },
            TtlFactor { condition: TtlCondition::LowConfidence(0.8), multiplier: 0.6 },
            TtlFactor { condition: TtlCondition::QueryType(QueryType::SemanticSearch), multiplier: 0.8 },
            TtlFactor { condition: TtlCondition::ResultStability(0.9), multiplier: 1.3 },
        ],
        max_ttl: Duration::from_secs(7200),  // 2 hours
        min_ttl: Duration::from_secs(180),   // 3 minutes
    },
    
    // Tier 3: Deep analysis - conservative caching
    TierCachePolicy {
        tier: ExecutionTier::Tier3Deep,
        base_ttl: Duration::from_secs(600),  // 10 minutes
        dynamic_ttl_factors: vec![
            TtlFactor { condition: TtlCondition::HighConfidence(0.97), multiplier: 1.2 },
            TtlFactor { condition: TtlCondition::LowConfidence(0.9), multiplier: 0.7 },
            TtlFactor { condition: TtlCondition::QueryType(QueryType::Debugging), multiplier: 0.5 },
            TtlFactor { condition: TtlCondition::QueryType(QueryType::SecurityAudit), multiplier: 0.3 },
        ],
        max_ttl: Duration::from_secs(1800),  // 30 minutes
        min_ttl: Duration::from_secs(60),    // 1 minute
    },
];
```

#### Dynamic TTL Calculation
```rust
fn calculate_dynamic_ttl(
    result: &TieredResult, 
    policy: &TierCachePolicy,
    access_history: &AccessHistory
) -> Duration {
    let mut ttl_multiplier = 1.0;
    
    for factor in &policy.dynamic_ttl_factors {
        match &factor.condition {
            TtlCondition::HighConfidence(threshold) => {
                if result.accuracy_estimate >= *threshold {
                    ttl_multiplier *= factor.multiplier;
                }
            },
            TtlCondition::LowConfidence(threshold) => {
                if result.accuracy_estimate < *threshold {
                    ttl_multiplier *= factor.multiplier;
                }
            },
            TtlCondition::QueryType(query_type) => {
                if std::mem::discriminant(&result.query_type) == std::mem::discriminant(query_type) {
                    ttl_multiplier *= factor.multiplier;
                }
            },
            TtlCondition::AccessFrequency(threshold) => {
                if access_history.access_count >= *threshold {
                    ttl_multiplier *= factor.multiplier;
                }
            },
            TtlCondition::ResultStability(threshold) => {
                if access_history.stability_score >= *threshold {
                    ttl_multiplier *= factor.multiplier;
                }
            },
        }
    }
    
    let calculated_ttl = Duration::from_secs_f32(policy.base_ttl.as_secs_f32() * ttl_multiplier);
    
    // Clamp to min/max bounds
    calculated_ttl.clamp(policy.min_ttl, policy.max_ttl)
}
```

#### Event-Based Cache Invalidation
```rust
enum CacheInvalidationEvent {
    FileModified { path: String, timestamp: u64 },
    CodebaseUpdate { commit_hash: String, affected_files: Vec<String> },
    DependencyUpdate { package: String, old_version: String, new_version: String },
    ConfigurationChange { component: String, change_type: String },
    QueryPatternChange { pattern: String, confidence_drop: f32 },
    UserFeedback { query: String, rating: i8 },
}

struct EventBasedInvalidator {
    invalidation_rules: Vec<InvalidationRule>,
}

struct InvalidationRule {
    event_pattern: EventPattern,
    cache_targets: CacheTargetPattern,
    confidence_threshold: f32,
}

enum EventPattern {
    FilePathPattern(Regex),
    QueryContentPattern(Regex),
    AccuracyDropPattern(f32),
    TimeWindow(Duration),
}

enum CacheTargetPattern {
    AllTiers,
    SpecificTier(ExecutionTier),
    QueryTypePattern(QueryType),
    CacheKeyPattern(Regex),
}

impl EventBasedInvalidator {
    fn process_event(&self, event: CacheInvalidationEvent, cache_manager: &mut TieredCacheManager) {
        match event {
            CacheInvalidationEvent::FileModified { path, .. } => {
                // Invalidate caches for queries that might reference this file
                let pattern = format!(".*{}.*", regex::escape(&path));
                cache_manager.invalidate_by_pattern(&pattern);
            },
            
            CacheInvalidationEvent::CodebaseUpdate { affected_files, .. } => {
                // Invalidate code structure and debugging related caches
                for file in affected_files {
                    cache_manager.invalidate_by_file_reference(&file);
                }
                cache_manager.invalidate_by_query_type(QueryType::CodeStructure);
                cache_manager.invalidate_by_query_type(QueryType::Debugging);
            },
            
            CacheInvalidationEvent::DependencyUpdate { package, .. } => {
                // Invalidate API usage and dependency-related caches
                let pattern = format!(".*{}.*", regex::escape(&package));
                cache_manager.invalidate_by_pattern(&pattern);
                cache_manager.invalidate_by_query_type(QueryType::APIUsage);
            },
            
            CacheInvalidationEvent::QueryPatternChange { confidence_drop, .. } => {
                // Invalidate caches with confidence below threshold
                if confidence_drop > 0.1 {
                    cache_manager.invalidate_by_confidence_threshold(0.9 - confidence_drop);
                }
            },
            
            CacheInvalidationEvent::UserFeedback { query, rating } => {
                // Invalidate specific query cache if feedback is negative
                if rating < 0 {
                    cache_manager.invalidate_specific_query(&query);
                }
            },
            
            _ => {} // Handle other event types
        }
    }
}
```

### 3. Tier Transitions - When to Upgrade/Downgrade Queries

#### Tier Selection Decision Tree
```
Initial Tier Selection:
├── Query Complexity == Simple
│   ├── Confidence Required < 90%
│   │   └── SELECT Tier1Fast
│   └── Confidence Required >= 90%
│       ├── User Budget < $0.005
│       │   └── SELECT Tier1Fast (with warning)
│       └── User Budget >= $0.005
│           └── SELECT Tier2Balanced
├── Query Complexity == Medium
│   ├── User Preference == FastestResponse
│   │   └── SELECT Tier1Fast
│   ├── User Preference == LowestCost
│   │   ├── Budget < $0.01
│   │   │   └── SELECT Tier1Fast
│   │   └── Budget >= $0.01
│   │       └── SELECT Tier2Balanced
│   └── User Preference == BestAccuracy || Balanced
│       └── SELECT Tier2Balanced
└── Query Complexity == Complex
    ├── User Preference == FastestResponse && Accuracy Required < 85%
    │   └── SELECT Tier1Fast (with accuracy warning)
    ├── User Preference == LowestCost && Budget < $0.01
    │   └── SELECT Tier2Balanced (compromise)
    └── Default
        └── SELECT Tier3Deep
```

#### Tier Escalation Algorithm
```rust
struct EscalationTrigger {
    condition: EscalationCondition,
    target_tier: ExecutionTier,
    confidence_threshold: f32,
}

enum EscalationCondition {
    LowConfidence(f32),              // Result confidence below threshold
    EmptyResults,                    // No results found
    UserDissatisfaction(i8),         // User feedback below threshold
    DebugQueryDetected,              // Query contains debug patterns
    SecurityQueryDetected,           // Query contains security patterns
    HighStakeQuery,                  // Query marked as high importance
    PreviousEscalationSuccessful,    // Pattern of successful escalations
}

const ESCALATION_RULES: &[EscalationTrigger] = &[
    EscalationTrigger {
        condition: EscalationCondition::LowConfidence(0.7),
        target_tier: ExecutionTier::Tier2Balanced,
        confidence_threshold: 0.9,
    },
    EscalationTrigger {
        condition: EscalationCondition::LowConfidence(0.85),
        target_tier: ExecutionTier::Tier3Deep,
        confidence_threshold: 0.95,
    },
    EscalationTrigger {
        condition: EscalationCondition::EmptyResults,
        target_tier: ExecutionTier::Tier2Balanced,
        confidence_threshold: 0.8,
    },
    EscalationTrigger {
        condition: EscalationCondition::DebugQueryDetected,
        target_tier: ExecutionTier::Tier3Deep,
        confidence_threshold: 0.95,
    },
    EscalationTrigger {
        condition: EscalationCondition::SecurityQueryDetected,
        target_tier: ExecutionTier::Tier3Deep,
        confidence_threshold: 0.97,
    },
];

fn should_escalate_tier(
    current_result: &TieredResult,
    user_context: &UserContext,
    query_analysis: &QueryClassification
) -> Option<ExecutionTier> {
    
    for rule in ESCALATION_RULES {
        let should_escalate = match &rule.condition {
            EscalationCondition::LowConfidence(threshold) => {
                current_result.accuracy_estimate < *threshold
            },
            EscalationCondition::EmptyResults => {
                current_result.results.is_empty()
            },
            EscalationCondition::UserDissatisfaction(min_rating) => {
                user_context.last_feedback_rating.unwrap_or(5) < *min_rating
            },
            EscalationCondition::DebugQueryDetected => {
                query_analysis.query_type == QueryType::Debugging
            },
            EscalationCondition::SecurityQueryDetected => {
                query_analysis.query_type == QueryType::SecurityAudit
            },
            EscalationCondition::HighStakeQuery => {
                user_context.query_importance == QueryImportance::High
            },
            EscalationCondition::PreviousEscalationSuccessful => {
                user_context.escalation_success_rate > 0.8 && 
                user_context.total_escalations > 5
            },
        };
        
        if should_escalate {
            // Check if target tier is actually higher than current
            let current_tier_level = tier_to_level(current_result.tier_used);
            let target_tier_level = tier_to_level(rule.target_tier);
            
            if target_tier_level > current_tier_level {
                // Check budget constraints
                if check_budget_constraint(user_context, rule.target_tier) {
                    return Some(rule.target_tier);
                }
            }
        }
    }
    
    None
}
```

#### Tier Downgrade Decision Logic
```rust
struct DowngradeOpportunity {
    condition: DowngradeCondition,
    target_tier: ExecutionTier,
    accuracy_impact: f32,
    cost_savings: u32,
}

enum DowngradeCondition {
    OverlyHighConfidence(f32),       // Confidence much higher than needed
    SimpleQueryPattern,              // Query pattern suggests simpler search adequate
    BudgetConstraint(u32),           // User approaching budget limit
    PerformancePreference,           // User prioritizes speed over accuracy
    CacheHitPredicted(f32),         // High probability of cache hit in lower tier
}

const DOWNGRADE_OPPORTUNITIES: &[DowngradeOpportunity] = &[
    DowngradeOpportunity {
        condition: DowngradeCondition::OverlyHighConfidence(0.95),
        target_tier: ExecutionTier::Tier1Fast,
        accuracy_impact: -0.05, // 5% accuracy loss expected
        cost_savings: 3, // 3 cents saved
    },
    DowngradeOpportunity {
        condition: DowngradeCondition::SimpleQueryPattern,
        target_tier: ExecutionTier::Tier1Fast,
        accuracy_impact: -0.03,
        cost_savings: 2,
    },
    DowngradeOpportunity {
        condition: DowngradeCondition::CacheHitPredicted(0.8),
        target_tier: ExecutionTier::Tier1Fast,
        accuracy_impact: 0.0, // No impact if cache hit
        cost_savings: 5, // Major savings from cache hit
    },
];

fn evaluate_downgrade_opportunity(
    planned_tier: ExecutionTier,
    query_classification: &QueryClassification,
    user_context: &UserContext,
    cache_predictor: &CachePredictor
) -> Option<(ExecutionTier, String)> {
    
    if planned_tier == ExecutionTier::Tier1Fast {
        return None; // Already at lowest tier
    }
    
    for opportunity in DOWNGRADE_OPPORTUNITIES {
        let should_downgrade = match &opportunity.condition {
            DowngradeCondition::OverlyHighConfidence(threshold) => {
                // If we expect much higher confidence than needed
                user_context.min_accuracy.unwrap_or(0.85) < *threshold - 0.1
            },
            DowngradeCondition::SimpleQueryPattern => {
                query_classification.complexity == QueryComplexity::Simple &&
                query_classification.query_type == QueryType::SpecialCharacters
            },
            DowngradeCondition::BudgetConstraint(threshold) => {
                user_context.cost_budget_remaining < *threshold
            },
            DowngradeCondition::PerformancePreference => {
                user_context.performance_preference == PerformancePreference::FastestResponse
            },
            DowngradeCondition::CacheHitPredicted(probability) => {
                cache_predictor.predict_cache_hit_probability(
                    &user_context.current_query, 
                    opportunity.target_tier
                ) > *probability
            },
        };
        
        if should_downgrade {
            let reason = format!(
                "Downgrade to {:?}: {} accuracy impact, {}¢ cost savings", 
                opportunity.target_tier,
                if opportunity.accuracy_impact < 0.0 { 
                    format!("{:.1}%", opportunity.accuracy_impact * 100.0) 
                } else { 
                    "no".to_string() 
                },
                opportunity.cost_savings
            );
            
            return Some((opportunity.target_tier, reason));
        }
    }
    
    None
}
```

### 4. Cost Tracking - Real-Time Cost Monitoring

#### Real-Time Cost Calculation
```rust
struct CostTracker {
    pricing_model: PricingModel,
    active_queries: HashMap<String, QueryCostState>,
    user_budgets: HashMap<String, UserBudgetState>,
    global_cost_state: GlobalCostState,
}

struct QueryCostState {
    query_id: String,
    user_id: String,
    tier: ExecutionTier,
    start_time: Instant,
    estimated_cost: u32,
    actual_cost: u32,
    cost_components: CostComponents,
}

struct CostComponents {
    api_calls: u32,           // cents
    compute_time: u32,        // cents
    storage_access: u32,      // cents
    cache_operations: u32,    // cents
    synthesis_processing: u32, // cents
}

struct UserBudgetState {
    user_id: String,
    current_period_usage: u32,
    daily_limit: u32,
    monthly_limit: u32,
    alert_thresholds: Vec<AlertThreshold>,
    cost_history: VecDeque<CostEvent>,
}

struct AlertThreshold {
    percentage: f32,
    alert_type: AlertType,
    notification_sent: bool,
}

enum AlertType {
    Warning,
    Critical,
    BudgetExceeded,
}

impl CostTracker {
    fn start_query_cost_tracking(&mut self, 
        query_id: &str, 
        user_id: &str, 
        tier: ExecutionTier,
        estimated_operations: &EstimatedOperations
    ) -> anyhow::Result<()> {
        
        let estimated_cost = self.calculate_estimated_cost(tier, estimated_operations);
        
        // Check if user can afford this query
        let user_budget = self.user_budgets.get(user_id).ok_or_else(|| 
            anyhow::anyhow!("User budget not found: {}", user_id)
        )?;
        
        if user_budget.current_period_usage + estimated_cost > user_budget.daily_limit {
            return Err(anyhow::anyhow!("Query would exceed daily budget limit"));
        }
        
        let query_state = QueryCostState {
            query_id: query_id.to_string(),
            user_id: user_id.to_string(),
            tier,
            start_time: Instant::now(),
            estimated_cost,
            actual_cost: 0,
            cost_components: CostComponents::default(),
        };
        
        self.active_queries.insert(query_id.to_string(), query_state);
        
        // Reserve estimated cost
        if let Some(user_budget) = self.user_budgets.get_mut(user_id) {
            user_budget.current_period_usage += estimated_cost;
        }
        
        Ok(())
    }
    
    fn update_query_cost(&mut self, 
        query_id: &str, 
        cost_component: CostComponentUpdate
    ) -> anyhow::Result<()> {
        
        let query_state = self.active_queries.get_mut(query_id).ok_or_else(|| 
            anyhow::anyhow!("Query not found in cost tracking: {}", query_id)
        )?;
        
        // Update specific cost component
        match cost_component {
            CostComponentUpdate::ApiCall { service, cost } => {
                query_state.cost_components.api_calls += cost;
                query_state.actual_cost += cost;
            },
            CostComponentUpdate::ComputeTime { duration_ms, cost_per_ms } => {
                let compute_cost = (duration_ms as f32 * cost_per_ms) as u32;
                query_state.cost_components.compute_time += compute_cost;
                query_state.actual_cost += compute_cost;
            },
            CostComponentUpdate::StorageAccess { bytes_accessed, cost } => {
                query_state.cost_components.storage_access += cost;
                query_state.actual_cost += cost;
            },
            CostComponentUpdate::CacheOperation { operation_type, cost } => {
                query_state.cost_components.cache_operations += cost;
                query_state.actual_cost += cost;
            },
        }
        
        // Check for cost overruns
        if query_state.actual_cost > query_state.estimated_cost * 2 {
            self.trigger_cost_overrun_alert(query_id)?;
        }
        
        Ok(())
    }
    
    fn finalize_query_cost(&mut self, query_id: &str) -> anyhow::Result<CostSummary> {
        let query_state = self.active_queries.remove(query_id).ok_or_else(|| 
            anyhow::anyhow!("Query not found: {}", query_id)
        )?;
        
        let duration = query_state.start_time.elapsed();
        
        // Update user budget with actual cost
        if let Some(user_budget) = self.user_budgets.get_mut(&query_state.user_id) {
            // Remove estimated cost reservation and add actual cost
            user_budget.current_period_usage = user_budget.current_period_usage
                .saturating_sub(query_state.estimated_cost)
                .saturating_add(query_state.actual_cost);
            
            // Record cost event
            user_budget.cost_history.push_back(CostEvent {
                timestamp: Instant::now(),
                query_id: query_id.to_string(),
                tier: query_state.tier,
                estimated_cost: query_state.estimated_cost,
                actual_cost: query_state.actual_cost,
                duration,
            });
            
            // Trim history to last 1000 events
            if user_budget.cost_history.len() > 1000 {
                user_budget.cost_history.pop_front();
            }
            
            // Check alert thresholds
            self.check_budget_alerts(&query_state.user_id)?;
        }
        
        // Update global cost tracking
        self.global_cost_state.total_queries += 1;
        self.global_cost_state.total_cost += query_state.actual_cost;
        
        Ok(CostSummary {
            query_id: query_id.to_string(),
            tier: query_state.tier,
            estimated_cost: query_state.estimated_cost,
            actual_cost: query_state.actual_cost,
            cost_components: query_state.cost_components,
            duration,
            cost_efficiency: query_state.estimated_cost as f32 / query_state.actual_cost.max(1) as f32,
        })
    }
    
    fn calculate_estimated_cost(&self, tier: ExecutionTier, operations: &EstimatedOperations) -> u32 {
        let base_cost = match tier {
            ExecutionTier::Tier1Fast => 0,      // Local operations only
            ExecutionTier::Tier2Balanced => 1,  // Minimal API usage
            ExecutionTier::Tier3Deep => 3,      // Full synthesis pipeline
        };
        
        let api_cost = operations.api_calls * self.pricing_model.api_cost_per_call;
        let compute_cost = operations.estimated_compute_ms * self.pricing_model.compute_cost_per_ms;
        let storage_cost = operations.estimated_storage_gb * self.pricing_model.storage_cost_per_gb;
        
        base_cost + api_cost + compute_cost + storage_cost
    }
    
    fn check_budget_alerts(&mut self, user_id: &str) -> anyhow::Result<()> {
        let user_budget = self.user_budgets.get_mut(user_id).ok_or_else(|| 
            anyhow::anyhow!("User budget not found: {}", user_id)
        )?;
        
        let usage_percentage = user_budget.current_period_usage as f32 / user_budget.daily_limit as f32;
        
        for threshold in &mut user_budget.alert_thresholds {
            if usage_percentage >= threshold.percentage && !threshold.notification_sent {
                self.send_budget_alert(user_id, &threshold.alert_type, usage_percentage)?;
                threshold.notification_sent = true;
            }
        }
        
        Ok(())
    }
    
    fn get_real_time_cost_analytics(&self, user_id: &str) -> anyhow::Result<RealTimeCostAnalytics> {
        let user_budget = self.user_budgets.get(user_id).ok_or_else(|| 
            anyhow::anyhow!("User budget not found: {}", user_id)
        )?;
        
        let current_burn_rate = self.calculate_burn_rate(user_id);
        let projected_daily_cost = self.project_daily_cost(user_id);
        let cost_per_query_avg = self.calculate_avg_query_cost(user_id);
        
        Ok(RealTimeCostAnalytics {
            current_usage: user_budget.current_period_usage,
            daily_limit: user_budget.daily_limit,
            usage_percentage: user_budget.current_period_usage as f32 / user_budget.daily_limit as f32,
            current_burn_rate,
            projected_daily_cost,
            cost_per_query_avg,
            queries_executed_today: user_budget.cost_history.len(),
            tier_cost_breakdown: self.calculate_tier_cost_breakdown(user_id),
            time_until_budget_exhausted: self.calculate_time_until_budget_exhausted(user_id),
        })
    }
}
```

#### Cost Optimization Triggers
```rust
struct CostOptimizationTrigger {
    condition: CostCondition,
    optimization_action: OptimizationAction,
    threshold: CostThreshold,
}

enum CostCondition {
    BurnRateExceeded(f32),           // Cost per hour exceeds threshold
    QueryCostSpike(f32),             // Individual query cost spike
    BudgetPercentageReached(f32),    // Percentage of budget consumed
    TierCostEfficiencyLow(f32),      // Cost vs accuracy ratio poor
    ApiCostAnomaly(f32),            // Unexpected API cost increase
}

enum OptimizationAction {
    DowngradeTier,                   // Switch to lower tier
    EnableAggressiveCaching,         // Increase cache hit rate
    BatchSimilarQueries,            // Group related queries
    ReduceApiCalls,                 // Minimize external API usage
    AlertUser,                      // Notify user of cost situation
    EnforceBudgetLimit,             // Hard stop on spending
}

const COST_OPTIMIZATION_TRIGGERS: &[CostOptimizationTrigger] = &[
    CostOptimizationTrigger {
        condition: CostCondition::BurnRateExceeded(2.0), // $0.02/hour
        optimization_action: OptimizationAction::DowngradeTier,
        threshold: CostThreshold { value: 2.0, window: Duration::from_secs(3600) },
    },
    CostOptimizationTrigger {
        condition: CostCondition::BudgetPercentageReached(0.8), // 80% of budget
        optimization_action: OptimizationAction::EnableAggressiveCaching,
        threshold: CostThreshold { value: 0.8, window: Duration::from_secs(86400) },
    },
    CostOptimizationTrigger {
        condition: CostCondition::BudgetPercentageReached(0.95), // 95% of budget
        optimization_action: OptimizationAction::EnforceBudgetLimit,
        threshold: CostThreshold { value: 0.95, window: Duration::from_secs(86400) },
    },
];
```

## Implementation Priority

### High Priority (Phase 6 Core)
1. **Query Classification System**: Pattern matching and feature extraction
2. **Basic Tier Selection**: Simple decision tree for tier routing
3. **TTL-Based Caching**: Time-based cache invalidation
4. **Real-Time Cost Tracking**: Basic cost monitoring and budget enforcement

### Medium Priority (Phase 6 Enhancement)
1. **Advanced Tier Transitions**: Escalation and downgrade logic
2. **Event-Based Cache Invalidation**: Intelligent cache management
3. **Cost Optimization Triggers**: Automated cost control
4. **Performance Prediction**: Machine learning for tier selection

### Low Priority (Phase 6 Optimization)
1. **Adaptive Classification**: Learning-based query classification
2. **Predictive Caching**: Proactive cache warming
3. **Dynamic Pricing**: Cost model adaptation
4. **Advanced Analytics**: Complex cost and performance analytics

## Mathematical Constants and Thresholds

```rust
// Query classification parameters
pub const CLASSIFICATION_CONFIDENCE_THRESHOLD: f32 = 0.7;
pub const FEATURE_WEIGHT_DECAY: f32 = 0.95;
pub const COMPLEXITY_SCORE_SIMPLE_THRESHOLD: f32 = 0.3;
pub const COMPLEXITY_SCORE_COMPLEX_THRESHOLD: f32 = 0.7;

// Cache management constants
pub const CACHE_HIT_PREDICTION_ACCURACY: f32 = 0.85;
pub const DYNAMIC_TTL_MIN_MULTIPLIER: f32 = 0.3;
pub const DYNAMIC_TTL_MAX_MULTIPLIER: f32 = 3.0;
pub const RESULT_STABILITY_THRESHOLD: f32 = 0.9;

// Cost tracking parameters
pub const COST_OVERRUN_ALERT_MULTIPLIER: f32 = 2.0;
pub const BUDGET_WARNING_THRESHOLD: f32 = 0.8;
pub const BUDGET_CRITICAL_THRESHOLD: f32 = 0.95;
pub const BURN_RATE_CALCULATION_WINDOW_HOURS: u64 = 4;

// Tier transition thresholds
pub const ESCALATION_CONFIDENCE_THRESHOLD: f32 = 0.7;
pub const DOWNGRADE_OPPORTUNITY_THRESHOLD: f32 = 0.95;
pub const TIER_EFFICIENCY_MIN_THRESHOLD: f32 = 0.8;
pub const CACHE_HIT_PROBABILITY_THRESHOLD: f32 = 0.8;
```

This comprehensive algorithmic foundation enables the Phase 6 Tiered Execution System to make intelligent, cost-aware routing decisions while maintaining performance and accuracy guarantees across all query types and user preferences.