//! Test support utilities for the LLMKG project

pub mod assertions;
pub mod builders;
pub mod data;
pub mod fixtures;
pub mod scenarios;

pub use builders::{
    AttentionManagerBuilder, 
    CognitivePatternBuilder, 
    CognitiveOrchestratorBuilder, 
    WorkingMemoryBuilder,
    QueryContextBuilder,
    PatternParametersBuilder
};
pub use assertions::{CognitiveAssertions, PatternAssertions};
pub use data::{TestQueries, PerformanceTestData, create_standard_test_entities};
pub use scenarios::{TestScenario, AttentionScenario, get_test_scenarios, get_attention_scenarios, create_scenario_entities, create_scenario_entities_in_graph};