//! Test support utilities for the LLMKG project

pub mod assertions;
pub mod builders;
pub mod data;
pub mod fixtures;
pub mod scenarios;
pub mod test_utils;

pub use builders::{
    AttentionManagerBuilder, 
    CognitivePatternBuilder, 
    CognitiveOrchestratorBuilder, 
    WorkingMemoryBuilder,
    QueryContextBuilder,
    PatternParametersBuilder,
    build_test_cognitive_orchestrator,
    build_test_attention_manager,
    build_test_working_memory,
    build_test_brain_metrics_collector,
    build_test_performance_monitor,
    build_test_neural_server,
    build_test_federation_coordinator,
    build_test_brain_graph
};
pub use assertions::{CognitiveAssertions, PatternAssertions};
pub use data::{TestQueries, PerformanceTestData, create_standard_test_entities};
pub use scenarios::{TestScenario, AttentionScenario, get_test_scenarios, get_attention_scenarios, create_scenario_entities, create_scenario_entities_in_graph};