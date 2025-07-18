// Test stubs for Phase 4 functionality that isn't fully implemented yet
// These are clearly marked as test-only code and don't extend production types

use anyhow::Result;
use std::sync::Arc;
use std::time::Duration;
use llmkg::learning::types::*;
use llmkg::cognitive::phase4_integration::Phase4CognitiveSystem;

/// Test-only wrapper for Phase4LearningSystem to add test methods
/// This is NOT an extension of the production type
pub struct TestPhase4LearningSystem {
    inner: Arc<llmkg::learning::phase4_integration::Phase4LearningSystem>,
}

impl TestPhase4LearningSystem {
    pub fn new(system: Arc<llmkg::learning::phase4_integration::Phase4LearningSystem>) -> Self {
        Self { inner: system }
    }

    /// Test-only method to simulate feedback processing
    pub async fn test_process_feedback(&self, feedback: UserFeedback) -> Result<()> {
        // In a real test, this would use the actual feedback processing API
        // For now, we just log it
        println!("TEST: Processing feedback with satisfaction={:.2}", feedback.satisfaction_score);
        Ok(())
    }

    /// Test-only method to simulate time passage
    pub async fn test_simulate_time(&self, duration: Duration) -> Result<()> {
        // In a real implementation, this might trigger decay processes
        println!("TEST: Simulating {} seconds of time passage", duration.as_secs());
        tokio::time::sleep(Duration::from_millis(10)).await; // Small delay to simulate work
        Ok(())
    }
}

/// Test-only wrapper for AdaptiveLearningSystem
pub struct TestAdaptiveLearningSystem {
    inner: Arc<llmkg::learning::adaptive_learning::AdaptiveLearningSystem>,
}

impl TestAdaptiveLearningSystem {
    pub fn new(system: Arc<llmkg::learning::adaptive_learning::AdaptiveLearningSystem>) -> Self {
        Self { inner: system }
    }

    /// Test-only method to check system health
    pub async fn test_measure_health(&self) -> Result<f32> {
        // Return a mock health value for testing
        // In real implementation, this would aggregate actual metrics
        Ok(0.85) // 85% health
    }

    /// Test-only method to inject performance data
    pub async fn test_inject_performance(&self, performance: f32, satisfaction: f32) -> Result<()> {
        println!("TEST: Injecting performance={:.2}, satisfaction={:.2}", performance, satisfaction);
        Ok(())
    }
}

/// Mock emergency result for testing
#[derive(Debug)]
pub struct MockEmergencyResult {
    pub success: bool,
    pub actions_taken: Vec<String>,
    pub performance_recovery: f32,
    pub time_to_recovery: Duration,
}

/// Test helper to create a minimal Phase4CognitiveSystem
/// This uses actual constructors where possible
pub async fn create_minimal_test_phase4_system() -> Result<Phase4CognitiveSystem> {
    use llmkg::core::{
        brain_enhanced_graph::BrainEnhancedGraph,
        sdr_storage::SDRStorage,
        activation_engine::ActivationPropagationEngine,
    };
    use llmkg::cognitive::{
        working_memory::WorkingMemorySystem,
        attention_manager::AttentionManager,
        inhibitory_logic::CompetitiveInhibitionSystem,
        orchestrator::CognitiveOrchestrator,
        phase3_integration::IntegratedCognitiveSystem,
    };
    use llmkg::learning::phase4_integration::Phase4LearningSystem;

    // Create all required components
    let brain_graph = Arc::new(BrainEnhancedGraph::new().await?);
    let sdr_storage = Arc::new(SDRStorage::new().await?);
    let activation_engine = Arc::new(ActivationPropagationEngine::new().await?);
    let working_memory = Arc::new(WorkingMemorySystem::new().await?);
    let attention_manager = Arc::new(AttentionManager::new().await?);
    let inhibition_system = Arc::new(CompetitiveInhibitionSystem::new().await?);
    
    // Create Phase 3 system
    let phase3_system = Arc::new(IntegratedCognitiveSystem::new(
        brain_graph.clone(),
        sdr_storage.clone(),
        activation_engine.clone(),
        working_memory.clone(),
        attention_manager.clone(),
        inhibition_system.clone(),
    ).await?);
    
    // Create orchestrator
    let orchestrator = Arc::new(CognitiveOrchestrator::new().await?);
    
    // Create Phase 4 learning system
    let phase4_learning = Arc::new(Phase4LearningSystem::new(
        phase3_system.clone(),
        brain_graph,
        sdr_storage,
        activation_engine,
        attention_manager,
        working_memory,
        inhibition_system,
        orchestrator,
    ).await?);
    
    // Create Phase 4 cognitive system
    Phase4CognitiveSystem::new(phase3_system, phase4_learning).await
}

/// Test data builders that don't pollute production code

pub struct TestLearningTask {
    pub name: String,
    pub complexity: f32,
    pub required_patterns: Vec<llmkg::cognitive::types::CognitivePatternType>,
    pub success_threshold: f32,
}

impl TestLearningTask {
    pub fn new(name: &str, complexity: f32) -> Self {
        Self {
            name: name.to_string(),
            complexity,
            required_patterns: vec![],
            success_threshold: 0.7,
        }
    }
}

pub struct TestMetaLearningResult {
    pub task_performance: f32,
    pub insights: Vec<String>,
}

/// Mock implementations for testing complex scenarios
pub mod mocks {
    use super::*;
    
    pub struct MockOptimizationResult {
        pub strategy: String,
        pub efficiency_gain: f32,
        pub success: bool,
    }
    
    pub async fn mock_attempt_optimization(strategy: &str) -> Result<MockOptimizationResult> {
        // Simulate optimization attempt
        tokio::time::sleep(Duration::from_millis(50)).await;
        
        Ok(MockOptimizationResult {
            strategy: strategy.to_string(),
            efficiency_gain: 0.1,
            success: true,
        })
    }
    
    pub async fn mock_emergency_adaptation(
        trigger: llmkg::learning::adaptive_learning::EmergencyTrigger
    ) -> Result<MockEmergencyResult> {
        // Simulate emergency handling
        tokio::time::sleep(Duration::from_millis(100)).await;
        
        Ok(MockEmergencyResult {
            success: true,
            actions_taken: vec![
                "Cleared caches".to_string(),
                "Reduced memory usage".to_string(),
            ],
            performance_recovery: 0.8,
            time_to_recovery: Duration::from_secs(2),
        })
    }
}

/// Test assertions that provide better error messages
#[macro_export]
macro_rules! assert_learning_occurred {
    ($update:expr) => {
        {
            let total_changes = $update.strengthened_connections.len() + 
                               $update.weakened_connections.len() + 
                               $update.new_connections.len();
            assert!(
                total_changes > 0,
                "No learning occurred: 0 connections modified (strengthened={}, weakened={}, new={})",
                $update.strengthened_connections.len(),
                $update.weakened_connections.len(),
                $update.new_connections.len()
            );
        }
    };
}

#[macro_export]
macro_rules! assert_within_range {
    ($value:expr, $min:expr, $max:expr, $name:expr) => {
        assert!(
            $value >= $min && $value <= $max,
            "{} = {} is outside valid range [{}, {}]",
            $name, $value, $min, $max
        );
    };
}