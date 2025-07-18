//! Learning coordination and orchestration for Phase 4 Learning System

use super::types::*;
use super::emergency::EmergencyProtocols;
use crate::learning::types::*;
use std::collections::HashMap;
use std::time::SystemTime;
use uuid::Uuid;
use anyhow::Result;

/// Central coordinator for all learning activities
#[derive(Debug, Clone)]
pub struct LearningCoordinator {
    pub active_learning_sessions: HashMap<Uuid, ActiveLearningSession>,
    pub learning_schedule: LearningSchedule,
    pub coordination_state: CoordinationState,
    pub emergency_protocols: EmergencyProtocols,
}

impl LearningCoordinator {
    /// Create new learning coordinator
    pub fn new() -> Self {
        Self {
            active_learning_sessions: HashMap::new(),
            learning_schedule: LearningSchedule {
                scheduled_sessions: Vec::new(),
                recurring_schedules: Vec::new(),
                conditional_triggers: Vec::new(),
            },
            coordination_state: CoordinationState {
                current_coordination_mode: CoordinationMode::Balanced,
                coordination_effectiveness: 0.8,
                inter_system_communication_quality: 0.7,
                learning_coherence_score: 0.6,
                last_coordination_update: SystemTime::now(),
            },
            emergency_protocols: EmergencyProtocols::new(),
        }
    }
    
    /// Start a new learning session
    pub fn start_learning_session(
        &mut self,
        session_type: LearningSessionType,
        participants: Vec<LearningParticipant>,
        resources: ResourceAllocation,
    ) -> Result<Uuid> {
        let session_id = Uuid::new_v4();
        
        let session = ActiveLearningSession {
            session_id,
            session_type,
            start_time: SystemTime::now(),
            expected_duration: resources.time_budget,
            participants,
            progress: LearningProgress {
                completion_percentage: 0.0,
                milestones_achieved: Vec::new(),
                current_phase: "Initialization".to_string(),
                estimated_remaining_time: resources.time_budget,
                performance_impact_so_far: 0.0,
            },
            resources_allocated: resources,
        };
        
        self.active_learning_sessions.insert(session_id, session);
        Ok(session_id)
    }
    
    /// Update progress for a learning session
    pub fn update_session_progress(
        &mut self,
        session_id: Uuid,
        progress: LearningProgress,
    ) -> Result<()> {
        if let Some(session) = self.active_learning_sessions.get_mut(&session_id) {
            session.progress = progress;
            Ok(())
        } else {
            Err(anyhow::anyhow!("Session not found: {}", session_id))
        }
    }
    
    /// Complete a learning session
    pub fn complete_session(&mut self, session_id: Uuid) -> Result<ActiveLearningSession> {
        self.active_learning_sessions
            .remove(&session_id)
            .ok_or_else(|| anyhow::anyhow!("Session not found: {}", session_id))
    }
    
    /// Get coordination strategy for current state
    pub fn get_coordination_strategy(&self, assessment: &SystemAssessment) -> LearningStrategy {
        let strategy_type = if assessment.overall_health < 0.5 {
            StrategyType::Emergency
        } else if assessment.readiness_for_learning > 0.8 {
            StrategyType::Aggressive
        } else if assessment.risk_factors.len() > 3 {
            StrategyType::Conservative
        } else {
            StrategyType::Balanced
        };
        
        let coordination_approach = match strategy_type {
            StrategyType::Emergency => CoordinationApproach::Emergency,
            StrategyType::Aggressive => CoordinationApproach::Parallel,
            StrategyType::Conservative => CoordinationApproach::Sequential,
            _ => CoordinationApproach::Synchronized,
        };
        
        LearningStrategy {
            strategy_type,
            priority_areas: assessment.learning_opportunities.clone(),
            resource_allocation: self.calculate_resource_allocation(&coordination_approach),
            coordination_approach: coordination_approach.clone(),
            safety_level: self.calculate_safety_level(assessment),
            expected_duration: self.estimate_duration(&coordination_approach),
        }
    }
    
    /// Calculate resource allocation based on coordination approach
    fn calculate_resource_allocation(&self, approach: &CoordinationApproach) -> crate::learning::adaptive_learning::ResourceRequirement {
        match approach {
            CoordinationApproach::Emergency => crate::learning::adaptive_learning::ResourceRequirement {
                memory_mb: 256.0,
                cpu_cores: 0.5,
                storage_mb: 50.0,
                network_bandwidth_mbps: 5.0,
            },
            CoordinationApproach::Parallel => crate::learning::adaptive_learning::ResourceRequirement {
                memory_mb: 1024.0,
                cpu_cores: 2.0,
                storage_mb: 200.0,
                network_bandwidth_mbps: 20.0,
            },
            CoordinationApproach::Sequential => crate::learning::adaptive_learning::ResourceRequirement {
                memory_mb: 512.0,
                cpu_cores: 1.0,
                storage_mb: 100.0,
                network_bandwidth_mbps: 10.0,
            },
            CoordinationApproach::Synchronized => crate::learning::adaptive_learning::ResourceRequirement {
                memory_mb: 768.0,
                cpu_cores: 1.5,
                storage_mb: 150.0,
                network_bandwidth_mbps: 15.0,
            },
        }
    }
    
    /// Calculate safety level based on system assessment
    fn calculate_safety_level(&self, assessment: &SystemAssessment) -> f32 {
        let base_safety = 0.8;
        let health_factor = assessment.overall_health * 0.2;
        let risk_factor = (5 - assessment.risk_factors.len().min(5)) as f32 * 0.04;
        
        (base_safety + health_factor + risk_factor).min(1.0)
    }
    
    /// Estimate duration based on coordination approach
    fn estimate_duration(&self, approach: &CoordinationApproach) -> std::time::Duration {
        match approach {
            CoordinationApproach::Emergency => std::time::Duration::from_secs(60),
            CoordinationApproach::Parallel => std::time::Duration::from_secs(300),
            CoordinationApproach::Sequential => std::time::Duration::from_secs(600),
            CoordinationApproach::Synchronized => std::time::Duration::from_secs(450),
        }
    }
    
    /// Execute coordination for learning session
    pub fn execute_coordination(
        &mut self,
        strategy: &LearningStrategy,
        session_id: Uuid,
    ) -> Result<CoordinationResult> {
        // Determine participants based on strategy
        let participants = self.select_participants(strategy);
        
        // Update coordination state
        self.coordination_state.current_coordination_mode = match strategy.strategy_type {
            StrategyType::Emergency => CoordinationMode::EmergencyMode,
            StrategyType::Conservative => CoordinationMode::ConservationMode,
            StrategyType::Aggressive => CoordinationMode::OptimizationFocused,
            _ => CoordinationMode::Balanced,
        };
        
        self.coordination_state.last_coordination_update = SystemTime::now();
        
        // Generate synchronization points
        let synchronization_points = self.generate_synchronization_points(&strategy.coordination_approach);
        
        Ok(CoordinationResult {
            session_id,
            coordination_mode: self.coordination_state.current_coordination_mode.clone(),
            participants_activated: participants,
            resource_allocation: strategy.resource_allocation.clone(),
            synchronization_points,
        })
    }
    
    /// Select appropriate participants for learning strategy
    fn select_participants(&self, strategy: &LearningStrategy) -> Vec<LearningParticipant> {
        let mut participants = Vec::new();
        
        match strategy.strategy_type {
            StrategyType::Emergency => {
                participants.push(LearningParticipant::CognitiveOrchestrator);
                participants.push(LearningParticipant::HomeostasisSystem);
            },
            StrategyType::Aggressive => {
                participants.push(LearningParticipant::HebbianEngine);
                participants.push(LearningParticipant::OptimizationAgent);
                participants.push(LearningParticipant::AdaptiveLearning);
            },
            StrategyType::Conservative => {
                participants.push(LearningParticipant::HomeostasisSystem);
                participants.push(LearningParticipant::AdaptiveLearning);
            },
            _ => {
                participants.push(LearningParticipant::HebbianEngine);
                participants.push(LearningParticipant::HomeostasisSystem);
                participants.push(LearningParticipant::AdaptiveLearning);
            },
        }
        
        participants
    }
    
    /// Generate synchronization points for coordination approach
    fn generate_synchronization_points(&self, approach: &CoordinationApproach) -> Vec<String> {
        match approach {
            CoordinationApproach::Synchronized => vec![
                "All systems ready".to_string(),
                "Learning phase complete".to_string(),
                "Optimization complete".to_string(),
                "Final validation".to_string(),
            ],
            CoordinationApproach::Sequential => vec![
                "Hebbian learning complete".to_string(),
                "Homeostasis complete".to_string(),
                "Optimization complete".to_string(),
            ],
            CoordinationApproach::Parallel => vec![
                "All parallel processes launched".to_string(),
                "All parallel processes complete".to_string(),
            ],
            CoordinationApproach::Emergency => vec![
                "Emergency assessment complete".to_string(),
                "Emergency response complete".to_string(),
            ],
        }
    }
    
    /// Check if emergency intervention is needed
    pub fn check_emergency_conditions(&self, assessment: &SystemAssessment) -> Option<super::emergency::EmergencyType> {
        if assessment.overall_health < 0.3 {
            Some(super::emergency::EmergencyType::PerformanceCollapse)
        } else if assessment.risk_factors.iter().any(|r| r.contains("overload")) {
            Some(super::emergency::EmergencyType::SystemOverload)
        } else if assessment.risk_factors.iter().any(|r| r.contains("divergence")) {
            Some(super::emergency::EmergencyType::LearningDivergence)
        } else {
            None
        }
    }
}

impl Default for LearningCoordinator {
    fn default() -> Self {
        Self::new()
    }
}