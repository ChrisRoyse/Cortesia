use crate::core::triple::Triple;
use crate::error::{GraphError, Result};
use crate::agents::construction::{KnowledgeAgent, ConstructionTask, ConstructionResult, TaskType, WorkItem};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use chrono::{DateTime, Utc};
use uuid;

/// Multi-agent coordinator for knowledge graph construction
pub struct MultiAgentCoordinator {
    agents: Arc<RwLock<HashMap<AgentId, Arc<dyn KnowledgeAgent>>>>,
    consensus_protocol: ConsensusProtocol,
    merge_strategy: MergeStrategy,
    coordination_channel: mpsc::Sender<CoordinationMessage>,
    active_tasks: Arc<RwLock<HashMap<String, CoordinationTask>>>,
}

impl MultiAgentCoordinator {
    pub fn new(consensus_protocol: ConsensusProtocol, merge_strategy: MergeStrategy) -> Self {
        let (sender, mut receiver) = mpsc::channel(1000);
        let active_tasks = Arc::new(RwLock::new(HashMap::new()));
        
        // Spawn coordination message handler
        let active_tasks_clone = active_tasks.clone();
        tokio::spawn(async move {
            while let Some(message) = receiver.recv().await {
                Self::handle_coordination_message(message, &active_tasks_clone).await;
            }
        });
        
        Self {
            agents: Arc::new(RwLock::new(HashMap::new())),
            consensus_protocol,
            merge_strategy,
            coordination_channel: sender,
            active_tasks,
        }
    }

    pub async fn register_agent(&self, agent_id: AgentId, agent: Arc<dyn KnowledgeAgent>) -> Result<()> {
        let mut agents = self.agents.write().await;
        agents.insert(agent_id, agent);
        Ok(())
    }

    pub async fn unregister_agent(&self, agent_id: &AgentId) -> Result<bool> {
        let mut agents = self.agents.write().await;
        Ok(agents.remove(agent_id).is_some())
    }

    pub async fn coordinate_construction(&self, task: ConstructionTask) -> Result<ConstructionResult> {
        let task_id = format!("task_{}", uuid::Uuid::new_v4());
        
        // Create coordination task
        let coordination_task = CoordinationTask {
            id: task_id.clone(),
            original_task: task.clone(),
            assigned_agents: Vec::new(),
            results: Vec::new(),
            status: TaskStatus::Pending,
            created_at: Utc::now(),
        };
        
        // Store task
        {
            let mut tasks = self.active_tasks.write().await;
            tasks.insert(task_id.clone(), coordination_task);
        }
        
        // Distribute work among agents
        let work_assignments = self.distribute_work(&task).await?;
        
        // Execute work on agents
        let mut agent_results = Vec::new();
        for (agent_id, work_item) in work_assignments {
            let result = self.execute_on_agent(&agent_id, work_item).await?;
            agent_results.push((agent_id, result));
        }
        
        // Achieve consensus on results
        let consensus_result = self.achieve_consensus(&agent_results).await?;
        
        // Merge results coherently
        let final_result = self.merge_results(consensus_result, &task).await?;
        
        // Update task status
        {
            let mut tasks = self.active_tasks.write().await;
            if let Some(task) = tasks.get_mut(&task_id) {
                task.status = TaskStatus::Completed;
                task.results = agent_results.into_iter().map(|(_, result)| result).collect();
            }
        }
        
        Ok(final_result)
    }

    async fn distribute_work(&self, task: &ConstructionTask) -> Result<Vec<(AgentId, WorkItem)>> {
        let agents = self.agents.read().await;
        let mut work_assignments = Vec::new();
        
        match &task.task_type {
            TaskType::TripleExtraction { text } => {
                // Divide text among agents
                let chunk_size = text.len() / agents.len().max(1);
                let mut start = 0;
                
                for (agent_id, agent) in agents.iter() {
                    if start < text.len() {
                        let end = (start + chunk_size).min(text.len());
                        let chunk = &text[start..end];
                        
                        if agent.get_capabilities().can_extract_triples {
                            work_assignments.push((agent_id.clone(), WorkItem::ExtractTriples {
                                text: chunk.to_string(),
                                context: task.context.clone(),
                            }));
                        }
                        
                        start = end;
                    }
                }
            }
            TaskType::FactValidation { facts } => {
                // Distribute facts among agents
                let facts_per_agent = facts.len() / agents.len().max(1);
                let mut start = 0;
                
                for (agent_id, agent) in agents.iter() {
                    if start < facts.len() {
                        let end = (start + facts_per_agent).min(facts.len());
                        let agent_facts = facts[start..end].to_vec();
                        
                        if agent.get_capabilities().can_validate_facts {
                            work_assignments.push((agent_id.clone(), WorkItem::ValidateFacts {
                                facts: agent_facts,
                                validation_strategy: task.validation_strategy.clone(),
                            }));
                        }
                        
                        start = end;
                    }
                }
            }
            TaskType::EntityResolution { entities } => {
                // Assign entity resolution to specialized agents
                for (agent_id, agent) in agents.iter() {
                    if agent.get_capabilities().can_resolve_entities {
                        work_assignments.push((agent_id.clone(), WorkItem::ResolveEntities {
                            entities: entities.clone(),
                            resolution_strategy: task.resolution_strategy.clone(),
                        }));
                    }
                }
            }
        }
        
        Ok(work_assignments)
    }

    async fn execute_on_agent(&self, agent_id: &AgentId, work_item: WorkItem) -> Result<ConstructionResult> {
        let agents = self.agents.read().await;
        
        if let Some(agent) = agents.get(agent_id) {
            agent.execute_work(work_item).await
        } else {
            Err(GraphError::InvalidInput(format!("Agent not found: {}", agent_id.0)))
        }
    }

    async fn achieve_consensus(&self, agent_results: &[(AgentId, ConstructionResult)]) -> Result<ConsensusResult> {
        match self.consensus_protocol {
            ConsensusProtocol::Majority => {
                self.majority_consensus(agent_results).await
            }
            ConsensusProtocol::Weighted => {
                self.weighted_consensus(agent_results).await
            }
            ConsensusProtocol::ByzantineFaultTolerant => {
                self.byzantine_fault_tolerant_consensus(agent_results).await
            }
        }
    }

    async fn majority_consensus(&self, agent_results: &[(AgentId, ConstructionResult)]) -> Result<ConsensusResult> {
        let mut fact_votes: HashMap<String, usize> = HashMap::new();
        let mut all_facts = Vec::new();
        
        for (_, result) in agent_results {
            for triple in &result.extracted_triples {
                let fact_key = format!("{}:{}:{}", triple.subject, triple.predicate, triple.object);
                *fact_votes.entry(fact_key).or_insert(0) += 1;
                all_facts.push(triple.clone());
            }
        }
        
        // Keep facts that got majority votes
        let majority_threshold = agent_results.len() / 2;
        let consensus_facts: Vec<Triple> = all_facts.into_iter()
            .filter(|triple| {
                let fact_key = format!("{}:{}:{}", triple.subject, triple.predicate, triple.object);
                fact_votes.get(&fact_key).unwrap_or(&0) > &majority_threshold
            })
            .collect();
        
        Ok(ConsensusResult {
            consensus_facts,
            confidence: 0.8,
            participating_agents: agent_results.len(),
            consensus_method: "majority".to_string(),
        })
    }

    async fn weighted_consensus(&self, agent_results: &[(AgentId, ConstructionResult)]) -> Result<ConsensusResult> {
        let agents = self.agents.read().await;
        let mut fact_scores: HashMap<String, f64> = HashMap::new();
        let mut all_facts = Vec::new();
        
        for (agent_id, result) in agent_results {
            let agent_weight = agents.get(agent_id)
                .map(|a| a.get_performance().accuracy)
                .unwrap_or(0.5);
            
            for triple in &result.extracted_triples {
                let fact_key = format!("{}:{}:{}", triple.subject, triple.predicate, triple.object);
                *fact_scores.entry(fact_key).or_insert(0.0) += agent_weight * triple.confidence as f64;
                all_facts.push(triple.clone());
            }
        }
        
        // Keep facts with high weighted scores
        let threshold = 0.6;
        let consensus_facts: Vec<Triple> = all_facts.into_iter()
            .filter(|triple| {
                let fact_key = format!("{}:{}:{}", triple.subject, triple.predicate, triple.object);
                fact_scores.get(&fact_key).unwrap_or(&0.0) > &threshold
            })
            .collect();
        
        Ok(ConsensusResult {
            consensus_facts,
            confidence: 0.85,
            participating_agents: agent_results.len(),
            consensus_method: "weighted".to_string(),
        })
    }

    async fn byzantine_fault_tolerant_consensus(&self, agent_results: &[(AgentId, ConstructionResult)]) -> Result<ConsensusResult> {
        // Simplified Byzantine fault tolerance - assumes up to 1/3 agents can be faulty
        let max_faulty = agent_results.len() / 3;
        let min_agreement = agent_results.len() - max_faulty;
        
        let mut fact_votes: HashMap<String, usize> = HashMap::new();
        let mut all_facts = Vec::new();
        
        for (_, result) in agent_results {
            for triple in &result.extracted_triples {
                let fact_key = format!("{}:{}:{}", triple.subject, triple.predicate, triple.object);
                *fact_votes.entry(fact_key).or_insert(0) += 1;
                all_facts.push(triple.clone());
            }
        }
        
        // Keep facts with sufficient agreement
        let consensus_facts: Vec<Triple> = all_facts.into_iter()
            .filter(|triple| {
                let fact_key = format!("{}:{}:{}", triple.subject, triple.predicate, triple.object);
                fact_votes.get(&fact_key).unwrap_or(&0) >= &min_agreement
            })
            .collect();
        
        Ok(ConsensusResult {
            consensus_facts,
            confidence: 0.9,
            participating_agents: agent_results.len(),
            consensus_method: "byzantine_fault_tolerant".to_string(),
        })
    }

    async fn merge_results(&self, consensus: ConsensusResult, task: &ConstructionTask) -> Result<ConstructionResult> {
        let merged_result = match self.merge_strategy {
            MergeStrategy::Union => {
                ConstructionResult {
                    task_id: format!("merged_{}", task.id),
                    extracted_triples: consensus.consensus_facts,
                    confidence: consensus.confidence,
                    processing_time_ms: 0, // Would aggregate timing
                    agent_contributions: HashMap::new(),
                }
            }
            MergeStrategy::Intersection => {
                // Only keep facts that all agents agreed on
                ConstructionResult {
                    task_id: format!("merged_{}", task.id),
                    extracted_triples: consensus.consensus_facts,
                    confidence: consensus.confidence * 1.1, // Higher confidence for intersection
                    processing_time_ms: 0,
                    agent_contributions: HashMap::new(),
                }
            }
            MergeStrategy::WeightedMerge => {
                // Already handled in weighted consensus
                ConstructionResult {
                    task_id: format!("merged_{}", task.id),
                    extracted_triples: consensus.consensus_facts,
                    confidence: consensus.confidence,
                    processing_time_ms: 0,
                    agent_contributions: HashMap::new(),
                }
            }
        };
        
        Ok(merged_result)
    }

    async fn handle_coordination_message(message: CoordinationMessage, active_tasks: &Arc<RwLock<HashMap<String, CoordinationTask>>>) {
        match message {
            CoordinationMessage::TaskProgress { task_id, progress } => {
                let mut tasks = active_tasks.write().await;
                if let Some(task) = tasks.get_mut(&task_id) {
                    task.status = TaskStatus::InProgress(progress);
                }
            }
            CoordinationMessage::AgentError { agent_id, error } => {
                eprintln!("Agent {} error: {}", agent_id.0, error);
            }
            CoordinationMessage::TaskComplete { task_id } => {
                let mut tasks = active_tasks.write().await;
                if let Some(task) = tasks.get_mut(&task_id) {
                    task.status = TaskStatus::Completed;
                }
            }
        }
    }

    pub async fn get_coordination_stats(&self) -> Result<CoordinationStats> {
        let agents = self.agents.read().await;
        let tasks = self.active_tasks.read().await;
        
        let active_tasks = tasks.values().filter(|t| matches!(t.status, TaskStatus::InProgress(_))).count();
        let completed_tasks = tasks.values().filter(|t| matches!(t.status, TaskStatus::Completed)).count();
        
        Ok(CoordinationStats {
            registered_agents: agents.len(),
            active_tasks,
            completed_tasks,
            total_tasks: tasks.len(),
        })
    }
}

/// Agent identifier
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct AgentId(pub String);

/// Consensus protocols
#[derive(Debug, Clone)]
pub enum ConsensusProtocol {
    Majority,
    Weighted,
    ByzantineFaultTolerant,
}

/// Merge strategies
#[derive(Debug, Clone)]
pub enum MergeStrategy {
    Union,
    Intersection,
    WeightedMerge,
}

/// Coordination task
#[derive(Debug, Clone)]
struct CoordinationTask {
    id: String,
    original_task: ConstructionTask,
    assigned_agents: Vec<AgentId>,
    results: Vec<ConstructionResult>,
    status: TaskStatus,
    created_at: DateTime<Utc>,
}

/// Task status
#[derive(Debug, Clone)]
enum TaskStatus {
    Pending,
    InProgress(f32), // progress percentage
    Completed,
    Failed(String),
}

/// Consensus result
#[derive(Debug, Clone)]
pub struct ConsensusResult {
    pub consensus_facts: Vec<Triple>,
    pub confidence: f64,
    pub participating_agents: usize,
    pub consensus_method: String,
}

/// Coordination messages
#[derive(Debug, Clone)]
pub enum CoordinationMessage {
    TaskProgress { task_id: String, progress: f32 },
    AgentError { agent_id: AgentId, error: String },
    TaskComplete { task_id: String },
}

/// Coordination statistics
#[derive(Debug, Clone)]
pub struct CoordinationStats {
    pub registered_agents: usize,
    pub active_tasks: usize,
    pub completed_tasks: usize,
    pub total_tasks: usize,
}



/// Agent coordination interface
pub trait AgentCoordination {
    fn get_agent_id(&self) -> AgentId;
    fn get_coordination_capabilities(&self) -> CoordinationCapabilities;
}

/// Coordination capabilities
#[derive(Debug, Clone)]
pub struct CoordinationCapabilities {
    pub can_coordinate: bool,
    pub supports_consensus: bool,
    pub fault_tolerance: bool,
    pub max_concurrent_tasks: usize,
}


