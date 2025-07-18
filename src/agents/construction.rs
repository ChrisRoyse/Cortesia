use crate::core::triple::Triple;
use crate::error::{GraphError, Result};
use std::collections::HashMap;
use std::time::SystemTime;
use async_trait::async_trait;
use uuid;

/// Knowledge construction agent interface
#[async_trait]
pub trait KnowledgeAgent: Send + Sync {
    /// Get the agent's unique identifier
    fn get_agent_id(&self) -> String;
    
    /// Get the agent's capabilities
    fn get_capabilities(&self) -> AgentCapabilities;
    
    /// Get the agent's performance metrics
    fn get_performance(&self) -> AgentPerformance;
    
    /// Execute a work item and return results
    async fn execute_work(&self, work_item: WorkItem) -> Result<ConstructionResult>;
    
    /// Update the agent's configuration
    async fn update_config(&mut self, config: AgentConfig) -> Result<()>;
    
    /// Check if the agent is healthy and ready to work
    async fn health_check(&self) -> Result<HealthStatus>;
}

/// Knowledge construction task
#[derive(Debug, Clone)]
pub struct ConstructionTask {
    pub id: String,
    pub task_type: TaskType,
    pub context: Option<String>,
    pub validation_strategy: Option<String>,
    pub resolution_strategy: Option<String>,
}

/// Types of construction tasks
#[derive(Debug, Clone)]
pub enum TaskType {
    TripleExtraction { text: String },
    FactValidation { facts: Vec<Triple> },
    EntityResolution { entities: Vec<String> },
}

/// Work item for individual agents
#[derive(Debug, Clone)]
pub enum WorkItem {
    ExtractTriples { text: String, context: Option<String> },
    ValidateFacts { facts: Vec<Triple>, validation_strategy: Option<String> },
    ResolveEntities { entities: Vec<String>, resolution_strategy: Option<String> },
}

/// Result of a construction operation
#[derive(Debug, Clone)]
pub struct ConstructionResult {
    pub task_id: String,
    pub extracted_triples: Vec<Triple>,
    pub confidence: f64,
    pub processing_time_ms: u64,
    pub agent_contributions: HashMap<String, AgentContribution>,
}

/// Individual agent's contribution to the result
#[derive(Debug, Clone)]
pub struct AgentContribution {
    pub agent_id: String,
    pub triples_contributed: Vec<Triple>,
    pub confidence_scores: Vec<f64>,
    pub processing_time_ms: u64,
    pub validation_results: Vec<ValidationResult>,
}

/// Agent capabilities description
#[derive(Debug, Clone)]
pub struct AgentCapabilities {
    pub can_extract_triples: bool,
    pub can_validate_facts: bool,
    pub can_resolve_entities: bool,
    pub supported_languages: Vec<String>,
    pub max_text_length: usize,
    pub specializations: Vec<String>,
}

/// Agent performance metrics
#[derive(Debug, Clone)]
pub struct AgentPerformance {
    pub accuracy: f64,
    pub processing_speed: f64, // triples per second
    pub reliability: f64,
    pub total_tasks_completed: u64,
    pub average_confidence: f64,
    pub last_updated: SystemTime,
}

/// Agent configuration
#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub max_concurrent_tasks: usize,
    pub confidence_threshold: f64,
    pub timeout_seconds: u64,
    pub retry_attempts: usize,
    pub custom_parameters: HashMap<String, String>,
}

/// Health status of an agent
#[derive(Debug, Clone)]
pub struct HealthStatus {
    pub is_healthy: bool,
    pub status_message: String,
    pub last_check: SystemTime,
    pub resource_usage: ResourceUsage,
}

/// Resource usage metrics
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    pub cpu_percent: f64,
    pub memory_mb: u64,
    pub active_tasks: usize,
    pub queue_size: usize,
}

/// Validation result for facts
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub fact: Triple,
    pub is_valid: bool,
    pub confidence: f64,
    pub validation_method: String,
    pub evidence: Vec<String>,
}

/// Task distribution strategy
pub enum TaskDistribution {
    RoundRobin,
    LoadBased,
    CapabilityBased,
    PerformanceBased,
}

/// Default knowledge agent implementation
pub struct DefaultKnowledgeAgent {
    agent_id: String,
    capabilities: AgentCapabilities,
    performance: AgentPerformance,
    config: AgentConfig,
    task_count: u64,
}

impl DefaultKnowledgeAgent {
    pub fn new(agent_id: String) -> Self {
        Self {
            agent_id,
            capabilities: AgentCapabilities {
                can_extract_triples: true,
                can_validate_facts: true,
                can_resolve_entities: true,
                supported_languages: vec!["en".to_string()],
                max_text_length: 10000,
                specializations: vec!["general".to_string()],
            },
            performance: AgentPerformance {
                accuracy: 0.85,
                processing_speed: 100.0,
                reliability: 0.95,
                total_tasks_completed: 0,
                average_confidence: 0.8,
                last_updated: SystemTime::now(),
            },
            config: AgentConfig {
                max_concurrent_tasks: 5,
                confidence_threshold: 0.7,
                timeout_seconds: 30,
                retry_attempts: 3,
                custom_parameters: HashMap::new(),
            },
            task_count: 0,
        }
    }
}

#[async_trait]
impl KnowledgeAgent for DefaultKnowledgeAgent {
    fn get_agent_id(&self) -> String {
        self.agent_id.clone()
    }
    
    fn get_capabilities(&self) -> AgentCapabilities {
        self.capabilities.clone()
    }
    
    fn get_performance(&self) -> AgentPerformance {
        self.performance.clone()
    }
    
    async fn execute_work(&self, work_item: WorkItem) -> Result<ConstructionResult> {
        let start_time = SystemTime::now();
        let task_id = format!("task_{}", uuid::Uuid::new_v4());
        
        let extracted_triples = match work_item {
            WorkItem::ExtractTriples { text, context: _ } => {
                self.extract_triples_from_text(&text).await?
            }
            WorkItem::ValidateFacts { facts, validation_strategy: _ } => {
                self.validate_fact_triples(&facts).await?
            }
            WorkItem::ResolveEntities { entities, resolution_strategy: _ } => {
                self.resolve_entity_triples(&entities).await?
            }
        };
        
        let processing_time = start_time.elapsed().unwrap_or_default().as_millis() as u64;
        
        let mut agent_contributions = HashMap::new();
        agent_contributions.insert(
            self.agent_id.clone(),
            AgentContribution {
                agent_id: self.agent_id.clone(),
                triples_contributed: extracted_triples.clone(),
                confidence_scores: vec![0.85; extracted_triples.len()],
                processing_time_ms: processing_time,
                validation_results: Vec::new(),
            },
        );
        
        Ok(ConstructionResult {
            task_id,
            extracted_triples,
            confidence: 0.85,
            processing_time_ms: processing_time,
            agent_contributions,
        })
    }
    
    async fn update_config(&mut self, config: AgentConfig) -> Result<()> {
        self.config = config;
        Ok(())
    }
    
    async fn health_check(&self) -> Result<HealthStatus> {
        Ok(HealthStatus {
            is_healthy: true,
            status_message: "Agent is operational".to_string(),
            last_check: SystemTime::now(),
            resource_usage: ResourceUsage {
                cpu_percent: 25.0,
                memory_mb: 128,
                active_tasks: 0,
                queue_size: 0,
            },
        })
    }
}

impl DefaultKnowledgeAgent {
    async fn extract_triples_from_text(&self, text: &str) -> Result<Vec<Triple>> {
        // Simplified triple extraction
        let mut triples = Vec::new();
        
        // Basic pattern matching for simple sentences
        let sentences: Vec<&str> = text.split('.').collect();
        
        for (i, sentence) in sentences.iter().enumerate() {
            if sentence.trim().is_empty() {
                continue;
            }
            
            // Very basic subject-predicate-object extraction
            let words: Vec<&str> = sentence.split_whitespace().collect();
            if words.len() >= 3 {
                let subject = words[0].to_string();
                let predicate = words[1].to_string();
                let object = words[2..].join(" ");
                
                triples.push(Triple {
                    subject,
                    predicate,
                    object,
                    confidence: 0.7,
                    source: None,
                });
            }
            
            // Limit the number of triples extracted
            if i >= 10 {
                break;
            }
        }
        
        Ok(triples)
    }
    
    async fn validate_fact_triples(&self, facts: &[Triple]) -> Result<Vec<Triple>> {
        // Simplified fact validation - return facts with updated confidence
        let mut validated_triples = Vec::new();
        
        for fact in facts {
            let mut validated_fact = fact.clone();
            
            // Simple validation based on triple structure
            let validation_score = if !fact.subject.is_empty() 
                && !fact.predicate.is_empty() 
                && !fact.object.is_empty() {
                0.8
            } else {
                0.3
            };
            
            validated_fact.confidence = validation_score as f32;
            
            if validation_score >= self.config.confidence_threshold {
                validated_triples.push(validated_fact);
            }
        }
        
        Ok(validated_triples)
    }
    
    async fn resolve_entity_triples(&self, entities: &[String]) -> Result<Vec<Triple>> {
        // Simplified entity resolution - create resolution triples
        let mut resolution_triples = Vec::new();
        
        for entity in entities {
            if !entity.is_empty() {
                resolution_triples.push(Triple {
                    subject: entity.clone(),
                    predicate: "is_a".to_string(),
                    object: "entity".to_string(),
                    confidence: 0.75,
                    source: None,
                });
            }
        }
        
        Ok(resolution_triples)
    }
}

/// Specialized agent for scientific text
pub struct ScientificKnowledgeAgent {
    base_agent: DefaultKnowledgeAgent,
    scientific_vocabulary: HashMap<String, String>,
}

impl ScientificKnowledgeAgent {
    pub fn new(agent_id: String) -> Self {
        let mut base_agent = DefaultKnowledgeAgent::new(agent_id);
        base_agent.capabilities.specializations = vec!["scientific".to_string(), "research".to_string()];
        base_agent.performance.accuracy = 0.92; // Higher accuracy for specialized domain
        
        let mut scientific_vocabulary = HashMap::new();
        scientific_vocabulary.insert("protein".to_string(), "biological_entity".to_string());
        scientific_vocabulary.insert("gene".to_string(), "genetic_element".to_string());
        scientific_vocabulary.insert("experiment".to_string(), "research_activity".to_string());
        
        Self {
            base_agent,
            scientific_vocabulary,
        }
    }
}

#[async_trait]
impl KnowledgeAgent for ScientificKnowledgeAgent {
    fn get_agent_id(&self) -> String {
        self.base_agent.get_agent_id()
    }
    
    fn get_capabilities(&self) -> AgentCapabilities {
        self.base_agent.get_capabilities()
    }
    
    fn get_performance(&self) -> AgentPerformance {
        self.base_agent.get_performance()
    }
    
    async fn execute_work(&self, work_item: WorkItem) -> Result<ConstructionResult> {
        // Enhanced extraction for scientific text
        match work_item {
            WorkItem::ExtractTriples { text, context } => {
                let enhanced_text = self.preprocess_scientific_text(&text);
                let enhanced_work = WorkItem::ExtractTriples { 
                    text: enhanced_text, 
                    context 
                };
                self.base_agent.execute_work(enhanced_work).await
            }
            _ => self.base_agent.execute_work(work_item).await,
        }
    }
    
    async fn update_config(&mut self, config: AgentConfig) -> Result<()> {
        self.base_agent.update_config(config).await
    }
    
    async fn health_check(&self) -> Result<HealthStatus> {
        self.base_agent.health_check().await
    }
}

impl ScientificKnowledgeAgent {
    fn preprocess_scientific_text(&self, text: &str) -> String {
        let mut processed = text.to_string();
        
        // Replace scientific terms with normalized forms
        for (term, normalized) in &self.scientific_vocabulary {
            processed = processed.replace(term, normalized);
        }
        
        processed
    }
}

/// Agent manager for coordinating multiple agents
pub struct AgentManager {
    agents: HashMap<String, Box<dyn KnowledgeAgent>>,
    distribution_strategy: TaskDistribution,
    performance_monitor: AgentPerformanceMonitor,
}

impl AgentManager {
    pub fn new(distribution_strategy: TaskDistribution) -> Self {
        Self {
            agents: HashMap::new(),
            distribution_strategy,
            performance_monitor: AgentPerformanceMonitor::new(),
        }
    }
    
    pub async fn register_agent(&mut self, agent: Box<dyn KnowledgeAgent>) -> Result<()> {
        let agent_id = agent.get_agent_id();
        self.agents.insert(agent_id.clone(), agent);
        self.performance_monitor.add_agent(agent_id).await;
        Ok(())
    }
    
    pub async fn distribute_task(&self, task: ConstructionTask) -> Result<Vec<(String, WorkItem)>> {
        let available_agents: Vec<&String> = self.agents.keys().collect();
        
        if available_agents.is_empty() {
            return Err(GraphError::InvalidInput("No agents available".to_string()));
        }
        
        match self.distribution_strategy {
            TaskDistribution::RoundRobin => {
                self.distribute_round_robin(&task, &available_agents).await
            }
            TaskDistribution::CapabilityBased => {
                self.distribute_by_capability(&task, &available_agents).await
            }
            TaskDistribution::PerformanceBased => {
                self.distribute_by_performance(&task, &available_agents).await
            }
            TaskDistribution::LoadBased => {
                self.distribute_by_load(&task, &available_agents).await
            }
        }
    }
    
    async fn distribute_round_robin(
        &self,
        task: &ConstructionTask,
        agent_ids: &[&String],
    ) -> Result<Vec<(String, WorkItem)>> {
        let work_item = self.create_work_item(task)?;
        let selected_agent = agent_ids.first().unwrap();
        Ok(vec![((*selected_agent).clone(), work_item)])
    }
    
    async fn distribute_by_capability(
        &self,
        task: &ConstructionTask,
        agent_ids: &[&String],
    ) -> Result<Vec<(String, WorkItem)>> {
        let mut suitable_agents = Vec::new();
        
        for &agent_id in agent_ids {
            if let Some(agent) = self.agents.get(agent_id) {
                let capabilities = agent.get_capabilities();
                let is_suitable = match &task.task_type {
                    TaskType::TripleExtraction { .. } => capabilities.can_extract_triples,
                    TaskType::FactValidation { .. } => capabilities.can_validate_facts,
                    TaskType::EntityResolution { .. } => capabilities.can_resolve_entities,
                };
                
                if is_suitable {
                    suitable_agents.push(agent_id.clone());
                }
            }
        }
        
        if suitable_agents.is_empty() {
            return Err(GraphError::InvalidInput("No suitable agents found".to_string()));
        }
        
        let work_item = self.create_work_item(task)?;
        Ok(vec![(suitable_agents[0].clone(), work_item)])
    }
    
    async fn distribute_by_performance(
        &self,
        task: &ConstructionTask,
        agent_ids: &[&String],
    ) -> Result<Vec<(String, WorkItem)>> {
        let mut agent_performances = Vec::new();
        
        for &agent_id in agent_ids {
            if let Some(agent) = self.agents.get(agent_id) {
                let performance = agent.get_performance();
                agent_performances.push((agent_id.clone(), performance.accuracy));
            }
        }
        
        // Sort by performance (accuracy)
        agent_performances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        if agent_performances.is_empty() {
            return Err(GraphError::InvalidInput("No agents available".to_string()));
        }
        
        let work_item = self.create_work_item(task)?;
        Ok(vec![(agent_performances[0].0.clone(), work_item)])
    }
    
    async fn distribute_by_load(
        &self,
        task: &ConstructionTask,
        agent_ids: &[&String],
    ) -> Result<Vec<(String, WorkItem)>> {
        // Simplified load-based distribution (would check actual load in real implementation)
        self.distribute_round_robin(task, agent_ids).await
    }
    
    fn create_work_item(&self, task: &ConstructionTask) -> Result<WorkItem> {
        match &task.task_type {
            TaskType::TripleExtraction { text } => {
                Ok(WorkItem::ExtractTriples {
                    text: text.clone(),
                    context: task.context.clone(),
                })
            }
            TaskType::FactValidation { facts } => {
                Ok(WorkItem::ValidateFacts {
                    facts: facts.clone(),
                    validation_strategy: task.validation_strategy.clone(),
                })
            }
            TaskType::EntityResolution { entities } => {
                Ok(WorkItem::ResolveEntities {
                    entities: entities.clone(),
                    resolution_strategy: task.resolution_strategy.clone(),
                })
            }
        }
    }
}

/// Performance monitoring for agents
pub struct AgentPerformanceMonitor {
    performance_history: HashMap<String, Vec<AgentPerformance>>,
}

impl AgentPerformanceMonitor {
    pub fn new() -> Self {
        Self {
            performance_history: HashMap::new(),
        }
    }
    
    pub async fn add_agent(&mut self, agent_id: String) {
        self.performance_history.insert(agent_id, Vec::new());
    }
    
    pub async fn record_performance(&mut self, agent_id: String, performance: AgentPerformance) {
        if let Some(history) = self.performance_history.get_mut(&agent_id) {
            history.push(performance);
            
            // Keep only recent performance records
            if history.len() > 100 {
                history.drain(0..50);
            }
        }
    }
    
    pub fn get_agent_stats(&self, agent_id: &str) -> Option<AgentStats> {
        if let Some(history) = self.performance_history.get(agent_id) {
            if history.is_empty() {
                return None;
            }
            
            let total_tasks: u64 = history.iter().map(|p| p.total_tasks_completed).sum();
            let avg_accuracy: f64 = history.iter().map(|p| p.accuracy).sum::<f64>() / history.len() as f64;
            let avg_speed: f64 = history.iter().map(|p| p.processing_speed).sum::<f64>() / history.len() as f64;
            let avg_reliability: f64 = history.iter().map(|p| p.reliability).sum::<f64>() / history.len() as f64;
            
            Some(AgentStats {
                agent_id: agent_id.to_string(),
                total_tasks,
                average_accuracy: avg_accuracy,
                average_processing_speed: avg_speed,
                average_reliability: avg_reliability,
                performance_trend: self.calculate_trend(history),
            })
        } else {
            None
        }
    }
    
    fn calculate_trend(&self, history: &[AgentPerformance]) -> PerformanceTrend {
        if history.len() < 2 {
            return PerformanceTrend::Stable;
        }
        
        let recent = &history[history.len() - 5..];
        let older = &history[..5.min(history.len() - 5)];
        
        if recent.is_empty() || older.is_empty() {
            return PerformanceTrend::Stable;
        }
        
        let recent_avg = recent.iter().map(|p| p.accuracy).sum::<f64>() / recent.len() as f64;
        let older_avg = older.iter().map(|p| p.accuracy).sum::<f64>() / older.len() as f64;
        
        let threshold = 0.05;
        if recent_avg > older_avg + threshold {
            PerformanceTrend::Improving
        } else if recent_avg < older_avg - threshold {
            PerformanceTrend::Declining
        } else {
            PerformanceTrend::Stable
        }
    }
}

/// Agent statistics
#[derive(Debug, Clone)]
pub struct AgentStats {
    pub agent_id: String,
    pub total_tasks: u64,
    pub average_accuracy: f64,
    pub average_processing_speed: f64,
    pub average_reliability: f64,
    pub performance_trend: PerformanceTrend,
}

/// Performance trend indicators
#[derive(Debug, Clone)]
pub enum PerformanceTrend {
    Improving,
    Stable,
    Declining,
}

