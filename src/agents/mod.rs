//! Agent-based system for distributed knowledge graph construction and coordination

use std::sync::Arc;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use uuid::Uuid;

#[cfg(target_arch = "wasm32")]
use alloc::{string::String, vec::Vec, collections::BTreeMap as HashMap};

/// Unique identifier for agents
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentId(pub Uuid);

impl AgentId {
    /// Create a new random agent ID
    pub fn new() -> Self {
        AgentId(Uuid::new_v4())
    }
}

impl Default for AgentId {
    fn default() -> Self {
        Self::new()
    }
}

/// Complexity level for tasks and processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Complexity {
    /// Simple, straightforward tasks
    Low,
    /// Moderate complexity requiring some coordination
    Medium,
    /// Complex tasks requiring significant resources
    High,
    /// Extremely complex tasks requiring special handling
    Critical,
}

impl Complexity {
    /// Get a numeric representation of complexity (0-3)
    pub fn as_u8(&self) -> u8 {
        match self {
            Complexity::Low => 0,
            Complexity::Medium => 1,
            Complexity::High => 2,
            Complexity::Critical => 3,
        }
    }
}

/// Types of tasks that agents can perform
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TaskType {
    /// Extract entities from text
    EntityExtraction { 
        text: String,
        language: Option<String>,
    },
    /// Build relationships between entities
    RelationshipConstruction {
        source_entity: String,
        target_entity: String,
        context: Option<String>,
    },
    /// Validate graph integrity
    GraphValidation {
        subgraph_id: Option<String>,
        validation_rules: Vec<String>,
    },
    /// Optimize graph structure
    GraphOptimization {
        optimization_type: String,
        parameters: HashMap<String, String>,
    },
    /// Coordinate multiple sub-tasks
    Coordination {
        subtasks: Vec<Task>,
        strategy: String,
    },
    /// Custom task type
    Custom {
        task_name: String,
        parameters: HashMap<String, String>,
    },
}

/// A task to be performed by an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(PartialEq)]
pub struct Task {
    /// Unique task identifier
    pub id: Uuid,
    /// Type of task to perform
    pub task_type: TaskType,
    /// Priority level (0-100, higher is more important)
    pub priority: u8,
    /// Estimated complexity
    pub complexity: Complexity,
    /// Optional deadline for task completion
    pub deadline: Option<std::time::SystemTime>,
    /// Metadata for the task
    pub metadata: HashMap<String, String>,
}

impl Task {
    /// Create a new task
    pub fn new(task_type: TaskType, priority: u8, complexity: Complexity) -> Self {
        Task {
            id: Uuid::new_v4(),
            task_type,
            priority,
            complexity,
            deadline: None,
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to the task
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Set a deadline for the task
    pub fn with_deadline(mut self, deadline: std::time::SystemTime) -> Self {
        self.deadline = Some(deadline);
        self
    }
}

/// Result of executing a task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    /// The task that was executed
    pub task_id: Uuid,
    /// Whether the task succeeded
    pub success: bool,
    /// Result data (if successful)
    pub data: Option<serde_json::Value>,
    /// Error message (if failed)
    pub error: Option<String>,
    /// Execution duration
    pub duration: std::time::Duration,
    /// Performance metrics
    pub metrics: HashMap<String, f64>,
}

impl TaskResult {
    /// Create a successful task result
    pub fn success(task_id: Uuid, data: serde_json::Value, duration: std::time::Duration) -> Self {
        TaskResult {
            task_id,
            success: true,
            data: Some(data),
            error: None,
            duration,
            metrics: HashMap::new(),
        }
    }

    /// Create a failed task result
    pub fn failure(task_id: Uuid, error: String, duration: std::time::Duration) -> Self {
        TaskResult {
            task_id,
            success: false,
            data: None,
            error: Some(error),
            duration,
            metrics: HashMap::new(),
        }
    }
}

/// Request sent to agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Request {
    /// Unique request ID
    pub id: Uuid,
    /// The task to be performed
    pub task: Task,
    /// Sender agent ID
    pub sender: AgentId,
    /// Optional timeout for the request
    pub timeout: Option<std::time::Duration>,
}

impl Request {
    /// Create a new request
    pub fn new(task: Task, sender: AgentId) -> Self {
        Request {
            id: Uuid::new_v4(),
            task,
            sender,
            timeout: None,
        }
    }

    /// Set a timeout for the request
    pub fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }
}

/// Response from agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
    /// The request ID this responds to
    pub request_id: Uuid,
    /// The result of the task
    pub result: TaskResult,
    /// Responding agent ID
    pub responder: AgentId,
}

impl Response {
    /// Create a new response
    pub fn new(request_id: Uuid, result: TaskResult, responder: AgentId) -> Self {
        Response {
            request_id,
            result,
            responder,
        }
    }
}

/// Status of an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStatus {
    /// Agent ID
    pub id: AgentId,
    /// Whether the agent is active
    pub active: bool,
    /// Current load (0.0 - 1.0)
    pub load: f32,
    /// Number of tasks in queue
    pub queue_size: usize,
    /// Performance metrics
    pub metrics: HashMap<String, f64>,
}

/// Base trait for all agents
/// 
/// Note: This trait uses a boxed future to maintain object safety while still
/// supporting async operations. This allows for dynamic dispatch of agents.
pub trait Agent: Send + Sync {
    /// Get the agent's unique identifier
    fn id(&self) -> AgentId;
    
    /// Get the agent's capabilities
    fn capabilities(&self) -> Vec<TaskType>;
    
    /// Check if the agent can handle a specific task type
    fn can_handle(&self, task: &Task) -> bool;
    
    /// Process a request and return a response
    /// 
    /// Returns a boxed future to maintain trait object safety
    fn process(&self, request: Request) -> std::pin::Pin<Box<dyn std::future::Future<Output = Response> + Send + '_>>;
    
    /// Get the agent's current status
    fn status(&self) -> AgentStatus;
    
    /// Shutdown the agent gracefully
    fn shutdown(&mut self) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + '_>>;
}

/// Agent specialized in constructing knowledge graph elements
pub struct ConstructionAgent {
    id: AgentId,
    graph: Arc<crate::KnowledgeGraph>,
    extractor: Arc<crate::AdvancedEntityExtractor>,
    active: std::sync::atomic::AtomicBool,
    queue_size: std::sync::atomic::AtomicUsize,
}

impl ConstructionAgent {
    /// Create a new construction agent
    pub fn new(
        graph: Arc<crate::KnowledgeGraph>,
        extractor: Arc<crate::AdvancedEntityExtractor>,
    ) -> Self {
        ConstructionAgent {
            id: AgentId::new(),
            graph,
            extractor,
            active: std::sync::atomic::AtomicBool::new(true),
            queue_size: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Extract entities from text
    async fn extract_entities(&self, text: &str) -> Result<Vec<crate::ExtractionEntity>, String> {
        self.extractor
            .extract_entities(text)
            .await
            .map_err(|e| format!("Entity extraction failed: {:?}", e))
    }

    /// Build relationships between entities
    async fn build_relationships(
        &self,
        _source: &str,
        _target: &str,
        _context: Option<&str>,
    ) -> Result<(), String> {
        // Implementation would use the graph to create relationships
        // This is a placeholder
        Ok(())
    }
}

impl Agent for ConstructionAgent {
    fn id(&self) -> AgentId {
        self.id
    }

    fn capabilities(&self) -> Vec<TaskType> {
        vec![
            TaskType::EntityExtraction { text: String::new(), language: None },
            TaskType::RelationshipConstruction {
                source_entity: String::new(),
                target_entity: String::new(),
                context: None,
            },
        ]
    }

    fn can_handle(&self, task: &Task) -> bool {
        matches!(
            &task.task_type,
            TaskType::EntityExtraction { .. } | TaskType::RelationshipConstruction { .. }
        )
    }

    fn process(&self, request: Request) -> std::pin::Pin<Box<dyn std::future::Future<Output = Response> + Send + '_>> {
        Box::pin(async move {
            let start = std::time::Instant::now();
            self.queue_size.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

            let result = match &request.task.task_type {
                TaskType::EntityExtraction { text, .. } => {
                    match self.extract_entities(text).await {
                        Ok(entities) => TaskResult::success(
                            request.task.id,
                            serde_json::json!({ "entities": entities }),
                            start.elapsed(),
                        ),
                        Err(e) => TaskResult::failure(request.task.id, e, start.elapsed()),
                    }
                }
                TaskType::RelationshipConstruction { source_entity, target_entity, context } => {
                    match self.build_relationships(source_entity, target_entity, context.as_deref()).await {
                        Ok(_) => TaskResult::success(
                            request.task.id,
                            serde_json::json!({ "relationship_created": true }),
                            start.elapsed(),
                        ),
                        Err(e) => TaskResult::failure(request.task.id, e, start.elapsed()),
                    }
                }
                _ => TaskResult::failure(
                    request.task.id,
                    "Unsupported task type".to_string(),
                    start.elapsed(),
                ),
            };

            self.queue_size.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);

            Response::new(request.id, result, self.id)
        })
    }

    fn status(&self) -> AgentStatus {
        AgentStatus {
            id: self.id,
            active: self.active.load(std::sync::atomic::Ordering::Relaxed),
            load: 0.0, // Would calculate based on queue size and processing
            queue_size: self.queue_size.load(std::sync::atomic::Ordering::Relaxed),
            metrics: HashMap::new(),
        }
    }

    fn shutdown(&mut self) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + '_>> {
        Box::pin(async move {
            self.active.store(false, std::sync::atomic::Ordering::Relaxed);
            // Would wait for queued tasks to complete
        })
    }
}

/// Agent responsible for coordinating other agents
pub struct CoordinationAgent {
    id: AgentId,
    agents: std::sync::RwLock<HashMap<AgentId, Arc<dyn Agent>>>,
    active: std::sync::atomic::AtomicBool,
}

impl CoordinationAgent {
    /// Create a new coordination agent
    pub fn new() -> Self {
        CoordinationAgent {
            id: AgentId::new(),
            agents: std::sync::RwLock::new(HashMap::new()),
            active: std::sync::atomic::AtomicBool::new(true),
        }
    }

    /// Register an agent with the coordinator
    pub fn register_agent(&self, agent: Arc<dyn Agent>) -> Result<(), String> {
        let mut agents = self.agents.write().unwrap();
        agents.insert(agent.id(), agent);
        Ok(())
    }

    /// Find an agent capable of handling a task
    pub fn find_agent_for_task(&self, task: &Task) -> Option<Arc<dyn Agent>> {
        let agents = self.agents.read().unwrap();
        agents
            .values()
            .find(|agent| agent.can_handle(task))
            .cloned()
    }

    /// Distribute tasks among agents
    async fn distribute_tasks(&self, tasks: Vec<Task>) -> Vec<TaskResult> {
        let mut results = Vec::new();

        for task in tasks {
            if let Some(agent) = self.find_agent_for_task(&task) {
                let request = Request::new(task, self.id);
                let response = agent.process(request).await;
                results.push(response.result);
            } else {
                results.push(TaskResult::failure(
                    task.id,
                    "No agent available for task".to_string(),
                    std::time::Duration::from_secs(0),
                ));
            }
        }

        results
    }
}

impl Agent for CoordinationAgent {
    fn id(&self) -> AgentId {
        self.id
    }

    fn capabilities(&self) -> Vec<TaskType> {
        vec![TaskType::Coordination {
            subtasks: vec![],
            strategy: String::new(),
        }]
    }

    fn can_handle(&self, task: &Task) -> bool {
        matches!(&task.task_type, TaskType::Coordination { .. })
    }

    fn process(&self, request: Request) -> std::pin::Pin<Box<dyn std::future::Future<Output = Response> + Send + '_>> {
        Box::pin(async move {
            let start = std::time::Instant::now();

            let result = match &request.task.task_type {
                TaskType::Coordination { subtasks, strategy } => {
                    let results = self.distribute_tasks(subtasks.clone()).await;
                    let all_succeeded = results.iter().all(|r| r.success);

                    if all_succeeded {
                        TaskResult::success(
                            request.task.id,
                            serde_json::json!({
                                "strategy": strategy,
                                "results": results,
                            }),
                            start.elapsed(),
                        )
                    } else {
                        TaskResult::failure(
                            request.task.id,
                            "Some subtasks failed".to_string(),
                            start.elapsed(),
                        )
                    }
                }
                _ => TaskResult::failure(
                    request.task.id,
                    "Unsupported task type".to_string(),
                    start.elapsed(),
                ),
            };

            Response::new(request.id, result, self.id)
        })
    }

    fn status(&self) -> AgentStatus {
        let agents = self.agents.read().unwrap();
        
        AgentStatus {
            id: self.id,
            active: self.active.load(std::sync::atomic::Ordering::Relaxed),
            load: 0.0, // Would calculate based on coordinated agents
            queue_size: agents.len(),
            metrics: HashMap::new(),
        }
    }

    fn shutdown(&mut self) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + '_>> {
        Box::pin(async move {
            self.active.store(false, std::sync::atomic::Ordering::Relaxed);
            
            // Shutdown all managed agents
            let agents = self.agents.read().unwrap();
            for agent in agents.values() {
                // Would call shutdown on each agent
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_id_creation() {
        let id1 = AgentId::new();
        let id2 = AgentId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_complexity_ordering() {
        assert!(Complexity::Low.as_u8() < Complexity::Medium.as_u8());
        assert!(Complexity::Medium.as_u8() < Complexity::High.as_u8());
        assert!(Complexity::High.as_u8() < Complexity::Critical.as_u8());
    }

    #[test]
    fn test_task_creation() {
        let task = Task::new(
            TaskType::EntityExtraction {
                text: "Test text".to_string(),
                language: Some("en".to_string()),
            },
            50,
            Complexity::Medium,
        );

        assert_eq!(task.priority, 50);
        assert_eq!(task.complexity, Complexity::Medium);
    }

    #[test]
    fn test_task_result_creation() {
        let task_id = Uuid::new_v4();
        let duration = std::time::Duration::from_secs(1);

        let success_result = TaskResult::success(
            task_id,
            serde_json::json!({"data": "test"}),
            duration,
        );
        assert!(success_result.success);
        assert!(success_result.data.is_some());
        assert!(success_result.error.is_none());

        let failure_result = TaskResult::failure(
            task_id,
            "Error occurred".to_string(),
            duration,
        );
        assert!(!failure_result.success);
        assert!(failure_result.data.is_none());
        assert!(failure_result.error.is_some());
    }
}