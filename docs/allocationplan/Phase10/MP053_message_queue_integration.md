# MP053: Message Queue Integration

## Task Description
Integrate message queue systems for asynchronous graph algorithm coordination, distributed processing, and reliable job scheduling across neuromorphic components.

## Prerequisites
- MP001-MP050 completed
- Understanding of message queue patterns
- Knowledge of distributed systems and async processing
- Familiarity with job scheduling and worker patterns

## Detailed Steps

1. Create `src/neuromorphic/messaging/queue.rs`

2. Implement message queue abstraction:
   ```rust
   use serde::{Serialize, Deserialize};
   use tokio::sync::mpsc;
   use uuid::Uuid;
   use std::collections::HashMap;
   use async_trait::async_trait;
   
   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub enum GraphMessage {
       AlgorithmRequest {
           id: Uuid,
           algorithm: String,
           graph_data: GraphData,
           parameters: HashMap<String, serde_json::Value>,
           priority: Priority,
           deadline: Option<chrono::DateTime<chrono::Utc>>,
       },
       AlgorithmResponse {
           request_id: Uuid,
           result: Result<AlgorithmResult, AlgorithmError>,
           execution_time: std::time::Duration,
           worker_id: String,
       },
       PartialResult {
           request_id: Uuid,
           progress: f32,
           intermediate_data: serde_json::Value,
       },
       WorkerHeartbeat {
           worker_id: String,
           status: WorkerStatus,
           current_jobs: Vec<Uuid>,
           resource_usage: ResourceUsage,
       },
       SystemCommand {
           command_type: SystemCommandType,
           target: Option<String>,
           parameters: HashMap<String, serde_json::Value>,
       },
   }
   
   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub enum Priority {
       Low = 1,
       Normal = 2,
       High = 3,
       Critical = 4,
   }
   
   #[async_trait]
   pub trait MessageQueue: Send + Sync {
       async fn publish(&self, topic: &str, message: GraphMessage) -> Result<(), QueueError>;
       async fn subscribe(&self, topic: &str) -> Result<MessageStream, QueueError>;
       async fn create_consumer_group(&self, group_id: &str, topics: Vec<String>) -> Result<ConsumerGroup, QueueError>;
       async fn ack_message(&self, message_id: &str) -> Result<(), QueueError>;
       async fn nack_message(&self, message_id: &str, requeue: bool) -> Result<(), QueueError>;
   }
   ```

3. Implement Redis-based message queue:
   ```rust
   use redis::{Client, aio::Connection, AsyncCommands};
   
   pub struct RedisMessageQueue {
       client: Client,
       connection_pool: deadpool_redis::Pool,
       retry_policy: RetryPolicy,
       serializer: MessageSerializer,
   }
   
   impl RedisMessageQueue {
       pub async fn new(redis_url: &str, pool_size: usize) -> Result<Self, QueueError> {
           let client = Client::open(redis_url)?;
           
           let config = deadpool_redis::Config::from_url(redis_url);
           let pool = config.create_pool(Some(
               deadpool_redis::Runtime::Tokio1
           ))?;
           
           Ok(Self {
               client,
               connection_pool: pool,
               retry_policy: RetryPolicy::exponential_backoff(),
               serializer: MessageSerializer::new(),
           })
       }
   }
   
   #[async_trait]
   impl MessageQueue for RedisMessageQueue {
       async fn publish(&self, topic: &str, message: GraphMessage) -> Result<(), QueueError> {
           let mut conn = self.connection_pool.get().await?;
           
           let serialized = self.serializer.serialize(&message)?;
           let message_id = Uuid::new_v4().to_string();
           
           // Use Redis Streams for reliable message delivery
           let _: String = conn.xadd(
               topic,
               "*",
               &[
                   ("id", message_id.as_str()),
                   ("data", serialized.as_str()),
                   ("timestamp", &chrono::Utc::now().timestamp().to_string()),
                   ("priority", &(message.get_priority() as u8).to_string()),
               ]
           ).await?;
           
           Ok(())
       }
       
       async fn subscribe(&self, topic: &str) -> Result<MessageStream, QueueError> {
           let mut conn = self.connection_pool.get().await?;
           
           // Create consumer group if it doesn't exist
           let _: Result<String, _> = conn.xgroup_create_mkstream(
               topic,
               "default_group",
               "0"
           ).await;
           
           let stream = MessageStream::new(conn, topic.to_string(), "default_group".to_string());
           Ok(stream)
       }
       
       async fn create_consumer_group(&self, group_id: &str, topics: Vec<String>) -> Result<ConsumerGroup, QueueError> {
           let mut connections = Vec::new();
           
           for topic in &topics {
               let mut conn = self.connection_pool.get().await?;
               
               // Create consumer group
               let _: Result<String, _> = conn.xgroup_create_mkstream(
                   topic,
                   group_id,
                   "0"
               ).await;
               
               connections.push((topic.clone(), conn));
           }
           
           Ok(ConsumerGroup::new(group_id.to_string(), connections))
       }
   }
   ```

4. Implement job scheduling and worker coordination:
   ```rust
   pub struct JobScheduler {
       message_queue: Arc<dyn MessageQueue>,
       worker_registry: Arc<Mutex<WorkerRegistry>>,
       job_tracker: Arc<Mutex<JobTracker>>,
       scheduler_config: SchedulerConfig,
   }
   
   #[derive(Debug, Clone)]
   pub struct SchedulerConfig {
       pub max_retries: u32,
       pub retry_delay: std::time::Duration,
       pub job_timeout: std::time::Duration,
       pub priority_queue_size: usize,
       pub worker_heartbeat_interval: std::time::Duration,
   }
   
   impl JobScheduler {
       pub async fn schedule_algorithm(&self, request: AlgorithmRequest) -> Result<Uuid, SchedulerError> {
           let job_id = Uuid::new_v4();
           
           // Create job entry
           let job = Job {
               id: job_id,
               algorithm: request.algorithm.clone(),
               priority: request.priority,
               created_at: chrono::Utc::now(),
               deadline: request.deadline,
               retry_count: 0,
               status: JobStatus::Queued,
           };
           
           // Register job
           {
               let mut tracker = self.job_tracker.lock().await;
               tracker.register_job(job);
           }
           
           // Select appropriate worker queue based on algorithm requirements
           let queue_name = self.select_worker_queue(&request).await?;
           
           // Publish to message queue
           let message = GraphMessage::AlgorithmRequest {
               id: job_id,
               algorithm: request.algorithm,
               graph_data: request.graph_data,
               parameters: request.parameters,
               priority: request.priority,
               deadline: request.deadline,
           };
           
           self.message_queue.publish(&queue_name, message).await?;
           
           Ok(job_id)
       }
       
       async fn select_worker_queue(&self, request: &AlgorithmRequest) -> Result<String, SchedulerError> {
           let registry = self.worker_registry.lock().await;
           
           // Find workers capable of handling this algorithm
           let capable_workers = registry.find_workers_for_algorithm(&request.algorithm);
           
           if capable_workers.is_empty() {
               return Err(SchedulerError::NoCapableWorkers(request.algorithm.clone()));
           }
           
           // Load balancing: select worker with lowest current load
           let optimal_worker = capable_workers.iter()
               .min_by_key(|worker| worker.current_job_count())
               .unwrap();
           
           Ok(format!("worker_queue_{}", optimal_worker.id))
       }
       
       pub async fn start_job_monitor(&self) {
           let mut interval = tokio::time::interval(std::time::Duration::from_secs(30));
           
           loop {
               interval.tick().await;
               
               let mut tracker = self.job_tracker.lock().await;
               let expired_jobs = tracker.find_expired_jobs(self.scheduler_config.job_timeout);
               
               for job in expired_jobs {
                   if job.retry_count < self.scheduler_config.max_retries {
                       // Retry job
                       tracing::warn!("Retrying job {} (attempt {})", job.id, job.retry_count + 1);
                       self.retry_job(job).await;
                   } else {
                       // Mark as failed
                       tracing::error!("Job {} failed after {} retries", job.id, job.retry_count);
                       tracker.mark_job_failed(job.id, "Maximum retries exceeded".to_string());
                   }
               }
           }
       }
   }
   ```

5. Implement distributed worker coordination:
   ```rust
   pub struct DistributedWorker {
       worker_id: String,
       capabilities: Vec<String>,
       message_queue: Arc<dyn MessageQueue>,
       current_jobs: Arc<Mutex<HashMap<Uuid, RunningJob>>>,
       resource_monitor: ResourceMonitor,
   }
   
   #[derive(Debug, Clone)]
   pub struct RunningJob {
       pub id: Uuid,
       pub algorithm: String,
       pub started_at: chrono::DateTime<chrono::Utc>,
       pub progress: f32,
       pub cancellation_token: tokio_util::sync::CancellationToken,
   }
   
   impl DistributedWorker {
       pub async fn start(&self) -> Result<(), WorkerError> {
           // Start heartbeat task
           let heartbeat_task = self.start_heartbeat();
           
           // Start message processing
           let processing_task = self.start_message_processing();
           
           // Start resource monitoring
           let monitoring_task = self.resource_monitor.start_monitoring();
           
           // Wait for any task to complete (shouldn't happen in normal operation)
           tokio::select! {
               result = heartbeat_task => {
                   tracing::error!("Heartbeat task ended: {:?}", result);
               }
               result = processing_task => {
                   tracing::error!("Processing task ended: {:?}", result);
               }
               result = monitoring_task => {
                   tracing::error!("Monitoring task ended: {:?}", result);
               }
           }
           
           Ok(())
       }
       
       async fn start_message_processing(&self) -> Result<(), WorkerError> {
           let queue_name = format!("worker_queue_{}", self.worker_id);
           let mut message_stream = self.message_queue.subscribe(&queue_name).await?;
           
           while let Some(message) = message_stream.next().await {
               match message {
                   Ok(graph_message) => {
                       if let Err(e) = self.process_message(graph_message).await {
                           tracing::error!("Failed to process message: {}", e);
                       }
                   }
                   Err(e) => {
                       tracing::error!("Error receiving message: {}", e);
                   }
               }
           }
           
           Ok(())
       }
       
       async fn process_message(&self, message: GraphMessage) -> Result<(), WorkerError> {
           match message {
               GraphMessage::AlgorithmRequest { id, algorithm, graph_data, parameters, .. } => {
                   // Check if we can handle this algorithm
                   if !self.capabilities.contains(&algorithm) {
                       return Err(WorkerError::UnsupportedAlgorithm(algorithm));
                   }
                   
                   // Check resource availability
                   if !self.resource_monitor.has_capacity().await {
                       return Err(WorkerError::InsufficientResources);
                   }
                   
                   // Execute algorithm
                   let cancellation_token = tokio_util::sync::CancellationToken::new();
                   let running_job = RunningJob {
                       id,
                       algorithm: algorithm.clone(),
                       started_at: chrono::Utc::now(),
                       progress: 0.0,
                       cancellation_token: cancellation_token.clone(),
                   };
                   
                   {
                       let mut jobs = self.current_jobs.lock().await;
                       jobs.insert(id, running_job);
                   }
                   
                   // Spawn algorithm execution task
                   let worker_id = self.worker_id.clone();
                   let message_queue = self.message_queue.clone();
                   let current_jobs = self.current_jobs.clone();
                   
                   tokio::spawn(async move {
                       let result = Self::execute_algorithm(
                           &algorithm,
                           graph_data,
                           parameters,
                           cancellation_token,
                       ).await;
                       
                       // Send response
                       let response = GraphMessage::AlgorithmResponse {
                           request_id: id,
                           result,
                           execution_time: chrono::Utc::now().signed_duration_since(
                               current_jobs.lock().await.get(&id).unwrap().started_at
                           ).to_std().unwrap_or_default(),
                           worker_id,
                       };
                       
                       let _ = message_queue.publish("algorithm_responses", response).await;
                       
                       // Remove from current jobs
                       current_jobs.lock().await.remove(&id);
                   });
               }
               GraphMessage::SystemCommand { command_type, .. } => {
                   self.handle_system_command(command_type).await?;
               }
               _ => {
                   tracing::warn!("Received unexpected message type");
               }
           }
           
           Ok(())
       }
   }
   ```

## Expected Output
```rust
pub trait MessageQueue {
    async fn publish(&self, topic: &str, message: GraphMessage) -> Result<(), QueueError>;
    async fn subscribe(&self, topic: &str) -> Result<MessageStream, QueueError>;
    async fn create_worker_pool(&self, pool_config: WorkerPoolConfig) -> Result<WorkerPool, QueueError>;
}

#[derive(Debug)]
pub enum QueueError {
    ConnectionError(String),
    SerializationError(serde_json::Error),
    TimeoutError,
    ConsumerGroupError(String),
    PublishError(String),
    SubscriptionError(String),
}

pub struct QueueMetrics {
    messages_published: AtomicU64,
    messages_consumed: AtomicU64,
    active_consumers: AtomicUsize,
    average_processing_time: AtomicU64,
    failed_deliveries: AtomicU64,
}
```

## Verification Steps
1. Test message delivery guarantees under network failures
2. Verify job scheduling fairness across priority levels
3. Test worker failover and job redistribution
4. Benchmark message throughput with large payloads
5. Validate dead letter queue handling
6. Test consumer group rebalancing

## Time Estimate
25 minutes

## Dependencies
- MP001-MP050: Previous implementations
- redis: Redis client for message queuing
- deadpool-redis: Connection pooling
- serde: Message serialization
- tokio: Async runtime
- uuid: Job identification