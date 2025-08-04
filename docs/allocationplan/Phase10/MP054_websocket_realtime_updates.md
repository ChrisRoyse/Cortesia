# MP054: WebSocket Real-time Updates

## Task Description
Implement WebSocket infrastructure for real-time graph algorithm visualization, live progress updates, and bidirectional communication with client applications.

## Prerequisites
- MP001-MP050 completed
- Understanding of WebSocket protocol
- Knowledge of real-time communication patterns
- Familiarity with event-driven architectures

## Detailed Steps

1. Create `src/neuromorphic/realtime/websocket.rs`

2. Implement WebSocket server with connection management:
   ```rust
   use tokio_tungstenite::{WebSocketStream, accept_async};
   use tokio::net::{TcpListener, TcpStream};
   use futures_util::{SinkExt, StreamExt};
   use serde::{Serialize, Deserialize};
   use uuid::Uuid;
   use std::collections::HashMap;
   use std::sync::Arc;
   use tokio::sync::{RwLock, mpsc};
   
   pub struct WebSocketServer {
       connections: Arc<RwLock<HashMap<Uuid, ConnectionHandle>>>,
       event_broadcaster: EventBroadcaster,
       authentication: Arc<dyn Authentication>,
       rate_limiter: RateLimiter,
       config: WebSocketConfig,
   }
   
   #[derive(Debug, Clone)]
   pub struct WebSocketConfig {
       pub bind_address: String,
       pub port: u16,
       pub max_connections: usize,
       pub heartbeat_interval: std::time::Duration,
       pub connection_timeout: std::time::Duration,
       pub max_message_size: usize,
       pub compression_enabled: bool,
   }
   
   pub struct ConnectionHandle {
       pub id: Uuid,
       pub user_id: Option<String>,
       pub subscriptions: Vec<String>,
       pub tx: mpsc::UnboundedSender<RealtimeMessage>,
       pub created_at: chrono::DateTime<chrono::Utc>,
       pub last_heartbeat: Arc<RwLock<chrono::DateTime<chrono::Utc>>>,
       pub metadata: HashMap<String, serde_json::Value>,
   }
   
   impl WebSocketServer {
       pub async fn new(config: WebSocketConfig) -> Result<Self, WebSocketError> {
           Ok(Self {
               connections: Arc::new(RwLock::new(HashMap::new())),
               event_broadcaster: EventBroadcaster::new(),
               authentication: Arc::new(DefaultAuthentication::new()),
               rate_limiter: RateLimiter::new(100, std::time::Duration::from_secs(60)),
               config,
           })
       }
       
       pub async fn start(&self) -> Result<(), WebSocketError> {
           let listener = TcpListener::bind(format!("{}:{}", self.config.bind_address, self.config.port)).await?;
           tracing::info!("WebSocket server listening on {}:{}", self.config.bind_address, self.config.port);
           
           // Start heartbeat task
           let heartbeat_task = self.start_heartbeat_monitor();
           
           // Start connection cleanup task
           let cleanup_task = self.start_connection_cleanup();
           
           loop {
               tokio::select! {
                   accept_result = listener.accept() => {
                       match accept_result {
                           Ok((stream, addr)) => {
                               tracing::debug!("New connection from: {}", addr);
                               let server = self.clone();
                               tokio::spawn(async move {
                                   if let Err(e) = server.handle_connection(stream).await {
                                       tracing::error!("Connection error: {}", e);
                                   }
                               });
                           }
                           Err(e) => {
                               tracing::error!("Failed to accept connection: {}", e);
                           }
                       }
                   }
                   _ = heartbeat_task => {
                       tracing::error!("Heartbeat task ended unexpectedly");
                       break;
                   }
                   _ = cleanup_task => {
                       tracing::error!("Cleanup task ended unexpectedly");
                       break;
                   }
               }
           }
           
           Ok(())
       }
   }
   ```

3. Implement real-time message types and handlers:
   ```rust
   #[derive(Debug, Clone, Serialize, Deserialize)]
   #[serde(tag = "type", content = "data")]
   pub enum RealtimeMessage {
       // Client -> Server
       Subscribe { channels: Vec<String> },
       Unsubscribe { channels: Vec<String> },
       AlgorithmStart { algorithm: String, parameters: serde_json::Value },
       AlgorithmStop { job_id: Uuid },
       Heartbeat { timestamp: u64 },
       
       // Server -> Client
       AlgorithmProgress {
           job_id: Uuid,
           progress: f32,
           current_step: String,
           nodes_processed: u64,
           edges_processed: u64,
           intermediate_results: Option<serde_json::Value>,
       },
       NodeActivation {
           node_id: u64,
           activation_level: f32,
           timestamp: u64,
           algorithm: String,
       },
       EdgeTraversal {
           from_node: u64,
           to_node: u64,
           weight: f32,
           timestamp: u64,
           algorithm: String,
       },
       PathDiscovered {
           job_id: Uuid,
           path: Vec<u64>,
           cost: f32,
           is_optimal: bool,
       },
       AlgorithmComplete {
           job_id: Uuid,
           result: AlgorithmResult,
           execution_time: u64,
           statistics: AlgorithmStatistics,
       },
       Error {
           error_code: String,
           message: String,
           job_id: Option<Uuid>,
       },
       ConnectionEstablished {
           connection_id: Uuid,
           server_time: u64,
           supported_features: Vec<String>,
       },
   }
   
   impl WebSocketServer {
       async fn handle_connection(&self, stream: TcpStream) -> Result<(), WebSocketError> {
           // Check connection limit
           if self.connections.read().await.len() >= self.config.max_connections {
               tracing::warn!("Connection limit reached, rejecting new connection");
               return Err(WebSocketError::ConnectionLimitReached);
           }
           
           let ws_stream = accept_async(stream).await?;
           let connection_id = Uuid::new_v4();
           
           let (tx, rx) = mpsc::unbounded_channel();
           let connection = ConnectionHandle {
               id: connection_id,
               user_id: None,
               subscriptions: Vec::new(),
               tx,
               created_at: chrono::Utc::now(),
               last_heartbeat: Arc::new(RwLock::new(chrono::Utc::now())),
               metadata: HashMap::new(),
           };
           
           // Register connection
           {
               let mut connections = self.connections.write().await;
               connections.insert(connection_id, connection);
           }
           
           // Send connection established message
           let established_msg = RealtimeMessage::ConnectionEstablished {
               connection_id,
               server_time: chrono::Utc::now().timestamp_millis() as u64,
               supported_features: vec![
                   "graph_visualization".to_string(),
                   "algorithm_progress".to_string(),
                   "real_time_updates".to_string(),
               ],
           };
           
           if let Some(conn) = self.connections.read().await.get(&connection_id) {
               let _ = conn.tx.send(established_msg);
           }
           
           // Handle WebSocket communication
           self.handle_websocket_messages(connection_id, ws_stream, rx).await?;
           
           // Clean up connection
           {
               let mut connections = self.connections.write().await;
               connections.remove(&connection_id);
           }
           
           tracing::info!("Connection {} closed", connection_id);
           Ok(())
       }
       
       async fn handle_websocket_messages(
           &self,
           connection_id: Uuid,
           mut ws_stream: WebSocketStream<TcpStream>,
           mut outbound_rx: mpsc::UnboundedReceiver<RealtimeMessage>,
       ) -> Result<(), WebSocketError> {
           loop {
               tokio::select! {
                   // Handle incoming messages from client
                   msg = ws_stream.next() => {
                       match msg {
                           Some(Ok(msg)) => {
                               if msg.is_text() || msg.is_binary() {
                                   if let Err(e) = self.process_client_message(connection_id, msg).await {
                                       tracing::error!("Error processing client message: {}", e);
                                   }
                               } else if msg.is_close() {
                                   tracing::info!("Client closed connection: {}", connection_id);
                                   break;
                               }
                           }
                           Some(Err(e)) => {
                               tracing::error!("WebSocket error: {}", e);
                               break;
                           }
                           None => break,
                       }
                   }
                   
                   // Handle outbound messages to client
                   outbound_msg = outbound_rx.recv() => {
                       match outbound_msg {
                           Some(msg) => {
                               let serialized = serde_json::to_string(&msg)?;
                               if let Err(e) = ws_stream.send(tokio_tungstenite::tungstenite::Message::Text(serialized)).await {
                                   tracing::error!("Failed to send message: {}", e);
                                   break;
                               }
                           }
                           None => break,
                       }
                   }
               }
           }
           
           Ok(())
       }
   }
   ```

4. Implement event broadcasting and subscription management:
   ```rust
   pub struct EventBroadcaster {
       subscribers: Arc<RwLock<HashMap<String, Vec<Uuid>>>>,
       connection_registry: Arc<RwLock<HashMap<Uuid, mpsc::UnboundedSender<RealtimeMessage>>>>,
   }
   
   impl EventBroadcaster {
       pub fn new() -> Self {
           Self {
               subscribers: Arc::new(RwLock::new(HashMap::new())),
               connection_registry: Arc::new(RwLock::new(HashMap::new())),
           }
       }
       
       pub async fn subscribe(&self, connection_id: Uuid, channel: String) {
           let mut subscribers = self.subscribers.write().await;
           subscribers.entry(channel)
               .or_insert_with(Vec::new)
               .push(connection_id);
       }
       
       pub async fn unsubscribe(&self, connection_id: Uuid, channel: &str) {
           let mut subscribers = self.subscribers.write().await;
           if let Some(channel_subscribers) = subscribers.get_mut(channel) {
               channel_subscribers.retain(|&id| id != connection_id);
           }
       }
       
       pub async fn broadcast_to_channel(&self, channel: &str, message: RealtimeMessage) {
           let subscribers = self.subscribers.read().await;
           let registry = self.connection_registry.read().await;
           
           if let Some(channel_subscribers) = subscribers.get(channel) {
               for &connection_id in channel_subscribers {
                   if let Some(tx) = registry.get(&connection_id) {
                       if let Err(_) = tx.send(message.clone()) {
                           tracing::warn!("Failed to send message to connection {}", connection_id);
                       }
                   }
               }
           }
       }
       
       pub async fn broadcast_algorithm_progress(&self, job_id: Uuid, progress: AlgorithmProgress) {
           let message = RealtimeMessage::AlgorithmProgress {
               job_id,
               progress: progress.percentage,
               current_step: progress.current_step,
               nodes_processed: progress.nodes_processed,
               edges_processed: progress.edges_processed,
               intermediate_results: progress.intermediate_results,
           };
           
           self.broadcast_to_channel(&format!("algorithm_{}", job_id), message).await;
           self.broadcast_to_channel("all_algorithms", message).await;
       }
       
       pub async fn broadcast_node_activation(&self, activation: NodeActivation) {
           let message = RealtimeMessage::NodeActivation {
               node_id: activation.node_id,
               activation_level: activation.level,
               timestamp: activation.timestamp,
               algorithm: activation.algorithm,
           };
           
           self.broadcast_to_channel("node_activations", message.clone()).await;
           self.broadcast_to_channel(&format!("algorithm_{}", activation.algorithm), message).await;
       }
   }
   ```

5. Implement connection monitoring and health management:
   ```rust
   pub struct ConnectionMonitor {
       connections: Arc<RwLock<HashMap<Uuid, ConnectionHandle>>>,
       heartbeat_interval: std::time::Duration,
       connection_timeout: std::time::Duration,
   }
   
   impl ConnectionMonitor {
       pub async fn start_heartbeat_monitor(&self) {
           let mut interval = tokio::time::interval(self.heartbeat_interval);
           
           loop {
               interval.tick().await;
               
               let now = chrono::Utc::now();
               let mut dead_connections = Vec::new();
               
               {
                   let connections = self.connections.read().await;
                   for (connection_id, connection) in connections.iter() {
                       let last_heartbeat = *connection.last_heartbeat.read().await;
                       
                       if now.signed_duration_since(last_heartbeat).to_std().unwrap_or_default() > self.connection_timeout {
                           dead_connections.push(*connection_id);
                       }
                   }
               }
               
               // Remove dead connections
               if !dead_connections.is_empty() {
                   let mut connections = self.connections.write().await;
                   for connection_id in dead_connections {
                       tracing::warn!("Removing dead connection: {}", connection_id);
                       connections.remove(&connection_id);
                   }
               }
           }
       }
       
       pub async fn get_connection_statistics(&self) -> ConnectionStatistics {
           let connections = self.connections.read().await;
           
           let total_connections = connections.len();
           let mut authenticated_connections = 0;
           let mut subscription_counts = HashMap::new();
           
           for connection in connections.values() {
               if connection.user_id.is_some() {
                   authenticated_connections += 1;
               }
               
               for subscription in &connection.subscriptions {
                   *subscription_counts.entry(subscription.clone()).or_insert(0) += 1;
               }
           }
           
           ConnectionStatistics {
               total_connections,
               authenticated_connections,
               subscription_counts,
               uptime: std::time::SystemTime::now()
                   .duration_since(std::time::UNIX_EPOCH)
                   .unwrap_or_default(),
           }
       }
   }
   
   #[derive(Debug, Serialize)]
   pub struct ConnectionStatistics {
       pub total_connections: usize,
       pub authenticated_connections: usize,
       pub subscription_counts: HashMap<String, usize>,
       pub uptime: std::time::Duration,
   }
   ```

## Expected Output
```rust
pub trait RealtimeUpdates {
    async fn broadcast_algorithm_progress(&self, job_id: Uuid, progress: AlgorithmProgress) -> Result<(), RealtimeError>;
    async fn broadcast_node_activation(&self, activation: NodeActivation) -> Result<(), RealtimeError>;
    async fn subscribe_to_updates(&self, connection_id: Uuid, channels: Vec<String>) -> Result<(), RealtimeError>;
    async fn get_active_connections(&self) -> Result<Vec<ConnectionInfo>, RealtimeError>;
}

#[derive(Debug)]
pub enum WebSocketError {
    ConnectionError(tokio_tungstenite::tungstenite::Error),
    SerializationError(serde_json::Error),
    AuthenticationError(String),
    RateLimitExceeded,
    ConnectionLimitReached,
    InvalidMessage(String),
}

pub struct RealtimeMetrics {
    active_connections: AtomicUsize,
    messages_sent: AtomicU64,
    messages_received: AtomicU64,
    subscription_count: AtomicUsize,
    average_latency: AtomicU64,
}
```

## Verification Steps
1. Test WebSocket connection handling under high concurrency
2. Verify real-time message delivery guarantees
3. Test connection resilience during network interruptions
4. Benchmark message throughput and latency
5. Validate subscription management and filtering
6. Test authentication and authorization flows

## Time Estimate
25 minutes

## Dependencies
- MP001-MP050: Previous implementations
- tokio-tungstenite: WebSocket implementation
- serde: Message serialization
- uuid: Connection identification
- tokio: Async runtime
- futures-util: Stream utilities