# MP051: Event Streaming Integration

## Task Description
Integrate graph algorithm events with streaming infrastructure for real-time processing and distributed coordination across neuromorphic components.

## Prerequisites
- MP001-MP050 completed
- Understanding of event-driven architecture
- Knowledge of streaming systems and message brokers
- Async Rust patterns and tokio runtime

## Detailed Steps

1. Create `src/neuromorphic/integration/event_stream.rs`

2. Implement event producer for graph operations:
   ```rust
   use tokio::sync::mpsc;
   use serde::{Serialize, Deserialize};
   use std::collections::VecDeque;
   
   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub enum GraphEvent {
       NodeActivation { node_id: u64, activation_level: f32, timestamp: u64 },
       EdgeTraversal { from: u64, to: u64, weight: f32, algorithm: String },
       AlgorithmStart { algorithm: String, params: serde_json::Value },
       AlgorithmComplete { algorithm: String, result: GraphResult, duration_ms: u64 },
       CorticalColumnUpdate { column_id: u64, state: ColumnState },
       PathDiscovery { path: Vec<u64>, cost: f32, heuristic: Option<f32> },
   }
   
   pub struct GraphEventProducer {
       tx: mpsc::UnboundedSender<GraphEvent>,
       buffer: VecDeque<GraphEvent>,
       batch_size: usize,
       flush_interval: std::time::Duration,
   }
   
   impl GraphEventProducer {
       pub fn new(batch_size: usize) -> (Self, mpsc::UnboundedReceiver<GraphEvent>) {
           let (tx, rx) = mpsc::unbounded_channel();
           let producer = Self {
               tx,
               buffer: VecDeque::with_capacity(batch_size),
               batch_size,
               flush_interval: std::time::Duration::from_millis(100),
           };
           (producer, rx)
       }
       
       pub async fn publish_event(&mut self, event: GraphEvent) -> Result<(), StreamError> {
           self.buffer.push_back(event.clone());
           
           if self.buffer.len() >= self.batch_size {
               self.flush_buffer().await?;
           }
           
           self.tx.send(event).map_err(|_| StreamError::ChannelClosed)
       }
       
       async fn flush_buffer(&mut self) -> Result<(), StreamError> {
           while let Some(event) = self.buffer.pop_front() {
               // Serialize and send to external streaming system
               let serialized = bincode::serialize(&event)?;
               // Send to Kafka/Redis/etc here
           }
           Ok(())
       }
   }
   ```

3. Implement event consumer with backpressure handling:
   ```rust
   pub struct GraphEventConsumer {
       rx: mpsc::UnboundedReceiver<GraphEvent>,
       handlers: HashMap<String, Box<dyn EventHandler + Send + Sync>>,
       metrics: StreamMetrics,
   }
   
   #[async_trait]
   pub trait EventHandler {
       async fn handle_event(&self, event: &GraphEvent) -> Result<(), HandlerError>;
       fn event_types(&self) -> Vec<String>;
   }
   
   impl GraphEventConsumer {
       pub async fn run(&mut self) -> Result<(), StreamError> {
           let mut interval = tokio::time::interval(std::time::Duration::from_millis(10));
           
           loop {
               tokio::select! {
                   event = self.rx.recv() => {
                       match event {
                           Some(evt) => self.process_event(evt).await?,
                           None => break,
                       }
                   }
                   _ = interval.tick() => {
                       self.metrics.report_throughput().await;
                   }
               }
           }
           Ok(())
       }
       
       async fn process_event(&mut self, event: GraphEvent) -> Result<(), StreamError> {
           let event_type = self.get_event_type(&event);
           
           if let Some(handler) = self.handlers.get(&event_type) {
               handler.handle_event(&event).await
                   .map_err(|e| StreamError::HandlerError(e))?;
               self.metrics.increment_processed();
           }
           
           Ok(())
       }
   }
   ```

4. Create streaming infrastructure adapter:
   ```rust
   pub struct StreamingInfrastructure {
       kafka_producer: Option<rdkafka::producer::FutureProducer>,
       redis_client: Option<redis::aio::Connection>,
       nats_client: Option<async_nats::Client>,
   }
   
   impl StreamingInfrastructure {
       pub async fn new(config: &StreamConfig) -> Result<Self, StreamError> {
           let kafka_producer = if config.enable_kafka {
               Some(Self::create_kafka_producer(&config.kafka_config).await?)
           } else { None };
           
           let redis_client = if config.enable_redis {
               Some(Self::create_redis_client(&config.redis_config).await?)
           } else { None };
           
           Ok(Self { kafka_producer, redis_client, nats_client: None })
       }
       
       pub async fn publish_to_kafka(&self, topic: &str, event: &GraphEvent) -> Result<(), StreamError> {
           if let Some(producer) = &self.kafka_producer {
               let payload = bincode::serialize(event)?;
               let record = rdkafka::producer::FutureRecord::to(topic)
                   .payload(&payload)
                   .key(&event.get_partition_key());
               
               producer.send(record, std::time::Duration::from_secs(1)).await
                   .map_err(|e| StreamError::KafkaError(e.0))?;
           }
           Ok(())
       }
   }
   ```

5. Implement event ordering and deduplication:
   ```rust
   pub struct OrderedEventProcessor {
       sequence_tracker: HashMap<String, u64>,
       out_of_order_buffer: BTreeMap<u64, GraphEvent>,
       deduplication_cache: lru::LruCache<String, u64>,
   }
   
   impl OrderedEventProcessor {
       pub fn process_event(&mut self, event: GraphEvent) -> Result<Vec<GraphEvent>, StreamError> {
           let sequence = event.get_sequence_number();
           let event_id = event.get_event_id();
           
           // Check for duplicates
           if self.deduplication_cache.contains(&event_id) {
               return Ok(vec![]);
           }
           
           let expected_sequence = self.sequence_tracker
               .get(&event.get_stream_key())
               .unwrap_or(&0) + 1;
           
           if sequence == expected_sequence {
               // Process in order
               self.sequence_tracker.insert(event.get_stream_key(), sequence);
               self.deduplication_cache.put(event_id, sequence);
               
               let mut result = vec![event];
               
               // Check if we can process buffered events
               while let Some((seq, buffered_event)) = self.out_of_order_buffer.remove(&(expected_sequence + 1)) {
                   result.push(buffered_event);
                   self.sequence_tracker.insert(event.get_stream_key(), seq);
               }
               
               Ok(result)
           } else if sequence > expected_sequence {
               // Buffer out-of-order event
               self.out_of_order_buffer.insert(sequence, event);
               Ok(vec![])
           } else {
               // Duplicate or very old event
               Ok(vec![])
           }
       }
   }
   ```

## Expected Output
```rust
pub trait EventStreaming {
    async fn stream_events(&mut self) -> Result<EventStream, StreamError>;
    async fn subscribe_to_events(&self, event_types: Vec<String>) -> Result<EventSubscription, StreamError>;
    async fn publish_event(&self, event: GraphEvent) -> Result<(), StreamError>;
}

#[derive(Debug)]
pub enum StreamError {
    SerializationError(bincode::Error),
    ChannelClosed,
    KafkaError(rdkafka::error::KafkaError),
    RedisError(redis::RedisError),
    HandlerError(HandlerError),
    ConfigurationError(String),
}

pub struct StreamMetrics {
    events_processed: AtomicU64,
    events_failed: AtomicU64,
    throughput_per_second: AtomicU64,
    last_report_time: Arc<Mutex<Instant>>,
}
```

## Verification Steps
1. Test event throughput under load (>10k events/sec)
2. Verify event ordering preservation across restarts
3. Test failure recovery and replay mechanisms
4. Benchmark latency from publish to consumption
5. Validate backpressure handling under stress
6. Test cross-service event propagation

## Time Estimate
25 minutes

## Dependencies
- MP001-MP050: Previous implementations
- tokio: Async runtime
- rdkafka: Kafka client (optional)
- redis: Redis client (optional)
- bincode: Serialization
- lru: Deduplication cache