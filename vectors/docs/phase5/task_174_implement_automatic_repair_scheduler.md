# Task 112: Implement Automatic Repair Scheduler

## Prerequisites Check
- [ ] Task 111 completed: repair module foundation created
- [ ] RepairJob and RepairPriority structures are defined
- [ ] Run: `cargo check` (should pass)

## Context
Implement the scheduler that manages automatic repair jobs with priority queuing and execution.

## Task Objective
Create the AutoRepairScheduler struct that manages repair job queues and execution.

## Steps
1. Add repair scheduler struct:
   ```rust
   /// Automatic repair scheduler
   pub struct AutoRepairScheduler {
       /// Configuration
       config: AutoRepairConfig,
       /// Priority-based job queues
       job_queues: Arc<RwLock<HashMap<RepairPriority, Vec<RepairJob>>>>,
       /// Currently running jobs
       running_jobs: Arc<RwLock<HashMap<String, RepairJob>>>,
       /// Completed jobs history
       completed_jobs: Arc<RwLock<Vec<CompletedRepairJob>>>,
       /// Last health check timestamp
       last_health_check: Arc<RwLock<Instant>>,
   }
   
   /// Completed repair job record
   #[derive(Debug, Clone)]
   pub struct CompletedRepairJob {
       /// Original job
       pub job: RepairJob,
       /// Repair result
       pub result: RepairResult,
       /// Total execution time
       pub execution_time: Duration,
       /// Completion timestamp
       pub completed_at: Instant,
   }
   ```
2. Add constructor and basic methods:
   ```rust
   impl AutoRepairScheduler {
       /// Create new automatic repair scheduler
       pub fn new(config: AutoRepairConfig) -> Self {
           let mut job_queues = HashMap::new();
           job_queues.insert(RepairPriority::Critical, Vec::new());
           job_queues.insert(RepairPriority::High, Vec::new());
           job_queues.insert(RepairPriority::Medium, Vec::new());
           job_queues.insert(RepairPriority::Low, Vec::new());
           
           Self {
               config,
               job_queues: Arc::new(RwLock::new(job_queues)),
               running_jobs: Arc::new(RwLock::new(HashMap::new())),
               completed_jobs: Arc::new(RwLock::new(Vec::new())),
               last_health_check: Arc::new(RwLock::new(Instant::now())),
           }
       }
       
       /// Get current configuration
       pub fn config(&self) -> &AutoRepairConfig {
           &self.config
       }
       
       /// Update configuration
       pub fn update_config(&mut self, config: AutoRepairConfig) {
           self.config = config;
       }
   }
   ```
3. Add job scheduling methods:
   ```rust
   impl AutoRepairScheduler {
       /// Schedule a repair job
       pub async fn schedule_repair_job(
           &self,
           doc_id: String,
           priority: RepairPriority,
           trigger: RepairTrigger,
           strategy: SyncStrategy,
       ) -> String {
           let job_id = Uuid::new_v4().to_string();
           let now = Instant::now();
           
           let deadline = match priority {
               RepairPriority::Critical => now + Duration::from_secs(self.config.critical_deadline_seconds),
               RepairPriority::High => now + Duration::from_secs(self.config.high_deadline_seconds),
               RepairPriority::Medium => now + Duration::from_secs(3600), // 1 hour
               RepairPriority::Low => now + Duration::from_secs(24 * 3600), // 24 hours
           };
           
           let job = RepairJob {
               id: job_id.clone(),
               doc_id,
               priority: priority.clone(),
               trigger,
               strategy,
               created_at: now,
               deadline,
               retry_count: 0,
               max_retries: 3,
           };
           
           let mut queues = self.job_queues.write().await;
           if let Some(queue) = queues.get_mut(&priority) {
               queue.push(job);
               // Sort by deadline (earliest first)
               queue.sort_by_key(|job| job.deadline);
           }
           
           job_id
       }
       
       /// Get next job to execute
       pub async fn get_next_job(&self) -> Option<RepairJob> {
           if !self.config.enabled {
               return None;
           }
           
           let running_count = {
               let running = self.running_jobs.read().await;
               running.len()
           };
           
           if running_count >= self.config.max_concurrent_jobs {
               return None;
           }
           
           let mut queues = self.job_queues.write().await;
           
           // Check queues in priority order
           for priority in [RepairPriority::Critical, RepairPriority::High, RepairPriority::Medium, RepairPriority::Low] {
               if let Some(queue) = queues.get_mut(&priority) {
                   if let Some(job) = queue.drain(..).next() {
                       return Some(job);
                   }
               }
           }
           
           None
       }
   }
   ```
4. Add job execution tracking:
   ```rust
   impl AutoRepairScheduler {
       /// Mark job as started
       pub async fn start_job(&self, job: RepairJob) {
           let mut running = self.running_jobs.write().await;
           running.insert(job.id.clone(), job);
       }
       
       /// Complete a job and record result
       pub async fn complete_job(
           &self,
           job_id: &str,
           result: RepairResult,
           execution_time: Duration,
       ) {
           let job = {
               let mut running = self.running_jobs.write().await;
               running.remove(job_id)
           };
           
           if let Some(job) = job {
               let completed_job = CompletedRepairJob {
                   job,
                   result,
                   execution_time,
                   completed_at: Instant::now(),
               };
               
               let mut completed = self.completed_jobs.write().await;
               completed.push(completed_job);
               
               // Keep only recent history (last 1000 jobs)
               if completed.len() > 1000 {
                   completed.remove(0);
               }
           }
       }
       
       /// Retry failed job
       pub async fn retry_job(&self, mut job: RepairJob) -> bool {
           if job.retry_count >= job.max_retries {
               return false;
           }
           
           job.retry_count += 1;
           
           // Add delay before retry
           tokio::time::sleep(Duration::from_secs(self.config.retry_delay_seconds)).await;
           
           let mut queues = self.job_queues.write().await;
           if let Some(queue) = queues.get_mut(&job.priority) {
               queue.push(job);
               queue.sort_by_key(|job| job.deadline);
           }
           
           true
       }
   }
   ```
5. Verify compilation

## Success Criteria
- [ ] AutoRepairScheduler with priority-based job queues
- [ ] Job scheduling with deadline calculation
- [ ] Priority-based job execution order
- [ ] Concurrent job limit enforcement
- [ ] Job execution tracking with start/complete cycle
- [ ] Retry mechanism with configurable delays
- [ ] Completed job history management
- [ ] Compiles without errors

## Time: 6 minutes

## Next Task
Task 113 will implement repair execution engine.

## Notes
Scheduler provides priority-based job management with proper resource limits and retry mechanisms for robust repair operations.