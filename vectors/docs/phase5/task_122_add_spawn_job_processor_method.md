# Task 122: Add Spawn Job Processor Method

## Prerequisites Check
- [ ] Task 121 completed: Is running method added
- [ ] Run: `cargo check` (should pass)

## Context
Add job processing task spawner for AutoRepairSystem.

## Task Objective
Implement spawn_job_processor() method that handles repair job execution loop.

## Steps
1. Add spawn job processor method to AutoRepairSystem:
   ```rust
   impl AutoRepairSystem {
       /// Spawn job processing task
       fn spawn_job_processor(&self) -> tokio::task::JoinHandle<()> {
           let scheduler = Arc::clone(&self.scheduler);
           let execution_engine = Arc::clone(&self.execution_engine);
           let running = Arc::clone(&self.running);
           
           tokio::spawn(async move {
               while *running.read().await {
                   // Get next job
                   if let Some(job) = scheduler.read().await.get_next_job().await {
                       let job_id = job.id.clone();
                       
                       // Start job
                       scheduler.read().await.start_job(job.clone()).await;
                       
                       // Execute job
                       let start_time = Instant::now();
                       let result = execution_engine.execute_repair_job(job.clone()).await;
                       let execution_time = start_time.elapsed();
                       
                       // Complete job
                       if result.success {
                           scheduler.read().await.complete_job(&job_id, result, execution_time).await;
                       } else {
                           // Retry if possible
                           if !scheduler.read().await.retry_job(job).await {
                               scheduler.read().await.complete_job(&job_id, result, execution_time).await;
                           }
                       }
                   } else {
                       // No jobs available, sleep briefly
                       tokio::time::sleep(Duration::from_secs(1)).await;
                   }
               }
           })
       }
   }
   ```

## Success Criteria
- [ ] Job processor spawner implemented
- [ ] Complete job execution cycle
- [ ] Retry logic included
- [ ] Compiles without errors

## Time: 6 minutes