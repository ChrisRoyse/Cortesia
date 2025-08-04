# Task 101: Distributed Locking for Multi-Instance Support

**Estimated Time:** 10 minutes  
**Prerequisites:** Task 100  
**Dependencies:** Graceful shutdown implemented

## Objective
Implement distributed locking to support multiple instances accessing the same Tantivy index safely.

## Context
You're adding distributed locking to allow multiple instances of the vector search system to coordinate access to the shared Tantivy index. This prevents index corruption when running multiple instances for high availability or load balancing. The implementation must work on Windows network shares and local filesystems.

## Task Details

### What You Need to Do

1. **Create distributed lock module** (`src/distributed_lock.rs`):
```rust
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use std::fs::{File, OpenOptions};
use anyhow::{Result, anyhow};
use serde::{Serialize, Deserialize};
use tracing::{info, warn, debug};

#[derive(Debug, Serialize, Deserialize)]
struct LockFile {
    owner_id: String,
    process_id: u32,
    acquired_at: u64,
    expires_at: u64,
    hostname: String,
}

pub struct DistributedLock {
    lock_path: PathBuf,
    owner_id: String,
    lease_duration: Duration,
    held: Arc<AtomicBool>,
}

impl DistributedLock {
    pub fn new(index_path: &Path, owner_id: String) -> Self {
        let lock_path = index_path.join(".tantivy_lock");
        
        Self {
            lock_path,
            owner_id,
            lease_duration: Duration::from_secs(30),
            held: Arc::new(AtomicBool::new(false)),
        }
    }
    
    pub async fn acquire(&self) -> Result<LockGuard> {
        let mut retry_count = 0;
        const MAX_RETRIES: u32 = 10;
        
        loop {
            match self.try_acquire().await {
                Ok(guard) => return Ok(guard),
                Err(e) if retry_count < MAX_RETRIES => {
                    warn!("Failed to acquire lock (attempt {}/{}): {}", 
                          retry_count + 1, MAX_RETRIES, e);
                    
                    // Exponential backoff
                    let wait_ms = 100 * (2_u64.pow(retry_count.min(5)));
                    tokio::time::sleep(Duration::from_millis(wait_ms)).await;
                    
                    retry_count += 1;
                }
                Err(e) => return Err(e),
            }
        }
    }
    
    async fn try_acquire(&self) -> Result<LockGuard> {
        // Check if lock file exists and is still valid
        if self.lock_path.exists() {
            match self.read_lock_file() {
                Ok(lock_info) => {
                    let now = SystemTime::now()
                        .duration_since(UNIX_EPOCH)?
                        .as_secs();
                    
                    if now < lock_info.expires_at {
                        // Lock is still held by another process
                        if self.is_process_alive(&lock_info) {
                            return Err(anyhow!(
                                "Lock held by {} (PID: {}) until {}",
                                lock_info.hostname,
                                lock_info.process_id,
                                lock_info.expires_at
                            ));
                        } else {
                            // Process is dead, we can take over
                            info!("Previous lock holder (PID: {}) is dead, taking over", 
                                  lock_info.process_id);
                        }
                    } else {
                        info!("Previous lock expired, acquiring new lock");
                    }
                }
                Err(e) => {
                    warn!("Could not read lock file, attempting to acquire: {}", e);
                }
            }
        }
        
        // Try to create lock file atomically
        self.create_lock_file().await?;
        
        // Start lease renewal
        let guard = LockGuard::new(self);
        guard.start_renewal().await;
        
        Ok(guard)
    }
    
    async fn create_lock_file(&self) -> Result<()> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_secs();
        
        let lock_info = LockFile {
            owner_id: self.owner_id.clone(),
            process_id: std::process::id(),
            acquired_at: now,
            expires_at: now + self.lease_duration.as_secs(),
            hostname: hostname::get()?.to_string_lossy().to_string(),
        };
        
        // Use exclusive file creation for atomicity
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            use winapi::um::winbase::CREATE_NEW;
            
            let file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .custom_flags(CREATE_NEW)
                .open(&self.lock_path)?;
                
            serde_json::to_writer(file, &lock_info)?;
        }
        
        #[cfg(not(windows))]
        {
            use std::os::unix::fs::OpenOptionsExt;
            
            let file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o644)
                .open(&self.lock_path)?;
                
            serde_json::to_writer(file, &lock_info)?;
        }
        
        self.held.store(true, Ordering::Relaxed);
        info!("Lock acquired by {} (PID: {})", lock_info.hostname, lock_info.process_id);
        
        Ok(())
    }
    
    fn read_lock_file(&self) -> Result<LockFile> {
        let file = File::open(&self.lock_path)?;
        Ok(serde_json::from_reader(file)?)
    }
    
    fn is_process_alive(&self, lock_info: &LockFile) -> bool {
        // Check if process is on same machine
        if let Ok(hostname) = hostname::get() {
            if hostname.to_string_lossy() != lock_info.hostname {
                // Different machine, assume alive
                return true;
            }
        }
        
        // Check if process exists
        #[cfg(windows)]
        {
            use winapi::um::processthreadsapi::OpenProcess;
            use winapi::um::winnt::PROCESS_QUERY_LIMITED_INFORMATION;
            use winapi::um::handleapi::CloseHandle;
            
            unsafe {
                let handle = OpenProcess(
                    PROCESS_QUERY_LIMITED_INFORMATION,
                    0,
                    lock_info.process_id
                );
                
                if !handle.is_null() {
                    CloseHandle(handle);
                    true
                } else {
                    false
                }
            }
        }
        
        #[cfg(not(windows))]
        {
            use nix::sys::signal::{self, Signal};
            use nix::unistd::Pid;
            
            // Send signal 0 to check if process exists
            match signal::kill(Pid::from_raw(lock_info.process_id as i32), Signal::SIGCONT) {
                Ok(_) => true,
                Err(_) => false,
            }
        }
    }
    
    async fn renew_lease(&self) -> Result<()> {
        if !self.held.load(Ordering::Relaxed) {
            return Err(anyhow!("Lock not held"));
        }
        
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_secs();
        
        let mut lock_info = self.read_lock_file()?;
        
        // Verify we still own the lock
        if lock_info.owner_id != self.owner_id {
            self.held.store(false, Ordering::Relaxed);
            return Err(anyhow!("Lock stolen by another process"));
        }
        
        // Update expiration
        lock_info.expires_at = now + self.lease_duration.as_secs();
        
        // Atomic rename for update
        let temp_path = self.lock_path.with_extension("tmp");
        let file = File::create(&temp_path)?;
        serde_json::to_writer(file, &lock_info)?;
        
        std::fs::rename(&temp_path, &self.lock_path)?;
        
        debug!("Lock lease renewed until {}", lock_info.expires_at);
        Ok(())
    }
    
    fn release(&self) -> Result<()> {
        if self.held.load(Ordering::Relaxed) {
            std::fs::remove_file(&self.lock_path)?;
            self.held.store(false, Ordering::Relaxed);
            info!("Lock released");
        }
        Ok(())
    }
}

pub struct LockGuard {
    lock: Arc<DistributedLock>,
    renewal_handle: Option<tokio::task::JoinHandle<()>>,
}

impl LockGuard {
    fn new(lock: &DistributedLock) -> Self {
        Self {
            lock: Arc::new(lock.clone()),
            renewal_handle: None,
        }
    }
    
    async fn start_renewal(&mut self) {
        let lock = self.lock.clone();
        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(10));
            
            loop {
                interval.tick().await;
                
                if !lock.held.load(Ordering::Relaxed) {
                    break;
                }
                
                if let Err(e) = lock.renew_lease().await {
                    error!("Failed to renew lease: {}", e);
                    break;
                }
            }
        });
        
        self.renewal_handle = Some(handle);
    }
}

impl Drop for LockGuard {
    fn drop(&mut self) {
        // Cancel renewal task
        if let Some(handle) = self.renewal_handle.take() {
            handle.abort();
        }
        
        // Release lock
        if let Err(e) = self.lock.release() {
            error!("Failed to release lock: {}", e);
        }
    }
}

// Integration with DocumentIndexer
pub struct LockedIndexer {
    indexer: DocumentIndexer,
    lock: DistributedLock,
}

impl LockedIndexer {
    pub async fn new(index_path: &Path) -> Result<Self> {
        let owner_id = format!("{}_{}", hostname::get()?.to_string_lossy(), std::process::id());
        let lock = DistributedLock::new(index_path, owner_id);
        
        // Acquire lock before creating indexer
        let _guard = lock.acquire().await?;
        
        let indexer = DocumentIndexer::new(index_path)?;
        
        Ok(Self { indexer, lock })
    }
    
    pub async fn index_file(&mut self, file_path: &Path) -> Result<()> {
        let _guard = self.lock.acquire().await?;
        self.indexer.index_file(file_path)
    }
}
```

2. **Add test for distributed locking**:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_distributed_lock() -> Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let lock1 = DistributedLock::new(temp_dir.path(), "instance1".to_string());
        let lock2 = DistributedLock::new(temp_dir.path(), "instance2".to_string());
        
        // First lock should succeed
        let guard1 = lock1.acquire().await?;
        
        // Second lock should fail immediately
        match lock2.try_acquire().await {
            Err(_) => {}, // Expected
            Ok(_) => panic!("Second lock should not succeed"),
        }
        
        // Release first lock
        drop(guard1);
        
        // Now second lock should succeed
        let _guard2 = lock2.acquire().await?;
        
        Ok(())
    }
}
```

## Success Criteria
- [ ] Multiple instances coordinate index access
- [ ] Lock renewal prevents expiration during use
- [ ] Dead process detection works correctly
- [ ] Works on Windows network shares
- [ ] Lock acquisition has proper retry logic
- [ ] No index corruption with concurrent access

## Common Pitfalls to Avoid
- Don't rely on file locking (not portable)
- Handle clock skew between machines
- Ensure atomic lock file operations
- Clean up locks from dead processes
- Test on actual network shares

## Context for Next Task
Task 102 will implement index backup and restore functionality.