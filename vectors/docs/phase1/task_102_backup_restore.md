# Task 102: Index Backup and Restore Implementation

**Estimated Time:** 10 minutes  
**Prerequisites:** Task 101  
**Dependencies:** Distributed locking in place

## Objective
Implement backup and restore functionality for Tantivy indexes with incremental backup support.

## Context
You're implementing backup and restore capabilities for the vector search system's Tantivy index. This allows for disaster recovery, migration between environments, and point-in-time recovery. The implementation must handle large indexes efficiently with incremental backups and work reliably on Windows.

## Task Details

### What You Need to Do

1. **Create backup/restore module** (`src/backup.rs`):
```rust
use std::path::{Path, PathBuf};
use std::fs::{self, File};
use std::io::{Read, Write, BufReader, BufWriter};
use tar::{Builder, Archive};
use flate2::Compression;
use flate2::write::GzEncoder;
use flate2::read::GzDecoder;
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};
use anyhow::{Result, anyhow};
use tracing::{info, warn, debug};

#[derive(Debug, Serialize, Deserialize)]
pub struct BackupManifest {
    pub version: String,
    pub created_at: DateTime<Utc>,
    pub index_path: PathBuf,
    pub backup_type: BackupType,
    pub files: Vec<BackupFile>,
    pub total_size: u64,
    pub checksum: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum BackupType {
    Full,
    Incremental { base_backup: String },
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BackupFile {
    pub path: PathBuf,
    pub size: u64,
    pub modified: SystemTime,
    pub checksum: String,
}

pub struct BackupManager {
    index_path: PathBuf,
    backup_dir: PathBuf,
    last_backup_manifest: Option<BackupManifest>,
}

impl BackupManager {
    pub fn new(index_path: PathBuf, backup_dir: PathBuf) -> Result<Self> {
        fs::create_dir_all(&backup_dir)?;
        
        // Load last backup manifest if exists
        let last_backup_manifest = Self::find_latest_backup(&backup_dir)?;
        
        Ok(Self {
            index_path,
            backup_dir,
            last_backup_manifest,
        })
    }
    
    pub async fn create_backup(&mut self, incremental: bool) -> Result<PathBuf> {
        info!("Starting {} backup of index", 
              if incremental { "incremental" } else { "full" });
        
        // Acquire distributed lock for consistency
        let lock = DistributedLock::new(&self.index_path, "backup".to_string());
        let _guard = lock.acquire().await?;
        
        // Force index flush
        self.flush_index()?;
        
        // Determine backup type
        let backup_type = if incremental && self.last_backup_manifest.is_some() {
            BackupType::Incremental {
                base_backup: self.last_backup_manifest.as_ref().unwrap()
                    .created_at.to_rfc3339(),
            }
        } else {
            BackupType::Full
        };
        
        // Collect files to backup
        let files_to_backup = self.collect_backup_files(&backup_type)?;
        
        if files_to_backup.is_empty() {
            info!("No changes since last backup");
            return Ok(self.backup_dir.join("no_changes.txt"));
        }
        
        // Create backup archive
        let backup_path = self.create_backup_archive(&files_to_backup, backup_type)?;
        
        info!("Backup completed: {:?}", backup_path);
        Ok(backup_path)
    }
    
    fn collect_backup_files(&self, backup_type: &BackupType) -> Result<Vec<BackupFile>> {
        let mut files = Vec::new();
        let base_time = match backup_type {
            BackupType::Incremental { .. } => {
                self.last_backup_manifest.as_ref()
                    .map(|m| m.created_at)
                    .unwrap_or_else(|| Utc::now() - chrono::Duration::days(365))
            }
            BackupType::Full => Utc::now() - chrono::Duration::days(365),
        };
        
        // Walk index directory
        for entry in walkdir::WalkDir::new(&self.index_path)
            .follow_links(false)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if !entry.file_type().is_file() {
                continue;
            }
            
            let path = entry.path();
            let metadata = entry.metadata()?;
            let modified = metadata.modified()?;
            
            // Check if file needs backup
            let should_backup = match backup_type {
                BackupType::Full => true,
                BackupType::Incremental { .. } => {
                    let file_time: DateTime<Utc> = modified.into();
                    file_time > base_time
                }
            };
            
            if should_backup {
                let relative_path = path.strip_prefix(&self.index_path)?;
                
                files.push(BackupFile {
                    path: relative_path.to_path_buf(),
                    size: metadata.len(),
                    modified,
                    checksum: self.calculate_checksum(path)?,
                });
            }
        }
        
        Ok(files)
    }
    
    fn create_backup_archive(&mut self, files: &[BackupFile], backup_type: BackupType) -> Result<PathBuf> {
        let timestamp = Utc::now().format("%Y%m%d_%H%M%S");
        let backup_name = format!("backup_{}_{}.tar.gz",
            match &backup_type {
                BackupType::Full => "full",
                BackupType::Incremental { .. } => "incr",
            },
            timestamp
        );
        
        let backup_path = self.backup_dir.join(&backup_name);
        let tar_gz = File::create(&backup_path)?;
        let enc = GzEncoder::new(tar_gz, Compression::default());
        let mut tar = Builder::new(enc);
        
        // Add manifest
        let manifest = BackupManifest {
            version: "1.0".to_string(),
            created_at: Utc::now(),
            index_path: self.index_path.clone(),
            backup_type,
            files: files.to_vec(),
            total_size: files.iter().map(|f| f.size).sum(),
            checksum: self.calculate_archive_checksum(files)?,
        };
        
        let manifest_json = serde_json::to_string_pretty(&manifest)?;
        let mut header = tar::Header::new_gnu();
        header.set_path("manifest.json")?;
        header.set_size(manifest_json.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        
        tar.append(&header, manifest_json.as_bytes())?;
        
        // Add files
        for file in files {
            let full_path = self.index_path.join(&file.path);
            tar.append_file(&file.path, &mut File::open(full_path)?)?;
        }
        
        tar.finish()?;
        
        // Update last backup manifest
        self.last_backup_manifest = Some(manifest);
        
        Ok(backup_path)
    }
    
    pub async fn restore_backup(&self, backup_path: &Path) -> Result<()> {
        info!("Starting restore from {:?}", backup_path);
        
        // Acquire distributed lock
        let lock = DistributedLock::new(&self.index_path, "restore".to_string());
        let _guard = lock.acquire().await?;
        
        // Read and validate backup
        let tar_gz = File::open(backup_path)?;
        let tar = GzDecoder::new(tar_gz);
        let mut archive = Archive::new(tar);
        
        // Extract manifest first
        let manifest = self.extract_manifest(&mut archive)?;
        
        // Validate checksums
        self.validate_backup(&manifest, &mut archive)?;
        
        // Perform restore based on type
        match manifest.backup_type {
            BackupType::Full => self.restore_full(&mut archive)?,
            BackupType::Incremental { ref base_backup } => {
                self.restore_incremental(&mut archive, base_backup)?
            }
        }
        
        info!("Restore completed successfully");
        Ok(())
    }
    
    fn restore_full(&self, archive: &mut Archive<GzDecoder<File>>) -> Result<()> {
        // Clear existing index
        if self.index_path.exists() {
            warn!("Removing existing index for full restore");
            fs::remove_dir_all(&self.index_path)?;
        }
        
        fs::create_dir_all(&self.index_path)?;
        
        // Extract all files
        for entry in archive.entries()? {
            let mut entry = entry?;
            let path = entry.path()?;
            
            if path.to_str() == Some("manifest.json") {
                continue; // Skip manifest
            }
            
            let dest_path = self.index_path.join(path);
            
            if let Some(parent) = dest_path.parent() {
                fs::create_dir_all(parent)?;
            }
            
            entry.unpack(&dest_path)?;
            debug!("Restored: {:?}", dest_path);
        }
        
        Ok(())
    }
    
    fn restore_incremental(&self, archive: &mut Archive<GzDecoder<File>>, base_backup: &str) -> Result<()> {
        // Verify base backup exists
        info!("Applying incremental backup on top of base: {}", base_backup);
        
        if !self.index_path.exists() {
            return Err(anyhow!("Cannot apply incremental backup: index does not exist"));
        }
        
        // Apply incremental changes
        for entry in archive.entries()? {
            let mut entry = entry?;
            let path = entry.path()?;
            
            if path.to_str() == Some("manifest.json") {
                continue;
            }
            
            let dest_path = self.index_path.join(path);
            
            if let Some(parent) = dest_path.parent() {
                fs::create_dir_all(parent)?;
            }
            
            entry.unpack(&dest_path)?;
            debug!("Updated: {:?}", dest_path);
        }
        
        Ok(())
    }
    
    fn calculate_checksum(&self, path: &Path) -> Result<String> {
        use sha2::{Sha256, Digest};
        
        let mut file = File::open(path)?;
        let mut hasher = Sha256::new();
        let mut buffer = [0; 8192];
        
        loop {
            let bytes_read = file.read(&mut buffer)?;
            if bytes_read == 0 {
                break;
            }
            hasher.update(&buffer[..bytes_read]);
        }
        
        Ok(format!("{:x}", hasher.finalize()))
    }
    
    fn flush_index(&self) -> Result<()> {
        // Force Tantivy to flush all pending writes
        let index = Index::open_in_dir(&self.index_path)?;
        let mut writer = index.writer(50_000_000)?;
        writer.commit()?;
        Ok(())
    }
}
```

2. **Add backup/restore tests**:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_full_backup_restore() -> Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let index_path = temp_dir.path().join("index");
        let backup_dir = temp_dir.path().join("backups");
        
        // Create and populate index
        let mut indexer = DocumentIndexer::new(&index_path)?;
        let test_file = temp_dir.path().join("test.rs");
        fs::write(&test_file, "pub fn test() {}")?;
        indexer.index_file(&test_file)?;
        
        // Create backup
        let mut backup_mgr = BackupManager::new(index_path.clone(), backup_dir)?;
        let backup_path = backup_mgr.create_backup(false).await?;
        
        // Clear index
        fs::remove_dir_all(&index_path)?;
        
        // Restore
        backup_mgr.restore_backup(&backup_path).await?;
        
        // Verify index works
        let search = SearchEngine::new(&index_path)?;
        let results = search.search("test")?;
        assert!(!results.is_empty());
        
        Ok(())
    }
}
```

## Success Criteria
- [ ] Full backup captures entire index
- [ ] Incremental backup only includes changes
- [ ] Restore recovers index to working state
- [ ] Checksums validate backup integrity
- [ ] Compression reduces backup size >50%
- [ ] Works with large indexes (>1GB)

## Common Pitfalls to Avoid
- Don't backup while index is being modified
- Ensure atomic backup operations
- Handle corrupted backup files gracefully
- Test restore on different machines
- Verify Windows path compatibility in archives

## Context for Next Task
Task 103 will implement index optimization and compaction.