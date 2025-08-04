# Ultimate RAG System - Disaster Recovery Procedures

## Executive Summary

This document provides comprehensive disaster recovery procedures for the Ultimate RAG System, a multi-tier search architecture achieving 95-97% accuracy through Tantivy indexes, LanceDB vector storage, multi-embedding systems, and Git temporal analysis.

**CRITICAL**: This system has no prior operational context. All procedures are derived from architectural analysis and must be validated during implementation.

## System Architecture Overview

### Core Components Requiring Protection
1. **Tantivy Text Indexes** - Full-text search capabilities
2. **LanceDB Vector Storage** - ACID transactional vector database (3072-dimensional embeddings)
3. **Multi-Embedding Cache Layers** - 7 specialized embedding services cache
4. **Git Analysis Cache** - Temporal analysis and commit history data
5. **Configuration and Secrets** - API keys, service configurations, system parameters

### Data Flow Dependencies
```
Documents → Chunking → Multi-Embedding → Vector Storage (LanceDB)
         → Text Processing → Tantivy Indexes
         → Git Analysis → Temporal Cache
```

## Recovery Time Objective (RTO) / Recovery Point Objective (RPO) Requirements

### Tier 1: Fast Local Search (Critical)
- **Components**: Tantivy indexes, cached results, exact match data
- **RTO**: < 5 minutes (business critical)
- **RPO**: < 15 minutes (minimal data loss acceptable)
- **Availability Target**: 99.9%
- **Backup Frequency**: Continuous incremental + hourly snapshots

### Tier 2: Balanced Hybrid Search (High Priority)
- **Components**: LanceDB vector storage, embedding caches, multi-method coordination
- **RTO**: < 30 minutes
- **RPO**: < 1 hour
- **Availability Target**: 99.5%
- **Backup Frequency**: Every 4 hours + daily full backup

### Tier 3: Deep Analysis (Standard)
- **Components**: Git temporal analysis, complex reasoning chains, comprehensive reports
- **RTO**: < 2 hours
- **RPO**: < 4 hours
- **Availability Target**: 99.0%
- **Backup Frequency**: Daily full backup + weekly comprehensive archive

## Critical Scenarios and Response Procedures

### Scenario 1: Data Corruption

#### 1.1 Tantivy Index Corruption
**Detection Signs:**
- Search results return empty or inconsistent results
- Index integrity check failures
- Rust panic errors during search operations

**Immediate Response (Execute within 5 minutes):**
```powershell
# 1. Stop all search services immediately
Stop-Service -Name "UltimateRAGSearchEngine" -Force

# 2. Verify corruption extent
tantivy-cli check-integrity --index-path "C:\data\rag\tantivy_indexes"

# 3. Isolate corrupted index segments
Move-Item "C:\data\rag\tantivy_indexes\*.corrupted" "C:\recovery\corrupted_segments\$(Get-Date -Format 'yyyyMMdd_HHmmss')"
```

**Recovery Procedure:**
```powershell
# Option A: Restore from latest backup (Preferred - RTO < 5 minutes)
robocopy "C:\backups\tantivy\latest" "C:\data\rag\tantivy_indexes" /MIR /R:3 /W:5

# Option B: Rebuild from source documents (RTO < 2 hours)
.\scripts\rebuild_tantivy_index.ps1 -SourcePath "C:\data\documents" -IndexPath "C:\data\rag\tantivy_indexes" -ParallelWorkers 8

# 3. Validate recovery
tantivy-cli search --index-path "C:\data\rag\tantivy_indexes" --query "test query"

# 4. Restart services
Start-Service -Name "UltimateRAGSearchEngine"
```

#### 1.2 LanceDB Vector Database Corruption
**Detection Signs:**
- Vector similarity search failures
- ACID transaction rollback errors
- Arrow columnar data format errors

**Immediate Response:**
```powershell
# 1. Enable LanceDB recovery mode
$env:LANCEDB_RECOVERY_MODE = "true"

# 2. Export uncorrupted data immediately
.\scripts\lancedb_emergency_export.ps1 -DatabasePath "C:\data\rag\lancedb" -OutputPath "C:\recovery\lancedb_export_$(Get-Date -Format 'yyyyMMdd_HHmmss')"

# 3. Verify transaction log integrity
lancedb-cli verify-logs --db-path "C:\data\rag\lancedb"
```

**Recovery Procedure:**
```powershell
# 1. Restore from latest consistent backup
Remove-Item "C:\data\rag\lancedb" -Recurse -Force
robocopy "C:\backups\lancedb\latest_consistent" "C:\data\rag\lancedb" /MIR

# 2. Apply transaction logs since backup
.\scripts\apply_lancedb_transaction_logs.ps1 -DatabasePath "C:\data\rag\lancedb" -LogPath "C:\logs\lancedb_transactions"

# 3. Rebuild corrupted vectors if necessary
.\scripts\rebuild_embeddings.ps1 -EmbeddingService "all" -BatchSize 100
```

### Scenario 2: Service Failures

#### 2.1 Embedding Service Cascade Failure
**Detection Signs:**
- Multiple embedding services returning errors
- Cache hit rate drops below 10%
- Query processing times exceed 30 seconds

**Immediate Response:**
```powershell
# 1. Activate emergency fallback mode
$env:RAG_EMERGENCY_MODE = "true"
$env:RAG_USE_LOCAL_EMBEDDINGS_ONLY = "true"

# 2. Switch to local BGE-M3 model only
.\scripts\activate_emergency_embeddings.ps1 -LocalModelPath "C:\models\bge-m3-local"

# 3. Reduce system load
.\scripts\throttle_requests.ps1 -MaxConcurrentRequests 5 -QueueSize 100
```

**Recovery Procedure:**
```powershell
# 1. Diagnose service failures systematically
foreach ($service in @("VoyageCode2", "E5Mistral", "CodeBERT", "SQLCoder", "BERTConfig", "StackTraceBERT")) {
    .\scripts\test_embedding_service.ps1 -ServiceName $service -TestQuery "test embedding generation"
}

# 2. Restore working services incrementally
.\scripts\restore_embedding_services.ps1 -ServicesConfig "C:\config\embedding_services.json"

# 3. Warm up caches systematically
.\scripts\warm_embedding_caches.ps1 -CacheStrategy "progressive" -PriorityQueries "C:\data\priority_queries.txt"
```

### Scenario 3: Network Partitions

#### 3.1 Split-Brain Prevention
**Prevention Measures:**
```powershell
# Implement distributed consensus mechanism
# File: scripts\prevent_split_brain.ps1

param(
    [Parameter(Mandatory=$true)]
    [string]$NodeId,
    [Parameter(Mandatory=$true)]
    [string[]]$PeerNodes
)

# 1. Implement quorum-based decision making
$quorumSize = [Math]::Floor($PeerNodes.Count / 2) + 1

# 2. Check network connectivity to peers
$activePeers = @()
foreach ($peer in $PeerNodes) {
    if (Test-Connection -ComputerName $peer -Count 1 -Quiet) {
        $activePeers += $peer
    }
}

# 3. Determine if this node should remain active
if ($activePeers.Count -ge $quorumSize) {
    Write-Host "Quorum achieved. Node $NodeId remaining active."
    exit 0
} else {
    Write-Host "Quorum lost. Node $NodeId entering safe mode."
    .\scripts\enter_safe_mode.ps1
    exit 1
}
```

### Scenario 4: Storage Failures

#### 4.1 Disk Crash Recovery
**Immediate Response:**
```powershell
# 1. Assess storage damage
Get-WmiObject -Class Win32_LogicalDisk | Where-Object { $_.DriveType -eq 3 } | Select-Object DeviceID, Size, FreeSpace, VolumeName

# 2. Activate emergency read-only mode
$env:RAG_READ_ONLY_MODE = "true"

# 3. Begin emergency backup to alternative storage
robocopy "C:\data\rag" "D:\emergency_backup\rag_$(Get-Date -Format 'yyyyMMdd_HHmmss')" /MIR /R:1 /W:1 /MT:8
```

**Recovery Procedure:**
```powershell
# 1. Setup replacement storage
Format-Volume -DriveLetter E -FileSystem NTFS -AllocationUnitSize 4096 -NewFileSystemLabel "RAG_Recovery"

# 2. Restore from most recent backup
.\scripts\restore_complete_system.ps1 -BackupLocation "\\backup-server\rag-backups\latest" -TargetLocation "E:\data\rag"

# 3. Update configuration for new storage location
.\scripts\update_storage_paths.ps1 -OldPath "C:\data\rag" -NewPath "E:\data\rag"
```

### Scenario 5: Catastrophic Loss

#### 5.1 Complete System Failure Recovery
**Prerequisites:**
- Offsite backup availability
- Clean Windows environment
- Network connectivity

**Complete Recovery Procedure:**
```powershell
# Phase 1: Environment Reconstruction (30 minutes)
# 1. Install Rust toolchain
winget install Rustlang.Rustup

# 2. Install required dependencies
cargo install tantivy-cli
cargo install ripgrep

# 3. Setup directory structure
New-Item -ItemType Directory -Force -Path @(
    "C:\data\rag\tantivy_indexes",
    "C:\data\rag\lancedb", 
    "C:\data\rag\embeddings_cache",
    "C:\data\rag\git_analysis_cache",
    "C:\logs\rag",
    "C:\config\rag"
)

# Phase 2: Data Restoration (2-4 hours depending on data size)
# 1. Restore Tantivy indexes
robocopy "\\offsite-backup\rag\tantivy_indexes" "C:\data\rag\tantivy_indexes" /MIR /R:3 /W:10

# 2. Restore LanceDB
robocopy "\\offsite-backup\rag\lancedb" "C:\data\rag\lancedb" /MIR /R:3 /W:10

# 3. Restore embedding caches
robocopy "\\offsite-backup\rag\embeddings_cache" "C:\data\rag\embeddings_cache" /MIR /R:3 /W:10

# Phase 3: Service Restoration (30 minutes)
# 1. Restore configuration
robocopy "\\offsite-backup\rag\config" "C:\config\rag" /MIR

# 2. Install and configure services
.\scripts\install_rag_services.ps1 -ConfigPath "C:\config\rag"

# 3. Validate complete system
.\scripts\validate_complete_system.ps1 -TestSuite "comprehensive"
```

## Backup Strategies

### Incremental Backup Strategy
```powershell
# File: scripts\incremental_backup.ps1
param(
    [Parameter(Mandatory=$true)]
    [string]$BackupDestination,
    [string]$LogPath = "C:\logs\rag\backup.log"
)

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

# 1. Tantivy incremental backup (detects changed segments)
robocopy "C:\data\rag\tantivy_indexes" "$BackupDestination\tantivy\$timestamp" /MIR /XO /R:3 /W:5 /LOG+:"$LogPath"

# 2. LanceDB transaction log backup
Copy-Item "C:\data\rag\lancedb\*.log" "$BackupDestination\lancedb_logs\$timestamp\" -Force

# 3. Embedding cache changed files only
robocopy "C:\data\rag\embeddings_cache" "$BackupDestination\embeddings_cache\$timestamp" /M /R:3 /W:5

# 4. Verify backup integrity
.\scripts\verify_backup_integrity.ps1 -BackupPath "$BackupDestination\$timestamp"
```

### Full Backup Strategy
```powershell
# File: scripts\full_backup.ps1
param(
    [Parameter(Mandatory=$true)]
    [string]$BackupDestination
)

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$fullBackupPath = "$BackupDestination\full_backup_$timestamp"

# 1. Stop services for consistent backup
Stop-Service -Name "UltimateRAGSearchEngine" -Force
Start-Sleep -Seconds 10

# 2. Create full system backup
robocopy "C:\data\rag" "$fullBackupPath\data" /MIR /R:3 /W:5
robocopy "C:\config\rag" "$fullBackupPath\config" /MIR /R:3 /W:5
robocopy "C:\logs\rag" "$fullBackupPath\logs" /MIR /R:3 /W:5

# 3. Create backup manifest
@{
    Timestamp = $timestamp
    DataSize = (Get-ChildItem "$fullBackupPath\data" -Recurse | Measure-Object -Property Length -Sum).Sum
    ConfigFiles = (Get-ChildItem "$fullBackupPath\config" -Recurse).Count
    LogFiles = (Get-ChildItem "$fullBackupPath\logs" -Recurse).Count
    SystemVersion = (Get-Content "C:\config\rag\version.txt" -ErrorAction SilentlyContinue)
} | ConvertTo-Json | Out-File "$fullBackupPath\manifest.json"

# 4. Restart services
Start-Service -Name "UltimateRAGSearchEngine"
```

### Continuous Backup Strategy
```powershell
# File: scripts\continuous_backup.ps1
# Implements real-time file system monitoring and backup

param(
    [Parameter(Mandatory=$true)]
    [string]$WatchPath = "C:\data\rag",
    [Parameter(Mandatory=$true)]
    [string]$BackupDestination
)

# 1. Setup file system watcher
$watcher = New-Object System.IO.FileSystemWatcher
$watcher.Path = $WatchPath
$watcher.Filter = "*.*"
$watcher.IncludeSubdirectories = $true
$watcher.EnableRaisingEvents = $true

# 2. Define backup action
$action = {
    $path = $Event.SourceEventArgs.FullPath
    $changeType = $Event.SourceEventArgs.ChangeType
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
    
    if ($changeType -eq "Created" -or $changeType -eq "Changed") {
        $relativePath = $path.Replace("C:\data\rag\", "")
        $backupPath = "$BackupDestination\continuous\$timestamp\$relativePath"
        
        try {
            $backupDir = Split-Path $backupPath -Parent
            if (!(Test-Path $backupDir)) {
                New-Item -ItemType Directory -Force -Path $backupDir
            }
            Copy-Item $path $backupPath -Force
            Write-Host "Backed up: $relativePath"
        } catch {
            Write-Warning "Failed to backup $path`: $_"
        }
    }
}

# 3. Register event handlers
Register-ObjectEvent -InputObject $watcher -EventName "Created" -Action $action
Register-ObjectEvent -InputObject $watcher -EventName "Changed" -Action $action

Write-Host "Continuous backup monitoring started for $WatchPath"
Write-Host "Press Ctrl+C to stop monitoring"

try {
    while ($true) {
        Start-Sleep -Seconds 1
    }
} finally {
    $watcher.EnableRaisingEvents = $false
    $watcher.Dispose()
}
```

## Windows-Specific VSS Integration

### Volume Shadow Copy Service (VSS) Integration
```powershell
# File: scripts\vss_backup.ps1
param(
    [Parameter(Mandatory=$true)]
    [string]$VolumePath = "C:",
    [Parameter(Mandatory=$true)]
    [string]$BackupDestination
)

# 1. Create VSS snapshot
$shadowCopy = (Get-WmiObject -Class Win32_ShadowCopy | Where-Object { $_.VolumeName -eq "$VolumePath\" } | Sort-Object InstallDate -Descending | Select-Object -First 1)

if (-not $shadowCopy) {
    # Create new shadow copy
    $result = (Get-WmiObject -Class Win32_ShadowCopy).Create("$VolumePath\", "ClientAccessible")
    if ($result.ReturnValue -eq 0) {
        $shadowCopy = Get-WmiObject -Class Win32_ShadowCopy | Where-Object { $_.ID -eq $result.ShadowID }
        Write-Host "Created shadow copy: $($shadowCopy.DeviceObject)"
    } else {
        throw "Failed to create shadow copy. Return value: $($result.ReturnValue)"
    }
}

# 2. Mount shadow copy
$shadowPath = $shadowCopy.DeviceObject + "\"
$mountPoint = "C:\VSS_Mount_$(Get-Date -Format 'yyyyMMddHHmmss')"

try {
    # Create symbolic link to shadow copy
    cmd /c "mklink /D `"$mountPoint`" `"$shadowPath`""
    
    # 3. Perform backup from consistent snapshot
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $vssBackupPath = "$BackupDestination\vss_backup_$timestamp"
    
    robocopy "$mountPoint\data\rag" "$vssBackupPath\data" /MIR /R:3 /W:5
    robocopy "$mountPoint\config\rag" "$vssBackupPath\config" /MIR /R:3 /W:5
    
    # 4. Create VSS backup manifest
    @{
        Timestamp = $timestamp
        ShadowCopyId = $shadowCopy.ID
        ShadowCopyCreated = $shadowCopy.InstallDate
        BackupPath = $vssBackupPath
        DataIntegrity = "VSS_CONSISTENT"
    } | ConvertTo-Json | Out-File "$vssBackupPath\vss_manifest.json"
    
    Write-Host "VSS backup completed successfully: $vssBackupPath"
    
} finally {
    # 5. Cleanup mount point
    if (Test-Path $mountPoint) {
        cmd /c "rmdir `"$mountPoint`""
    }
}

# Optional: Cleanup old shadow copies (keep last 5)
$allShadows = Get-WmiObject -Class Win32_ShadowCopy | Where-Object { $_.VolumeName -eq "$VolumePath\" } | Sort-Object InstallDate -Descending
if ($allShadows.Count -gt 5) {
    $oldShadows = $allShadows | Select-Object -Skip 5
    foreach ($oldShadow in $oldShadows) {
        $oldShadow.Delete()
        Write-Host "Deleted old shadow copy: $($oldShadow.ID)"
    }
}
```

### VSS-Aware Application Backup
```powershell
# File: scripts\vss_application_backup.ps1
# Coordinates with RAG system for application-consistent backups

param(
    [Parameter(Mandatory=$true)]
    [string]$BackupDestination
)

# 1. Signal application to prepare for backup
Write-Host "Signaling RAG system to prepare for VSS backup..."
$env:RAG_VSS_BACKUP_PREPARING = "true"

# Wait for application to reach consistent state
$timeout = 60  # seconds
$elapsed = 0
while ($elapsed -lt $timeout) {
    if (Test-Path "C:\data\rag\.vss_ready") {
        Write-Host "RAG system ready for VSS backup"
        break
    }
    Start-Sleep -Seconds 2
    $elapsed += 2
}

if ($elapsed -ge $timeout) {
    Write-Warning "Timeout waiting for RAG system to prepare for backup"
    $env:RAG_VSS_BACKUP_PREPARING = "false"
    exit 1
}

try {
    # 2. Perform VSS backup
    .\scripts\vss_backup.ps1 -VolumePath "C:" -BackupDestination $BackupDestination
    
} finally {
    # 3. Signal backup completion
    $env:RAG_VSS_BACKUP_PREPARING = "false"
    Remove-Item "C:\data\rag\.vss_ready" -ErrorAction SilentlyContinue
}
```

## Recovery Procedures Step-by-Step

### Data Validation After Recovery
```powershell
# File: scripts\validate_recovery.ps1
param(
    [Parameter(Mandatory=$true)]
    [string]$RecoveryPath,
    [switch]$Comprehensive
)

Write-Host "Starting post-recovery validation..."

# 1. Validate Tantivy indexes
Write-Host "Validating Tantivy indexes..."
$tantivyResult = & tantivy-cli check-integrity --index-path "$RecoveryPath\tantivy_indexes"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Tantivy index validation failed"
    exit 1
}

# 2. Validate LanceDB database
Write-Host "Validating LanceDB database..."
$lancedbResult = & lancedb-cli verify --db-path "$RecoveryPath\lancedb"
if ($LASTEXITCODE -ne 0) {
    Write-Error "LanceDB validation failed"
    exit 1
}

# 3. Test embedding cache integrity
Write-Host "Validating embedding caches..."
$cacheFiles = Get-ChildItem "$RecoveryPath\embeddings_cache" -Recurse -File
$corruptedCaches = @()

foreach ($cacheFile in $cacheFiles) {
    try {
        $content = Get-Content $cacheFile.FullName -Raw | ConvertFrom-Json
        if (-not $content.embedding -or $content.embedding.Count -eq 0) {
            $corruptedCaches += $cacheFile.FullName
        }
    } catch {
        $corruptedCaches += $cacheFile.FullName
    }
}

if ($corruptedCaches.Count -gt 0) {
    Write-Warning "Found $($corruptedCaches.Count) corrupted cache files:"
    $corruptedCaches | ForEach-Object { Write-Warning "  $_" }
}

# 4. Comprehensive testing if requested
if ($Comprehensive) {
    Write-Host "Running comprehensive system test..."
    
    # Test search functionality
    .\scripts\test_search_functionality.ps1 -TestQueries "C:\data\test_queries.txt"
    
    # Test embedding generation
    .\scripts\test_embedding_services.ps1 -AllServices
    
    # Test Git analysis
    .\scripts\test_git_analysis.ps1 -RepoPath "C:\data\test_repos"
}

Write-Host "Recovery validation completed successfully"
```

### Failover Mechanisms
```powershell
# File: scripts\automated_failover.ps1
param(
    [Parameter(Mandatory=$true)]
    [string]$PrimaryNode,
    [Parameter(Mandatory=$true)]
    [string]$SecondaryNode
)

# 1. Health check implementation
function Test-NodeHealth {
    param([string]$NodeAddress)
    
    try {
        $response = Invoke-RestMethod -Uri "http://$NodeAddress/health" -TimeoutSec 5
        return $response.status -eq "healthy"
    } catch {
        return $false
    }
}

# 2. Failover orchestration
$failoverTriggered = $false
$healthCheckInterval = 30  # seconds
$failureThreshold = 3     # consecutive failures

$primaryFailures = 0

while ($true) {
    $primaryHealthy = Test-NodeHealth -NodeAddress $PrimaryNode
    
    if ($primaryHealthy) {
        $primaryFailures = 0
        if ($failoverTriggered) {
            Write-Host "Primary node recovered. Consider failback operation."
        }
    } else {
        $primaryFailures++
        Write-Warning "Primary node health check failed ($primaryFailures/$failureThreshold)"
        
        if ($primaryFailures -ge $failureThreshold -and -not $failoverTriggered) {
            Write-Host "Triggering failover to secondary node..."
            
            # Update DNS/load balancer to point to secondary
            .\scripts\update_dns_record.ps1 -Target $SecondaryNode
            
            # Start secondary node services
            Invoke-Command -ComputerName $SecondaryNode -ScriptBlock {
                Start-Service -Name "UltimateRAGSearchEngine"
            }
            
            # Update monitoring systems
            .\scripts\notify_failover.ps1 -PrimaryNode $PrimaryNode -SecondaryNode $SecondaryNode
            
            $failoverTriggered = $true
            Write-Host "Failover completed successfully"
        }
    }
    
    Start-Sleep -Seconds $healthCheckInterval
}
```

## Testing Procedures for DR

### DR Test Scenarios
```powershell
# File: scripts\dr_test_scenarios.ps1
param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("CorruptionTest", "ServiceFailureTest", "NetworkPartitionTest", "StorageFailureTest", "CatastrophicTest")]
    [string]$TestScenario,
    [switch]$ExecuteRecovery = $false
)

switch ($TestScenario) {
    "CorruptionTest" {
        Write-Host "=== DR Test: Data Corruption Scenario ==="
        
        # 1. Create backup of current state
        .\scripts\create_test_backup.ps1
        
        # 2. Simulate corruption
        Write-Host "Simulating Tantivy index corruption..."
        $corruptFile = Get-ChildItem "C:\data\rag\tantivy_indexes" -File | Get-Random
        $corruptContent = [System.Text.Encoding]::UTF8.GetBytes("CORRUPTED_DATA_$(Get-Random)")
        [System.IO.File]::WriteAllBytes($corruptFile.FullName, $corruptContent)
        
        # 3. Detect corruption
        Write-Host "Testing corruption detection..."
        $detectionResult = & tantivy-cli check-integrity --index-path "C:\data\rag\tantivy_indexes"
        
        if ($ExecuteRecovery) {
            # 4. Execute recovery procedure
            Write-Host "Executing recovery procedure..."
            .\scripts\recover_from_corruption.ps1 -ComponentType "Tantivy"
            
            # 5. Validate recovery
            .\scripts\validate_recovery.ps1 -RecoveryPath "C:\data\rag" -Comprehensive
        } else {
            Write-Host "Test complete. Use -ExecuteRecovery to test recovery procedures."
            .\scripts\restore_test_backup.ps1
        }
    }
    
    "ServiceFailureTest" {
        Write-Host "=== DR Test: Service Failure Scenario ==="
        
        # 1. Simulate embedding service failures
        $env:RAG_SIMULATE_EMBEDDING_FAILURES = "VoyageCode2,E5Mistral,CodeBERT"
        
        # 2. Test fallback mechanisms
        .\scripts\test_embedding_fallback.ps1
        
        # 3. Test local-only mode
        $env:RAG_USE_LOCAL_EMBEDDINGS_ONLY = "true"
        .\scripts\test_local_embedding_mode.ps1
        
        # Cleanup
        Remove-Item Env:RAG_SIMULATE_EMBEDDING_FAILURES -ErrorAction SilentlyContinue
        Remove-Item Env:RAG_USE_LOCAL_EMBEDDINGS_ONLY -ErrorAction SilentlyContinue
    }
    
    "NetworkPartitionTest" {
        Write-Host "=== DR Test: Network Partition Scenario ==="
        
        # Simulate network partition using Windows Firewall
        Write-Host "Simulating network partition..."
        New-NetFirewallRule -DisplayName "DR_Test_Block" -Direction Inbound -Action Block -Protocol TCP -LocalPort 8080
        
        # Test split-brain prevention
        .\scripts\test_split_brain_prevention.ps1
        
        if ($ExecuteRecovery) {
            # Test partition recovery
            Remove-NetFirewallRule -DisplayName "DR_Test_Block"
            .\scripts\test_partition_recovery.ps1
        }
    }
    
    "StorageFailureTest" {
        Write-Host "=== DR Test: Storage Failure Scenario ==="
        
        # Simulate disk space exhaustion
        $testFile = "C:\data\rag\test_space_fill.tmp"
        $availableSpace = (Get-WmiObject -Class Win32_LogicalDisk | Where-Object { $_.DeviceID -eq "C:" }).FreeSpace
        $fillSize = $availableSpace - 100MB  # Leave 100MB free
        
        Write-Host "Simulating storage exhaustion..."
        fsutil file createnew $testFile $fillSize
        
        # Test emergency backup procedures
        .\scripts\test_emergency_backup.ps1
        
        # Cleanup
        Remove-Item $testFile -Force -ErrorAction SilentlyContinue
    }
    
    "CatastrophicTest" {
        Write-Host "=== DR Test: Catastrophic Failure Scenario ==="
        Write-Warning "This test requires a separate test environment!"
        
        if ($ExecuteRecovery) {
            # Test complete system rebuild
            .\scripts\test_complete_system_rebuild.ps1 -BackupSource "\\test-backup-server\rag-test-backups"
        } else {
            Write-Host "Catastrophic test requires -ExecuteRecovery flag and isolated test environment"
        }
    }
}

Write-Host "DR test scenario '$TestScenario' completed."
```

## Incident Response Playbooks

### Incident Classification Matrix
```
Priority 1 (Critical - 5 min response):
- Complete system failure
- Data corruption affecting search accuracy
- Security breach
- Data loss > RPO thresholds

Priority 2 (High - 30 min response):
- Single component failure with degraded service
- Performance degradation > 300% baseline
- Backup failures

Priority 3 (Medium - 2 hour response):
- Cache performance issues
- Non-critical component warnings
- Capacity planning alerts
```

### Incident Response Procedures
```powershell
# File: scripts\incident_response.ps1
param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("P1", "P2", "P3")]
    [string]$Priority,
    [Parameter(Mandatory=$true)]
    [string]$IncidentType,
    [Parameter(Mandatory=$true)]
    [string]$Description
)

$incidentId = "INC-$(Get-Date -Format 'yyyyMMdd-HHmmss')-$(Get-Random -Maximum 9999)"

# 1. Log incident
$incident = @{
    Id = $incidentId
    Priority = $Priority
    Type = $IncidentType
    Description = $Description
    Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Status = "INVESTIGATING"
} | ConvertTo-Json

$incident | Out-File "C:\logs\rag\incidents\$incidentId.json"

Write-Host "=== INCIDENT RESPONSE INITIATED ==="
Write-Host "Incident ID: $incidentId"
Write-Host "Priority: $Priority"
Write-Host "Type: $IncidentType"

switch ($Priority) {
    "P1" {
        Write-Host "=== P1 CRITICAL INCIDENT RESPONSE ==="
        
        # Immediate actions (within 5 minutes)
        Write-Host "1. Activating emergency mode..."
        $env:RAG_EMERGENCY_MODE = "true"
        
        Write-Host "2. Collecting system state..."
        .\scripts\collect_system_state.ps1 -OutputPath "C:\logs\rag\incidents\$incidentId\"
        
        Write-Host "3. Initiating emergency notifications..."
        .\scripts\send_emergency_notification.ps1 -IncidentId $incidentId -Priority $Priority
        
        Write-Host "4. Starting emergency backup..."
        Start-Job -ScriptBlock { .\scripts\emergency_backup.ps1 -IncidentId $using:incidentId }
        
        # Component-specific responses
        switch ($IncidentType) {
            "DATA_CORRUPTION" {
                Write-Host "5. Executing data corruption response..."
                .\scripts\respond_to_corruption.ps1 -IncidentId $incidentId
            }
            "COMPLETE_FAILURE" {
                Write-Host "5. Executing complete failure response..."
                .\scripts\respond_to_complete_failure.ps1 -IncidentId $incidentId
            }
            "SECURITY_BREACH" {
                Write-Host "5. Executing security incident response..."
                .\scripts\respond_to_security_breach.ps1 -IncidentId $incidentId
            }
        }
    }
    
    "P2" {
        Write-Host "=== P2 HIGH PRIORITY INCIDENT RESPONSE ==="
        
        Write-Host "1. Collecting diagnostic information..."
        .\scripts\collect_diagnostics.ps1 -IncidentId $incidentId
        
        Write-Host "2. Attempting automated recovery..."
        .\scripts\attempt_automated_recovery.ps1 -IncidentType $IncidentType
        
        Write-Host "3. Notifying on-call team..."
        .\scripts\send_notification.ps1 -IncidentId $incidentId -Priority $Priority
    }
    
    "P3" {
        Write-Host "=== P3 MEDIUM PRIORITY INCIDENT RESPONSE ==="
        
        Write-Host "1. Logging incident for investigation..."
        .\scripts\queue_for_investigation.ps1 -IncidentId $incidentId
        
        Write-Host "2. Collecting trend data..."
        .\scripts\analyze_trends.ps1 -IncidentType $IncidentType -TimeWindow "24h"
    }
}

Write-Host "Initial incident response completed for $incidentId"
```

## Backup Storage Requirements

### Storage Capacity Planning
```powershell
# File: scripts\calculate_backup_requirements.ps1

# Current system size analysis
$dataPath = "C:\data\rag"
$currentSize = (Get-ChildItem $dataPath -Recurse | Measure-Object -Property Length -Sum).Sum

Write-Host "=== Backup Storage Requirements Analysis ==="
Write-Host "Current system size: $([math]::Round($currentSize / 1GB, 2)) GB"

# Growth projections
$dailyGrowthRate = 0.05  # 5% daily growth
$retentionPeriods = @{
    "Hourly" = 24      # Keep 24 hourly backups
    "Daily" = 30       # Keep 30 daily backups  
    "Weekly" = 12      # Keep 12 weekly backups
    "Monthly" = 12     # Keep 12 monthly backups
}

$totalStorageRequired = 0

foreach ($period in $retentionPeriods.Keys) {
    $count = $retentionPeriods[$period]
    
    switch ($period) {
        "Hourly" {
            # Incremental backups (estimated 10% of full size)
            $backupSize = $currentSize * 0.1
            $totalPeriodSize = $backupSize * $count
        }
        "Daily" {
            # Differential backups (estimated 30% of full size)  
            $backupSize = $currentSize * 0.3
            $totalPeriodSize = $backupSize * $count
        }
        "Weekly" {
            # Full backups
            $backupSize = $currentSize
            $totalPeriodSize = $backupSize * $count
        }
        "Monthly" {
            # Compressed full backups (estimated 70% of full size)
            $backupSize = $currentSize * 0.7
            $totalPeriodSize = $backupSize * $count
        }
    }
    
    $totalStorageRequired += $totalPeriodSize
    Write-Host "$period backups: $([math]::Round($totalPeriodSize / 1GB, 2)) GB ($count backups @ $([math]::Round($backupSize / 1GB, 2)) GB each)"
}

# Add safety margin
$safetyMargin = $totalStorageRequired * 0.2  # 20% safety margin
$totalWithMargin = $totalStorageRequired + $safetyMargin

Write-Host ""
Write-Host "Total storage required: $([math]::Round($totalStorageRequired / 1GB, 2)) GB"
Write-Host "With 20% safety margin: $([math]::Round($totalWithMargin / 1GB, 2)) GB"
Write-Host ""
Write-Host "Recommended storage allocation: $([math]::Round($totalWithMargin / 1TB, 2)) TB"
```

### Backup Storage Configuration
```powershell
# File: scripts\configure_backup_storage.ps1
param(
    [Parameter(Mandatory=$true)]
    [string]$PrimaryBackupPath,
    [Parameter(Mandatory=$true)]
    [string]$SecondaryBackupPath,
    [string]$OffsiteBackupPath = $null
)

Write-Host "=== Configuring Backup Storage ==="

# 1. Primary backup storage (local/fast)
Write-Host "Configuring primary backup storage: $PrimaryBackupPath"
$primaryDrive = Split-Path $PrimaryBackupPath -Qualifier

if (!(Test-Path $PrimaryBackupPath)) {
    New-Item -ItemType Directory -Force -Path $PrimaryBackupPath
}

# Configure primary storage for performance
$primaryVolume = Get-Volume -DriveLetter $primaryDrive.TrimEnd(':')
if ($primaryVolume.FileSystemType -ne "NTFS") {
    Write-Warning "Primary backup drive should use NTFS for optimal performance"
}

# 2. Secondary backup storage (local/redundant)
Write-Host "Configuring secondary backup storage: $SecondaryBackupPath"
if (!(Test-Path $SecondaryBackupPath)) {
    New-Item -ItemType Directory -Force -Path $SecondaryBackupPath
}

# 3. Offsite backup storage (if configured)
if ($OffsiteBackupPath) {
    Write-Host "Configuring offsite backup storage: $OffsiteBackupPath"
    if (!(Test-Path $OffsiteBackupPath)) {
        New-Item -ItemType Directory -Force -Path $OffsiteBackupPath
    }
}

# 4. Create backup configuration
$backupConfig = @{
    Primary = @{
        Path = $PrimaryBackupPath
        Type = "Local_Fast"
        RetentionPolicy = @{
            Hourly = 24
            Daily = 7
        }
        CompressionLevel = "None"  # Fast backup/restore
    }
    Secondary = @{
        Path = $SecondaryBackupPath
        Type = "Local_Redundant"
        RetentionPolicy = @{
            Daily = 30
            Weekly = 12
        }
        CompressionLevel = "Standard"  # Balanced
    }
    Offsite = @{
        Path = $OffsiteBackupPath
        Type = "Remote_Archive"
        RetentionPolicy = @{
            Weekly = 12
            Monthly = 24
            Yearly = 7
        }
        CompressionLevel = "Maximum"  # Space efficient
    }
}

$backupConfig | ConvertTo-Json -Depth 4 | Out-File "C:\config\rag\backup_config.json"

Write-Host "Backup storage configuration completed"
Write-Host "Configuration saved to: C:\config\rag\backup_config.json"
```

## Monitoring and Alerting

### Health Monitoring System
```powershell
# File: scripts\health_monitoring.ps1
param(
    [int]$MonitoringInterval = 60,  # seconds
    [string]$ConfigPath = "C:\config\rag\monitoring_config.json"
)

# Load monitoring configuration
$config = Get-Content $ConfigPath | ConvertFrom-Json

function Test-ComponentHealth {
    param(
        [string]$ComponentName,
        [hashtable]$ComponentConfig
    )
    
    $healthStatus = @{
        Component = $ComponentName
        Status = "Unknown"
        Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Metrics = @{}
        Alerts = @()
    }
    
    switch ($ComponentName) {
        "Tantivy" {
            try {
                # Test index accessibility and integrity
                $indexPath = $ComponentConfig.IndexPath
                $testResult = & tantivy-cli check-integrity --index-path $indexPath --timeout=30
                
                if ($LASTEXITCODE -eq 0) {
                    $healthStatus.Status = "Healthy"
                } else {
                    $healthStatus.Status = "Unhealthy"
                    $healthStatus.Alerts += "Index integrity check failed"
                }
                
                # Collect metrics
                $indexSize = (Get-ChildItem $indexPath -Recurse | Measure-Object -Property Length -Sum).Sum
                $healthStatus.Metrics.IndexSizeGB = [math]::Round($indexSize / 1GB, 2)
                
            } catch {
                $healthStatus.Status = "Error"
                $healthStatus.Alerts += "Failed to check Tantivy health: $_"
            }
        }
        
        "LanceDB" {
            try {
                # Test database connectivity and transaction capability
                $dbPath = $ComponentConfig.DatabasePath
                $testResult = & lancedb-cli verify --db-path $dbPath --timeout=30
                
                if ($LASTEXITCODE -eq 0) {
                    $healthStatus.Status = "Healthy"
                } else {
                    $healthStatus.Status = "Unhealthy"
                    $healthStatus.Alerts += "Database verification failed"
                }
                
                # Collect metrics
                $dbSize = (Get-ChildItem $dbPath -Recurse | Measure-Object -Property Length -Sum).Sum
                $healthStatus.Metrics.DatabaseSizeGB = [math]::Round($dbSize / 1GB, 2)
                
            } catch {
                $healthStatus.Status = "Error"
                $healthStatus.Alerts += "Failed to check LanceDB health: $_"
            }
        }
        
        "EmbeddingServices" {
            $services = @("VoyageCode2", "E5Mistral", "BGE_M3", "CodeBERT", "SQLCoder", "BERTConfig", "StackTraceBERT")
            $healthyServices = 0
            
            foreach ($service in $services) {
                try {
                    $testResult = Invoke-RestMethod -Uri "http://localhost:8080/test_embedding/$service" -TimeoutSec 10
                    if ($testResult.status -eq "ok") {
                        $healthyServices++
                    }
                } catch {
                    $healthStatus.Alerts += "Embedding service $service is unreachable"
                }
            }
            
            $healthStatus.Metrics.HealthyServicesCount = $healthyServices
            $healthStatus.Metrics.TotalServicesCount = $services.Count
            
            if ($healthyServices -eq $services.Count) {
                $healthStatus.Status = "Healthy"
            } elseif ($healthyServices -gt 0) {
                $healthStatus.Status = "Degraded"
            } else {
                $healthStatus.Status = "Unhealthy"
            }
        }
        
        "SystemResources" {
            # CPU Usage
            $cpu = Get-Counter "\Processor(_Total)\% Processor Time" -SampleInterval 1 -MaxSamples 3
            $avgCpuUsage = ($cpu.CounterSamples | Measure-Object -Property CookedValue -Average).Average
            
            # Memory Usage
            $memory = Get-WmiObject -Class Win32_OperatingSystem
            $memoryUsagePercent = (($memory.TotalVisibleMemorySize - $memory.FreePhysicalMemory) / $memory.TotalVisibleMemorySize) * 100
            
            # Disk Usage
            $disk = Get-WmiObject -Class Win32_LogicalDisk | Where-Object { $_.DeviceID -eq "C:" }
            $diskUsagePercent = (($disk.Size - $disk.FreeSpace) / $disk.Size) * 100
            
            $healthStatus.Metrics.CpuUsagePercent = [math]::Round($avgCpuUsage, 2)
            $healthStatus.Metrics.MemoryUsagePercent = [math]::Round($memoryUsagePercent, 2)
            $healthStatus.Metrics.DiskUsagePercent = [math]::Round($diskUsagePercent, 2)
            
            # Determine health status based on thresholds
            if ($avgCpuUsage -gt 90 -or $memoryUsagePercent -gt 90 -or $diskUsagePercent -gt 90) {
                $healthStatus.Status = "Critical"
                $healthStatus.Alerts += "Resource usage critical"
            } elseif ($avgCpuUsage -gt 70 -or $memoryUsagePercent -gt 70 -or $diskUsagePercent -gt 80) {
                $healthStatus.Status = "Warning"
                $healthStatus.Alerts += "High resource usage detected"
            } else {
                $healthStatus.Status = "Healthy"
            }
        }
    }
    
    return $healthStatus
}

# Main monitoring loop
Write-Host "Starting health monitoring system..."
Write-Host "Monitoring interval: $MonitoringInterval seconds"

while ($true) {
    $overallHealth = @{
        Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Components = @{}
        OverallStatus = "Healthy"
        CriticalAlerts = @()
    }
    
    # Check each component
    foreach ($component in $config.Components.PSObject.Properties) {
        $componentName = $component.Name
        $componentConfig = $component.Value
        
        $healthStatus = Test-ComponentHealth -ComponentName $componentName -ComponentConfig $componentConfig
        $overallHealth.Components[$componentName] = $healthStatus
        
        # Update overall status
        if ($healthStatus.Status -eq "Error" -or $healthStatus.Status -eq "Critical") {
            $overallHealth.OverallStatus = "Critical"
            $overallHealth.CriticalAlerts += $healthStatus.Alerts
        } elseif ($healthStatus.Status -eq "Unhealthy" -and $overallHealth.OverallStatus -ne "Critical") {
            $overallHealth.OverallStatus = "Unhealthy"
        } elseif ($healthStatus.Status -eq "Warning" -and $overallHealth.OverallStatus -eq "Healthy") {
            $overallHealth.OverallStatus = "Warning"
        }
    }
    
    # Log health status
    $healthLogPath = "C:\logs\rag\health\health_$(Get-Date -Format 'yyyyMMdd').json"
    $overallHealth | ConvertTo-Json -Depth 4 | Out-File $healthLogPath -Append
    
    # Send alerts if necessary
    if ($overallHealth.OverallStatus -eq "Critical") {
        .\scripts\send_critical_alert.ps1 -HealthStatus $overallHealth
    }
    
    # Console output
    Write-Host "$(Get-Date -Format 'HH:mm:ss') - Overall Status: $($overallHealth.OverallStatus)"
    
    Start-Sleep -Seconds $MonitoringInterval
}
```

## Summary

This disaster recovery plan provides comprehensive protection for the Ultimate RAG System with:

- **RTO/RPO Compliance**: Tier-based recovery objectives from 5 minutes to 2 hours
- **Multi-Layer Backup Strategy**: Incremental, full, and continuous backup approaches
- **Windows-Specific Integration**: VSS snapshots and Windows service management
- **Automated Recovery**: PowerShell scripts for rapid restoration
- **Comprehensive Testing**: DR test scenarios and validation procedures
- **Incident Response**: Prioritized response playbooks with automated notifications

**Critical Note**: All procedures require validation during implementation as this system analysis was performed with no prior operational context. Regular DR testing is essential to ensure procedures remain current and effective.

**Next Steps**:
1. Implement monitoring and alerting systems
2. Test all DR procedures in isolated environment
3. Establish offsite backup infrastructure
4. Train operations team on incident response procedures
5. Schedule regular DR drills and procedure updates