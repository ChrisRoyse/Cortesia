# Git File Watcher Design for MCP Server

## Executive Summary

This document specifies a comprehensive Git file watcher system for the LLMKG MCP server that provides intelligent, incremental reindexing capabilities. The system integrates with the existing neuromorphic temporal analysis framework to deliver sub-100ms response times while maintaining biological accuracy through Git-aware change detection.

## Architecture Overview

```typescript
┌───────────────────────────────────────────────────────────────────┐
│                     Git File Watcher System                      │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────┐    ┌──────────────────────────────────┐  │
│  │   FileSystemWatcher │    │       Git Integration            │  │
│  │                     │    │                                  │  │
│  │ • chokidar-based    │────┼──┐ ┌─────────────────────────────┐ │  │
│  │ • Cross-platform    │    │  └─┤   PostCommitHandler        │ │  │
│  │ • Debounced events  │    │    │ • Immediate reindexing     │ │  │
│  │ • Ignore patterns   │    │    │ • Diff-based updates       │ │  │
│  └─────────────────────┘    │    │ • Conflict resolution      │ │  │
│                             │    └─────────────────────────────┘ │  │
│  ┌─────────────────────┐    │                                  │  │
│  │  ChangeDetector     │    │  ┌─────────────────────────────┐ │  │
│  │                     │────┼──┤   PostCheckoutHandler      │ │  │
│  │ • Git diff analysis │    │  │ • Branch switch detection  │ │  │
│  │ • Content hashing   │    │  │ • Bulk reindexing          │ │  │
│  │ • Metadata tracking │    │  │ • Context preservation     │ │  │
│  └─────────────────────┘    │  └─────────────────────────────┘ │  │
│                             │                                  │  │
│  ┌─────────────────────┐    │  ┌─────────────────────────────┐ │  │
│  │ IncrementalIndexer  │    │  │    ReindexingEngine         │ │  │
│  │                     │────┼──┤ • Background processing     │ │  │
│  │ • Changed files only│    │  │ • Batch operations          │ │  │
│  │ • Embedding updates │    │  │ • Priority queuing          │ │  │
│  │ • Index consistency │    │  │ • Progress tracking         │ │  │
│  └─────────────────────┘    │  └─────────────────────────────┘ │  │
│                             │                                  │  │
│  ┌─────────────────────┐    │  ┌─────────────────────────────┐ │  │
│  │   MCPNotifier       │    │  │    PerformanceOptimizer    │ │  │
│  │                     │────┼──┤ • Batching strategies       │ │  │
│  │ • Progress updates  │    │  │ • Resource management       │ │  │
│  │ • Status broadcasts │    │  │ • Adaptive throttling       │ │  │
│  │ • Error notifications│   │  │ • Memory optimization       │ │  │
│  └─────────────────────┘    │  └─────────────────────────────┘ │  │
└───────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. File System Watcher

Cross-platform file monitoring using chokidar with intelligent filtering and debouncing:

```typescript
import chokidar from 'chokidar';
import { EventEmitter } from 'events';
import path from 'path';
import { minimatch } from 'minimatch';

interface WatcherConfig {
  debounceMs: number;
  ignorePatterns: string[];
  watchPatterns: string[];
  persistent: boolean;
  usePolling: boolean;
  pollingInterval: number;
}

class GitAwareFileWatcher extends EventEmitter {
  private watcher: chokidar.FSWatcher | null = null;
  private debounceTimers = new Map<string, NodeJS.Timeout>();
  private changeBuffer = new Map<string, FileChangeEvent>();
  
  constructor(
    private rootPath: string,
    private config: WatcherConfig = {
      debounceMs: 300,
      ignorePatterns: [
        '**/node_modules/**',
        '**/target/**',
        '**/.git/**',
        '**/build/**',
        '**/*.tmp',
        '**/*.log',
        '**/coverage/**'
      ],
      watchPatterns: ['**/*'],
      persistent: true,
      usePolling: false,
      pollingInterval: 1000
    }
  ) {
    super();
  }

  async start(): Promise<void> {
    if (this.watcher) {
      throw new Error('Watcher already started');
    }

    this.watcher = chokidar.watch(this.config.watchPatterns, {
      cwd: this.rootPath,
      ignored: this.config.ignorePatterns,
      persistent: this.config.persistent,
      usePolling: this.config.usePolling,
      interval: this.config.pollingInterval,
      binaryInterval: this.config.pollingInterval * 2,
      depth: 10,
      awaitWriteFinish: {
        stabilityThreshold: 100,
        pollInterval: 50
      }
    });

    this.watcher
      .on('add', (filePath: string) => this.handleFileEvent('add', filePath))
      .on('change', (filePath: string) => this.handleFileEvent('change', filePath))
      .on('unlink', (filePath: string) => this.handleFileEvent('delete', filePath))
      .on('addDir', (dirPath: string) => this.handleDirectoryEvent('add', dirPath))
      .on('unlinkDir', (dirPath: string) => this.handleDirectoryEvent('delete', dirPath))
      .on('error', (error: Error) => this.emit('error', error))
      .on('ready', () => this.emit('ready'));

    console.log(`[GitFileWatcher] Started watching: ${this.rootPath}`);
  }

  private handleFileEvent(eventType: 'add' | 'change' | 'delete', filePath: string): void {
    if (this.shouldIgnoreFile(filePath)) {
      return;
    }

    const absolutePath = path.resolve(this.rootPath, filePath);
    const changeEvent: FileChangeEvent = {
      type: eventType,
      path: absolutePath,
      relativePath: filePath,
      timestamp: new Date(),
      size: this.getFileSize(absolutePath)
    };

    // Clear existing debounce timer
    const existingTimer = this.debounceTimers.get(absolutePath);
    if (existingTimer) {
      clearTimeout(existingTimer);
    }

    // Buffer the change
    this.changeBuffer.set(absolutePath, changeEvent);

    // Set new debounce timer
    const debounceTimer = setTimeout(() => {
      const bufferedEvent = this.changeBuffer.get(absolutePath);
      if (bufferedEvent) {
        this.changeBuffer.delete(absolutePath);
        this.debounceTimers.delete(absolutePath);
        this.emit('fileChange', bufferedEvent);
      }
    }, this.config.debounceMs);

    this.debounceTimers.set(absolutePath, debounceTimer);
  }

  private shouldIgnoreFile(filePath: string): boolean {
    return this.config.ignorePatterns.some(pattern => 
      minimatch(filePath, pattern, { dot: true })
    );
  }

  private getFileSize(filePath: string): number | undefined {
    try {
      return require('fs').statSync(filePath).size;
    } catch {
      return undefined;
    }
  }

  async stop(): Promise<void> {
    if (this.watcher) {
      await this.watcher.close();
      this.watcher = null;
    }

    // Clear all debounce timers
    for (const timer of this.debounceTimers.values()) {
      clearTimeout(timer);
    }
    this.debounceTimers.clear();
    this.changeBuffer.clear();

    console.log('[GitFileWatcher] Stopped watching');
  }
}

interface FileChangeEvent {
  type: 'add' | 'change' | 'delete';
  path: string;
  relativePath: string;
  timestamp: Date;
  size?: number;
}
```

### 2. Git Integration System

Integration with Git hooks and repository analysis for intelligent change detection:

```typescript
import { exec } from 'child_process';
import { promisify } from 'util';
import crypto from 'crypto';
import fs from 'fs/promises';

const execAsync = promisify(exec);

interface GitCommitInfo {
  hash: string;
  author: string;
  timestamp: Date;
  message: string;
  changedFiles: string[];
}

interface GitDiffResult {
  addedFiles: string[];
  modifiedFiles: string[];
  deletedFiles: string[];
  renamedFiles: Array<{ from: string; to: string }>;
}

class GitIntegrationManager {
  constructor(private repositoryPath: string) {}

  async setupHooks(): Promise<void> {
    const hooksDir = path.join(this.repositoryPath, '.git', 'hooks');
    
    // Post-commit hook for immediate reindexing
    const postCommitHook = `#!/bin/sh
# LLMKG MCP Server - Post-commit hook
# Trigger immediate reindexing of changed files

if command -v node >/dev/null 2>&1; then
  node -e "
    const { MCPServer } = require('./mcp-server');
    const server = new MCPServer();
    server.triggerPostCommitReindex(process.env.GIT_COMMIT || 'HEAD');
  " 2>/dev/null || true
fi
`;

    // Post-checkout hook for branch switches
    const postCheckoutHook = `#!/bin/sh
# LLMKG MCP Server - Post-checkout hook
# Handle branch switches and bulk reindexing

PREV_HEAD=$1
NEW_HEAD=$2
BRANCH_FLAG=$3

if [ "$BRANCH_FLAG" = "1" ]; then
  if command -v node >/dev/null 2>&1; then
    node -e "
      const { MCPServer } = require('./mcp-server');
      const server = new MCPServer();
      server.triggerBranchSwitch('$PREV_HEAD', '$NEW_HEAD');
    " 2>/dev/null || true
  fi
fi
`;

    await fs.writeFile(path.join(hooksDir, 'post-commit'), postCommitHook, { mode: 0o755 });
    await fs.writeFile(path.join(hooksDir, 'post-checkout'), postCheckoutHook, { mode: 0o755 });
    
    console.log('[GitIntegration] Git hooks installed successfully');
  }

  async getCommitDiff(fromCommit: string, toCommit: string = 'HEAD'): Promise<GitDiffResult> {
    try {
      const { stdout } = await execAsync(
        `git diff --name-status ${fromCommit}..${toCommit}`,
        { cwd: this.repositoryPath }
      );

      const result: GitDiffResult = {
        addedFiles: [],
        modifiedFiles: [],
        deletedFiles: [],
        renamedFiles: []
      };

      for (const line of stdout.trim().split('\n')) {
        if (!line) continue;

        const [status, ...fileParts] = line.split('\t');
        const filePath = fileParts.join('\t');

        switch (status[0]) {
          case 'A':
            result.addedFiles.push(filePath);
            break;
          case 'M':
            result.modifiedFiles.push(filePath);
            break;
          case 'D':
            result.deletedFiles.push(filePath);
            break;
          case 'R':
            const [fromFile, toFile] = fileParts;
            result.renamedFiles.push({ from: fromFile, to: toFile });
            break;
        }
      }

      return result;
    } catch (error) {
      throw new Error(`Failed to get git diff: ${error.message}`);
    }
  }

  async getCurrentCommit(): Promise<string> {
    try {
      const { stdout } = await execAsync('git rev-parse HEAD', { cwd: this.repositoryPath });
      return stdout.trim();
    } catch (error) {
      throw new Error(`Failed to get current commit: ${error.message}`);
    }
  }

  async getCommitInfo(commitHash: string): Promise<GitCommitInfo> {
    try {
      const { stdout } = await execAsync(
        `git show --format="%H|%an|%at|%s" --name-only ${commitHash}`,
        { cwd: this.repositoryPath }
      );

      const lines = stdout.trim().split('\n');
      const [hash, author, timestamp, message] = lines[0].split('|');
      const changedFiles = lines.slice(2).filter(line => line.trim());

      return {
        hash,
        author,
        timestamp: new Date(parseInt(timestamp) * 1000),
        message,
        changedFiles
      };
    } catch (error) {
      throw new Error(`Failed to get commit info: ${error.message}`);
    }
  }

  async getBranchName(): Promise<string> {
    try {
      const { stdout } = await execAsync('git branch --show-current', { cwd: this.repositoryPath });
      return stdout.trim();
    } catch (error) {
      return 'main'; // fallback
    }
  }

  async getFileContentHash(filePath: string, commitHash?: string): Promise<string> {
    try {
      const gitPath = commitHash ? `${commitHash}:${filePath}` : filePath;
      const { stdout } = await execAsync(
        `git show ${gitPath}`,
        { cwd: this.repositoryPath }
      );
      
      return crypto.createHash('sha256').update(stdout).digest('hex');
    } catch (error) {
      // File might not exist in the specified commit
      return '';
    }
  }
}
```

### 3. Incremental Reindexing Engine

Efficient update system that only processes changed files while maintaining index consistency:

```typescript
interface IndexEntry {
  filePath: string;
  contentHash: string;
  lastModified: Date;
  commitHash: string;
  embedding?: number[];
  metadata: FileMetadata;
}

interface FileMetadata {
  size: number;
  language: string;
  author: string;
  lastCommitMessage: string;
  changeFrequency: number;
}

interface ReindexingJob {
  id: string;
  type: 'incremental' | 'bulk' | 'emergency';
  priority: number;
  files: string[];
  commitHash?: string;
  branchName?: string;
  createdAt: Date;
  estimatedDuration: number;
}

class IncrementalReindexingEngine extends EventEmitter {
  private indexMap = new Map<string, IndexEntry>();
  private jobQueue: ReindexingJob[] = [];
  private isProcessing = false;
  private batchSize = 50;
  private maxConcurrentJobs = 4;

  constructor(
    private gitManager: GitIntegrationManager,
    private embeddingService: EmbeddingService,
    private mcpNotifier: MCPNotificationService
  ) {
    super();
  }

  async handleFileChange(changeEvent: FileChangeEvent): Promise<void> {
    const jobId = crypto.randomUUID();
    
    const job: ReindexingJob = {
      id: jobId,
      type: 'incremental',
      priority: this.calculatePriority(changeEvent),
      files: [changeEvent.path],
      createdAt: new Date(),
      estimatedDuration: this.estimateProcessingTime([changeEvent.path])
    };

    this.enqueueJob(job);
    this.mcpNotifier.notifyJobQueued(job);
  }

  async handleCommitEvent(commitHash: string): Promise<void> {
    try {
      const currentCommit = await this.gitManager.getCurrentCommit();
      const diff = await this.gitManager.getCommitDiff(
        this.getLastProcessedCommit(),
        currentCommit
      );

      const allChangedFiles = [
        ...diff.addedFiles,
        ...diff.modifiedFiles,
        ...diff.renamedFiles.map(r => r.to)
      ];

      if (allChangedFiles.length === 0) {
        return;
      }

      const job: ReindexingJob = {
        id: crypto.randomUUID(),
        type: 'incremental',
        priority: 8, // High priority for commit-based changes
        files: allChangedFiles,
        commitHash: currentCommit,
        createdAt: new Date(),
        estimatedDuration: this.estimateProcessingTime(allChangedFiles)
      };

      this.enqueueJob(job);
      this.mcpNotifier.notifyCommitProcessing(commitHash, allChangedFiles.length);
    } catch (error) {
      this.emit('error', new Error(`Failed to handle commit event: ${error.message}`));
    }
  }

  async handleBranchSwitch(fromCommit: string, toCommit: string): Promise<void> {
    try {
      const diff = await this.gitManager.getCommitDiff(fromCommit, toCommit);
      
      const allChangedFiles = [
        ...diff.addedFiles,
        ...diff.modifiedFiles,
        ...diff.renamedFiles.map(r => r.to)
      ];

      // Handle deletions
      for (const deletedFile of diff.deletedFiles) {
        this.indexMap.delete(deletedFile);
      }

      // Handle renames
      for (const rename of diff.renamedFiles) {
        const oldEntry = this.indexMap.get(rename.from);
        if (oldEntry) {
          this.indexMap.delete(rename.from);
          this.indexMap.set(rename.to, { ...oldEntry, filePath: rename.to });
        }
      }

      if (allChangedFiles.length > 0) {
        const job: ReindexingJob = {
          id: crypto.randomUUID(),
          type: 'bulk',
          priority: 9, // Highest priority for branch switches
          files: allChangedFiles,
          commitHash: toCommit,
          branchName: await this.gitManager.getBranchName(),
          createdAt: new Date(),
          estimatedDuration: this.estimateProcessingTime(allChangedFiles)
        };

        this.enqueueJob(job);
        this.mcpNotifier.notifyBranchSwitch(fromCommit, toCommit, allChangedFiles.length);
      }
    } catch (error) {
      this.emit('error', new Error(`Failed to handle branch switch: ${error.message}`));
    }
  }

  private enqueueJob(job: ReindexingJob): void {
    // Insert job in priority order
    const insertIndex = this.jobQueue.findIndex(existingJob => 
      existingJob.priority < job.priority
    );
    
    if (insertIndex === -1) {
      this.jobQueue.push(job);
    } else {
      this.jobQueue.splice(insertIndex, 0, job);
    }

    this.processQueue();
  }

  private async processQueue(): Promise<void> {
    if (this.isProcessing || this.jobQueue.length === 0) {
      return;
    }

    this.isProcessing = true;

    try {
      while (this.jobQueue.length > 0) {
        const job = this.jobQueue.shift()!;
        await this.processJob(job);
      }
    } finally {
      this.isProcessing = false;
    }
  }

  private async processJob(job: ReindexingJob): Promise<void> {
    try {
      this.mcpNotifier.notifyJobStarted(job);
      
      const batches = this.createBatches(job.files, this.batchSize);
      let processedFiles = 0;

      for (const batch of batches) {
        const batchPromises = batch.map(filePath => 
          this.processFile(filePath, job.commitHash)
        );

        const results = await Promise.allSettled(batchPromises);
        
        // Handle results and errors
        for (let i = 0; i < results.length; i++) {
          const result = results[i];
          const filePath = batch[i];
          
          if (result.status === 'fulfilled') {
            processedFiles++;
            this.mcpNotifier.notifyFileProcessed(filePath, job.id);
          } else {
            this.emit('error', new Error(
              `Failed to process file ${filePath}: ${result.reason.message}`
            ));
          }
        }

        // Report progress
        const progress = processedFiles / job.files.length;
        this.mcpNotifier.notifyJobProgress(job.id, progress);

        // Small delay to prevent overwhelming the system
        await new Promise(resolve => setTimeout(resolve, 10));
      }

      this.mcpNotifier.notifyJobCompleted(job, processedFiles);
    } catch (error) {
      this.mcpNotifier.notifyJobFailed(job, error.message);
      this.emit('error', error);
    }
  }

  private async processFile(filePath: string, commitHash?: string): Promise<void> {
    try {
      // Get current file content hash
      const currentHash = await this.gitManager.getFileContentHash(filePath, commitHash);
      const existingEntry = this.indexMap.get(filePath);

      // Skip if file hasn't changed
      if (existingEntry && existingEntry.contentHash === currentHash) {
        return;
      }

      // Read file content
      const content = await fs.readFile(filePath, 'utf-8');
      
      // Generate embedding
      const embedding = await this.embeddingService.generateEmbedding(content);
      
      // Get file metadata
      const metadata = await this.extractFileMetadata(filePath, commitHash);

      // Update index
      const indexEntry: IndexEntry = {
        filePath,
        contentHash: currentHash,
        lastModified: new Date(),
        commitHash: commitHash || await this.gitManager.getCurrentCommit(),
        embedding,
        metadata
      };

      this.indexMap.set(filePath, indexEntry);
      
      // Persist to storage
      await this.persistIndexEntry(indexEntry);
      
    } catch (error) {
      throw new Error(`Failed to process file ${filePath}: ${error.message}`);
    }
  }

  private calculatePriority(changeEvent: FileChangeEvent): number {
    let priority = 5; // Base priority

    // Increase priority for recently modified files
    const timeSinceChange = Date.now() - changeEvent.timestamp.getTime();
    if (timeSinceChange < 5000) priority += 2; // Last 5 seconds

    // Increase priority for important file types
    const extension = path.extname(changeEvent.path);
    if (['.rs', '.ts', '.js', '.py', '.md'].includes(extension)) {
      priority += 1;
    }

    // Increase priority for smaller files (faster to process)
    if (changeEvent.size && changeEvent.size < 10000) { // 10KB
      priority += 1;
    }

    return Math.min(priority, 10); // Cap at 10
  }

  private estimateProcessingTime(files: string[]): number {
    // Rough estimation: 100ms per file + overhead
    return files.length * 100 + 500;
  }

  private createBatches<T>(items: T[], batchSize: number): T[][] {
    const batches: T[][] = [];
    for (let i = 0; i < items.length; i += batchSize) {
      batches.push(items.slice(i, i + batchSize));
    }
    return batches;
  }

  private async extractFileMetadata(filePath: string, commitHash?: string): Promise<FileMetadata> {
    try {
      const stats = await fs.stat(filePath);
      const commitInfo = commitHash ? 
        await this.gitManager.getCommitInfo(commitHash) :
        null;

      return {
        size: stats.size,
        language: this.detectLanguage(filePath),
        author: commitInfo?.author || 'unknown',
        lastCommitMessage: commitInfo?.message || '',
        changeFrequency: this.calculateChangeFrequency(filePath)
      };
    } catch (error) {
      return {
        size: 0,
        language: 'unknown',
        author: 'unknown',
        lastCommitMessage: '',
        changeFrequency: 0
      };
    }
  }

  private detectLanguage(filePath: string): string {
    const extension = path.extname(filePath).toLowerCase();
    const languageMap: Record<string, string> = {
      '.rs': 'rust',
      '.ts': 'typescript',
      '.js': 'javascript',
      '.py': 'python',
      '.md': 'markdown',
      '.json': 'json',
      '.yml': 'yaml',
      '.yaml': 'yaml',
      '.toml': 'toml'
    };
    
    return languageMap[extension] || 'text';
  }

  private calculateChangeFrequency(filePath: string): number {
    // This would integrate with the temporal analysis system
    // For now, return a placeholder
    return 1;
  }

  private getLastProcessedCommit(): string {
    // This would be persisted in the database
    // For now, return a placeholder
    return 'HEAD~1';
  }

  private async persistIndexEntry(entry: IndexEntry): Promise<void> {
    // This would integrate with the neuromorphic storage system
    // For now, just log
    console.log(`[IncrementalIndexer] Persisted entry for ${entry.filePath}`);
  }
}
```

### 4. MCP Notification System

Real-time notifications to connected LLMs about index updates and system status:

```typescript
interface MCPNotification {
  type: 'job_queued' | 'job_started' | 'job_progress' | 'job_completed' | 'job_failed' | 
        'file_processed' | 'commit_processing' | 'branch_switch' | 'system_status';
  timestamp: Date;
  data: any;
}

interface SystemStatus {
  isProcessing: boolean;
  queueLength: number;
  totalFilesProcessed: number;
  averageProcessingTime: number;
  lastError?: string;
  uptime: number;
}

class MCPNotificationService extends EventEmitter {
  private connectedClients = new Set<MCPConnection>();
  private notificationBuffer: MCPNotification[] = [];
  private maxBufferSize = 1000;
  private systemStatus: SystemStatus = {
    isProcessing: false,
    queueLength: 0,
    totalFilesProcessed: 0,
    averageProcessingTime: 0,
    uptime: Date.now()
  };

  constructor() {
    super();
    
    // Periodic status updates
    setInterval(() => {
      this.broadcastSystemStatus();
    }, 30000); // Every 30 seconds
  }

  registerClient(connection: MCPConnection): void {
    this.connectedClients.add(connection);
    
    connection.on('disconnect', () => {
      this.connectedClients.delete(connection);
    });

    // Send current system status to new client
    this.sendNotification(connection, {
      type: 'system_status',
      timestamp: new Date(),
      data: this.systemStatus
    });

    console.log(`[MCPNotifier] Client registered. Total clients: ${this.connectedClients.size}`);
  }

  notifyJobQueued(job: ReindexingJob): void {
    this.systemStatus.queueLength++;
    
    const notification: MCPNotification = {
      type: 'job_queued',
      timestamp: new Date(),
      data: {
        jobId: job.id,
        type: job.type,
        priority: job.priority,
        fileCount: job.files.length,
        estimatedDuration: job.estimatedDuration
      }
    };

    this.broadcast(notification);
  }

  notifyJobStarted(job: ReindexingJob): void {
    this.systemStatus.isProcessing = true;
    this.systemStatus.queueLength--;

    const notification: MCPNotification = {
      type: 'job_started',
      timestamp: new Date(),
      data: {
        jobId: job.id,
        type: job.type,
        fileCount: job.files.length,
        branchName: job.branchName
      }
    };

    this.broadcast(notification);
  }

  notifyJobProgress(jobId: string, progress: number): void {
    const notification: MCPNotification = {
      type: 'job_progress',
      timestamp: new Date(),
      data: {
        jobId,
        progress: Math.round(progress * 100), // Convert to percentage
        message: `Processing... ${Math.round(progress * 100)}% complete`
      }
    };

    this.broadcast(notification);
  }

  notifyJobCompleted(job: ReindexingJob, processedFiles: number): void {
    this.systemStatus.isProcessing = false;
    this.systemStatus.totalFilesProcessed += processedFiles;

    const notification: MCPNotification = {
      type: 'job_completed',
      timestamp: new Date(),
      data: {
        jobId: job.id,
        type: job.type,
        processedFiles,
        totalFiles: job.files.length,
        duration: Date.now() - job.createdAt.getTime(),
        successRate: (processedFiles / job.files.length) * 100
      }
    };

    this.broadcast(notification);
  }

  notifyJobFailed(job: ReindexingJob, error: string): void {
    this.systemStatus.isProcessing = false;
    this.systemStatus.lastError = error;

    const notification: MCPNotification = {
      type: 'job_failed',
      timestamp: new Date(),
      data: {
        jobId: job.id,
        type: job.type,
        error,
        fileCount: job.files.length
      }
    };

    this.broadcast(notification);
  }

  notifyFileProcessed(filePath: string, jobId: string): void {
    const notification: MCPNotification = {
      type: 'file_processed',
      timestamp: new Date(),
      data: {
        filePath,
        jobId,
        language: this.detectLanguage(filePath)
      }
    };

    this.broadcast(notification);
  }

  notifyCommitProcessing(commitHash: string, fileCount: number): void {
    const notification: MCPNotification = {
      type: 'commit_processing',
      timestamp: new Date(),
      data: {
        commitHash: commitHash.substring(0, 8), // Short hash
        fileCount,
        message: `Processing ${fileCount} files from commit ${commitHash.substring(0, 8)}`
      }
    };

    this.broadcast(notification);
  }

  notifyBranchSwitch(fromCommit: string, toCommit: string, fileCount: number): void {
    const notification: MCPNotification = {
      type: 'branch_switch',
      timestamp: new Date(),
      data: {
        fromCommit: fromCommit.substring(0, 8),
        toCommit: toCommit.substring(0, 8),
        fileCount,
        message: `Branch switch detected: ${fileCount} files to reindex`
      }
    };

    this.broadcast(notification);
  }

  private broadcastSystemStatus(): void {
    const notification: MCPNotification = {
      type: 'system_status',
      timestamp: new Date(),
      data: {
        ...this.systemStatus,
        uptime: Date.now() - this.systemStatus.uptime
      }
    };

    this.broadcast(notification);
  }

  private broadcast(notification: MCPNotification): void {
    // Add to buffer
    this.notificationBuffer.push(notification);
    if (this.notificationBuffer.length > this.maxBufferSize) {
      this.notificationBuffer.shift(); // Remove oldest
    }

    // Send to all connected clients
    for (const client of this.connectedClients) {
      this.sendNotification(client, notification);
    }

    console.log(`[MCPNotifier] Broadcasted ${notification.type} to ${this.connectedClients.size} clients`);
  }

  private sendNotification(client: MCPConnection, notification: MCPNotification): void {
    try {
      client.send({
        jsonrpc: '2.0',
        method: 'notifications/resources/updated',
        params: {
          uri: 'git://file-watcher/notifications',
          notification
        }
      });
    } catch (error) {
      console.error(`[MCPNotifier] Failed to send notification to client:`, error);
      this.connectedClients.delete(client);
    }
  }

  private detectLanguage(filePath: string): string {
    const extension = path.extname(filePath).toLowerCase();
    const languageMap: Record<string, string> = {
      '.rs': 'rust',
      '.ts': 'typescript',
      '.js': 'javascript',
      '.py': 'python',
      '.md': 'markdown'
    };
    
    return languageMap[extension] || 'text';
  }

  getNotificationHistory(count: number = 50): MCPNotification[] {
    return this.notificationBuffer.slice(-count);
  }

  getSystemStatus(): SystemStatus {
    return { ...this.systemStatus };
  }
}
```

### 5. Performance Optimization Strategies

Intelligent resource management and adaptive optimization:

```typescript
interface OptimizationMetrics {
  processingTimes: number[];
  memoryUsage: number[];
  queueLengths: number[];
  errorRates: number[];
  cacheHitRates: number[];
}

class PerformanceOptimizer {
  private metrics: OptimizationMetrics = {
    processingTimes: [],
    memoryUsage: [],
    queueLengths: [],
    errorRates: [],
    cacheHitRates: []
  };

  private config = {
    maxBatchSize: 100,
    minBatchSize: 10,
    targetProcessingTime: 1000, // 1 second
    maxMemoryUsage: 512 * 1024 * 1024, // 512MB
    adaptationInterval: 30000 // 30 seconds
  };

  constructor(private reindexingEngine: IncrementalReindexingEngine) {
    // Start adaptive optimization
    setInterval(() => {
      this.adaptConfiguration();
    }, this.config.adaptationInterval);

    // Monitor system resources
    setInterval(() => {
      this.collectMetrics();
    }, 5000);
  }

  private collectMetrics(): void {
    const memUsage = process.memoryUsage();
    this.metrics.memoryUsage.push(memUsage.heapUsed);

    // Keep only recent metrics (last 100 measurements)
    Object.keys(this.metrics).forEach(key => {
      const metric = this.metrics[key as keyof OptimizationMetrics];
      if (metric.length > 100) {
        metric.splice(0, metric.length - 100);
      }
    });
  }

  private adaptConfiguration(): void {
    const avgProcessingTime = this.calculateAverage(this.metrics.processingTimes);
    const avgMemoryUsage = this.calculateAverage(this.metrics.memoryUsage);
    const avgQueueLength = this.calculateAverage(this.metrics.queueLengths);

    // Adapt batch size based on processing time
    if (avgProcessingTime > this.config.targetProcessingTime) {
      // Processing too slow, reduce batch size
      this.reindexingEngine.batchSize = Math.max(
        this.config.minBatchSize,
        Math.floor(this.reindexingEngine.batchSize * 0.8)
      );
    } else if (avgProcessingTime < this.config.targetProcessingTime * 0.5) {
      // Processing fast, increase batch size
      this.reindexingEngine.batchSize = Math.min(
        this.config.maxBatchSize,
        Math.floor(this.reindexingEngine.batchSize * 1.2)
      );
    }

    // Adapt concurrent jobs based on memory usage
    if (avgMemoryUsage > this.config.maxMemoryUsage) {
      this.reindexingEngine.maxConcurrentJobs = Math.max(1, 
        this.reindexingEngine.maxConcurrentJobs - 1
      );
    } else if (avgMemoryUsage < this.config.maxMemoryUsage * 0.7) {
      this.reindexingEngine.maxConcurrentJobs = Math.min(8,
        this.reindexingEngine.maxConcurrentJobs + 1
      );
    }

    console.log(`[Optimizer] Adapted: batchSize=${this.reindexingEngine.batchSize}, ` +
               `concurrentJobs=${this.reindexingEngine.maxConcurrentJobs}`);
  }

  private calculateAverage(numbers: number[]): number {
    if (numbers.length === 0) return 0;
    return numbers.reduce((sum, num) => sum + num, 0) / numbers.length;
  }

  recordProcessingTime(time: number): void {
    this.metrics.processingTimes.push(time);
  }

  recordQueueLength(length: number): void {
    this.metrics.queueLengths.push(length);
  }

  recordErrorRate(rate: number): void {
    this.metrics.errorRates.push(rate);
  }

  recordCacheHitRate(rate: number): void {
    this.metrics.cacheHitRates.push(rate);
  }

  getOptimizationReport(): any {
    return {
      currentConfig: {
        batchSize: this.reindexingEngine.batchSize,
        maxConcurrentJobs: this.reindexingEngine.maxConcurrentJobs
      },
      metrics: {
        avgProcessingTime: this.calculateAverage(this.metrics.processingTimes),
        avgMemoryUsage: this.calculateAverage(this.metrics.memoryUsage),
        avgQueueLength: this.calculateAverage(this.metrics.queueLengths),
        avgErrorRate: this.calculateAverage(this.metrics.errorRates),
        avgCacheHitRate: this.calculateAverage(this.metrics.cacheHitRates)
      },
      adaptations: this.getAdaptationCount()
    };
  }

  private getAdaptationCount(): number {
    // This would track how many times the system has adapted
    return 0; // Placeholder
  }
}
```

## Integration with Existing System

### Temporal Analysis Integration

```typescript
// Integration with the existing Rust temporal analysis system
interface TemporalAnalysisIntegration {
  async enrichFileWithTemporalContext(
    filePath: string, 
    commitHash: string
  ): Promise<TemporalContext> {
    // Call to Rust temporal analysis system
    const analysisResult = await this.callRustAnalysis({
      file_path: filePath,
      commit_hash: commitHash,
      analysis_type: 'comprehensive'
    });

    return {
      authorExpertise: analysisResult.author_expertise,
      changePatterns: analysisResult.change_patterns,
      bugOriginInfo: analysisResult.bug_origin,
      featureTimeline: analysisResult.feature_timeline
    };
  }

  private async callRustAnalysis(params: any): Promise<any> {
    // This would interface with the Rust neuromorphic system
    // through FFI, WebAssembly, or IPC
    return {};
  }
}
```

## Configuration and Setup

### Environment Configuration

```typescript
interface GitWatcherConfig {
  repositoryPath: string;
  watcherSettings: {
    debounceMs: number;
    ignorePatterns: string[];
    batchSize: number;
    maxConcurrentJobs: number;
  };
  gitIntegration: {
    enableHooks: boolean;
    hookTypes: ('post-commit' | 'post-checkout')[];
    diffAnalysisDepth: number;
  };
  performance: {
    enableAdaptiveOptimization: boolean;
    targetProcessingTime: number;
    maxMemoryUsage: number;
  };
  notifications: {
    enableRealTimeUpdates: boolean;
    bufferSize: number;
    statusUpdateInterval: number;
  };
}

const defaultConfig: GitWatcherConfig = {
  repositoryPath: process.cwd(),
  watcherSettings: {
    debounceMs: 300,
    ignorePatterns: [
      '**/node_modules/**',
      '**/target/**',
      '**/.git/**',
      '**/build/**',
      '**/*.tmp',
      '**/*.log'
    ],
    batchSize: 50,
    maxConcurrentJobs: 4
  },
  gitIntegration: {
    enableHooks: true,
    hookTypes: ['post-commit', 'post-checkout'],
    diffAnalysisDepth: 50
  },
  performance: {
    enableAdaptiveOptimization: true,
    targetProcessingTime: 1000,
    maxMemoryUsage: 512 * 1024 * 1024
  },
  notifications: {
    enableRealTimeUpdates: true,
    bufferSize: 1000,
    statusUpdateInterval: 30000
  }
};
```

## Implementation Timeline

### Phase 1: Core Infrastructure (Week 1)
- [ ] File system watcher with chokidar
- [ ] Basic debouncing and filtering
- [ ] Git integration manager
- [ ] Hook installation system

### Phase 2: Incremental Indexing (Week 2)
- [ ] Change detection algorithms
- [ ] Incremental reindexing engine
- [ ] Job queue and prioritization
- [ ] Error handling and recovery

### Phase 3: MCP Integration (Week 3)
- [ ] Notification service
- [ ] Real-time progress updates
- [ ] Client connection management
- [ ] Status broadcasting

### Phase 4: Optimization (Week 4)
- [ ] Performance monitoring
- [ ] Adaptive configuration
- [ ] Resource management
- [ ] Load testing and tuning

## Testing Strategy

### Unit Tests
```typescript
describe('GitAwareFileWatcher', () => {
  it('should debounce rapid file changes', async () => {
    const watcher = new GitAwareFileWatcher('/test/repo');
    const changeEvents: FileChangeEvent[] = [];
    
    watcher.on('fileChange', (event) => changeEvents.push(event));
    
    // Simulate rapid changes
    watcher.handleFileEvent('change', 'test.ts');
    watcher.handleFileEvent('change', 'test.ts');
    watcher.handleFileEvent('change', 'test.ts');
    
    await new Promise(resolve => setTimeout(resolve, 400));
    
    expect(changeEvents).toHaveLength(1);
  });
});
```

### Integration Tests
```typescript
describe('End-to-End Git Integration', () => {
  it('should handle commit-based reindexing', async () => {
    const system = new GitFileWatcherSystem(testConfig);
    await system.start();
    
    // Simulate git commit
    await exec('git commit -m "test commit"', { cwd: testRepo });
    
    // Wait for processing
    await system.waitForCompletion();
    
    expect(system.getProcessedFileCount()).toBeGreaterThan(0);
  });
});
```

## Monitoring and Observability

### Metrics Collection
- Processing latency (p50, p95, p99)
- Queue length and processing rate
- Memory usage and garbage collection
- Error rates and failure patterns
- Cache hit rates and effectiveness

### Health Checks
- File system accessibility
- Git repository health
- MCP connection status
- Background job processing
- Resource utilization

### Alerting
- Processing delays exceeding thresholds
- High error rates
- Memory usage approaching limits
- Git operation failures
- Client disconnections

## Security Considerations

### File System Security
- Validate all file paths
- Respect file system permissions
- Prevent directory traversal attacks
- Limit resource consumption

### Git Integration Security
- Validate Git repository integrity
- Secure hook installation
- Prevent injection attacks in Git commands
- Monitor for suspicious repository changes

### MCP Security
- Authenticate client connections
- Encrypt sensitive notifications
- Rate limit client requests
- Validate notification payloads

## Future Enhancements

### Advanced Features
1. **Semantic Change Detection**: Use LLM analysis to understand semantic significance of changes
2. **Predictive Reindexing**: Anticipate changes based on development patterns
3. **Cross-Repository Analysis**: Track changes across multiple related repositories
4. **AI-Powered Prioritization**: Use machine learning to optimize processing order
5. **Distributed Processing**: Scale across multiple nodes for large repositories

### Integration Possibilities
1. **IDE Extensions**: Real-time feedback in development environments
2. **CI/CD Integration**: Automated testing triggered by semantic changes
3. **Code Review Enhancement**: Context-aware review suggestions
4. **Documentation Generation**: Automatic updates based on code changes
5. **Knowledge Graph Evolution**: Dynamic relationship updates based on code evolution

---

*This Git file watcher design provides a robust, efficient, and intelligent system for maintaining synchronized indexes in the MCP server while leveraging the existing neuromorphic temporal analysis capabilities.*