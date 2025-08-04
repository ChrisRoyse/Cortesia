# Phase 8: MCP Wrapper - Thin Node.js/TypeScript Layer for LLM Integration

## Objective
Create a lightweight MCP (Model Context Protocol) server that exposes the completed Rust RAG system's capabilities to LLMs through standardized MCP tools. This is a **thin wrapper** that preserves all the multi-method search excellence of the Rust implementation while providing seamless LLM integration.

## Duration
3-5 days (24-40 hours)

## Critical Understanding
- **ALL Phases 0-7 remain EXACTLY as implemented** - pure Rust system with 95-97% accuracy
- **NO changes to the core Rust system** - zero modifications to existing codebase
- **Thin wrapper approach** - MCP server communicates with Rust via FFI/subprocess/API
- **Preserve performance** - minimal overhead between LLM requests and Rust execution
- **Simple installation** - `npm install -g ultimate-rag-mcp` after Rust system is built

## Prerequisites
- **Rust System Complete**: All Phases 0-7 implemented and validated
- **Rust Binary Built**: Compiled and tested Rust executable
- **Node.js 18+**: For MCP server implementation
- **TypeScript**: For type-safe MCP tool definitions

## Technical Approach

### 1. MCP Server Architecture (Thin Wrapper)
```typescript
// MCP Server - communicates with existing Rust system
export class UltimateRagMcpServer {
    private rustExecutablePath: string;
    private configPath: string;
    
    constructor(rustPath: string, configPath: string) {
        this.rustExecutablePath = rustPath;
        this.configPath = configPath;
    }
    
    // Delegate all search operations to Rust system
    async handleToolCall(name: string, params: any): Promise<any> {
        switch (name) {
            case 'index_codebase':
                return this.callRustIndexer(params);
            case 'search':
                return this.callRustSearch(params);
            case 'semantic_search':
                return this.callRustSemanticSearch(params);
            case 'track_repository':
                return this.enableRustGitWatcher(params);
            case 'get_context':
                return this.callRustFileRetrieval(params);
            case 'explain_matches':
                return this.callRustExplanation(params);
            default:
                throw new Error(`Unknown tool: ${name}`);
        }
    }
}
```

### 2. Communication Strategies with Rust System

#### Option A: Direct Subprocess Communication (Recommended)
```typescript
class RustSubprocessAdapter {
    private rustBinaryPath: string;
    
    async callRustSearch(query: string, options: SearchOptions): Promise<SearchResult[]> {
        const command = this.rustBinaryPath;
        const args = [
            'search',
            '--query', query,
            '--tier', options.tier.toString(),
            '--codebase', options.codebasePath,
            '--format', 'json'
        ];
        
        const result = await this.executeRustCommand(command, args);
        return JSON.parse(result.stdout);
    }
    
    private async executeRustCommand(command: string, args: string[]): Promise<{stdout: string, stderr: string}> {
        return new Promise((resolve, reject) => {
            const child = spawn(command, args, { stdio: 'pipe' });
            
            let stdout = '';
            let stderr = '';
            
            child.stdout.on('data', (data) => stdout += data);
            child.stderr.on('data', (data) => stderr += data);
            
            child.on('close', (code) => {
                if (code === 0) {
                    resolve({ stdout, stderr });
                } else {
                    reject(new Error(`Rust process exited with code ${code}: ${stderr}`));
                }
            });
        });
    }
}
```

#### Option B: FFI via N-API (Advanced)
```typescript
// If performance is critical, use FFI bindings
import { createRequire } from 'module';
const require = createRequire(import.meta.url);

// Requires building Rust library with C FFI exports
const rustLibrary = require('./native/ultimate_rag.node');

class RustFFIAdapter {
    search(query: string, options: SearchOptions): SearchResult[] {
        // Direct FFI call to Rust library
        const jsonResult = rustLibrary.search(
            query,
            options.tier,
            options.codebasePath,
            JSON.stringify(options)
        );
        return JSON.parse(jsonResult);
    }
}
```

#### Option C: Local REST API (Fallback)
```typescript
// If Rust system exposes HTTP API
class RustHTTPAdapter {
    private baseUrl: string;
    
    constructor(baseUrl: string = 'http://localhost:8080') {
        this.baseUrl = baseUrl;
    }
    
    async search(query: string, options: SearchOptions): Promise<SearchResult[]> {
        const response = await fetch(`${this.baseUrl}/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, ...options })
        });
        
        if (!response.ok) {
            throw new Error(`Rust API error: ${response.statusText}`);
        }
        
        return response.json();
    }
}
```

### 3. MCP Tool Definitions

#### Tool 1: index_codebase
```typescript
const indexCodebaseTool: Tool = {
    name: 'index_codebase',
    description: 'Index a codebase using the Rust RAG system for search capabilities',
    inputSchema: {
        type: 'object',
        properties: {
            path: {
                type: 'string',
                description: 'Absolute path to codebase root directory'
            },
            include_patterns: {
                type: 'array',
                items: { type: 'string' },
                description: 'Glob patterns for files to include (optional)',
                default: ['**/*.rs', '**/*.py', '**/*.js', '**/*.ts', '**/*.go', '**/*.java']
            },
            exclude_patterns: {
                type: 'array',
                items: { type: 'string' },
                description: 'Glob patterns for files to exclude (optional)',
                default: ['node_modules/**', '.git/**', 'target/**', 'dist/**']
            },
            force_reindex: {
                type: 'boolean',
                description: 'Force complete reindexing (optional)',
                default: false
            },
            enable_git_hooks: {
                type: 'boolean',
                description: 'Enable automatic reindexing on git changes (optional)',
                default: true
            }
        },
        required: ['path']
    }
};

async function handleIndexCodebase(params: IndexCodebaseParams): Promise<IndexResult> {
    // Validate codebase path exists
    if (!fs.existsSync(params.path)) {
        throw new Error(`Codebase path does not exist: ${params.path}`);
    }
    
    // Call Rust indexer
    const rustResult = await rustAdapter.indexCodebase(params);
    
    return {
        success: true,
        files_indexed: rustResult.filesIndexed,
        embeddings_created: rustResult.embeddingsCreated,
        index_size_mb: rustResult.indexSizeMb,
        duration_ms: rustResult.durationMs,
        git_hooks_enabled: params.enable_git_hooks
    };
}
```

#### Tool 2: search (Multi-Method Tiered Search)
```typescript
const searchTool: Tool = {
    name: 'search',
    description: 'Search codebase using multi-method approach with tiered execution strategy',
    inputSchema: {
        type: 'object',
        properties: {
            query: {
                type: 'string',
                description: 'Search query (supports special characters, boolean logic, proximity)'
            },
            codebase_path: {
                type: 'string',
                description: 'Path to indexed codebase'
            },
            search_tier: {
                type: 'number',
                enum: [1, 2, 3],
                description: 'Search tier: 1=Fast(85-90%), 2=Balanced(92-95%), 3=Deep(95-97%)',
                default: 2
            },
            max_results: {
                type: 'number',
                description: 'Maximum number of results to return',
                default: 20,
                minimum: 1,
                maximum: 100
            },
            file_types: {
                type: 'array',
                items: { type: 'string' },
                description: 'Filter by file extensions (optional)'
            },
            include_content: {
                type: 'boolean',
                description: 'Include file content in results',
                default: true
            }
        },
        required: ['query', 'codebase_path']
    }
};

async function handleSearch(params: SearchParams): Promise<SearchResult> {
    // Validate inputs
    if (!params.query.trim()) {
        throw new Error('Query cannot be empty');
    }
    
    if (!fs.existsSync(params.codebase_path)) {
        throw new Error(`Codebase path does not exist: ${params.codebase_path}`);
    }
    
    // Call Rust search system with all its multi-method capabilities
    const rustResult = await rustAdapter.search(params.query, {
        codebasePath: params.codebase_path,
        tier: params.search_tier || 2,
        maxResults: params.max_results || 20,
        fileTypes: params.file_types,
        includeContent: params.include_content
    });
    
    return {
        results: rustResult.matches.map(match => ({
            file_path: match.filePath,
            content: params.include_content ? match.content : undefined,
            line_number: match.lineNumber,
            score: match.score,
            match_type: match.matchType, // 'exact' | 'semantic' | 'fuzzy' | 'ast' | 'hybrid'
            context_lines: match.contextLines,
            git_info: match.gitInfo ? {
                last_modified: match.gitInfo.lastModified,
                author: match.gitInfo.author,
                commit_hash: match.gitInfo.commitHash
            } : undefined
        })),
        total_found: rustResult.totalFound,
        search_duration_ms: rustResult.durationMs,
        tier_used: params.search_tier || 2,
        accuracy_estimate: rustResult.accuracyEstimate
    };
}
```

#### Tool 3: semantic_search (Vector-Only Search)
```typescript
const semanticSearchTool: Tool = {
    name: 'semantic_search',
    description: 'Semantic search using embeddings for conceptual matches',
    inputSchema: {
        type: 'object',
        properties: {
            query: {
                type: 'string',
                description: 'Natural language query for semantic matching'
            },
            codebase_path: {
                type: 'string',
                description: 'Path to indexed codebase'
            },
            max_results: {
                type: 'number',
                description: 'Maximum number of results',
                default: 10,
                minimum: 1,
                maximum: 50
            },
            similarity_threshold: {
                type: 'number',
                description: 'Minimum similarity score (0.0-1.0)',
                default: 0.7,
                minimum: 0.0,
                maximum: 1.0
            }
        },
        required: ['query', 'codebase_path']
    }
};

async function handleSemanticSearch(params: SemanticSearchParams): Promise<SemanticResult> {
    // Call Rust vector search system
    const rustResult = await rustAdapter.semanticSearch(params.query, {
        codebasePath: params.codebase_path,
        maxResults: params.max_results || 10,
        similarityThreshold: params.similarity_threshold || 0.7
    });
    
    return {
        results: rustResult.matches.map(match => ({
            file_path: match.filePath,
            content: match.content,
            similarity_score: match.similarityScore,
            chunk_index: match.chunkIndex,
            embedding_model: 'text-embedding-3-large',
            semantic_summary: match.semanticSummary
        })),
        total_found: rustResult.totalFound,
        search_duration_ms: rustResult.durationMs,
        embedding_model_used: 'text-embedding-3-large'
    };
}
```

#### Tool 4: track_repository (Git Integration)
```typescript
const trackRepositoryTool: Tool = {
    name: 'track_repository',
    description: 'Enable automatic reindexing when repository changes',
    inputSchema: {
        type: 'object',
        properties: {
            codebase_path: {
                type: 'string',
                description: 'Path to git repository root'
            },
            watch_patterns: {
                type: 'array',
                items: { type: 'string' },
                description: 'File patterns to watch for changes',
                default: ['**/*.rs', '**/*.py', '**/*.js', '**/*.ts']
            },
            debounce_ms: {
                type: 'number',
                description: 'Debounce delay for batch processing changes',
                default: 5000,
                minimum: 1000
            }
        },
        required: ['codebase_path']
    }
};

async function handleTrackRepository(params: TrackRepositoryParams): Promise<TrackingResult> {
    // Enable Rust git watcher
    const rustResult = await rustAdapter.enableGitWatcher({
        repositoryPath: params.codebase_path,
        watchPatterns: params.watch_patterns || ['**/*.rs', '**/*.py', '**/*.js', '**/*.ts'],
        debounceMs: params.debounce_ms || 5000
    });
    
    return {
        success: true,
        repository_path: params.codebase_path,
        watching_patterns: params.watch_patterns,
        git_hooks_installed: rustResult.gitHooksInstalled,
        watch_process_id: rustResult.watchProcessId
    };
}
```

#### Tool 5: get_context (File Retrieval)
```typescript
const getContextTool: Tool = {
    name: 'get_context',
    description: 'Retrieve specific file content with optional git history',
    inputSchema: {
        type: 'object',
        properties: {
            file_path: {
                type: 'string',
                description: 'Absolute or relative path to file'
            },
            codebase_path: {
                type: 'string',
                description: 'Root path of indexed codebase'
            },
            include_git_history: {
                type: 'boolean',
                description: 'Include recent git history for the file',
                default: false
            },
            include_related_files: {
                type: 'boolean',
                description: 'Include files with similar content/imports',
                default: false
            },
            line_range: {
                type: 'object',
                properties: {
                    start: { type: 'number', minimum: 1 },
                    end: { type: 'number', minimum: 1 }
                },
                description: 'Specific line range to retrieve (optional)'
            }
        },
        required: ['file_path', 'codebase_path']
    }
};

async function handleGetContext(params: GetContextParams): Promise<ContextResult> {
    // Call Rust file retrieval system
    const rustResult = await rustAdapter.getFileContext({
        filePath: params.file_path,
        codebasePath: params.codebase_path,
        includeGitHistory: params.include_git_history,
        includeRelatedFiles: params.include_related_files,
        lineRange: params.line_range
    });
    
    return {
        file_path: rustResult.filePath,
        content: rustResult.content,
        line_count: rustResult.lineCount,
        file_size_bytes: rustResult.fileSizeBytes,
        last_modified: rustResult.lastModified,
        git_history: params.include_git_history ? rustResult.gitHistory : undefined,
        related_files: params.include_related_files ? rustResult.relatedFiles : undefined,
        language: rustResult.detectedLanguage
    };
}
```

#### Tool 6: explain_matches (Search Explanation)
```typescript
const explainMatchesTool: Tool = {
    name: 'explain_matches',
    description: 'Get detailed explanation of why search results were returned',
    inputSchema: {
        type: 'object',
        properties: {
            query: {
                type: 'string',
                description: 'Original search query'
            },
            codebase_path: {
                type: 'string',
                description: 'Path to indexed codebase'
            },
            result_file_path: {
                type: 'string',
                description: 'Specific result file to explain'
            },
            explain_ranking: {
                type: 'boolean',
                description: 'Include ranking/scoring explanation',
                default: true
            }
        },
        required: ['query', 'codebase_path', 'result_file_path']
    }
};

async function handleExplainMatches(params: ExplainMatchesParams): Promise<ExplanationResult> {
    // Call Rust explanation system
    const rustResult = await rustAdapter.explainMatch({
        query: params.query,
        codebasePath: params.codebase_path,
        resultFilePath: params.result_file_path,
        explainRanking: params.explain_ranking
    });
    
    return {
        file_path: params.result_file_path,
        query: params.query,
        match_explanations: rustResult.explanations.map(exp => ({
            method: exp.method, // 'exact' | 'semantic' | 'fuzzy' | 'ast'
            score: exp.score,
            reason: exp.reason,
            matched_terms: exp.matchedTerms,
            context: exp.context
        })),
        ranking_explanation: params.explain_ranking ? {
            final_score: rustResult.ranking.finalScore,
            score_components: rustResult.ranking.scoreComponents,
            tier_used: rustResult.ranking.tierUsed,
            fusion_method: rustResult.ranking.fusionMethod
        } : undefined,
        alternatives_considered: rustResult.alternativesConsidered
    };
}
```

## Implementation Tasks

### Task 1: MCP Server Framework Setup (8 hours)
```typescript
// Package structure
ultimate-rag-mcp/
├── package.json
├── tsconfig.json
├── src/
│   ├── index.ts              // MCP server entry point
│   ├── tools/                // MCP tool definitions
│   ├── adapters/             // Rust communication adapters
│   ├── types/                // TypeScript interfaces
│   └── utils/                // Helper functions
├── bin/
│   └── ultimate-rag-mcp     // CLI executable
└── README.md

// package.json
{
  "name": "ultimate-rag-mcp",
  "version": "1.0.0",
  "description": "MCP server for Ultimate RAG Rust system",
  "main": "dist/index.js",
  "bin": {
    "ultimate-rag-mcp": "./bin/ultimate-rag-mcp"
  },
  "scripts": {
    "build": "tsc",
    "start": "node dist/index.js",
    "dev": "ts-node src/index.ts"
  },
  "dependencies": {
    "@modelcontextprotocol/sdk": "^0.5.0",
    "zod": "^3.22.0"
  },
  "devDependencies": {
    "typescript": "^5.0.0",
    "@types/node": "^20.0.0",
    "ts-node": "^10.9.0"
  }
}
```

### Task 2: Rust Communication Layer (8 hours)
```typescript
// src/adapters/rust-adapter.ts
export class RustAdapter {
    constructor(
        private rustBinaryPath: string,
        private configPath?: string
    ) {
        this.validateRustBinary();
    }
    
    private validateRustBinary(): void {
        if (!fs.existsSync(this.rustBinaryPath)) {
            throw new Error(`Rust binary not found: ${this.rustBinaryPath}`);
        }
        
        // Test that binary works
        try {
            execSync(`${this.rustBinaryPath} --version`, { timeout: 5000 });
        } catch (error) {
            throw new Error(`Rust binary validation failed: ${error.message}`);
        }
    }
    
    async indexCodebase(params: IndexCodebaseParams): Promise<IndexResult> {
        const args = [
            'index',
            '--path', params.path,
            '--output-format', 'json'
        ];
        
        if (params.include_patterns) {
            args.push('--include', params.include_patterns.join(','));
        }
        
        if (params.exclude_patterns) {
            args.push('--exclude', params.exclude_patterns.join(','));
        }
        
        if (params.force_reindex) {
            args.push('--force');
        }
        
        const result = await this.executeCommand(args);
        return JSON.parse(result.stdout);
    }
    
    async search(query: string, options: SearchOptions): Promise<RustSearchResult> {
        const args = [
            'search',
            '--query', query,
            '--codebase', options.codebasePath,
            '--tier', options.tier.toString(),
            '--max-results', options.maxResults.toString(),
            '--output-format', 'json'
        ];
        
        if (options.fileTypes) {
            args.push('--file-types', options.fileTypes.join(','));
        }
        
        if (!options.includeContent) {
            args.push('--no-content');
        }
        
        const result = await this.executeCommand(args);
        return JSON.parse(result.stdout);
    }
    
    private async executeCommand(args: string[]): Promise<{stdout: string, stderr: string}> {
        return new Promise((resolve, reject) => {
            const child = spawn(this.rustBinaryPath, args, {
                stdio: 'pipe',
                timeout: 30000 // 30 second timeout
            });
            
            let stdout = '';
            let stderr = '';
            
            child.stdout.on('data', (data) => stdout += data.toString());
            child.stderr.on('data', (data) => stderr += data.toString());
            
            child.on('close', (code, signal) => {
                if (signal) {
                    reject(new Error(`Process killed by signal: ${signal}`));
                } else if (code === 0) {
                    resolve({ stdout, stderr });
                } else {
                    reject(new Error(`Process exited with code ${code}: ${stderr}`));
                }
            });
            
            child.on('error', (error) => {
                reject(new Error(`Failed to start process: ${error.message}`));
            });
        });
    }
}
```

### Task 3: CLI Configuration & Installation (4 hours)
```bash
#!/usr/bin/env node
// bin/ultimate-rag-mcp

import { UltimateRagMcpServer } from '../dist/index.js';
import { program } from 'commander';
import path from 'path';
import fs from 'fs';

program
    .name('ultimate-rag-mcp')
    .description('MCP server for Ultimate RAG Rust system')
    .version('1.0.0')
    .option('-r, --rust-binary <path>', 'Path to Rust binary')
    .option('-c, --config <path>', 'Path to configuration file')
    .option('-p, --port <number>', 'Port for MCP server', '3000')
    .option('--stdio', 'Use stdio transport (for MCP client integration)')
    .parse();

const options = program.opts();

// Auto-detect Rust binary if not specified
const rustBinaryPath = options.rustBinary || autoDetectRustBinary();

function autoDetectRustBinary(): string {
    const possiblePaths = [
        path.join(process.cwd(), 'target', 'release', 'ultimate-rag'),
        path.join(process.cwd(), 'target', 'release', 'ultimate-rag.exe'),
        path.join(process.env.HOME || '', '.cargo', 'bin', 'ultimate-rag'),
        'ultimate-rag' // Assume it's in PATH
    ];
    
    for (const rustPath of possiblePaths) {
        if (fs.existsSync(rustPath)) {
            return rustPath;
        }
    }
    
    throw new Error(
        'Rust binary not found. Please specify with --rust-binary or ensure ' +
        'the Rust system is built and available in PATH'
    );
}

// Start MCP server
const server = new UltimateRagMcpServer(rustBinaryPath, options.config);

if (options.stdio) {
    server.connectStdio();
} else {
    server.listen(parseInt(options.port));
}

console.log(`Ultimate RAG MCP Server started with Rust binary: ${rustBinaryPath}`);
```

### Task 4: Error Handling & Validation (4 hours)
```typescript
// src/utils/validation.ts
export class ValidationError extends Error {
    constructor(message: string, public field?: string) {
        super(message);
        this.name = 'ValidationError';
    }
}

export class RustSystemError extends Error {
    constructor(message: string, public rustError?: string, public exitCode?: number) {
        super(message);
        this.name = 'RustSystemError';
    }
}

export function validateIndexParams(params: any): IndexCodebaseParams {
    if (!params.path || typeof params.path !== 'string') {
        throw new ValidationError('path is required and must be a string');
    }
    
    if (!path.isAbsolute(params.path)) {
        throw new ValidationError('path must be an absolute path');
    }
    
    if (!fs.existsSync(params.path)) {
        throw new ValidationError(`path does not exist: ${params.path}`);
    }
    
    if (!fs.lstatSync(params.path).isDirectory()) {
        throw new ValidationError('path must be a directory');
    }
    
    return {
        path: params.path,
        include_patterns: params.include_patterns || undefined,
        exclude_patterns: params.exclude_patterns || undefined,
        force_reindex: Boolean(params.force_reindex),
        enable_git_hooks: Boolean(params.enable_git_hooks)
    };
}

export function validateSearchParams(params: any): SearchParams {
    if (!params.query || typeof params.query !== 'string') {
        throw new ValidationError('query is required and must be a string');
    }
    
    if (!params.query.trim()) {
        throw new ValidationError('query cannot be empty');
    }
    
    if (!params.codebase_path || typeof params.codebase_path !== 'string') {
        throw new ValidationError('codebase_path is required and must be a string');
    }
    
    if (!fs.existsSync(params.codebase_path)) {
        throw new ValidationError(`codebase_path does not exist: ${params.codebase_path}`);
    }
    
    const tier = params.search_tier || 2;
    if (![1, 2, 3].includes(tier)) {
        throw new ValidationError('search_tier must be 1, 2, or 3');
    }
    
    const maxResults = params.max_results || 20;
    if (maxResults < 1 || maxResults > 100) {
        throw new ValidationError('max_results must be between 1 and 100');
    }
    
    return {
        query: params.query.trim(),
        codebase_path: params.codebase_path,
        search_tier: tier,
        max_results: maxResults,
        file_types: params.file_types || undefined,
        include_content: params.include_content !== false
    };
}
```

## Installation & Deployment

### Installation Process
```bash
# 1. Ensure Rust system is built and working
cd /path/to/rust/project
cargo build --release
./target/release/ultimate-rag --version

# 2. Install MCP server globally
npm install -g ultimate-rag-mcp

# 3. Verify installation
ultimate-rag-mcp --version

# 4. Test with your Rust binary
ultimate-rag-mcp --rust-binary ./target/release/ultimate-rag --help
```

### MCP Client Integration

#### Claude Desktop Configuration
```json
// ~/.config/claude/claude_desktop_config.json
{
  "mcpServers": {
    "ultimate-rag": {
      "command": "ultimate-rag-mcp",
      "args": [
        "--rust-binary", "/path/to/rust/ultimate-rag",
        "--stdio"
      ]
    }
  }
}
```

#### Generic MCP Client
```typescript
import { Client } from '@modelcontextprotocol/sdk/client/index.js';

const client = new Client({
  name: "ultimate-rag-client",
  version: "1.0.0"
});

await client.connect({
  command: "ultimate-rag-mcp",
  args: ["--rust-binary", "/path/to/rust/ultimate-rag", "--stdio"]
});

// Index a codebase
const indexResult = await client.request({
  method: "tools/call",
  params: {
    name: "index_codebase",
    arguments: {
      path: "/path/to/my/project",
      enable_git_hooks: true
    }
  }
});

// Search the codebase
const searchResult = await client.request({
  method: "tools/call",
  params: {
    name: "search",
    arguments: {
      query: "Result<T, E> AND HTTP",
      codebase_path: "/path/to/my/project",
      search_tier: 2
    }
  }
});
```

## Configuration Management

### Default Configuration
```json
// ~/.ultimate-rag-mcp/config.json
{
  "rust_binary_path": "auto-detect",
  "default_search_tier": 2,
  "max_concurrent_requests": 5,
  "request_timeout_ms": 30000,
  "auto_detect_codebases": true,
  "enable_git_integration": true,
  "log_level": "info",
  "cache_enabled": true,
  "cache_ttl_minutes": 10
}
```

### Runtime Configuration
```typescript
// The MCP server reads configuration and passes it to Rust
export interface MCPServerConfig {
    rustBinaryPath: string;
    defaultSearchTier: 1 | 2 | 3;
    maxConcurrentRequests: number;
    requestTimeoutMs: number;
    logLevel: 'debug' | 'info' | 'warn' | 'error';
    cacheEnabled: boolean;
    cacheTtlMinutes: number;
}

class ConfigManager {
    static load(configPath?: string): MCPServerConfig {
        const defaultConfig: MCPServerConfig = {
            rustBinaryPath: 'auto-detect',
            defaultSearchTier: 2,
            maxConcurrentRequests: 5,
            requestTimeoutMs: 30000,
            logLevel: 'info',
            cacheEnabled: true,
            cacheTtlMinutes: 10
        };
        
        if (configPath && fs.existsSync(configPath)) {
            const userConfig = JSON.parse(fs.readFileSync(configPath, 'utf8'));
            return { ...defaultConfig, ...userConfig };
        }
        
        return defaultConfig;
    }
}
```

## Testing Strategy

### Unit Tests
```typescript
// tests/rust-adapter.test.ts
describe('RustAdapter', () => {
    let adapter: RustAdapter;
    
    beforeEach(() => {
        // Mock Rust binary for testing
        adapter = new RustAdapter('./mocks/mock-ultimate-rag');
    });
    
    test('should validate rust binary exists', () => {
        expect(() => new RustAdapter('./nonexistent')).toThrow();
    });
    
    test('should execute search command correctly', async () => {
        const result = await adapter.search('test query', {
            codebasePath: '/test/path',
            tier: 2,
            maxResults: 10,
            includeContent: true
        });
        
        expect(result).toHaveProperty('matches');
        expect(result.matches).toBeInstanceOf(Array);
    });
});
```

### Integration Tests
```typescript
// tests/integration.test.ts
describe('MCP Integration', () => {
    let mcpServer: UltimateRagMcpServer;
    
    beforeAll(async () => {
        // Start server with test Rust binary
        mcpServer = new UltimateRagMcpServer('./target/release/ultimate-rag');
        await mcpServer.start();
    });
    
    test('should handle index_codebase tool call', async () => {
        const result = await mcpServer.handleToolCall('index_codebase', {
            path: '/test/codebase',
            force_reindex: true
        });
        
        expect(result.success).toBe(true);
        expect(result.files_indexed).toBeGreaterThan(0);
    });
    
    test('should handle search tool call', async () => {
        const result = await mcpServer.handleToolCall('search', {
            query: 'pub fn',
            codebase_path: '/test/codebase',
            search_tier: 2
        });
        
        expect(result.results).toBeInstanceOf(Array);
        expect(result.tier_used).toBe(2);
    });
});
```

## Dependencies and Timeline

### Prerequisites
- **Phase 7 Complete**: Rust system fully validated and working
- **Rust Binary Built**: Compiled executable with CLI interface
- **OpenAI API Key**: For embedding functionality (if used)
- **Node.js 18+**: Runtime environment for MCP server

### Timeline Breakdown
- **Day 1**: MCP server framework and tool definitions (8 hours)
- **Day 2**: Rust communication adapter implementation (8 hours)
- **Day 3**: Error handling, validation, and CLI setup (8 hours)
- **Day 4**: Testing, documentation, and packaging (8 hours)
- **Day 5**: Integration testing and deployment verification (8 hours)

### External Dependencies
```json
{
  "dependencies": {
    "@modelcontextprotocol/sdk": "^0.5.0",    // MCP protocol implementation
    "commander": "^11.0.0",                   // CLI argument parsing
    "zod": "^3.22.0"                         // Runtime type validation
  },
  "devDependencies": {
    "typescript": "^5.0.0",
    "@types/node": "^20.0.0",
    "jest": "^29.0.0",
    "@types/jest": "^29.0.0",
    "ts-node": "^10.9.0"
  }
}
```

## Success Criteria

### Functional Requirements
- **Tool Integration**: All 6 MCP tools correctly delegate to Rust system
- **Performance Preservation**: <10ms overhead between MCP and Rust calls
- **Error Handling**: Graceful error propagation from Rust to MCP client
- **Configuration**: Flexible configuration for different deployment scenarios
- **CLI Usability**: Simple installation and configuration process

### Quality Gates
- **Zero Data Loss**: No information lost in translation between MCP and Rust
- **Type Safety**: Full TypeScript type coverage for all interfaces
- **Error Resilience**: Handles Rust system failures gracefully
- **Performance**: Maintains Rust system's accuracy and speed guarantees
- **Documentation**: Clear setup and usage instructions

### Deployment Validation
- **npm Installation**: Global installation works on Windows, macOS, Linux
- **Rust Detection**: Auto-detects Rust binary in common locations
- **MCP Client Integration**: Works with Claude Desktop and other MCP clients
- **Multiple Codebases**: Can handle multiple indexed codebases simultaneously
- **Concurrent Requests**: Handles multiple LLM requests without conflicts

## Risk Mitigation

### Technical Risks
- **Rust Binary Dependency**: Clear error messages if Rust system not available
- **Process Communication**: Robust subprocess management with timeouts
- **Performance Overhead**: Minimal marshaling between TypeScript and Rust
- **Error Propagation**: Preserve Rust error context in MCP responses

### Operational Risks
- **Installation Complexity**: Auto-detection and clear error messages
- **Version Compatibility**: Pin to specific Rust system version
- **Configuration Drift**: Sensible defaults with override capabilities
- **Client Integration**: Test with multiple MCP clients

## Final Deliverable

A production-ready MCP server that:

### ✅ **Preserves All Rust Capabilities**
- **95-97% accuracy** maintained through direct delegation
- **Multi-method search** (exact, semantic, fuzzy, AST, hybrid)
- **Tiered execution** strategy (Tier 1: 85-90%, Tier 2: 92-95%, Tier 3: 95-97%)
- **Windows optimization** and all special character handling
- **Git integration** and automatic reindexing

### ✅ **Provides LLM Integration**
- **6 MCP tools** covering all major Rust system capabilities
- **Type-safe interfaces** with comprehensive validation
- **Error handling** that preserves Rust system error context
- **Performance preservation** with minimal overhead

### ✅ **Simple Installation & Usage**
- **Global npm package**: `npm install -g ultimate-rag-mcp`
- **Auto-detection** of Rust binary in common locations
- **CLI configuration** with sensible defaults
- **MCP client integration** tested with Claude Desktop

### ✅ **Production Quality**
- **Comprehensive testing** including integration tests
- **Error resilience** with graceful degradation
- **Performance monitoring** and timeout handling
- **Documentation** for setup, configuration, and troubleshooting

---

*Phase 8 provides a thin, efficient wrapper that exposes the full power of the Rust RAG system to LLMs through the standardized MCP protocol, preserving all accuracy and performance characteristics while enabling seamless integration with AI development workflows.*