# list_branches - Database Branch Discovery Tool

## Overview

The `list_branches` tool provides comprehensive visibility into all available knowledge graph database branches within the LLMKG system. It displays branch metadata including names, database IDs, creation timestamps, source relationships, descriptions, and activity status. This tool is essential for branch management, enabling users to understand the current branching structure and make informed decisions about branch operations.

## Implementation Details

### Handler Location
- **File**: `src/mcp/llm_friendly_server/handlers/temporal.rs`
- **Function**: `handle_list_branches`
- **Lines**: 221-271

### Core Functionality

The tool implements comprehensive branch discovery and reporting:

1. **Branch Enumeration**: List all available branches in the system
2. **Metadata Retrieval**: Fetch detailed information for each branch
3. **Status Reporting**: Show active/inactive status for branches
4. **Relationship Mapping**: Display parent-child branch relationships
5. **Creation Tracking**: Provide timestamps and source information
6. **Activity Monitoring**: Report branch usage and activity status

### Input Schema

```json
{
  "type": "object",
  "properties": {}
}
```

**Note**: This tool requires no input parameters as it lists all available branches in the system.

### Key Variables and Functions

#### Primary Handler Function
```rust
pub async fn handle_list_branches(
    _knowledge_engine: &Arc<RwLock<KnowledgeEngine>>,
    usage_stats: &Arc<RwLock<UsageStats>>,
    _params: Value,
) -> std::result::Result<(Value, String, Vec<String>), String>
```

#### Branch Manager Access
```rust
let branch_manager_arc = get_branch_manager().await
    .map_err(|e| format!("Failed to get branch manager: {}", e))?;

let branch_manager_guard = branch_manager_arc.read().await;
let branch_manager = branch_manager_guard.as_ref()
    .ok_or("Branch manager not initialized")?;
```

#### Branch Retrieval
```rust
let branches = branch_manager.list_branches().await
    .map_err(|e| format!("Failed to list branches: {}", e))?;
```

### Branch Information Structure

#### Branch Data Processing
```rust
let branch_data: Vec<Value> = branches.iter().map(|branch| {
    json!({
        "branch_name": branch.branch_name,
        "database_id": branch.database_id.as_str(),
        "created_at": branch.created_at.to_rfc3339(),
        "created_from": branch.created_from.as_str(),
        "description": branch.description,
        "is_active": branch.is_active
    })
}).collect();
```

**Branch Properties:**
- **branch_name**: Human-readable identifier for the branch
- **database_id**: Unique database identifier used for operations
- **created_at**: ISO 8601 timestamp of branch creation
- **created_from**: Source database ID from which this branch was created
- **description**: Optional user-provided description of branch purpose
- **is_active**: Boolean indicating if branch is currently accessible

#### Branch Metadata Schema
```rust
// Inferred from usage patterns
struct BranchInfo {
    branch_name: String,
    database_id: DatabaseId,
    created_at: DateTime<Utc>,
    created_from: DatabaseId,
    description: Option<String>,
    is_active: bool,
}
```

### Branch Manager Integration

#### Distributed Branch Management
```rust
use crate::mcp::llm_friendly_server::database_branching::{
    get_branch_manager, MergeStrategy
};
```

The branch manager provides:
- **Central Registry**: Maintains authoritative list of all branches
- **Metadata Storage**: Stores comprehensive branch information
- **State Tracking**: Monitors branch activity and status
- **Relationship Mapping**: Tracks parent-child branch relationships
- **Distributed Access**: Coordinates across multiple system nodes

#### Branch Discovery Process
```rust
let branches = branch_manager.list_branches().await
    .map_err(|e| format!("Failed to list branches: {}", e))?;
```

**Discovery Features:**
- **Complete Enumeration**: Lists all branches regardless of status
- **Metadata Inclusion**: Provides full branch information in single call
- **Real-time Status**: Reflects current branch states and activity
- **Relationship Context**: Shows how branches relate to each other

### Output Format

#### Comprehensive Branch Listing Response
```json
{
  "branches": [
    {
      "branch_name": "main",
      "database_id": "main",
      "created_at": "2024-01-01T00:00:00Z",
      "created_from": "initial",
      "description": "Main production database",
      "is_active": true
    },
    {
      "branch_name": "quantum-physics-exploration",
      "database_id": "main_quantum-physics-exploration",
      "created_at": "2024-01-15T10:30:00Z",
      "created_from": "main",
      "description": "Testing new quantum physics relationships",
      "is_active": true
    },
    {
      "branch_name": "historical-data-import",
      "database_id": "main_historical-data-import",
      "created_at": "2024-01-14T14:20:00Z",
      "created_from": "main",
      "description": "Importing historical scientific data",
      "is_active": false
    }
  ],
  "total_branches": 3
}
```

#### Human-Readable Message Format
```rust
let message = format!(
    "Found {} branches:\n\n{}",
    branches.len(),
    branches.iter()
        .take(5)
        .map(|b| format!("🌿 {} ({})", b.branch_name, b.database_id.as_str()))
        .collect::<Vec<_>>()
        .join("\n")
);
```

**Example Human-Readable Output:**
```
Found 3 branches:

🌿 main (main)
🌿 quantum-physics-exploration (main_quantum-physics-exploration)
🌿 historical-data-import (main_historical-data-import)
```

**Display Features:**
- **Tree Symbol**: Uses 🌿 emoji to visually represent branches
- **Compact Format**: Shows name and database ID for quick reference
- **Truncation**: Limits display to first 5 branches for readability
- **Complete Data**: Full details available in JSON response

### Branch Status Analysis

#### Active vs Inactive Branches
```rust
"is_active": branch.is_active
```

**Branch States:**
- **Active (true)**: Branch is available for read/write operations
- **Inactive (false)**: Branch exists but is not currently accessible

**State Implications:**
- **Active branches** can be used for all operations (read, write, compare, merge)
- **Inactive branches** may be temporarily unavailable or archived
- **State changes** are managed by the branch manager and system administrators

#### Creation Timestamp Analysis
```rust
"created_at": branch.created_at.to_rfc3339(),
```

**Timestamp Features:**
- **ISO 8601 Format**: Standard format for cross-system compatibility
- **UTC Timezone**: Normalized to UTC for consistency
- **Precision**: Includes seconds for precise creation tracking
- **Sorting**: Enables chronological analysis of branch creation

### Branch Relationship Mapping

#### Parent-Child Relationships
```rust
"created_from": branch.created_from.as_str(),
```

**Relationship Types:**
- **Root Branches**: Created from "initial" or system default
- **Feature Branches**: Created from "main" or other production branches
- **Sub-branches**: Created from existing feature branches
- **Merge Branches**: Temporary branches created during merge operations

**Relationship Benefits:**
- **Lineage Tracking**: Understand the evolution of branches
- **Merge Planning**: Identify appropriate merge targets
- **Dependency Analysis**: Understand branch interdependencies
- **Rollback Strategy**: Plan rollback paths through branch hierarchy

### Error Handling

#### Branch Manager Access Errors
```rust
let branch_manager_arc = get_branch_manager().await
    .map_err(|e| format!("Failed to get branch manager: {}", e))?;

let branch_manager_guard = branch_manager_arc.read().await;
let branch_manager = branch_manager_guard.as_ref()
    .ok_or("Branch manager not initialized")?;
```

#### Branch Discovery Errors
```rust
let branches = branch_manager.list_branches().await
    .map_err(|e| format!("Failed to list branches: {}", e))?;
```

**Error Scenarios:**
- **Manager Unavailable**: Branch manager service is not running
- **Network Issues**: Communication problems with distributed components
- **Permission Errors**: Insufficient access rights to list branches
- **System Overload**: Temporary unavailability due to high load
- **Configuration Issues**: Incorrect system configuration

### Performance Characteristics

#### Complexity Analysis
- **Branch Enumeration**: O(n) where n is the number of branches
- **Metadata Retrieval**: O(1) per branch for cached metadata
- **Response Formatting**: O(n) for JSON serialization

#### Memory Usage
- **Branch Metadata**: Lightweight structures for each branch
- **Response Data**: JSON representation of all branch information
- **Manager State**: Minimal overhead for branch manager access

#### Usage Statistics Impact
- **Weight**: 10 points per operation (lightweight discovery)
- **Operation Type**: `StatsOperation::ExecuteQuery`

### Integration Points

#### With Database Branching System
```rust
use crate::mcp::llm_friendly_server::database_branching::{
    get_branch_manager, MergeStrategy
};
```

#### With Federation Layer
The system integrates with distributed database federation:
- **Multi-Node Discovery**: Lists branches across all federation nodes
- **Consistent View**: Provides unified view of distributed branches
- **Load Balancing**: Distributes discovery requests across nodes
- **Fault Tolerance**: Handles node failures gracefully

### Best Practices for Developers

1. **Regular Monitoring**: Periodically list branches to understand system state
2. **Branch Cleanup**: Use listing to identify inactive or obsolete branches
3. **Relationship Analysis**: Understand branch lineage before making changes
4. **Status Verification**: Check branch activity status before operations
5. **Naming Conventions**: Use consistent naming patterns visible in listings

### Usage Examples

#### System Overview
```json
{}
```
**Use Case**: Get complete overview of all branches in the system

#### Branch Audit
```json
{}
```
**Use Case**: Audit branch usage and identify cleanup candidates

#### Pre-Operation Check
```json
{}
```
**Use Case**: Verify branch existence and status before performing operations

### Advanced Features

#### Branch Filtering (Conceptual)
While the current implementation lists all branches, future enhancements could include:
```json
{
  "filter": {
    "active_only": true,
    "created_after": "2024-01-01T00:00:00Z",
    "created_from": "main"
  }
}
```

#### Sorting Options (Conceptual)
```json
{
  "sort_by": "created_at",
  "sort_order": "desc"
}
```

### Suggestions System
```rust
let suggestions = vec![
    "Create new branches for experiments".to_string(),
    "Compare branches to track changes".to_string(),
    "Merge completed work back to main".to_string(),
];
```

### Use Cases and Applications

#### Development Workflow Management
- **Feature Tracking**: Monitor active feature development branches
- **Release Planning**: Identify branches ready for merge to main
- **Team Coordination**: See what branches teammates are working on
- **Resource Planning**: Understand branch storage and resource usage

#### System Administration
- **Branch Maintenance**: Identify inactive branches for cleanup
- **Storage Management**: Monitor branch count and storage implications
- **Performance Monitoring**: Track branch creation patterns
- **Backup Planning**: Understand branch structure for backup strategies

#### Research and Analysis
- **Experiment Tracking**: Monitor research experiment branches
- **Collaboration**: See shared research branches across teams
- **Version Management**: Understand knowledge evolution through branches
- **Comparative Analysis**: Identify branches for comparison studies

### Branch Lifecycle Visibility

#### Creation Patterns
```rust
"created_at": branch.created_at.to_rfc3339(),
"created_from": branch.created_from.as_str(),
```

**Pattern Analysis:**
- **Creation Frequency**: Understand branch creation patterns over time
- **Source Distribution**: See which branches are most commonly used as sources
- **Naming Trends**: Identify naming conventions and patterns
- **Team Activity**: Correlate branch creation with team activities

#### Branch Evolution
The listing provides insights into:
- **Branch Hierarchy**: Understand the tree structure of branches
- **Active Development**: Identify currently active development streams  
- **Historical Context**: See the evolution of knowledge through branching
- **Collaboration Patterns**: Understand how teams use branching

### Tool Integration Workflow

1. **Branch Manager Access**: Initialize connection to distributed branch management system
2. **Branch Discovery**: Retrieve comprehensive list of all available branches
3. **Metadata Enrichment**: Gather detailed information for each discovered branch
4. **Status Assessment**: Determine current activity and availability status
5. **Relationship Mapping**: Identify parent-child relationships between branches
6. **Response Formatting**: Structure branch information for both API and human consumption
7. **Suggestion Generation**: Provide contextual recommendations for branch management
8. **Usage Tracking**: Update system analytics for branch discovery effectiveness

This tool provides essential branch discovery capabilities for the LLMKG system, enabling comprehensive visibility into the distributed branching structure and facilitating informed decision-making about branch operations and management.