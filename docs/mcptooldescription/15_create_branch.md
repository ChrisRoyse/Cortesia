# create_branch - Knowledge Graph Branch Creation Tool

## Overview

The `create_branch` tool enables users to create new branches of the knowledge graph database, similar to git branches. This allows for safe experimentation with knowledge modifications without affecting the main database. Users can create isolated branches for testing new relationships, importing experimental data, or developing alternative knowledge structures that can later be merged back or discarded as needed.

## Implementation Details

### Handler Location
- **File**: `src/mcp/llm_friendly_server/handlers/temporal.rs`
- **Function**: `handle_create_branch`
- **Lines**: 154-218

### Core Functionality

The tool implements sophisticated database branching capabilities:

1. **Branch Creation**: Create new database branches from existing ones
2. **Version Management**: Integration with multi-database version manager
3. **Metadata Tracking**: Store branch descriptions and creation timestamps
4. **Branch Isolation**: Ensure complete data isolation between branches
5. **Database ID Generation**: Automatic unique identifier creation
6. **Branch Manager Integration**: Interface with distributed branch management system

### Input Schema

```json
{
  "type": "object",
  "properties": {
    "source_db_id": {
      "type": "string",
      "description": "The database ID to branch from (use 'main' for the main database)"
    },
    "branch_name": {
      "type": "string",
      "description": "Name for the new branch (e.g., 'experiment-1', 'feature-xyz')"
    },
    "description": {
      "type": "string",
      "description": "Optional description of what this branch is for"
    }
  },
  "required": ["source_db_id", "branch_name"]
}
```

### Key Variables and Functions

#### Primary Handler Function
```rust
pub async fn handle_create_branch(
    _knowledge_engine: &Arc<RwLock<KnowledgeEngine>>,
    usage_stats: &Arc<RwLock<UsageStats>>,
    params: Value,
    _version_manager: Arc<MultiDatabaseVersionManager>,
) -> std::result::Result<(Value, String, Vec<String>), String>
```

#### Input Processing Variables
```rust
let source_db_id = params.get("source_db_id")
    .and_then(|v| v.as_str())
    .ok_or("Missing required field: source_db_id")?;

let branch_name = params.get("branch_name")
    .and_then(|v| v.as_str())
    .ok_or("Missing required field: branch_name")?;

let description = params.get("description")
    .and_then(|v| v.as_str())
    .map(String::from);
```

### Branch Manager Integration

#### Branch Manager Access
```rust
use crate::mcp::llm_friendly_server::database_branching::{
    get_branch_manager, MergeStrategy
};

// Get or initialize branch manager
let branch_manager_arc = get_branch_manager().await
    .map_err(|e| format!("Failed to get branch manager: {}", e))?;

let branch_manager_guard = branch_manager_arc.read().await;
let branch_manager = branch_manager_guard.as_ref()
    .ok_or("Branch manager not initialized")?;
```

The system uses a sophisticated branch manager that handles:
- **Distributed Branching**: Manages branches across multiple database instances
- **Consistency Maintenance**: Ensures branch integrity and isolation
- **Resource Management**: Handles database connections and cleanup
- **Conflict Prevention**: Avoids naming conflicts and validation issues

#### Branch Creation Process
```rust
// Create the branch
let new_db_id = branch_manager.create_branch(
    &DatabaseId::new(source_db_id.to_string()),
    branch_name.to_string(),
    description.clone(),
).await
    .map_err(|e| format!("Failed to create branch: {}", e))?;
```

**Database ID System:**
```rust
use crate::federation::DatabaseId;

let new_db_id = branch_manager.create_branch(
    &DatabaseId::new(source_db_id.to_string()),
    branch_name.to_string(),
    description.clone(),
).await
```

The `DatabaseId` provides:
- **Unique Identification**: Each branch gets a unique database identifier
- **Federation Support**: Enables distributed database management
- **Reference Integrity**: Maintains relationships between parent and child branches

### Branch Creation Workflow

#### 1. Parameter Validation
```rust
let source_db_id = params.get("source_db_id")
    .and_then(|v| v.as_str())
    .ok_or("Missing required field: source_db_id")?;

let branch_name = params.get("branch_name")
    .and_then(|v| v.as_str())
    .ok_or("Missing required field: branch_name")?;
```

**Validation Requirements:**
- **source_db_id**: Must be a valid existing database ID
- **branch_name**: Must be a non-empty string suitable for identification
- **description**: Optional metadata for branch documentation

#### 2. Branch Manager Access
The system accesses a global branch manager that coordinates across the distributed system:

```rust
let branch_manager_arc = get_branch_manager().await
    .map_err(|e| format!("Failed to get branch manager: {}", e))?;
```

**Branch Manager Features:**
- **Async Operations**: All branch operations are non-blocking
- **Thread Safety**: Uses Arc<RwLock> for safe concurrent access
- **Error Handling**: Comprehensive error reporting for branch operations
- **Resource Management**: Automatic cleanup and connection pooling

#### 3. Branch Creation Execution
```rust
let new_db_id = branch_manager.create_branch(
    &DatabaseId::new(source_db_id.to_string()),
    branch_name.to_string(),
    description.clone(),
).await
    .map_err(|e| format!("Failed to create branch: {}", e))?;
```

**Creation Process:**
- **Copy Operation**: Creates a complete copy of the source database
- **Isolation**: Ensures complete data isolation between branches
- **Metadata Recording**: Stores branch creation metadata and relationships
- **ID Generation**: Assigns a unique database ID for the new branch

### Output Format

#### Comprehensive Branch Creation Response
```json
{
  "branch_name": "quantum-physics-exploration",
  "database_id": "main_quantum-physics-exploration",
  "source_database_id": "main",
  "description": "Testing new quantum physics relationships",
  "created_at": "2024-01-15T10:30:00Z",
  "status": "active"
}
```

#### Human-Readable Message Format
```rust
let message = format!(
    "Branch Created Successfully:\n\
    🌿 Branch Name: {}\n\
    📁 New Database ID: {}\n\
    🔗 Created from: {}\n\
    📝 Description: {}",
    branch_name,
    new_db_id.as_str(),
    source_db_id,
    description.as_deref().unwrap_or("No description")
);
```

**Example Human-Readable Output:**
```
Branch Created Successfully:
🌿 Branch Name: quantum-physics-exploration
📁 New Database ID: main_quantum-physics-exploration
🔗 Created from: main
📝 Description: Testing new quantum physics relationships
```

### Version Manager Integration

#### Multi-Database Version Manager
```rust
use crate::versioning::MultiDatabaseVersionManager;

pub async fn handle_create_branch(
    _knowledge_engine: &Arc<RwLock<KnowledgeEngine>>,
    usage_stats: &Arc<RwLock<UsageStats>>,
    params: Value,
    _version_manager: Arc<MultiDatabaseVersionManager>,
) -> std::result::Result<(Value, String, Vec<String>), String>
```

The version manager provides:
- **Version Tracking**: Maintains version history across branches
- **Rollback Capabilities**: Enables reverting to previous states
- **Merge Support**: Facilitates branch merging operations
- **Consistency Checking**: Validates branch states and integrity

### Error Handling

#### Parameter Validation Errors
```rust
let source_db_id = params.get("source_db_id")
    .and_then(|v| v.as_str())
    .ok_or("Missing required field: source_db_id")?;

let branch_name = params.get("branch_name")
    .and_then(|v| v.as_str())
    .ok_or("Missing required field: branch_name")?;
```

#### Branch Manager Errors
```rust
let branch_manager_arc = get_branch_manager().await
    .map_err(|e| format!("Failed to get branch manager: {}", e))?;

let branch_manager_guard = branch_manager_arc.read().await;
let branch_manager = branch_manager_guard.as_ref()
    .ok_or("Branch manager not initialized")?;
```

#### Branch Creation Errors
```rust
let new_db_id = branch_manager.create_branch(
    &DatabaseId::new(source_db_id.to_string()),
    branch_name.to_string(),
    description.clone(),
).await
    .map_err(|e| format!("Failed to create branch: {}", e))?;
```

**Common Error Scenarios:**
- **Invalid Source**: Source database ID doesn't exist
- **Naming Conflicts**: Branch name already exists
- **Resource Limits**: Insufficient resources to create branch
- **Permission Issues**: Insufficient permissions for branch creation
- **System Errors**: Database connection or storage issues

### Performance Characteristics

#### Complexity Analysis
- **Branch Creation**: O(n) where n is the size of the source database
- **Metadata Operations**: O(1) for branch registration and tracking
- **Resource Allocation**: Depends on underlying database implementation

#### Memory Usage
- **Branch Data**: Complete copy of source database (initially)
- **Metadata Storage**: Branch tracking and relationship information
- **Manager State**: Branch manager coordination structures

#### Usage Statistics Impact
- **Weight**: 100 points per operation (expensive database operation)
- **Operation Type**: `StatsOperation::ExecuteQuery`

### Integration Points

#### With Database Branching System
```rust
use crate::mcp::llm_friendly_server::database_branching::{
    get_branch_manager, MergeStrategy
};
```

#### With Federation Layer
```rust
use crate::federation::DatabaseId;
```

The federation layer enables:
- **Distributed Branching**: Branches can span multiple nodes
- **Load Distribution**: Balance branch creation across resources
- **Fault Tolerance**: Ensure branch availability despite node failures

#### With Version Management
```rust
use crate::versioning::MultiDatabaseVersionManager;
```

Version management provides:
- **History Tracking**: Maintain complete version history
- **Rollback Support**: Enable reverting to previous states
- **Merge Facilitation**: Support complex merge operations

### Best Practices for Developers

1. **Descriptive Naming**: Use clear, descriptive branch names that indicate purpose
2. **Documentation**: Always provide meaningful descriptions for branches
3. **Resource Management**: Be mindful of storage implications for large databases
4. **Cleanup**: Regularly remove unused branches to conserve resources
5. **Testing Strategy**: Use branches for safe experimentation and testing

### Usage Examples

#### Experimental Research Branch
```json
{
  "source_db_id": "main",
  "branch_name": "ai-ethics-research",
  "description": "Exploring ethical implications of AI development"
}
```

#### Feature Development Branch
```json
{
  "source_db_id": "main",
  "branch_name": "temporal-reasoning-feature",
  "description": "Adding temporal reasoning capabilities to the knowledge graph"
}
```

#### Data Import Branch
```json
{
  "source_db_id": "main",
  "branch_name": "wikipedia-import-2024",
  "description": "Importing selected Wikipedia data for knowledge expansion"
}
```

#### Testing Branch
```json
{
  "source_db_id": "main",
  "branch_name": "schema-validation-test",
  "description": "Testing new schema validation rules before production deployment"
}
```

### Suggestions System
```rust
let suggestions = vec![
    format!("Switch to the new branch with database_id: {}", new_db_id.as_str()),
    "Compare branches to see differences".to_string(),
    "Merge changes back when ready".to_string(),
];
```

### Branch Lifecycle Management

#### Branch States
- **Active**: Branch is available for read/write operations
- **Inactive**: Branch exists but is not currently accessible
- **Merged**: Branch has been merged back to its parent
- **Archived**: Branch is preserved for historical reference

#### Branch Operations
- **Creation**: Initial branch setup from source database
- **Modification**: Adding/updating knowledge within the branch
- **Comparison**: Analyzing differences between branches
- **Merging**: Integrating changes back to parent or other branches
- **Deletion**: Removing branches that are no longer needed

### Use Cases and Applications

#### Research and Experimentation
- **Hypothesis Testing**: Create branches to test new knowledge relationships
- **Data Source Integration**: Safely integrate new data sources
- **Algorithm Development**: Test new knowledge extraction algorithms
- **Schema Evolution**: Experiment with schema changes

#### Development Workflows
- **Feature Branches**: Isolate feature development work
- **Bug Fix Branches**: Fix issues without affecting main database
- **Rollback Branches**: Create safe points for rollback operations
- **Testing Branches**: Comprehensive testing of knowledge modifications

#### Collaborative Work
- **Team Isolation**: Different teams can work on separate branches
- **Peer Review**: Share branches for review before merging
- **Conflict Resolution**: Resolve conflicting changes through branching
- **Knowledge Versioning**: Maintain different versions of knowledge

### Tool Integration Workflow

1. **Input Processing**: Validate source database ID and branch name requirements
2. **Branch Manager Access**: Obtain and initialize the distributed branch manager
3. **Database ID Creation**: Generate unique identifier for the new branch
4. **Branch Creation**: Execute complete database copy with isolation guarantees
5. **Metadata Recording**: Store branch creation information and relationships
6. **Result Formatting**: Structure comprehensive branch creation response
7. **Suggestion Generation**: Provide next-step recommendations for branch usage
8. **Usage Tracking**: Update system analytics for branch creation effectiveness

This tool provides essential database branching capabilities for the LLMKG system, enabling safe experimentation, collaborative development, and sophisticated version management through a distributed branch management system that ensures data isolation and integrity.