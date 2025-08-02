# merge_branches - Branch Integration and Consolidation Tool

## Overview

The `merge_branches` tool provides sophisticated branch merging capabilities for the LLMKG system, enabling users to integrate changes from one knowledge graph branch into another. It supports multiple merge strategies for handling conflicts, provides detailed merge results, and ensures data integrity throughout the merge process. This tool is essential for collaborative development workflows where multiple branches need to be consolidated while maintaining knowledge graph consistency.

## Implementation Details

### Handler Location
- **File**: `src/mcp/llm_friendly_server/handlers/temporal.rs`
- **Function**: `handle_merge_branches`
- **Lines**: 340-419

### Core Functionality

The tool implements comprehensive branch merging capabilities:

1. **Multi-Strategy Merging**: Support different conflict resolution approaches
2. **Change Integration**: Merge additions, deletions, and modifications
3. **Conflict Resolution**: Handle contradictory information between branches
4. **Integrity Preservation**: Maintain knowledge graph consistency during merges
5. **Result Reporting**: Provide detailed merge outcome information
6. **Rollback Support**: Enable reverting merge operations if needed

### Input Schema

```json
{
  "type": "object",
  "properties": {
    "source_branch": {
      "type": "string",
      "description": "Branch to merge FROM (contains new changes)"
    },
    "target_branch": {
      "type": "string",
      "description": "Branch to merge INTO (will receive changes)"
    },
    "merge_strategy": {
      "type": "string",
      "description": "How to handle conflicts",
      "enum": ["accept_source", "accept_target", "manual"],
      "default": "accept_source"
    }
  },
  "required": ["source_branch", "target_branch"]
}
```

### Key Variables and Functions

#### Primary Handler Function
```rust
pub async fn handle_merge_branches(
    _knowledge_engine: &Arc<RwLock<KnowledgeEngine>>,
    usage_stats: &Arc<RwLock<UsageStats>>,
    params: Value,
) -> std::result::Result<(Value, String, Vec<String>), String>
```

#### Input Processing Variables
```rust
let source_branch = params.get("source_branch")
    .and_then(|v| v.as_str())
    .ok_or("Missing required field: source_branch")?;

let target_branch = params.get("target_branch")
    .and_then(|v| v.as_str())
    .ok_or("Missing required field: target_branch")?;

let merge_strategy_str = params.get("merge_strategy")
    .and_then(|v| v.as_str())
    .unwrap_or("accept_source");
```

### Merge Strategy System

#### Strategy Enumeration
```rust
use crate::mcp::llm_friendly_server::database_branching::{
    get_branch_manager, MergeStrategy
};

let merge_strategy = match merge_strategy_str {
    "accept_source" => MergeStrategy::AcceptSource,
    "accept_target" => MergeStrategy::AcceptTarget,
    "manual" => MergeStrategy::Manual,
    _ => return Err(format!("Invalid merge strategy: {}", merge_strategy_str))
};
```

#### Merge Strategy Implementations

**1. Accept Source Strategy**
```rust
MergeStrategy::AcceptSource
```
- **Behavior**: Source branch changes take precedence over target
- **Use Case**: When source contains authoritative updates
- **Conflict Resolution**: Source version is used for all conflicts
- **Risk Level**: Low (predictable outcomes)

**2. Accept Target Strategy**
```rust
MergeStrategy::AcceptTarget
```
- **Behavior**: Target branch is preserved, source changes are filtered
- **Use Case**: When target branch must remain unchanged
- **Conflict Resolution**: Target version is preserved for all conflicts
- **Risk Level**: Low (conservative approach)

**3. Manual Strategy**
```rust
MergeStrategy::Manual
```
- **Behavior**: Conflicts are flagged for human resolution
- **Use Case**: When careful conflict resolution is required
- **Conflict Resolution**: Requires manual intervention for conflicts
- **Risk Level**: High (requires expert knowledge)

### Branch Manager Integration

#### Branch Manager Access
```rust
let branch_manager_arc = get_branch_manager().await
    .map_err(|e| format!("Failed to get branch manager: {}", e))?;

let branch_manager_guard = branch_manager_arc.read().await;
let branch_manager = branch_manager_guard.as_ref()
    .ok_or("Branch manager not initialized")?;
```

#### Merge Execution
```rust
let merge_result = branch_manager.merge_branches(
    source_branch,
    target_branch,
    merge_strategy,
).await
    .map_err(|e| format!("Failed to merge branches: {}", e))?;
```

The branch manager's merge functionality provides:
- **Atomic Operations**: All-or-nothing merge execution
- **Conflict Detection**: Automatic identification of contradictory information
- **Strategy Application**: Consistent application of chosen merge strategy
- **Rollback Capability**: Ability to undo merge operations if needed

### Merge Result Structure

#### Merge Result Processing
```rust
// Inferred from usage patterns
struct MergeResult {
    success: bool,
    triples_added: usize,
    triples_removed: usize,
    conflicts_resolved: usize,
    message: String,
    // Additional merge metadata
}
```

#### Result Components
- **success**: Boolean indicating overall merge success
- **triples_added**: Number of new relationships added to target
- **triples_removed**: Number of relationships removed from target
- **conflicts_resolved**: Number of conflicts handled during merge
- **message**: Descriptive message about merge outcome

### Merge Algorithm Details

#### Change Detection Phase
The merge process begins by analyzing differences:
1. **Content Comparison**: Identify unique content in each branch
2. **Conflict Identification**: Find contradictory information
3. **Change Classification**: Categorize additions, deletions, modifications
4. **Impact Assessment**: Evaluate merge complexity and risks

#### Strategy Application Phase
Based on the chosen strategy:
1. **Conflict Resolution**: Apply strategy-specific conflict handling
2. **Change Integration**: Merge non-conflicting changes
3. **Validation**: Ensure knowledge graph consistency
4. **Rollback Preparation**: Prepare for potential merge reversal

#### Result Generation Phase
After merge completion:
1. **Statistics Collection**: Count changes and conflicts
2. **Validation Checks**: Verify merge integrity
3. **Result Compilation**: Generate comprehensive merge report
4. **Success Confirmation**: Confirm merge completion status

### Output Format

#### Comprehensive Merge Response
```json
{
  "source_branch": "quantum-physics-exploration",
  "target_branch": "main",
  "merge_strategy": "accept_source",
  "result": {
    "success": true,
    "triples_added": 57,
    "triples_removed": 0,
    "conflicts_resolved": 3,
    "message": "Merge completed successfully. All quantum physics relationships integrated."
  },
  "timestamp": "2024-01-15T14:30:00Z"
}
```

#### Human-Readable Message Format
```rust
let message = format!(
    "Merge Results:\n\
    ✅ Status: {}\n\
    🔀 {} → {}\n\
    📝 Strategy: {}\n\
    ➕ Triples Added: {}\n\
    ➖ Triples Removed: {}\n\
    🔧 Conflicts Resolved: {}\n\
    💬 {}",
    if merge_result.success { "Success" } else { "Failed" },
    source_branch, target_branch,
    merge_strategy_str,
    merge_result.triples_added,
    merge_result.triples_removed,
    merge_result.conflicts_resolved,
    merge_result.message
);
```

**Example Human-Readable Output:**
```
Merge Results:
✅ Status: Success
🔀 quantum-physics-exploration → main
📝 Strategy: accept_source
➕ Triples Added: 57
➖ Triples Removed: 0
🔧 Conflicts Resolved: 3
💬 Merge completed successfully. All quantum physics relationships integrated.
```

### Conflict Resolution Mechanisms

#### Conflict Types
The system handles various conflict types:
- **Contradictory Facts**: Same subject-predicate with different objects
- **Inconsistent Relationships**: Relationships that logically conflict
- **Temporal Conflicts**: Time-based inconsistencies
- **Quality Conflicts**: Different confidence levels for same information

#### Resolution Strategies

**Accept Source Conflicts**
```rust
MergeStrategy::AcceptSource
```
- **Implementation**: Source values override target values
- **Advantages**: Clear, deterministic outcomes
- **Disadvantages**: May lose valuable target information
- **Best For**: Authoritative updates from trusted sources

**Accept Target Conflicts**
```rust
MergeStrategy::AcceptTarget
```
- **Implementation**: Target values are preserved
- **Advantages**: Maintains stability of target branch
- **Disadvantages**: May ignore valuable source improvements
- **Best For**: Conservative merges where target must be preserved

**Manual Conflicts**
```rust
MergeStrategy::Manual
```
- **Implementation**: Conflicts are flagged for human review
- **Advantages**: Ensures careful consideration of all conflicts
- **Disadvantages**: Requires human intervention and expertise
- **Best For**: Critical merges requiring expert judgment

### Advanced Merge Features

#### Atomic Operations
```rust
let merge_result = branch_manager.merge_branches(
    source_branch,
    target_branch,
    merge_strategy,
).await
    .map_err(|e| format!("Failed to merge branches: {}", e))?;
```

All merge operations are atomic:
- **All-or-Nothing**: Either complete merge succeeds or no changes are made
- **Consistency Guarantee**: Knowledge graph remains consistent throughout
- **Rollback Capability**: Failed merges don't leave partial changes
- **Isolation**: Other operations don't interfere with merge process

#### Validation and Integrity
The merge process includes comprehensive validation:
- **Graph Consistency**: Ensures no orphaned or invalid relationships
- **Confidence Validation**: Maintains confidence score integrity
- **Temporal Consistency**: Preserves temporal relationship validity
- **Schema Compliance**: Ensures merged data follows schema requirements

### Error Handling

#### Parameter Validation
```rust
let source_branch = params.get("source_branch")
    .and_then(|v| v.as_str())
    .ok_or("Missing required field: source_branch")?;

let target_branch = params.get("target_branch")
    .and_then(|v| v.as_str())
    .ok_or("Missing required field: target_branch")?;
```

#### Strategy Validation
```rust
let merge_strategy = match merge_strategy_str {
    "accept_source" => MergeStrategy::AcceptSource,
    "accept_target" => MergeStrategy::AcceptTarget,
    "manual" => MergeStrategy::Manual,
    _ => return Err(format!("Invalid merge strategy: {}", merge_strategy_str))
};
```

#### Merge Execution Errors
```rust
let merge_result = branch_manager.merge_branches(
    source_branch,
    target_branch,
    merge_strategy,
).await
    .map_err(|e| format!("Failed to merge branches: {}", e))?;
```

**Error Scenarios:**
- **Branch Not Found**: Source or target branch doesn't exist
- **Permission Denied**: Insufficient rights to merge branches
- **Conflict Overload**: Too many conflicts for automatic resolution
- **Resource Limitations**: Insufficient resources for merge operation
- **Consistency Violations**: Merge would violate knowledge graph constraints

### Performance Characteristics

#### Complexity Analysis
- **Difference Detection**: O(n) where n is combined branch size
- **Conflict Resolution**: O(c) where c is number of conflicts
- **Integration**: O(m) where m is number of changes to apply
- **Validation**: O(n) for consistency checking

#### Memory Usage
- **Merge State**: Temporary structures for merge processing
- **Conflict Data**: Storage for identified conflicts and resolutions
- **Change Sets**: Lists of additions, deletions, and modifications
- **Rollback Information**: Data needed for potential merge reversal

#### Usage Statistics Impact
- **Weight**: 100 points per operation (expensive operation)
- **Operation Type**: `StatsOperation::ExecuteQuery`

### Integration Points

#### With Database Branching System
```rust
use crate::mcp::llm_friendly_server::database_branching::{
    get_branch_manager, MergeStrategy
};
```

The integration provides:
- **Distributed Merging**: Handle merges across federation nodes
- **Consistency Maintenance**: Ensure global consistency during merges
- **Conflict Resolution**: Apply sophisticated conflict resolution algorithms
- **Performance Optimization**: Efficient merge operations for large branches

#### With Version Management
The merge system integrates with version tracking:
- **Version History**: Record merge operations in version history
- **Rollback Support**: Enable reverting to pre-merge states
- **Branch Lineage**: Maintain branch relationship information
- **Change Tracking**: Record detailed change information

### Best Practices for Developers

1. **Pre-Merge Comparison**: Always compare branches before merging
2. **Strategy Selection**: Choose appropriate strategy based on merge context
3. **Backup Creation**: Create backups before significant merges
4. **Conflict Review**: Examine conflicts carefully in manual mode
5. **Result Validation**: Verify merge results meet expectations

### Usage Examples

#### Feature Integration
```json
{
  "source_branch": "feature-temporal-reasoning",
  "target_branch": "main",
  "merge_strategy": "accept_source"
}
```
**Use Case**: Integrate completed feature branch back to main

#### Bug Fix Merge
```json
{
  "source_branch": "hotfix-critical-bug",
  "target_branch": "main",
  "merge_strategy": "accept_source"
}
```
**Use Case**: Apply critical bug fix to production branch

#### Experimental Integration
```json
{
  "source_branch": "research-experiment",
  "target_branch": "main",
  "merge_strategy": "manual"
}
```
**Use Case**: Carefully integrate experimental changes with manual review

#### Conservative Update
```json
{
  "source_branch": "external-updates",
  "target_branch": "main",
  "merge_strategy": "accept_target"
}
```
**Use Case**: Apply external updates while preserving local changes

### Suggestions System
```rust
let suggestions = vec![
    "Verify the merge results".to_string(),
    "Consider creating a backup branch before major merges".to_string(),
    "Review any unresolved conflicts".to_string(),
];
```

### Advanced Merge Scenarios

#### Multi-Way Merges
While the current implementation focuses on two-branch merges, the system can handle:
- **Sequential Merges**: Chain multiple merge operations
- **Diamond Merges**: Merge branches that share common ancestry
- **Conflict Cascades**: Handle conflicts that create additional conflicts

#### Selective Merging
Future enhancements could support:
- **Path-Specific Merges**: Merge only specific knowledge domains
- **Entity-Specific Merges**: Merge changes related to specific entities
- **Time-Based Merges**: Merge changes from specific time periods

### Use Cases and Applications

#### Development Workflows
- **Feature Integration**: Merge completed features back to main development line
- **Bug Fix Application**: Apply fixes across multiple branches
- **Release Preparation**: Consolidate changes for release branches
- **Hotfix Deployment**: Rapidly deploy critical fixes

#### Research and Collaboration
- **Experiment Integration**: Merge successful experiments into main knowledge
- **Team Collaboration**: Integrate work from multiple research teams
- **Knowledge Synthesis**: Combine knowledge from different sources
- **Version Consolidation**: Merge different versions of domain knowledge

#### Data Management
- **Source Integration**: Merge knowledge from external data sources  
- **Quality Improvement**: Merge higher-quality versions of existing knowledge
- **Schema Evolution**: Merge branches with schema improvements
- **Cleanup Operations**: Merge branches with data cleaning improvements

### Tool Integration Workflow

1. **Input Processing**: Validate branch names and merge strategy selection
2. **Branch Manager Access**: Initialize connection to distributed branch management
3. **Pre-Merge Analysis**: Analyze differences and potential conflicts
4. **Strategy Application**: Apply selected conflict resolution strategy
5. **Change Integration**: Execute merge operations with integrity preservation
6. **Conflict Resolution**: Handle conflicts according to chosen strategy
7. **Validation**: Verify merge results and knowledge graph consistency
8. **Result Compilation**: Generate comprehensive merge outcome report
9. **Cleanup**: Perform any necessary post-merge cleanup operations
10. **Usage Tracking**: Update system analytics for merge effectiveness

This tool provides essential branch merging capabilities for the LLMKG system, enabling sophisticated integration of knowledge graph changes while maintaining data integrity and supporting collaborative development workflows through multiple conflict resolution strategies and comprehensive result reporting.