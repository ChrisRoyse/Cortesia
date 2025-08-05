# Task Execution Guide - Specialized Embedding System

## 🎯 **EXECUTION PHILOSOPHY**

Every task in this system follows **CLAUDE.md TDD methodology** with strict **10-minute atomic cycles**. This guide ensures consistent execution across all 500+ tasks.

## 📋 **TASK STRUCTURE TEMPLATE**

Every task follows this exact pattern:

```markdown
### Task XXX: [Task Name]
**Type**: [Foundation/Implementation/Integration/Testing/Optimization]
**Duration**: 10 minutes
**Dependencies**: [Previous task IDs]

#### TDD Cycle
1. **RED Phase**: [Write failing test description]
2. **GREEN Phase**: [Minimal implementation approach]
3. **REFACTOR Phase**: [Clean, optimized solution]

#### Verification
- [ ] [Specific success criteria]
- [ ] [Integration point validation]
- [ ] [Performance target met]
```

## 🔄 **TDD CYCLE EXECUTION**

### **RED Phase (3 minutes)**
```rust
// 1. Write the FAILING test first
#[test]
fn test_feature_not_implemented() {
    let component = ComponentUnderTest::new();
    let result = component.target_method("input");
    assert!(result.is_err()); // MUST FAIL initially
}

// 2. Run the test - verify it FAILS
cargo test test_feature_not_implemented
// Expected: compilation error or test failure
```

**Critical Rules**:
- Test MUST fail initially (red state)
- Test defines the exact behavior you want
- No implementation code written yet

### **GREEN Phase (5 minutes)**
```rust
// 3. Write MINIMAL code to make test pass
impl ComponentUnderTest {
    pub fn target_method(&self, input: &str) -> Result<Output> {
        // Simplest possible implementation
        Ok(Output::default())
    }
}

// 4. Run test again - verify it PASSES
cargo test test_feature_not_implemented
// Expected: test passes (green state)
```

**Critical Rules**:
- Minimal code only - no extra features
- Make the test pass as simply as possible
- No optimization or edge cases yet

### **REFACTOR Phase (2 minutes)**
```rust
// 5. Clean up and optimize while keeping tests green
impl ComponentUnderTest {
    pub fn target_method(&self, input: &str) -> Result<Output> {
        // Clean, proper implementation
        self.validate_input(input)?;
        let processed = self.process_input(input);
        self.generate_output(processed)
    }
}

// 6. Run tests continuously during refactoring
cargo test
// Expected: all tests stay green
```

**Critical Rules**:
- Tests must stay green throughout refactoring
- Improve code structure and readability
- Add error handling and edge cases

## 🛠️ **EXECUTION WORKFLOW**

### **1. Task Preparation (30 seconds)**
```bash
# Navigate to project
cd /path/to/vector-search

# Create branch for task
git checkout -b task-XXX-feature-name

# Verify starting state
cargo test
cargo check
```

### **2. Read Task Requirements (30 seconds)**
- Review task description and dependencies
- Understand exact deliverables
- Identify integration points

### **3. Execute TDD Cycle (9 minutes)**
Follow RED-GREEN-REFACTOR exactly as above

### **4. Verify and Commit (30 seconds)**
```bash
# Run all tests
cargo test

# Check for compilation issues
cargo check

# Commit atomic change
git add .
git commit -m "task-XXX: [brief description]

RED: [test description]
GREEN: [implementation description]  
REFACTOR: [improvements made]"
```

## 📊 **TASK CATEGORIES**

### **Foundation Tasks (000-099, 100-109, etc.)**
- Create basic structures and traits
- Define interfaces and error types
- Set up module organization
- **Duration**: Exactly 10 minutes each

### **Implementation Tasks (010-089, 110-189, etc.)**
- Implement core functionality
- Add business logic
- Handle data processing
- **Duration**: Exactly 10 minutes each

### **Integration Tasks (090-099, 190-199, etc.)**
- Connect components together
- Test real integration points
- Verify system boundaries
- **Duration**: Exactly 10 minutes each

### **Testing Tasks (050-059, 150-159, etc.)**
- Add comprehensive test coverage
- Create integration test suites
- Performance benchmarking
- **Duration**: Exactly 10 minutes each

### **Optimization Tasks (480-499, etc.)**
- Performance improvements
- Memory optimization
- Concurrency enhancements
- **Duration**: Exactly 10 minutes each

## ⚡ **PARALLEL EXECUTION STRATEGY**

### **Feature-Based Parallelism**
Different features can be implemented in parallel:

```bash
# Developer 1: Content Detection (000-099)
git checkout -b content-detection-feature

# Developer 2: Embedding Models (100-199)  
git checkout -b embedding-models-feature

# Developer 3: Vector Storage (200-299)
git checkout -b vector-storage-feature

# Developer 4: Git Watching (300-399)
git checkout -b git-watching-feature

# Developer 5: MCP Server (400-499)
git checkout -b mcp-server-feature
```

### **Task Dependencies**
```rust
// Dependencies are clearly marked in each task
Task 001 depends on: Task 000
Task 015 depends on: Tasks 010, 012
Task 200 depends on: Tasks 150-199 (embedding models complete)
```

## 🔍 **VERIFICATION CHECKLIST**

### **Per-Task Verification**
- [ ] Test follows TDD RED-GREEN-REFACTOR cycle
- [ ] Implementation is minimal but complete
- [ ] Code compiles without warnings
- [ ] All tests pass (including existing tests)
- [ ] Task completed in exactly 10 minutes
- [ ] Dependencies satisfied
- [ ] Integration points verified

### **Per-Feature Verification** 
- [ ] All tasks in feature complete (000-099, etc.)
- [ ] Feature integration tests pass
- [ ] Performance targets met
- [ ] Documentation updated
- [ ] Ready for next feature integration

### **System-Wide Verification**
- [ ] All features integrate correctly
- [ ] End-to-end tests pass
- [ ] Performance targets met system-wide
- [ ] MCP server responds correctly
- [ ] Git integration works in real repository

## 🚨 **COMMON PITFALLS**

### **❌ RED Phase Mistakes**
- Writing implementation code before test
- Test doesn't actually fail initially
- Test is too complex or tests multiple things
- Not running the test to verify failure

### **❌ GREEN Phase Mistakes**
- Implementing more than needed to pass test
- Adding optimization before basic functionality
- Not running test to verify it passes
- Implementing multiple features at once

### **❌ REFACTOR Phase Mistakes**
- Breaking existing tests during refactoring
- Adding new functionality instead of cleaning
- Not running tests continuously
- Optimizing without measuring performance

### **❌ General Execution Mistakes**
- Taking longer than 10 minutes per task
- Skipping TDD cycle steps
- Not committing after each task
- Working on multiple tasks simultaneously

## 🎯 **SUCCESS METRICS**

### **Individual Task Success**
- ✅ Completed in exactly 10 minutes
- ✅ Follows RED-GREEN-REFACTOR cycle
- ✅ All tests pass
- ✅ Code compiles cleanly
- ✅ Meets task verification criteria

### **Feature Success**
- ✅ All tasks in feature complete
- ✅ Feature integration tests pass
- ✅ Performance targets achieved
- ✅ Ready for system integration

### **System Success** 
- ✅ 98-99% search accuracy achieved
- ✅ < 500ms MCP response time
- ✅ Git integration works seamlessly
- ✅ LLM can use tools successfully
- ✅ Production ready with monitoring

## 🚀 **GETTING STARTED**

1. **Choose Starting Feature**: Content Detection (Tasks 000-099) recommended
2. **Set Up Environment**: Ensure Rust, git, and dependencies installed
3. **Start Task 000**: Follow TDD cycle exactly
4. **Track Progress**: Use verification checklist for each task
5. **Integrate Features**: Connect completed features progressively
6. **Deploy MCP Server**: Final integration and testing

**Remember**: The discipline of 10-minute atomic TDD cycles is what makes this system achievable and maintainable. Never skip the process, even if it feels slow initially.