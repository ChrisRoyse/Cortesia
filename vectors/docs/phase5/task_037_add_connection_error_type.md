# Task 037: Add Connection Error Type

## Context
Continuing vector store implementation. Need error handling for connections.

## Task Objective
Add ConnectionFailed variant to VectorStoreError enum.

## Steps
1. Open src/vector_store.rs
2. Find VectorStoreError enum
3. Add: ConnectionFailed(String),
4. Save file

## Success Criteria
- [ ] Error variant added
- [ ] File compiles

## Time: 3 minutes

## Next Task
Task 038: Add schema validation error type