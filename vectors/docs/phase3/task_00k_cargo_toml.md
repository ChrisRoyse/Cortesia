# Task 00k: Update Cargo.toml with Dependencies

**Estimated Time: 5 minutes**
**Lines of Code: 15**
**Prerequisites: Task 00j completed**

## Context

The foundation code requires specific dependencies to compile. This task adds the minimal required dependencies to Cargo.toml.

## Your Task

Add the required dependencies to the existing `Cargo.toml` file in the workspace.

## Required Implementation

Add these dependencies to the `[dependencies]` section in `Cargo.toml`:

```toml
# Error handling
anyhow = "1.0"
thiserror = "1.0"

# Core search engine (commented out until Phase 4)
# tantivy = "0.21"

# Serialization (for future use)
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Async runtime (for future use)
tokio = { version = "1.0", features = ["full"] }

# Utilities
uuid = { version = "1.0", features = ["v4"] }
```

Add to `[dev-dependencies]`:

```toml
tempfile = "3.0"
```

## Success Criteria

- [ ] `anyhow` and `thiserror` dependencies added (required for compilation)
- [ ] `serde`, `tokio`, `uuid` dependencies added (Phase 3 tasks may reference)
- [ ] `tempfile` added to dev-dependencies (needed for tests)
- [ ] Tantivy commented out (not needed until Phase 4)
- [ ] All version numbers specified correctly
- [ ] File compiles without errors

## Validation

Run `cargo check` - should compile without errors.

## Completion

This completes the Task 00 split. All Phase 3 tasks should now be able to compile and reference the foundation interfaces.