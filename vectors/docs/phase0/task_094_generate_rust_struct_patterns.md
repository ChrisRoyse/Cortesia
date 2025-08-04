# Micro-Task 094: Generate Rust Struct Patterns

## Objective
Create test files with Rust struct definitions including special characters.

## Context
Struct patterns test the parser's ability to handle complex type definitions.

## Prerequisites
- Task 093 completed (code generation validated)

## Time Estimate
10 minutes

## Instructions
1. Navigate to test data: `cd data/test_files`
2. Create `rust_patterns` directory: `mkdir rust_patterns`
3. Create `generate_structs.py`:
   ```python
   patterns = [
       "struct Point<T> { x: T, y: T }",
       "#[derive(Debug, Clone)]
struct Data<'a> { field: &'a str }",
       "pub struct Result<T, E> { ok: Option<T>, err: Option<E> }"
   ]
   
   with open('rust_patterns/struct_definitions.rs', 'w') as f:
       f.write('\n\n'.join(patterns))
   ```
4. Run: `python generate_structs.py`
5. Validate: `rustc --edition 2021 --crate-type lib rust_patterns/struct_definitions.rs`

## Expected Output
- `rust_patterns/struct_definitions.rs` created
- Multiple struct patterns with generics
- File validates as valid Rust

## Success Criteria
- [ ] Struct patterns generated
- [ ] Special characters included (<>, #[], 'a)
- [ ] Rust compilation validates syntax
- [ ] At least 5 different patterns

## Next Task
task_095_generate_rust_enum_patterns.md