# Micro-Phase 9.01: WASM Project Setup

## Objective
Initialize the WASM project structure for CortexKG web interface with proper directory layout and configuration.

## Prerequisites
- Rust toolchain installed (rustc, cargo)
- wasm-pack installed (`cargo install wasm-pack`)
- Node.js and npm installed

## Task Description
Create the WASM project directory structure and initialize the Rust library configured for WebAssembly compilation.

## Specific Actions

1. **Create WASM project directory**:
   ```bash
   mkdir -p cortexkg-wasm
   cd cortexkg-wasm
   ```

2. **Initialize Rust library project**:
   ```bash
   cargo init --lib
   ```

3. **Create directory structure**:
   ```
   cortexkg-wasm/
   ├── src/
   │   ├── lib.rs           # Main WASM entry point
   │   ├── allocation/      # Allocation engine bindings
   │   ├── query/           # Query processor bindings
   │   ├── storage/         # Browser storage integration
   │   └── utils/           # WASM utilities
   ├── tests/               # WASM tests
   ├── Cargo.toml
   └── build.sh             # Build script
   ```

4. **Update Cargo.toml for WASM**:
   ```toml
   [package]
   name = "cortexkg-wasm"
   version = "0.1.0"
   edition = "2021"

   [lib]
   crate-type = ["cdylib", "rlib"]

   [features]
   default = ["console_error_panic_hook"]

   [profile.release]
   opt-level = "s"
   lto = true
   ```

5. **Create basic lib.rs**:
   ```rust
   use wasm_bindgen::prelude::*;

   // When the `wee_alloc` feature is enabled, use `wee_alloc` as the global allocator
   #[cfg(feature = "wee_alloc")]
   #[global_allocator]
   static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

   #[wasm_bindgen]
   extern {
       fn alert(s: &str);
   }

   #[wasm_bindgen]
   pub fn greet() {
       alert("Hello from CortexKG WASM!");
   }
   ```

6. **Create build script**:
   ```bash
   #!/bin/bash
   wasm-pack build --target web --out-dir pkg
   ```

## Expected Outputs
- Initialized Rust library project configured for WASM
- Proper directory structure for module organization
- Basic WASM entry point that can be compiled
- Build script for easy compilation

## Validation
1. Run `cargo check` - should succeed
2. Directory structure matches specification
3. Cargo.toml has WASM-specific configuration
4. lib.rs contains valid WASM bindgen code

## Next Steps
- Configure WASM dependencies (micro-phase 9.02)
- Setup build optimization (micro-phase 9.03)