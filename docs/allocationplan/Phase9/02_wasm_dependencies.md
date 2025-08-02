# Micro-Phase 9.02: WASM Dependencies Configuration

## Objective
Configure all necessary WASM dependencies for CortexKG including wasm-bindgen, web-sys, and optimization libraries.

## Prerequisites
- Completed micro-phase 9.01 (WASM project setup)
- Cargo.toml file exists in project

## Task Description
Add and configure all required dependencies for WASM compilation, browser API access, and performance optimization.

## Specific Actions

1. **Update Cargo.toml with core dependencies**:
   ```toml
   [dependencies]
   wasm-bindgen = "0.2.92"
   wasm-bindgen-futures = "0.4.42"
   js-sys = "0.3.69"
   serde = { version = "1.0", features = ["derive"] }
   serde-wasm-bindgen = "0.6.5"

   # WASM-specific optimizations
   wee_alloc = { version = "0.4.5", optional = true }
   console_error_panic_hook = { version = "0.1.7", optional = true }

   # Web APIs
   web-sys = { version = "0.3.69", features = [
       "console",
       "Document",
       "Element",
       "HtmlElement",
       "HtmlCanvasElement",
       "CanvasRenderingContext2d",
       "Window",
       "Performance",
       "Storage",
       "IdbFactory",
       "IdbDatabase",
       "IdbObjectStore",
       "IdbRequest",
       "IdbTransaction",
       "IdbIndex",
       "IdbKeyRange",
       "Event",
       "EventTarget",
       "MouseEvent",
       "TouchEvent",
       "Touch",
       "TouchList",
       "RequestInit",
       "RequestMode",
       "Request",
       "Response",
       "Headers",
   ]}

   # SIMD support
   [target.'cfg(target_arch = "wasm32")'.dependencies]
   stdarch = { version = "0.1", features = ["wasm_simd128"] }
   ```

2. **Add development dependencies**:
   ```toml
   [dev-dependencies]
   wasm-bindgen-test = "0.3.42"
   ```

3. **Configure features**:
   ```toml
   [features]
   default = ["console_error_panic_hook", "wee_alloc"]
   simd = ["stdarch"]
   debug = ["console_error_panic_hook"]
   ```

4. **Create utils module for panic handling**:
   ```rust
   // src/utils.rs
   pub fn set_panic_hook() {
       #[cfg(feature = "console_error_panic_hook")]
       console_error_panic_hook::set_once();
   }

   #[wasm_bindgen]
   pub fn init_wasm() {
       set_panic_hook();
   }
   ```

5. **Update lib.rs to use utils**:
   ```rust
   mod utils;

   use wasm_bindgen::prelude::*;

   #[wasm_bindgen(start)]
   pub fn main() {
       utils::set_panic_hook();
   }
   ```

## Expected Outputs
- Complete dependency configuration in Cargo.toml
- All required web-sys features enabled
- Panic handling utilities created
- Optional features properly configured

## Validation
1. Run `cargo check --target wasm32-unknown-unknown`
2. All dependencies resolve without conflicts
3. Features compile correctly
4. No missing web-sys features for planned functionality

## Next Steps
- Setup build configuration (micro-phase 9.03)
- Create core WASM bindings (micro-phase 9.04)