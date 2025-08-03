//! WASM bindings for neuromorphic processing
//!
//! Provides web-compatible interface with SIMD acceleration for
//! browser-based neuromorphic computation.

use wasm_bindgen::prelude::*;

pub mod simd_bindings;
pub mod snn_wasm;
pub mod ttfs_wasm;

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global allocator
// Note: These features will be enabled when building for WASM target
// #[cfg(all(target_arch = "wasm32", feature = "wee_alloc"))]
// #[global_allocator]
// static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

/// Initialize the WASM module
#[wasm_bindgen(start)]
pub fn init() {
    // Set panic hook for better error messages in browser console
    // Will be enabled when console_error_panic_hook is added as dependency
    // #[cfg(feature = "console_error_panic_hook")]
    // console_error_panic_hook::set_once();
}
