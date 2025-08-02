# Micro-Phase 9.03: WASM Build Configuration

## Objective
Setup optimized build configuration for WASM module including size optimization, performance tuning, and automated build pipeline.

## Prerequisites
- Completed micro-phase 9.01 (project setup)
- Completed micro-phase 9.02 (dependencies configured)

## Task Description
Configure build system for optimal WASM output with focus on small bundle size and high performance.

## Specific Actions

1. **Update Cargo.toml with optimized release profile**:
   ```toml
   [profile.release]
   opt-level = "s"          # Optimize for size
   lto = true               # Enable Link Time Optimization
   codegen-units = 1        # Single codegen unit for better optimization
   panic = "abort"          # Smaller binary, no unwinding
   strip = true             # Strip symbols
   
   [profile.wasm]
   inherits = "release"
   opt-level = "z"          # Aggressive size optimization
   ```

2. **Create .cargo/config.toml**:
   ```toml
   [build]
   target = "wasm32-unknown-unknown"
   
   [target.wasm32-unknown-unknown]
   rustflags = [
       "-C", "link-arg=--no-entry",
       "-C", "link-arg=--export-dynamic",
       "-C", "target-feature=+simd128",
   ]
   ```

3. **Create wasm-pack.toml configuration**:
   ```toml
   [package]
   name = "cortexkg-wasm"
   
   [build]
   target = "web"
   out-dir = "./pkg"
   
   [profile.release]
   wasm-opt = ["-Os", "--enable-simd"]
   ```

4. **Create optimized build script**:
   ```bash
   #!/bin/bash
   # build.sh
   
   set -e
   
   echo "Building CortexKG WASM module..."
   
   # Clean previous builds
   rm -rf pkg
   
   # Build with wasm-pack
   wasm-pack build \
       --target web \
       --out-dir pkg \
       --release \
       --no-typescript
   
   # Additional optimization with wasm-opt
   if command -v wasm-opt &> /dev/null; then
       echo "Running wasm-opt..."
       wasm-opt -Os \
           --enable-simd \
           pkg/cortexkg_wasm_bg.wasm \
           -o pkg/cortexkg_wasm_bg_opt.wasm
       mv pkg/cortexkg_wasm_bg_opt.wasm pkg/cortexkg_wasm_bg.wasm
   fi
   
   # Report size
   echo "Build complete. WASM size:"
   ls -lh pkg/*.wasm | awk '{print $5, $9}'
   ```

5. **Create package.json for npm integration**:
   ```json
   {
     "name": "cortexkg-wasm-wrapper",
     "version": "0.1.0",
     "description": "WASM wrapper for CortexKG",
     "main": "pkg/cortexkg_wasm.js",
     "scripts": {
       "build": "./build.sh",
       "test": "wasm-pack test --headless --firefox",
       "size": "ls -lh pkg/*.wasm"
     },
     "devDependencies": {
       "webpack": "^5.90.0",
       "webpack-cli": "^5.1.0"
     }
   }
   ```

6. **Create webpack.config.js for bundling**:
   ```javascript
   const path = require('path');
   
   module.exports = {
     entry: './pkg/cortexkg_wasm.js',
     output: {
       path: path.resolve(__dirname, 'dist'),
       filename: 'cortexkg.bundle.js',
       library: 'CortexKG',
       libraryTarget: 'umd'
     },
     mode: 'production',
     experiments: {
       asyncWebAssembly: true
     },
     optimization: {
       minimize: true
     }
   };
   ```

## Expected Outputs
- Optimized Cargo build profiles
- WASM-specific build configuration
- Automated build script with optimization
- Size-optimized WASM output (<2MB target)
- Webpack configuration for JavaScript bundling

## Validation
1. Run `./build.sh` - build completes successfully
2. Check WASM file size: `ls -lh pkg/*.wasm`
3. Verify SIMD support: `wasm2wat pkg/*.wasm | grep -i simd`
4. Bundle size is under 2MB

## Next Steps
- Create core WASM bindings (micro-phase 9.04)
- Define memory-efficient data structures (micro-phase 9.05)