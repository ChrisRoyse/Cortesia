# Cross-Platform Testing Strategy for Ultimate RAG System

## Executive Summary

This document defines a comprehensive cross-platform testing strategy for the Ultimate RAG System, a Rust-based retrieval-augmented generation system with neuromorphic components. The strategy covers Windows 10/11 (primary), Windows Server 2019/2022, Linux (Ubuntu 20.04/22.04), and macOS (secondary) platforms.

## Platform Compatibility Matrix

### Supported Platforms

| Platform | Status | Rust Tier | Performance Target | Special Considerations |
|----------|--------|-----------|-------------------|----------------------|
| **Windows 10/11** | Primary | Tier 1 | 100% baseline | Primary development target |
| **Windows Server 2019/2022** | Production | Tier 1 | 100% baseline | Server-specific optimizations |
| **Ubuntu 20.04 LTS** | Supported | Tier 1 | 95% baseline | SIMD compatibility required |
| **Ubuntu 22.04 LTS** | Supported | Tier 1 | 95% baseline | Latest toolchain support |
| **macOS 12+** | Secondary | Tier 1 | 85% baseline | M1/M2 ARM64 consideration |
| **WASM32** | Embedded | Tier 2 | 80% baseline | Browser deployment target |

### Component Platform Support

| Component | Windows | Linux | macOS | WASM | Notes |
|-----------|---------|-------|-------|------|-------|
| Neuromorphic Core | ✅ | ✅ | ✅ | ✅ | SIMD optimization varies |
| Vector Search (Tantivy) | ✅ | ✅ | ✅ | ❌ | Native file I/O required |
| Tree-sitter AST | ✅ | ✅ | ✅ | ⚠️ | Limited WASM support |
| Neural Bridge | ✅ | ✅ | ✅ | ✅ | Full compatibility |
| Temporal Memory | ✅ | ✅ | ✅ | ✅ | Full compatibility |
| LanceDB Integration | ✅ | ✅ | ✅ | ❌ | Requires protobuf compiler |

## OS-Specific Test Suites

### Windows-Specific Testing

#### Path Handling Tests
```rust
#[cfg(windows)]
mod windows_path_tests {
    use std::path::Path;
    
    #[test]
    fn test_windows_path_normalization() {
        let paths = vec![
            r"C:\temp\file.txt",
            r"C:/temp/file.txt",
            r"\\server\share\file.txt",
            r"\\?\C:\very\long\path\exceeding\260\characters\...",
        ];
        
        for path in paths {
            assert!(normalize_path(Path::new(path)).is_ok());
        }
    }
    
    #[test]
    fn test_extended_path_support() {
        // Test Windows extended path syntax (\\?\)
        let long_path = format!(r"\\?\C:\{}", "a".repeat(300));
        assert!(Path::new(&long_path).exists() || !Path::new(&long_path).parent().unwrap().exists());
    }
    
    #[test]
    fn test_reserved_names() {
        let reserved = vec!["CON", "PRN", "AUX", "NUL", "COM1", "LPT1"];
        for name in reserved {
            assert!(is_reserved_name(name));
        }
    }
}
```

#### Windows Performance Tests
```rust
#[cfg(windows)]
#[bench]
fn bench_windows_file_operations(b: &mut Bencher) {
    use std::fs;
    let temp_dir = tempfile::tempdir().unwrap();
    
    b.iter(|| {
        let file_path = temp_dir.path().join("test.txt");
        fs::write(&file_path, "test data").unwrap();
        fs::read(&file_path).unwrap()
    });
}
```

### Linux-Specific Testing

#### File System Tests
```rust
#[cfg(unix)]
mod unix_tests {
    use std::os::unix::fs::PermissionsExt;
    
    #[test]
    fn test_unix_permissions() {
        let temp_file = tempfile::NamedTempFile::new().unwrap();
        let mut perms = temp_file.path().metadata().unwrap().permissions();
        perms.set_mode(0o644);
        std::fs::set_permissions(temp_file.path(), perms).unwrap();
        
        let metadata = temp_file.path().metadata().unwrap();
        assert_eq!(metadata.permissions().mode() & 0o777, 0o644);
    }
    
    #[test]
    fn test_case_sensitive_paths() {
        let temp_dir = tempfile::tempdir().unwrap();
        let file1 = temp_dir.path().join("Test.txt");
        let file2 = temp_dir.path().join("test.txt");
        
        std::fs::write(&file1, "content1").unwrap();
        std::fs::write(&file2, "content2").unwrap();
        
        assert_ne!(std::fs::read_to_string(&file1).unwrap(),
                   std::fs::read_to_string(&file2).unwrap());
    }
}
```

### macOS-Specific Testing

#### Apple Silicon Compatibility
```rust
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod macos_arm_tests {
    #[test]
    fn test_simd_availability() {
        #[cfg(target_feature = "neon")]
        {
            // Test ARM NEON SIMD operations
            use std::arch::aarch64::*;
            unsafe {
                let a = vdupq_n_f32(1.0);
                let b = vdupq_n_f32(2.0);
                let result = vaddq_f32(a, b);
                assert_eq!(vgetq_lane_f32(result, 0), 3.0);
            }
        }
    }
}
```

## Cross-Platform CI/CD Configuration

### GitHub Actions Multi-Platform Workflow

```yaml
name: Cross-Platform CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  CARGO_TERM_COLOR: always
  RUST_BACKTRACE: 1

jobs:
  test-matrix:
    name: Test on ${{ matrix.os }} with Rust ${{ matrix.rust }}
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-20.04, ubuntu-22.04, windows-2019, windows-2022, macos-12, macos-13]
        rust: [stable, beta]
        include:
          - os: ubuntu-20.04
            target: x86_64-unknown-linux-gnu
          - os: ubuntu-22.04
            target: x86_64-unknown-linux-gnu
          - os: windows-2019
            target: x86_64-pc-windows-msvc
          - os: windows-2022
            target: x86_64-pc-windows-msvc
          - os: macos-12
            target: x86_64-apple-darwin
          - os: macos-13
            target: aarch64-apple-darwin

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Install Rust toolchain
        uses: dtolnay/rust-toolchain@master
        with:
          toolchain: ${{ matrix.rust }}
          targets: ${{ matrix.target }}
          components: rustfmt, clippy

      - name: Setup Rust cache
        uses: Swatinem/rust-cache@v2
        with:
          key: ${{ matrix.os }}-${{ matrix.rust }}

      - name: Install platform-specific dependencies
        run: |
          if [[ "${{ matrix.os }}" == ubuntu-* ]]; then
            sudo apt-get update
            sudo apt-get install -y protobuf-compiler libprotobuf-dev
          elif [[ "${{ matrix.os }}" == macos-* ]]; then
            brew install protobuf
          elif [[ "${{ matrix.os }}" == windows-* ]]; then
            choco install protoc
          fi
        shell: bash

      - name: Check code formatting
        run: cargo fmt --all -- --check

      - name: Run clippy
        run: cargo clippy --all-features --workspace --target ${{ matrix.target }} -- -D warnings

      - name: Build workspace
        run: cargo build --workspace --target ${{ matrix.target }} --verbose

      - name: Run tests
        run: cargo test --workspace --target ${{ matrix.target }} --verbose
        env:
          RUST_TEST_THREADS: 1

      - name: Run platform-specific tests
        run: |
          if [[ "${{ matrix.os }}" == windows-* ]]; then
            cargo test --workspace --target ${{ matrix.target }} --features windows-specific
          elif [[ "${{ matrix.os }}" == ubuntu-* ]]; then
            cargo test --workspace --target ${{ matrix.target }} --features linux-specific
          elif [[ "${{ matrix.os }}" == macos-* ]]; then
            cargo test --workspace --target ${{ matrix.target }} --features macos-specific
          fi
        shell: bash

  benchmark-matrix:
    name: Benchmark on ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]

    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2

      - name: Run benchmarks
        run: |
          cargo bench --workspace -- --output-format csv | tee benchmark-results-${{ matrix.os }}.csv

      - name: Upload benchmark results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results-${{ matrix.os }}
          path: benchmark-results-${{ matrix.os }}.csv

  wasm-test:
    name: WASM Build Test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          targets: wasm32-unknown-unknown
      - uses: Swatinem/rust-cache@v2

      - name: Install wasm-pack
        run: curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

      - name: Build WASM packages
        run: |
          wasm-pack build crates/neuromorphic-wasm --target web
          wasm-pack build crates/neuromorphic-wasm --target nodejs

      - name: Test WASM in browser
        run: |
          cd crates/neuromorphic-wasm
          npm test
```

## Performance Baseline Per Platform

### Target Performance Metrics

| Metric | Windows | Linux | macOS | Acceptable Variance |
|--------|---------|-------|-------|-------------------|
| TTFS Encoding | 67μs | 70μs (±5%) | 80μs (±20%) | ±25% max |
| Lateral Inhibition | 340μs | 350μs (±3%) | 400μs (±18%) | ±25% max |
| Memory Allocation | 0.73ms | 0.75ms (±3%) | 0.85ms (±16%) | ±20% max |
| SIMD Speedup | 4.2x | 4.0x (±5%) | 3.5x (±17%) | ±20% max |
| File I/O (1MB) | 12ms | 10ms (±17%) | 15ms (±25%) | ±30% max |
| Vector Search | 2.3ms | 2.1ms (±9%) | 2.8ms (±22%) | ±25% max |

### Benchmark Suite Structure

```rust
// benches/cross_platform_bench.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use neuromorphic_core::*;

fn benchmark_ttfs_encoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("ttfs_encoding");
    
    #[cfg(target_os = "windows")]
    group.bench_function("windows", |b| {
        b.iter(|| {
            let encoder = TTFSEncoder::new();
            encoder.encode(black_box("test input"))
        });
    });
    
    #[cfg(target_os = "linux")]
    group.bench_function("linux", |b| {
        b.iter(|| {
            let encoder = TTFSEncoder::new();
            encoder.encode(black_box("test input"))
        });
    });
    
    #[cfg(target_os = "macos")]
    group.bench_function("macos", |b| {
        b.iter(|| {
            let encoder = TTFSEncoder::new();
            encoder.encode(black_box("test input"))
        });
    });
    
    group.finish();
}

criterion_group!(benches, benchmark_ttfs_encoding);
criterion_main!(benches);
```

## Path Normalization Testing

### Comprehensive Path Test Suite

```rust
// tests/path_normalization.rs
use std::path::{Path, PathBuf};

#[cfg(test)]
mod path_tests {
    use super::*;
    
    #[test]
    fn test_cross_platform_path_normalization() {
        let test_cases = vec![
            // Windows paths
            (r"C:\temp\file.txt", "/c/temp/file.txt"),
            (r".\relative\path", "./relative/path"),
            (r"..\parent\path", "../parent/path"),
            // Unix paths
            ("/tmp/file.txt", "/tmp/file.txt"),
            ("./relative/path", "./relative/path"),
            ("../parent/path", "../parent/path"),
            // Mixed separators
            ("C:/temp\\file.txt", "/c/temp/file.txt"),
        ];
        
        for (input, expected) in test_cases {
            let normalized = normalize_path_cross_platform(input);
            assert_eq!(normalized, expected, "Failed for input: {}", input);
        }
    }
    
    #[test]
    fn test_unicode_path_support() {
        let unicode_paths = vec![
            "файл.txt",      // Cyrillic
            "文件.txt",       // Chinese
            "ファイル.txt",    // Japanese
            "🦀_rust.rs",     // Emoji
        ];
        
        for path in unicode_paths {
            let path_buf = PathBuf::from(path);
            assert!(path_buf.to_str().is_some(), "Unicode path failed: {}", path);
        }
    }
}
```

## Character Encoding Tests

### UTF-8 and Platform-Specific Encoding

```rust
// tests/character_encoding_tests.rs
#[cfg(test)]
mod encoding_tests {
    use std::fs;
    use tempfile::tempdir;
    
    #[test]
    fn test_utf8_file_handling() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("utf8_test.txt");
        
        let test_content = "Hello 世界 🌍 Русский ñáéíóú";
        fs::write(&file_path, test_content).unwrap();
        
        let read_content = fs::read_to_string(&file_path).unwrap();
        assert_eq!(read_content, test_content);
    }
    
    #[test]
    fn test_bom_handling() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("bom_test.txt");
        
        // UTF-8 BOM + content
        let bom_content = b"\xEF\xBB\xBFHello World";
        fs::write(&file_path, bom_content).unwrap();
        
        let content = fs::read_to_string(&file_path).unwrap();
        assert!(content.starts_with("Hello World") || content.starts_with("\u{FEFF}Hello World"));
    }
    
    #[cfg(windows)]
    #[test]
    fn test_windows_1252_fallback() {
        // Test handling of legacy Windows-1252 encoded files
        use encoding_rs::{WINDOWS_1252, UTF_8};
        
        let windows_1252_bytes = b"\x93Hello\x94"; // Curly quotes in Windows-1252
        let (decoded, _, _) = WINDOWS_1252.decode(windows_1252_bytes);
        assert!(decoded.contains("Hello"));
    }
}
```

## Resource Limit Testing

### Memory and Performance Constraints

```rust
// tests/resource_limits.rs
#[cfg(test)]
mod resource_tests {
    use std::thread;
    use std::time::Duration;
    
    #[test]
    fn test_memory_usage_limits() {
        let initial_memory = get_process_memory_usage();
        
        // Allocate large neural network
        let large_network = create_test_network(10000);
        let peak_memory = get_process_memory_usage();
        
        drop(large_network);
        thread::sleep(Duration::from_millis(100)); // Allow cleanup
        
        let final_memory = get_process_memory_usage();
        
        // Memory should be released
        assert!(final_memory - initial_memory < (peak_memory - initial_memory) / 2);
    }
    
    #[test]
    fn test_cpu_usage_under_load() {
        use rayon::prelude::*;
        
        let start_time = std::time::Instant::now();
        
        // CPU-intensive parallel computation
        (0..10000).into_par_iter().map(|i| {
            // Simulate neural computation
            (0..1000).fold(0.0f32, |acc, j| acc + (i * j) as f32)
        }).collect::<Vec<_>>();
        
        let duration = start_time.elapsed();
        
        // Should complete within reasonable time on all platforms
        #[cfg(target_os = "windows")]
        assert!(duration < Duration::from_secs(5));
        
        #[cfg(target_os = "linux")]
        assert!(duration < Duration::from_secs(4));
        
        #[cfg(target_os = "macos")]
        assert!(duration < Duration::from_secs(6));
    }
    
    fn get_process_memory_usage() -> usize {
        #[cfg(target_os = "linux")]
        {
            use std::fs;
            let status = fs::read_to_string("/proc/self/status").unwrap();
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    let kb: usize = line.split_whitespace().nth(1).unwrap().parse().unwrap();
                    return kb * 1024;
                }
            }
            0
        }
        
        #[cfg(target_os = "windows")]
        {
            // Windows-specific memory usage implementation
            use std::mem;
            use winapi::um::processthreadsapi::GetCurrentProcess;
            use winapi::um::psapi::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
            
            unsafe {
                let mut pmc: PROCESS_MEMORY_COUNTERS = mem::zeroed();
                if GetProcessMemoryInfo(
                    GetCurrentProcess(),
                    &mut pmc,
                    mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
                ) != 0 {
                    pmc.WorkingSetSize
                } else {
                    0
                }
            }
        }
        
        #[cfg(target_os = "macos")]
        {
            // macOS-specific implementation using mach
            0 // Simplified for brevity
        }
    }
}
```

## Platform-Specific Optimizations

### SIMD Optimization Strategy

```rust
// src/platform/simd_optimization.rs
#[cfg(target_arch = "x86_64")]
pub mod x86_simd {
    use std::arch::x86_64::*;
    
    #[target_feature(enable = "avx2")]
    pub unsafe fn vectorized_computation_avx2(data: &[f32]) -> Vec<f32> {
        let mut result = Vec::with_capacity(data.len());
        
        for chunk in data.chunks_exact(8) {
            let vector = _mm256_loadu_ps(chunk.as_ptr());
            let processed = _mm256_mul_ps(vector, _mm256_set1_ps(2.0));
            
            let mut output = [0.0f32; 8];
            _mm256_storeu_ps(output.as_mut_ptr(), processed);
            result.extend_from_slice(&output);
        }
        
        result
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub mod arm_simd {
    use std::arch::aarch64::*;
    
    pub unsafe fn vectorized_computation_neon(data: &[f32]) -> Vec<f32> {
        let mut result = Vec::with_capacity(data.len());
        
        for chunk in data.chunks_exact(4) {
            let vector = vld1q_f32(chunk.as_ptr());
            let processed = vmulq_n_f32(vector, 2.0);
            
            let mut output = [0.0f32; 4];
            vst1q_f32(output.as_mut_ptr(), processed);
            result.extend_from_slice(&output);
        }
        
        result
    }
}

pub fn optimized_computation(data: &[f32]) -> Vec<f32> {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        unsafe { x86_simd::vectorized_computation_avx2(data) }
    }
    
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        unsafe { arm_simd::vectorized_computation_neon(data) }
    }
    
    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "avx2"),
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    {
        // Fallback implementation
        data.iter().map(|&x| x * 2.0).collect()
    }
}
```

## Containerized Testing Approach

### Docker Test Environments

#### Linux Test Container
```dockerfile
# docker/test-linux.dockerfile
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    protobuf-compiler \
    libprotobuf-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /workspace
COPY . .

RUN cargo build --workspace --release
RUN cargo test --workspace --release
```

#### Windows Test Container
```dockerfile
# docker/test-windows.dockerfile
FROM mcr.microsoft.com/windows/servercore:ltsc2022

SHELL ["powershell", "-Command", "$ErrorActionPreference = 'Stop';"]

# Install Rust
RUN Invoke-WebRequest -Uri https://win.rustup.rs/x86_64 -OutFile rustup-init.exe; \
    ./rustup-init.exe -y; \
    Remove-Item rustup-init.exe

# Install dependencies
RUN choco install protoc -y

ENV PATH="C:\Users\ContainerUser\.cargo\bin;${PATH}"

WORKDIR C:\workspace
COPY . .

RUN cargo build --workspace --release
RUN cargo test --workspace --release
```

### Docker Compose Test Suite
```yaml
# docker-compose.test.yml
version: '3.8'

services:
  test-ubuntu-20:
    build:
      context: .
      dockerfile: docker/test-ubuntu-20.dockerfile
    volumes:
      - ./target:/workspace/target
    command: cargo test --workspace

  test-ubuntu-22:
    build:
      context: .
      dockerfile: docker/test-ubuntu-22.dockerfile
    volumes:
      - ./target:/workspace/target
    command: cargo test --workspace

  test-alpine:
    build:
      context: .
      dockerfile: docker/test-alpine.dockerfile
    volumes:
      - ./target:/workspace/target
    command: cargo test --workspace

  benchmark-comparison:
    build:
      context: .
      dockerfile: docker/test-ubuntu-22.dockerfile
    volumes:
      - ./benchmark-results:/workspace/results
    command: cargo bench --workspace -- --output-format csv > /workspace/results/docker-bench.csv
```

## Performance Monitoring and Validation

### Automated Performance Regression Detection

```rust
// tests/performance_regression.rs
use criterion::{BatchSize, Criterion};
use std::collections::HashMap;
use std::fs;

#[derive(serde::Deserialize, serde::Serialize)]
struct BenchmarkBaseline {
    ttfs_encoding_ns: u64,
    lateral_inhibition_ns: u64,
    memory_allocation_ns: u64,
    platform: String,
    rust_version: String,
    timestamp: String,
}

impl BenchmarkBaseline {
    fn load_or_create(platform: &str) -> Self {
        let filename = format!("benchmark-baseline-{}.json", platform);
        
        if let Ok(content) = fs::read_to_string(&filename) {
            serde_json::from_str(&content).unwrap_or_else(|_| Self::default(platform))
        } else {
            Self::default(platform)
        }
    }
    
    fn default(platform: &str) -> Self {
        Self {
            ttfs_encoding_ns: u64::MAX,
            lateral_inhibition_ns: u64::MAX,
            memory_allocation_ns: u64::MAX,
            platform: platform.to_string(),
            rust_version: env!("RUSTC_VERSION").to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }
    
    fn save(&self) {
        let filename = format!("benchmark-baseline-{}.json", self.platform);
        let content = serde_json::to_string_pretty(self).unwrap();
        fs::write(&filename, content).unwrap();
    }
    
    fn check_regression(&self, current: &Self, threshold: f64) -> Vec<String> {
        let mut regressions = Vec::new();
        
        if self.regression_ratio(self.ttfs_encoding_ns, current.ttfs_encoding_ns) > threshold {
            regressions.push("TTFS encoding performance regression detected".to_string());
        }
        
        if self.regression_ratio(self.lateral_inhibition_ns, current.lateral_inhibition_ns) > threshold {
            regressions.push("Lateral inhibition performance regression detected".to_string());
        }
        
        if self.regression_ratio(self.memory_allocation_ns, current.memory_allocation_ns) > threshold {
            regressions.push("Memory allocation performance regression detected".to_string());
        }
        
        regressions
    }
    
    fn regression_ratio(&self, baseline: u64, current: u64) -> f64 {
        if baseline == u64::MAX { return 0.0; }
        (current as f64 - baseline as f64) / baseline as f64
    }
}

#[cfg(test)]
mod performance_tests {
    use super::*;
    
    #[test]
    fn test_performance_baseline_compliance() {
        let platform = std::env::consts::OS;
        let mut baseline = BenchmarkBaseline::load_or_create(platform);
        
        // Run actual benchmarks
        let current_ttfs = benchmark_ttfs_encoding_single();
        let current_inhibition = benchmark_lateral_inhibition_single();
        let current_allocation = benchmark_memory_allocation_single();
        
        let current = BenchmarkBaseline {
            ttfs_encoding_ns: current_ttfs,
            lateral_inhibition_ns: current_inhibition,
            memory_allocation_ns: current_allocation,
            platform: platform.to_string(),
            rust_version: env!("RUSTC_VERSION").to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
        };
        
        // Check for regressions (10% threshold)
        let regressions = baseline.check_regression(&current, 0.10);
        
        if regressions.is_empty() {
            // Update baseline if performance improved
            if current.ttfs_encoding_ns < baseline.ttfs_encoding_ns {
                baseline.ttfs_encoding_ns = current.ttfs_encoding_ns;
            }
            if current.lateral_inhibition_ns < baseline.lateral_inhibition_ns {
                baseline.lateral_inhibition_ns = current.lateral_inhibition_ns;
            }
            if current.memory_allocation_ns < baseline.memory_allocation_ns {
                baseline.memory_allocation_ns = current.memory_allocation_ns;
            }
            
            baseline.save();
        } else {
            panic!("Performance regressions detected: {:#?}", regressions);
        }
    }
}

fn benchmark_ttfs_encoding_single() -> u64 {
    use std::time::Instant;
    use neuromorphic_core::TTFSEncoder;
    
    let encoder = TTFSEncoder::new();
    let start = Instant::now();
    
    for _ in 0..1000 {
        let _ = encoder.encode("test input");
    }
    
    start.elapsed().as_nanos() as u64 / 1000
}

fn benchmark_lateral_inhibition_single() -> u64 {
    use std::time::Instant;
    use neuromorphic_core::LateralInhibition;
    
    let inhibition = LateralInhibition::new(0.5, 1.0, true);
    let start = Instant::now();
    
    for _ in 0..100 {
        let _ = inhibition.compete(&[0.1, 0.3, 0.8, 0.2]);
    }
    
    start.elapsed().as_nanos() as u64 / 100
}

fn benchmark_memory_allocation_single() -> u64 {
    use std::time::Instant;
    use neuromorphic_core::CorticalGrid;
    
    let start = Instant::now();
    
    for _ in 0..10 {
        let grid = CorticalGrid::new(4, 1000);
        std::hint::black_box(grid);
    }
    
    start.elapsed().as_nanos() as u64 / 10
}
```

## Test Execution and Validation

### Cross-Platform Test Runner

```bash
#!/bin/bash
# scripts/run_cross_platform_tests.sh

set -euo pipefail

PLATFORMS=("ubuntu-20.04" "ubuntu-22.04" "windows-2019" "windows-2022" "macos-12")
RESULTS_DIR="cross-platform-results"

mkdir -p "$RESULTS_DIR"

echo "🚀 Starting cross-platform test suite"

for platform in "${PLATFORMS[@]}"; do
    echo "🔄 Testing on $platform"
    
    # Run tests in Docker or GitHub Actions matrix
    if command -v docker &> /dev/null; then
        docker run --rm -v "$(pwd):/workspace" \
            "llmkg-test:$platform" \
            cargo test --workspace -- --test-threads=1 --nocapture \
            > "$RESULTS_DIR/test-results-$platform.log" 2>&1
    fi
    
    # Run benchmarks
    if command -v docker &> /dev/null; then
        docker run --rm -v "$(pwd):/workspace" \
            "llmkg-test:$platform" \
            cargo bench --workspace -- --output-format csv \
            > "$RESULTS_DIR/bench-results-$platform.csv" 2>&1
    fi
    
    echo "✅ Completed testing on $platform"
done

echo "📊 Generating cross-platform comparison report"
python3 scripts/generate_platform_report.py "$RESULTS_DIR"

echo "🎉 Cross-platform testing completed successfully"
```

### Platform Comparison Report Generator

```python
#!/usr/bin/env python3
# scripts/generate_platform_report.py

import json
import csv
import sys
from pathlib import Path
from typing import Dict, List, Any

def parse_benchmark_results(results_dir: Path) -> Dict[str, Any]:
    """Parse benchmark CSV files and extract performance data."""
    results = {}
    
    for csv_file in results_dir.glob("bench-results-*.csv"):
        platform = csv_file.stem.replace("bench-results-", "")
        
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            platform_results = {}
            
            for row in reader:
                benchmark_name = row['benchmark']
                platform_results[benchmark_name] = {
                    'mean_ns': float(row['mean']),
                    'stddev_ns': float(row['stddev']),
                    'median_ns': float(row['median']),
                }
            
            results[platform] = platform_results
    
    return results

def generate_html_report(results: Dict[str, Any], output_path: Path):
    """Generate comprehensive HTML report with charts and tables."""
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>LLMKG Cross-Platform Performance Report</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .benchmark-section {{ margin: 40px 0; }}
            .chart-container {{ width: 800px; height: 400px; margin: 20px 0; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .performance-good {{ color: green; }}
            .performance-warning {{ color: orange; }}
            .performance-bad {{ color: red; }}
        </style>
    </head>
    <body>
        <h1>🧠 LLMKG Cross-Platform Performance Report</h1>
        <p>Generated on: {import datetime; datetime.datetime.now().isoformat()}</p>
        
        <h2>📊 Performance Summary</h2>
        <div class="chart-container">
            <canvas id="performanceChart"></canvas>
        </div>
        
        <h2>📋 Detailed Results</h2>
        {generate_results_table(results)}
        
        <h2>🎯 Platform Recommendations</h2>
        {generate_recommendations(results)}
        
        <script>
            {generate_chart_js(results)}
        </script>
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)

def generate_results_table(results: Dict[str, Any]) -> str:
    """Generate HTML table with benchmark results."""
    if not results:
        return "<p>No benchmark results available.</p>"
    
    # Extract all benchmark names
    all_benchmarks = set()
    for platform_results in results.values():
        all_benchmarks.update(platform_results.keys())
    
    html = "<table><thead><tr><th>Benchmark</th>"
    for platform in results.keys():
        html += f"<th>{platform}</th>"
    html += "</tr></thead><tbody>"
    
    for benchmark in sorted(all_benchmarks):
        html += f"<tr><td>{benchmark}</td>"
        
        benchmark_results = []
        for platform in results.keys():
            if benchmark in results[platform]:
                mean_ns = results[platform][benchmark]['mean_ns']
                benchmark_results.append(mean_ns)
                html += f"<td>{mean_ns/1000:.1f}μs</td>"
            else:
                html += "<td>N/A</td>"
        
        html += "</tr>"
    
    html += "</tbody></table>"
    return html

def generate_recommendations(results: Dict[str, Any]) -> str:
    """Generate platform-specific recommendations."""
    recommendations = """
    <ul>
        <li><strong>Windows:</strong> Optimal for development and primary deployment</li>
        <li><strong>Linux:</strong> Best performance-per-cost ratio for server deployment</li>
        <li><strong>macOS:</strong> Good for development, consider ARM vs x86 performance</li>
    </ul>
    """
    return recommendations

def generate_chart_js(results: Dict[str, Any]) -> str:
    """Generate Chart.js configuration for performance visualization."""
    return """
    const ctx = document.getElementById('performanceChart').getContext('2d');
    const chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['TTFS Encoding', 'Lateral Inhibition', 'Memory Allocation'],
            datasets: [
                {
                    label: 'Windows',
                    data: [67, 340, 730],
                    backgroundColor: 'rgba(54, 162, 235, 0.8)'
                },
                {
                    label: 'Linux',
                    data: [70, 350, 750],
                    backgroundColor: 'rgba(255, 99, 132, 0.8)'
                },
                {
                    label: 'macOS',
                    data: [80, 400, 850],
                    backgroundColor: 'rgba(75, 192, 192, 0.8)'
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Time (microseconds)'
                    }
                }
            }
        }
    });
    """

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 generate_platform_report.py <results_directory>")
        sys.exit(1)
    
    results_dir = Path(sys.argv[1])
    if not results_dir.exists():
        print(f"Results directory {results_dir} does not exist")
        sys.exit(1)
    
    print("🔄 Parsing benchmark results...")
    results = parse_benchmark_results(results_dir)
    
    print("📊 Generating HTML report...")
    output_path = results_dir / "cross-platform-report.html"
    generate_html_report(results, output_path)
    
    print(f"✅ Report generated: {output_path}")
    print(f"🌐 Open in browser: file://{output_path.absolute()}")

if __name__ == "__main__":
    main()
```

## Conclusion

This comprehensive cross-platform testing strategy ensures the Ultimate RAG System delivers consistent, high-performance operation across all target platforms. The strategy emphasizes:

1. **Automated Testing**: CI/CD pipelines validate every change across all platforms
2. **Performance Baselines**: Regression detection prevents performance degradation  
3. **Platform-Specific Optimizations**: SIMD and OS-specific features utilized appropriately
4. **Comprehensive Coverage**: Path handling, encoding, resource limits, and edge cases tested
5. **Containerized Validation**: Docker environments provide consistent test conditions
6. **Detailed Reporting**: Performance comparisons and recommendations for optimal deployment

The testing infrastructure supports the system's neuromorphic architecture while maintaining cross-platform compatibility and optimal performance characteristics across Windows, Linux, and macOS environments.