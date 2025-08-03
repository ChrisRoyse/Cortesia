# CortexKG - Neuromorphic Knowledge Graph System

[![Rust CI](https://github.com/ChrisRoyse/LLMKG/actions/workflows/rust.yml/badge.svg)](https://github.com/ChrisRoyse/LLMKG/actions/workflows/rust.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust Version](https://img.shields.io/badge/rust-1.75+-orange.svg)](https://www.rust-lang.org)

## Overview

CortexKG is a cutting-edge neuromorphic knowledge graph system that leverages spiking neural networks (SNNs) with Time-to-First-Spike (TTFS) encoding for efficient knowledge representation and reasoning. The system implements biologically-inspired cortical columns with lateral inhibition for optimal resource allocation.

## Features

- **Spiking Neural Networks**: Biologically-inspired neural computation with temporal dynamics
- **TTFS Encoding**: Efficient information encoding using spike timing
- **Lateral Inhibition**: Competition-based resource allocation between cortical columns
- **Memory Versioning**: Temporal memory branches with consolidation support
- **WASM Support**: Browser-compatible with SIMD optimization
- **Cross-Database Bridge**: Pattern emergence across knowledge bases

## Architecture

```
CortexKG/
├── crates/
│   ├── neuromorphic-core/      # Core SNN structures and TTFS implementation
│   ├── snn-allocation-engine/  # Spike-based allocation with lateral inhibition
│   ├── temporal-memory/        # Memory versioning and consolidation
│   ├── neural-bridge/          # Cross-database pattern detection
│   ├── neuromorphic-wasm/      # WASM bindings with SIMD
│   └── snn-mocks/             # Testing infrastructure
└── docs/                       # Documentation and specifications
```

## Quick Start

### Prerequisites

- Rust 1.75 or later
- `just` command runner (optional): `cargo install just`

### Installation

```bash
# Clone the repository
git clone https://github.com/ChrisRoyse/LLMKG.git
cd LLMKG

# Build the project
cargo build --workspace

# Run tests
cargo test --workspace

# Build for WASM
cargo build --target wasm32-unknown-unknown -p neuromorphic-wasm
```

### Development Commands

If you have `just` installed:

```bash
just init      # Initialize development environment
just test      # Run all tests
just fmt       # Format code
just clippy    # Run lints
just docs      # Generate documentation
just pre-commit # Run all checks before committing
```

## Project Status

**Current Phase**: 0.1 - Foundation ✅

- [x] Workspace structure established
- [x] All 6 core crates created
- [x] Build system configured
- [x] CI/CD pipeline ready
- [x] Error handling framework
- [ ] Phase 1: Neural implementation (Next)

## Implementation Notes

See [IMPLEMENTATION_NOTES.md](docs/IMPLEMENTATION_NOTES.md) for detailed implementation decisions and deviations from the original specification.

## Performance

The system is optimized for:
- **Native**: Platform-specific memory allocators (jemalloc/mimalloc)
- **WASM**: Minimal binary size with wee_alloc
- **SIMD**: Hardware acceleration for neural computations

## Contributing

Contributions are welcome! Please ensure:

1. All tests pass: `cargo test --workspace`
2. Code is formatted: `cargo fmt --all`
3. No clippy warnings: `cargo clippy --all-features`
4. Documentation is updated

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by biological cortical column architecture
- Built with Rust's safety and performance guarantees
- Optimized for both native and web deployment

## Contact

- GitHub Issues: [https://github.com/ChrisRoyse/LLMKG/issues](https://github.com/ChrisRoyse/LLMKG/issues)
- Project Lead: Chris Royse

---

*CortexKG - Bringing biological intelligence to knowledge graphs*