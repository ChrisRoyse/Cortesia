# Development commands for CortexKG
# Install just: cargo install just

# Default recipe - show available commands
default:
    @just --list

# Run all checks before committing
pre-commit: fmt clippy test check-wasm
    @echo "✅ All checks passed!"

# Format code
fmt:
    cargo fmt --all

# Run clippy lints
clippy:
    cargo clippy --all-features --workspace -- -D warnings

# Run tests
test:
    cargo test --all-features --workspace

# Build the project
build:
    cargo build --all-features --workspace

# Build release version
release:
    cargo build --release --all-features --workspace

# Check WASM build
check-wasm:
    cargo build --target wasm32-unknown-unknown -p neuromorphic-wasm

# Run benchmarks
bench:
    cargo bench --workspace

# Clean build artifacts
clean:
    cargo clean

# Run security audit
audit:
    cargo audit

# Update dependencies
update:
    cargo update

# Generate documentation
docs:
    cargo doc --all-features --workspace --no-deps --open

# Watch for changes and rebuild
watch:
    cargo watch -x check -x test

# Check for outdated dependencies
outdated:
    cargo outdated

# Initialize development environment
init:
    rustup target add wasm32-unknown-unknown
    cargo install cargo-watch cargo-audit cargo-outdated wasm-pack just

# Run development server (when implemented)
dev:
    @echo "Development server not yet implemented"

# Package WASM module
pack-wasm:
    wasm-pack build crates/neuromorphic-wasm --target web --out-dir ../../dist/wasm