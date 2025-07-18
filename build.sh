#!/bin/bash
# LLMKG Build Script
# Builds the knowledge graph library for multiple targets

set -e

echo "🚀 Building LLMKG - LLM Knowledge Graph"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Clean previous builds
print_status "Cleaning previous builds..."
cargo clean
rm -rf pkg/ dist/

# Create output directories
mkdir -p dist/{wasm,native,docs}

# Check if wasm-pack is installed
if ! command -v wasm-pack &> /dev/null; then
    print_warning "wasm-pack not found. Installing..."
    curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
fi

# Build WASM version
print_status "Building WebAssembly version..."
wasm-pack build --target web --out-dir pkg/web --features "wasm" --release
if [ $? -eq 0 ]; then
    print_success "WASM build completed"
    cp -r pkg/web/* dist/wasm/
else
    print_error "WASM build failed"
    exit 1
fi

# Build Node.js version
print_status "Building Node.js version..."
wasm-pack build --target nodejs --out-dir pkg/node --features "wasm" --release
if [ $? -eq 0 ]; then
    print_success "Node.js build completed"
    cp -r pkg/node/* dist/wasm/node/
else
    print_error "Node.js build failed"
    exit 1
fi

# Build native library
print_status "Building native library..."
cargo build --release --features "native"
if [ $? -eq 0 ]; then
    print_success "Native build completed"
    cp target/release/libllmkg.* dist/native/ 2>/dev/null || true
else
    print_error "Native build failed"
    exit 1
fi

# Optimize WASM
if command -v wasm-opt &> /dev/null; then
    print_status "Optimizing WASM..."
    wasm-opt -Os -o dist/wasm/llmkg_bg.wasm dist/wasm/llmkg_bg.wasm
    print_success "WASM optimization completed"
else
    print_warning "wasm-opt not found. Skipping WASM optimization."
fi

# Generate documentation
print_status "Generating documentation..."
cargo doc --no-deps --features "native"
if [ $? -eq 0 ]; then
    cp -r target/doc/* dist/docs/
    print_success "Documentation generated"
else
    print_warning "Documentation generation failed"
fi

# Run tests
print_status "Running tests..."
cargo test --features "native"
if [ $? -eq 0 ]; then
    print_success "All tests passed"
else
    print_warning "Some tests failed"
fi

# Run benchmarks
print_status "Running benchmarks..."
cargo bench --features "native" || print_warning "Benchmark execution failed"

# Package size analysis
print_status "Analyzing package sizes..."
echo "📊 Package Size Analysis:"
echo "Native library: $(du -h dist/native/* 2>/dev/null | tail -1 | cut -f1 || echo 'N/A')"
echo "WASM module: $(du -h dist/wasm/*.wasm 2>/dev/null | cut -f1 || echo 'N/A')"
echo "Total dist size: $(du -sh dist/ | cut -f1)"

# Create deployment package
print_status "Creating deployment package..."
cd dist
tar -czf llmkg-release.tar.gz *
cd ..

# Create npm package info for WASM
cat > dist/wasm/package.json << EOF
{
  "name": "llmkg",
  "version": "0.1.0",
  "description": "Lightning-fast knowledge graph optimized for LLM integration",
  "main": "llmkg.js",
  "types": "llmkg.d.ts",
  "files": [
    "llmkg.js",
    "llmkg.d.ts", 
    "llmkg_bg.wasm"
  ],
  "keywords": ["llm", "knowledge-graph", "wasm", "embeddings", "rag"],
  "author": "LLMKG Team",
  "license": "MIT",
  "repository": {
    "type": "git",
    "url": "https://github.com/llmkg/llmkg"
  }
}
EOF

print_success "Build completed successfully! 🎉"
echo ""
echo "📦 Build Artifacts:"
echo "  • WASM package: dist/wasm/"
echo "  • Native library: dist/native/"
echo "  • Documentation: dist/docs/"
echo "  • Release package: dist/llmkg-release.tar.gz"
echo ""
echo "🚀 Quick Start:"
echo "  • Web: Import dist/wasm/llmkg.js"
echo "  • Node.js: npm install dist/wasm/"
echo "  • Native: Link against dist/native/libllmkg.*"
echo ""
echo "📖 View documentation: open dist/docs/llmkg/index.html"