# Integration Test Docker Environment
FROM rust:1.75-bullseye as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    llvm \
    clang \
    libclang-dev \
    nodejs \
    npm \
    python3 \
    python3-pip \
    valgrind \
    && rm -rf /var/lib/apt/lists/*

# Install Rust WASM target and tools
RUN rustup target add wasm32-unknown-unknown
RUN cargo install wasm-pack wasm-bindgen-cli

# Install Node.js testing tools
RUN npm install -g @web/test-runner chromedriver

# Install Python analysis tools
RUN pip3 install pytest matplotlib pandas numpy

# Set up workspace
WORKDIR /workspace
COPY . .

# Build the project
RUN cargo build --release --all-features

# Build WASM version
RUN wasm-pack build --target web --out-dir pkg

# Final test runner image
FROM rust:1.75-bullseye

# Copy system dependencies
RUN apt-get update && apt-get install -y \
    libssl1.1 \
    ca-certificates \
    nodejs \
    npm \
    python3 \
    python3-pip \
    valgrind \
    htop \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Copy built artifacts
COPY --from=builder /workspace/target /workspace/target
COPY --from=builder /workspace/pkg /workspace/pkg
COPY --from=builder /workspace/tests /workspace/tests
COPY --from=builder /workspace/Cargo.toml /workspace/Cargo.toml
COPY --from=builder /workspace/Cargo.lock /workspace/Cargo.lock

# Copy test scripts
COPY scripts/integration-test-runner.sh /usr/local/bin/
COPY scripts/generate-test-report.py /usr/local/bin/
RUN chmod +x /usr/local/bin/integration-test-runner.sh

WORKDIR /workspace

# Environment variables
ENV RUST_BACKTRACE=full
ENV RUST_LOG=info
ENV LLMKG_TEST_ENV=docker

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD cargo test --test health_check || exit 1

# Default command
CMD ["/usr/local/bin/integration-test-runner.sh"]

# Multi-stage build for different scenarios
FROM rust:1.75-bullseye as federation-test

# Install additional tools for federation testing
RUN apt-get update && apt-get install -y \
    docker.io \
    docker-compose \
    jq \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY tests/integration/federation/ /federation-tests/
COPY scripts/federation-test-runner.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/federation-test-runner.sh

CMD ["/usr/local/bin/federation-test-runner.sh"]

# Performance testing image
FROM rust:1.75-bullseye as performance-test

# Install performance monitoring tools
RUN apt-get update && apt-get install -y \
    perf-tools-unstable \
    linux-tools-generic \
    sysstat \
    iotop \
    && rm -rf /var/lib/apt/lists/*

COPY tests/integration/performance/ /performance-tests/
COPY scripts/performance-test-runner.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/performance-test-runner.sh

CMD ["/usr/local/bin/performance-test-runner.sh"]