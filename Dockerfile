# Build stage
FROM rust:1.88-slim-bookworm AS builder

# Install required system dependencies for tonic-build (protobuf-compiler) and postgres (libpq-dev)
RUN apt-get update && apt-get install -y protobuf-compiler libpq-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN cargo build --release

# Runtime stage
FROM debian:bookworm-slim

# Install CA certificates to enable TLS connections to GCP/gRPC services
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/target/release/ai_llm /app/ai_llm

# Cloud Run sets the PORT env variable; we default to 8080 if not set
ENV PORT=8080

EXPOSE 8080

CMD ["/app/ai_llm"]
