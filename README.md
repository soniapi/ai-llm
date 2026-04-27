# AI-LLM

**Dynamic Hypothesis Generator**

This repository contains the `ai_llm` project, a pure Rust implementation of an LLM built from scratch. It is designed specifically for dynamic hypothesis generation based on database schemas and statistics.

Unlike typical LLM implementations, this project relies on a custom, "from scratch" Transformer architecture and does not use external machine learning libraries (like PyTorch or TensorFlow) or external LLM APIs (like OpenAI).

## Architecture

The `ai_llm` project is an HTTP web service deployed on Google Cloud Run.

1.  **Frontend (HTTP API):** Uses `axum` to serve an HTTP API. The core public endpoint is a `GET /generate`.
2.  **Backend Integration (gRPC):** Communicates with the `ai-infra` backend via gRPC (using `tonic` and `prost`) to fetch database schema context dynamically.
3.  **Model:** A custom-built Transformer model (defined in `src/lib.rs` and `src/transformer.rs`) that takes the schema context as a prompt and generates a hypothesis.

## Prerequisites

To build and run this project locally, you will need:

*   **Rust:** Version 1.88 or newer (to avoid dependency conflicts).
*   **System Dependencies:** `protobuf-compiler` and `libpq-dev`.

On Debian/Ubuntu:
```bash
sudo apt-get update
sudo apt-get install protobuf-compiler libpq-dev
```

## Running the Service

You can build and run the service locally using standard Cargo commands:

```bash
cargo build
cargo run
```

By default, the server listens on `http://localhost:8080` (or the port defined by the `PORT` environment variable).

## Testing

Integration tests verify the HTTP routing and stub the external `ai-infra` gRPC interactions:

```bash
cargo test
```

## Usage Example

A Rust client example is provided in the `examples/` directory to demonstrate how to call the public `/generate` endpoint.

To run the example against the live Google Cloud Run environment (`https://ai-llm-5u7ahgmduq-uc.a.run.app`):

```bash
cargo run --example client
```

To run the example against a local server:

```bash
cargo run --example client http://localhost:8080
```

## Deployment

The service is configured to be deployed to Google Cloud Run. It utilizes a multi-stage Dockerfile based on a Debian-slim image. The deployment process is handled via GitHub Actions.

## License

This project is dual-licensed under the Apache License 2.0 (`LICENSE-ALv2.md`) and the MIT License (`LICENSE-MIT.md`).
