# ai-llm

**Dynamic Hypothesis Generator**

This repository contains the `ai_llm` project, a Rust implementation of an LLM. It is designed specifically for dynamic hypothesis generation based on database schemas and statistics.

## Architecture

The `ai_llm` project is a standalone modulith and a microservice deployed on Google Cloud Run.

1. **Training:** The `ai-llm-training` module (`ai-llm-training/src/lib.rs`) contains the self-supervised training data pipeline loop. It calls ai_infra::establish_connection(), and runs direct SQL queries to return raw PostgreSQL data that stream directly into Rust Vec memory structures.
2. **Model:** A custom-built Transformer model (defined in `ai-llm-inference/src/lib.rs` and `ai-llm-inference/src/transformer.rs`) that takes the schema context as a prompt and generates a hypothesis.
3. **Backend Integration (gRPC):** Communicates with the `ai-infra` backend via gRPC (using `tonic` and `prost`) to fetch database schema context dynamically. Note: gRPC is exclusively used for fetching schema metadata and statistics for inference prompts, and is *not* used for streaming the large volume of training table data (which is handled via direct SQL queries in the training module).
4. **Frontend (REST API):** Uses `axum` to serve a REST API. The core public endpoint is a `GET /generate`.


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
cargo run -p ai-llm-inference
```

By default, the server listens on `http://localhost:8080` (or the port defined by the `PORT` environment variable).

## Testing

Integration tests verify the REST API routing and stub the external `ai-infra` gRPC interactions:

```bash
cargo test
```

## Usage Example

A Rust client example is provided in the `ai-llm-inference/examples/` directory to demonstrate how to call the public `/generate` endpoint.

To run the example against the live Google Cloud Run environment (`https://ai-llm-5u7ahgmduq-uc.a.run.app`):

```bash
cargo run -p ai-llm-inference --example client
```

To run the example against a local server:

```bash
cargo run -p ai-llm-inference --example client http://localhost:8080
```

## Deployment

The service is configured to be deployed to Google Cloud Run. It utilizes a multi-stage Dockerfile based on a Debian-slim image. The deployment process is handled via GitHub Actions.

## License

This project is dual-licensed under the Apache License 2.0 (`LICENSE-ALv2.md`) and the MIT License (`LICENSE-MIT.md`).
