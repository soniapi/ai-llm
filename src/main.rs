mod attention;
mod model;
mod tensor;
mod tokenizer;

use model::EncoderBlock;
use tokenizer::{GrpcMockPayload, NumericalTokenizer};

fn main() {
    println!("--- Pure Rust LLM Proof of Concept: Dynamic Hypothesis Generation ---");

    // 1. Define network architecture (Small for PoC)
    let embedding_dim = 128;
    let ffn_dim = 512;

    // 2. Initialize the Tokenizer and the Encoder Block (with random weights)
    let tokenizer = NumericalTokenizer::new(embedding_dim);
    let encoder = EncoderBlock::new(embedding_dim, ffn_dim);

    // 3. Receive Mock gRPC payload from ai-infra
    let grpc_payloads = vec![
        GrpcMockPayload {
            column_name: "s".to_string(),
            average: 145000.0,
        },
        GrpcMockPayload {
            column_name: "c".to_string(),
            average: 0.5,
        },
        GrpcMockPayload {
            column_name: "p".to_string(),
            average: 100.0,
        },
    ];
    println!("\nReceived Database Context from ai-infra:");
    for payload in &grpc_payloads {
        println!(
            " - Column: {}, Average: {}",
            payload.column_name, payload.average
        );
    }

    // 4. Tokenize and embed the data into the Rust Matrix Backend (Tensor)
    let embedded_input = tokenizer.encode(&grpc_payloads);
    println!("\nInitial Embedded Input: {:?}", embedded_input);

    // 5. Pass the data through the Encoder (Bidirectional Self-Attention + FFN)
    let contextualized_output = encoder.forward(&embedded_input);

    // 6. Output the final state
    println!("\nForward Pass Complete!");
    println!("The input tensor has been mathematically transformed.");
    println!("Final Contextualized Output: {:?}", contextualized_output);
    println!(
        "(The Decoder would now use this tensor to generate a hypothesis like: 'Test population where s > 145000')"
    );
}
