use crate::tensor::Tensor;

// This simulates the gRPC data arriving from ai-infra
pub struct GrpcMockPayload {
    pub column_name: String,
    pub average: f32,
}

// A highly simplified Mock Tokenizer that embeds the input into a tensor
pub struct NumericalTokenizer {
    pub embedding_dim: usize,
}

impl NumericalTokenizer {
    pub fn new(embedding_dim: usize) -> Self {
        NumericalTokenizer { embedding_dim }
    }

    pub fn encode(&self, payloads: &[GrpcMockPayload]) -> Tensor {
        let seq_len = payloads.len();
        // Since we don't have a real learned embedding table in this PoC,
        // we'll simulate the embedding process by generating a random tensor
        // representing the "tokenized context" of these columns.

        println!(
            "Tokenizer converting {} gRPC elements into a {}x{} tensor...",
            seq_len, seq_len, self.embedding_dim
        );

        Tensor::random(seq_len, self.embedding_dim)
    }
}
