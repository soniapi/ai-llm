use crate::attention::scaled_dot_product_attention;
use crate::tensor::Tensor;

pub struct EncoderBlock {
    // Attention Weights
    pub w_q: Tensor,
    pub w_k: Tensor,
    pub w_v: Tensor,
    pub w_o: Tensor, // Output projection

    // FeedForward Network Weights
    pub ffn_w1: Tensor,
    pub ffn_w2: Tensor,
}

impl EncoderBlock {
    pub fn new(dim: usize, ffn_dim: usize) -> Self {
        EncoderBlock {
            w_q: Tensor::random(dim, dim),
            w_k: Tensor::random(dim, dim),
            w_v: Tensor::random(dim, dim),
            w_o: Tensor::random(dim, dim),
            ffn_w1: Tensor::random(dim, ffn_dim),
            ffn_w2: Tensor::random(ffn_dim, dim),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // --- Step 1: Self-Attention ---
        // Project inputs to Queries, Keys, and Values
        let q = x.matmul(&self.w_q);
        let k = x.matmul(&self.w_k);
        let v = x.matmul(&self.w_v);

        // Calculate Attention
        let attention_output = scaled_dot_product_attention(&q, &k, &v);

        // Project the output
        let proj_output = attention_output.matmul(&self.w_o);

        // Residual Connection (add input x back)
        let mut x_residual = x.add(&proj_output); // Note: Skipping LayerNorm for simplicity

        // --- Step 2: FeedForward Network ---
        let hidden = x_residual.matmul(&self.ffn_w1).relu();
        let ffn_output = hidden.matmul(&self.ffn_w2);

        // Second Residual Connection
        x_residual = x_residual.add(&ffn_output);

        x_residual
    }
}
