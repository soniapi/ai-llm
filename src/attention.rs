use crate::tensor::Tensor;

pub fn scaled_dot_product_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Tensor {
    // 1. MatMul: Q * K.T
    // Q is [SeqLen x Dim], K is [SeqLen x Dim], so K.T is [Dim x SeqLen]
    // The result raw_scores is [SeqLen x SeqLen]
    let raw_scores = q.matmul(&k.transpose());

    // 2. Scale by sqrt(Dim)
    let d_k = k.cols as f32;
    let scaled_scores = raw_scores.divide_by_scalar(d_k.sqrt());

    // 3. Softmax to get attention probabilities
    let attention_weights = scaled_scores.softmax();

    // 4. MatMul: Weights * V
    // weights is [SeqLen x SeqLen], V is [SeqLen x Dim]
    // The result final_context is [SeqLen x Dim]
    let final_context = attention_weights.matmul(v);

    final_context
}
