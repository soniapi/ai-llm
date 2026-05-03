use crate::tensor::Tensor;

pub struct Linear {
    pub weight: Tensor,
    pub bias: Tensor,
    pub weight_grad: Tensor,
    pub bias_grad: Tensor,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        // Random initialization would go here, we just use 0.01 for dummy initialization
        let weight = Tensor::new(vec![0.01; in_features * out_features], vec![in_features, out_features]);
        let bias = Tensor::zeros(vec![out_features]);
        let weight_grad = Tensor::zeros(vec![in_features, out_features]);
        let bias_grad = Tensor::zeros(vec![out_features]);
        Self { weight, bias, weight_grad, bias_grad }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.matmul(&self.weight).broadcast_add(&self.bias)
    }

    pub fn zero_grad(&mut self) {
        for val in self.weight_grad.data.iter_mut() { *val = 0.0; }
        for val in self.bias_grad.data.iter_mut() { *val = 0.0; }
    }

    pub fn step(&mut self, lr: f32) {
        for (w, g) in self.weight.data.iter_mut().zip(self.weight_grad.data.iter()) {
            *w -= lr * g;
        }
        for (b, g) in self.bias.data.iter_mut().zip(self.bias_grad.data.iter()) {
            *b -= lr * g;
        }
    }
}

pub struct LayerNorm {
    pub weight: Tensor,
    pub bias: Tensor,
    pub weight_grad: Tensor,
    pub bias_grad: Tensor,
    pub eps: f32,
}

impl LayerNorm {
    pub fn new(features: usize) -> Self {
        Self {
            weight: Tensor::new(vec![1.0; features], vec![features]),
            bias: Tensor::zeros(vec![features]),
            weight_grad: Tensor::zeros(vec![features]),
            bias_grad: Tensor::zeros(vec![features]),
            eps: 1e-5,
        }
    }

    pub fn zero_grad(&mut self) {
        for val in self.weight_grad.data.iter_mut() { *val = 0.0; }
        for val in self.bias_grad.data.iter_mut() { *val = 0.0; }
    }

    pub fn step(&mut self, lr: f32) {
        for (w, g) in self.weight.data.iter_mut().zip(self.weight_grad.data.iter()) {
            *w -= lr * g;
        }
        for (b, g) in self.bias.data.iter_mut().zip(self.bias_grad.data.iter()) {
            *b -= lr * g;
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // A very simplified LayerNorm
        let mut result_data = x.data.clone();
        let last_dim = *x.shape.last().unwrap();
        let num_rows = x.data.len() / last_dim;

        for i in 0..num_rows {
            let start = i * last_dim;
            let end = start + last_dim;

            let mut sum = 0.0;
            for j in start..end {
                sum += x.data[j];
            }
            let mean = sum / last_dim as f32;

            let mut var_sum = 0.0;
            for j in start..end {
                let diff = x.data[j] - mean;
                var_sum += diff * diff;
            }
            let var = var_sum / last_dim as f32;
            let std = (var + self.eps).sqrt();

            for j in start..end {
                let norm_val = (x.data[j] - mean) / std;
                result_data[j] = norm_val * self.weight.data[j - start] + self.bias.data[j - start];
            }
        }

        Tensor::new(result_data, x.shape.clone())
    }
}

pub struct SelfAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
    pub d_k: f32,
}

impl SelfAttention {
    pub fn new(d_model: usize) -> Self {
        Self {
            q_proj: Linear::new(d_model, d_model),
            k_proj: Linear::new(d_model, d_model),
            v_proj: Linear::new(d_model, d_model),
            out_proj: Linear::new(d_model, d_model),
            d_k: (d_model as f32).sqrt(),
        }
    }

    pub fn zero_grad(&mut self) {
        self.q_proj.zero_grad();
        self.k_proj.zero_grad();
        self.v_proj.zero_grad();
        self.out_proj.zero_grad();
    }

    pub fn step(&mut self, lr: f32) {
        self.q_proj.step(lr);
        self.k_proj.step(lr);
        self.v_proj.step(lr);
        self.out_proj.step(lr);
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Simplified Single-head Attention for dummy LLM
        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        // Q * K^T (Assuming 2D x where shape is [seq_len, d_model])
        let seq_len = q.shape[0];
        let d_model = q.shape[1];

        let mut scores = Tensor::zeros(vec![seq_len, seq_len]);
        for i in 0..seq_len {
            for j in 0..seq_len {
                let mut sum = 0.0;
                for d in 0..d_model {
                    sum += q.data[i * d_model + d] * k.data[j * d_model + d];
                }
                scores.data[i * seq_len + j] = sum / self.d_k;
            }
        }

        let attn_weights = scores.softmax();
        let attn_output = attn_weights.matmul(&v);
        self.out_proj.forward(&attn_output)
    }
}

pub struct FeedForward {
    pub linear1: Linear,
    pub linear2: Linear,
}

impl FeedForward {
    pub fn new(d_model: usize, d_ff: usize) -> Self {
        Self {
            linear1: Linear::new(d_model, d_ff),
            linear2: Linear::new(d_ff, d_model),
        }
    }

    pub fn zero_grad(&mut self) {
        self.linear1.zero_grad();
        self.linear2.zero_grad();
    }

    pub fn step(&mut self, lr: f32) {
        self.linear1.step(lr);
        self.linear2.step(lr);
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut h = self.linear1.forward(x);
        // ReLU activation
        for val in h.data.iter_mut() {
            *val = val.max(0.0);
        }
        self.linear2.forward(&h)
    }
}

pub struct TransformerBlock {
    pub attention: SelfAttention,
    pub norm1: LayerNorm,
    pub ff: FeedForward,
    pub norm2: LayerNorm,
}

impl TransformerBlock {
    pub fn new(d_model: usize) -> Self {
        Self {
            attention: SelfAttention::new(d_model),
            norm1: LayerNorm::new(d_model),
            ff: FeedForward::new(d_model, d_model * 4),
            norm2: LayerNorm::new(d_model),
        }
    }

    pub fn zero_grad(&mut self) {
        self.attention.zero_grad();
        self.norm1.zero_grad();
        self.ff.zero_grad();
        self.norm2.zero_grad();
    }

    pub fn step(&mut self, lr: f32) {
        self.attention.step(lr);
        self.norm1.step(lr);
        self.ff.step(lr);
        self.norm2.step(lr);
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let attn_out = self.attention.forward(&self.norm1.forward(x));
        let x_add = x.add(&attn_out); // Residual

        let ff_out = self.ff.forward(&self.norm2.forward(&x_add));
        x_add.add(&ff_out) // Residual
    }
}
