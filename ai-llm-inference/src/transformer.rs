use crate::tensor::Tensor;

pub struct Linear {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let weight = Tensor::new(vec![0.01; in_features * out_features], vec![in_features, out_features]);
        let bias = Tensor::zeros(vec![out_features]);
        Self { weight, bias }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.matmul(&self.weight).broadcast_add(&self.bias)
    }

    pub fn zero_grad(&self) {
        self.weight.zero_grad();
        self.bias.zero_grad();
    }

    pub fn step(&self, lr: f32) {
        self.weight.step(lr);
        self.bias.step(lr);
    }
}

pub struct LayerNorm {
    pub weight: Tensor,
    pub bias: Tensor,
    pub eps: f32,
}

impl LayerNorm {
    pub fn new(features: usize) -> Self {
        Self {
            weight: Tensor::new(vec![1.0; features], vec![features]),
            bias: Tensor::zeros(vec![features]),
            eps: 1e-5,
        }
    }

    pub fn zero_grad(&self) {
        self.weight.zero_grad();
        self.bias.zero_grad();
    }

    pub fn step(&self, lr: f32) {
        self.weight.step(lr);
        self.bias.step(lr);
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.layer_norm(&self.weight, &self.bias, self.eps)
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

    pub fn zero_grad(&self) {
        self.q_proj.zero_grad();
        self.k_proj.zero_grad();
        self.v_proj.zero_grad();
        self.out_proj.zero_grad();
    }

    pub fn step(&self, lr: f32) {
        self.q_proj.step(lr);
        self.k_proj.step(lr);
        self.v_proj.step(lr);
        self.out_proj.step(lr);
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        let scores = q.matmul(&k.transpose()).mul_scalar(1.0 / self.d_k);
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

    pub fn zero_grad(&self) {
        self.linear1.zero_grad();
        self.linear2.zero_grad();
    }

    pub fn step(&self, lr: f32) {
        self.linear1.step(lr);
        self.linear2.step(lr);
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let h = self.linear1.forward(x).relu();
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

    pub fn zero_grad(&self) {
        self.attention.zero_grad();
        self.norm1.zero_grad();
        self.ff.zero_grad();
        self.norm2.zero_grad();
    }

    pub fn step(&self, lr: f32) {
        self.attention.step(lr);
        self.norm1.step(lr);
        self.ff.step(lr);
        self.norm2.step(lr);
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let attn_out = self.attention.forward(&self.norm1.forward(x));
        let x_add = x.add(&attn_out);

        let ff_out = self.ff.forward(&self.norm2.forward(&x_add));
        x_add.add(&ff_out)
    }
}
