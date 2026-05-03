use std::f32;

#[derive(Clone, Debug)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let expected_len: usize = shape.iter().product();
        assert_eq!(data.len(), expected_len, "Data length does not match shape");
        Self { data, shape }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let len: usize = shape.iter().product();
        Self {
            data: vec![0.0; len],
            shape,
        }
    }

    // A very simple 2D matmul
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape.len(), 2, "Matmul only implemented for 2D tensors");
        assert_eq!(other.shape.len(), 2, "Matmul only implemented for 2D tensors");
        assert_eq!(self.shape[1], other.shape[0], "Inner dimensions must match");

        let m = self.shape[0];
        let k = self.shape[1];
        let n = other.shape[1];

        let mut result = Tensor::zeros(vec![m, n]);

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    sum += self.data[i * k + p] * other.data[p * n + j];
                }
                result.data[i * n + j] = sum;
            }
        }

        result
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "Shapes must match for addition");
        let data: Vec<f32> = self.data.iter().zip(other.data.iter()).map(|(a, b)| a + b).collect();
        Tensor::new(data, self.shape.clone())
    }

    pub fn broadcast_add(&self, other: &Tensor) -> Tensor {
        // Simple broadcast add for 2D + 1D (bias)
        assert_eq!(self.shape.len(), 2);
        assert_eq!(other.shape.len(), 1);
        assert_eq!(self.shape[1], other.shape[0]);

        let mut data = self.data.clone();
        for i in 0..self.shape[0] {
            for j in 0..self.shape[1] {
                data[i * self.shape[1] + j] += other.data[j];
            }
        }
        Tensor::new(data, self.shape.clone())
    }

    // Apply Softmax along the last dimension
    pub fn softmax(&self) -> Tensor {
        let mut result_data = self.data.clone();
        let last_dim = *self.shape.last().unwrap();
        let num_rows = self.data.len() / last_dim;

        for i in 0..num_rows {
            let start = i * last_dim;
            let end = start + last_dim;

            let mut max_val = std::f32::NEG_INFINITY;
            for j in start..end {
                if self.data[j] > max_val {
                    max_val = self.data[j];
                }
            }

            let mut sum_exp = 0.0;
            for j in start..end {
                result_data[j] = (self.data[j] - max_val).exp();
                sum_exp += result_data[j];
            }

            for j in start..end {
                result_data[j] /= sum_exp;
            }
        }

        Tensor::new(result_data, self.shape.clone())
    }

    /// Computes the Cross Entropy loss between logits (2D tensor [seq_len, vocab_size])
    /// and target indices (1D array of length seq_len).
    pub fn cross_entropy(&self, targets: &[usize]) -> f32 {
        let probs = self.softmax();
        let seq_len = self.shape[0];
        let vocab_size = self.shape[1];

        let mut loss = 0.0;
        for i in 0..seq_len {
            let target_idx = targets[i];
            let prob = probs.data[i * vocab_size + target_idx];
            loss -= prob.ln(); // -log(p)
        }

        loss / (seq_len as f32)
    }

    /// Computes the gradient of Cross Entropy loss w.r.t the logits.
    /// Returns a tensor of the same shape as self.
    pub fn cross_entropy_grad(&self, targets: &[usize]) -> Tensor {
        let mut grad = self.softmax(); // dL/dz = p_i - y_i
        let seq_len = self.shape[0];
        let vocab_size = self.shape[1];

        for i in 0..seq_len {
            let target_idx = targets[i];
            grad.data[i * vocab_size + target_idx] -= 1.0;
        }

        // Average over batch/sequence length
        for val in grad.data.iter_mut() {
            *val /= seq_len as f32;
        }

        grad
    }
}
