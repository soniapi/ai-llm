use std::f32;

use std::sync::{Arc, RwLock};

#[derive(Clone)]
pub enum Op {
    Leaf,
    Matmul(Tensor, Tensor),
    Add(Tensor, Tensor),
    BroadcastAdd(Tensor, Tensor),
    MulScalar(Tensor, f32),
    Transpose(Tensor),
    Relu(Tensor),
    Softmax(Tensor),
    LayerNorm(Tensor, Tensor, Tensor, f32), // input, weight, bias, eps
}

pub struct TensorData {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub grad: Option<Vec<f32>>,
    pub op: Op,
}

#[derive(Clone)]
pub struct Tensor {
    pub inner: Arc<RwLock<TensorData>>,
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.inner.read().unwrap();
        f.debug_struct("Tensor")
         .field("shape", &inner.shape)
         .field("data", &inner.data)
         .finish()
    }
}

impl Tensor {


    pub fn shape(&self) -> Vec<usize> {
        self.inner.read().unwrap().shape.clone()
    }

    pub fn data(&self) -> Vec<f32> {
        self.inner.read().unwrap().data.clone()
    }

    pub fn grad(&self) -> Option<Vec<f32>> {
        self.inner.read().unwrap().grad.clone()
    }




    pub fn set_grad(&self, grad: Vec<f32>) {
        self.inner.write().unwrap().grad = Some(grad);
    }

    pub fn add_to_grad(&self, grad: &[f32]) {
        let mut inner = self.inner.write().unwrap();
        if inner.grad.is_none() {
            inner.grad = Some(vec![0.0; inner.data.len()]);
        }
        let g = inner.grad.as_mut().unwrap();
        for (i, val) in grad.iter().enumerate() {
            g[i] += val;
        }
    }

    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let expected_len: usize = shape.iter().product();
        assert_eq!(data.len(), expected_len, "Data length does not match shape");
        Self {
            inner: Arc::new(RwLock::new(TensorData {
                data,
                shape,
                grad: None,
                op: Op::Leaf,
            })),
        }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let len: usize = shape.iter().product();
        Self {
            inner: Arc::new(RwLock::new(TensorData {
                data: vec![0.0; len],
                shape,
                grad: None,
                op: Op::Leaf,
            })),
        }
    }

    pub fn zero_grad(&self) {
        let mut inner = self.inner.write().unwrap();
        if let Some(g) = inner.grad.as_mut() {
            for val in g.iter_mut() {
                *val = 0.0;
            }
        }
    }

    pub fn step(&self, lr: f32) {
        let mut inner = self.inner.write().unwrap();
        let grad_clone = inner.grad.clone();
        if let Some(g) = grad_clone {
            for (w, grad) in inner.data.iter_mut().zip(g.iter()) {
                *w -= lr * grad;
            }
        }
    }

    pub fn cross_entropy(&self, targets: &[usize]) -> f32 {
        let probs = self.softmax();
        let probs_inner = probs.inner.read().unwrap();
        let seq_len = probs_inner.shape[0];
        let vocab_size = probs_inner.shape[1];

        let mut loss = 0.0;
        for i in 0..seq_len {
            let target_idx = targets[i];
            let prob = probs_inner.data[i * vocab_size + target_idx];
            loss -= prob.ln(); // -log(p)
        }

        loss / (seq_len as f32)
    }

    pub fn cross_entropy_grad(&self, targets: &[usize]) -> Tensor {
        let grad = self.softmax(); // dL/dz = p_i - y_i
        let seq_len = grad.shape()[0];
        let vocab_size = grad.shape()[1];

        {

            let mut inner = grad.inner.write().unwrap();
            let grad_data = &mut inner.data;
            for i in 0..seq_len {
                let target_idx = targets[i];
                grad_data[i * vocab_size + target_idx] -= 1.0;
            }

            for val in grad_data.iter_mut() {
                *val /= seq_len as f32;
            }
        }

        grad
    }
}

impl Tensor {

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let s_shape = self.shape();
        let o_shape = other.shape();
        assert_eq!(s_shape.len(), 2, "Matmul only implemented for 2D tensors");
        assert_eq!(o_shape.len(), 2, "Matmul only implemented for 2D tensors");
        assert_eq!(s_shape[1], o_shape[0], "Inner dimensions must match");

        let m = s_shape[0];
        let k = s_shape[1];
        let n = o_shape[1];

        let mut data = vec![0.0; m * n];
        let s_data = self.data();
        let o_data = other.data();

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    sum += s_data[i * k + p] * o_data[p * n + j];
                }
                data[i * n + j] = sum;
            }
        }

        Self {
            inner: Arc::new(RwLock::new(TensorData {
                data,
                shape: vec![m, n],
                grad: None,
                op: Op::Matmul(self.clone(), other.clone()),
            })),
        }
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        let s_shape = self.shape();
        let o_shape = other.shape();
        assert_eq!(s_shape, o_shape, "Shapes must match for addition");

        let s_data = self.data();
        let o_data = other.data();
        let data: Vec<f32> = s_data.iter().zip(o_data.iter()).map(|(a, b)| a + b).collect();

        Self {
            inner: Arc::new(RwLock::new(TensorData {
                data,
                shape: s_shape,
                grad: None,
                op: Op::Add(self.clone(), other.clone()),
            })),
        }
    }

    pub fn broadcast_add(&self, other: &Tensor) -> Tensor {
        let s_shape = self.shape();
        let o_shape = other.shape();
        assert_eq!(s_shape.len(), 2);
        assert_eq!(o_shape.len(), 1);
        assert_eq!(s_shape[1], o_shape[0]);

        let mut data = self.data().clone();
        let o_data = other.data();
        for i in 0..s_shape[0] {
            for j in 0..s_shape[1] {
                data[i * s_shape[1] + j] += o_data[j];
            }
        }

        Self {
            inner: Arc::new(RwLock::new(TensorData {
                data,
                shape: s_shape,
                grad: None,
                op: Op::BroadcastAdd(self.clone(), other.clone()),
            })),
        }
    }

    pub fn mul_scalar(&self, scalar: f32) -> Tensor {
        let data: Vec<f32> = self.data().iter().map(|&x| x * scalar).collect();
        Self {
            inner: Arc::new(RwLock::new(TensorData {
                data,
                shape: self.shape(),
                grad: None,
                op: Op::MulScalar(self.clone(), scalar),
            })),
        }
    }

    pub fn transpose(&self) -> Tensor {
        let shape = self.shape();
        assert_eq!(shape.len(), 2, "Transpose only implemented for 2D tensors");

        let m = shape[0];
        let n = shape[1];
        let mut data = vec![0.0; m * n];
        let s_data = self.data();

        for i in 0..m {
            for j in 0..n {
                data[j * m + i] = s_data[i * n + j];
            }
        }

        Self {
            inner: Arc::new(RwLock::new(TensorData {
                data,
                shape: vec![n, m],
                grad: None,
                op: Op::Transpose(self.clone()),
            })),
        }
    }

    pub fn relu(&self) -> Tensor {
        let data: Vec<f32> = self.data().iter().map(|&x| x.max(0.0)).collect();
        Self {
            inner: Arc::new(RwLock::new(TensorData {
                data,
                shape: self.shape(),
                grad: None,
                op: Op::Relu(self.clone()),
            })),
        }
    }

    pub fn softmax(&self) -> Tensor {
        let shape = self.shape();
        let mut result_data = self.data().clone();
        let last_dim = *shape.last().unwrap();
        let num_rows = result_data.len() / last_dim;

        for i in 0..num_rows {
            let start = i * last_dim;
            let end = start + last_dim;

            let mut max_val = std::f32::NEG_INFINITY;
            for j in start..end {
                if result_data[j] > max_val {
                    max_val = result_data[j];
                }
            }

            let mut sum_exp = 0.0;
            for j in start..end {
                result_data[j] = (result_data[j] - max_val).exp();
                sum_exp += result_data[j];
            }

            for j in start..end {
                result_data[j] /= sum_exp;
            }
        }

        Self {
            inner: Arc::new(RwLock::new(TensorData {
                data: result_data,
                shape,
                grad: None,
                op: Op::Softmax(self.clone()),
            })),
        }
    }

    pub fn layer_norm(&self, weight: &Tensor, bias: &Tensor, eps: f32) -> Tensor {
        let shape = self.shape();
        let mut result_data = self.data().clone();
        let last_dim = *shape.last().unwrap();
        let num_rows = result_data.len() / last_dim;
        let w_data = weight.data();
        let b_data = bias.data();

        for i in 0..num_rows {
            let start = i * last_dim;
            let end = start + last_dim;

            let mut sum = 0.0;
            for j in start..end {
                sum += result_data[j];
            }
            let mean = sum / last_dim as f32;

            let mut var_sum = 0.0;
            for j in start..end {
                let diff = result_data[j] - mean;
                var_sum += diff * diff;
            }
            let var = var_sum / last_dim as f32;
            let std = (var + eps).sqrt();

            for j in start..end {
                let norm_val = (result_data[j] - mean) / std;
                result_data[j] = norm_val * w_data[j - start] + b_data[j - start];
            }
        }

        Self {
            inner: Arc::new(RwLock::new(TensorData {
                data: result_data,
                shape,
                grad: None,
                op: Op::LayerNorm(self.clone(), weight.clone(), bias.clone(), eps),
            })),
        }
    }
}

impl Tensor {
    pub fn backward(&self, grad: Option<&Tensor>) {
        let topo = self.build_topo();

        // Initialize gradient of the root node
        if let Some(g) = grad {
            self.set_grad(g.data().clone());
        } else {
            let inner = self.inner.read().unwrap();
            self.set_grad(vec![1.0; inner.data.len()]);
        }

        // Iterate backwards over the topological sort
        for t in topo.iter().rev() {
            let inner = t.inner.read().unwrap();
            let grad_opt = inner.grad.clone();

            if grad_opt.is_none() {
                continue;
            }
            let g = grad_opt.unwrap();

            match &inner.op {
                Op::Leaf => {}
                Op::Matmul(a, b) => {
                    // dL/dA = dL/dC * B^T
                    // dL/dB = A^T * dL/dC
                    let m = a.shape()[0];
                    let k = a.shape()[1];
                    let n = b.shape()[1];

                    let a_data = a.data();
                    let b_data = b.data();

                    let mut grad_a = vec![0.0; m * k];
                    let mut grad_b = vec![0.0; k * n];

                    for i in 0..m {
                        for j in 0..n {
                            let g_val = g[i * n + j];
                            for p in 0..k {
                                grad_a[i * k + p] += g_val * b_data[p * n + j];
                                grad_b[p * n + j] += a_data[i * k + p] * g_val;
                            }
                        }
                    }

                    a.add_to_grad(&grad_a);
                    b.add_to_grad(&grad_b);
                }
                Op::Add(a, b) => {
                    a.add_to_grad(&g);
                    b.add_to_grad(&g);
                }
                Op::BroadcastAdd(a, b) => {
                    a.add_to_grad(&g);

                    let a_shape = a.shape();
                    let mut grad_b = vec![0.0; b.shape()[0]];
                    for i in 0..a_shape[0] {
                        for j in 0..a_shape[1] {
                            grad_b[j] += g[i * a_shape[1] + j];
                        }
                    }
                    b.add_to_grad(&grad_b);
                }
                Op::MulScalar(a, scalar) => {
                    let grad_a: Vec<f32> = g.iter().map(|&x| x * scalar).collect();
                    a.add_to_grad(&grad_a);
                }
                Op::Transpose(a) => {
                    let shape = a.shape();
                    let m = shape[0];
                    let n = shape[1];
                    let mut grad_a = vec![0.0; m * n];
                    for i in 0..m {
                        for j in 0..n {
                            grad_a[i * n + j] = g[j * m + i];
                        }
                    }
                    a.add_to_grad(&grad_a);
                }
                Op::Relu(a) => {
                    let a_data = a.data();
                    let grad_a: Vec<f32> = g.iter().zip(a_data.iter()).map(|(&g_val, &a_val)| {
                        if a_val > 0.0 { g_val } else { 0.0 }
                    }).collect();
                    a.add_to_grad(&grad_a);
                }
                Op::Softmax(a) => {
                    let a_shape = a.shape();
                    let last_dim = *a_shape.last().unwrap();
                    let num_rows = inner.data.len() / last_dim;
                    let mut grad_a = vec![0.0; inner.data.len()];

                    for i in 0..num_rows {
                        let start = i * last_dim;
                        let end = start + last_dim;

                        for j in start..end {
                            let p_j = inner.data[j];
                            let mut sum = 0.0;
                            for k in start..end {
                                let p_k = inner.data[k];
                                let jacobian = if j == k { p_j * (1.0 - p_j) } else { -p_j * p_k };
                                sum += jacobian * g[k];
                            }
                            grad_a[j] = sum;
                        }
                    }
                    a.add_to_grad(&grad_a);
                }
                Op::LayerNorm(a, weight, _bias, eps) => {
                    let a_shape = a.shape();
                    let last_dim = *a_shape.last().unwrap();
                    let num_rows = inner.data.len() / last_dim;

                    let mut grad_a = vec![0.0; inner.data.len()];
                    let mut grad_w = vec![0.0; last_dim];
                    let mut grad_b = vec![0.0; last_dim];

                    let a_data = a.data();
                    let w_data = weight.data();

                    for i in 0..num_rows {
                        let start = i * last_dim;
                        let end = start + last_dim;

                        let mut sum = 0.0;
                        for j in start..end { sum += a_data[j]; }
                        let mean = sum / last_dim as f32;

                        let mut var_sum = 0.0;
                        for j in start..end {
                            let diff = a_data[j] - mean;
                            var_sum += diff * diff;
                        }
                        let var = var_sum / last_dim as f32;
                        let std = (var + *eps).sqrt();

                        for j in start..end {
                            let norm_val = (a_data[j] - mean) / std;
                            let idx = j - start;

                            grad_w[idx] += g[j] * norm_val;
                            grad_b[idx] += g[j];

                            // Simplified backward for LayerNorm wrt input
                            let dx_hat = g[j] * w_data[idx];
                            grad_a[j] += dx_hat / std;
                        }

                        // We omit the full exact backprop through mean and var for this simplified model
                    }

                    a.add_to_grad(&grad_a);
                    weight.add_to_grad(&grad_w);
                    _bias.add_to_grad(&grad_b);
                }
            }
        }
    }

    fn build_topo(&self) -> Vec<Tensor> {
        let mut topo = Vec::new();
        let mut visited = std::collections::HashSet::new();
        self.build_topo_internal(&mut topo, &mut visited);
        topo
    }

    fn build_topo_internal(&self, topo: &mut Vec<Tensor>, visited: &mut std::collections::HashSet<usize>) {
        let ptr = Arc::as_ptr(&self.inner) as usize;
        if visited.insert(ptr) {
            let inner = self.inner.read().unwrap();
            match &inner.op {
                Op::Leaf => {}
                Op::Matmul(a, b) => {
                    a.build_topo_internal(topo, visited);
                    b.build_topo_internal(topo, visited);
                }
                Op::Add(a, b) => {
                    a.build_topo_internal(topo, visited);
                    b.build_topo_internal(topo, visited);
                }
                Op::BroadcastAdd(a, b) => {
                    a.build_topo_internal(topo, visited);
                    b.build_topo_internal(topo, visited);
                }
                Op::MulScalar(a, _) => {
                    a.build_topo_internal(topo, visited);
                }
                Op::Transpose(a) => {
                    a.build_topo_internal(topo, visited);
                }
                Op::Relu(a) => {
                    a.build_topo_internal(topo, visited);
                }
                Op::Softmax(a) => {
                    a.build_topo_internal(topo, visited);
                }
                Op::LayerNorm(a, w, b, _) => {
                    a.build_topo_internal(topo, visited);
                    w.build_topo_internal(topo, visited);
                    b.build_topo_internal(topo, visited);
                }
            }
            topo.push(self.clone());
        }
    }
}
