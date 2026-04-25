use std::fmt;

#[derive(Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl Tensor {
    pub fn new(data: Vec<f32>, rows: usize, cols: usize) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "Data length must match rows * cols"
        );
        Tensor { data, rows, cols }
    }

    pub fn random(rows: usize, cols: usize) -> Self {

        use rand::RngExt;
        let mut rng = rand::rng();
        // Initialize with small random weights roughly normally distributed around 0
        let data = (0..rows * cols)
            .map(|_| rng.random_range(-0.1..0.1))
            .collect();
        Tensor { data, rows, cols }
    }

    pub fn transpose(&self) -> Self {
        let mut new_data = vec![0.0; self.data.len()];
        for r in 0..self.rows {
            for c in 0..self.cols {
                new_data[c * self.rows + r] = self.data[r * self.cols + c];
            }
        }
        Tensor::new(new_data, self.cols, self.rows)
    }

    pub fn matmul(&self, other: &Tensor) -> Self {
        assert_eq!(
            self.cols, other.rows,
            "Incompatible dimensions for matmul: {}x{} and {}x{}",
            self.rows, self.cols, other.rows, other.cols
        );
        let mut result = vec![0.0; self.rows * other.cols];

        // O(N^3) standard matrix multiplication
        for r in 0..self.rows {
            for c in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.data[r * self.cols + k] * other.data[k * other.cols + c];
                }
                result[r * other.cols + c] = sum;
            }
        }
        Tensor::new(result, self.rows, other.cols)
    }

    pub fn divide_by_scalar(&self, scalar: f32) -> Self {
        let new_data = self.data.iter().map(|&x| x / scalar).collect();
        Tensor::new(new_data, self.rows, self.cols)
    }

    pub fn add(&self, other: &Tensor) -> Self {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let new_data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Tensor::new(new_data, self.rows, self.cols)
    }

    // Applies softmax row by row (probabilities sum to 1.0 per row)
    pub fn softmax(&self) -> Self {
        let mut new_data = vec![0.0; self.data.len()];

        for r in 0..self.rows {
            let row_start = r * self.cols;
            let row_end = row_start + self.cols;
            let row_slice = &self.data[row_start..row_end];

            // Subtract max for numerical stability
            let mut max_val = f32::NEG_INFINITY;
            for &val in row_slice {
                if val > max_val {
                    max_val = val;
                }
            }

            let mut sum_exp = 0.0;
            for c in 0..self.cols {
                let exp_val = (row_slice[c] - max_val).exp();
                new_data[row_start + c] = exp_val;
                sum_exp += exp_val;
            }

            for c in 0..self.cols {
                new_data[row_start + c] /= sum_exp;
            }
        }

        Tensor::new(new_data, self.rows, self.cols)
    }

    // ReLU activation
    pub fn relu(&self) -> Self {
        let new_data = self
            .data
            .iter()
            .map(|&x| if x > 0.0 { x } else { 0.0 })
            .collect();
        Tensor::new(new_data, self.rows, self.cols)
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor(shape: {}x{})", self.rows, self.cols)
    }
}
