pub mod activations;
pub mod conv_ops;
pub mod init;
pub mod loss;
pub mod ops;
pub mod pool;

use std::fmt;

/// A multi-dimensional tensor backed by contiguous `Vec<f32>` with row-major strides.
#[derive(Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

impl Tensor {
    pub fn zeros(shape: &[usize]) -> Self {
        let size: usize = shape.iter().product();
        Self {
            data: vec![0.0; size],
            shape: shape.to_vec(),
            strides: compute_strides(shape),
        }
    }

    pub fn from_data(data: Vec<f32>, shape: &[usize]) -> Self {
        let size: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            size,
            "data len {} != shape product {}",
            data.len(),
            size
        );
        Self {
            data,
            shape: shape.to_vec(),
            strides: compute_strides(shape),
        }
    }

    pub fn from_slice(data: &[f32], shape: &[usize]) -> Self {
        Self::from_data(data.to_vec(), shape)
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Reshape (must be contiguous and same total elements).
    pub fn view(&self, new_shape: &[usize]) -> Self {
        let new_size: usize = new_shape.iter().product();
        assert_eq!(
            self.numel(),
            new_size,
            "view: {} elements != {}",
            self.numel(),
            new_size
        );
        Self::from_data(self.data.clone(), new_shape)
    }

    /// Transpose last two dimensions (creates new contiguous tensor).
    pub fn t(&self) -> Self {
        assert!(self.ndim() >= 2, "transpose requires at least 2D");
        let n = self.ndim();
        let rows = self.shape[n - 2];
        let cols = self.shape[n - 1];
        let batch: usize = self.shape[..n - 2].iter().product();
        let mat_size = rows * cols;

        let mut out = vec![0.0f32; self.data.len()];
        for b in 0..batch {
            let src = &self.data[b * mat_size..(b + 1) * mat_size];
            let dst = &mut out[b * mat_size..(b + 1) * mat_size];
            for r in 0..rows {
                for c in 0..cols {
                    dst[c * rows + r] = src[r * cols + c];
                }
            }
        }

        let mut new_shape = self.shape.clone();
        new_shape[n - 2] = cols;
        new_shape[n - 1] = rows;
        Self::from_data(out, &new_shape)
    }

    /// Fill with a scalar value.
    pub fn fill(&mut self, val: f32) {
        self.data.fill(val);
    }

    /// Add another tensor element-wise (in-place).
    pub fn add_inplace(&mut self, other: &Tensor) {
        assert_eq!(self.shape, other.shape, "add_inplace shape mismatch");
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += b;
        }
    }

    /// Scale all elements by a scalar.
    pub fn scale(&mut self, s: f32) {
        for x in self.data.iter_mut() {
            *x *= s;
        }
    }
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tensor(shape={:?}, data[..5]={:?})",
            self.shape,
            &self.data[..self.data.len().min(5)]
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let t = Tensor::zeros(&[3, 4]);
        assert_eq!(t.numel(), 12);
        assert!(t.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_transpose() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let tt = t.t();
        assert_eq!(tt.shape, vec![3, 2]);
        assert_eq!(tt.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_view() {
        let t = Tensor::zeros(&[2, 3, 4]);
        let v = t.view(&[6, 4]);
        assert_eq!(v.shape, vec![6, 4]);
        assert_eq!(v.numel(), 24);
    }
}
