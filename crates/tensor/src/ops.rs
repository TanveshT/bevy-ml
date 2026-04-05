use crate::Tensor;

/// Dot product of two slices.
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Matrix multiply: A[M×K] × B[K×N] → C[M×N].
/// Supports batched: A[..×M×K] × B[..×K×N] → C[..×M×N] (batch dims must match).
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    assert!(a.ndim() >= 2 && b.ndim() >= 2);
    let m = a.shape[a.ndim() - 2];
    let k_a = a.shape[a.ndim() - 1];
    let k_b = b.shape[b.ndim() - 2];
    let n = b.shape[b.ndim() - 1];
    assert_eq!(k_a, k_b, "matmul: inner dims mismatch {} vs {}", k_a, k_b);
    let k = k_a;

    let batch: usize = a.shape[..a.ndim() - 2].iter().product();
    let batch_b: usize = b.shape[..b.ndim() - 2].iter().product();
    assert!(
        batch == batch_b || batch == 1 || batch_b == 1,
        "matmul batch mismatch: {} vs {}",
        batch,
        batch_b
    );

    let mat_a = m * k;
    let mat_b = k * n;
    let mat_c = m * n;
    let mut out = vec![0.0f32; batch.max(batch_b) * mat_c];

    for bi in 0..batch.max(batch_b) {
        let a_off = (bi % batch) * mat_a;
        let b_off = (bi % batch_b) * mat_b;
        let c_off = bi * mat_c;
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a.data[a_off + i * k + p] * b.data[b_off + p * n + j];
                }
                out[c_off + i * n + j] = sum;
            }
        }
    }

    let mut shape = a.shape[..a.ndim() - 2].to_vec();
    if batch_b > batch {
        shape = b.shape[..b.ndim() - 2].to_vec();
    }
    shape.push(m);
    shape.push(n);
    Tensor::from_data(out, &shape)
}

/// Linear layer: output = input × W^T + bias
/// input: [batch, in_features], W: [out_features, in_features], bias: [out_features]
/// output: [batch, out_features]
pub fn linear(input: &Tensor, weight: &Tensor, bias: &Tensor) -> Tensor {
    let batch = input.shape[0];
    let in_f = input.shape[1];
    let out_f = weight.shape[0];
    assert_eq!(weight.shape[1], in_f);
    assert_eq!(bias.shape[0], out_f);

    let mut out = vec![0.0f32; batch * out_f];
    for s in 0..batch {
        let inp = &input.data[s * in_f..(s + 1) * in_f];
        for j in 0..out_f {
            let w_row = &weight.data[j * in_f..(j + 1) * in_f];
            out[s * out_f + j] = dot_product(inp, w_row) + bias.data[j];
        }
    }
    Tensor::from_data(out, &[batch, out_f])
}

/// Backward for linear layer.
/// Returns (d_input, d_weight, d_bias).
pub fn linear_backward(
    d_output: &Tensor, // [batch, out_f]
    input: &Tensor,    // [batch, in_f]
    weight: &Tensor,   // [out_f, in_f]
) -> (Tensor, Tensor, Tensor) {
    let batch = d_output.shape[0];
    let out_f = d_output.shape[1];
    let in_f = input.shape[1];

    // d_input = d_output × W  [batch, in_f]
    let mut d_input = vec![0.0f32; batch * in_f];
    for s in 0..batch {
        for j in 0..out_f {
            let d_oj = d_output.data[s * out_f + j];
            if d_oj != 0.0 {
                for k in 0..in_f {
                    d_input[s * in_f + k] += d_oj * weight.data[j * in_f + k];
                }
            }
        }
    }

    // d_weight = d_output^T × input  [out_f, in_f]
    let mut d_weight = vec![0.0f32; out_f * in_f];
    for s in 0..batch {
        for j in 0..out_f {
            let d_oj = d_output.data[s * out_f + j];
            if d_oj != 0.0 {
                for k in 0..in_f {
                    d_weight[j * in_f + k] += d_oj * input.data[s * in_f + k];
                }
            }
        }
    }

    // d_bias = sum over batch of d_output  [out_f]
    let mut d_bias = vec![0.0f32; out_f];
    for s in 0..batch {
        for (j, d_bias_j) in d_bias.iter_mut().enumerate() {
            *d_bias_j += d_output.data[s * out_f + j];
        }
    }

    (
        Tensor::from_data(d_input, &[batch, in_f]),
        Tensor::from_data(d_weight, &[out_f, in_f]),
        Tensor::from_data(d_bias, &[out_f]),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_2x3_3x2() {
        let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = Tensor::from_data(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]);
        let c = matmul(&a, &b);
        assert_eq!(c.shape, vec![2, 2]);
        assert_eq!(c.data, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_linear() {
        let input = Tensor::from_data(vec![1.0, 2.0, 3.0], &[1, 3]);
        let weight = Tensor::from_data(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3]);
        let bias = Tensor::from_data(vec![0.5, -0.5], &[2]);
        let out = linear(&input, &weight, &bias);
        assert_eq!(out.shape, vec![1, 2]);
        assert!((out.data[0] - 1.5).abs() < 1e-6);
        assert!((out.data[1] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_linear_backward_shapes() {
        let input = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let weight = Tensor::from_data(vec![0.1; 6], &[2, 3]);
        let d_output = Tensor::from_data(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let (di, dw, db) = linear_backward(&d_output, &input, &weight);
        assert_eq!(di.shape, vec![2, 3]);
        assert_eq!(dw.shape, vec![2, 3]);
        assert_eq!(db.shape, vec![2]);
    }
}
