use crate::Tensor;

/// ReLU: max(0, x) element-wise.
pub fn relu(input: &Tensor) -> Tensor {
    let data: Vec<f32> = input.data.iter().map(|&x| x.max(0.0)).collect();
    Tensor::from_data(data, &input.shape)
}

/// ReLU backward: d_output * (input > 0).
pub fn relu_backward(d_output: &Tensor, input: &Tensor) -> Tensor {
    let data: Vec<f32> = d_output
        .data
        .iter()
        .zip(input.data.iter())
        .map(|(&d, &x)| if x > 0.0 { d } else { 0.0 })
        .collect();
    Tensor::from_data(data, &input.shape)
}

/// Log-softmax along the last dimension.
/// input: [..., C] → output: [..., C]
pub fn log_softmax(input: &Tensor) -> Tensor {
    let c = *input.shape.last().unwrap();
    let batch: usize = input.data.len() / c;
    let mut out = vec![0.0f32; input.data.len()];

    for s in 0..batch {
        let row = &input.data[s * c..(s + 1) * c];
        let out_row = &mut out[s * c..(s + 1) * c];
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = row.iter().map(|&x| (x - max).exp()).sum();
        let log_sum_exp = max + sum_exp.ln();
        for (o, &x) in out_row.iter_mut().zip(row.iter()) {
            *o = x - log_sum_exp;
        }
    }
    Tensor::from_data(out, &input.shape)
}

/// Sigmoid: 1 / (1 + exp(-x)).
pub fn sigmoid(input: &Tensor) -> Tensor {
    let data: Vec<f32> = input
        .data
        .iter()
        .map(|&x| 1.0 / (1.0 + (-x).exp()))
        .collect();
    Tensor::from_data(data, &input.shape)
}

/// Dropout: randomly zero elements with probability p. Returns (output, mask).
/// In eval mode, call this with p=0 or skip.
pub fn dropout(input: &Tensor, p: f32, rng: &mut impl rand::Rng) -> (Tensor, Vec<bool>) {
    let scale = 1.0 / (1.0 - p);
    let mut mask = vec![true; input.data.len()];
    let data: Vec<f32> = input
        .data
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            if rng.random_range(0.0f32..1.0) < p {
                mask[i] = false;
                0.0
            } else {
                x * scale
            }
        })
        .collect();
    (Tensor::from_data(data, &input.shape), mask)
}

/// Dropout backward: d_output * mask * scale.
pub fn dropout_backward(d_output: &Tensor, mask: &[bool], p: f32) -> Tensor {
    let scale = 1.0 / (1.0 - p);
    let data: Vec<f32> = d_output
        .data
        .iter()
        .zip(mask.iter())
        .map(|(&d, &m)| if m { d * scale } else { 0.0 })
        .collect();
    Tensor::from_data(data, &d_output.shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        let t = Tensor::from_data(vec![-1.0, 0.0, 1.0, 2.0], &[4]);
        let r = relu(&t);
        assert_eq!(r.data, vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_log_softmax_sums_to_one() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0], &[1, 3]);
        let ls = log_softmax(&t);
        let sum_exp: f32 = ls.data.iter().map(|&x| x.exp()).sum();
        assert!((sum_exp - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_relu_backward() {
        let input = Tensor::from_data(vec![-1.0, 2.0, 0.0, 3.0], &[4]);
        let d_out = Tensor::from_data(vec![1.0, 1.0, 1.0, 1.0], &[4]);
        let d_in = relu_backward(&d_out, &input);
        assert_eq!(d_in.data, vec![0.0, 1.0, 0.0, 1.0]);
    }
}
