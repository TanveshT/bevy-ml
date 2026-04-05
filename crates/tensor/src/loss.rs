use crate::Tensor;

/// NLL loss: -log_probs[target] averaged over batch.
/// log_probs: [batch, C], targets: &[usize]
/// Returns scalar loss and per-sample gradient.
pub fn nll_loss(log_probs: &Tensor, targets: &[usize]) -> (f32, Tensor) {
    let batch = log_probs.shape[0];
    let c = log_probs.shape[1];
    assert_eq!(targets.len(), batch);

    let mut total = 0.0f32;
    let mut grad = vec![0.0f32; batch * c];

    for s in 0..batch {
        let t = targets[s];
        debug_assert!(t < c);
        total -= log_probs.data[s * c + t];

        // Gradient: softmax(i) - 1{i==t}, divided by batch
        for i in 0..c {
            let softmax_i = log_probs.data[s * c + i].exp();
            grad[s * c + i] = if i == t {
                (softmax_i - 1.0) / batch as f32
            } else {
                softmax_i / batch as f32
            };
        }
    }

    (total / batch as f32, Tensor::from_data(grad, &[batch, c]))
}

/// Argmax along last dimension. Returns indices.
pub fn argmax(tensor: &Tensor) -> Vec<usize> {
    let c = *tensor.shape.last().unwrap();
    let batch = tensor.data.len() / c;
    (0..batch)
        .map(|s| {
            let row = &tensor.data[s * c..(s + 1) * c];
            row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::log_softmax;

    #[test]
    fn test_nll_loss_gradient_sums_to_zero_per_sample() {
        let logits = Tensor::from_data(vec![1.0, 2.0, 3.0, 1.0, 1.0, 1.0], &[2, 3]);
        let log_probs = log_softmax(&logits);
        let (_, grad) = nll_loss(&log_probs, &[2, 0]);
        // Each row of grad should sum to 0 (softmax sums to 1, subtract 1 at target)
        let sum0: f32 = grad.data[0..3].iter().sum();
        let sum1: f32 = grad.data[3..6].iter().sum();
        assert!(sum0.abs() < 1e-5);
        assert!(sum1.abs() < 1e-5);
    }

    #[test]
    fn test_argmax() {
        let t = Tensor::from_data(vec![0.1, 0.9, 0.0, 0.3, 0.1, 0.6], &[2, 3]);
        assert_eq!(argmax(&t), vec![1, 2]);
    }
}
