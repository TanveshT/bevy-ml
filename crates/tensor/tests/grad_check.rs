//! Numerical gradient checks for tensor ops.
//! For each op, compare analytical gradient against finite-difference approximation.

use ecs_ml_tensor::Tensor;
use ecs_ml_tensor::{activations, conv_ops, loss, ops, pool};

const EPS: f32 = 1e-4;
const TOL: f32 = 2e-2; // relative tolerance (f32 finite-difference precision)

fn numerical_grad<F: Fn(&Tensor) -> f32>(x: &Tensor, f: F) -> Tensor {
    let mut grad = vec![0.0f32; x.data.len()];
    for (i, g) in grad.iter_mut().enumerate() {
        let mut x_plus = x.clone();
        let mut x_minus = x.clone();
        x_plus.data[i] += EPS;
        x_minus.data[i] -= EPS;
        *g = (f(&x_plus) - f(&x_minus)) / (2.0 * EPS);
    }
    Tensor::from_data(grad, &x.shape)
}

fn check_close(analytical: &Tensor, numerical: &Tensor, name: &str) {
    assert_eq!(analytical.shape, numerical.shape, "{name}: shape mismatch");
    for (i, (&a, &n)) in analytical
        .data
        .iter()
        .zip(numerical.data.iter())
        .enumerate()
    {
        let denom = a.abs().max(n.abs()).max(1e-6);
        let rel_err = (a - n).abs() / denom;
        assert!(
            rel_err < TOL,
            "{name}[{i}]: analytical={a:.6} numerical={n:.6} rel_err={rel_err:.6}"
        );
    }
}

// ===== Linear =====

#[test]
fn grad_check_linear_weight() {
    let input = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let weight = Tensor::from_data(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3]);
    let bias = Tensor::from_data(vec![0.0, 0.0], &[2]);
    let d_output = Tensor::from_data(vec![1.0, 0.5, 0.5, 1.0], &[2, 2]);

    let (_, dw_analytical, _) = ops::linear_backward(&d_output, &input, &weight);

    let dw_numerical = numerical_grad(&weight, |w| {
        let out = ops::linear(&input, w, &bias);
        out.data
            .iter()
            .zip(d_output.data.iter())
            .map(|(&o, &d)| o * d)
            .sum()
    });

    check_close(&dw_analytical, &dw_numerical, "linear_dw");
}

#[test]
fn grad_check_linear_input() {
    let input = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let weight = Tensor::from_data(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3]);
    let bias = Tensor::from_data(vec![0.1, -0.1], &[2]);
    let d_output = Tensor::from_data(vec![1.0, 0.5, 0.5, 1.0], &[2, 2]);

    let (di_analytical, _, _) = ops::linear_backward(&d_output, &input, &weight);

    let di_numerical = numerical_grad(&input, |inp| {
        let out = ops::linear(inp, &weight, &bias);
        out.data
            .iter()
            .zip(d_output.data.iter())
            .map(|(&o, &d)| o * d)
            .sum()
    });

    check_close(&di_analytical, &di_numerical, "linear_di");
}

#[test]
fn grad_check_linear_bias() {
    let input = Tensor::from_data(vec![1.0, 2.0, 3.0], &[1, 3]);
    let weight = Tensor::from_data(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3]);
    let bias = Tensor::from_data(vec![0.1, -0.1], &[2]);
    let d_output = Tensor::from_data(vec![1.0, 0.5], &[1, 2]);

    let (_, _, db_analytical) = ops::linear_backward(&d_output, &input, &weight);

    let db_numerical = numerical_grad(&bias, |b| {
        let out = ops::linear(&input, &weight, b);
        out.data
            .iter()
            .zip(d_output.data.iter())
            .map(|(&o, &d)| o * d)
            .sum()
    });

    check_close(&db_analytical, &db_numerical, "linear_db");
}

// ===== ReLU =====

#[test]
fn grad_check_relu() {
    // Avoid zero (non-differentiable)
    let input = Tensor::from_data(vec![-2.0, 1.0, -0.5, 3.0, 0.1, -1.0], &[2, 3]);
    let d_output = Tensor::from_data(vec![1.0; 6], &[2, 3]);

    let di_analytical = activations::relu_backward(&d_output, &input);

    let di_numerical = numerical_grad(&input, |inp| {
        let out = activations::relu(inp);
        out.data
            .iter()
            .zip(d_output.data.iter())
            .map(|(&o, &d)| o * d)
            .sum()
    });

    check_close(&di_analytical, &di_numerical, "relu_di");
}

// ===== Conv2d =====

#[test]
fn grad_check_conv2d_weight() {
    let input = Tensor::from_data((0..18).map(|i| i as f32 * 0.1).collect(), &[1, 2, 3, 3]);
    let weight = Tensor::from_data(vec![0.1; 8], &[1, 2, 2, 2]);
    let bias = Tensor::from_data(vec![0.0], &[1]);
    let padding = 0;
    let stride = 1;

    let out = conv_ops::direct_conv2d(&input, &weight, &bias, padding, stride);
    let d_output = Tensor::from_data(vec![1.0; out.data.len()], &out.shape);

    let (_, dw_analytical, _) =
        conv_ops::direct_conv2d_backward(&d_output, &input, &weight, padding, stride);

    let dw_numerical = numerical_grad(&weight, |w| {
        let out = conv_ops::direct_conv2d(&input, w, &bias, padding, stride);
        out.data
            .iter()
            .zip(d_output.data.iter())
            .map(|(&o, &d)| o * d)
            .sum()
    });

    check_close(&dw_analytical, &dw_numerical, "conv2d_dw");
}

#[test]
fn grad_check_conv2d_input() {
    let input = Tensor::from_data((0..18).map(|i| i as f32 * 0.1).collect(), &[1, 2, 3, 3]);
    let weight = Tensor::from_data(
        vec![0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.15, -0.15],
        &[1, 2, 2, 2],
    );
    let bias = Tensor::from_data(vec![0.05], &[1]);
    let padding = 0;
    let stride = 1;

    let out = conv_ops::direct_conv2d(&input, &weight, &bias, padding, stride);
    let d_output = Tensor::from_data(vec![1.0; out.data.len()], &out.shape);

    let (di_analytical, _, _) =
        conv_ops::direct_conv2d_backward(&d_output, &input, &weight, padding, stride);

    let di_numerical = numerical_grad(&input, |inp| {
        let out = conv_ops::direct_conv2d(inp, &weight, &bias, padding, stride);
        out.data
            .iter()
            .zip(d_output.data.iter())
            .map(|(&o, &d)| o * d)
            .sum()
    });

    check_close(&di_analytical, &di_numerical, "conv2d_di");
}

// ===== MaxPool2d =====

#[test]
fn grad_check_max_pool2d() {
    // Use distinct values so argmax is unambiguous
    let data: Vec<f32> = (0..16).map(|i| i as f32 + 0.5).collect();
    let input = Tensor::from_data(data, &[1, 1, 4, 4]);

    let (out, indices) = pool::max_pool2d(&input, 2);
    let d_output = Tensor::from_data(vec![1.0; out.data.len()], &out.shape);

    let di_analytical = pool::max_pool2d_backward(&d_output, &indices, &input.shape);

    let di_numerical = numerical_grad(&input, |inp| {
        let (out, _) = pool::max_pool2d(inp, 2);
        out.data
            .iter()
            .zip(d_output.data.iter())
            .map(|(&o, &d)| o * d)
            .sum()
    });

    check_close(&di_analytical, &di_numerical, "max_pool2d_di");
}

// ===== LogSoftmax + NLL Loss (combined) =====

#[test]
fn grad_check_log_softmax_nll() {
    // The analytical gradient from nll_loss is (softmax - one_hot) / N,
    // which is the gradient w.r.t. the *logits* before log_softmax.
    // So we check the full chain: logits → log_softmax → nll_loss.
    let logits = Tensor::from_data(vec![1.0, 2.0, 0.5, 0.5, 1.5, 1.0], &[2, 3]);
    let targets = vec![1, 0];

    let log_probs = activations::log_softmax(&logits);
    let (_, grad_analytical) = loss::nll_loss(&log_probs, &targets);

    // Numerical gradient w.r.t. logits
    let grad_numerical = numerical_grad(&logits, |x| {
        let lp = activations::log_softmax(x);
        let (l, _) = loss::nll_loss(&lp, &targets);
        l
    });

    check_close(&grad_analytical, &grad_numerical, "log_softmax_nll_grad");
}
