use bevy_ecs::world::World;

use ecs_ml_tensor::ops;
use ecs_ml_tensor::{Tensor, activations, conv_ops, loss, pool};

use crate::components::*;
use crate::dispatch_row_size;
use crate::resources::*;

// --- contiguous_iter helpers ---

/// Read weight matrix from row entities via contiguous_iter.
/// Returns a Tensor [out_features, in_features] assembled from the contiguous column.
fn read_contiguous_weights<const N: usize>(
    world: &mut World,
    layer_idx: usize,
) -> (Tensor, Tensor) {
    let mut query = world.query::<(&WeightRow<N>, &BiasValue, &RowIndex, &LayerIndex)>();

    // Collect rows for this layer, sorted by RowIndex
    let mut rows: Vec<(usize, &[f32; N], f32)> = Vec::new();
    for (wr, bv, ri, li) in query.iter(world) {
        if li.0 == layer_idx {
            rows.push((ri.0, &wr.0, bv.0));
        }
    }
    rows.sort_by_key(|(ri, ..)| *ri);

    let out_f = rows.len();
    let mut weight_data = Vec::with_capacity(out_f * N);
    let mut bias_data = Vec::with_capacity(out_f);
    for (_, row, b) in &rows {
        weight_data.extend_from_slice(*row);
        bias_data.push(*b);
    }

    (
        Tensor::from_data(weight_data, &[out_f, N]),
        Tensor::from_data(bias_data, &[out_f]),
    )
}

/// Read weight matrix using contiguous_iter for SIMD-friendly access.
/// Falls back to per-entity iteration if contiguous_iter isn't available.
fn read_contiguous_weights_fast<const N: usize>(
    world: &mut World,
    layer_idx: usize,
) -> (Tensor, Tensor) {
    let mut query = world.query::<(&WeightRow<N>, &BiasValue, &RowIndex, &LayerIndex)>();

    // Try contiguous_iter first — gives &[WeightRow<N>] slices directly from BlobVec columns
    if let Some(contiguous) = query.contiguous_iter(world) {
        let mut rows_with_meta: Vec<(usize, usize, usize)> = Vec::new(); // (layer_idx, row_idx, position_in_slice)
        let mut all_weight_slices: Vec<&[WeightRow<N>]> = Vec::new();
        let mut all_bias_slices: Vec<&[BiasValue]> = Vec::new();
        let mut all_row_idx_slices: Vec<&[RowIndex]> = Vec::new();
        let mut all_layer_idx_slices: Vec<&[LayerIndex]> = Vec::new();

        for (weights, biases, row_indices, layer_indices) in contiguous {
            let base = all_weight_slices.iter().map(|s| s.len()).sum::<usize>();
            for (pos, li) in layer_indices.iter().enumerate() {
                if li.0 == layer_idx {
                    rows_with_meta.push((li.0, row_indices[pos].0, base + pos));
                }
            }
            all_weight_slices.push(weights);
            all_bias_slices.push(biases);
            all_row_idx_slices.push(row_indices);
            all_layer_idx_slices.push(layer_indices);
        }

        rows_with_meta.sort_by_key(|(_, ri, _)| *ri);
        let out_f = rows_with_meta.len();
        let mut weight_data = Vec::with_capacity(out_f * N);
        let mut bias_data = Vec::with_capacity(out_f);

        for (_, _, global_pos) in &rows_with_meta {
            // Find which slice and offset
            let mut remaining = *global_pos;
            for (si, slice) in all_weight_slices.iter().enumerate() {
                if remaining < slice.len() {
                    weight_data.extend_from_slice(&slice[remaining].0);
                    bias_data.push(all_bias_slices[si][remaining].0);
                    break;
                }
                remaining -= slice.len();
            }
        }

        return (
            Tensor::from_data(weight_data, &[out_f, N]),
            Tensor::from_data(bias_data, &[out_f]),
        );
    }

    // Fallback
    read_contiguous_weights::<N>(world, layer_idx)
}

/// Write gradients back to row entities.
fn write_contiguous_grads<const N: usize>(
    world: &mut World,
    layer_idx: usize,
    dw: &Tensor, // [out_f, N]
    db: &Tensor, // [out_f]
) {
    let mut query = world.query::<(&mut GradRow<N>, &mut BiasGradValue, &RowIndex, &LayerIndex)>();

    // Collect entity info first to avoid borrow issues
    let mut targets: Vec<(usize, usize)> = Vec::new(); // (row_idx, entity_position)
    {
        for (pos, (_, _, ri, li)) in query.iter(world).enumerate() {
            if li.0 == layer_idx {
                targets.push((ri.0, pos));
            }
        }
    }

    // Write gradients
    for (mut gr, mut bgv, ri, li) in query.iter_mut(world) {
        if li.0 == layer_idx {
            let row = ri.0;
            let grad_row = &dw.data[row * N..(row + 1) * N];
            for (dst, &src) in gr.0.iter_mut().zip(grad_row.iter()) {
                *dst += src;
            }
            bgv.0 += db.data[row];
        }
    }
}

// --- Exclusive forward pass ---

/// Forward pass as an exclusive system. Uses contiguous_iter for Linear layers.
pub fn forward_pass(world: &mut World) {
    // Gather layer metadata
    let mut layer_info: Vec<(usize, LayerType)> = Vec::new();
    {
        let mut query = world.query::<(&LayerKind, &LayerIndex)>();
        for (kind, idx) in query.iter(world) {
            layer_info.push((idx.0, kind.0.clone()));
        }
    }
    layer_info.sort_by_key(|(idx, _)| *idx);

    let eval_mode = world.resource::<TrainingState>().eval_mode;
    let mut current = world.resource::<InputBatch>().data.clone();

    for (i, kind) in &layer_info {
        let i = *i;
        world.resource_mut::<ActivationStorage>().inputs[i] = current.clone();

        let output = match kind {
            LayerType::Linear { in_features, .. } => {
                let in_f = *in_features;
                // Use contiguous_iter path: read weight matrix from row entities
                let (w, b) =
                    dispatch_row_size!(in_f, |N| read_contiguous_weights_fast::<N>(world, i));
                ops::linear(&current, &w, &b)
            }
            LayerType::ReLU => activations::relu(&current),
            LayerType::Conv2d {
                padding, stride, ..
            } => {
                let (w, b) = {
                    let mut q = world.query::<(&LayerWeights, &LayerBias, &LayerIndex)>();
                    let mut found = None;
                    for (lw, lb, li) in q.iter(world) {
                        if li.0 == i {
                            found = Some((lw.0.clone(), lb.0.clone()));
                            break;
                        }
                    }
                    found.unwrap()
                };
                conv_ops::direct_conv2d(&current, &w, &b, *padding, *stride)
            }
            LayerType::MaxPool2d { kernel_size } => {
                let (out, indices) = pool::max_pool2d(&current, *kernel_size);
                world.resource_mut::<ActivationStorage>().pool_indices[i] = Some(indices);
                out
            }
            LayerType::Flatten => {
                let batch = current.shape[0];
                let flat_size: usize = current.shape[1..].iter().product();
                current.view(&[batch, flat_size])
            }
            LayerType::LogSoftmax => activations::log_softmax(&current),
            LayerType::Dropout { p } => {
                if !eval_mode {
                    let mut rng = rand::rng();
                    let (out, mask) = activations::dropout(&current, *p, &mut rng);
                    world.resource_mut::<ActivationStorage>().dropout_masks[i] = Some(mask);
                    out
                } else {
                    current.clone()
                }
            }
        };

        world.resource_mut::<ActivationStorage>().outputs[i] = output.clone();
        current = output;
    }
}

/// Compute loss from the final layer's output.
pub fn compute_loss(world: &mut World) {
    let num_layers = {
        let mut q = world.query::<&LayerIndex>();
        q.iter(world).count()
    };
    if num_layers == 0 {
        return;
    }
    // LayerIndex values may not be contiguous with row entities present.
    // Find the max LayerIndex among layer entities (those with LayerKind).
    let last_idx = {
        let mut q = world.query::<(&LayerIndex, &LayerKind)>();
        q.iter(world).map(|(li, _)| li.0).max().unwrap()
    };

    let output = world.resource::<ActivationStorage>().outputs[last_idx].clone();
    let targets = world.resource::<InputBatch>().targets.clone();

    let (batch_loss, grad) = loss::nll_loss(&output, &targets);

    let predictions = loss::argmax(&output);
    let correct = predictions
        .iter()
        .zip(targets.iter())
        .filter(|(p, t)| p == t)
        .count();

    let mut loss_output = world.resource_mut::<LossOutput>();
    loss_output.batch_loss = batch_loss;
    loss_output.correct = correct;

    world.resource_mut::<GradientStorage>().output_grads[last_idx] = grad;
}

/// Backward pass as an exclusive system. Uses contiguous_iter for Linear gradient writes.
pub fn backward_pass(world: &mut World) {
    let eval_mode = world.resource::<TrainingState>().eval_mode;
    if eval_mode {
        return;
    }

    // Gather layer metadata and weights
    let mut layer_data: Vec<(usize, LayerType, Tensor)> = Vec::new();
    {
        let mut q = world.query::<(&LayerWeights, &LayerKind, &LayerIndex)>();
        for (lw, lk, li) in q.iter(world) {
            layer_data.push((li.0, lk.0.clone(), lw.0.clone()));
        }
    }
    layer_data.sort_by_key(|(idx, ..)| *idx);

    let num_layers = layer_data.len();

    for rev_i in (0..num_layers).rev() {
        let (i, ref kind, ref weight) = layer_data[rev_i];
        let layer_input = world.resource::<ActivationStorage>().inputs[i].clone();
        let d_output = world.resource::<GradientStorage>().output_grads[i].clone();

        let d_input = match kind {
            LayerType::Linear { in_features, .. } => {
                let in_f = *in_features;
                // Read weights from row entities (contiguous path)
                let (w, _b) =
                    dispatch_row_size!(in_f, |N| read_contiguous_weights_fast::<N>(world, i));
                let (di, dw, db) = ops::linear_backward(&d_output, &layer_input, &w);

                // Write gradients to row entities
                dispatch_row_size!(in_f, |N| write_contiguous_grads::<N>(world, i, &dw, &db));
                di
            }
            LayerType::ReLU => activations::relu_backward(&d_output, &layer_input),
            LayerType::Conv2d {
                padding, stride, ..
            } => {
                let (di, dw, db) = conv_ops::direct_conv2d_backward(
                    &d_output,
                    &layer_input,
                    weight,
                    *padding,
                    *stride,
                );
                // Write to layer entity gradients
                let mut q = world.query::<(&mut LayerGrad, &mut LayerBiasGrad, &LayerIndex)>();
                for (mut lg, mut lbg, li) in q.iter_mut(world) {
                    if li.0 == i {
                        lg.0.add_inplace(&dw);
                        lbg.0.add_inplace(&db);
                        break;
                    }
                }
                di
            }
            LayerType::MaxPool2d { .. } => {
                let indices = world.resource::<ActivationStorage>().pool_indices[i]
                    .as_ref()
                    .unwrap()
                    .clone();
                pool::max_pool2d_backward(&d_output, &indices, &layer_input.shape)
            }
            LayerType::Flatten => Tensor::from_data(d_output.data.clone(), &layer_input.shape),
            LayerType::LogSoftmax => d_output.clone(),
            LayerType::Dropout { p } => {
                let mask = world.resource::<ActivationStorage>().dropout_masks[i].clone();
                if let Some(mask) = &mask {
                    activations::dropout_backward(&d_output, mask, *p)
                } else {
                    d_output.clone()
                }
            }
        };

        if i > 0 {
            world.resource_mut::<GradientStorage>().output_grads[i - 1] = d_input;
        }
    }
}

/// Record metrics after each batch.
pub fn record_metrics(world: &mut World) {
    let eval_mode = world.resource::<TrainingState>().eval_mode;
    if eval_mode {
        return;
    }

    {
        let state = world.resource::<TrainingState>();
        let batch_idx = state.batch_idx;
        let epoch = state.epoch;
        let total_batches = state.total_batches;
        let total_epochs = state.total_epochs;
        let learning_rate = state.learning_rate;
        let batch_loss = world.resource::<LossOutput>().batch_loss;

        let mut metrics = world.resource_mut::<MetricsHistory>();
        metrics.current_batch = batch_idx;
        metrics.current_epoch = epoch;
        metrics.total_batches = total_batches;
        metrics.total_epochs = total_epochs;
        metrics.learning_rate = learning_rate;
        metrics.current_loss = batch_loss;
        metrics.batch_losses.push(batch_loss);
    }

    // Compute per-layer stats from row entities for Linear layers,
    // and from LayerWeights for Conv2d layers.
    let mut layer_info: Vec<(usize, LayerType)> = Vec::new();
    {
        let mut q = world.query::<(&LayerKind, &LayerIndex)>();
        for (lk, li) in q.iter(world) {
            layer_info.push((li.0, lk.0.clone()));
        }
    }
    layer_info.sort_by_key(|(idx, _)| *idx);

    let mut stats: Vec<(usize, String, f32, f32, f32)> = Vec::new();
    for (i, kind) in &layer_info {
        match kind {
            LayerType::Linear {
                in_features,
                out_features,
            } => {
                let in_f = *in_features;
                let (w, _) = dispatch_row_size!(in_f, |N| read_contiguous_weights::<N>(world, *i));
                if w.numel() == 0 {
                    continue;
                }
                let name = format!("Linear({in_features}->{out_features})");
                let mean: f32 = w.data.iter().sum::<f32>() / w.numel() as f32;
                let variance: f32 =
                    w.data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / w.numel() as f32;
                let std = variance.sqrt();
                // Grad norm from row entities
                let grad_norm = dispatch_row_size!(in_f, |N| compute_grad_norm::<N>(world, *i));
                stats.push((*i, name, mean, std, grad_norm));
            }
            LayerType::Conv2d {
                in_channels,
                out_channels,
                kernel_size,
                ..
            } => {
                let mut q = world.query::<(&LayerWeights, &LayerGrad, &LayerIndex)>();
                for (lw, lg, li) in q.iter(world) {
                    if li.0 == *i && lw.0.numel() > 0 {
                        let name = format!(
                            "Conv2d({in_channels}->{out_channels},{kernel_size}x{kernel_size})"
                        );
                        let mean: f32 = lw.0.data.iter().sum::<f32>() / lw.0.numel() as f32;
                        let variance: f32 =
                            lw.0.data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>()
                                / lw.0.numel() as f32;
                        let std = variance.sqrt();
                        let grad_norm: f32 = lg.0.data.iter().map(|x| x * x).sum::<f32>().sqrt();
                        stats.push((*i, name, mean, std, grad_norm));
                        break;
                    }
                }
            }
            _ => {}
        }
    }
    stats.sort_by_key(|(idx, ..)| *idx);
    world.resource_mut::<MetricsHistory>().layer_stats = stats
        .into_iter()
        .map(|(_, n, m, s, g)| (n, m, s, g))
        .collect();
}

fn compute_grad_norm<const N: usize>(world: &mut World, layer_idx: usize) -> f32 {
    let mut query = world.query::<(&GradRow<N>, &LayerIndex)>();
    let mut sum_sq = 0.0f32;
    for (gr, li) in query.iter(world) {
        if li.0 == layer_idx {
            for &v in gr.0.iter() {
                sum_sq += v * v;
            }
        }
    }
    sum_sq.sqrt()
}
