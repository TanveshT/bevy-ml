use bevy_ecs::resource::Resource;
use bevy_ecs::world::World;

use ecs_ml_core::components::*;
use ecs_ml_core::dispatch_row_size;
use ecs_ml_core::resources::TrainingState;

#[derive(Resource)]
pub struct SgdConfig {
    pub momentum: f32,
}

/// SGD step as exclusive system. Updates row entities (Linear) and layer entities (Conv2d).
pub fn sgd_step(world: &mut World) {
    let eval_mode = world.resource::<TrainingState>().eval_mode;
    if eval_mode {
        return;
    }
    let lr = world.resource::<TrainingState>().learning_rate;
    let momentum = world.resource::<SgdConfig>().momentum;

    // Gather layer info
    let mut layer_info: Vec<(usize, LayerType)> = Vec::new();
    {
        let mut q = world.query::<(&LayerKind, &LayerIndex)>();
        for (lk, li) in q.iter(world) {
            layer_info.push((li.0, lk.0.clone()));
        }
    }

    for (i, kind) in &layer_info {
        match kind {
            LayerType::Linear { in_features, .. } => {
                let in_f = *in_features;
                dispatch_row_size!(in_f, |N| sgd_update_rows::<N>(world, *i, lr, momentum));
            }
            LayerType::Conv2d { .. } => {
                sgd_update_layer_entity(world, *i, lr);
            }
            _ => {}
        }
    }
}

/// SGD update for Linear layer row entities via contiguous_iter_mut.
fn sgd_update_rows<const N: usize>(world: &mut World, layer_idx: usize, lr: f32, _momentum: f32) {
    let mut query = world.query::<(
        &mut WeightRow<N>,
        &mut GradRow<N>,
        &mut BiasValue,
        &mut BiasGradValue,
        &LayerIndex,
    )>();

    // Try contiguous_iter_mut for SIMD-friendly batch update
    if let Some(contiguous) = query.contiguous_iter_mut(world) {
        for (mut weights, mut grads, mut biases, mut bias_grads, layer_indices) in contiguous {
            for (idx, li) in layer_indices.iter().enumerate() {
                if li.0 != layer_idx {
                    continue;
                }
                // Plain SGD: w -= lr * grad
                let w_slice = &mut weights[idx].0;
                let g_slice = &grads[idx].0;
                for (w, &g) in w_slice.iter_mut().zip(g_slice.iter()) {
                    *w -= lr * g;
                }
                // Bias
                biases[idx].0 -= lr * bias_grads[idx].0;

                // Zero grads
                grads[idx].0 = [0.0f32; N];
                bias_grads[idx].0 = 0.0;
            }
        }
        return;
    }

    // Fallback: per-entity iteration
    for (mut wr, mut gr, mut bv, mut bgv, li) in query.iter_mut(world) {
        if li.0 != layer_idx {
            continue;
        }
        for (w, &g) in wr.0.iter_mut().zip(gr.0.iter()) {
            *w -= lr * g;
        }
        bv.0 -= lr * bgv.0;
        gr.0 = [0.0f32; N];
        bgv.0 = 0.0;
    }
}

/// SGD update for Conv2d (still uses LayerWeights/LayerGrad).
fn sgd_update_layer_entity(world: &mut World, layer_idx: usize, lr: f32) {
    let mut q = world.query::<(
        &mut LayerWeights,
        &mut LayerBias,
        &mut LayerGrad,
        &mut LayerBiasGrad,
        &LayerIndex,
    )>();
    for (mut w, mut b, mut g, mut bg, li) in q.iter_mut(world) {
        if li.0 == layer_idx && w.0.numel() > 0 {
            for (wv, &gv) in w.0.data.iter_mut().zip(g.0.data.iter()) {
                *wv -= lr * gv;
            }
            for (bv, &bgv) in b.0.data.iter_mut().zip(bg.0.data.iter()) {
                *bv -= lr * bgv;
            }
            g.0.fill(0.0);
            bg.0.fill(0.0);
            break;
        }
    }
}

/// Initialize momentum buffers (placeholder — momentum with row entities TBD).
pub fn init_momentum(_world: &mut World) {
    // TODO: momentum with row entities
}
