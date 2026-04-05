use bevy_ecs::resource::Resource;
use bevy_ecs::world::World;

use ecs_ml_core::components::*;
use ecs_ml_core::dispatch_row_size;
use ecs_ml_core::resources::TrainingState;

#[derive(Resource)]
pub struct AdamConfig {
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub step: usize,
}

/// Adam step as exclusive system.
pub fn adam_step(world: &mut World) {
    let eval_mode = world.resource::<TrainingState>().eval_mode;
    if eval_mode {
        return;
    }

    // Increment step and read config
    {
        let mut config = world.resource_mut::<AdamConfig>();
        config.step += 1;
    }
    let config = world.resource::<AdamConfig>();
    let b1 = config.beta1;
    let b2 = config.beta2;
    let eps = config.eps;
    let t = config.step as f32;
    let lr = world.resource::<TrainingState>().learning_rate;
    let bc1 = 1.0 - b1.powf(t);
    let bc2 = 1.0 - b2.powf(t);

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
                dispatch_row_size!(in_f, |N| adam_update_rows::<N>(
                    world, *i, lr, b1, b2, eps, bc1, bc2
                ));
            }
            LayerType::Conv2d { .. } => {
                adam_update_layer_entity(world, *i, lr, b1, b2, eps, bc1, bc2);
            }
            _ => {}
        }
    }
}

/// Initialize Adam state — currently a no-op for row entities (state stored inline).
/// For Conv2d layers, we'd need AdamFirstMoment/AdamSecondMoment components.
pub fn init_adam_state(_world: &mut World) {
    // Row entities: Adam m/v state is managed per-row during update
    // Conv2d: TODO — add moment components to layer entities
}

// Per-row Adam state stored as components
use bevy_ecs::component::Component;

#[derive(Component)]
pub struct AdamM1Row<const N: usize>(pub [f32; N]);

#[derive(Component)]
pub struct AdamM2Row<const N: usize>(pub [f32; N]);

#[derive(Component)]
pub struct AdamBiasM1(pub f32);

#[derive(Component)]
pub struct AdamBiasM2(pub f32);

#[allow(clippy::too_many_arguments)]
fn adam_update_rows<const N: usize>(
    world: &mut World,
    layer_idx: usize,
    lr: f32,
    b1: f32,
    b2: f32,
    eps: f32,
    bc1: f32,
    bc2: f32,
) {
    // Check if Adam state components exist; if not, spawn them
    let needs_init = {
        let mut q = world.query::<(&WeightRow<N>, Option<&AdamM1Row<N>>, &LayerIndex)>();
        let mut needs = false;
        for (_, m1, li) in q.iter(world) {
            if li.0 == layer_idx && m1.is_none() {
                needs = true;
                break;
            }
        }
        needs
    };

    if needs_init {
        // Collect entities that need Adam state
        let mut entities: Vec<bevy_ecs::entity::Entity> = Vec::new();
        {
            let mut q = world.query::<(bevy_ecs::entity::Entity, &WeightRow<N>, &LayerIndex)>();
            for (e, _, li) in q.iter(world) {
                if li.0 == layer_idx {
                    entities.push(e);
                }
            }
        }
        for e in entities {
            world.entity_mut(e).insert((
                AdamM1Row::<N>([0.0f32; N]),
                AdamM2Row::<N>([0.0f32; N]),
                AdamBiasM1(0.0),
                AdamBiasM2(0.0),
            ));
        }
    }

    // Now do the Adam update
    let mut query = world.query::<(
        &mut WeightRow<N>,
        &mut GradRow<N>,
        &mut AdamM1Row<N>,
        &mut AdamM2Row<N>,
        &mut BiasValue,
        &mut BiasGradValue,
        &mut AdamBiasM1,
        &mut AdamBiasM2,
        &LayerIndex,
    )>();

    for (mut wr, mut gr, mut m1, mut m2, mut bv, mut bgv, mut bm1, mut bm2, li) in
        query.iter_mut(world)
    {
        if li.0 != layer_idx {
            continue;
        }

        // Weight update
        for i in 0..N {
            let gi = gr.0[i];
            m1.0[i] = b1 * m1.0[i] + (1.0 - b1) * gi;
            m2.0[i] = b2 * m2.0[i] + (1.0 - b2) * gi * gi;
            let m_hat = m1.0[i] / bc1;
            let v_hat = m2.0[i] / bc2;
            wr.0[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }

        // Bias update
        let bg = bgv.0;
        bm1.0 = b1 * bm1.0 + (1.0 - b1) * bg;
        bm2.0 = b2 * bm2.0 + (1.0 - b2) * bg * bg;
        let bm_hat = bm1.0 / bc1;
        let bv_hat = bm2.0 / bc2;
        bv.0 -= lr * bm_hat / (bv_hat.sqrt() + eps);

        // Zero grads
        gr.0 = [0.0f32; N];
        bgv.0 = 0.0;
    }
}

#[allow(clippy::too_many_arguments)]
fn adam_update_layer_entity(
    world: &mut World,
    layer_idx: usize,
    lr: f32,
    _b1: f32,
    _b2: f32,
    _eps: f32,
    _bc1: f32,
    _bc2: f32,
) {
    // For Conv2d — simple SGD-like update for now (Adam moments on layer entity TBD)
    let mut q = world.query::<(
        &mut LayerWeights,
        &mut LayerBias,
        &mut LayerGrad,
        &mut LayerBiasGrad,
        &LayerIndex,
    )>();
    for (mut w, mut b, mut g, mut bg, li) in q.iter_mut(world) {
        if li.0 == layer_idx && w.0.numel() > 0 {
            // Simple gradient descent as placeholder
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
