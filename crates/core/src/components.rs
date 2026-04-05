use bevy_ecs::component::Component;
use ecs_ml_tensor::Tensor;

/// Weight matrix for a layer.
#[derive(Component)]
pub struct LayerWeights(pub Tensor);

/// Bias vector for a layer.
#[derive(Component)]
pub struct LayerBias(pub Tensor);

/// Weight gradient accumulator.
#[derive(Component)]
pub struct LayerGrad(pub Tensor);

/// Bias gradient accumulator.
#[derive(Component)]
pub struct LayerBiasGrad(pub Tensor);

/// Position of this layer in the network (0-indexed).
#[derive(Component)]
pub struct LayerIndex(pub usize);

/// What kind of layer this is.
#[derive(Component)]
pub struct LayerKind(pub LayerType);

/// Layer type enum — determines forward/backward dispatch.
#[derive(Clone, Debug)]
pub enum LayerType {
    Linear {
        in_features: usize,
        out_features: usize,
    },
    Conv2d {
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        padding: usize,
        stride: usize,
    },
    ReLU,
    MaxPool2d {
        kernel_size: usize,
    },
    Flatten,
    LogSoftmax,
    Dropout {
        p: f32,
    },
}

// --- Row-entity components for contiguous_iter ---

/// A single row of a weight matrix, stored inline in Bevy's BlobVec column.
/// With N entities of `WeightRow<N>`, the column IS the weight matrix — contiguous
/// floats, zero indirection, SIMD-ready via `contiguous_iter`.
#[derive(Component)]
#[repr(transparent)]
pub struct WeightRow<const N: usize>(pub [f32; N]);

/// Gradient accumulator for one weight row, same inline layout.
#[derive(Component)]
#[repr(transparent)]
pub struct GradRow<const N: usize>(pub [f32; N]);

/// Per-row bias value (one per output neuron).
#[derive(Component)]
pub struct BiasValue(pub f32);

/// Per-row bias gradient.
#[derive(Component)]
pub struct BiasGradValue(pub f32);

/// Which row within the layer (0..out_features).
#[derive(Component)]
pub struct RowIndex(pub usize);

/// Dispatch macro: match a runtime `usize` against common sizes, calling
/// a const-generic function for the matched size. Falls back to a closure
/// for unsupported sizes.
///
/// Usage: `dispatch_row_size!(in_features, |N| { body using WeightRow<N> }, fallback_expr)`
/// Dispatch macro: match a runtime `usize` to a const-generic function call.
///
/// Usage: `dispatch_row_size!(size_expr, |N| func_name::<N>(arg1, arg2))`
#[macro_export]
macro_rules! dispatch_row_size {
    ($size:expr, |$n:ident| $body:expr) => {
        match $size {
            10 => { const $n: usize = 10; $body }
            16 => { const $n: usize = 16; $body }
            32 => { const $n: usize = 32; $body }
            64 => { const $n: usize = 64; $body }
            128 => { const $n: usize = 128; $body }
            256 => { const $n: usize = 256; $body }
            512 => { const $n: usize = 512; $body }
            784 => { const $n: usize = 784; $body }
            1024 => { const $n: usize = 1024; $body }
            1568 => { const $n: usize = 1568; $body }
            2048 => { const $n: usize = 2048; $body }
            4096 => { const $n: usize = 4096; $body }
            other => panic!("Unsupported row size {other} for contiguous_iter. Add it to dispatch_row_size! in components.rs"),
        }
    };
}
