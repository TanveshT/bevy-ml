use bevy_app::App;
use bevy_ecs::world::World;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use ecs_ml_core::components::*;
use ecs_ml_core::dispatch_row_size;
use ecs_ml_core::resources::*;
use ecs_ml_tensor::Tensor;
use ecs_ml_tensor::init;

/// Builder for defining a neural network architecture.
pub struct NetworkBuilder {
    layers: Vec<LayerType>,
    seed: u64,
}

impl Default for NetworkBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl NetworkBuilder {
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            seed: 42,
        }
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn linear(mut self, in_features: usize, out_features: usize) -> Self {
        self.layers.push(LayerType::Linear {
            in_features,
            out_features,
        });
        self
    }

    pub fn relu(mut self) -> Self {
        self.layers.push(LayerType::ReLU);
        self
    }

    pub fn conv2d(mut self, in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        self.layers.push(LayerType::Conv2d {
            in_channels,
            out_channels,
            kernel_size,
            padding: kernel_size / 2,
            stride: 1,
        });
        self
    }

    pub fn conv2d_full(
        mut self,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        padding: usize,
        stride: usize,
    ) -> Self {
        self.layers.push(LayerType::Conv2d {
            in_channels,
            out_channels,
            kernel_size,
            padding,
            stride,
        });
        self
    }

    pub fn max_pool2d(mut self, kernel_size: usize) -> Self {
        self.layers.push(LayerType::MaxPool2d { kernel_size });
        self
    }

    pub fn flatten(mut self) -> Self {
        self.layers.push(LayerType::Flatten);
        self
    }

    pub fn log_softmax(mut self) -> Self {
        self.layers.push(LayerType::LogSoftmax);
        self
    }

    pub fn dropout(mut self, p: f32) -> Self {
        self.layers.push(LayerType::Dropout { p });
        self
    }

    /// Build the network: spawn layer entities (and row entities for Linear layers)
    /// and allocate storage resources.
    pub fn build(self, app: &mut App, batch_size: usize) {
        let num_layers = self.layers.len();
        let mut rng = SmallRng::seed_from_u64(self.seed);

        let world: &mut World = app.world_mut();

        // Spawn layer entities
        for (i, layer) in self.layers.iter().enumerate() {
            match layer {
                LayerType::Linear {
                    in_features,
                    out_features,
                } => {
                    let in_f = *in_features;
                    let out_f = *out_features;

                    // He-initialize full weight matrix, then distribute to row entities
                    let w = init::he_init(&[out_f, in_f], in_f, &mut rng);

                    // Spawn row entities: one per output neuron, each with WeightRow<IN>
                    dispatch_row_size!(in_f, |N| spawn_linear_rows::<N>(world, i, &w, out_f));

                    // Layer entity gets dummy weights (data lives in row entities)
                    // but keeps bias (one vec, not per-row for simplicity in non-linear layers)
                    let b = Tensor::zeros(&[out_f]);
                    let bias_grad = Tensor::zeros(&[out_f]);
                    world.spawn((
                        LayerWeights(Tensor::zeros(&[0])),
                        LayerBias(b),
                        LayerGrad(Tensor::zeros(&[0])),
                        LayerBiasGrad(bias_grad),
                        LayerIndex(i),
                        LayerKind(layer.clone()),
                    ));
                }
                LayerType::Conv2d {
                    in_channels,
                    out_channels,
                    kernel_size,
                    ..
                } => {
                    let fan_in = in_channels * kernel_size * kernel_size;
                    let w = init::he_init(
                        &[*out_channels, *in_channels, *kernel_size, *kernel_size],
                        fan_in,
                        &mut rng,
                    );
                    let b = Tensor::zeros(&[*out_channels]);
                    let grad = Tensor::zeros(&w.shape);
                    let bias_grad = Tensor::zeros(&b.shape);
                    world.spawn((
                        LayerWeights(w),
                        LayerBias(b),
                        LayerGrad(grad),
                        LayerBiasGrad(bias_grad),
                        LayerIndex(i),
                        LayerKind(layer.clone()),
                    ));
                }
                _ => {
                    // Non-parametric layers
                    world.spawn((
                        LayerWeights(Tensor::zeros(&[0])),
                        LayerBias(Tensor::zeros(&[0])),
                        LayerGrad(Tensor::zeros(&[0])),
                        LayerBiasGrad(Tensor::zeros(&[0])),
                        LayerIndex(i),
                        LayerKind(layer.clone()),
                    ));
                }
            }
        }

        // Allocate activation and gradient storage
        let empty = || Tensor::zeros(&[0]);
        world.insert_resource(ActivationStorage {
            inputs: (0..num_layers).map(|_| empty()).collect(),
            outputs: (0..num_layers).map(|_| empty()).collect(),
            pool_indices: (0..num_layers).map(|_| None).collect(),
            dropout_masks: (0..num_layers).map(|_| None).collect(),
        });
        world.insert_resource(GradientStorage {
            output_grads: (0..num_layers).map(|_| empty()).collect(),
        });
        world.insert_resource(LossOutput {
            batch_loss: 0.0,
            correct: 0,
        });

        // Determine input dimension from the first layer
        let input_dim = match &self.layers[0] {
            LayerType::Linear { in_features, .. } => vec![batch_size, *in_features],
            LayerType::Conv2d { in_channels, .. } => vec![batch_size, *in_channels, 28, 28],
            _ => vec![batch_size, 1],
        };
        world.insert_resource(InputBatch {
            data: Tensor::zeros(&input_dim),
            targets: vec![0; batch_size],
            batch_size,
        });
    }
}

/// Spawn row entities for a Linear layer. Called via dispatch_row_size! macro.
fn spawn_linear_rows<const N: usize>(
    world: &mut World,
    layer_idx: usize,
    weight_matrix: &Tensor,
    out_features: usize,
) {
    for row in 0..out_features {
        let mut row_data = [0.0f32; N];
        row_data.copy_from_slice(&weight_matrix.data[row * N..(row + 1) * N]);

        world.spawn((
            WeightRow::<N>(row_data),
            GradRow::<N>([0.0f32; N]),
            BiasValue(0.0),
            BiasGradValue(0.0),
            LayerIndex(layer_idx),
            RowIndex(row),
        ));
    }
}
