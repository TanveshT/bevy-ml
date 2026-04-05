use bevy_ecs::resource::Resource;
use ecs_ml_tensor::Tensor;

#[derive(Resource)]
pub struct TrainingState {
    pub epoch: usize,
    pub batch_idx: usize,
    pub learning_rate: f32,
    pub eval_mode: bool,
    pub total_batches: usize,
    pub total_epochs: usize,
}

#[derive(Resource)]
pub struct InputBatch {
    pub data: Tensor,
    pub targets: Vec<usize>,
    pub batch_size: usize,
}

/// Stores activations for all layers, indexed by LayerIndex.
#[derive(Resource)]
pub struct ActivationStorage {
    /// Input to each layer (pre-activation for layers that transform).
    pub inputs: Vec<Tensor>,
    /// Output of each layer.
    pub outputs: Vec<Tensor>,
    /// Max-pool argmax indices (layer_idx → indices), only populated for MaxPool layers.
    pub pool_indices: Vec<Option<Vec<usize>>>,
    /// Dropout masks (layer_idx → mask), only populated for Dropout layers.
    pub dropout_masks: Vec<Option<Vec<bool>>>,
}

#[derive(Resource)]
pub struct GradientStorage {
    /// Gradient of loss w.r.t. each layer's output.
    pub output_grads: Vec<Tensor>,
}

#[derive(Resource)]
pub struct LossOutput {
    pub batch_loss: f32,
    pub correct: usize,
}

#[derive(Resource, Clone)]
pub struct MetricsHistory {
    pub train_losses: Vec<f32>,
    pub train_accuracies: Vec<f32>,
    pub test_accuracies: Vec<f32>,
    pub epoch_times_ms: Vec<u64>,
    pub current_epoch: usize,
    pub current_batch: usize,
    pub total_batches: usize,
    pub total_epochs: usize,
    pub learning_rate: f32,
    pub current_loss: f32,
    /// Per-batch loss values for the current epoch (for live chart).
    pub batch_losses: Vec<f32>,
    /// Per-layer stats: (name, weight_mean, weight_std, grad_norm)
    pub layer_stats: Vec<(String, f32, f32, f32)>,
}

impl Default for MetricsHistory {
    fn default() -> Self {
        Self {
            train_losses: Vec::new(),
            train_accuracies: Vec::new(),
            test_accuracies: Vec::new(),
            epoch_times_ms: Vec::new(),
            current_epoch: 0,
            current_batch: 0,
            total_batches: 0,
            total_epochs: 0,
            learning_rate: 0.0,
            current_loss: 0.0,
            batch_losses: Vec::new(),
            layer_stats: Vec::new(),
        }
    }
}
