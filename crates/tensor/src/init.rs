use crate::Tensor;
use rand::Rng;

/// He initialization: uniform in [-sqrt(2/fan_in), sqrt(2/fan_in)].
pub fn he_init(shape: &[usize], fan_in: usize, rng: &mut impl Rng) -> Tensor {
    let std = (2.0 / fan_in as f32).sqrt();
    let size: usize = shape.iter().product();
    let data: Vec<f32> = (0..size).map(|_| rng.random_range(-std..std)).collect();
    Tensor::from_data(data, shape)
}

/// Xavier initialization: uniform in [-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out))].
pub fn xavier_init(shape: &[usize], fan_in: usize, fan_out: usize, rng: &mut impl Rng) -> Tensor {
    let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();
    let size: usize = shape.iter().product();
    let data: Vec<f32> = (0..size).map(|_| rng.random_range(-limit..limit)).collect();
    Tensor::from_data(data, shape)
}
