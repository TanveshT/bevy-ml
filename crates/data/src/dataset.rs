/// Trait for datasets that provide (image, label) pairs.
pub trait Dataset {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Get a single sample's feature vector.
    fn get_features(&self, idx: usize) -> &[f32];
    /// Get a single sample's label.
    fn get_label(&self, idx: usize) -> usize;
    /// Feature dimensionality (flat).
    fn feature_dim(&self) -> usize;
    /// Number of classes.
    fn num_classes(&self) -> usize;
}
